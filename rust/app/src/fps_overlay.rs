//! On-screen FPS counter overlay.
//!
//! A prerequisite for the upcoming mobile-adaptive-quality work: adaptive
//! quality decisions need an actual visible frame-time signal to test
//! against, so this makes a live frame-rate reading visible rather than
//! only queryable in code.
//!
//! ## Why this isn't just `bevy_diagnostic::FrameTimeDiagnosticsPlugin`
//!
//! An earlier version used that plugin directly and was reported unstable
//! on real GPU hardware: "jumping between 200 fps and 0.5 fps, when the
//! real fps was closer to 1." Investigated empirically (not just from
//! source reading) by logging raw per-frame deltas on this sandbox's
//! software-rendered (llvmpipe) backend: after an initial handful of
//! frames, the underlying `Time<Real>` delta this app measures *is*
//! accurate (it settled to a stable ~10-13fps here, matching the actual
//! shader cost) -- so this is **not** a case of Bevy's pipelined rendering
//! (main app one frame ahead of the render thread) decoupling the
//! measurement from real GPU-bound frame pacing; the render/main-app sync
//! point (see `bevy_render::pipelined_rendering`) does gate `Time<Real>`
//! correctly. (A `bevy_render::diagnostic::RenderDiagnosticsPlugin` GPU
//! timestamp signal was considered and rejected: it requires manual
//! per-render-node instrumentation we don't have around Bevy's built-in 2D
//! pipeline, and critically only supports GPU timestamps on Vulkan/DX12 --
//! not WebGPU or WebGL2, i.e. not our actual mobile/web deploy target.)
//!
//! Two real, separate causes remain, both addressed below without needing
//! a different measurement source:
//!  1. **A startup transient**: the first several frames render before the
//!     marcher's (expensive, async-compiled) pipeline is ready, so they're
//!     nearly free and read as a spurious high-fps burst (observed ~100fps
//!     for ~3-5 frames here) before dropping to the true steady rate.
//!  2. **Genuine, large content-dependent cost**: this shader's cost is
//!     wildly different for a sky/background pixel (a handful of `map()`
//!     calls before the `t > MAX_DIST` miss) versus a pixel that hits dense
//!     fractal detail (up to 256 march steps + 6 normal taps + a shadow
//!     ray). Panning the camera between mostly-sky and mostly-fractal
//!     framing is a *real* multi-hundred-times swing in per-frame GPU cost,
//!     not a bug -- and FPS (a reciprocal-of-time metric) visually
//!     exaggerates this compared to frame time (doubling frame time from
//!     8ms to 16ms reads as "125 -> 62", the same relative change from
//!     800ms to 1600ms reads as "1.25 -> 0.6", both proportionally
//!     identical but the small-number case *looks* more like noise).
//!
//! The fix: average *frame time* (not already-inverted fps values) over a
//! rolling ~1-second window -- `frames_in_window / sum(times_in_window)` --
//! rather than Bevy's default short EMA smoothing of instantaneous fps
//! samples. This damps both the startup transient (it ages out of the
//! window within ~1s) and short camera-pan-driven swings, while still
//! genuinely tracking real, sustained changes in scene cost. Frame time in
//! ms is also displayed alongside fps, since it doesn't have fps's
//! reciprocal-metric exaggeration and is the more stable-*looking* of the
//! two even though both are derived from the same underlying average.

use std::collections::VecDeque;
use std::time::Duration;

use bevy::prelude::*;

/// How long a window of recent frame times to average over. Long enough to
/// smooth both the startup transient and brief camera-pan cost swings;
/// short enough to still track real, sustained changes in scene cost
/// within about a second.
const WINDOW_SECONDS: f64 = 1.0;

/// Always keep at least this many recent samples, even if their combined
/// duration already exceeds `WINDOW_SECONDS` -- without this floor, a
/// single sample *longer* than the whole window (a real stall, e.g. the
/// pipeline-compile transient's first expensive frame) would evict every
/// other sample down to itself alone, and the "average" would just be that
/// one frame's instantaneous reciprocal -- reintroducing the exact
/// single-frame-dominates instability this windowing exists to damp.
const MIN_SAMPLES: usize = 30;

pub struct FpsOverlayPlugin;

impl Plugin for FpsOverlayPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FrameTimeWindow>()
            .init_resource::<DebugTwistAccum>()
            .init_resource::<PhaseTimings>()
            .add_systems(Startup, spawn_fps_overlay)
            .add_systems(Update, (record_frame_time, update_fps_text).chain())
            .add_systems(
                Update,
                (update_orbit_debug_text, update_marbles_debug_text, update_phase_timings_text),
            );
    }
}

/// Running total of every `delta_radians` ever passed to
/// `CameraOrbit::roll` (Q/E and two-finger twist alike) -- display-only,
/// never read by any real camera math. The arcball `CameraOrbit` has no
/// standalone "roll" value anymore (twist is just part of `orientation`,
/// same as everything else), but a numeric roll readout has been genuinely
/// useful for diagnosing three rounds of live camera bugs, so this keeps it
/// available without reintroducing a second piece of state the real camera
/// logic has to stay in sync with -- incremented at each `orbit.roll(...)`
/// call site (`camera.rs`'s `orbit_camera_input`, `touch.rs`'s
/// `touch_camera_input`), never read there.
#[derive(Resource, Default)]
pub struct DebugTwistAccum(pub f32);

/// Rolling window of recent per-frame wall-clock times (seconds), used to
/// compute a windowed-average fps/frame-time pair -- see the module doc for
/// why this is more stable than Bevy's default EMA-of-instantaneous-fps.
#[derive(Resource, Default)]
struct FrameTimeWindow {
    samples: VecDeque<f64>,
    window_sum: f64,
}

impl FrameTimeWindow {
    fn push(&mut self, delta_secs: f64) {
        self.samples.push_back(delta_secs);
        self.window_sum += delta_secs;
        while self.window_sum > WINDOW_SECONDS && self.samples.len() > MIN_SAMPLES {
            if let Some(oldest) = self.samples.pop_front() {
                self.window_sum -= oldest;
            }
        }
    }

    /// `(fps, frame_time_ms)` averaged over the current window, or `None`
    /// if no non-zero-duration samples have been recorded yet.
    fn averaged(&self) -> Option<(f64, f64)> {
        if self.samples.is_empty() || self.window_sum <= 0.0 {
            return None;
        }
        let avg_frame_secs = self.window_sum / self.samples.len() as f64;
        Some((1.0 / avg_frame_secs, avg_frame_secs * 1000.0))
    }
}

fn record_frame_time(time: Res<Time>, mut window: ResMut<FrameTimeWindow>) {
    let delta = time.delta_secs_f64();
    if delta > 0.0 {
        window.push(delta);
    }
}

/// **What this measures, and what it doesn't**: wall-clock CPU time spent
/// inside each named `Update`/`FixedUpdate` system that prepares a frame --
/// the physics tick, and each render pass's per-frame uniform computation
/// (`update_material`/`update_coarse_material`/`update_shadow_material`).
/// This is *not* GPU execution time, and not even full CPU-side command
/// encoding/submission time (that happens later, in Bevy's Render sub-app,
/// out of easy reach from an ordinary `Update`-schedule system without much
/// deeper render-graph instrumentation than this overlay's existing
/// "genuinely useful, cheap to build" bar justifies). `fps_overlay.rs`'s
/// module doc already covers why real GPU timestamps aren't available at
/// all on this project's actual deploy target (WebGPU) --
/// `RenderDiagnosticsPlugin` only supports them on native Vulkan/DX12.
///
/// So: if every phase here reads small and roughly flat, that's real
/// evidence the bottleneck is GPU-side (shader execution), not something
/// this readout can see directly -- in that case the actual diagnostic tool
/// is the `?mrrm=0`/`?shadowlod=0` query-param toggles (`mrrm.rs`/
/// `shadow_pass.rs`) compared against the FPS line's *total* frame time,
/// which genuinely does reflect real end-to-end GPU cost (turn a pass off,
/// watch whether frame time actually drops). This readout's job is
/// narrower but still real: catching a CPU-side regression (an expensive
/// system, an accidental per-frame allocation storm, multiplayer rollback
/// resimulation cost) that *would* show up here directly.
#[derive(Resource, Default)]
pub struct PhaseTimings {
    phases: Vec<(&'static str, FrameTimeWindow)>,
}

impl PhaseTimings {
    /// Records one call's elapsed time for `phase`, creating its rolling
    /// window on first use. Insertion order (not alphabetical/hash order)
    /// is what the readout displays in, so call this the first time in
    /// whatever order should read top-to-bottom.
    pub fn record(&mut self, phase: &'static str, elapsed: Duration) {
        match self.phases.iter_mut().find(|(name, _)| *name == phase) {
            Some((_, window)) => window.push(elapsed.as_secs_f64()),
            None => {
                let mut window = FrameTimeWindow::default();
                window.push(elapsed.as_secs_f64());
                self.phases.push((phase, window));
            }
        }
    }
}

/// Marker for the `Text` entity showing the live FPS number.
#[derive(Component)]
struct FpsText;

/// Marker for the `Text` entity showing live camera state (accumulated
/// twist + `forward`) -- a numeric readout precise enough to verify
/// camera-direction fixes against exact expected values from a screenshot,
/// rather than eyeballing whether the rendered fractal "looks" rotated
/// correctly.
#[derive(Component)]
struct OrbitDebugText;

/// Marker for the `Text` entity showing every marble's live state
/// (multiplayer milestone 0) -- exact positions/velocities are what let a
/// live build's marble-vs-marble collision be verified precisely (are they
/// actually separating, actually bouncing) instead of squinting at
/// screenshots for spheres that can be a few hundredths of a world unit
/// across. Worth keeping past this milestone's own verification too: a
/// future rollback engine (milestone 1) and real networking (milestone 2)
/// will keep needing exactly this kind of visibility into every marble's
/// state, not just the local player's.
#[derive(Component)]
struct MarblesDebugText;

/// Marker for the `Text` entity showing [`PhaseTimings`]'s per-phase
/// CPU breakdown -- see that resource's doc for exactly what this does and
/// doesn't measure.
#[derive(Component)]
struct PhaseTimingsText;

fn spawn_fps_overlay(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(6.0),
                left: Val::Px(6.0),
                padding: UiRect::axes(Val::Px(6.0), Val::Px(3.0)),
                flex_direction: FlexDirection::Column,
                ..default()
            },
            // Semi-transparent dark backing so the text reads over any part
            // of the (arbitrarily bright/colorful) fractal render.
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.55)),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("FPS: --"),
                TextFont {
                    font_size: 18.0,
                    ..default()
                },
                // Bright, high-contrast against the dark backing panel.
                TextColor(Color::srgb(0.2, 1.0, 0.4)),
                FpsText,
            ));
            parent.spawn((
                Text::new("twist: -- forward: --"),
                TextFont {
                    font_size: 14.0,
                    ..default()
                },
                TextColor(Color::srgb(0.6, 0.9, 1.0)),
                OrbitDebugText,
            ));
            parent.spawn((
                Text::new("marbles: --"),
                TextFont {
                    font_size: 14.0,
                    ..default()
                },
                TextColor(Color::srgb(1.0, 0.75, 0.4)),
                MarblesDebugText,
            ));
            parent.spawn((
                Text::new("phases: --"),
                TextFont {
                    font_size: 12.0,
                    ..default()
                },
                TextColor(Color::srgb(0.75, 0.75, 0.75)),
                PhaseTimingsText,
            ));
        });
}

fn update_phase_timings_text(
    timings: Res<PhaseTimings>,
    mut text: Query<&mut Text, With<PhaseTimingsText>>,
) {
    let Ok(mut text) = text.single_mut() else {
        return;
    };
    if timings.phases.is_empty() {
        return;
    }
    let parts: Vec<String> = timings
        .phases
        .iter()
        .filter_map(|(name, window)| window.averaged().map(|(_, ms)| format!("{name}={ms:.2}ms")))
        .collect();
    text.0 = format!("cpu phases: {}", parts.join(" "));
}

fn update_orbit_debug_text(
    orbit: Res<crate::camera::CameraOrbit>,
    twist_debug: Res<DebugTwistAccum>,
    touch_debug: Res<crate::touch::TouchDebugInfo>,
    mut text: Query<&mut Text, With<OrbitDebugText>>,
) {
    let Ok(mut text) = text.single_mut() else {
        return;
    };
    let f = orbit.forward();
    // `touches` line always shows the live count (0/1/2+), and the raw
    // swipe delta + the arcball formula's real intermediates (`screen_dir`,
    // `angle`) whenever a single-finger swipe actually happened this frame
    // -- lets a real on-device repro (twist to some roll, then swipe) be
    // diagnosed from a screenshot of exactly what `drag` computed, rather
    // than a description of it. See `touch::TouchDebugInfo`'s doc.
    let touch_line = match (touch_debug.swipe_delta, touch_debug.screen_dir, touch_debug.angle_deg) {
        (Some(d), Some(s), Some(a)) => format!(
            "touches: {} delta: ({:.1}, {:.1}) screen_dir: ({:.2}, {:.2}, {:.2}) angle: {:.2}deg",
            touch_debug.active_count, d.x, d.y, s.x, s.y, s.z, a
        ),
        _ => format!("touches: {}", touch_debug.active_count),
    };
    text.0 = format!(
        "twist: {:.1}deg forward: ({:.3}, {:.3}, {:.3})\n{touch_line}",
        twist_debug.0.to_degrees(),
        f.x,
        f.y,
        f.z
    );
}

/// See [`MarblesDebugText`]'s doc for why this exists. One line per marble:
/// index, position, and speed (`|vel|`, not the full vector -- position is
/// what matters for "are they separated," speed for "did collision actually
/// impart an impulse, or is it just sitting still").
fn update_marbles_debug_text(
    marble_state: Res<crate::physics_sys::MarbleState>,
    mut text: Query<&mut Text, With<MarblesDebugText>>,
) {
    let Ok(mut text) = text.single_mut() else {
        return;
    };
    let mut lines = vec![format!("marbles: {}", marble_state.marbles.len())];
    for (i, m) in marble_state.marbles.iter().enumerate() {
        let marker = if i == marble_state.local_player_index { "*" } else { " " };
        lines.push(format!(
            "{marker}{i}: ({:.3}, {:.3}, {:.3}) |vel|={:.4}",
            m.pos.x,
            m.pos.y,
            m.pos.z,
            m.vel.length()
        ));
    }
    text.0 = lines.join("\n");
}

fn update_fps_text(window: Res<FrameTimeWindow>, mut text: Query<&mut Text, With<FpsText>>) {
    let Ok(mut text) = text.single_mut() else {
        return;
    };
    let Some((fps, frame_ms)) = window.averaged() else {
        return;
    };
    text.0 = format!("FPS: {fps:.1} ({frame_ms:.1}ms)");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_window_has_no_average() {
        let window = FrameTimeWindow::default();
        assert!(window.averaged().is_none());
    }

    #[test]
    fn steady_frame_time_gives_matching_fps() {
        let mut window = FrameTimeWindow::default();
        // 60fps steady-state: 1/60s per frame, well under the 1s window.
        for _ in 0..30 {
            window.push(1.0 / 60.0);
        }
        let (fps, ms) = window.averaged().unwrap();
        assert!((fps - 60.0).abs() < 0.5, "expected ~60fps, got {fps}");
        assert!((ms - 16.667).abs() < 0.5, "expected ~16.7ms, got {ms}");
    }

    #[test]
    fn a_single_expensive_frame_does_not_dominate_the_average() {
        // Regression check for the reported "jumping between 200fps and
        // 0.5fps" instability: a lone very slow frame amid many fast ones
        // (e.g. the startup pipeline-compile transient, or one frame that
        // panned across dense fractal detail) should be damped by the
        // window average, not make the displayed number swing wildly.
        let mut window = FrameTimeWindow::default();
        for _ in 0..50 {
            window.push(1.0 / 200.0); // a burst of spuriously "free" frames
        }
        window.push(2.0); // one frame that took 2 real seconds
        let (fps, _) = window.averaged().unwrap();
        // Not anywhere near the instantaneous 0.5fps that single frame's
        // reciprocal would suggest, and not anywhere near the preceding
        // 200fps burst either -- damped toward something in between.
        assert!(fps > 1.0, "a single slow frame swung the average too low: {fps}");
        assert!(fps < 100.0, "a single slow frame didn't move the average at all: {fps}");
    }

    #[test]
    fn window_ages_out_old_samples_beyond_one_second() {
        let mut window = FrameTimeWindow::default();
        // Push 2 seconds' worth of slow (0.5fps) frames, then switch to a
        // sustained fast rate -- after enough fast frames accumulate past
        // the 1s window, the old slow samples should be fully aged out.
        for _ in 0..4 {
            window.push(0.5);
        }
        for _ in 0..300 {
            window.push(1.0 / 300.0); // fills much more than 1s at this rate
        }
        let (fps, _) = window.averaged().unwrap();
        assert!(
            fps > 100.0,
            "old slow samples should have aged out of the window, got {fps}"
        );
    }
}
