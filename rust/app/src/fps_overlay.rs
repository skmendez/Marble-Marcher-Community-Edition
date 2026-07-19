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
            .add_systems(Startup, spawn_fps_overlay)
            .add_systems(Update, (record_frame_time, update_fps_text).chain());
    }
}

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

/// Marker for the `Text` entity showing the live FPS number.
#[derive(Component)]
struct FpsText;

fn spawn_fps_overlay(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(6.0),
                left: Val::Px(6.0),
                padding: UiRect::axes(Val::Px(6.0), Val::Px(3.0)),
                ..default()
            },
            // Semi-transparent dark backing so the text reads over any part
            // of the (arbitrarily bright/colorful) fractal render.
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.55)),
        ))
        .with_child((
            Text::new("FPS: --"),
            TextFont {
                font_size: 18.0,
                ..default()
            },
            // Bright, high-contrast against the dark backing panel.
            TextColor(Color::srgb(0.2, 1.0, 0.4)),
            FpsText,
        ));
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
