//! `?perfprobe=1` (web) / `MM_PERFPROBE=1` (native): an automated GPU
//! relative-cost breakdown.
//!
//! The existing "cpu phases" overlay (`fps_overlay.rs`'s `PhaseTimings`)
//! only measures CPU-side command *recording* time -- when every phase
//! there reads small and flat (the common case on this renderer: all four
//! phases cumulatively under 0.1ms while real frame time is 15-20ms+), that
//! tells you the bottleneck is GPU-side shader execution, but not *where*.
//! Real GPU timestamp queries aren't a viable first step here (investigated
//! this session: `bevy_render::diagnostic::RenderDiagnosticsPlugin` is
//! architecturally incompatible with the WebGPU backend -- confirmed
//! `write_timestamp()` mid-pass unconditionally panics there per spec, only
//! the pass-descriptor `timestamp_writes` begin/end mechanism works, and
//! wiring that requires forking Bevy's built-in `Mesh2dPipeline` render-graph
//! node into a custom one; even then it's Chromium-only, no Firefox/Safari).
//!
//! This is the cheap, universal alternative `fps_overlay.rs`'s own module
//! doc already points at: toggle a pass off/down and watch whether *real*
//! end-to-end frame time (the existing FPS overlay's own windowed average,
//! which genuinely is GPU-inclusive wall-clock time) actually moves. This
//! module automates that into a self-cycling sequence of several-second
//! windows -- baseline, no shadow pass, no coarse pass, a clamped fine-pass
//! step budget -- logging each window's average frame time to the console
//! (and the on-screen overlay) once it completes, so a relative cost
//! ranking can be read off a live session (including on a real phone, where
//! opening devtools to read console output is the only realistic profiling
//! tool available at all) without needing to manually flip query params and
//! reload between each comparison.
//!
//! Opt-in only (`perfprobe_enabled()`, same query-param-then-env-var
//! pattern as `mrrm::mrrm_enabled`/`shadow_pass::shadow_lod_enabled`) --
//! `perfprobe_tick` is a no-op whenever it's off, so this has zero behavior
//! change and negligible overhead (one resource read, one bool check) on
//! every normal run.

use bevy::prelude::*;
use web_time::Instant;

use crate::mrrm::CoarseCamera;
use crate::shadow_pass::ShadowCamera;

/// How long each probe window runs before its average is logged and the
/// next window starts. Long enough that a handful of startup-transient or
/// camera-pan frames don't skew the average (`fps_overlay.rs`'s own
/// `WINDOW_SECONDS` reasoning applies here too, just at a coarser grain
/// since this is a manual/console-read tool, not a live HUD number).
const WINDOW_DURATION_SECS: f64 = 4.0;

/// The fine pass's step budget during the "low fine steps" window --
/// deliberately far below any value that would look visually correct (real
/// tuning candidates for `MAX_STEPS` belong to Milestone 2's algorithmic
/// work, informed by this probe's output, not this diagnostic) -- this only
/// needs to isolate "how much of the fine pass's cost is its march loop
/// itself" as a rough upper bound on that lever's potential payoff.
const LOW_FINE_STEPS: f32 = 8.0;

/// `0.0` is `SceneUniforms::misc2.z`'s "no override" sentinel (see its doc)
/// -- every window other than `LowFineSteps` uses this.
const NO_FINE_STEP_OVERRIDE: f32 = 0.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ProbeWindow {
    Baseline,
    NoCoarsePass,
    NoShadowPass,
    LowFineSteps,
}

impl ProbeWindow {
    const CYCLE: [ProbeWindow; 4] = [
        ProbeWindow::Baseline,
        ProbeWindow::NoCoarsePass,
        ProbeWindow::NoShadowPass,
        ProbeWindow::LowFineSteps,
    ];

    fn label(self) -> &'static str {
        match self {
            ProbeWindow::Baseline => "baseline",
            ProbeWindow::NoCoarsePass => "no_coarse_pass",
            ProbeWindow::NoShadowPass => "no_shadow_pass",
            ProbeWindow::LowFineSteps => "low_fine_steps",
        }
    }

    fn coarse_camera_active(self) -> bool {
        !matches!(self, ProbeWindow::NoCoarsePass)
    }

    fn shadow_camera_active(self) -> bool {
        !matches!(self, ProbeWindow::NoShadowPass)
    }

    fn fine_max_steps_override(self) -> f32 {
        match self {
            ProbeWindow::LowFineSteps => LOW_FINE_STEPS,
            _ => NO_FINE_STEP_OVERRIDE,
        }
    }
}

/// `?perfprobe=1`/`MM_PERFPROBE=1` gate -- same `OnceLock` + query-param-
/// then-env-var layering as `mrrm::mrrm_enabled`/
/// `shadow_pass::shadow_lod_enabled`.
pub fn perfprobe_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let value =
            crate::web_config::query_param("perfprobe").or_else(|| std::env::var("MM_PERFPROBE").ok());
        matches!(value.as_deref(), Some("1") | Some("true"))
    })
}

/// Live probe state -- always inserted (`init_resource`), regardless of
/// whether `perfprobe_enabled()` is true, so `render::update_material` can
/// unconditionally read `fine_max_steps_override` without an `Option`
/// param; `perfprobe_tick` simply never advances it past its `Default` when
/// the probe is off, so `fine_max_steps_override` stays `0.0` (no override)
/// forever in that case.
#[derive(Resource)]
pub struct PerfProbeState {
    window_index: usize,
    window_started: Instant,
    sum_secs: f64,
    count: u32,
    /// Set on `perfprobe_tick`'s first real call -- `window_started` is
    /// otherwise stamped at resource-construction time (`Startup`, before
    /// the pipeline's first-render shader-compile transient), which would
    /// otherwise burn part of `Baseline`'s window on that transient and
    /// could even finalize it with zero recorded frames if compilation ever
    /// took longer than `WINDOW_DURATION_SECS`. The first tick instead
    /// re-stamps `window_started` to *that* frame and returns without
    /// finalizing/advancing anything, so `Baseline`'s timer only starts
    /// once real frames are actually happening.
    started: bool,
    /// Mirrors `ProbeWindow::fine_max_steps_override` for whichever window
    /// is currently active -- `render::update_material_impl` copies this
    /// straight into `SceneUniforms::misc2.z` every frame.
    pub fine_max_steps_override: f32,
}

impl Default for PerfProbeState {
    fn default() -> Self {
        Self {
            window_index: 0,
            window_started: Instant::now(),
            sum_secs: 0.0,
            count: 0,
            started: false,
            fine_max_steps_override: NO_FINE_STEP_OVERRIDE,
        }
    }
}

/// `Update` system (must run before `update_material`/`update_coarse_material`/
/// `update_shadow_material` in `main.rs`'s chained system list, so a window
/// transition's camera-active/step-override changes apply on the very frame
/// they happen, not one frame late): accumulates this frame's delta into the
/// current window, and once `WINDOW_DURATION_SECS` has elapsed, logs the
/// finished window's average frame time and advances to the next window in
/// `ProbeWindow::CYCLE` (wrapping back to `Baseline` after `LowFineSteps`,
/// so a session left running keeps producing fresh comparable samples).
pub fn perfprobe_tick(
    time: Res<Time>,
    mut state: ResMut<PerfProbeState>,
    mut coarse_cameras: Query<&mut Camera, (With<CoarseCamera>, Without<ShadowCamera>)>,
    mut shadow_cameras: Query<&mut Camera, (With<ShadowCamera>, Without<CoarseCamera>)>,
) {
    if !perfprobe_enabled() {
        return;
    }

    if !state.started {
        state.started = true;
        state.window_started = Instant::now();
        return;
    }

    let delta = time.delta_secs_f64();
    if delta > 0.0 {
        state.sum_secs += delta;
        state.count += 1;
    }

    if state.window_started.elapsed().as_secs_f64() < WINDOW_DURATION_SECS {
        return;
    }

    let finished = ProbeWindow::CYCLE[state.window_index];
    if state.count > 0 {
        let avg_ms = (state.sum_secs / state.count as f64) * 1000.0;
        info!(
            "perfprobe: window '{}' finished -- avg frame time {:.3}ms ({:.1} fps) over {} frames",
            finished.label(),
            avg_ms,
            1000.0 / avg_ms,
            state.count,
        );
    } else {
        warn!("perfprobe: window '{}' finished with zero frames recorded", finished.label());
    }

    state.window_index = (state.window_index + 1) % ProbeWindow::CYCLE.len();
    state.window_started = Instant::now();
    state.sum_secs = 0.0;
    state.count = 0;

    let next = ProbeWindow::CYCLE[state.window_index];
    state.fine_max_steps_override = next.fine_max_steps_override();
    for mut camera in &mut coarse_cameras {
        camera.is_active = next.coarse_camera_active();
    }
    for mut camera in &mut shadow_cameras {
        camera.is_active = next.shadow_camera_active();
    }
    info!(
        "perfprobe: entering window '{}' (coarse_active={} shadow_active={} fine_max_steps={})",
        next.label(),
        next.coarse_camera_active(),
        next.shadow_camera_active(),
        if next.fine_max_steps_override() > 0.0 {
            next.fine_max_steps_override().to_string()
        } else {
            // Kept in sync by hand with `codegen.rs`'s `FINE_MAX_STEPS` --
            // this is just a diagnostic log label, not a real dependency
            // (the shader itself always reads the real constant), but a
            // stale value here would mislead whoever's reading this output.
            "default(128)".to_string()
        },
    );
}

/// Marker for the on-screen text entity showing the currently-active probe
/// window -- only spawned/updated when `perfprobe_enabled()`, so a normal
/// (non-probe) run's overlay is unchanged. `pub(crate)`: it appears in
/// `update_perfprobe_overlay_text`'s public query type, which `main.rs` (a
/// different module) needs to be able to name when registering that system
/// -- same reasoning as `mrrm::CoarseQuad`'s doc.
#[derive(Component)]
pub(crate) struct PerfProbeText;

pub fn spawn_perfprobe_overlay(mut commands: Commands) {
    if !perfprobe_enabled() {
        return;
    }
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            // Bottom-left: `fps_overlay.rs`'s debug overlay lives top-left
            // and `net.rs`'s share-link panel lives top-right (both visible
            // at once when `?debug=1` and multiplayer UI are both up) --
            // this avoids overlapping either.
            bottom: Val::Px(6.0),
            left: Val::Px(6.0),
            padding: UiRect::axes(Val::Px(6.0), Val::Px(3.0)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.55)),
        children![(
            Text::new("perfprobe: warming up (see console for results)"),
            TextFont {
                font_size: 14.0,
                ..default()
            },
            TextColor(Color::srgb(1.0, 0.9, 0.3)),
            PerfProbeText,
        )],
    ));
}

pub fn update_perfprobe_overlay_text(
    state: Res<PerfProbeState>,
    mut text: Query<&mut Text, With<PerfProbeText>>,
) {
    if !perfprobe_enabled() {
        return;
    }
    let Ok(mut text) = text.single_mut() else {
        return;
    };
    let current = ProbeWindow::CYCLE[state.window_index];
    let remaining = (WINDOW_DURATION_SECS - state.window_started.elapsed().as_secs_f64()).max(0.0);
    text.0 = format!(
        "perfprobe: window '{}' ({:.1}s left) -- results logged to console",
        current.label(),
        remaining,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cycle_covers_every_window_exactly_once() {
        let labels: Vec<_> = ProbeWindow::CYCLE.iter().map(|w| w.label()).collect();
        assert_eq!(labels.len(), 4);
        assert!(labels.contains(&"baseline"));
        assert!(labels.contains(&"no_coarse_pass"));
        assert!(labels.contains(&"no_shadow_pass"));
        assert!(labels.contains(&"low_fine_steps"));
    }

    #[test]
    fn only_low_fine_steps_overrides_the_step_budget() {
        for window in ProbeWindow::CYCLE {
            let overridden = window.fine_max_steps_override() != NO_FINE_STEP_OVERRIDE;
            assert_eq!(overridden, window == ProbeWindow::LowFineSteps, "{window:?}");
        }
    }

    #[test]
    fn exactly_one_window_disables_each_pass() {
        let coarse_off = ProbeWindow::CYCLE.iter().filter(|w| !w.coarse_camera_active()).count();
        let shadow_off = ProbeWindow::CYCLE.iter().filter(|w| !w.shadow_camera_active()).count();
        assert_eq!(coarse_off, 1);
        assert_eq!(shadow_off, 1);
    }

    #[test]
    fn default_state_has_no_override_and_starts_at_baseline() {
        let state = PerfProbeState::default();
        assert_eq!(state.fine_max_steps_override, NO_FINE_STEP_OVERRIDE);
        assert_eq!(state.window_index, 0);
        assert_eq!(ProbeWindow::CYCLE[state.window_index], ProbeWindow::Baseline);
    }
}
