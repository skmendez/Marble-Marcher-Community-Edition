//! Adaptive internal render resolution: the pure decision logic that decides
//! how many pixels the (expensive, per-pixel) ray marcher should actually
//! render this "tick" of the controller, given a live averaged frame time.
//!
//! This module deliberately contains **no Bevy types** in its core
//! functions (`next_scale`, `target_pixel_size`) — same separation as
//! `touch.rs`'s gesture math: plain-data functions that are trivially
//! unit-tested, called from a thin Bevy-facing system
//! (`adjust_resolution_scale`) that wires in the real frame-time signal
//! (`fps_overlay::FrameTimeWindow`, already-debugged rolling-average timer —
//! reused here rather than duplicated) and a real `Resource`.
//!
//! The actual render-target creation/resize/present-blit machinery driven by
//! this decision lives in `present.rs`.

use bevy::prelude::*;

use crate::fps_overlay::FrameTimeWindow;

/// Never render the marcher below 1/4 of native resolution — much lower
/// than this and the upscaled image stops reading as "the same scene, just
/// softer" and starts reading as broken/blocky, which is worse for
/// perceived quality than a lower frame rate would be.
pub const MIN_SCALE: f32 = 0.25;

/// Never render above native resolution — there is no image-quality benefit
/// (the present pass only ever *downscales or matches*, never magnifies
/// past native without also upscaling from supersampled data we don't have),
/// and it would only cost more GPU time for zero visible gain.
pub const MAX_SCALE: f32 = 1.0;

/// Only step resolution *down* once the ~1s-averaged frame time
/// (`FrameTimeWindow::averaged`, already damped against single-frame
/// outliers and the startup pipeline-compile transient — see
/// `fps_overlay.rs`'s module doc) is sustainably worse than the 16.67ms
/// (60fps) target, not the instant it ticks over. 19ms is about 14% over
/// target (~53fps) — comfortably outside normal frame-to-frame jitter for a
/// scene that's actually hitting close to target, so this only fires for a
/// genuine, sustained shortfall.
const SCALE_DOWN_THRESHOLD_MS: f64 = 19.0;

/// Only step resolution *up* once frame time is sustainably *better* than
/// target by a similar margin (14ms is about 16% under the 16.67ms target,
/// ~71fps). Paired with `SCALE_DOWN_THRESHOLD_MS` this creates a ~5ms dead
/// zone (14-19ms) around the 16.67ms target where the scale is left alone —
/// this is the hysteresis: without a dead zone, a scale step that lands
/// frame time just on the other side of a single threshold would
/// immediately qualify to step back the other way next tick, oscillating
/// every adjustment interval instead of settling.
const SCALE_UP_THRESHOLD_MS: f64 = 14.0;

/// Multiplicative step per adjustment tick. Gradual (5%) rather than
/// jumping straight to a clamp extreme, so a resolution change isn't a
/// visible "pop" — it compounds over repeated sustained-bad/good ticks,
/// reaching the clamp range within a handful of adjustment intervals (at
/// ~1 per `ADJUST_INTERVAL_SECS`, that's a few seconds, not instant).
const SCALE_STEP_DOWN: f32 = 0.95;
const SCALE_STEP_UP: f32 = 1.05;

/// How often (seconds) the controller re-evaluates the scale. Originally
/// 0.25 (~4/sec), tuned assuming each adjustment was a cheap in-place
/// texture resize — every *actual* scale change (not just re-evaluation)
/// now rebuilds the render target as a whole new `Image` asset
/// (`present::resize_marcher_render_target`'s doc explains why: resizing
/// the same asset in place while a camera is actively rendering into it is
/// a known upstream Bevy bug that permanently freezes that camera's
/// output). That's a meaningfully heavier operation than a plain resize,
/// and at 4/sec during a multi-second convergence (observed: 14 real
/// resizes in 8 seconds under sustained load) it was visibly janky —
/// reported as "a lot of screen jitter," not the smooth ramp this was
/// supposed to be. Throttled much coarser here (a few adjustments over
/// several seconds, not per-frame) trades a bit of convergence speed for
/// actually looking smooth, which matters more for a background
/// quality/perf tradeoff than shaving a couple of seconds off how fast it
/// reacts to a real, sustained slowdown.
pub const ADJUST_INTERVAL_SECS: f64 = 1.5;

/// Given the current internal-resolution scale and the latest averaged
/// frame time (ms), returns the next scale to use: steps down if frame time
/// is sustainably above target, up if sustainably below, otherwise leaves it
/// unchanged (the dead zone — see `SCALE_DOWN_THRESHOLD_MS`/
/// `SCALE_UP_THRESHOLD_MS` docs above) — always clamped to
/// `[MIN_SCALE, MAX_SCALE]`.
pub fn next_scale(current_scale: f32, avg_frame_ms: f64) -> f32 {
    let stepped = if avg_frame_ms > SCALE_DOWN_THRESHOLD_MS {
        current_scale * SCALE_STEP_DOWN
    } else if avg_frame_ms < SCALE_UP_THRESHOLD_MS {
        current_scale * SCALE_STEP_UP
    } else {
        current_scale
    };
    stepped.clamp(MIN_SCALE, MAX_SCALE)
}

/// Rounds a native (window) pixel size and a scale factor into an actual
/// target pixel size for the offscreen render target, flooring each
/// dimension at 1px (a 0-sized `Image`/texture is invalid) — used by
/// `present::resize_marcher_render_target` both to compute the desired size
/// and, by comparing against the render target's *current* stored size, to
/// decide whether an actual (GPU-resource-touching) resize is warranted at
/// all, rather than resizing on every tiny float `scale` change.
pub fn target_pixel_size(native_size: UVec2, scale: f32) -> UVec2 {
    UVec2::new(
        ((native_size.x as f32 * scale).round() as u32).max(1),
        ((native_size.y as f32 * scale).round() as u32).max(1),
    )
}

/// Current adaptive internal-resolution scale (fraction of native window
/// pixel size the marcher actually renders at), plus the controller's own
/// throttle state.
#[derive(Resource)]
pub struct AdaptiveResolution {
    pub scale: f32,
    /// Wall-clock time (`Time::elapsed_secs_f64`) at/after which the next
    /// adjustment is allowed to run — see `ADJUST_INTERVAL_SECS`.
    next_adjust_at: f64,
}

impl Default for AdaptiveResolution {
    fn default() -> Self {
        // Start at native resolution: the controller only ever scales down
        // once real, sustained frame-time evidence justifies it (and scales
        // back up once that evidence goes away), rather than presuming
        // degraded quality upfront.
        Self { scale: MAX_SCALE, next_adjust_at: 0.0 }
    }
}

/// `MM_FORCE_RES_SCALE=<float>` pins `AdaptiveResolution::scale` to a fixed
/// value (still clamped to `[MIN_SCALE, MAX_SCALE]`) instead of letting the
/// controller react to frame time — a testing aid for exercising the
/// render-target-resize path (`present::resize_marcher_render_target`) at a
/// known, reproducible scale (e.g. confirming the offscreen image actually
/// shrinks and the upscaled result actually looks softer), matching this
/// codebase's other `MM_*` env-var testing hooks (`render.rs`'s `MM_SCENE`,
/// `main.rs`'s `MM_WINDOW_SIZE`). `std::env::var` always returns `Err` on
/// wasm32-unknown-unknown (no OS environment in a browser), so this has no
/// effect in the deployed web build.
fn forced_scale() -> Option<f32> {
    std::env::var("MM_FORCE_RES_SCALE").ok()?.parse().ok()
}

/// `Update` system: re-evaluates `AdaptiveResolution::scale` from the live
/// `FrameTimeWindow` average, throttled to `ADJUST_INTERVAL_SECS`. Does
/// nothing (not even the throttle-timer bookkeeping) if no frame-time
/// samples have been recorded yet — see `FrameTimeWindow::averaged`'s
/// `None` case (startup, before any frame has been timed).
pub fn adjust_resolution_scale(
    time: Res<Time>,
    frame_window: Res<FrameTimeWindow>,
    mut adaptive: ResMut<AdaptiveResolution>,
) {
    if let Some(forced) = forced_scale() {
        adaptive.scale = forced.clamp(MIN_SCALE, MAX_SCALE);
        return;
    }

    let now = time.elapsed_secs_f64();
    if now < adaptive.next_adjust_at {
        return;
    }
    let Some((_fps, avg_frame_ms)) = frame_window.averaged() else {
        return;
    };
    adaptive.next_adjust_at = now + ADJUST_INTERVAL_SECS;
    adaptive.scale = next_scale(adaptive.scale, avg_frame_ms);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dead_zone_leaves_scale_unchanged() {
        // Comfortably within the 14-19ms dead zone around the 16.67ms
        // target: this is the hysteresis itself, so it must be a no-op
        // across the whole zone, not just exactly at the target.
        assert_eq!(next_scale(1.0, 16.67), 1.0);
        assert_eq!(next_scale(0.5, 15.0), 0.5);
        assert_eq!(next_scale(0.5, 18.5), 0.5);
        assert_eq!(next_scale(0.7, 14.0), 0.7); // boundary itself: not "< 14"
        assert_eq!(next_scale(0.7, 19.0), 0.7); // boundary itself: not "> 19"
    }

    #[test]
    fn sustained_slow_frames_scale_down() {
        let scaled = next_scale(1.0, 25.0);
        assert!(scaled < 1.0, "expected scale to drop under sustained slow frames, got {scaled}");
    }

    #[test]
    fn sustained_fast_frames_scale_up() {
        let scaled = next_scale(0.5, 8.0);
        assert!(scaled > 0.5, "expected scale to rise under sustained fast frames, got {scaled}");
    }

    #[test]
    fn does_not_flip_flop_at_the_boundary() {
        // Regression check for the exact failure mode hysteresis exists to
        // prevent: repeatedly evaluating at a single frame-time value must
        // move monotonically (or stay put), never oscillate back and forth.
        let mut scale = 1.0f32;
        let mut prev = scale;
        for _ in 0..20 {
            scale = next_scale(scale, 25.0); // sustained-slow throughout
            assert!(scale <= prev, "scale rose while frame time stayed slow: {prev} -> {scale}");
            prev = scale;
        }
    }

    #[test]
    fn clamps_to_min_scale() {
        let mut scale = 1.0f32;
        for _ in 0..200 {
            scale = next_scale(scale, 100.0);
        }
        assert_eq!(scale, MIN_SCALE);
    }

    #[test]
    fn clamps_to_max_scale() {
        let mut scale = MIN_SCALE;
        for _ in 0..200 {
            scale = next_scale(scale, 1.0);
        }
        assert_eq!(scale, MAX_SCALE);
    }

    #[test]
    fn sustained_high_frame_time_moves_toward_target_over_iterations() {
        let mut scale = 1.0f32;
        for _ in 0..10 {
            scale = next_scale(scale, 30.0);
        }
        assert!(
            scale < 0.7,
            "expected substantial scale reduction after 10 sustained-slow ticks, got {scale}"
        );
    }

    #[test]
    fn sustained_low_frame_time_moves_toward_max_over_iterations() {
        let mut scale = 0.25f32;
        for _ in 0..10 {
            scale = next_scale(scale, 5.0);
        }
        assert!(
            scale > 0.35,
            "expected substantial scale increase after 10 sustained-fast ticks, got {scale}"
        );
    }

    #[test]
    fn target_pixel_size_matches_native_at_full_scale() {
        assert_eq!(target_pixel_size(UVec2::new(1280, 720), 1.0), UVec2::new(1280, 720));
    }

    #[test]
    fn target_pixel_size_scales_down_proportionally() {
        assert_eq!(target_pixel_size(UVec2::new(1280, 720), 0.5), UVec2::new(640, 360));
    }

    #[test]
    fn target_pixel_size_floors_at_one_pixel() {
        assert_eq!(target_pixel_size(UVec2::new(4, 4), 0.01), UVec2::new(1, 1));
    }
}
