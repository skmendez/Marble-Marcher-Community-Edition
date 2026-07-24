//! Live (no-reload) debug toggles for `mrrm` and the step-heat view mode --
//! the two flags this app's investigation into MRRM's coarse-warm-start
//! aliasing (see `mrrm.rs`'s module doc, `codegen.rs`'s `CoarseStepHeat`/
//! `CoarseStepDistance` view modes) needs to flip *without* losing camera
//! position, since a reload resets the marble/camera back to the scene's
//! spawn point.
//!
//! Every other `?key=value` flag in this app (`config::Config`'s fields) is
//! deliberately read once at startup and applied via a page reload
//! (`web/index.html`'s `setUpConfigPanel` doc) -- that stays true here too
//! for every flag except these two. `mrrm`/`stepheat` still seed their
//! *initial* value from `Config` (so a shared `?mrrm=0` link still works,
//! and native/no-JS behavior is unchanged), but after startup this
//! resource's value is re-read from JS every frame instead of being fixed
//! for the life of the window -- see `seed_live_debug_toggles`/
//! `poll_live_debug_toggles` below.
//!
//! Deliberately a new sibling resource rather than making `Config` itself
//! mutable: `Config`'s own doc commits to "read once at startup, never
//! mutated" as a design choice for every *other* flag, and violating that
//! for two fields would make every future reader of `Config` re-check
//! whether a given field is actually still startup-only. `LiveDebugToggles`
//! makes the "this one mutates every frame" fact visible at the type level
//! instead.

use bevy::prelude::*;

use crate::config::Config;

/// The fine pass's step-heat debug view mode (`SceneUniforms.misc3.x`,
/// `render.rs`) -- a 4-value integer selector, not a bool, now that the
/// coarse pass's own step count/hit distance are also visualizable. Numeric
/// values are the exact `misc3.x` wire encoding read by `codegen.rs`'s
/// `MARCHER::fragment` (`0.0`/`1.0`/`2.0`/`3.0`) -- keep `as_uniform_value`
/// in sync with the `scene.misc3.x > 0.5`/`> 1.5`/`> 2.5` comparisons there
/// if this enum ever grows another variant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DebugViewMode {
    #[default]
    Off,
    /// The original `?stepheat=1` behavior: the *fine* pass's own per-pixel
    /// `iters`, normalized against `FINE_MAX_STEPS`.
    FineStepHeat,
    /// The MRRM coarse pre-pass's own per-texel `iters`, upscaled onto the
    /// fine image and normalized against `MAX_STEPS` (the coarse pass's own,
    /// larger budget -- `codegen.rs`'s `COARSE_MARCHER` doc). Only ever
    /// constructed via `from_i32`, below, which only the wasm-only
    /// `poll_live_debug_toggles` calls outside of this module's own tests --
    /// so a plain (non-test) native build never constructs this variant,
    /// hence the `allow`.
    #[allow(dead_code)]
    CoarseStepHeat,
    /// The MRRM coarse pre-pass's own cached hit distance, upscaled onto
    /// the fine image via `coarse_distance_color`'s teal/white/magenta
    /// gradient. Same native-only-dead-code reasoning as `CoarseStepHeat`.
    #[allow(dead_code)]
    CoarseStepDistance,
}

impl DebugViewMode {
    /// JS-side encoding (`web/index.html`'s `viewMode` dropdown) and this
    /// enum's own discriminant happen to already agree (`0..=3` in
    /// declaration order), but this match is written out explicitly rather
    /// than relying on that so a future reordering of the enum can't
    /// silently desync the wire value. Only called from the wasm-only
    /// `poll_live_debug_toggles` outside of tests, so a plain native build
    /// never calls it -- same reasoning as `CoarseStepHeat`'s `allow`.
    #[allow(dead_code)]
    fn from_i32(v: i32) -> Self {
        match v {
            1 => Self::FineStepHeat,
            2 => Self::CoarseStepHeat,
            3 => Self::CoarseStepDistance,
            _ => Self::Off,
        }
    }

    /// The `SceneUniforms.misc3.x` value this mode writes, consumed by
    /// `codegen.rs`'s `MARCHER::fragment` (`scene.misc3.x > 0.5`/`> 1.5`/
    /// `> 2.5`).
    pub fn as_uniform_value(self) -> f32 {
        match self {
            Self::Off => 0.0,
            Self::FineStepHeat => 1.0,
            Self::CoarseStepHeat => 2.0,
            Self::CoarseStepDistance => 3.0,
        }
    }
}

/// `mrrm`/step-heat-view-mode, live-toggleable with no page reload --
/// everything else stays on `Config` (module doc above). Seeded once from
/// `Config` at `Startup` (`seed_live_debug_toggles`), then re-read from JS
/// every `Update` frame on wasm (`poll_live_debug_toggles`); native has no
/// live-toggle mechanism (no JS panel to poll), so these two fields are
/// simply fixed at their seeded value for the life of a native run, same as
/// every `Config` field already is there.
#[derive(Resource, Clone, Copy, Debug)]
pub struct LiveDebugToggles {
    pub mrrm_enabled: bool,
    pub view_mode: DebugViewMode,
}

pub fn seed_live_debug_toggles(mut commands: Commands, config: Res<Config>) {
    commands.insert_resource(LiveDebugToggles {
        mrrm_enabled: config.mrrm_enabled,
        view_mode: if config.step_heat_enabled {
            DebugViewMode::FineStepHeat
        } else {
            DebugViewMode::Off
        },
    });
}

/// Re-reads `mrrm`/`viewMode` from `web/index.html`'s `window.mmLiveDebug`
/// object (via `net.rs`'s `js_bridge::live_mrrm_enabled`/`live_view_mode`)
/// every frame -- cheap (two `extern "C"` calls into already-marshalled JS
/// state, same cost class as `net.rs`'s `poll_net_status`), and must run
/// before `render::update_frame_data` so a same-frame toggle reaches this
/// frame's uniforms rather than landing one frame late (`main.rs`'s
/// `Update` chain ordering).
#[cfg(target_arch = "wasm32")]
pub fn poll_live_debug_toggles(mut toggles: ResMut<LiveDebugToggles>) {
    toggles.mrrm_enabled = crate::net::js_bridge::live_mrrm_enabled();
    toggles.view_mode = DebugViewMode::from_i32(crate::net::js_bridge::live_view_mode());
}

/// Native has no live-toggle JS to poll -- a true no-op, leaving
/// `LiveDebugToggles` at whatever `seed_live_debug_toggles` set it to for
/// the life of the run (matches every other flag's native behavior).
#[cfg(not(target_arch = "wasm32"))]
pub fn poll_live_debug_toggles(_toggles: ResMut<LiveDebugToggles>) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_i32_round_trips_through_as_uniform_value() {
        for mode in [
            DebugViewMode::Off,
            DebugViewMode::FineStepHeat,
            DebugViewMode::CoarseStepHeat,
            DebugViewMode::CoarseStepDistance,
        ] {
            let wire = mode.as_uniform_value();
            assert_eq!(DebugViewMode::from_i32(wire as i32), mode);
        }
    }

    #[test]
    fn from_i32_falls_back_to_off_for_an_out_of_range_value() {
        assert_eq!(DebugViewMode::from_i32(-1), DebugViewMode::Off);
        assert_eq!(DebugViewMode::from_i32(4), DebugViewMode::Off);
    }
}
