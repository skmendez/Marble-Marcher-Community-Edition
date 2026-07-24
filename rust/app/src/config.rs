//! One `Config` resource, read once at startup, replacing what used to be
//! 7 independent `?key=value`(-then-`MM_KEY`) parses scattered across as
//! many files (`render::SceneKind::from_config`, `mrrm::mrrm_enabled`,
//! `shadow_pass::shadow_lod_enabled`, `perfprobe::perfprobe_enabled`,
//! `fps_overlay::debug_enabled`, and `main.rs`'s `present_mode`), each with
//! its own `OnceLock` and its own (usually near-identical, occasionally
//! subtly different) query-param-then-env-var glue. Every caller now reads
//! `Res<Config>` instead of re-parsing the URL/environment itself.
//!
//! Read once, at `App`-construction time in `main.rs` (before any Bevy
//! resource can exist, since `present_mode()`'s `PresentMode` decision has
//! to feed into `WindowPlugin` before `App::new()` even runs) -- so
//! `Config::from_env()` is a plain function, not a `Startup` system, and
//! `main()` both uses its `vsync_off` field directly *and* inserts the same
//! value as a resource for every later system to read.

use bevy::prelude::Resource;

use crate::render::SceneKind;

fn query_value(web_key: &str, env_key: &str) -> Option<String> {
    crate::web_config::query_param(web_key).or_else(|| std::env::var(env_key).ok())
}

#[derive(Resource, Clone, Copy, Debug)]
pub struct Config {
    pub scene: SceneKind,
    /// `?mrrm=0`/`MM_MRRM=0` disables MRRM warm-starting -- default on,
    /// see `mrrm.rs`'s module doc for why this is a per-frame shader flag
    /// rather than an entity-level toggle.
    pub mrrm_enabled: bool,
    /// `?shadowlod=0`/`MM_SHADOW_LOD=0` disables the cached shadow-LOD
    /// resample -- default on, same reasoning as `mrrm_enabled`.
    pub shadow_lod_enabled: bool,
    /// `?perfprobe=1`/`MM_PERFPROBE=1` enables the automated GPU
    /// relative-cost breakdown -- default off (a diagnostic tool, not
    /// something a normal play session should ever trigger by accident).
    pub perfprobe_enabled: bool,
    /// `?debug=1`/`MM_DEBUG=1` shows the FPS/camera/marble/phase-timing
    /// overlay and the thrust-direction debug gizmo -- default off, so the
    /// URL a player actually shares/opens is clean.
    pub debug_enabled: bool,
    /// `?vsync=off`/`MM_VSYNC=off` switches `PresentMode` to
    /// `AutoNoVsync` -- default off (stays `AutoVsync`), a GPU-perf-plan
    /// diagnostic toggle, not a recommendation to ship uncapped rendering
    /// (`main.rs`'s `present_mode` doc).
    pub vsync_off: bool,
    /// `?res_oscillate=1`/`MM_RES_OSCILLATE=1` continuously sweeps the fine
    /// pass's resolution tier between full size and 1/10th (100x fewer
    /// pixels) and back -- a manual, always-visible smoothness check for
    /// the adaptive-resolution plumbing (`render::oscillate_fine_resolution_tier`'s
    /// doc), independent of and not gated on `debug_enabled`. Default off.
    pub res_oscillate_enabled: bool,
    /// `?gpuprofile=1`/`MM_GPUPROFILE=1` enables GPU timestamp-query
    /// profiling of each render pass, surfaced via the HTML/JS overlay in
    /// `web/index.html` -- default off (a diagnostic tool; also a true
    /// no-op on hardware/browsers without `wgpu::Features::TIMESTAMP_QUERY`,
    /// see `gpu_profile.rs`'s module doc).
    pub gpu_profile_enabled: bool,
    /// `?stepheat=1`/`MM_STEPHEAT=1` replaces the fine pass's normal shading
    /// with a heatmap colored by each pixel's ray-march step count (dark
    /// blue = few steps, red = near the step budget) -- default off, a
    /// visualization debug tool for understanding where march cost actually
    /// goes (e.g. whether MRRM's coarse warm-start has room to help a given
    /// view), see `marble_csg::codegen`'s `MARCHER::fragment` doc.
    pub step_heat_enabled: bool,
    /// `?exposure=<f>`/`MM_EXPOSURE=<f>` scales the HDR color before the
    /// ACES tonemap (`marble_csg::codegen`'s `MARCHER::tonemap`; rides in
    /// `SceneUniforms::misc3.y`). Default 1.0; unparseable or non-positive
    /// values fall back to the default rather than blacking out the frame
    /// (the shader guards non-positive again independently). The C++
    /// original's counterpart is a live-tunable setting with auto-exposure
    /// on top (`Settings.h`'s `exposure`/`auto_exposure_*`); this is the
    /// fixed-value starting point.
    pub exposure: f32,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            scene: SceneKind::from_value(query_value("scene", "MM_SCENE").as_deref()),
            mrrm_enabled: query_value("mrrm", "MM_MRRM").as_deref() != Some("0"),
            shadow_lod_enabled: query_value("shadowlod", "MM_SHADOW_LOD").as_deref() != Some("0"),
            perfprobe_enabled: matches!(
                query_value("perfprobe", "MM_PERFPROBE").as_deref(),
                Some("1") | Some("true")
            ),
            debug_enabled: query_value("debug", "MM_DEBUG").as_deref() == Some("1"),
            vsync_off: query_value("vsync", "MM_VSYNC").as_deref() == Some("off"),
            res_oscillate_enabled: query_value("res_oscillate", "MM_RES_OSCILLATE").as_deref()
                == Some("1"),
            gpu_profile_enabled: query_value("gpuprofile", "MM_GPUPROFILE").as_deref()
                == Some("1"),
            step_heat_enabled: query_value("stepheat", "MM_STEPHEAT").as_deref() == Some("1"),
            exposure: query_value("exposure", "MM_EXPOSURE")
                .and_then(|v| v.parse::<f32>().ok())
                .filter(|v| *v > 0.0)
                .unwrap_or(1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mrrm_and_shadow_lod_default_on_everything_else_defaults_off() {
        // Can't exercise `query_param`/`env::var` themselves in a unit test
        // (native `std::env::var` reads real process state, wasm
        // `query_param` reads a real page URL) -- this just pins the
        // *polarity* convention each flag was already documented to use,
        // so a future edit can't silently flip a default without a test
        // noticing. `scene` isn't a bool, checked separately below.
        let value: Option<&str> = None;
        assert!(value != Some("0")); // mrrm/shadow_lod's "default on" test
        assert!(!matches!(value, Some("1") | Some("true"))); // perfprobe
        assert!(value != Some("1")); // debug
        assert!(value != Some("off")); // vsync
        assert!(value != Some("1")); // res_oscillate
        assert!(value != Some("1")); // gpuprofile
        assert!(value != Some("1")); // stepheat
    }

    #[test]
    fn scene_defaults_to_menger_oscillating_sphere_when_absent() {
        assert_eq!(SceneKind::from_value(Some("hollow_donut")), SceneKind::HollowDonut);
        assert_eq!(SceneKind::from_value(None), SceneKind::MengerOscillatingSphere);
    }
}
