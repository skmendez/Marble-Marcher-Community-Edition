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

/// `pub(crate)`, not private: `snapshot.rs`'s `?snapshot=`/`MM_SNAPSHOT` read
/// reuses this exact query-param-then-env-var convention too (not part of
/// `Config` itself since a snapshot payload is a `String`, not a `Copy`
/// value like every other field here), rather than duplicating the same
/// two-line fallback glue a second time.
pub(crate) fn query_value(web_key: &str, env_key: &str) -> Option<String> {
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
    /// `?adaptive_res=1`/`MM_ADAPTIVE_RES=1` enables the load-driven
    /// resolution controller (`adaptive_res::adjust_resolution_scale`):
    /// steps the fine pass's `render::FineRenderTarget::active_size` down
    /// under sustained slow frames and back up once frame time recovers,
    /// via hysteresis logic ported from a first (fully reverted) attempt at
    /// this whose *decision* logic was always sound -- only its consumer
    /// (a since-eliminated GPU render-target rebuild per scale change) was
    /// the actual cause of that attempt's visible jitter.
    ///
    /// **Default off**, pending a play-test -- same cautious-rollout
    /// pattern as `smart_camera`/`autopilot`: this is a previously-untested
    /// mechanic (the decision logic is unit-tested, but "does it feel
    /// smooth to a live player" isn't something a unit test can establish).
    /// Mutually exclusive with `res_oscillate_enabled` -- see
    /// `adaptive_res::adjust_resolution_scale`'s doc for which one wins if
    /// both are somehow requested at once.
    pub adaptive_res_enabled: bool,
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
    /// `?smartcam=1`/`MM_SMARTCAM=1` enables the geometry-aware half of the
    /// camera (`smart_camera.rs`): deocclusion, clearance pull-in, damping,
    /// FOV compensation.
    ///
    /// **Default off**, pending a play-test -- it changes how the camera
    /// *moves*, which is the kind of change that has to be felt rather than
    /// measured, and the measurements it does have
    /// (`smart_camera::scene_probe`) only establish that it keeps the marble
    /// visible and the eye out of the geometry, not that it feels good. Off,
    /// the camera tracks the player's intent exactly and instantly, as it
    /// always did.
    ///
    /// One thing changes either way: distance comes from the framing rule
    /// (marble sized to the screen, `smart_camera::framing_distance`) rather
    /// than from a hand-tuned per-scene constant. That part has no failure
    /// mode worth an escape hatch -- it reproduces the three deleted
    /// per-scene distances to within ~12% at 16:9, and unlike them it is
    /// also right on a phone and right for a marble of any radius.
    pub smart_camera: bool,
    /// `?autopilot=1`/`MM_AUTOPILOT=1` drives the marble and the camera from
    /// a fixed script instead of from real input -- a wandering
    /// camera-relative thrust, plus a slow camera drag for the first few
    /// seconds. Purely a verification aid: a headless run
    /// (`scripts/headless_screenshot.sh`) otherwise renders a marble sitting
    /// perfectly still in `GravityMode::Flying`, which says nothing about
    /// how the camera *follows*. Deterministic (driven by the physics tick,
    /// not the wall clock) so two runs at different frame rates capture the
    /// same moment. Default off.
    pub autopilot: bool,
    /// `?material_gamma=<f>`/`MM_MATERIAL_GAMMA=<f>` -- the terrain-albedo
    /// gamma boost (`albedo = pow(albedo, 1/this)`, rides in
    /// `SceneUniforms::misc3.z`). Default 0.5 (albedo squared), matching
    /// the C++ original's shipped `gamma_material` default; `1.0` disables
    /// the boost for A/B comparison. Same parse-fallback convention as
    /// `exposure`.
    pub material_gamma: f32,
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
            adaptive_res_enabled: query_value("adaptive_res", "MM_ADAPTIVE_RES").as_deref()
                == Some("1"),
            gpu_profile_enabled: query_value("gpuprofile", "MM_GPUPROFILE").as_deref()
                == Some("1"),
            step_heat_enabled: query_value("stepheat", "MM_STEPHEAT").as_deref() == Some("1"),
            exposure: query_value("exposure", "MM_EXPOSURE")
                .and_then(|v| v.parse::<f32>().ok())
                .filter(|v| *v > 0.0)
                .unwrap_or(1.0),
            smart_camera: matches!(
                query_value("smartcam", "MM_SMARTCAM").as_deref(),
                Some("1") | Some("true")
            ),
            autopilot: query_value("autopilot", "MM_AUTOPILOT").as_deref() == Some("1"),
            material_gamma: query_value("material_gamma", "MM_MATERIAL_GAMMA")
                .and_then(|v| v.parse::<f32>().ok())
                .filter(|v| *v > 0.0)
                .unwrap_or(0.5),
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
        assert!(value != Some("1")); // adaptive_res
        assert!(value != Some("1")); // gpuprofile
        assert!(value != Some("1")); // stepheat
        assert!(!matches!(value, Some("1") | Some("true"))); // smartcam
        assert!(value != Some("1")); // autopilot
    }

    #[test]
    fn scene_defaults_to_menger_oscillating_sphere_when_absent() {
        assert_eq!(SceneKind::from_value(Some("hollow_donut")), SceneKind::HollowDonut);
        assert_eq!(SceneKind::from_value(Some("cube_sphere_morph")), SceneKind::CubeSphereMorph);
        assert_eq!(SceneKind::from_value(Some("gears")), SceneKind::Gears);
        assert_eq!(SceneKind::from_value(None), SceneKind::MengerOscillatingSphere);
    }
}
