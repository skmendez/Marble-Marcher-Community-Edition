//! Marble Marcher CSG renderer on Bevy — see rust/DESIGN.md §8.
//!
//! Builds the demo CSG scene, generates its WGSL, and renders it into a
//! fullscreen `Material2d` quad. A marble spawns at the demo level's start
//! position and simulates at a fixed 60 Hz (M5, DESIGN.md §7) against the
//! same `Object`/`Params` the shader renders, driven by WASD (camera-yaw
//! relative) with `R` to force a respawn; the orbit camera follows it.

mod camera;
mod config;
mod debug_gizmos;
mod debug_screenshot;
mod fps_overlay;
mod gpu;
mod gpu_profile;
mod live_debug;
mod mrrm;
mod net;
mod perfprobe;
mod physics_sys;
mod render;
mod shadow_pass;
mod step_data;
mod touch;
mod web_config;

use bevy::prelude::*;
use bevy::sprite::Material2dPlugin;
use bevy::window::WindowResolution;

use camera::{orbit_camera_input, CameraOrbit};
use config::Config;
use debug_gizmos::draw_thrust_debug;
use debug_screenshot::DebugScreenshotPlugin;
use fps_overlay::FpsOverlayPlugin;
use gpu::MarcherGpuPlugin;
use gpu_profile::GpuProfilePlugin;
use live_debug::{poll_live_debug_toggles, seed_live_debug_toggles};
use mrrm::{resize_coarse_render_target, setup_mrrm_pipeline, sync_coarse_quad_scale, CoarseMarcherMaterial};
use net::{
    handle_copy_button_click, poll_net_status, setup_networking, spawn_net_ui, sync_net_ui_text,
    update_copy_button_visibility, update_copy_feedback, CopyFeedback,
};
use perfprobe::{perfprobe_tick, spawn_perfprobe_overlay, update_perfprobe_overlay_text, PerfProbeState};
use physics_sys::{marble_physics_tick, PendingSceneSync};
use render::{
    apply_pending_scene_sync, finalize_marble_cubemap, oscillate_fine_resolution_tier,
    resize_fine_render_target, setup, sync_fine_render_target_and_present, sync_quad_scale,
    update_frame_data, FineMarcherMaterial, PresentMaterial,
};
use shadow_pass::{resize_shadow_render_target, setup_shadow_pipeline, sync_shadow_quad_scale, ShadowMarcherMaterial};
use step_data::{setup_step_data_pipeline, StepDataMaterial, StepDataPlugin};
use touch::{touch_camera_input, TouchDebugInfo};

/// `MM_WINDOW_SIZE=WxH` overrides the window's starting resolution — mainly
/// useful for testing on a software (CPU) Vulkan/GL fallback, where this
/// per-pixel ray marcher is far more expensive than on real GPU hardware.
fn window_resolution() -> WindowResolution {
    std::env::var("MM_WINDOW_SIZE")
        .ok()
        .and_then(|s| {
            let (w, h) = s.split_once('x')?;
            Some(WindowResolution::new(w.parse().ok()?, h.parse().ok()?))
        })
        .unwrap_or_else(|| WindowResolution::new(1280.0, 720.0))
}

/// `config.vsync_off` switches `PresentMode` from the default `AutoVsync`
/// to `AutoNoVsync` -- a diagnostic toggle (perf plan milestone 3) to tell
/// whether an observed frame-rate ceiling is a real GPU-throughput limit
/// (frame time stays the same with vsync off) or compositor/vsync pacing
/// (frame time drops). Read once at `App` construction (via `Config::
/// from_env`, below -- before any Bevy resource can exist, since
/// `PresentMode` has to feed into `WindowPlugin` up front) since it's fixed
/// for the life of the window in this app (no runtime toggle needed). Not a
/// recommendation to ship uncapped/tearing-prone rendering by default --
/// `AutoVsync` stays the default; this only exists so the question can
/// actually be answered on a real high-refresh device, which this dev
/// environment doesn't have.
fn present_mode(config: &Config) -> bevy::window::PresentMode {
    if config.vsync_off {
        bevy::window::PresentMode::AutoNoVsync
    } else {
        bevy::window::PresentMode::AutoVsync
    }
}

fn main() {
    // Read once, here, rather than as a `Startup` system: `present_mode`
    // has to feed into `WindowPlugin` before `App::new()` even runs, so no
    // resource could exist yet regardless (`config::Config`'s module doc).
    // Inserted as a resource below so every other system reads `Res<Config>`
    // instead of re-parsing the URL/environment itself.
    let config = Config::from_env();
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Marble Marcher CSG (Bevy)".into(),
                resolution: window_resolution(),
                // Web-only (no-op on native, per bevy_window 0.16.1's
                // `Window::fit_canvas_to_parent` doc comment): makes the
                // wasm/WebGPU build's canvas track its parent element's
                // *actual* CSS size (bevy_winit sets the canvas's inline
                // style to `width/height: 100%` of its parent at creation
                // — see bevy_winit 0.16.1 `system.rs::create_windows`),
                // which winit's own `ResizeObserver` on the canvas then
                // reports as real `WindowResized` events. Without this,
                // winit hard-codes the canvas's inline pixel size to the
                // startup `window_resolution()` (`web_sys::set_canvas_size`
                // at window creation), which beats `web/index.html`'s CSS
                // `canvas { width: 100vw; height: 100vh }` rule (inline
                // style always wins over a stylesheet type-selector) — so a
                // browser/viewport resize only visually stretches the
                // fixed-resolution canvas via that CSS instead of actually
                // re-rendering at the new size, which reads as blur/
                // stretching. `sync_quad_scale`/`update_material`
                // (render.rs) already recompute the quad scale and shader
                // aspect from `Window` every frame, so once real
                // `WindowResized` events flow, resize-follow is free.
                fit_canvas_to_parent: true,
                present_mode: present_mode(&config),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(config)
        // `MarcherGpuPlugin` owns the persistent per-frame GPU buffers every
        // material's bind group references (gpu.rs) -- added before the
        // material plugins so the buffers resource exists whenever a
        // material prepare system first runs.
        .add_plugins(MarcherGpuPlugin)
        // Registers a replacement render-graph node for the fine pass's
        // GPU timestamp queries (`?gpuprofile=1`) -- see `gpu_profile.rs`'s
        // module doc for why this needs its own node rather than Bevy's
        // built-in render diagnostics.
        .add_plugins(GpuProfilePlugin)
        // Cumulative ray-march step-count estimate (`?gpuprofile=1`'s
        // companion feature, `step_data.rs`) -- registers the
        // `EstimatedSteps` resource and its JS-reporting system. Does NOT add
        // `bevy_render`'s `GpuReadbackPlugin` itself: `DefaultPlugins`
        // already adds that unconditionally (`step_data.rs`'s `StepDataPlugin`
        // doc), so doing so again here would panic at startup.
        .add_plugins(StepDataPlugin)
        .add_plugins(Material2dPlugin::<FineMarcherMaterial>::default())
        .add_plugins(Material2dPlugin::<CoarseMarcherMaterial>::default())
        .add_plugins(Material2dPlugin::<ShadowMarcherMaterial>::default())
        .add_plugins(Material2dPlugin::<PresentMaterial>::default())
        .add_plugins(Material2dPlugin::<StepDataMaterial>::default())
        .add_plugins(FpsOverlayPlugin)
        .add_plugins(DebugScreenshotPlugin)
        .insert_resource(Time::<Fixed>::from_hz(60.0))
        .init_resource::<CameraOrbit>()
        .init_resource::<TouchDebugInfo>()
        .init_resource::<CopyFeedback>()
        .init_resource::<PendingSceneSync>()
        .init_resource::<PerfProbeState>()
        // `setup` (below) inserts `SceneState`/`MarbleState`/
        // `MultiplayerSession` directly rather than `init_resource`-ing any
        // of them here -- none has a scene-independent `Default` to speak
        // of (`MultiplayerSession::new_solo` needs the scene's actual
        // initial marble list, physics_sys.rs's doc), and `setup` is
        // chained first among these `Startup` systems, so every one of
        // them exists well before `FixedUpdate`'s `marble_physics_tick`
        // ever runs.
        // `setup_mrrm_pipeline` (mrrm.rs) needs `setup`'s `SceneState` (the
        // scene tree + params buffer, to build the coarse shader/material)
        // and corrects `setup`'s placeholder `FineMarcherMaterial::coarse`
        // handle once the real coarse render target exists.
        // `setup_shadow_pipeline` (shadow_pass.rs) needs both `setup`'s
        // `SceneState` and `setup_mrrm_pipeline`'s `CoarseRenderTarget`
        // (this pass's own warm-start source), and corrects `setup`'s
        // placeholder `FineMarcherMaterial::shadow` handle in turn.
        // `setup_step_data_pipeline` (step_data.rs) also needs
        // `setup_mrrm_pipeline`'s `CoarseRenderTarget` (this pass warm-starts
        // identically to the fine pass) -- a true no-op (spawns nothing) if
        // `?gpuprofile=1` is off.
        .add_systems(
            Startup,
            (
                // Only needs `Config`, already inserted above (not a
                // `Startup` system's output) -- placed first purely by
                // convention, not because anything later in this chain
                // depends on `LiveDebugToggles` existing yet.
                seed_live_debug_toggles,
                setup,
                setup_mrrm_pipeline,
                setup_shadow_pipeline,
                setup_step_data_pipeline,
                setup_networking,
                spawn_net_ui,
                spawn_perfprobe_overlay,
            )
                .chain(),
        )
        .add_systems(FixedUpdate, marble_physics_tick)
        .add_systems(
            Update,
            (
                // Split into two nested, individually-chained groups purely
                // because Bevy's `.chain()`-tuple trait impl has a fixed max
                // arity (20) and this list has since grown past it -- the
                // outer tuple is `.chain()`d too, so full sequential
                // ordering across both groups is preserved exactly as if
                // this were still one flat chained tuple.
                (
                    apply_pending_scene_sync,
                    resize_fine_render_target,
                    resize_coarse_render_target,
                    resize_shadow_render_target,
                    // Debug-only (`?res_oscillate=1`) manual smoothness check
                    // -- must run after `resize_fine_render_target` (so it
                    // sees this frame's up-to-date `max_size` if a real
                    // resize also just happened) and before
                    // `sync_fine_render_target_and_present` (so its
                    // `active_size` write applies on the same frame, not one
                    // frame late).
                    oscillate_fine_resolution_tier,
                    sync_fine_render_target_and_present,
                    sync_quad_scale,
                    sync_coarse_quad_scale,
                    sync_shadow_quad_scale,
                    orbit_camera_input,
                    touch_camera_input,
                    finalize_marble_cubemap,
                    // Must run before the three `update_*_material` systems
                    // below: a probe window transition's camera-active/
                    // step-override changes need to apply on the same frame
                    // they happen, not one frame late (see
                    // `perfprobe::perfprobe_tick`'s doc).
                    perfprobe_tick,
                )
                    .chain(),
                (
                    // Must run before `update_frame_data`: a same-frame
                    // `mrrm`/view-mode toggle (`live_debug.rs`) needs to
                    // reach this frame's uniforms, not land one frame late.
                    poll_live_debug_toggles,
                    // Writes all three passes' uniforms + the marble list
                    // into `MarcherFrameData` (render.rs) -- replaced the
                    // three per-pass `update_material`/
                    // `update_coarse_material`/`update_shadow_material`
                    // systems when per-frame data moved to persistent GPU
                    // buffers (gpu.rs).
                    update_frame_data,
                    update_perfprobe_overlay_text,
                    draw_thrust_debug.run_if(|config: Res<Config>| config.debug_enabled),
                    poll_net_status,
                    update_copy_button_visibility,
                    handle_copy_button_click,
                    update_copy_feedback,
                    sync_net_ui_text,
                )
                    .chain(),
            )
                .chain(),
        )
        .run();
}
