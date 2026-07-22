//! Marble Marcher CSG renderer on Bevy — see rust/DESIGN.md §8.
//!
//! Builds the demo CSG scene, generates its WGSL, and renders it into a
//! fullscreen `Material2d` quad. A marble spawns at the demo level's start
//! position and simulates at a fixed 60 Hz (M5, DESIGN.md §7) against the
//! same `Object`/`Params` the shader renders, driven by WASD (camera-yaw
//! relative) with `R` to force a respawn; the orbit camera follows it.

mod camera;
mod debug_gizmos;
mod debug_screenshot;
mod fps_overlay;
mod gpu;
mod mrrm;
mod net;
mod perfprobe;
mod physics_sys;
mod render;
mod shadow_pass;
mod touch;
mod web_config;

use bevy::prelude::*;
use bevy::sprite::Material2dPlugin;
use bevy::window::WindowResolution;

use camera::{orbit_camera_input, CameraOrbit};
use debug_gizmos::draw_thrust_debug;
use debug_screenshot::DebugScreenshotPlugin;
use fps_overlay::{debug_enabled, FpsOverlayPlugin};
use gpu::MarcherGpuPlugin;
use mrrm::{resize_coarse_render_target, setup_mrrm_pipeline, sync_coarse_quad_scale, CoarseMarcherMaterial};
use net::{
    handle_copy_button_click, poll_net_status, setup_networking, spawn_net_ui, sync_net_ui_text,
    update_copy_button_visibility, update_copy_feedback, CopyFeedback,
};
use perfprobe::{perfprobe_tick, spawn_perfprobe_overlay, update_perfprobe_overlay_text, PerfProbeState};
use physics_sys::{marble_physics_tick, PendingSceneSync};
use render::{apply_pending_scene_sync, finalize_marble_cubemap, setup, sync_quad_scale, update_frame_data, FineMarcherMaterial};
use shadow_pass::{resize_shadow_render_target, setup_shadow_pipeline, sync_shadow_quad_scale, ShadowMarcherMaterial};
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

/// `?vsync=off` (web) / `MM_VSYNC=off` (native) switches `PresentMode` from
/// the default `AutoVsync` to `AutoNoVsync` -- a diagnostic toggle (perf
/// plan milestone 3) to tell whether an observed frame-rate ceiling is a
/// real GPU-throughput limit (frame time stays the same with vsync off) or
/// compositor/vsync pacing (frame time drops). Same query-param-then-env
/// layering as `mrrm::mrrm_enabled`/`shadow_pass::shadow_lod_enabled`, read
/// once at `App` construction since `PresentMode` is fixed for the life of
/// the window in this app (no runtime toggle needed). Not a recommendation
/// to ship uncapped/tearing-prone rendering by default -- `AutoVsync` stays
/// the default; this only exists so the question can actually be answered
/// on a real high-refresh device, which this dev environment doesn't have.
fn present_mode() -> bevy::window::PresentMode {
    let value = crate::web_config::query_param("vsync").or_else(|| std::env::var("MM_VSYNC").ok());
    if value.as_deref() == Some("off") {
        bevy::window::PresentMode::AutoNoVsync
    } else {
        bevy::window::PresentMode::AutoVsync
    }
}

fn main() {
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
                present_mode: present_mode(),
                ..default()
            }),
            ..default()
        }))
        // `MarcherGpuPlugin` owns the persistent per-frame GPU buffers every
        // material's bind group references (gpu.rs) -- added before the
        // material plugins so the buffers resource exists whenever a
        // material prepare system first runs.
        .add_plugins(MarcherGpuPlugin)
        .add_plugins(Material2dPlugin::<FineMarcherMaterial>::default())
        .add_plugins(Material2dPlugin::<CoarseMarcherMaterial>::default())
        .add_plugins(Material2dPlugin::<ShadowMarcherMaterial>::default())
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
        .add_systems(
            Startup,
            (
                setup,
                setup_mrrm_pipeline,
                setup_shadow_pipeline,
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
                apply_pending_scene_sync,
                resize_coarse_render_target,
                resize_shadow_render_target,
                sync_quad_scale,
                sync_coarse_quad_scale,
                sync_shadow_quad_scale,
                orbit_camera_input,
                touch_camera_input,
                finalize_marble_cubemap,
                // Must run before the three `update_*_material` systems below:
                // a probe window transition's camera-active/step-override
                // changes need to apply on the same frame they happen, not one
                // frame late (see `perfprobe::perfprobe_tick`'s doc).
                perfprobe_tick,
                // Writes all three passes' uniforms + the marble list into
                // `MarcherFrameData` (render.rs) -- replaced the three
                // per-pass `update_material`/`update_coarse_material`/
                // `update_shadow_material` systems when per-frame data moved
                // to persistent GPU buffers (gpu.rs).
                update_frame_data,
                update_perfprobe_overlay_text,
                draw_thrust_debug.run_if(debug_enabled),
                poll_net_status,
                update_copy_button_visibility,
                handle_copy_button_click,
                update_copy_feedback,
                sync_net_ui_text,
            )
                .chain(),
        )
        .run();
}
