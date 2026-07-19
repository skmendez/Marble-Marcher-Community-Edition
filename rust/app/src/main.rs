//! Marble Marcher CSG renderer on Bevy — see rust/DESIGN.md §8.
//!
//! Builds the demo CSG scene, generates its WGSL, and renders it into a
//! fullscreen `Material2d` quad. A marble spawns at the demo level's start
//! position and simulates at a fixed 60 Hz (M5, DESIGN.md §7) against the
//! same `Object`/`Params` the shader renders, driven by WASD (camera-yaw
//! relative) with `R` to force a respawn; the orbit camera follows it.

mod camera;
mod debug_screenshot;
mod fps_overlay;
mod physics_sys;
mod render;
mod touch;

use bevy::prelude::*;
use bevy::sprite::Material2dPlugin;
use bevy::window::WindowResolution;

use camera::{orbit_camera_input, CameraOrbit};
use debug_screenshot::DebugScreenshotPlugin;
use fps_overlay::FpsOverlayPlugin;
use physics_sys::marble_physics_tick;
use render::{setup, sync_quad_scale, update_material, MarcherMaterial};
use touch::touch_camera_input;

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
                ..default()
            }),
            ..default()
        }))
        .add_plugins(Material2dPlugin::<MarcherMaterial>::default())
        .add_plugins(FpsOverlayPlugin)
        .add_plugins(DebugScreenshotPlugin)
        .insert_resource(Time::<Fixed>::from_hz(60.0))
        .init_resource::<CameraOrbit>()
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, marble_physics_tick)
        .add_systems(
            Update,
            (
                sync_quad_scale,
                orbit_camera_input,
                touch_camera_input,
                update_material,
            )
                .chain(),
        )
        .run();
}
