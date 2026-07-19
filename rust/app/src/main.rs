//! Marble Marcher CSG renderer on Bevy — see rust/DESIGN.md §8.
//!
//! Builds the demo CSG scene, generates its WGSL, and renders it into a
//! fullscreen `Material2d` quad. A marble spawns at the demo level's start
//! position and simulates at a fixed 60 Hz (M5, DESIGN.md §7) against the
//! same `Object`/`Params` the shader renders, driven by WASD (camera-yaw
//! relative) with `R` to force a respawn; the orbit camera follows it.

mod camera;
mod debug_screenshot;
mod physics_sys;
mod render;

use bevy::prelude::*;
use bevy::sprite::Material2dPlugin;
use bevy::window::WindowResolution;

use camera::{orbit_camera_input, CameraOrbit};
use debug_screenshot::DebugScreenshotPlugin;
use physics_sys::{marble_physics_tick, MarbleState};
use render::{setup, sync_quad_scale, update_material, MarcherMaterial};

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
                ..default()
            }),
            ..default()
        }))
        .add_plugins(Material2dPlugin::<MarcherMaterial>::default())
        .add_plugins(DebugScreenshotPlugin)
        .insert_resource(Time::<Fixed>::from_hz(60.0))
        .init_resource::<CameraOrbit>()
        .init_resource::<MarbleState>()
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, marble_physics_tick)
        .add_systems(
            Update,
            (sync_quad_scale, orbit_camera_input, update_material).chain(),
        )
        .run();
}
