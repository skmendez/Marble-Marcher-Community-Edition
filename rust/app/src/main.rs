//! Marble Marcher CSG renderer on Bevy — see rust/DESIGN.md §8.
//!
//! M4: builds the demo CSG scene, generates its WGSL, and renders it into a
//! fullscreen `Material2d` quad, with a free-orbit camera and a proof-of
//! -concept parameter animation (no shader recompile). Marble physics/input
//! land in M5 — `marble.w` stays 0 here.

mod camera;
mod render;

use bevy::prelude::*;
use bevy::sprite::Material2dPlugin;

use camera::{orbit_camera_input, CameraOrbit};
use render::{setup, sync_quad_scale, update_material, MarcherMaterial};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Marble Marcher CSG (Bevy)".into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(Material2dPlugin::<MarcherMaterial>::default())
        .init_resource::<CameraOrbit>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (sync_quad_scale, orbit_camera_input, update_material).chain(),
        )
        .run();
}
