//! Free-orbit camera around the scene origin (DESIGN.md §8). Not the M5 game
//! camera (which orbits the marble) — this is purely for looking at the demo
//! scene while there's no marble to follow yet.

use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;

const PITCH_LIMIT: f32 = 1.5;
const MIN_DISTANCE: f32 = 0.5;
const MAX_DISTANCE: f32 = 20.0;
const YAW_SENSITIVITY: f32 = 0.006;
const PITCH_SENSITIVITY: f32 = 0.006;
const ZOOM_SENSITIVITY: f32 = 0.5;

/// Yaw/pitch/distance around a fixed target at the scene origin.
#[derive(Resource, Clone, Copy, Debug)]
pub struct CameraOrbit {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}

impl Default for CameraOrbit {
    fn default() -> Self {
        Self {
            yaw: 0.8,
            pitch: 0.35,
            distance: 8.0,
        }
    }
}

impl CameraOrbit {
    /// Eye position and right-handed camera basis (right, up, forward)
    /// looking at the origin (DESIGN.md §8): up = +Y,
    /// forward = normalize(target - eye), right = normalize(cross(forward, up)),
    /// up' = cross(right, forward).
    pub fn eye_and_basis(&self) -> (Vec3, Vec3, Vec3, Vec3) {
        let target = Vec3::ZERO;
        let offset = self.distance
            * Vec3::new(
                self.pitch.cos() * self.yaw.sin(),
                self.pitch.sin(),
                self.pitch.cos() * self.yaw.cos(),
            );
        let eye = target + offset;
        let world_up = Vec3::Y;
        let forward = (target - eye).normalize();
        let right = forward.cross(world_up).normalize();
        let up = right.cross(forward);
        (eye, right, up, forward)
    }
}

/// Left-drag to orbit, scroll wheel to zoom.
pub fn orbit_camera_input(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut motion: EventReader<MouseMotion>,
    mut wheel: EventReader<MouseWheel>,
    mut orbit: ResMut<CameraOrbit>,
) {
    if mouse_buttons.pressed(MouseButton::Left) {
        for ev in motion.read() {
            orbit.yaw -= ev.delta.x * YAW_SENSITIVITY;
            orbit.pitch =
                (orbit.pitch - ev.delta.y * PITCH_SENSITIVITY).clamp(-PITCH_LIMIT, PITCH_LIMIT);
        }
    } else {
        motion.clear();
    }

    for ev in wheel.read() {
        orbit.distance = (orbit.distance - ev.y * ZOOM_SENSITIVITY).clamp(MIN_DISTANCE, MAX_DISTANCE);
    }
}
