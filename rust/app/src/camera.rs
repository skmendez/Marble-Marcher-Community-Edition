//! Game camera: orbits a target (the marble, once M5 spawns one) via
//! yaw/pitch/distance (DESIGN.md §7/§8).

use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;

use marble_csg::scenes::beware_of_bumps;

const PITCH_LIMIT: f32 = 1.5;
const MIN_DISTANCE: f32 = 0.2;
const MAX_DISTANCE: f32 = 20.0;
const YAW_SENSITIVITY: f32 = 0.006;
const PITCH_SENSITIVITY: f32 = 0.006;
const ZOOM_SENSITIVITY: f32 = 0.5;

/// Yaw/pitch/distance around an externally supplied target (see
/// [`CameraOrbit::eye_and_basis`]) — the target itself (the marble's
/// position) lives in `MarbleState`, not here, so this resource stays pure
/// input state.
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
            // DESIGN.md §7: orbit_dist * marble_rad / 0.035 (Beware Of Bumps'
            // level values), i.e. the original MMCE camera-distance scaling.
            distance: beware_of_bumps::ORBIT_DIST * beware_of_bumps::MARBLE_RAD / 0.035,
        }
    }
}

impl CameraOrbit {
    /// Eye position and right-handed camera basis (right, up, forward)
    /// looking at `target` (DESIGN.md §8): up = +Y,
    /// forward = normalize(target - eye), right = normalize(cross(forward, up)),
    /// up' = cross(right, forward).
    pub fn eye_and_basis(&self, target: Vec3) -> (Vec3, Vec3, Vec3, Vec3) {
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
