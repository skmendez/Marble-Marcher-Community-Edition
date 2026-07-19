//! Game camera: orbits a target (the marble, once M5 spawns one) via
//! yaw/pitch/roll/distance (DESIGN.md §7/§8).

use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;

/// `pub(crate)` (not private) so `touch.rs`'s swipe handler can clamp pitch
/// identically to the mouse handler below, instead of duplicating the value.
pub(crate) const PITCH_LIMIT: f32 = 1.5;
const MIN_DISTANCE: f32 = 0.12;
const MAX_DISTANCE: f32 = 20.0;
pub(crate) const YAW_SENSITIVITY: f32 = 0.006;
pub(crate) const PITCH_SENSITIVITY: f32 = 0.006;
const ZOOM_SENSITIVITY: f32 = 0.5;

/// Yaw/pitch/roll/distance around an externally supplied target (see
/// [`CameraOrbit::eye_and_basis`]) — the target itself (the marble's
/// position) lives in `MarbleState`, not here, so this resource stays pure
/// input state.
#[derive(Resource, Clone, Copy, Debug)]
pub struct CameraOrbit {
    pub yaw: f32,
    pub pitch: f32,
    /// Camera roll around the forward (view) axis, radians. Driven only by
    /// the touch two-finger-rotate gesture (`touch.rs`) — there's no mouse
    /// equivalent today. Zero means "no roll": `eye_and_basis` must produce
    /// the exact same `right`/`up` as before this field existed when
    /// `roll == 0.0` (see that method's doc).
    pub roll: f32,
    pub distance: f32,
}

impl Default for CameraOrbit {
    fn default() -> Self {
        Self {
            // Aimed roughly along the outward surface normal at the demo
            // scene's actual resting spot (computed once, offline, from
            // Object::nearest_point at the settled marble position — see
            // rust/csg/examples/ in git history) rather than an arbitrary
            // angle: with the marble sitting flush against the fractal
            // surface (radius-thin clearance), most viewing directions look
            // straight into solid geometry or get blocked by the nearby
            // decorative "creme spheres" clutter, and this is the direction
            // that was actually verified (via a close-up screenshot) to see
            // it. Not a general solution — a real fix would auto-orient the
            // camera to the marble's contact normal at spawn/settle time,
            // which isn't implemented yet.
            yaw: -1.448,
            pitch: 0.899,
            roll: 0.0,
            // Much closer than DESIGN.md §7's original
            // `orbit_dist * marble_rad / 0.035` MMCE-scaling formula
            // (~1.77): that reads as "the marble is a barely-visible speck"
            // at normal viewing distances (verified this session — even
            // with a saturated marker color, it was only a handful of
            // pixels across, ~960x540 screenshot). Picked by comparing
            // rendered screenshots at several candidate distances: 0.45
            // made the marble clearly visible but still smallish (~35px
            // diameter in a 540px-tall frame); 0.2 reads much better — the
            // marble is unmistakably the visual focus (~60px diameter,
            // ~11% of frame height) while the surrounding fractal surface
            // detail and creme-sphere background are still clearly
            // visible, with no clipping/occlusion artifacts at this
            // yaw/pitch. `MIN_DISTANCE` below is set just under this so
            // scroll-zoom still has a little headroom to go closer.
            distance: 0.2,
        }
    }
}

impl CameraOrbit {
    /// Eye position and right-handed camera basis (right, up, forward)
    /// looking at `target` (DESIGN.md §8): up = +Y,
    /// forward = normalize(target - eye), right = normalize(cross(forward, up)),
    /// up' = cross(right, forward) — then, if `roll != 0`, `right`/`up` are
    /// additionally rotated within their own plane by `roll` radians around
    /// `forward` (a pure camera roll: `forward` is unaffected, `right`/`up`
    /// stay orthonormal). At `roll == 0.0` this is `cos(0)=1, sin(0)=0`, so
    /// `right`/`up` come out identical to the pre-roll computation —
    /// existing native mouse/keyboard framing is unaffected by this field's
    /// existence.
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

        let (sin_r, cos_r) = self.roll.sin_cos();
        let rolled_right = right * cos_r + up * sin_r;
        let rolled_up = up * cos_r - right * sin_r;

        (eye, rolled_right, rolled_up, forward)
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
