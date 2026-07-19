//! Game camera: orbits a target (the marble) via a free 3D orientation
//! quaternion + distance (DESIGN.md §7/§8).

use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;

pub(crate) const YAW_SENSITIVITY: f32 = 0.006;
pub(crate) const PITCH_SENSITIVITY: f32 = 0.006;
const MIN_DISTANCE: f32 = 0.12;
const MAX_DISTANCE: f32 = 20.0;
const ZOOM_SENSITIVITY: f32 = 0.5;

/// Full 3D orbit orientation + distance around an externally supplied
/// target (see [`CameraOrbit::eye_and_basis`]) — the target itself (the
/// marble's position) lives in `MarbleState`, not here, so this resource
/// stays pure input state.
///
/// `orientation` replaces an earlier yaw/pitch/roll Euler-angle
/// representation. Composing every rotation (mouse/touch drag, two-finger
/// roll) directly onto this quaternion, instead of decomposing into and
/// reconstructing from separate persistent angles each frame, is what
/// avoids gimbal lock — there's no "pitch" value that can approach +/-90
/// degrees and collapse yaw/roll onto the same axis (reported as feeling
/// gimbal-locked when rotating what looked like the Y axis, under the old
/// representation). This representation has no such singularity anywhere:
/// `drag`/`roll` below only ever apply an *incremental* rotation relative
/// to whatever the current orientation already is.
#[derive(Resource, Clone, Copy, Debug)]
pub struct CameraOrbit {
    pub orientation: Quat,
    pub distance: f32,
}

impl CameraOrbit {
    /// Builds the orientation quaternion equivalent to the old yaw/pitch
    /// Euler-angle scheme, for setting a specific *starting* view (this
    /// type's `Default` impl, and `render.rs`'s per-scene camera presets) —
    /// not used for ongoing input, which composes incremental rotations
    /// directly (`drag`/`roll`) rather than ever reconstructing from a
    /// yaw/pitch pair. `eye_and_basis` derives `forward = orientation *
    /// Vec3::NEG_Z`; this composition reproduces the exact same forward
    /// direction the old `Vec3::new(pitch.cos()*yaw.sin(), pitch.sin(),
    /// pitch.cos()*yaw.cos())` offset formula did, so any old yaw/pitch
    /// preset value still gives the identical starting view.
    pub fn orientation_from_yaw_pitch(yaw: f32, pitch: f32) -> Quat {
        Quat::from_rotation_y(yaw) * Quat::from_rotation_x(-pitch)
    }
}

impl Default for CameraOrbit {
    fn default() -> Self {
        Self {
            // Same starting view as the old yaw=-1.448/pitch=0.899 Euler
            // angles (aimed at the demo scene's marble resting-surface
            // normal — computed once, offline, from Object::nearest_point
            // at the settled marble position — see rust/csg/examples/ in
            // git history), expressed via `orientation_from_yaw_pitch`.
            // With the marble sitting flush against the fractal surface
            // (radius-thin clearance), most viewing directions look
            // straight into solid geometry or get blocked by the nearby
            // decorative "creme spheres" clutter, and this is the direction
            // that was actually verified (via a close-up screenshot) to see
            // it. Not a general solution — a real fix would auto-orient the
            // camera to the marble's contact normal at spawn/settle time,
            // which isn't implemented yet.
            orientation: Self::orientation_from_yaw_pitch(-1.448, 0.899),
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
    /// Eye position and orthonormal camera basis (right, up, forward)
    /// looking at `target`, `distance` back along `-forward` (DESIGN.md §8).
    pub fn eye_and_basis(&self, target: Vec3) -> (Vec3, Vec3, Vec3, Vec3) {
        let forward = self.forward();
        let right = self.orientation * Vec3::X;
        let up = self.orientation * Vec3::Y;
        let eye = target - forward * self.distance;
        (eye, right, up, forward)
    }

    /// The direction the camera is currently looking, independent of any
    /// target (`physics_sys.rs` uses this alone, decomposed back into a
    /// yaw/pitch pair, to feed `marble_csg::step_marble`'s camera-relative
    /// thrust — see that call site's doc for why that's safe).
    pub fn forward(&self) -> Vec3 {
        self.orientation * Vec3::NEG_Z
    }

    /// Applies a horizontal/vertical drag (mouse drag or single-finger
    /// swipe) as incremental rotations: horizontal around the *world* Y
    /// axis (so a horizontal drag always spins the view the same way
    /// regardless of current pitch — a stable "compass" feel), vertical
    /// around the camera's *own current* local right axis (so it always
    /// tilts relative to however the camera is presently facing). This
    /// ordering — world-axis yaw pre-multiplied, local-axis pitch
    /// post-multiplied — is the standard gimbal-lock-free orbit/FPS camera
    /// composition. No clamping of the result: a full quaternion has no
    /// pole to clamp away from, and clamping isn't expressible without
    /// decomposing back into Euler angles, which would reintroduce the
    /// exact gimbal-lock hazard this representation exists to avoid — the
    /// camera can now genuinely orbit all the way over the top/bottom, as
    /// requested ("truly global" movement).
    pub(crate) fn drag(&mut self, delta: Vec2) {
        let yaw = Quat::from_rotation_y(-delta.x * YAW_SENSITIVITY);
        let pitch = Quat::from_rotation_x(delta.y * PITCH_SENSITIVITY);
        // Renormalize every step: repeated quaternion multiplication can
        // accumulate tiny floating-point drift away from unit length over
        // a long play session, which would otherwise slowly skew
        // `eye_and_basis`'s basis vectors away from orthonormal.
        self.orientation = (yaw * self.orientation * pitch).normalize();
    }

    /// Applies a two-finger-rotate roll increment around the camera's own
    /// current forward axis (post-multiply, i.e. in the camera's local
    /// frame — a pure screen-space roll, `forward` unaffected).
    pub(crate) fn roll(&mut self, delta_radians: f32) {
        self.orientation = (self.orientation * Quat::from_rotation_z(delta_radians)).normalize();
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
            orbit.drag(ev.delta);
        }
    } else {
        motion.clear();
    }

    for ev in wheel.read() {
        orbit.distance = (orbit.distance - ev.y * ZOOM_SENSITIVITY).clamp(MIN_DISTANCE, MAX_DISTANCE);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The old (pre-quaternion) offset-direction formula, reimplemented
    /// standalone here purely as an independent oracle for
    /// `orientation_from_yaw_pitch`'s regression test below -- not used by
    /// any non-test code.
    fn old_forward(yaw: f32, pitch: f32) -> Vec3 {
        -Vec3::new(pitch.cos() * yaw.sin(), pitch.sin(), pitch.cos() * yaw.cos())
    }

    #[test]
    fn orientation_from_yaw_pitch_matches_old_formula() {
        for (yaw, pitch) in [(0.0, 0.0), (0.8, 0.35), (-1.448, 0.899), (2.5, -1.2), (0.3, 1.4)] {
            let orientation = CameraOrbit::orientation_from_yaw_pitch(yaw, pitch);
            let got = orientation * Vec3::NEG_Z;
            let want = old_forward(yaw, pitch);
            assert!(
                got.distance(want) < 1e-4,
                "yaw={yaw} pitch={pitch}: got forward {got:?}, want {want:?}"
            );
        }
    }

    #[test]
    fn eye_and_basis_matches_old_right_up_derivation() {
        // The old code derived right/up as forward.cross(world_up) /
        // right.cross(forward) rather than rotating basis vectors by the
        // quaternion directly -- confirm the two approaches agree (at
        // zero roll) for a few presets, not just that `forward` does.
        for (yaw, pitch) in [(0.0, 0.0), (0.8, 0.35), (-1.448, 0.899)] {
            let orbit = CameraOrbit {
                orientation: CameraOrbit::orientation_from_yaw_pitch(yaw, pitch),
                distance: 1.0,
            };
            let (_, right, up, forward) = orbit.eye_and_basis(Vec3::ZERO);
            let want_right = forward.cross(Vec3::Y).normalize();
            let want_up = want_right.cross(forward);
            assert!(right.distance(want_right) < 1e-4, "yaw={yaw} pitch={pitch}: right mismatch");
            assert!(up.distance(want_up) < 1e-4, "yaw={yaw} pitch={pitch}: up mismatch");
        }
    }

    #[test]
    fn eye_is_distance_back_along_forward_from_target() {
        let orbit = CameraOrbit { orientation: CameraOrbit::orientation_from_yaw_pitch(0.4, 0.2), distance: 3.0 };
        let target = Vec3::new(1.0, 2.0, 3.0);
        let (eye, _, _, forward) = orbit.eye_and_basis(target);
        assert!((eye - target).length() - 3.0 < 1e-4);
        assert!((target - eye).normalize().distance(forward) < 1e-4);
    }

    #[test]
    fn repeated_vertical_drag_orbits_past_the_old_pitch_clamp_with_no_lock() {
        // The old representation clamped pitch to +-1.5 rad (~86 degrees,
        // already nearly vertical) and could never exceed that -- a real
        // limitation the "truly global" rework removes. Demonstrate that
        // concretely: drag vertically by a precisely known *total* of
        // exactly PI radians (180 degrees, using many small steps so no
        // single step is implausibly large) -- comfortably past the old
        // clamp, over the top, and back down the other side. The old
        // system was physically incapable of ever reaching this state at
        // all; this one lands exactly where continuous rotation predicts,
        // with a still-orthonormal, non-degenerate basis.
        let mut orbit = CameraOrbit { orientation: Quat::IDENTITY, distance: 1.0 };
        let steps = 100;
        let total_pitch = std::f32::consts::PI;
        let per_step_delta_y = -(total_pitch / PITCH_SENSITIVITY) / steps as f32;
        for _ in 0..steps {
            orbit.drag(Vec2::new(0.0, per_step_delta_y));
        }
        let forward = orbit.forward();
        assert!(forward.is_finite(), "basis degenerated: {forward:?}");
        // Starting forward was (0, 0, -1); after a PI (half-turn) pitch
        // rotation over the top, it must land close to (0, 0, 1) --
        // exactly the opposite horizontal direction, having passed through
        // straight-up along the way. No Euler-angle clamp could ever reach
        // this, since it never leaves +-1.5 rad of the start.
        assert!(
            forward.distance(Vec3::new(0.0, 0.0, 1.0)) < 1e-3,
            "expected to land flipped to the opposite side after a PI pitch rotation, got {forward:?}"
        );
        // Basis must still be orthonormal after many compositions (checks
        // the per-step renormalization is actually keeping drift in check).
        let (_, right, up, _) = orbit.eye_and_basis(Vec3::ZERO);
        assert!((right.length() - 1.0).abs() < 1e-3);
        assert!((up.length() - 1.0).abs() < 1e-3);
        assert!(right.dot(up).abs() < 1e-3);
    }

    #[test]
    fn roll_does_not_change_forward() {
        let mut orbit = CameraOrbit { orientation: CameraOrbit::orientation_from_yaw_pitch(0.8, 0.35), distance: 1.0 };
        let forward_before = orbit.forward();
        orbit.roll(1.2);
        let forward_after = orbit.forward();
        assert!(
            forward_before.distance(forward_after) < 1e-4,
            "roll must not change the view direction: {forward_before:?} -> {forward_after:?}"
        );
    }

    #[test]
    fn roll_does_change_right_and_up() {
        let mut orbit = CameraOrbit { orientation: CameraOrbit::orientation_from_yaw_pitch(0.8, 0.35), distance: 1.0 };
        let (_, right_before, up_before, _) = orbit.eye_and_basis(Vec3::ZERO);
        orbit.roll(1.2);
        let (_, right_after, up_after, _) = orbit.eye_and_basis(Vec3::ZERO);
        assert!(right_before.distance(right_after) > 0.1, "roll should visibly change `right`");
        assert!(up_before.distance(up_after) > 0.1, "roll should visibly change `up`");
    }
}
