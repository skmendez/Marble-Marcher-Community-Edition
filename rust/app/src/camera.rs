//! Game camera: orbits a target (the marble) via a free 3D orientation
//! quaternion + distance (DESIGN.md §7/§8).
//!
//! Arcball/trackball model ("the camera sits on a sphere around the
//! marble, Google Earth-style"): a swipe moves the camera a distance
//! across that sphere proportional to the swipe's magnitude, in the swiped
//! direction, without introducing any twist of its own; a two-finger twist
//! changes heading and *only* heading. This replaced an earlier
//! yaw(world-Y)/pitch(local-right)/roll(separate scalar) decomposition
//! that went through three rounds of live-reported bugs (gimbal lock at
//! the poles under Euler angles; a real pan-vs-twist bug once `roll` was
//! pulled out as a separate scalar with `drag`'s pitch still rotating
//! around a roll-tilted axis; then a pitch-dependent yaw/pitch gain
//! mismatch once roll-compensation was added to fix *that*) — each fix was
//! a compensation formula patched onto a mismatch between two differently
//! defined rotation axes, not a removal of the mismatch. The arcball
//! design has no such mismatch: every swipe is a *single* rotation whose
//! axis is always perpendicular to `forward` (so it can never twist the
//! view around its own look direction) and whose angle depends only on
//! swipe magnitude (so sphere-distance-per-pixel is uniform regardless of
//! pitch) — see `drag`'s doc for the exact construction and why both of
//! those hold by construction, not by tuning.

use bevy::input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::prelude::*;

use crate::physics_sys::MarbleState;

/// Radians of rotation per pixel of swipe magnitude (`drag`) — same
/// numeric value the old yaw/pitch decomposition used for both axes, kept
/// so a swipe of a given pixel length still produces roughly the same
/// on-screen motion as before.
pub(crate) const DRAG_SENSITIVITY: f32 = 0.006;
/// Focal length (`1/tan(halfFOV)`) baked into `cam_forward.w` everywhere a
/// `SceneUniforms` is written (`render.rs`/`mrrm.rs`/`shadow_pass.rs`, each
/// currently as their own `forward.extend(1.5)` literal, pre-existing
/// duplication not touched here) — exposed here too so `debug_gizmos.rs`'s
/// screen-space projection uses the exact same value rather than a fifth,
/// independently-chosen copy.
pub(crate) const FOCAL_LENGTH: f32 = 1.5;
const MIN_DISTANCE: f32 = 0.12;
const MAX_DISTANCE: f32 = 20.0;
const ZOOM_SENSITIVITY: f32 = 0.5;
/// `MouseWheel::unit == Pixel` (trackpads, and some mice) reports raw pixel
/// deltas rather than wheel "notches" -- a single physical gesture's
/// magnitude can be 10-100x a `Line` event's. Dividing by this before
/// applying `ZOOM_SENSITIVITY` (see `orbit_camera_input`) puts pixel deltas
/// on the same per-line scale line deltas already use; ~100 CSS pixels per
/// line is the standard DOM wheel-event convention (matches Chrome's
/// default `deltaMode`-conversion factor). Without this, a single trackpad
/// swipe on macOS got read as an enormous number of "lines" and jumped the
/// camera straight to `MIN_DISTANCE`/`MAX_DISTANCE` instead of smoothly
/// zooming -- reported live as "scrolling jumps between fog and 100% zoom".
const PIXELS_PER_LINE: f32 = 100.0;
/// The camera is never allowed closer than this multiple of the marble's
/// own radius, on top of the flat `MIN_DISTANCE` floor -- `MIN_DISTANCE`
/// alone isn't enough for every scene: the Menger scenes' marble
/// (`render.rs::spawn_params`, `rad = 0.15`) is *larger* than
/// `MIN_DISTANCE = 0.12`, so a full zoom-in there put the eye literally
/// inside the marble's own geometry ("100% zoom... puts you inside the
/// marble"). `1.5x` clears the surface with a comfortable margin while
/// staying well under every scene's own tuned default distance
/// (`render.rs::setup`'s per-scene `camera_orbit.distance` overrides), so
/// zoom-in headroom is preserved everywhere.
const MIN_DISTANCE_MARBLE_RADII: f32 = 1.5;

/// Full 3D orbit orientation + distance around an externally supplied
/// target (see [`CameraOrbit::eye_and_basis`]) — the target itself (the
/// marble's position) lives in `MarbleState`, not here, so this resource
/// stays pure input state.
///
/// A single quaternion, with no separate `roll`/yaw/pitch fields: `drag`
/// and `roll` below only ever compose an *incremental* rotation onto
/// whatever `orientation` already is, never decompose into or reconstruct
/// from stored angles — which is what avoids gimbal lock (no "pitch" value
/// that can approach the pole and collapse yaw/roll onto the same axis)
/// *and* what avoids the roll-vs-orientation sync bugs the previous
/// (`orientation` + separate `roll: f32`) representation went through —
/// there's nothing left to fall out of sync, since roll is just part of
/// `orientation` like everything else.
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
    /// looking at `target`, `distance` back along `-forward` (DESIGN.md
    /// §8). `right`/`up` already include whatever twist has accumulated —
    /// `orientation` *is* the rendered frame now, there's no separate
    /// unrolled frame to reconcile it with.
    pub fn eye_and_basis(&self, target: Vec3) -> (Vec3, Vec3, Vec3, Vec3) {
        let forward = self.forward();
        let right = self.orientation * Vec3::X;
        let up = self.orientation * Vec3::Y;
        let eye = target - forward * self.distance;
        (eye, right, up, forward)
    }

    /// The direction the camera is currently looking, independent of any
    /// target (`physics_sys.rs` passes this, alongside `eye_and_basis`'s
    /// `right`, into `marble_csg::step_marble`'s camera-relative thrust).
    pub fn forward(&self) -> Vec3 {
        self.orientation * Vec3::NEG_Z
    }

    /// Applies a swipe (mouse drag or single-finger touch) as a single
    /// rotation that slides the camera across its sphere in the swiped
    /// direction, by an angle proportional to the swipe's magnitude —
    /// the arcball/trackball construction (module doc).
    ///
    /// `screen_dir` is the swipe's direction expressed in world space,
    /// built from the camera's *current* `right`/`up` (twist included —
    /// there's no "unrolled" frame to convert into first, unlike the
    /// previous design). `axis`, being `forward.cross(screen_dir)`, is
    /// perpendicular to both `forward` and `screen_dir` by construction:
    /// perpendicular to `forward` means rotating around it can *never* put
    /// a twist component into the result (there's nothing left over to
    /// spin the view around its own look direction with), and perpendicular
    /// to `screen_dir` combined with the vector triple product identity
    /// `(A×B)×A = B` (for unit, orthogonal `A`, `B`) means `forward` moves
    /// *exactly* along `screen_dir` — not just to first order, for the
    /// whole finite rotation. Because `angle` depends only on `delta`'s
    /// magnitude (not a separate yaw/pitch split with different gains at
    /// different pitches, the previous design's bug), the sphere-distance
    /// moved per pixel of swipe is the same at any pitch, any roll, any
    /// swipe direction — verified numerically across a pitch/roll sweep,
    /// not just at the one configuration each previous round happened to
    /// test.
    ///
    /// Sign convention matches the previously-shipped, already-tuned
    /// roll=0 behavior exactly (checked independently against the old
    /// formula for pure horizontal and pure vertical swipes) — this isn't
    /// a fourth flipped direction.
    pub(crate) fn drag(&mut self, delta: Vec2) {
        let Some((screen_dir, angle)) = self.drag_intermediates(delta) else {
            return;
        };
        let axis = self.forward().cross(screen_dir).normalize();
        // Renormalize every step: repeated quaternion multiplication can
        // accumulate tiny floating-point drift away from unit length over
        // a long play session, which would otherwise slowly skew
        // `eye_and_basis`'s basis vectors away from orthonormal.
        self.orientation = (Quat::from_axis_angle(axis, angle) * self.orientation).normalize();
    }

    /// `drag`'s `screen_dir`/`angle` (world-space swipe direction, rotation
    /// magnitude in radians), or `None` for a zero-length `delta` -- pulled
    /// out on its own so `touch.rs`'s live debug readout can display the
    /// exact intermediates `drag` actually computes, without a second,
    /// independently-written copy of the formula that could drift out of
    /// sync with the real one.
    pub(crate) fn drag_intermediates(&self, delta: Vec2) -> Option<(Vec3, f32)> {
        if delta.length_squared() < 1e-12 {
            return None;
        }
        let right = self.orientation * Vec3::X;
        let up = self.orientation * Vec3::Y;
        let screen_dir = (right * delta.x + up * delta.y).normalize();
        let angle = delta.length() * DRAG_SENSITIVITY;
        Some((screen_dir, angle))
    }

    /// Applies a two-finger-rotate twist increment: a *local*-frame
    /// rotation about the camera's own current forward axis (post-multiply
    /// by a local-Z rotation — the standard body-rotation identity, same
    /// one `drag`'s doc leans on for why a world-frame pre-multiply behaves
    /// differently from a local-frame post-multiply). Local Z is exactly
    /// `forward`'s axis, and a rotation about Z never touches the Z
    /// component it's rotating around, so this changes `right`/`up` but
    /// provably never `forward` (`roll_does_not_change_forward` below) —
    /// heading changes, nothing else, matching the "twist rotates the
    /// sphere in place" half of the arcball model. Reproduces the same
    /// tilt-direction sign the previous separate-`roll`-scalar design had.
    pub(crate) fn roll(&mut self, delta_radians: f32) {
        self.orientation = (self.orientation * Quat::from_rotation_z(delta_radians)).normalize();
    }
}

/// Radians/second applied while `Q`/`E` is held -- keyboard/mouse users have
/// no multi-touch gesture to roll with otherwise (touch's two-finger rotate
/// is the only other input path to `CameraOrbit::roll`), so this is a real
/// accessibility gap on desktop, not just a debug convenience -- though it
/// also happens to be the only reliable way to drive a nonzero roll from
/// browser automation (CDP), since synthetic multi-touch gestures don't
/// reliably reach this app's touch handling.
const KEYBOARD_ROLL_RATE: f32 = 1.5;

/// Left-drag to orbit, scroll wheel to zoom, `Q`/`E` to roll.
#[allow(clippy::too_many_arguments)] // SystemParam count, one more for the marble-radius zoom clamp
pub fn orbit_camera_input(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut motion: EventReader<MouseMotion>,
    mut wheel: EventReader<MouseWheel>,
    mut orbit: ResMut<CameraOrbit>,
    mut twist_debug: ResMut<crate::fps_overlay::DebugTwistAccum>,
    marble_state: Res<MarbleState>,
) {
    if mouse_buttons.pressed(MouseButton::Left) {
        for ev in motion.read() {
            orbit.drag(ev.delta);
        }
    } else {
        motion.clear();
    }

    let mut roll_dir = 0.0f32;
    if keys.pressed(KeyCode::KeyQ) {
        roll_dir += 1.0;
    }
    if keys.pressed(KeyCode::KeyE) {
        roll_dir -= 1.0;
    }
    if roll_dir != 0.0 {
        let delta = roll_dir * KEYBOARD_ROLL_RATE * time.delta_secs();
        orbit.roll(delta);
        twist_debug.0 += delta;
    }

    let min_distance = MIN_DISTANCE.max(marble_state.local_marble().rad * MIN_DISTANCE_MARBLE_RADII);
    for ev in wheel.read() {
        let lines = match ev.unit {
            MouseScrollUnit::Line => ev.y,
            MouseScrollUnit::Pixel => ev.y / PIXELS_PER_LINE,
        };
        orbit.distance = (orbit.distance - lines * ZOOM_SENSITIVITY).clamp(min_distance, MAX_DISTANCE);
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
        // quaternion directly -- confirm the two approaches agree for a
        // few presets, not just that `forward` does.
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
        let orbit = CameraOrbit {
            orientation: CameraOrbit::orientation_from_yaw_pitch(0.4, 0.2),
            distance: 3.0,
        };
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
        let per_step_delta_y = -(total_pitch / DRAG_SENSITIVITY) / steps as f32;
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
        let mut orbit = CameraOrbit {
            orientation: CameraOrbit::orientation_from_yaw_pitch(0.8, 0.35),
            distance: 1.0,
        };
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
        let mut orbit = CameraOrbit {
            orientation: CameraOrbit::orientation_from_yaw_pitch(0.8, 0.35),
            distance: 1.0,
        };
        let (_, right_before, up_before, _) = orbit.eye_and_basis(Vec3::ZERO);
        orbit.roll(1.2);
        let (_, right_after, up_after, _) = orbit.eye_and_basis(Vec3::ZERO);
        assert!(right_before.distance(right_after) > 0.1, "roll should visibly change `right`");
        assert!(up_before.distance(up_after) > 0.1, "roll should visibly change `up`");
    }

    #[test]
    fn roll_sign_matches_previously_shipped_tilt_direction() {
        // The old (separate-scalar) design's `eye_and_basis` computed
        // `right = right0*cos(roll) + up0*sin(roll)` -- i.e. for a small
        // positive roll, `right` tilts *toward* `up0`. Confirm the new
        // local-Z-post-multiply `roll` reproduces the same sign, not just
        // "changes right/up somehow" (the test above).
        let base = CameraOrbit::orientation_from_yaw_pitch(0.8, 0.35);
        let mut orbit = CameraOrbit { orientation: base, distance: 1.0 };
        let right0 = base * Vec3::X;
        let up0 = base * Vec3::Y;
        let small_roll = 0.01;
        orbit.roll(small_roll);
        let (_, right, _, _) = orbit.eye_and_basis(Vec3::ZERO);
        let predicted = (right0 + up0 * small_roll).normalize();
        assert!(
            right.distance(predicted) < 1e-3,
            "expected right to tilt toward up0 for positive roll (old sign convention), \
             got {right:?}, predicted {predicted:?}"
        );
    }

    #[test]
    fn drag_matches_old_sign_convention_at_zero_roll() {
        // The old shipped formula at roll=0: yaw_angle = -delta.x*SENS
        // (world-Y, pre-multiplied), pitch_angle = delta.y*SENS (local
        // right0, post-multiplied). Confirm the new single-rotation
        // `drag` moves `forward` in exactly the same direction for pure
        // horizontal and pure vertical swipes at a level, unrolled start
        // -- this round must not silently flip a direction that was
        // already correct and already user-verified.
        fn old_drag(o: Quat, delta: Vec2) -> Quat {
            let yaw = Quat::from_rotation_y(-delta.x * DRAG_SENSITIVITY);
            let pitch = Quat::from_rotation_x(delta.y * DRAG_SENSITIVITY);
            (yaw * o * pitch).normalize()
        }
        let base = CameraOrbit::orientation_from_yaw_pitch(0.0, 0.0);
        for delta in [Vec2::new(10.0, 0.0), Vec2::new(-10.0, 0.0), Vec2::new(0.0, 10.0), Vec2::new(0.0, -10.0)] {
            let mut new_orbit = CameraOrbit { orientation: base, distance: 1.0 };
            new_orbit.drag(delta);
            let old_orientation = old_drag(base, delta);

            let new_delta = (new_orbit.forward() - (base * Vec3::NEG_Z)).normalize();
            let old_delta = (old_orientation * Vec3::NEG_Z - (base * Vec3::NEG_Z)).normalize();
            assert!(
                new_delta.distance(old_delta) < 1e-2,
                "delta={delta:?}: new drag moved forward {new_delta:?}, old drag moved {old_delta:?} -- direction mismatch"
            );
        }
    }

    #[test]
    fn drag_moves_by_uniform_angle_regardless_of_pitch() {
        // The bug that broke round 3: yaw and pitch had different gains on
        // `forward` except exactly at pitch=0, so a fixed-magnitude swipe
        // moved the camera different *amounts* (and, mixed with roll,
        // different *directions*) depending on current pitch. The arcball
        // design has no yaw/pitch split at all -- confirm a fixed-magnitude
        // swipe rotates `forward` by *exactly* the same angle at several
        // very different pitches (and with roll thrown in, since the old
        // bug was specifically a roll+pitch interaction).
        let delta = Vec2::new(30.0, -18.0);
        let mut angles = Vec::new();
        for pitch_deg in [0.0, 30.0, 60.0, 80.0, -45.0] {
            let pitch = pitch_deg_to_rad(pitch_deg);
            let base = CameraOrbit::orientation_from_yaw_pitch(0.8, pitch) * Quat::from_rotation_z(0.6458); // ~37deg roll thrown in
            let mut orbit = CameraOrbit { orientation: base, distance: 1.0 };
            let f0 = orbit.forward();
            orbit.drag(delta);
            let f1 = orbit.forward();
            angles.push(f0.angle_between(f1));
        }
        let first = angles[0];
        for (pitch_deg, ang) in [0.0, 30.0, 60.0, 80.0, -45.0].into_iter().zip(angles.iter()) {
            assert!(
                (ang - first).abs() < 1e-4,
                "pitch={pitch_deg}: rotated {ang} rad, expected {first} rad (uniform regardless of pitch)"
            );
        }
    }

    fn pitch_deg_to_rad(d: f32) -> f32 {
        d.to_radians()
    }

    #[test]
    fn drag_out_and_back_returns_to_exact_original_orientation() {
        // The bug that broke round 2 (and was only incompletely fixed by
        // round 3): a swipe should never leave any residual twist behind.
        // The strongest testable form: drag by delta, then immediately
        // drag by -delta from the *resulting* state (recomputing
        // screen_dir/axis fresh each time, exactly as real input does) --
        // must land back at exactly the starting orientation, full
        // right/up/forward included, not just forward.
        for (pitch_deg, roll, delta) in [
            (0.0, 0.0, Vec2::new(20.0, 0.0)),
            (40.0, 1.3, Vec2::new(-15.0, 22.0)),
            (-60.0, -2.1, Vec2::new(8.0, -30.0)),
            (75.0, 0.7, Vec2::new(-40.0, -5.0)),
        ] {
            let pitch = pitch_deg_to_rad(pitch_deg);
            let base = CameraOrbit::orientation_from_yaw_pitch(0.3, pitch) * Quat::from_rotation_z(roll);
            let mut orbit = CameraOrbit { orientation: base, distance: 1.0 };
            orbit.drag(delta);
            orbit.drag(-delta);
            let ang = base.angle_between(orbit.orientation);
            // Exact in infinite precision (verified algebraically: the
            // second drag's rotation is provably the exact inverse of the
            // first, via the rotation-conjugation identity R*Rot(axis,th)*
            // R^-1 = Rot(R*axis,th)) -- this tolerance is `f32`-precision
            // headroom for a handful of chained trig/cross/normalize ops at
            // up to a ~14 degree single-step rotation, not slack for a real
            // bug: a genuine twist-leak would be comparable in size to the
            // swipe's own rotation angle (tenths of a radian), not 1e-3.
            assert!(
                ang < 5e-3,
                "pitch={pitch_deg} roll={roll} delta={delta:?}: out-and-back left {ang} rad of residual twist"
            );
        }
    }

    #[test]
    fn drag_matches_real_device_repro_at_45_degree_roll() {
        // Regression test grounded in an actual live repro, not a
        // synthetic case: an on-device screenshot showed `roll: 46.2deg`,
        // `forward: (-0.290, -0.800, -0.526)`, and a bottom-to-top swipe
        // giving `delta: (-0.1, -1.6)` -- under the *previous* (gain-
        // mismatched) design this nearly-pure-vertical swipe leaked ~30%
        // onto the right/left axis (dot ~-0.30), matching the reported
        // "swipe up tilts left" bug. Reconstruct that exact state under the
        // new design (roll composed via `roll()`, not a field) and confirm
        // the swipe stays cleanly on-axis.
        let pitch = 0.8_f32.asin(); // sin(pitch) = -forward.y = 0.800
        let yaw = (0.290_f32 / pitch.cos()).asin(); // cos(pitch)*sin(yaw) = -forward.x = 0.290
        let mut orbit = CameraOrbit {
            orientation: CameraOrbit::orientation_from_yaw_pitch(yaw, pitch),
            distance: 1.0,
        };
        orbit.roll(46.2_f32.to_radians());
        let forward_before = orbit.forward();
        assert!(
            forward_before.distance(Vec3::new(-0.290, -0.800, -0.526)) < 0.02,
            "reconstructed state should reproduce the screenshot's forward, got {forward_before:?}"
        );

        let (_, right, up, _) = orbit.eye_and_basis(Vec3::ZERO);
        orbit.drag(Vec2::new(-0.1, -1.6));
        let moved = (orbit.forward() - forward_before).normalize();

        let cross_axis_leak = moved.dot(right.normalize()).abs();
        assert!(
            cross_axis_leak < 0.1,
            "a near-pure-vertical swipe at roll=46.2deg leaked {cross_axis_leak:.3} onto \
             `right` (pre-fix this was ~0.30) -- moved={moved:?} right={right:?}"
        );
        let up_alignment = moved.dot(up.normalize()).abs();
        assert!(
            up_alignment > 0.95,
            "expected the swipe to be predominantly aligned with `up`/`-up`, got \
             dot={up_alignment:.3} moved={moved:?} up={up:?}"
        );
    }
}
