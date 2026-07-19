//! Touch input: swipe (1 finger) orbits the camera exactly like mouse-drag;
//! a 2-finger gesture computes pinch (marble thrust, applied in
//! `physics_sys::marble_physics_tick`) and rotate (camera roll) *concurrently*
//! from the same two touch points, since a real two-finger gesture is
//! usually a mix of both at once, not one or the other.
//!
//! Gesture math is implemented as plain-data free functions (below
//! `pinch_delta_distance`/`rotate_delta_angle`/`sort_by_id`) so it's
//! unit-testable without constructing a real `bevy::input::touch::Touches`
//! resource — its only mutation entry point, `process_touch_event`, is
//! private to `bevy_input`, so tests use the small owned [`TouchSnapshot`]
//! type instead. The Bevy systems (`touch_camera_input` here, and the pinch
//! read directly in `physics_sys::marble_physics_tick`) are thin wrappers
//! that convert real `Touch`es into snapshots and call into this math.

use bevy::input::touch::Touches;
use bevy::prelude::*;

use crate::camera::CameraOrbit;

/// A minimal, owned snapshot of one touch's current and previous-frame
/// position — decoupled from `bevy::input::touch::Touch` (a borrowed,
/// bevy_input-internal-construction-only type) purely so the gesture math
/// below is unit-testable with plain data.
#[derive(Clone, Copy, Debug)]
struct TouchSnapshot {
    id: u64,
    previous_position: Vec2,
    position: Vec2,
}

/// Sorts by touch id (ascending) for a stable, deterministic pairing across
/// frames — `Touches` is backed by a `HashMap`, whose iteration order is not
/// guaranteed to stay consistent between frames for the same 2 ids, which
/// would otherwise risk silently swapping which touch is "first" mid-gesture
/// (flipping the sign of every pinch/rotate computation for a frame).
fn sort_by_id(mut snapshots: Vec<TouchSnapshot>) -> Vec<TouchSnapshot> {
    snapshots.sort_by_key(|t| t.id);
    snapshots
}

/// Returns up to 2 currently-pressed touches, sorted by id. A 3rd+
/// simultaneous touch (rare, but possible on some devices) is ignored rather
/// than tracked, per the "up to 2" spec — extending to more touches isn't
/// needed for these gestures and would need a real multi-touch gesture
/// disambiguation scheme.
fn active_snapshots_sorted(touches: &Touches) -> [Option<TouchSnapshot>; 2] {
    let all = sort_by_id(
        touches
            .iter()
            .map(|t| TouchSnapshot {
                id: t.id(),
                previous_position: t.previous_position(),
                position: t.position(),
            })
            .collect(),
    );
    let mut out: [Option<TouchSnapshot>; 2] = [None, None];
    for (slot, snap) in out.iter_mut().zip(all) {
        *slot = Some(snap);
    }
    out
}

/// Change in inter-touch distance this frame (`current - previous`),
/// **not** the absolute distance — this is what makes the pinch a
/// continuous, velocity-like force (proportional to how fast the fingers
/// are moving apart/together) rather than a one-shot impulse proportional
/// to how far apart they happen to be. Positive = fingers spreading apart
/// ("zoom in" motion); negative = fingers coming together ("zoom out"
/// motion).
fn pinch_delta_distance(prev_a: Vec2, prev_b: Vec2, cur_a: Vec2, cur_b: Vec2) -> f32 {
    let cur_dist = cur_a.distance(cur_b);
    let prev_dist = prev_a.distance(prev_b);
    cur_dist - prev_dist
}

/// Converts a pinch-distance delta (screen pixels/frame) into a `dy` value
/// in `step_marble`'s WASD convention (physics_sys.rs's doc comment:
/// `dy = -1` is W / toward-camera, `dy = +1` is S / away-from-camera).
/// Pinch IN (fingers together, distance decreasing) pulls the marble
/// TOWARD the camera, matching W — so a *negative* `pinch_delta_distance`
/// (distance shrinking) must produce a *negative* `dy`, i.e. no sign flip
/// needed at all. (An earlier version of this function negated the delta,
/// on the belief that `dy = +1`/S was the toward-camera direction — that
/// belief was backwards, verified empirically against a live build; see
/// physics_sys.rs's WASD doc for how. That bug is exactly why pinching in
/// used to send the marble away from the camera instead of toward it.)
/// `PINCH_SENSITIVITY` is a first-pass value (not yet feel-tuned on a real
/// touchscreen): chosen so a brisk ~20px/frame pinch motion saturates to
/// the same |dy| = 1 magnitude a held WASD key gives; `step_marble` already
/// clamps combined `(dx, dy)` magnitude to 1, so overshooting this is
/// harmless, just not meaningfully "more" force.
const PINCH_SENSITIVITY: f32 = 0.05;

fn pinch_dy(prev_a: Vec2, prev_b: Vec2, cur_a: Vec2, cur_b: Vec2) -> f32 {
    let delta = pinch_delta_distance(prev_a, prev_b, cur_a, cur_b);
    (PINCH_SENSITIVITY * delta).clamp(-1.0, 1.0)
}

/// Largest single-frame rotation treated as a genuine gesture rather than a
/// tracking artifact — see `rotate_delta_angle`'s doc. About 130 degrees:
/// comfortably above any plausible intentional two-finger twist in one
/// ~16ms frame (the existing test suite's fastest case is a 90-degree
/// rotation), comfortably below the ~180-degree jump a same-frame swap of
/// which touch is "first" would produce.
const MAX_PLAUSIBLE_ROTATE_DELTA: f32 = 2.3;

/// Change in the angle of the line between two touches, this frame vs last
/// (radians, wrapped to `[-PI, PI]` so a gesture crossing the +/-PI seam
/// doesn't register as a near-full-turn jump, then clamped to
/// `MAX_PLAUSIBLE_ROTATE_DELTA`). Fed 1:1 into `CameraOrbit::roll` — the
/// standard, expected mapping for a two-finger rotate gesture (turn your
/// fingers by X radians, the view rolls by X radians).
///
/// The clamp guards against a specific failure mode reported as "pinching
/// sometimes sends the ball sideways instead of straight toward/away from
/// the camera": `active_snapshots_sorted` pairs touches by sorting on their
/// `id`, assuming each id stays bound to the same physical finger for the
/// gesture's duration. If a touch's reported id ever changes mid-gesture
/// (a real possibility on some touchscreens/browsers, not just a
/// theoretical one), the two snapshots this function compares could
/// silently swap which one is "a" and which is "b" between frames — the
/// pinch distance math is symmetric to that swap (harmless), but this
/// angle is not: a swap flips it by very close to +/-PI in one single
/// frame. Nothing about that spurious near-180-degree roll would be
/// visually distinguishable from "camera suddenly spun," which combined
/// with an otherwise-straight pinch push reads exactly as "the ball went
/// sideways." Clamping (not dropping to zero) still lets a genuinely fast
/// real rotation register, just not an implausible single-frame near-flip.
fn rotate_delta_angle(prev_a: Vec2, prev_b: Vec2, cur_a: Vec2, cur_b: Vec2) -> f32 {
    let prev_angle = (prev_b - prev_a).to_angle();
    let cur_angle = (cur_b - cur_a).to_angle();
    wrap_angle(cur_angle - prev_angle).clamp(-MAX_PLAUSIBLE_ROTATE_DELTA, MAX_PLAUSIBLE_ROTATE_DELTA)
}

fn wrap_angle(a: f32) -> f32 {
    use std::f32::consts::PI;
    (a + PI).rem_euclid(2.0 * PI) - PI
}

/// Per-frame result of reading the current 2-touch gesture, if any — read by
/// `physics_sys::marble_physics_tick` (for `pinch_dy`) and applied to
/// `CameraOrbit::roll` here (for `rotate_delta`).
#[derive(Clone, Copy, Debug, Default)]
pub struct TwoFingerGesture {
    pub pinch_dy: f32,
    pub rotate_delta: f32,
}

/// Computes the current-frame two-finger gesture (pinch + rotate,
/// concurrently) from `Touches`, or `None` if fewer than 2 touches are
/// active. Shared by `touch_camera_input` (applies `rotate_delta` to roll)
/// and `physics_sys::marble_physics_tick` (applies `pinch_dy` to marble
/// thrust) so both read the exact same pair of touches each frame.
pub fn read_two_finger_gesture(touches: &Touches) -> Option<TwoFingerGesture> {
    let [a, b] = active_snapshots_sorted(touches);
    let (a, b) = (a?, b?);
    Some(TwoFingerGesture {
        pinch_dy: pinch_dy(a.previous_position, b.previous_position, a.position, b.position),
        rotate_delta: rotate_delta_angle(
            a.previous_position,
            b.previous_position,
            a.position,
            b.position,
        ),
    })
}

/// Touch-driven camera control (`Update` schedule, alongside
/// `orbit_camera_input`): exactly 1 active touch swipes (`CameraOrbit::drag`,
/// identical feel to mouse-drag — same sensitivity constants, applied via
/// the same method); 2+ applies this frame's rotate delta to `roll` (the
/// pinch half of a 2-touch gesture is read separately, directly inside the
/// physics tick — see `physics_sys.rs`).
pub fn touch_camera_input(touches: Res<Touches>, mut orbit: ResMut<CameraOrbit>) {
    let active_count = touches.iter().count();
    if active_count == 1 {
        if let Some(touch) = touches.iter().next() {
            orbit.drag(touch.delta());
        }
    } else if active_count >= 2 {
        if let Some(gesture) = read_two_finger_gesture(&touches) {
            orbit.roll(gesture.rotate_delta);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(id: u64, previous_position: Vec2, position: Vec2) -> TouchSnapshot {
        TouchSnapshot { id, previous_position, position }
    }

    #[test]
    fn pinch_in_moves_marble_toward_camera() {
        // Fingers 10 units apart, moving to 6 units apart (pinching in) --
        // must give a negative dy (toward camera, per physics_sys.rs's W
        // convention -- see that file's doc for how this was verified).
        let prev_a = Vec2::new(-5.0, 0.0);
        let prev_b = Vec2::new(5.0, 0.0);
        let cur_a = Vec2::new(-3.0, 0.0);
        let cur_b = Vec2::new(3.0, 0.0);
        let dy = pinch_dy(prev_a, prev_b, cur_a, cur_b);
        assert!(dy < 0.0, "pinch-in should give negative (toward-camera) dy, got {dy}");
    }

    #[test]
    fn pinch_out_moves_marble_away_from_camera() {
        // Fingers 6 units apart, spreading to 10 units apart -- must give a
        // positive dy (away from camera, S convention).
        let prev_a = Vec2::new(-3.0, 0.0);
        let prev_b = Vec2::new(3.0, 0.0);
        let cur_a = Vec2::new(-5.0, 0.0);
        let cur_b = Vec2::new(5.0, 0.0);
        let dy = pinch_dy(prev_a, prev_b, cur_a, cur_b);
        assert!(dy > 0.0, "pinch-out should give positive (away-from-camera) dy, got {dy}");
    }

    #[test]
    fn no_pinch_motion_gives_zero_dy() {
        let a = Vec2::new(-4.0, 1.0);
        let b = Vec2::new(4.0, 1.0);
        assert_eq!(pinch_dy(a, b, a, b), 0.0);
    }

    #[test]
    fn pinch_dy_saturates_to_unit_range() {
        // A huge, unrealistic single-frame pinch-in motion must still clamp
        // to [-1, 1] (step_marble's own combined-magnitude clamp assumes
        // roughly WASD-scale inputs).
        let prev_a = Vec2::new(-500.0, 0.0);
        let prev_b = Vec2::new(500.0, 0.0);
        let cur_a = Vec2::ZERO;
        let cur_b = Vec2::ZERO;
        let dy = pinch_dy(prev_a, prev_b, cur_a, cur_b);
        assert_eq!(dy, -1.0);
    }

    #[test]
    fn rotate_delta_matches_known_angle_change() {
        // Two touches on the horizontal axis (angle 0) rotating 90 degrees
        // counterclockwise to the vertical axis (angle PI/2).
        let prev_a = Vec2::new(-5.0, 0.0);
        let prev_b = Vec2::new(5.0, 0.0);
        let cur_a = Vec2::new(0.0, -5.0);
        let cur_b = Vec2::new(0.0, 5.0);
        let delta = rotate_delta_angle(prev_a, prev_b, cur_a, cur_b);
        assert!(
            (delta - std::f32::consts::FRAC_PI_2).abs() < 1e-4,
            "expected a +PI/2 rotation, got {delta}"
        );
    }

    #[test]
    fn rotate_delta_zero_when_touches_do_not_rotate() {
        let a = Vec2::new(-4.0, 2.0);
        let b = Vec2::new(4.0, 2.0);
        assert!(rotate_delta_angle(a, b, a, b).abs() < 1e-6);
    }

    #[test]
    fn rotate_delta_wraps_across_the_seam() {
        // Angle goes from just past +PI to just past -PI (a small clockwise
        // nudge across the wraparound seam) -- must report a small delta,
        // not a near-2*PI jump.
        let ang_before = std::f32::consts::PI - 0.05;
        let ang_after = -std::f32::consts::PI + 0.05;
        let prev_a = Vec2::ZERO;
        let prev_b = Vec2::new(ang_before.cos(), ang_before.sin()) * 5.0;
        let cur_a = Vec2::ZERO;
        let cur_b = Vec2::new(ang_after.cos(), ang_after.sin()) * 5.0;
        let delta = rotate_delta_angle(prev_a, prev_b, cur_a, cur_b);
        assert!(
            delta.abs() < 0.2,
            "expected a small delta across the wrap seam, got {delta}"
        );
    }

    #[test]
    fn sort_by_id_is_deterministic_regardless_of_input_order() {
        // Stand-in for HashMap iteration order not being guaranteed stable
        // across frames for the same key set: insert id 7 before id 3 and
        // confirm sorting always yields id 3 first.
        let unsorted = vec![
            snap(7, Vec2::new(1.0, 1.0), Vec2::new(1.0, 1.0)),
            snap(3, Vec2::new(2.0, 2.0), Vec2::new(2.0, 2.0)),
        ];
        let sorted = sort_by_id(unsorted);
        assert_eq!(sorted[0].id, 3);
        assert_eq!(sorted[1].id, 7);
    }
}
