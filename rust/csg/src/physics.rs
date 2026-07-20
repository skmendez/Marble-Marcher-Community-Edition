//! M5: marble/collider physics against an [`Object`], pure logic (glam only,
//! no bevy — see rust/DESIGN.md §7).
//!
//! Direct port of `Scene::UpdateMarble` / `Scene::MarbleCollision`
//! (src/Scene.cpp), supporting **both** physics models present in this
//! repo's C++ history as [`GravityMode`]:
//!
//! - [`GravityMode::Rolling`]: the original upstream behavior — gravity on,
//!   kill plane on, camera-**yaw**-relative rolling
//!   (`v = (dx*cos - dy*sin, 0, -dy*cos - dx*sin)`, horizontal only).
//! - [`GravityMode::Flying`]: this branch's in-progress experimental
//!   mechanic ("new camera/movement mechanics" commit) — no gravity
//!   (`force = 0`), no kill plane, true 3D camera-**yaw-and-pitch**-relative
//!   thrust (so `W`/`S` fly along wherever the camera is actually looking,
//!   including up/down, not just horizontally). Collision/bounce off
//!   geometry still applies in both modes — only gravity and the movement
//!   formula differ; see [`step_marble`]'s doc for the exact per-mode math,
//!   derived from `Scene::MakeCameraRotation`/`FOCAL_DIST` (src/Scene.h/.cpp).
//!
//! The `std::cerr` debug print in `MarbleCollision` is not ported in either
//! mode.

use glam::{Vec2, Vec3};

use crate::{Object, Params};

/// Which physics model [`step_marble`] uses. See the module doc for what
/// each mode is a port of. Defaults to `Flying`: the default deployed scene
/// (`SceneKind::MengerSponge`, see `app/src/render.rs`) has no authored
/// level data (start position tuned to rest on a surface, kill plane,
/// etc.) the way the demo level does, so free-flight is the sensible
/// default there; `G` still toggles to `Rolling` at any time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GravityMode {
    /// Original MMCE marble-rolling physics: gravity, kill plane, horizontal
    /// camera-yaw-relative movement.
    Rolling,
    /// This branch's experimental zero-gravity free-flight mechanic: no
    /// gravity, no kill plane, full 3D camera-relative thrust.
    #[default]
    Flying,
}

/// A single point-collider sample: an offset from the owning body's origin
/// plus a radius. The marble is the one-sample case (`offset = 0`). This is
/// the hook for future CSG-vs-CSG collision (point-shell sampling of one
/// body against the other's DE) without reworking the physics API — see
/// DESIGN.md §7 and MILESTONES.md's "Later" section.
#[derive(Clone, Copy, Debug)]
pub struct SamplePoint {
    pub offset: Vec3,
    pub radius: f32,
}

/// Frame-rate-locked physics constants (per 60 Hz tick), C++ `Scene.cpp`
/// file-scope `static const` values (ported verbatim — DESIGN.md §7), plus
/// which [`GravityMode`] to simulate.
#[derive(Clone, Copy, Debug)]
pub struct PhysicsConfig {
    pub ground_force: f32,
    pub air_force: f32,
    pub ground_friction: f32,
    pub air_friction: f32,
    pub gravity: f32,
    pub ground_ratio: f32,
    pub bounce: f32,
    pub mode: GravityMode,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            ground_force: 0.008,
            air_force: 0.004,
            ground_friction: 0.99,
            air_friction: 0.995,
            gravity: 0.005,
            ground_ratio: 1.15,
            bounce: 1.2,
            mode: GravityMode::default(),
        }
    }
}

/// `sqrt(3)`, C++ `Scene.h`'s `#define FOCAL_DIST 1.73205080757` — the
/// camera-space distance used to build the [`GravityMode::Flying`] thrust
/// direction (see `step_marble`).
const FOCAL_DIST: f32 = 1.732_050_8;

/// Projects `v` onto the XZ (horizontal) plane and renormalizes, for
/// [`GravityMode::Rolling`]'s horizontal-only movement (`step_marble`) --
/// falls back to `default_direction` (always some horizontal unit vector)
/// in the degenerate case of `v` pointing exactly straight up or down,
/// where the projection is zero and has no well-defined direction.
fn horizontal(v: Vec3, default_direction: Vec3) -> Vec3 {
    let flat = Vec3::new(v.x, 0.0, v.z);
    if flat.length_squared() > 1e-12 {
        flat.normalize()
    } else {
        default_direction
    }
}

/// The C++ `MarbleCollision` hard-codes this threshold (`marble_rad *
/// 0.001f`) rather than exposing it as a tunable constant; kept as a literal
/// here too.
const CRUSH_RATIO: f32 = 0.001;

/// Physics substeps per tick (C++ `Scene.cpp`'s file-scope `static const int
/// num_phys_steps = 6;`). Gravity and position integration are split across
/// this many substeps, with a full collision resolution after each one —
/// see [`step_marble`]'s doc for why this substepping (not a single big step
/// per tick) is load-bearing.
pub const NUM_PHYS_STEPS: u32 = 6;

/// The marble body: world-space position, velocity, and radius.
#[derive(Clone, Copy, Debug)]
pub struct Marble {
    pub pos: Vec3,
    pub vel: Vec3,
    pub rad: f32,
    /// The raw camera-relative thrust vector `step_marble` computed on its
    /// most recent call (`v` in that function's doc — before the per-tick
    /// force-scale `f` multiply), or `Vec3::ZERO` if that tick's input was
    /// `Vec2::ZERO`. Exposed purely for visualization (`debug_gizmos.rs`):
    /// callers that want to verify thrust direction can read this directly
    /// instead of re-deriving an approximation of it from yaw/pitch, which
    /// is exactly the class of bug (`step_marble`'s doc) this field exists
    /// to let a caller check against *without* repeating.
    pub last_thrust: Vec3,
}

impl Marble {
    /// Spawns a marble at `start` with zero velocity (C++ `ResetLevel`'s
    /// marble init: `marble_pos = start; marble_vel = 0`).
    pub fn spawn(start: Vec3, rad: f32) -> Self {
        Self {
            pos: start,
            vel: Vec3::ZERO,
            rad,
            last_thrust: Vec3::ZERO,
        }
    }

    /// Resets position to `start` and zeroes velocity in place (crush /
    /// kill-plane respawn).
    pub fn respawn(&mut self, start: Vec3) {
        self.pos = start;
        self.vel = Vec3::ZERO;
        self.last_thrust = Vec3::ZERO;
    }
}

/// Result of a [`collide`] query: whether any sample was resting on/near a
/// surface, whether any sample was crushed (fully embedded), and the
/// corrected body position (push-outs from all non-crushed, overlapping
/// samples summed back onto `body_pos` — DESIGN.md §7).
#[derive(Clone, Copy, Debug)]
pub struct CollisionOutcome {
    pub on_ground: bool,
    pub crushed: bool,
    pub pos: Vec3,
}

/// Exact port of `Scene::MarbleCollision` (src/Scene.cpp:1072, minus the
/// debug `std::cerr` print), generalized to a body made of `samples` point
/// colliders (DESIGN.md §7's collider abstraction). Each sample is queried
/// at `body_pos + sample.offset`; per-sample logic is identical to the C++:
///
/// ```text
/// de = obj.de(vec4(pos, 1))
/// if de >= rad { on_ground |= de < rad * ground_ratio; continue }
/// if de < rad * 0.001 { crushed = true; break }   // C++ returns immediately
/// np = obj.nearest_point(vec4(pos, 1))
/// d = np - pos; dn = normalize(d); dv = dot(vel, dn)
/// pos -= dn * rad - d
/// vel -= dn * (dv * bounce)
/// on_ground = true
/// ```
///
/// `vel` is mutated in place (matches the C++ `marble_vel` in/out); the
/// corrected position is returned rather than mutated in place, since a
/// crushed outcome must NOT move the body (the caller decides how to
/// respond — `step_marble` respawns).
pub fn collide(
    obj: &Object,
    params: &Params,
    body_pos: Vec3,
    vel: &mut Vec3,
    samples: &[SamplePoint],
    cfg: &PhysicsConfig,
) -> CollisionOutcome {
    let mut on_ground = false;
    let mut crushed = false;
    let mut pos = body_pos;

    for sample in samples {
        let rad = sample.radius;
        let sample_pos = pos + sample.offset;
        let de = obj.de(sample_pos.extend(1.0), params);

        if de >= rad {
            on_ground |= de < rad * cfg.ground_ratio;
            continue;
        }

        if de < rad * CRUSH_RATIO {
            crushed = true;
            break; // C++ returns immediately on crush; no further samples matter.
        }

        let np = obj.nearest_point(sample_pos.extend(1.0), params);
        let d = np - sample_pos;
        let dn = d.normalize();
        let dv = vel.dot(dn);
        pos -= dn * rad - d;
        *vel -= dn * (dv * cfg.bounce);
        on_ground = true;
    }

    CollisionOutcome {
        on_ground,
        crushed,
        pos,
    }
}

/// Outcome of a single [`step_marble`] tick, for callers that want to react
/// to respawns (sound effects, UI, etc.).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepEvent {
    None,
    RespawnedCrushed,
    RespawnedFell,
}

/// One 60 Hz physics tick — direct port of `Scene::UpdateMarble`'s
/// non-free-camera branch, branching on `cfg.mode` (see the module doc for
/// what each [`GravityMode`] is a port of and why gravity/kill-plane/the
/// movement formula are the only things that differ between them —
/// substepping, collision, and friction are identical in both modes).
///
/// ⚠ The substep structure below is load-bearing, in both modes — an
/// earlier draft of this port (and of DESIGN.md §7) ran
/// gravity/integration/collision once per tick with the full velocity, and
/// that tunnels: `marble_pos += marble_vel` in one big jump skips over thin
/// fractal struts, and — worse — a single collision correction per tick
/// leaves the *tangential* component of velocity untouched (only the normal
/// component is resolved), so a marble resting on a sloped strut drifts
/// sideways for many ticks (friction alone decays it far too slowly) until
/// it slides off and free-falls. The real C++ avoids both problems by
/// substepping: gravity and position integration are split across
/// [`NUM_PHYS_STEPS`] substeps, with a full collision resolution after
/// *each* substep (`Scene::UpdateMarble`, src/Scene.cpp — `num_phys_steps`).
/// Keyboard input force and friction are still applied once per full tick,
/// after the substep loop, exactly as in the C++.
///
/// `input` is `(dx, dy)` in C++'s convention: `dx` is the strafe axis, `dy`
/// is the forward/back axis (see the app-side WASD mapping for the sign
/// convention). `cam_forward`/`cam_right` are the camera's actual current
/// unit basis vectors (e.g. `CameraOrbit::forward()`/`orientation * Vec3::X`
/// on the app side) — **not** yaw/pitch angles re-derived via trig, which an
/// earlier version of this function took instead. That version had a real
/// bug: its `Flying`-mode formula's vertical component came out with the
/// opposite sign from `cam_forward`'s actual vertical component (confirmed
/// by direct derivation, not just suspicion — the horizontal terms matched
/// `cam_forward` exactly, only the vertical term's sign differed), so
/// thrust was only ever exactly toward/away from the camera when pitch was
/// zero, and visibly *not* toward/away from the camera at any other tilt —
/// reported as "the movement force isn't directly towards or away from the
/// camera." Taking the real basis vectors directly instead of re-deriving
/// an approximation of them from angles eliminates that whole class of
/// bug (there's no second, independently-written trig formula that has to
/// happen to agree in sign convention with the first).
///
/// The full 3D `cam_forward` is conceptually only used in
/// [`GravityMode::Flying`] (the original rolling movement is horizontal-only
/// and ignores the camera's vertical tilt, matching the pristine C++) —
/// `Rolling` derives its horizontal-only direction by projecting
/// `cam_forward`/`cam_right` onto the XZ plane and renormalizing
/// (falling back to `Vec3::NEG_Z`/`Vec3::X` in the degenerate case of
/// looking exactly straight up or down, where that projection is zero).
/// Order, matching the C++ statement order in `UpdateMarble`/`MarbleCollision`:
///
/// 1. [`NUM_PHYS_STEPS`] substeps, each: gravity (`Rolling`: `vel.y -= rad *
///    gravity / NUM_PHYS_STEPS`; `Flying`: none — C++'s `force = 0;`
///    override), then `pos += vel / NUM_PHYS_STEPS`, then collide (marble as
///    one [`SamplePoint`]) at the new `pos` — `on_ground` accumulates via OR
///    across substeps, and a crush respawns to `start` immediately,
///    reporting [`StepEvent::RespawnedCrushed`] in **both** modes (the C++
///    instead sets `pos.y = -9999` and, in `Rolling` mode, lets the
///    *end-of-tick* kill-plane check below catch it on the same tick — with
///    the kill-plane check disabled entirely in `Flying` mode, a
///    C++-faithful crush there would leave the marble permanently stuck at
///    `y = -9999`; we short-circuit to an immediate respawn in both modes
///    instead, which matches `Rolling`'s observable outcome exactly and
///    gives `Flying` a sane one instead of an unreachable void).
/// 2. input force, once per full tick: `f = rad * (on_ground ?
///    ground_force : air_force)`, then per mode:
///    - `Rolling`: `v = dy*horizontal(cam_forward) + dx*horizontal(cam_right)`
///      (horizontal only, see above).
///    - `Flying`: full 3D camera-relative thrust: `v = dy*FOCAL_DIST*cam_forward
///      + dx*cam_right` — i.e. `W`/`S` fly along wherever the camera is
///      actually pointed (including up/down), `A`/`D` strafe along its
///        actual current right vector.
///
///    Either way: `vel += v * f`.
/// 3. friction, once per full tick: `vel *= on_ground ? ground_friction :
///    air_friction`.
/// 4. kill-plane (`Rolling` only — disabled in `Flying`, matching the C++'s
///    commented-out check): `pos.y < kill_y` -> respawn to `start`, report
///    [`StepEvent::RespawnedFell`].
#[allow(clippy::too_many_arguments)]
pub fn step_marble(
    marble: &mut Marble,
    obj: &Object,
    params: &Params,
    input: Vec2,
    cam_forward: Vec3,
    cam_right: Vec3,
    cfg: &PhysicsConfig,
    kill_y: f32,
    start: Vec3,
) -> StepEvent {
    // C++ normalizes (dx, dy) to unit magnitude when the combined input
    // exceeds 1 (e.g. two WASD keys held at once) so diagonal movement
    // isn't faster than axis-aligned movement.
    let mut dx = input.x;
    let mut dy = input.y;
    let mag2 = dx * dx + dy * dy;
    if mag2 > 1.0 {
        let mag = mag2.sqrt();
        dx /= mag;
        dy /= mag;
    }

    let steps = NUM_PHYS_STEPS as f32;
    let mut on_ground = false;

    for _ in 0..NUM_PHYS_STEPS {
        if cfg.mode == GravityMode::Rolling {
            marble.vel.y -= marble.rad * cfg.gravity / steps;
        }
        marble.pos += marble.vel / steps;

        let samples = [SamplePoint {
            offset: Vec3::ZERO,
            radius: marble.rad,
        }];
        let outcome = collide(obj, params, marble.pos, &mut marble.vel, &samples, cfg);
        if outcome.crushed {
            marble.respawn(start);
            return StepEvent::RespawnedCrushed;
        }
        marble.pos = outcome.pos;
        on_ground |= outcome.on_ground;
    }

    // Input force, once per full tick.
    let f = marble.rad * if on_ground { cfg.ground_force } else { cfg.air_force };
    let v = match cfg.mode {
        GravityMode::Rolling => {
            dy * horizontal(cam_forward, Vec3::NEG_Z) + dx * horizontal(cam_right, Vec3::X)
        }
        GravityMode::Flying => dy * FOCAL_DIST * cam_forward + dx * cam_right,
    };
    marble.last_thrust = v;
    marble.vel += v * f;

    // Friction, once per full tick.
    marble.vel *= if on_ground { cfg.ground_friction } else { cfg.air_friction };

    // Kill plane (Rolling only).
    if cfg.mode == GravityMode::Rolling && marble.pos.y < kill_y {
        marble.respawn(start);
        return StepEvent::RespawnedFell;
    }

    StepEvent::None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenes::{beware_of_bumps, demo_scene, set_fractal_params};
    use crate::ScalarValue;

    fn setup_demo() -> (Object, Params) {
        let mut params = Params::new();
        let (object, handles) = demo_scene(&mut params);
        set_fractal_params(
            &mut params,
            &handles,
            beware_of_bumps::SCALE,
            beware_of_bumps::ANG1,
            beware_of_bumps::ANG2,
            beware_of_bumps::SHIFT,
            beware_of_bumps::COLOR,
            beware_of_bumps::ITERS,
        );
        (object, params)
    }

    /// A marble dropped somewhat above the demo scene's start position falls
    /// (y decreases) and, within a few thousand ticks, comes to rest on a
    /// surface: on_ground (via `de` bracketed around `rad`), tiny velocity,
    /// and `de(pos)` within `[rad*0.5, rad*ground_ratio*1.5]`.
    ///
    /// (The level's authored start position is already resting on solid
    /// ground — see [`marble_at_start_is_immediately_stable`] — so this test
    /// drops from `start + 0.5` on Y to actually exercise the fall.)
    #[test]
    fn marble_falls_and_settles() {
        let (object, params) = setup_demo();
        // Explicit Rolling: these tests exercise gravity/kill-plane
        // behavior, which is no longer PhysicsConfig::default()'s mode.
        let cfg = PhysicsConfig {
            mode: GravityMode::Rolling,
            ..PhysicsConfig::default()
        };
        let rad = beware_of_bumps::MARBLE_RAD;
        let start = beware_of_bumps::START;
        let drop_from = start + Vec3::new(0.0, 0.5, 0.0);
        let mut marble = Marble::spawn(drop_from, rad);

        let y0 = marble.pos.y;
        const MAX_TICKS: u32 = 20_000;
        let mut settled_at = None;
        let mut min_y_seen = y0;

        for tick in 0..MAX_TICKS {
            let event = step_marble(
                &mut marble,
                &object,
                &params,
                Vec2::ZERO,
                Vec3::NEG_Z,
                Vec3::X,
                &cfg,
                beware_of_bumps::KILL_Y,
                start,
            );
            assert_eq!(
                event,
                StepEvent::None,
                "marble should not respawn while settling onto the start platform (tick {tick})"
            );
            min_y_seen = min_y_seen.min(marble.pos.y);

            let de = object.de(marble.pos.extend(1.0), &params);
            let speed = marble.vel.length();
            // "Settled" here means small relative to the marble's own scale,
            // not near-machine-zero: resting-contact jitter from the
            // collision correction's bounce term (never purely dissipative
            // for the tangential component) stabilizes around 1e-2..5e-2
            // times `rad`, not lower — confirmed empirically by tracing this
            // exact scenario (see rust/csg/examples/step_probe.rs).
            if speed < 0.05 * rad && de >= rad * 0.5 && de <= rad * cfg.ground_ratio * 1.5 {
                settled_at = Some((tick, speed, de));
                break;
            }
        }

        assert!(
            min_y_seen < y0,
            "marble never fell (min y {min_y_seen} vs start y {y0})"
        );

        let (tick, speed, de) =
            settled_at.expect("marble did not settle onto a surface within MAX_TICKS");
        eprintln!(
            "settled after {} ticks: |vel|={:.6e}, de/rad={:.4}",
            tick + 1,
            speed,
            de / rad
        );
    }

    /// The level's authored start position is already resting on solid
    /// ground (this is what a real level's spawn point means): a marble
    /// spawned exactly at `beware_of_bumps::START` with zero velocity stays
    /// essentially motionless — no perceptible fall, no drift — for many
    /// ticks. This is also the regression test for the tangential-drift bug
    /// a single-substep-per-tick version of `step_marble` had (see its doc
    /// comment): that version let the marble creep sideways off this exact
    /// ledge and fall through the kill plane within ~100 ticks.
    #[test]
    fn marble_at_start_is_immediately_stable() {
        let (object, params) = setup_demo();
        // Explicit Rolling: these tests exercise gravity/kill-plane
        // behavior, which is no longer PhysicsConfig::default()'s mode.
        let cfg = PhysicsConfig {
            mode: GravityMode::Rolling,
            ..PhysicsConfig::default()
        };
        let rad = beware_of_bumps::MARBLE_RAD;
        let start = beware_of_bumps::START;
        let mut marble = Marble::spawn(start, rad);

        const TICKS: u32 = 600; // 10 seconds at 60 Hz
        for tick in 0..TICKS {
            let event = step_marble(
                &mut marble,
                &object,
                &params,
                Vec2::ZERO,
                Vec3::NEG_Z,
                Vec3::X,
                &cfg,
                beware_of_bumps::KILL_Y,
                start,
            );
            assert_eq!(
                event,
                StepEvent::None,
                "marble should not respawn while resting at the authored start (tick {tick})"
            );
        }

        let horizontal_drift = (marble.pos - start).with_y(0.0).length();
        assert!(
            horizontal_drift < 0.1 * rad,
            "marble drifted {horizontal_drift} sideways off its start ledge (rad={rad})"
        );
        assert!(
            marble.vel.length() < 1e-3 * rad,
            "marble should have settled to near-zero velocity, got |vel|={}",
            marble.vel.length()
        );
    }

    /// `collide` reports `crushed` when the body is deeply embedded, and
    /// `step_marble` respawns to `start` with zero velocity on crush.
    #[test]
    fn crush_respawns_to_start() {
        // A trivial object the marble starts deeply embedded in (de very
        // negative -> well below the `rad*0.001` crush threshold).
        let params = Params::new();
        let object = Object::Sphere {
            radius: ScalarValue::Const(5.0),
        };
        let rad = 0.02;
        let start = Vec3::new(1.0, 2.0, 3.0);
        let mut marble = Marble::spawn(Vec3::ZERO, rad);
        marble.vel = Vec3::new(0.1, 0.2, 0.3);

        let event = step_marble(
            &mut marble,
            &object,
            &params,
            Vec2::ZERO,
            Vec3::NEG_Z,
            Vec3::X,
            &PhysicsConfig::default(),
            -1000.0,
            start,
        );

        assert_eq!(event, StepEvent::RespawnedCrushed);
        assert_eq!(marble.pos, start);
        assert_eq!(marble.vel, Vec3::ZERO);
    }

    /// A marble falling through empty space (far from all geometry) hits
    /// the kill plane and respawns to `start` with zero velocity.
    #[test]
    fn kill_plane_respawns_falling_marble() {
        let (object, params) = setup_demo();
        let rad = beware_of_bumps::MARBLE_RAD;
        // Far outside both the classic fractal's cuboid (half-extent 6) and
        // the creme-spheres bounding sphere (radius 6): DE stays large and
        // positive the whole way down, so the marble free-falls to the kill
        // plane without ever colliding.
        let start = beware_of_bumps::START;
        let drop_from = Vec3::new(50.0, 50.0, 50.0);
        let mut marble = Marble::spawn(drop_from, rad);
        let kill_y = beware_of_bumps::KILL_Y;

        // Explicit Rolling: this test relies on gravity to make the marble
        // fall at all, which is no longer PhysicsConfig::default()'s mode.
        let cfg = PhysicsConfig {
            mode: GravityMode::Rolling,
            ..PhysicsConfig::default()
        };

        const MAX_TICKS: u32 = 20_000;
        let mut event = StepEvent::None;
        for _ in 0..MAX_TICKS {
            event = step_marble(
                &mut marble,
                &object,
                &params,
                Vec2::ZERO,
                Vec3::NEG_Z,
                Vec3::X,
                &cfg,
                kill_y,
                start,
            );
            if event != StepEvent::None {
                break;
            }
        }

        assert_eq!(event, StepEvent::RespawnedFell);
        assert_eq!(marble.pos, start);
        assert_eq!(marble.vel, Vec3::ZERO);
    }

    /// After any non-crushed `collide()` call, the push-out actually
    /// separates the body from the surface: `de(corrected_pos) >= rad*0.9`.
    #[test]
    fn collide_pushes_out_to_surface() {
        let params = Params::new();
        let object = Object::Sphere {
            radius: ScalarValue::Const(5.0),
        };
        let rad = 0.1;
        // Slightly overlapping the sphere's surface from outside
        // (de = 0.005, i.e. 0.05*rad -- above the crush threshold, below
        // `rad`, so this exercises the nearest-point push-out branch).
        let body_pos = Vec3::new(5.005, 0.0, 0.0);
        let mut vel = Vec3::new(-0.5, 0.0, 0.0);
        // Explicit Rolling: these tests exercise gravity/kill-plane
        // behavior, which is no longer PhysicsConfig::default()'s mode.
        let cfg = PhysicsConfig {
            mode: GravityMode::Rolling,
            ..PhysicsConfig::default()
        };

        let samples = [SamplePoint {
            offset: Vec3::ZERO,
            radius: rad,
        }];
        let outcome = collide(&object, &params, body_pos, &mut vel, &samples, &cfg);

        assert!(!outcome.crushed);
        assert!(outcome.on_ground);
        let de = object.de(outcome.pos.extend(1.0), &params);
        assert!(
            de >= rad * 0.9,
            "push-out should separate the body from the surface: de={de}, rad={rad}"
        );
    }

    /// Sanity check on the collider abstraction itself: a multi-sample body
    /// (e.g. a coarse point-shell) reports crush if any sample is embedded,
    /// even when another sample is merely touching the surface.
    #[test]
    fn multi_sample_collide_reports_crush_from_any_sample() {
        let params = Params::new();
        let object = Object::Sphere {
            radius: ScalarValue::Const(5.0),
        };
        let mut vel = Vec3::ZERO;
        let samples = [
            SamplePoint {
                offset: Vec3::new(5.005, 0.0, 0.0),
                radius: 0.1,
            }, // fine: just outside, de=0.005 -- push-out branch
            SamplePoint {
                offset: Vec3::ZERO,
                radius: 0.1,
            }, // body_pos itself is the sphere's center: de=-5, crushed
        ];
        let outcome = collide(
            &object,
            &params,
            Vec3::ZERO,
            &mut vel,
            &samples,
            &PhysicsConfig::default(),
        );
        assert!(outcome.crushed);
    }

    /// [`GravityMode::Flying`]: no gravity — a marble floating in empty
    /// space with zero input stays exactly where it is (no free-fall).
    #[test]
    fn flying_mode_has_no_gravity() {
        let params = Params::new();
        // Far from all geometry, matching kill_plane_respawns_falling_marble's
        // "empty space" setup, but this time we're checking it does NOT fall.
        let object = Object::Sphere {
            radius: ScalarValue::Const(0.01),
        };
        let rad = 0.02;
        let start = Vec3::new(50.0, 50.0, 50.0);
        let mut marble = Marble::spawn(start, rad);
        let cfg = PhysicsConfig {
            mode: GravityMode::Flying,
            ..PhysicsConfig::default()
        };

        for _ in 0..600 {
            let event = step_marble(
                &mut marble,
                &object,
                &params,
                Vec2::ZERO,
                Vec3::NEG_Z,
                Vec3::X,
                &cfg,
                beware_of_bumps::KILL_Y,
                start,
            );
            assert_eq!(event, StepEvent::None);
        }
        assert_eq!(marble.pos, start, "flying mode must not apply gravity");
    }

    /// [`GravityMode::Flying`]: no kill plane — a marble given downward
    /// thrust flies straight through where `kill_y` would trigger a respawn
    /// in `Rolling` mode, and never respawns.
    #[test]
    fn flying_mode_has_no_kill_plane() {
        let params = Params::new();
        let object = Object::Sphere {
            radius: ScalarValue::Const(0.01),
        };
        let rad = 0.02;
        let start = Vec3::new(50.0, 50.0, 50.0);
        let mut marble = Marble::spawn(start, rad);
        let cfg = PhysicsConfig {
            mode: GravityMode::Flying,
            ..PhysicsConfig::default()
        };
        let kill_y = start.y - 1.0; // would trigger almost immediately in Rolling mode

        // dy > 0 with cam_forward pointing straight down flies downward per
        // step_marble's Flying-mode formula (v = dy * FOCAL_DIST * cam_forward).
        for _ in 0..600 {
            let event = step_marble(
                &mut marble,
                &object,
                &params,
                Vec2::new(0.0, 1.0),
                Vec3::NEG_Y, // straight down, regardless of yaw/pitch conventions
                Vec3::X,
                &cfg,
                kill_y,
                start,
            );
            assert_eq!(event, StepEvent::None, "Flying mode must never respawn from a kill plane");
        }
        assert!(
            marble.pos.y < kill_y,
            "expected the marble to have flown below kill_y ({}), got y={}",
            kill_y,
            marble.pos.y
        );
    }
}
