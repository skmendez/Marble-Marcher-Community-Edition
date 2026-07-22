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

use glam::{Mat3, Quat, Vec2, Vec3};

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
    // Reused across every contacting sample below instead of letting
    // `nearest_point` allocate a fresh fold-history `Vec` per call --
    // `nearest_point_scratch`'s own doc guarantees it's left empty after
    // each call, so it's always safe to reuse as-is for the next one.
    let mut hist = Vec::new();

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

        let np = obj.nearest_point_scratch(sample_pos.extend(1.0), params, &mut hist);
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

/// One player's per-tick input: WASD-style `(dx, dy)` (see [`step_marble`]'s
/// doc for the sign convention) plus the *camera orientation* they had at
/// that exact tick, as a single quaternion rather than separate
/// `cam_forward`/`cam_right` vectors.
///
/// This shape exists for multiplayer, not just this milestone's local
/// multi-marble loop: thrust direction depends on the controlling client's
/// live camera orientation (see [`step_marble`]'s doc on `cam_forward`/
/// `cam_right`), so once marbles are networked, a remote client resimulating
/// *your* marble needs *your* camera orientation at that tick, not theirs —
/// camera orientation stops being purely local presentation state and
/// becomes part of the reproducible per-tick input. A single `Quat` is a
/// clean, self-contained, serializable value for a future rollback engine to
/// snapshot/replay per player per tick, and it reuses the exact same
/// `orientation * Vec3::NEG_Z` / `orientation * Vec3::X` derivation
/// `physics_sys.rs` already does for the app's real `CameraOrbit` — no new
/// "aim direction" representation to invent or get the sign convention wrong
/// on (see [`Self::cam_basis`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlayerInput {
    pub dx: f32,
    pub dy: f32,
    pub orientation: Quat,
}

impl PlayerInput {
    /// Derives `(cam_forward, cam_right)` from `orientation`, matching
    /// `CameraOrbit::forward`/`CameraOrbit::eye_and_basis`'s own
    /// `orientation * Vec3::NEG_Z` / `orientation * Vec3::X` exactly — kept
    /// in one place so every consumer (this module, and a future rollback
    /// engine replaying logged inputs) agrees on the same derivation rather
    /// than each re-deriving it.
    pub fn cam_basis(&self) -> (Vec3, Vec3) {
        (self.orientation * Vec3::NEG_Z, self.orientation * Vec3::X)
    }
}

/// Inverse of [`PlayerInput::cam_basis`]: the unique unit quaternion `q` with
/// `q * Vec3::NEG_Z == cam_forward` and `q * Vec3::X == cam_right`, for
/// `cam_forward`/`cam_right` that are already known-orthonormal (as
/// `CameraOrbit`'s always are). Used only to let [`step_marble`]'s existing
/// vector-based signature stay a thin wrapper over [`step_marbles`] without
/// changing its callers or its extensive existing test coverage.
///
/// This is *not* the kind of reconstruction that caused the real bug
/// `step_marble`'s doc describes (re-deriving forward/right from yaw/pitch
/// *angles* via a second, independently-written trig formula that had to
/// happen to agree in sign convention with the first) — it's a single,
/// standard, mathematically well-defined operation on an already-orthonormal
/// basis, verified by a round-trip test (`orientation_from_basis_roundtrips`)
/// rather than just assumed correct.
fn orientation_from_basis(cam_forward: Vec3, cam_right: Vec3) -> Quat {
    // `up = cross(right, forward)` matches `CameraOrbit::eye_and_basis`'s own
    // `up0 = cross(right0, forward)` handedness convention exactly.
    let up = cam_right.cross(cam_forward);
    // `orientation`'s rotation matrix columns are `orientation * X/Y/Z` --
    // and `orientation * Z == -cam_forward`, *not* `cam_forward`, since
    // `forward` is defined as `orientation * NEG_Z` throughout this
    // codebase (`CameraOrbit::forward`). Using `cam_forward` directly as
    // the third column here builds a *reflection* (determinant -1, not a
    // valid rotation) rather than the rotation that actually produces it --
    // caught by `orientation_from_basis_roundtrips` below, which failed
    // outright (not subtly) for exactly this reason on the first attempt.
    Quat::from_mat3(&Mat3::from_cols(cam_right, up, -cam_forward))
}

/// One 60 Hz physics tick for a single marble — thin wrapper over
/// [`step_marbles`] (which has the full algorithm doc: substep structure,
/// per-mode movement formula, the `cam_forward`/`cam_right`-not-angles bug
/// history) for the one-marble case, kept so every existing caller/test
/// using this exact vector-based signature needs no changes.
/// `cam_forward`/`cam_right` are converted to an equivalent
/// [`PlayerInput::orientation`] via [`orientation_from_basis`] (a safe,
/// well-defined reconstruction from an already-orthonormal basis — see that
/// function's doc for why this is *not* the class of bug `step_marbles`'
/// doc describes) purely so there's one substep implementation to maintain,
/// not two that can silently drift apart.
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
    let mut marbles = [*marble];
    let player_input = PlayerInput {
        dx: input.x,
        dy: input.y,
        orientation: orientation_from_basis(cam_forward, cam_right),
    };
    let events = step_marbles(
        &mut marbles,
        &[player_input],
        obj,
        params,
        cfg,
        kill_y,
        &[start],
    );
    *marble = marbles[0];
    events[0]
}

/// Sphere-vs-sphere collision between every pair of `marbles`, resolved in
/// **stable slice-index order** (`i < j`, never a `HashMap` or any other
/// unordered-iteration structure) — this determinism discipline doesn't
/// matter for anything *yet* (no networking exists in this milestone), but a
/// future rollback engine's resimulation depends on every client reaching
/// bit-identical results from the same inputs, so the discipline is baked in
/// from the start rather than retrofitted once it's load-bearing.
///
/// Equal-mass elastic response regardless of radius (a simplifying
/// assumption — scaling the impulse by an actual per-marble mass, e.g.
/// proportional to `rad^3`, is a reasonable future refinement, not needed
/// here) using the same `bounce` restitution [`collide`] already applies to
/// marble-vs-fractal collisions, so a marble bounces off another marble with
/// the same "springiness" it bounces off geometry with.
///
/// Called once per physics substep (not once per tick) inside
/// [`step_marbles`] — the same anti-tunneling reason [`NUM_PHYS_STEPS`]
/// substepping exists for marble-vs-fractal collision (this function's
/// caller's doc): two fast-moving marbles can still pass through each other
/// in one big per-tick position update.
pub fn collide_marbles(marbles: &mut [Marble], bounce: f32) {
    let n = marbles.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let delta = marbles[j].pos - marbles[i].pos;
            let dist = delta.length();
            let min_dist = marbles[i].rad + marbles[j].rad;
            if dist >= min_dist || dist < 1e-6 {
                // `dist < 1e-6`: exactly-coincident centers have no
                // well-defined push-out normal — an extremely unlikely edge
                // case (two marbles spawned/teleported to the same point),
                // skip rather than divide by ~zero.
                continue;
            }
            let normal = delta / dist;
            let overlap = min_dist - dist;
            // Push equally apart along the normal (mirrors `collide`'s own
            // push-out-to-surface correction, just symmetric between two
            // moving bodies instead of one moving body and static geometry).
            marbles[i].pos -= normal * (overlap * 0.5);
            marbles[j].pos += normal * (overlap * 0.5);

            // Standard equal-mass 1D-along-normal collision response with
            // restitution `bounce`: each marble's velocity picks up
            // `(1+bounce)/2 * sep_speed` along the normal (derivable from
            // conserving momentum with `m_i == m_j` and defining `bounce` as
            // the post/pre approach-speed ratio) — at `bounce == 1` this
            // reduces to the textbook "swap the normal-component velocities"
            // equal-mass elastic collision.
            let rel_vel = marbles[j].vel - marbles[i].vel;
            let sep_speed = rel_vel.dot(normal);
            if sep_speed < 0.0 {
                // Only resolve when actually approaching (`sep_speed < 0`,
                // `normal` pointing i->j) — a pair already separating needs
                // no velocity correction, just the position push-out above.
                let impulse = normal * (sep_speed * (1.0 + bounce) * 0.5);
                marbles[i].vel += impulse;
                marbles[j].vel -= impulse;
            }
        }
    }
}

/// One 60 Hz physics tick for `marbles.len()` bodies sharing one
/// `obj`/`cfg`/`kill_y`, each driven by its own [`PlayerInput`] — direct
/// generalization of the original single-marble port of `Scene::UpdateMarble`'s
/// non-free-camera branch (`Scene::MarbleCollision`, src/Scene.cpp), branching
/// on `cfg.mode` (see the module doc for what each [`GravityMode`] is a port
/// of and why gravity/kill-plane/the movement formula are the only things
/// that differ between them — substepping, collision, and friction are
/// identical in both modes), with [`collide_marbles`] added as one more
/// per-substep phase for marble-vs-marble collision.
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
/// after the substep loop, exactly as in the C++. The same anti-tunneling
/// reasoning is why [`collide_marbles`] also runs once per substep, not once
/// per tick — see that function's doc.
///
/// `PlayerInput::dx`/`dy` follow C++'s convention: `dx` is the strafe axis,
/// `dy` is the forward/back axis (see the app-side WASD mapping for the sign
/// convention). [`PlayerInput::cam_basis`] derives the camera's actual
/// current unit basis vectors (`orientation * Vec3::NEG_Z` /
/// `orientation * Vec3::X`) — **not** yaw/pitch angles re-derived via trig,
/// which an earlier version of this function took instead. That version had
/// a real bug: its `Flying`-mode formula's vertical component came out with
/// the opposite sign from `cam_forward`'s actual vertical component
/// (confirmed by direct derivation, not just suspicion — the horizontal
/// terms matched `cam_forward` exactly, only the vertical term's sign
/// differed), so thrust was only ever exactly toward/away from the camera
/// when pitch was zero, and visibly *not* toward/away from the camera at any
/// other tilt — reported as "the movement force isn't directly towards or
/// away from the camera." Taking the real basis vectors directly instead of
/// re-deriving an approximation of them from angles eliminates that whole
/// class of bug (there's no second, independently-written trig formula that
/// has to happen to agree in sign convention with the first).
///
/// The full 3D `cam_forward` is conceptually only used in
/// [`GravityMode::Flying`] (the original rolling movement is horizontal-only
/// and ignores the camera's vertical tilt, matching the pristine C++) —
/// `Rolling` derives its horizontal-only direction by projecting
/// `cam_forward`/`cam_right` onto the XZ plane and renormalizing
/// (falling back to `Vec3::NEG_Z`/`Vec3::X` in the degenerate case of
/// looking exactly straight up or down, where that projection is zero).
/// Order, matching the C++ statement order in `UpdateMarble`/`MarbleCollision`,
/// applied to every marble in the same stable index order every phase (for
/// the same determinism reason [`collide_marbles`] documents):
///
/// 1. [`NUM_PHYS_STEPS`] substeps, each, per marble: gravity (`Rolling`:
///    `vel.y -= rad * gravity / NUM_PHYS_STEPS`; `Flying`: none — C++'s
///    `force = 0;` override), then `pos += vel / NUM_PHYS_STEPS`, then
///    collide (marble as one [`SamplePoint`]) at the new `pos` —
///    `on_ground` accumulates via OR across substeps, and a crush respawns
///    to `start` immediately, reporting [`StepEvent::RespawnedCrushed`] in
///    **both** modes (the C++ instead sets `pos.y = -9999` and, in
///    `Rolling` mode, lets the *end-of-tick* kill-plane check below catch
///    it on the same tick — with the kill-plane check disabled entirely in
///    `Flying` mode, a C++-faithful crush there would leave the marble
///    permanently stuck at `y = -9999`; we short-circuit to an immediate
///    respawn in both modes instead, which matches `Rolling`'s observable
///    outcome exactly and gives `Flying` a sane one instead of an
///    unreachable void) — then, once every marble has had this phase,
///    [`collide_marbles`] resolves marble-vs-marble collision for the
///    substep.
/// 2. input force, once per full tick per marble: `f = rad * (on_ground ?
///    ground_force : air_force)`, then per mode:
///    - `Rolling`: `v = dy*horizontal(cam_forward) + dx*horizontal(cam_right)`
///      (horizontal only, see above).
///    - `Flying`: full 3D camera-relative thrust: `v = dy*FOCAL_DIST*cam_forward
///      + dx*cam_right` — i.e. `W`/`S` fly along wherever the camera is
///      actually pointed (including up/down), `A`/`D` strafe along its
///        actual current right vector.
///
///    Either way: `vel += v * f`.
/// 3. friction, once per full tick per marble: `vel *= on_ground ?
///    ground_friction : air_friction`.
/// 4. kill-plane (`Rolling` only — disabled in `Flying`, matching the C++'s
///    commented-out check), once per full tick per marble: `pos.y < kill_y`
///    -> respawn to `start`, report [`StepEvent::RespawnedFell`].
///
/// A marble that's crushed (by the fractal) part-way through a tick stops
/// being touched for the rest of that tick — no further substeps, no input
/// force/friction/kill-plane — matching the original single-marble
/// `step_marble`'s semantics of returning immediately on crush, just
/// per-marble instead of terminating the whole call.
///
/// `marbles`, `inputs`, and `starts` must be the same length (asserted).
#[allow(clippy::too_many_arguments)]
pub fn step_marbles(
    marbles: &mut [Marble],
    inputs: &[PlayerInput],
    obj: &Object,
    params: &Params,
    cfg: &PhysicsConfig,
    kill_y: f32,
    starts: &[Vec3],
) -> Vec<StepEvent> {
    assert_eq!(marbles.len(), inputs.len(), "one input per marble");
    assert_eq!(marbles.len(), starts.len(), "one start position per marble");

    let steps = NUM_PHYS_STEPS as f32;
    let mut on_ground = vec![false; marbles.len()];
    let mut events = vec![StepEvent::None; marbles.len()];

    for _ in 0..NUM_PHYS_STEPS {
        for i in 0..marbles.len() {
            if events[i] != StepEvent::None {
                continue; // crushed earlier this tick -- see this fn's doc.
            }
            if cfg.mode == GravityMode::Rolling {
                marbles[i].vel.y -= marbles[i].rad * cfg.gravity / steps;
            }
            marbles[i].pos += marbles[i].vel / steps;

            let samples = [SamplePoint {
                offset: Vec3::ZERO,
                radius: marbles[i].rad,
            }];
            let outcome = collide(obj, params, marbles[i].pos, &mut marbles[i].vel, &samples, cfg);
            if outcome.crushed {
                marbles[i].respawn(starts[i]);
                events[i] = StepEvent::RespawnedCrushed;
                continue;
            }
            marbles[i].pos = outcome.pos;
            on_ground[i] |= outcome.on_ground;
        }

        collide_marbles(marbles, cfg.bounce);
    }

    for i in 0..marbles.len() {
        if events[i] != StepEvent::None {
            continue;
        }

        // C++ normalizes (dx, dy) to unit magnitude when the combined input
        // exceeds 1 (e.g. two WASD keys held at once) so diagonal movement
        // isn't faster than axis-aligned movement.
        let mut dx = inputs[i].dx;
        let mut dy = inputs[i].dy;
        let mag2 = dx * dx + dy * dy;
        if mag2 > 1.0 {
            let mag = mag2.sqrt();
            dx /= mag;
            dy /= mag;
        }
        let (cam_forward, cam_right) = inputs[i].cam_basis();

        // Input force, once per full tick.
        let f = marbles[i].rad * if on_ground[i] { cfg.ground_force } else { cfg.air_force };
        let v = match cfg.mode {
            GravityMode::Rolling => {
                dy * horizontal(cam_forward, Vec3::NEG_Z) + dx * horizontal(cam_right, Vec3::X)
            }
            GravityMode::Flying => dy * FOCAL_DIST * cam_forward + dx * cam_right,
        };
        marbles[i].last_thrust = v;
        marbles[i].vel += v * f;

        // Friction, once per full tick.
        marbles[i].vel *= if on_ground[i] { cfg.ground_friction } else { cfg.air_friction };

        // Kill plane (Rolling only).
        if cfg.mode == GravityMode::Rolling && marbles[i].pos.y < kill_y {
            marbles[i].respawn(starts[i]);
            events[i] = StepEvent::RespawnedFell;
        }
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenes::{beware_of_bumps, demo_scene, set_fractal_params};
    use crate::{ScalarValue, Vec3Value};

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

    /// `orientation_from_basis` must exactly invert `PlayerInput::cam_basis`
    /// (round-trip: forward/right in, orientation out, forward/right back
    /// out must match the originals) -- this is what makes `step_marble`'s
    /// wrapper over `step_marbles` safe. Caught a real bug on the first
    /// attempt (a reflection instead of a rotation, wrong handedness) via
    /// `flying_mode_has_no_kill_plane` failing outright, not subtly -- this
    /// test exists so that class of mistake fails here directly instead of
    /// via a seemingly-unrelated test elsewhere.
    #[test]
    fn orientation_from_basis_roundtrips() {
        let cases = [
            (Vec3::NEG_Z, Vec3::X),
            (Vec3::NEG_Y, Vec3::X),
            (Vec3::Y, Vec3::X),
            (Vec3::X, Vec3::NEG_Z),
            (Vec3::new(0.267, -0.802, -0.535).normalize(), {
                // Any unit vector orthogonal to the forward above.
                let f = Vec3::new(0.267, -0.802, -0.535).normalize();
                Vec3::new(0.0, -f.z, f.y).normalize()
            }),
        ];
        for (forward, right) in cases {
            let orientation = orientation_from_basis(forward, right);
            assert!(
                (orientation.length() - 1.0).abs() < 1e-4,
                "expected a unit quaternion for forward={forward:?} right={right:?}, got {orientation:?}"
            );
            let got_forward = orientation * Vec3::NEG_Z;
            let got_right = orientation * Vec3::X;
            assert!(
                got_forward.distance(forward) < 1e-4,
                "forward round-trip failed: wanted {forward:?}, got {got_forward:?}"
            );
            assert!(
                got_right.distance(right) < 1e-4,
                "right round-trip failed: wanted {right:?}, got {got_right:?}"
            );
        }
    }

    /// Two overlapping marbles push apart to exactly touching (not
    /// overlapping, not flung apart past each other) and exchange velocity
    /// along the collision normal, conserving total momentum (equal-mass
    /// elastic response -- `collide_marbles`'s doc).
    #[test]
    fn collide_marbles_separates_overlapping_pair_and_conserves_momentum() {
        let mut marbles = [
            Marble {
                pos: Vec3::new(-0.05, 0.0, 0.0),
                vel: Vec3::new(1.0, 0.0, 0.0),
                rad: 0.1,
                last_thrust: Vec3::ZERO,
            },
            Marble {
                pos: Vec3::new(0.05, 0.0, 0.0),
                vel: Vec3::new(-1.0, 0.0, 0.0),
                rad: 0.1,
                last_thrust: Vec3::ZERO,
            },
        ];
        // Overlap: centers 0.1 apart, combined radius 0.2.
        let momentum_before: Vec3 = marbles.iter().map(|m| m.vel).sum();

        collide_marbles(&mut marbles, 1.0);

        let dist = marbles[0].pos.distance(marbles[1].pos);
        assert!(
            (dist - 0.2).abs() < 1e-4,
            "expected centers pushed apart to exactly touching (0.2), got {dist}"
        );
        // Head-on equal-mass elastic (bounce=1) collision: velocities swap.
        assert!(marbles[0].vel.distance(Vec3::new(-1.0, 0.0, 0.0)) < 1e-4);
        assert!(marbles[1].vel.distance(Vec3::new(1.0, 0.0, 0.0)) < 1e-4);
        let momentum_after: Vec3 = marbles.iter().map(|m| m.vel).sum();
        assert!(
            momentum_before.distance(momentum_after) < 1e-4,
            "momentum not conserved: {momentum_before:?} -> {momentum_after:?}"
        );
    }

    /// A pair already separating (moving apart) is left velocity-unchanged
    /// by the collision response -- only the position push-out applies.
    #[test]
    fn collide_marbles_does_not_add_velocity_to_a_separating_pair() {
        let mut marbles = [
            Marble {
                pos: Vec3::new(-0.05, 0.0, 0.0),
                vel: Vec3::new(-1.0, 0.0, 0.0),
                rad: 0.1,
                last_thrust: Vec3::ZERO,
            },
            Marble {
                pos: Vec3::new(0.05, 0.0, 0.0),
                vel: Vec3::new(1.0, 0.0, 0.0),
                rad: 0.1,
                last_thrust: Vec3::ZERO,
            },
        ];
        collide_marbles(&mut marbles, 1.0);
        assert!(marbles[0].vel.distance(Vec3::new(-1.0, 0.0, 0.0)) < 1e-6);
        assert!(marbles[1].vel.distance(Vec3::new(1.0, 0.0, 0.0)) < 1e-6);
    }

    /// Non-overlapping marbles are left completely untouched.
    #[test]
    fn collide_marbles_leaves_non_overlapping_pair_untouched() {
        let mut marbles = [
            Marble::spawn(Vec3::ZERO, 0.1),
            Marble::spawn(Vec3::new(10.0, 0.0, 0.0), 0.1),
        ];
        marbles[0].vel = Vec3::new(1.0, 2.0, 3.0);
        let before = marbles;
        collide_marbles(&mut marbles, 1.0);
        assert_eq!(marbles[0].pos, before[0].pos);
        assert_eq!(marbles[1].pos, before[1].pos);
        assert_eq!(marbles[0].vel, before[0].vel);
        assert_eq!(marbles[1].vel, before[1].vel);
    }

    fn flying_input(dx: f32, dy: f32, forward: Vec3, right: Vec3) -> PlayerInput {
        PlayerInput {
            dx,
            dy,
            orientation: orientation_from_basis(forward, right),
        }
    }

    /// `step_marbles` with a single marble and a single-element everything
    /// must behave identically to `step_marble` -- exercised directly rather
    /// than just trusting `step_marble`'s own wrapper tests, since it's the
    /// wrapper's *inverse* correspondence (does the new N-marble engine
    /// still do the right thing for N=1, not just "does the old signature
    /// still compile").
    #[test]
    fn step_marbles_matches_step_marble_for_a_single_marble() {
        let (object, params) = setup_demo();
        let cfg = PhysicsConfig {
            mode: GravityMode::Rolling,
            ..PhysicsConfig::default()
        };
        let rad = beware_of_bumps::MARBLE_RAD;
        let start = beware_of_bumps::START;
        let drop_from = start + Vec3::new(0.0, 0.5, 0.0);

        let mut via_step_marble = Marble::spawn(drop_from, rad);
        let mut via_step_marbles = [Marble::spawn(drop_from, rad)];
        let input = flying_input(0.3, -0.4, Vec3::NEG_Z, Vec3::X);

        for _ in 0..300 {
            step_marble(
                &mut via_step_marble,
                &object,
                &params,
                Vec2::new(input.dx, input.dy),
                Vec3::NEG_Z,
                Vec3::X,
                &cfg,
                beware_of_bumps::KILL_Y,
                start,
            );
            step_marbles(
                &mut via_step_marbles,
                &[input],
                &object,
                &params,
                &cfg,
                beware_of_bumps::KILL_Y,
                &[start],
            );
        }

        assert!(
            via_step_marble.pos.distance(via_step_marbles[0].pos) < 1e-4,
            "step_marble and step_marbles(N=1) diverged: {:?} vs {:?}",
            via_step_marble.pos,
            via_step_marbles[0].pos
        );
    }

    /// N marbles dropped at well-separated points above a large flat-topped
    /// box each individually fall and settle onto it, mirroring
    /// `marble_falls_and_settles` but for a multi-marble slice -- confirms
    /// `step_marbles` didn't break marble-vs-fractal collision for any
    /// individual marble while adding marble-vs-marble collision. A big
    /// cuboid (not the demo level, and not a sphere) on purpose:
    /// `beware_of_bumps::START` sits on a resting ledge only about 1x the
    /// marble's own radius wide (`render.rs`'s `animate_fractal` doc), so
    /// offsetting away from it by enough to avoid marble-vs-marble contact
    /// reliably walks off the ledge into open space (confirmed by an
    /// earlier version of this test doing exactly that); a *sphere*'s
    /// surface curves away from any offset point, so straight-down gravity
    /// has a tangential component there and the marble never truly stops
    /// rolling (also confirmed by an earlier version of this test: settled
    /// to a small but persistently nonzero terminal creep speed, not a
    /// `step_marbles` bug either -- just not a flat resting surface). A
    /// large cuboid's flat top face gives every offset within it a genuine,
    /// unsloped resting spot, isolating what this test actually wants to
    /// check.
    #[test]
    fn step_marbles_each_marble_falls_and_settles_independently() {
        let params = Params::new();
        let half_extent = 20.0;
        let object = Object::Cuboid {
            half_extent: Vec3Value::Const(glam::Vec3::splat(half_extent)),
        };
        let cfg = PhysicsConfig {
            mode: GravityMode::Rolling,
            ..PhysicsConfig::default()
        };
        let rad = 0.1;
        // Well separated in X/Z (>> 2*rad apart) so they don't collide with
        // each other, purely testing independent marble-vs-fractal settling
        // here (marble-vs-marble collision is covered by the tests above and
        // the anti-tunneling test below); all comfortably within the top
        // face's flat interior, away from its edges/corners.
        let starts = [
            Vec3::new(0.0, half_extent + 0.5, 0.0),
            Vec3::new(2.0, half_extent + 0.5, 0.0),
            Vec3::new(0.0, half_extent + 0.5, 2.0),
        ];
        let mut marbles: Vec<Marble> = starts.iter().map(|s| Marble::spawn(*s, rad)).collect();
        let inputs = vec![flying_input(0.0, 0.0, Vec3::NEG_Z, Vec3::X); starts.len()];
        let kill_y = -1000.0;

        for _ in 0..5_000 {
            step_marbles(&mut marbles, &inputs, &object, &params, &cfg, kill_y, &starts);
        }

        for (i, m) in marbles.iter().enumerate() {
            let de = object.de(m.pos.extend(1.0), &params);
            assert!(
                m.vel.length() < 0.05 * rad,
                "marble {i} did not settle: |vel|={}",
                m.vel.length()
            );
            assert!(
                de >= rad * 0.5 && de <= rad * cfg.ground_ratio * 1.5,
                "marble {i} not resting on a surface: de={de}, rad={rad}"
            );
        }
    }

    /// Two marbles on a direct collision course actually collide and
    /// separate -- not pass through each other -- across
    /// [`NUM_PHYS_STEPS`]-substepped ticks, the direct anti-tunneling
    /// regression test `collide_marbles`'s doc describes.
    #[test]
    fn step_marbles_fast_marbles_do_not_tunnel_through_each_other() {
        let params = Params::new();
        // Far from any geometry -- isolates marble-vs-marble collision from
        // marble-vs-fractal.
        let object = Object::Sphere {
            radius: ScalarValue::Const(0.001),
        };
        let rad = 0.1;
        let cfg = PhysicsConfig {
            mode: GravityMode::Flying, // no gravity: isolate the collision itself
            bounce: 1.0,
            ..PhysicsConfig::default()
        };
        let start_a = Vec3::new(-2.0, 100.0, 100.0);
        let start_b = Vec3::new(2.0, 100.0, 100.0);
        let mut marbles = [Marble::spawn(start_a, rad), Marble::spawn(start_b, rad)];
        // `vel = 0.6` is a per-*tick* displacement (`pos += vel / steps` each
        // of `NUM_PHYS_STEPS` substeps sums back to `vel` over a full tick) --
        // 3x the combined touching radius (0.2), so a single *unsubstepped*
        // per-tick jump starting anywhere within striking range would cross
        // clean past the other marble without ever registering as
        // "touching". Per-*substep* displacement (`vel / 6 = 0.1`) is still
        // comfortably under the combined radius, so the discrete
        // per-substep overlap check `collide_marbles` does six times a tick
        // reliably catches what one check per tick would have missed. (A
        // marble fast enough to cross the entire combined radius within a
        // single *substep* would still tunnel -- the same limitation
        // `collide`'s marble-vs-fractal check already has, not something
        // `collide_marbles` claims to fix beyond.)
        marbles[0].vel = Vec3::new(0.6, 0.0, 0.0);
        marbles[1].vel = Vec3::new(-0.6, 0.0, 0.0);
        let inputs = [
            flying_input(0.0, 0.0, Vec3::NEG_Z, Vec3::X),
            flying_input(0.0, 0.0, Vec3::NEG_Z, Vec3::X),
        ];
        let starts = [start_a, start_b];

        let mut crossed = false;
        for _ in 0..30 {
            step_marbles(&mut marbles, &inputs, &object, &params, &cfg, -1e6, &starts);
            if marbles[0].pos.x > marbles[1].pos.x {
                crossed = true;
                break;
            }
        }
        assert!(
            !crossed,
            "marbles tunneled through each other instead of colliding: {:?}",
            marbles.iter().map(|m| m.pos).collect::<Vec<_>>()
        );
        // And they must have actually visibly interacted (bounced back),
        // not just frozen in place -- confirms the collision fired at all.
        assert!(
            marbles[0].vel.x < 0.0,
            "marble 0 should have bounced back (negative x velocity), got {}",
            marbles[0].vel.x
        );
        assert!(
            marbles[1].vel.x > 0.0,
            "marble 1 should have bounced back (positive x velocity), got {}",
            marbles[1].vel.x
        );
    }
}
