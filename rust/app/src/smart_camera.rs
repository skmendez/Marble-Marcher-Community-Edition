//! The smart camera: what actually renders, as opposed to what the player
//! asked for. Design and research notes in `rust/CAMERA.md`; this module is
//! §4 of that document.
//!
//! # The split
//!
//! [`crate::camera::CameraOrbit`] is the player's **intent** — a free 3D
//! arcball orientation plus a zoom multiplier, written 1:1 by every input
//! path with no smoothing whatsoever. [`CameraRig`] is the **realized**
//! camera: the same thing, except that it (a) frames the marble at a
//! sensible on-screen size instead of at a hand-tuned world distance, (b)
//! refuses to let geometry get between itself and the marble, and (c) moves
//! with damping instead of teleporting. The two are kept in lockstep by
//! every input event (`camera::apply_drag`/`apply_roll` rotate both by the
//! same rotation) and diverge only through this module's own corrections,
//! which decay back toward intent as soon as the view is clear again.
//!
//! That split is what makes "respect the player" mechanical rather than a
//! matter of tuning: there is no code path in which a correction competes
//! with an input, because inputs move both states and corrections move only
//! one.
//!
//! # Two invariants, not two behaviors
//!
//! The realized camera is stored as an orientation and a *distance*, and
//! the eye is always derived as `focus - forward * distance`. So:
//!
//! * **I1**: the camera is always looking at (a smoothed version of) the
//!   marble. It cannot drift off-target, because no state exists in which
//!   it points anywhere else.
//! * **I2**: `distance <= free_distance` along that direction
//!   ([`marble_csg::visibility::sweep`]), so the eye is always inside the
//!   region of space that is *visible from the marble*. It cannot be inside
//!   geometry, and it cannot be on the far side of a wall — not because
//!   something pushes it out afterwards, but because no other state is
//!   representable.
//!
//! A physically-simulated "drone" camera (a collider springing toward an
//! ideal pose) was the obvious alternative and is rejected for exactly this
//! reason: it can end up behind a wall and stay there, and then no amount of
//! damping helps. Here, being blocked is not a state the camera can be *in*;
//! it is a force that moves it.
//!
//! # What runs each frame
//!
//! A sphere trace from the marble outward along the view ray
//! ([`marble_csg::visibility::sweep`]) answers the solver's three questions
//! at once — how far back the eye can sit, how much of the marble is
//! visible from there, and what is in the way. Around that:
//!
//! 1. the focus point springs toward the marble (with a hard lag clamp, so
//!    smoothing can never cost more than a fixed fraction of the frame),
//! 2. the view direction slides tangentially away from whatever is blocking
//!    it, at a rate proportional to how blocked it is, and decays back
//!    toward the player's intent when clear,
//! 3. the ray is re-swept wherever that left it, because the step that
//!    follows is the one that actually places the eye and must not run on a
//!    frame-old idea of where the walls are,
//! 4. if that direction has nowhere for the camera to be at all — or has
//!    been hopeless for long enough, or is merely far tighter than the
//!    framing rule wants — a small ring of alternatives is searched and the
//!    best is committed to,
//! 5. the distance is pulled in fast and pushed back out slowly,
//! 6. and the field of view widens (only) when geometry has forced the
//!    camera closer than framing wanted, which keeps the marble's on-screen
//!    size near target even in a tunnel.
//!
//! Steps 2, 4 and 5 each have a hold or commitment attached, and every one
//! of them is there because its absence produced a specific measured
//! failure (thrash between candidates, a reposition immediately undone, a
//! camera pumping in and out on a flickering strut) — see the constants.

use bevy::prelude::*;

use marble_csg::visibility::{sweep, Sdf, SweepConfig};

use crate::camera::{CameraOrbit, FOCAL_LENGTH, MAX_DISTANCE, MIN_DISTANCE_MARBLE_RADII};

/// Target on-screen marble size for pointer input, as a fraction of the
/// *shorter* screen dimension (height in landscape, width in portrait).
///
/// Taking the shorter dimension as the reference is what lets one rule
/// cover both a landscape desktop window and a portrait phone: "1/6 of the
/// height" on a 16:9 monitor and "about 1/4 of the width" on a phone are the
/// same instruction once expressed this way. Cross-checked against the
/// distances that were previously hand-tuned by screenshot per scene: this
/// value reproduces the demo scene's `0.2` to within ~10% and the Menger
/// scenes' `1.2` to within ~12% (`rust/CAMERA.md` §4.2).
pub const POINTER_TARGET_FRACTION: f32 = 1.0 / 6.0;

/// Target on-screen marble size for touch input — bigger, because the
/// physical screen is smaller and a finger covers a chunk of it. Between
/// the "1/4 to 1/3 of screen width" the design brief asked for.
pub const TOUCH_TARGET_FRACTION: f32 = 0.28;

/// Widest the field of view may open when geometry forces the camera closer
/// than the framing rule wants (§4.8): `1.0` is a 90° vertical FOV, against
/// the default [`FOCAL_LENGTH`]'s 67°. Only ever *widened* from the default,
/// and slowly ([`FOCAL_TAU`]) — FOV pumping is a well-documented nausea
/// trigger, and this is the one lever here that changes the projection
/// itself rather than where the camera is.
const MIN_FOCAL_LENGTH: f32 = 1.0;

/// Focus-point smoothing time. Short on purpose: the marble *is* the
/// gameplay, and a camera that lags it reads as input latency. This exists
/// to take the edge off collision chatter, not to add cinematic drift.
const FOCUS_TAU: f32 = 0.10;

/// Hard cap on how far the smoothed focus may trail the real marble, as a
/// fraction of the current camera distance. Applies across the frame only --
/// the depth component of the trail is removed outright (see step 1 of
/// [`solve`]) -- so this bounds how far off centre a fast-moving marble can
/// drift, and nothing else. A pure spring has no bound at all: at high
/// marble speed it degenerates into the marble sitting near the edge of
/// frame, which is exactly the framing failure this camera exists to
/// prevent. `0.25` keeps it comfortably inside the middle of the picture no
/// matter how fast it is travelling.
const MAX_FOCUS_LAG_FRACTION: f32 = 0.25;

/// Distance-spring smoothing when *shortening* (something is in the way).
/// Near-immediate: lag here means the obstruction is visibly in front of
/// the marble, or the eye is inside it.
const PULL_IN_TAU: f32 = 0.05;
/// Distance-spring smoothing when *lengthening* (the way is clear again).
/// Deliberately ~7x slower than pulling in — the asymmetry is what stops a
/// picket fence of Menger struts from pumping the camera in and out.
const PUSH_OUT_TAU: f32 = 0.35;
/// ...and it may not even start until the camera has had room to grow into
/// for this long (Cinemachine calls this the deoccluder's "smoothing time":
/// hold at the near point briefly so a picket fence doesn't pump the
/// camera).
///
/// The timer keys off *room* (the distance goal being above where the camera
/// currently is), not off the view being fully clear. Keying it off
/// visibility -- which is what this did first -- deadlocks in a tight,
/// busy space: a view that dips to 0.95 visible for one frame every so often
/// resets the timer forever, so the camera pulls in once and then never
/// pushes back out, however much room opens up. That is exactly what
/// HollowDonut's probe caught: a camera pinned 0.25 from a marble that had
/// 0.6-1.3 of clearance available the whole time.
const PUSH_OUT_HOLD: f32 = 0.4;

/// Peak rate (rad/s) at which the view direction slides around an
/// obstruction, reached at zero visibility and scaled down continuously by
/// how visible the marble actually is. ~90°/s: fast enough to clear a
/// pillar in well under a second, slow enough not to feel like the camera
/// wrestled control away.
const SLIDE_RATE: f32 = 1.6;
/// The rate the slide escalates to once the marble has been *completely*
/// hidden for [`PANIC_AFTER`] — sliding that hasn't found an opening by
/// then is up against something concave (a dead-end pocket, an inside
/// corner) that needs a bigger angular move to escape.
const SLIDE_PANIC_RATE: f32 = 3.4;
const PANIC_AFTER: f32 = 0.35;

/// Rate (rad/s, ~150°/s) used to travel to a direction the search has
/// committed to, as opposed to the gentle slide that eases around an
/// obstruction in place.
///
/// Faster than [`SLIDE_RATE`] for a concrete reason found in the HollowDonut
/// probe rather than by taste: the thing the camera is repositioning
/// *around* moves too. A marble circling the inside of that tube sweeps the
/// usable directions around with it at close to 90°/s -- the same speed the
/// gentle slide travels -- so a chase at that rate never converges and the
/// camera stays pinned close to the marble indefinitely. A repositioning
/// move has to outrun the thing it is chasing to ever arrive.
const REPOSITION_RATE: f32 = 2.6;

/// How long the view must stay clear before the camera starts drifting back
/// toward the player's intent, and how fast it does so. Slower than the
/// slide: getting *out* of the way is urgent, going back is not.
const RECOVER_HOLD: f32 = 0.25;
const RECOVER_TAU: f32 = 0.6;

/// Hard cap on how far the realized camera may deviate from intent
/// (~110°). Without it, a marble in a long dead-end corridor can walk the
/// camera arbitrarily far from where the player pointed it. With it, the
/// worst case is a bounded disagreement that resolves the moment the marble
/// is back in the open.
const MAX_CORRECTION: f32 = 1.9;

/// FOV-widening smoothing time (§4.8's rate limit).
const FOCAL_TAU: f32 = 0.5;

/// The camera's own collision radius, in marble radii — Cinemachine's
/// "camera radius", and also what keeps the eye out of the near-surface band
/// where this renderer's normal estimation degenerates into speckle (see the
/// per-scene camera-distance comments this replaced in `render::setup`).
///
/// Scaled to the *marble*, which is the game's unit of length, and
/// emphatically not to the framing distance, which is what it used to be
/// (`0.08 * desired`). That coupling is wrong in principle -- the camera's
/// physical size has nothing to do with how far back it happens to want to
/// sit -- and catastrophic in practice: at `cube_sphere_morph`'s `zoom =
/// 3.3` it made the probe ball `0.645`, over four marble radii, so the very
/// first sample of every march (taken on the marble's own surface) reported
/// blocked whenever the marble was near anything at all. Reported from play
/// as the camera diving at the marble on approach to any surface; captured
/// on device as `vis 0.00 d 0.225/4.816 (free 0.000) ... steps 1`.
const CAMERA_RADIUS_MARBLE_RADII: f32 = 0.35;

/// How much of the framing distance the camera wants to keep behind it
/// before it starts resisting being *rotated* into geometry, and how far it
/// looks ahead when deciding which way is "into".
///
/// The player orbiting the camera into a wall is a different problem from
/// something moving into the shot, and it wants a different answer. Pulling
/// in is the right response to an occluder crossing the sightline; it is the
/// wrong response to the player swinging the camera behind a pillar, where
/// it reads as the camera suddenly diving at the marble for no reason the
/// player can see (reported from play: rotating slightly near a large
/// structure took the distance from 1.411 to 0.279 in one motion, with the
/// marble fully visible the whole time).
///
/// Every third-person camera writeup that treats this case puts *rotation*
/// first in the resolution order and dollying second -- see `rust/CAMERA.md`
/// §12. So near geometry, the orbit becomes a constrained one: the component
/// of the player's rotation that would drive the camera into a surface is
/// removed, and what remains slides it along the surface instead. The dolly
/// stays as the safety net it always was, but it stops being the *first*
/// answer, so it rarely has anywhere dramatic to go.
const WALL_COMFORT_FRACTION: f32 = 0.85;
/// ...and the fraction below which the into-the-surface component of a
/// rotation is removed *entirely*, so the camera simply will not be orbited
/// any further into a wall. Between the two the removal ramps, which is what
/// keeps sliding smooth rather than making the controls snag at a threshold.
///
/// Together these are the "stay some distance away" the reported behavior
/// was missing: rotation alone can never cost the camera more than 40% of
/// its framing distance. Geometry that genuinely gets between the camera and
/// the marble still can, via the dolly, because that is a real occlusion and
/// the camera must answer it.
const WALL_FLOOR_FRACTION: f32 = 0.6;
/// Finite-difference angle for estimating which way clearance improves
/// (radians, ~3 degrees). Big enough to read past `de` noise on fractal
/// surfaces, small enough to still be a local gradient.
const WALL_GRADIENT_EPS: f32 = 0.05;
/// ...and the smallest clearance change over that angle, as a fraction of
/// the framing distance, that counts as a real direction to slide along.
///
/// Load-bearing rather than hygiene. When the camera is pressed straight at
/// a wall the gradient is a *minimum*: every direction improves, by almost
/// nothing, and normalising that near-zero vector produces a confident but
/// meaningless "into the wall" direction -- which then blocks the player's
/// rotation in every direction at once, exactly where they most need to get
/// out. Below this threshold there is no wall direction to speak of, so
/// nothing is constrained.
const WALL_GRADIENT_MIN_FRACTION: f32 = 0.02;

/// How long after the player stops steering before *elective* repositioning
/// (the cramped search) is allowed to run again. Safety repositioning -- no
/// camera position on this ray at all, or a sustained total block -- is
/// never gated: it does not compete with the player, it rescues them.
const ELECTIVE_INPUT_IDLE: f32 = 0.5;

/// A view is "cramped" when geometry allows less than this fraction of the
/// distance framing wants. Being cramped is not an emergency -- the marble
/// is perfectly visible, just too big in frame -- but it is worth *looking*
/// for a roomier direction, because the alternative is a camera that settles
/// half a metre from the marble inside a tunnel and stays there: nothing
/// else in the solver pushes back out, since an unobstructed close-up looks
/// entirely fine to the occlusion logic. This is Cinemachine's "shot
/// quality" idea (distance-from-optimal counts alongside obstruction), and
/// in this game it is the difference between HollowDonut's camera riding the
/// tube wall and it finding the open middle.
const CRAMPED_FRACTION: f32 = 0.5;
/// ...but only after being cramped this long, so a moment of tight quarters
/// while rolling past a strut doesn't start a search.
const CRAMPED_HOLD: f32 = 0.4;

/// How long the decay back toward the player's intent stays suppressed after
/// a search commitment finishes.
///
/// Without it the solver and the decay form a limit cycle in any scene where
/// the player's intended direction is a cramped one: search picks a roomier
/// direction, the camera arrives, the view is no longer cramped, so the
/// decay immediately drags it back to the cramped intent, which triggers the
/// search again. Measured in HollowDonut's tube as the camera spending half
/// the run pinned close to the marble despite a clear 1.4-unit shot being
/// continuously available. A reposition the solver went to the trouble of
/// making is worth keeping for a few seconds.
///
/// This never blocks the player: their input writes the realized camera
/// directly (`camera::apply_drag`), so a drag moves the view instantly
/// whether or not a lockout is running.
const RECOVER_LOCKOUT: f32 = 2.5;

/// How long the solver commits to a direction the search picked before it is
/// willing to reconsider (Cinemachine's deoccluder has the same idea as its
/// "smoothing time"). See the commitment block in [`solve`] for why this is
/// load-bearing rather than an optimisation.
const SEARCH_COMMIT: f32 = 0.4;

/// Candidate directions tried by the emergency search ([`search_direction`])
/// when the current view ray has no usable camera position at all: two rings
/// (40° and 80° off the current direction) of four, spaced around the
/// camera's own right/up axes. Frame-free by construction -- there is no
/// world "up" in this game to build a candidate set around (`rust/CAMERA.md`
/// §2) -- and small enough that the search stays affordable even in the one
/// situation that triggers it every frame (a marble deep inside a tube).
const SEARCH_ANGLES: [f32; 2] = [0.7, 1.4];

/// Step budget for the per-frame sightline march. 24 steps costs ~16 µs on
/// the most expensive scene (`csg/examples/de_bench.rs`); real marches
/// terminate in a handful of steps.
const SWEEP_MAX_STEPS: u32 = 24;

/// Largest simulation step the solver will take in one go. A wasm hitch (a
/// GC pause, a pipeline compile) otherwise arrives as one enormous `dt` and
/// teleports the camera — every spring here is frame-rate independent, but
/// none of them is *hitch* independent.
const MAX_DT: f32 = 1.0 / 20.0;

/// The realized camera — what renders, and what the marble's control frame
/// is derived from. See the module doc for how this relates to
/// [`CameraOrbit`] (the player's intent) and for the two invariants it
/// maintains.
#[derive(Resource, Clone, Copy, Debug)]
pub struct CameraRig {
    /// Realized orientation. `forward = orientation * NEG_Z` points at
    /// [`Self::focus`] by construction (invariant I1).
    pub orientation: Quat,
    /// Realized distance from [`Self::focus`] to the eye.
    pub distance: f32,
    /// Smoothed marble position — what the camera actually looks at.
    pub focus: Vec3,
    /// Realized focal length (`1/tan(halfFOV)`), fed to
    /// `SceneUniforms::cam_forward.w`. Equals [`FOCAL_LENGTH`] except while
    /// the FOV is widened to compensate for a forced-close camera.
    pub focal_length: f32,
    focus_vel: Vec3,
    distance_vel: f32,
    blocked_for: f32,
    clear_for: f32,
    cramped_for: f32,
    /// Seconds since the distance goal last *tightened*. See
    /// [`PUSH_OUT_HOLD`].
    room_for: f32,
    /// Previous frame's distance goal, so `room_for` can tell "the
    /// constraint just moved in" from "the camera is simply already at the
    /// goal".
    last_goal: f32,
    /// The direction [`search_direction`] most recently committed to, the
    /// free distance it promised, and how long the commitment still has to
    /// run. See the commitment block in [`solve`].
    search_dir: Option<Vec3>,
    search_free: f32,
    search_hold: f32,
    /// Time left on the post-reposition hold. See [`RECOVER_LOCKOUT`].
    recover_lockout: f32,
    /// The intent as of the previous solve, so this one can recover the
    /// player's rotation *as a delta* and apply it under the wall-slide
    /// constraint (see [`WALL_COMFORT_FRACTION`]) instead of the input
    /// systems applying it to the realized camera directly.
    last_intent: Quat,
    /// Seconds since the player last rotated the camera. Gates elective
    /// repositioning only ([`ELECTIVE_INPUT_IDLE`]).
    input_idle_for: f32,
    /// Cleared on construction, set the first time [`solve`] runs: the very
    /// first frame snaps rather than springs (there is no previous state to
    /// spring *from*, and starting at distance 0 would put the eye inside
    /// the marble for the first several frames).
    initialized: bool,
    /// Last solve's diagnostics, for the `?debug=1` overlay and the probe
    /// harness. Never read by the solver itself.
    pub debug: RigDebug,
}

/// Read-only diagnostics from the last [`solve`] — everything needed to
/// answer "why is the camera where it is?" without a debugger.
#[derive(Clone, Copy, Debug, Default)]
pub struct RigDebug {
    /// [`marble_csg::visibility::Sweep::visibility`]: 1 = clear shot.
    pub visibility: f32,
    /// How far back the camera *could* sit along the current view ray.
    pub free_distance: f32,
    /// How far back it *wants* to sit (framing rule × zoom).
    pub desired_distance: f32,
    /// The marble's actual on-screen size, as a fraction of the shorter
    /// screen dimension — directly comparable with the target fraction.
    pub screen_fraction: f32,
    /// Angle between the realized camera and the player's intent (radians).
    pub deviation: f32,
    /// Distance field value at the eye: how much clearance the camera
    /// actually has. Negative would mean "inside geometry" (I2 violated).
    pub eye_clearance: f32,
    /// The probe ball's radius this frame ([`CAMERA_RADIUS_MARBLE_RADII`]) --
    /// the number to compare `eye_clearance` against, and the one whose
    /// mis-scaling caused the camera to dive at the marble near any surface.
    /// Shown in the `?debug=1` overlay next to the clearance for that
    /// reason, and because it doubles as a quick check of *which* build is
    /// running: a bundle without the fix does not print it.
    pub camera_radius: f32,
    /// `de` evaluations spent on this frame's sightline march.
    pub steps: u32,
}

impl Default for CameraRig {
    fn default() -> Self {
        Self {
            orientation: Quat::IDENTITY,
            distance: 1.0,
            focus: Vec3::ZERO,
            focal_length: FOCAL_LENGTH,
            focus_vel: Vec3::ZERO,
            distance_vel: 0.0,
            blocked_for: 0.0,
            clear_for: 0.0,
            cramped_for: 0.0,
            room_for: 0.0,
            last_goal: 0.0,
            search_dir: None,
            search_free: 0.0,
            search_hold: 0.0,
            recover_lockout: 0.0,
            last_intent: Quat::IDENTITY,
            input_idle_for: 0.0,
            initialized: false,
            debug: RigDebug::default(),
        }
    }
}

impl CameraRig {
    /// Eye position and orthonormal basis for rendering (`render.rs`) and
    /// for screen-space projection (`debug_gizmos.rs`).
    pub fn eye_and_basis(&self) -> (Vec3, Vec3, Vec3, Vec3) {
        CameraOrbit::basis_from(self.orientation, self.focus, self.distance)
    }

    pub fn eye(&self) -> Vec3 {
        self.focus - (self.orientation * Vec3::NEG_Z) * self.distance
    }

    /// Forces the next [`solve`] to snap rather than spring — for a scene
    /// switch or a respawn, where springing would send the camera flying
    /// across the level.
    pub fn reset(&mut self) {
        self.initialized = false;
    }
}

/// Everything [`solve`] needs from the rest of the app, gathered into one
/// plain-data struct so the solver itself is a pure function of (state,
/// input, geometry) and can be unit-tested against analytic worlds with no
/// Bevy `App`, no scene tree, and no window.
#[derive(Clone, Copy, Debug)]
pub struct SolveInput {
    pub marble_pos: Vec3,
    pub marble_radius: f32,
    /// The player's intended orientation ([`CameraOrbit::orientation`]).
    ///
    /// In/out: [`solve`] can push it, but only when the realized camera has
    /// been forced [`MAX_CORRECTION`] away from it and still needs to go
    /// further (see the cap in `solve`). Nothing else ever writes it -- the
    /// input systems own it.
    pub intent: Quat,
    /// The player's zoom multiplier ([`CameraOrbit::zoom`]).
    pub zoom: f32,
    /// Window width / height.
    pub aspect: f32,
    /// [`POINTER_TARGET_FRACTION`] or [`TOUCH_TARGET_FRACTION`].
    pub target_fraction: f32,
    pub dt: f32,
    /// `false` (the default until this has been play-tested; `?smartcam=1`
    /// turns it on) keeps the framing rule but disables every
    /// geometry-aware and time-based behavior: the rig then tracks intent
    /// exactly, which is the pre-smart-camera behavior and the A/B baseline.
    pub smart: bool,
}

/// Distance at which a sphere of radius `radius` covers `fraction` of the
/// shorter screen dimension, at focal length `focal` and the given aspect.
///
/// From the projection the shader uses (`marble_csg::codegen`'s ray setup:
/// `ndc_y = y_cam·f/z_cam`, `ndc_x = x_cam·f/(z_cam·aspect)`, NDC spanning
/// `[-1, 1]` across the window), a sphere of radius `r` at distance `d`
/// spans `f·r/sqrt(d² − r²)` of the screen's *height*, and `1/aspect` times
/// that of its width. Referencing the shorter dimension (hence
/// `min(1, aspect)`) and solving for `d`:
///
/// ```text
/// d = sqrt(r² + (f·r / (fraction · min(1, aspect)))²)
/// ```
///
/// The `r²` term is the exact-silhouette correction; it only matters when
/// the marble fills much of the frame, which is precisely the tight-space
/// case where being exact is worth the one extra `sqrt`.
pub fn framing_distance(radius: f32, focal: f32, fraction: f32, aspect: f32) -> f32 {
    let fraction = fraction.max(1e-3);
    let short_side = aspect.clamp(1e-3, 1.0);
    let tangential = focal * radius / (fraction * short_side);
    (radius * radius + tangential * tangential).sqrt()
}

/// Inverse of [`framing_distance`]: what fraction of the shorter screen
/// dimension the marble actually covers right now. Diagnostics only.
pub fn screen_fraction(radius: f32, focal: f32, distance: f32, aspect: f32) -> f32 {
    let short_side = aspect.clamp(1e-3, 1.0);
    let denom = (distance * distance - radius * radius).max(1e-9).sqrt();
    focal * radius / (denom * short_side)
}

/// Exact critically damped spring step (ζ = 1: settles as fast as possible
/// without overshoot), frame-rate independent by construction — the closed
/// form of `x(t) = (A + Bt)·e^{-ωt}` evaluated at `dt`, not an
/// `exp`-corrected lerp. `tau` is the time constant; `ω = 2/tau`.
fn spring(x: &mut f32, v: &mut f32, target: f32, tau: f32, dt: f32) {
    if dt <= 0.0 {
        return;
    }
    let omega = 2.0 / tau.max(1e-4);
    let dx = *x - target;
    let b = *v + omega * dx;
    let e = (-omega * dt).exp();
    *x = target + (dx + b * dt) * e;
    *v = (*v - omega * b * dt) * e;
}

fn spring_vec3(x: &mut Vec3, v: &mut Vec3, target: Vec3, tau: f32, dt: f32) {
    let (mut xa, mut va) = (*x, *v);
    for i in 0..3 {
        let (mut xi, mut vi) = (xa[i], va[i]);
        spring(&mut xi, &mut vi, target[i], tau, dt);
        xa[i] = xi;
        va[i] = vi;
    }
    *x = xa;
    *v = va;
}

/// Frame-rate-independent exponential approach factor for a time constant
/// `tau` — `1 - e^{-dt/tau}`, the thing a bare `lerp(a, b, 0.1)` should
/// always have been.
fn approach(tau: f32, dt: f32) -> f32 {
    1.0 - (-dt / tau.max(1e-4)).exp()
}

/// The rotation axis that slides the view around `blocker`: perpendicular
/// to the sightline (so it never introduces twist, the same construction
/// `CameraOrbit::drag` relies on) and oriented so that rotating *positively*
/// about it moves the eye toward the blocker's free side.
///
/// Rotating `u` about `axis = u × n_t` by `+θ` moves `u` toward `n_t` — by
/// the triple-product identity `(u × n_t) × u = n_t` for orthonormal
/// `u`, `n_t`. So with `n_t` the surface's outward direction projected
/// perpendicular to the sightline, the camera peeks around the obstruction
/// rather than into it.
fn slide_axis(sdf: &impl Sdf, blocker: Vec3, u: Vec3, orientation: Quat, eps: f32) -> Option<Vec3> {
    let outward = sdf.outward(blocker, eps);
    let mut tangent = outward - u * outward.dot(u);
    if tangent.length_squared() < 1e-10 {
        // Head-on: the obstruction's normal is along the sightline, so it
        // offers no direction to slide toward (a flat wall squarely between
        // camera and marble). Any perpendicular will do to break the tie;
        // using the camera's own screen-right keeps the resulting motion
        // horizontal on screen, which reads as a deliberate pan rather than
        // an arbitrary tumble. Falling back again to screen-up covers the
        // degenerate case where right happens to be parallel to `u`, which
        // it can't be for a well-formed basis but is cheap to guard.
        let right = orientation * Vec3::X;
        tangent = right - u * right.dot(u);
        if tangent.length_squared() < 1e-10 {
            let up = orientation * Vec3::Y;
            tangent = up - u * up.dot(u);
        }
    }
    let tangent = tangent.normalize_or_zero();
    if tangent == Vec3::ZERO {
        return None;
    }
    let axis = u.cross(tangent);
    if axis.length_squared() < 1e-10 {
        return None;
    }
    Some(axis.normalize())
}

/// Best alternative direction when the current one is unusable — the
/// "whisker" search of `rust/CAMERA.md` §4.5, reached only in an emergency.
///
/// The tangential slide handles the ordinary case (something has moved into
/// the shot; ease around it). What it cannot handle is a direction with *no*
/// viable camera position at all — where even the closest allowed distance
/// would put the eye inside a wall — because sliding is a slow, continuous
/// motion and that situation needs an answer this frame. That happens in
/// genuinely tight interiors: HollowDonut's tube (free radius `0.85`) and
/// the Menger scenes' recursive tunnels, where the free space around the
/// marble is barely larger than the camera wants to be.
///
/// Returns the best candidate direction and the free distance along it, or
/// `None` if nothing meaningfully beats what the caller already has.
///
/// Candidates are scored on room-for-the-camera *and* visibility
/// (`min(free, desired)/desired + visibility`, both already produced by the
/// one sweep each candidate costs) rather than on free distance alone: in a
/// tube, the roomiest direction and the one that can actually see the marble
/// are usually the same, but where they differ, a camera that can sit far
/// back and see nothing is not a rescue. Deliberately *not* scored on
/// agreement with the player's intent — that preference is expressed by the
/// decay back toward intent (which resumes the moment the emergency ends),
/// and letting it compete here would mean declining a viable shot in favour
/// of an unviable one the player happens to have asked for.
#[allow(clippy::too_many_arguments)]
fn search_direction(
    sdf: &impl Sdf,
    focus: Vec3,
    u: Vec3,
    orientation: Quat,
    probe_dist: f32,
    desired: f32,
    cfg: SweepConfig,
    current_score: f32,
) -> Option<(Vec3, f32)> {
    let right = orientation * Vec3::X;
    let up = orientation * Vec3::Y;
    // Room is credited past the framing distance, up to twice it, rather
    // than saturating there. Two directions that both "have enough room"
    // are not equally good: the roomier one is usually the one whose
    // usable-ness survives the marble moving, which is what a committed
    // reposition needs. (In HollowDonut, the tube's *axial* direction has
    // half again the clearance of the radial one and, unlike it, does not
    // sweep around as the marble circles the tube wall.)
    let score = |free: f32, vis: f32| (free / desired.max(1e-6)).min(2.0) + vis;
    let mut best: Option<(Vec3, f32, f32)> = None;

    // Candidate zero: straight out along the local clearance gradient at the
    // marble itself -- "away from whatever surface is nearest". In a tunnel
    // or against a wall that points into the open middle, which is where the
    // camera wants to be, and it costs one `outward` (4 `de` calls) plus one
    // sweep. Without it the ring below has to hill-climb toward the same
    // answer 40° at a time, which in HollowDonut's tube measured at about a
    // second and a half of the marble filling the frame while it got there.
    let outward = sdf.outward(focus, cfg.camera_radius * 0.25);
    if outward != Vec3::ZERO {
        let sw = sweep(sdf, focus, outward, probe_dist, cfg);
        best = Some((outward, sw.free_distance, score(sw.free_distance, sw.visibility)));
    }

    for angle in SEARCH_ANGLES {
        for axis in [right, up, -right, -up] {
            // Rotate `u` toward `axis` by `angle` -- `u × axis` is
            // perpendicular to the view ray, so like every other rotation in
            // this module it can introduce no twist.
            let rot_axis = u.cross(axis);
            if rot_axis.length_squared() < 1e-10 {
                continue;
            }
            let candidate = Quat::from_axis_angle(rot_axis.normalize(), angle) * u;
            let sw = sweep(sdf, focus, candidate, probe_dist, cfg);
            let s = score(sw.free_distance, sw.visibility);
            if best.is_none_or(|(_, _, best_score)| s > best_score) {
                best = Some((candidate, sw.free_distance, s));
            }
        }
    }
    // Only worth acting on if it is a real improvement: a marginal one would
    // just make the camera twitch between equally-bad directions.
    best.filter(|(_, _, s)| *s > current_score + 0.25).map(|(dir, free, _)| (dir, free))
}

/// Applies a requested change of view direction, with the component that
/// would drive the camera into a surface removed — collide-and-slide, in
/// the angular domain (see [`WALL_COMFORT_FRACTION`]).
///
/// `wanted` is where the camera would go if geometry did not exist: the
/// player's own rotation this frame, plus any decay back toward their
/// intent. Returns the orientation to actually adopt.
///
/// The removal is proportional, not a threshold: at the comfort distance
/// nothing is removed, and it ramps to full removal as clearance goes to
/// zero. So sliding along a wall stays smooth, and a camera that is merely
/// *near* geometry still turns freely. Only the into-the-surface component
/// is ever touched — pushing away from a wall, or along it, is untouched, so
/// the player is never stuck against one.
fn constrain_rotation(
    sdf: &impl Sdf,
    from: Quat,
    wanted: Quat,
    focus: Vec3,
    desired: f32,
    probe_dist: f32,
    cfg: SweepConfig,
) -> Quat {
    let u_from = -(from * Vec3::NEG_Z);
    let u_wanted = -(wanted * Vec3::NEG_Z);
    let motion = u_wanted - u_from;
    if motion.length_squared() < 1e-12 {
        return wanted;
    }

    let comfort = WALL_COMFORT_FRACTION * desired;
    let free_wanted = sweep(sdf, focus, u_wanted, probe_dist, cfg).free_distance;
    if free_wanted >= comfort {
        return wanted; // nothing worth resisting
    }

    // Which way does clearance improve, in the camera's own screen plane?
    // Two extra sweeps; only ever paid while actually near a surface.
    let right = from * Vec3::X;
    let up = from * Vec3::Y;
    let probe = |d: Vec3| sweep(sdf, focus, d.normalize(), probe_dist, cfg).free_distance;
    let gradient = Vec2::new(
        probe(u_wanted + right * WALL_GRADIENT_EPS) - free_wanted,
        probe(u_wanted + up * WALL_GRADIENT_EPS) - free_wanted,
    );
    if gradient.length() < WALL_GRADIENT_MIN_FRACTION * desired {
        // Equally tight in every direction -- a tunnel, a pocket, or the
        // camera pressed square against a face. No wall to slide along, so
        // nothing to constrain (see `WALL_GRADIENT_MIN_FRACTION`).
        return wanted;
    }
    let into_wall = -gradient.normalize();

    let motion_2d = Vec2::new(motion.dot(right), motion.dot(up));
    let into = motion_2d.dot(into_wall);
    if into <= 0.0 {
        return wanted; // moving along the surface, or away from it
    }
    let floor = WALL_FLOOR_FRACTION * desired;
    let strength = ((comfort - free_wanted) / (comfort - floor).max(1e-4)).clamp(0.0, 1.0);
    let slid = motion_2d - into_wall * (into * strength);
    let u_slid = (u_from + right * slid.x + up * slid.y).normalize_or_zero();
    if u_slid == Vec3::ZERO {
        return wanted;
    }
    // A pure swing, exactly like every other rotation this module applies:
    // it moves `forward` and cannot introduce twist.
    (Quat::from_rotation_arc(u_from, u_slid) * from).normalize()
}

/// One frame of camera solving. Pure: no globals, no time source, no
/// rendering — everything it needs is in `input` and `sdf`, which is what
/// makes the behavior testable against analytic worlds (see this module's
/// tests) rather than only observable by eye.
pub fn solve(rig: &mut CameraRig, input: &mut SolveInput, sdf: &impl Sdf) {
    let dt = input.dt.clamp(0.0, MAX_DT);
    let radius = input.marble_radius.max(1e-5);

    // How far back the framing rule wants the camera, before geometry gets
    // a say. Uses the *base* focal length, not the possibly-widened current
    // one: FOV widening is a response to being unable to achieve this
    // distance, so feeding the widened value back in here would be a
    // feedback loop (widen -> want to be closer -> widen more).
    let min_distance = radius * MIN_DISTANCE_MARBLE_RADII;
    let desired = framing_distance(radius, FOCAL_LENGTH, input.target_fraction, input.aspect)
        * input.zoom.max(0.01);
    let desired = desired.clamp(min_distance, MAX_DISTANCE);

    if !rig.initialized {
        rig.orientation = input.intent;
        rig.focus = input.marble_pos;
        rig.focus_vel = Vec3::ZERO;
        rig.distance = desired;
        rig.distance_vel = 0.0;
        rig.focal_length = FOCAL_LENGTH;
        rig.blocked_for = 0.0;
        rig.clear_for = 0.0;
        rig.cramped_for = 0.0;
        rig.room_for = 0.0;
        rig.last_goal = desired;
        rig.search_dir = None;
        rig.search_hold = 0.0;
        rig.recover_lockout = 0.0;
        rig.last_intent = input.intent;
        rig.input_idle_for = ELECTIVE_INPUT_IDLE;
        rig.initialized = true;
    }

    if !input.smart {
        // Framing rule only (the default; `?smartcam=1` opts in to the
        // rest): track intent exactly, no smoothing, no geometry awareness.
        // This is both the shipped default until the solver has been
        // play-tested and the A/B baseline the probe harness measures
        // against. Deliberately not "disable the whole module" -- the
        // framing rule has no failure mode that needs an escape hatch,
        // whereas the geometry-aware behaviors are the ones worth being able
        // to switch off when diagnosing a feel complaint.
        rig.orientation = input.intent;
        rig.last_intent = input.intent;
        rig.focus = input.marble_pos;
        rig.focus_vel = Vec3::ZERO;
        rig.distance_vel = 0.0;
        rig.focal_length = FOCAL_LENGTH;
        let camera_radius = CAMERA_RADIUS_MARBLE_RADII * radius;
        let sw = sweep(
            sdf,
            rig.focus,
            -(rig.orientation * Vec3::NEG_Z),
            desired,
            SweepConfig {
                camera_radius,
                target_radius: radius,
                min_camera_distance: min_distance,
                max_steps: SWEEP_MAX_STEPS,
            },
        );
        // The one geometry-aware thing that stays on with the flag off:
        // don't put the eye inside a wall. Not a feel change -- the camera
        // still points exactly where the player says, instantly, with no
        // damping and no auto-rotation -- but it is what the deleted
        // per-scene distance constants were *for* (HollowDonut's `0.6` was
        // chosen because the tube's interior free radius is `0.85`), so
        // dropping them without this would leave the flag-off default
        // strictly worse than before this feature existed. The sweep it
        // needs is already being run for the debug overlay.
        rig.distance = desired.min(sw.free_distance).max(min_distance);
        rig.debug = RigDebug {
            visibility: sw.visibility,
            free_distance: sw.free_distance,
            desired_distance: desired,
            screen_fraction: screen_fraction(radius, FOCAL_LENGTH, rig.distance, input.aspect),
            deviation: 0.0,
            eye_clearance: sdf.de(rig.eye()),
            camera_radius,
            steps: sw.steps,
        };
        return;
    }

    // --- 1. focus: spring toward the marble, across the frame only ---
    spring_vec3(&mut rig.focus, &mut rig.focus_vel, input.marble_pos, FOCUS_TAU, dt);
    let lag = rig.focus - input.marble_pos;
    let max_lag = MAX_FOCUS_LAG_FRACTION * rig.distance;
    if lag.length_squared() > max_lag * max_lag {
        rig.focus = input.marble_pos + lag.normalize() * max_lag;
    }
    // Smoothing applies *across* the picture, never along the view axis.
    //
    // A spring following a moving target necessarily trails it, by roughly
    // `speed * FOCUS_TAU`. Across the frame that is harmless and even
    // desirable -- the marble leads slightly in the direction it is
    // travelling. Along the view axis it is neither: it silently adds the
    // trailing distance to the camera's distance (a marble flying away at 3
    // units/s sat ~20% further back than the framing rule asked for), and
    // then, the instant the marble stops -- which for a marble means hitting
    // something -- all of it unwinds at the spring's rate. That reads as the
    // camera whipping in toward the marble on every collision, reported from
    // play, and reproduced in `a_marble_that_stops_dead_does_not_pull_the_
    // camera_in` below.
    //
    // Zeroing the depth component leaves the eye exactly `distance` from
    // the marble's own view plane at all times, so how far away the camera
    // is stays purely the distance solver's business (§4.4) -- damped,
    // asymmetric and geometry-aware -- rather than something the follow
    // spring gets an unowned say in.
    let forward = rig.orientation * Vec3::NEG_Z;
    let depth_error = (rig.focus - input.marble_pos).dot(forward);
    rig.focus -= forward * depth_error;
    rig.focus_vel -= forward * rig.focus_vel.dot(forward);

    // --- 2. the player's own rotation, under the wall-slide constraint ---
    // The input systems write [`CameraOrbit`] only; the realized camera picks
    // their rotation up here as a delta, so it can be *projected* when it
    // would drive the camera into a surface. In open space the projection is
    // inert and this is exactly the rotation the player asked for, applied
    // the same frame they asked for it -- input stays 1:1 and undamped,
    // which is the whole contract (`camera::apply_drag`'s doc).
    let camera_radius = CAMERA_RADIUS_MARBLE_RADII * radius;
    let probe_dist_for_input = desired.max(rig.distance).max(min_distance);
    let input_cfg = SweepConfig {
        camera_radius,
        target_radius: radius,
        min_camera_distance: min_distance,
        max_steps: SWEEP_MAX_STEPS,
    };
    let player_delta = input.intent * rig.last_intent.inverse();
    rig.last_intent = input.intent;
    let delta_angle = player_delta.angle_between(Quat::IDENTITY);
    if delta_angle > 1e-4 {
        rig.input_idle_for = 0.0;
        let wanted = (player_delta * rig.orientation).normalize();
        rig.orientation = constrain_rotation(
            sdf,
            rig.orientation,
            wanted,
            rig.focus,
            desired,
            probe_dist_for_input,
            input_cfg,
        );
    } else {
        rig.input_idle_for += dt;
    }

    // --- 3. one march along the current sightline ---
    let u = -(rig.orientation * Vec3::NEG_Z); // focus -> eye
    // Probe at least as far as the camera currently is: shrinking the probe
    // to `desired` alone would report "clear" for a camera that is already
    // further out than that and about to be pulled in.
    let probe_dist = desired.max(rig.distance).max(min_distance);
    let sweep_cfg = SweepConfig {
        camera_radius,
        target_radius: radius,
        // Nothing inside the closest the camera may ever sit counts as an
        // obstruction -- see `SweepConfig::min_camera_distance`.
        min_camera_distance: min_distance,
        max_steps: SWEEP_MAX_STEPS,
    };
    let sw = sweep(sdf, rig.focus, u, probe_dist, sweep_cfg);

    // --- 4. direction: slide away from the obstruction, decay to intent ---
    let direction_before = rig.orientation;
    // A search commitment (below) outranks both the slide and the decay back
    // to intent for as long as it lasts. Letting them run alongside it is a
    // tug of war with a committed rotation, and it measurably was one: in
    // HollowDonut's tube the solver spent a second and a half rotating
    // toward a direction with a clear 1.4-unit shot while the decay dragged
    // it back toward the cramped one the player had last pointed at, netting
    // ~0 progress and a lot of travel.
    let committed = rig.search_dir.is_some();
    if sw.visibility < 1.0 {
        rig.blocked_for += dt;
        rig.clear_for = 0.0;
        if let Some(blocker) = sw.blocker {
            if let Some(axis) = slide_axis(sdf, blocker, u, rig.orientation, camera_radius * 0.25)
                .filter(|_| !committed)
            {
                let rate = if rig.blocked_for > PANIC_AFTER && sw.visibility <= 0.02 {
                    SLIDE_PANIC_RATE
                } else {
                    SLIDE_RATE
                };
                // Proportional to how blocked the view is: a barely-clipped
                // shot drifts imperceptibly, a fully blocked one moves with
                // purpose, and there is no threshold in between to chatter
                // across (which is the whole reason visibility is continuous).
                let theta = rate * (1.0 - sw.visibility) * dt;
                rig.orientation = (Quat::from_axis_angle(axis, theta) * rig.orientation).normalize();
            }
        }
    } else {
        rig.clear_for += dt;
        rig.blocked_for = 0.0;
    }

    // Decay back toward the player's intent -- but not while cramped (as of
    // the previous frame; this runs before this frame's re-sweep). Recovery
    // and the cramped search below pull in opposite directions by
    // construction: intent is where the camera got stuck in the tight spot
    // in the first place, so letting recovery run there would just undo the
    // search's work every frame and leave the camera oscillating in place.
    rig.recover_lockout = (rig.recover_lockout - dt).max(0.0);
    // Also waits for the player to stop steering: while they are dragging,
    // the intent is moving under their hand and the realized camera is
    // already tracking it directly, so a decay toward it does nothing but
    // fight the wall-slide constraint on the same frame.
    if rig.clear_for > RECOVER_HOLD
        && rig.cramped_for <= 0.0
        && !committed
        && rig.recover_lockout <= 0.0
        && rig.input_idle_for > ELECTIVE_INPUT_IDLE
    {
        // Through the same constraint as the player's own rotation: the
        // decay is also a motion toward intent, and intent is exactly where
        // the camera was prevented from going if the player has been
        // pushing it into a wall. Unconstrained, it would undo the slide a
        // little at a time and put the dive back.
        let wanted = rig.orientation.slerp(input.intent, approach(RECOVER_TAU, dt)).normalize();
        rig.orientation = constrain_rotation(
            sdf,
            rig.orientation,
            wanted,
            rig.focus,
            desired,
            probe_dist,
            sweep_cfg,
        );
    }

    // Bound the disagreement with the player (see MAX_CORRECTION) -- by
    // dragging the *intent* along, not by hauling the camera back.
    //
    // Hauling the camera back was the first version, and it deadlocks: a
    // marble travelling down a curved tunnel keeps rotating which directions
    // are usable, while the intent quaternion sits wherever the player last
    // left it, so the deviation grows for reasons that have nothing to do
    // with the camera misbehaving. Once it pins against the cap, the camera
    // is forbidden from going anywhere it can actually see from. Letting the
    // intent follow along is both the honest reading (the player's "behind
    // the marble" means behind it *now*, not in the direction the marble was
    // heading a corner ago) and the standard behavior -- a stick-relative
    // heading gets carried around corners in every third-person game.
    //
    // Note this only ever fires at the cap, i.e. after ~110° of forced
    // deviation. Ordinary sliding never reaches it, so an obstruction that
    // clears still springs the camera back to exactly where the player
    // pointed it.
    let mut deviation = rig.orientation.angle_between(input.intent);
    if deviation > MAX_CORRECTION {
        let excess = (deviation - MAX_CORRECTION) / deviation;
        input.intent = input.intent.slerp(rig.orientation, excess).normalize();
        deviation = MAX_CORRECTION;
    }

    // --- 5. re-sweep along wherever the direction ended up ---
    // Everything above moved the view ray, and `sw` describes it as it was at
    // the *start* of the frame. The distance solve below is the step that
    // actually puts the eye somewhere, so running it on a frame-old free
    // distance is precisely how the eye ends up inside a wall the camera has
    // just turned toward -- found by the HollowDonut probe, where the marble
    // hugs a curved tube wall and which directions are usable changes from
    // frame to frame. Skipped entirely when the direction didn't move, which
    // is the settled, nothing-in-the-way case.
    let u = -(rig.orientation * Vec3::NEG_Z);
    let mut sw = if rig.orientation.angle_between(direction_before) > 1e-4 {
        sweep(sdf, rig.focus, u, probe_dist, sweep_cfg)
    } else {
        sw
    };

    // --- 6. emergency: this direction has nowhere for the camera to be ---
    // A free distance below the minimum means every point on this ray that
    // isn't inside the marble is inside the geometry: the distance solve
    // below has no legal answer and will floor at the minimum, burying the
    // eye. Sliding out of that at ~90°/s would leave it buried for a good
    // fraction of a second, so this searches a ring of alternatives and
    // swings to the best -- quickly if the eye is about to be trapped,
    // immediately if it already is. The one place the camera is allowed to
    // move faster than the player can follow, and strictly a rescue.
    //
    // The second trigger catches the other case sliding cannot solve: a view
    // that has been fully blocked long enough to demonstrate that easing
    // around the obstruction isn't finding the way out (a long curved
    // tunnel, where the opening is nowhere near where the blocker's surface
    // normal points).
    // Third trigger, and the only non-urgent one: a view that is perfectly
    // clear but has been pinned far closer than the framing rule wants for
    // long enough that it is not just a passing squeeze (see
    // `CRAMPED_FRACTION`).
    if sw.free_distance < CRAMPED_FRACTION * desired {
        rig.cramped_for += dt;
    } else {
        rig.cramped_for = 0.0;
    }

    let hopeless = sw.free_distance < min_distance * 1.05;
    let stuck = rig.blocked_for > PANIC_AFTER && sw.visibility <= 0.05;
    // Elective only, so it waits for the player to stop steering
    // ([`ELECTIVE_INPUT_IDLE`]) -- a reposition that fights the hand on the
    // controls is worse than the cramped shot it is trying to fix. The two
    // triggers above are safety, and are never gated.
    let cramped = rig.cramped_for > CRAMPED_HOLD && rig.input_idle_for > ELECTIVE_INPUT_IDLE;

    // Pick a target direction, or keep the one already committed to. The
    // commitment is what makes this converge: re-running the search every
    // frame and rotating a little toward whatever won *that* frame lets the
    // camera thrash between two nearly-equal candidates and make no progress
    // toward either (measured: it doubled the HollowDonut probe's total
    // camera travel while leaving it just as cramped). Committing for
    // `SEARCH_COMMIT` seconds turns the same machinery into a decision that
    // gets seen through -- and costs nine fewer sweeps on every frame in
    // between.
    rig.search_hold = (rig.search_hold - dt).max(0.0);
    if rig.search_hold <= 0.0 {
        rig.search_dir = None;
    }
    if rig.search_dir.is_none() && (hopeless || stuck || cramped) {
        let current_score = (sw.free_distance.min(desired) / desired.max(1e-6)) + sw.visibility;
        if let Some((best_dir, best_free)) = search_direction(
            sdf,
            rig.focus,
            u,
            rig.orientation,
            // Probe half again as far as the framing rule wants, so the
            // score above can actually tell a merely-adequate direction from
            // a roomy one (it credits clearance past `desired`; a probe that
            // stops at `desired` would report every adequate direction as
            // identical). Only a search pays this.
            desired * 1.6,
            desired,
            sweep_cfg,
            current_score,
        ) {
            rig.search_dir = Some(best_dir);
            rig.search_free = best_free;
            rig.search_hold = SEARCH_COMMIT;
            rig.recover_lockout = RECOVER_LOCKOUT;
        }
    }

    if let Some(target) = rig.search_dir {
        let swing = Quat::from_rotation_arc(u, target);
        let (axis, angle) = swing.to_axis_angle();
        if angle < 1e-3 {
            // Arrived: release the commitment so the next frame is free to
            // resume ordinary sliding (or to search again from here).
            rig.search_dir = None;
            rig.search_hold = 0.0;
        } else {
            // Already trapped -- or would be, at the distance this frame is
            // about to settle on. A rate limit there would only decide how
            // many frames get rendered from inside a wall.
            let trapped = sdf.de(rig.focus + u * rig.distance.max(min_distance)) < 0.0;
            // Rate matches urgency: instant when already trapped, panic
            // speed when there is nowhere to stand at all, and the ordinary
            // slide rate when this is only about framing -- a camera that
            // whipped around at rescue speed because the shot was a bit
            // tight would read as the camera taking over.
            let rate = if hopeless || stuck { SLIDE_PANIC_RATE } else { REPOSITION_RATE };
            let step = if trapped { angle } else { (rate * dt).min(angle) };
            rig.orientation = (Quat::from_axis_angle(axis, step) * rig.orientation).normalize();
            // Interpolate the free distance the same way the direction was:
            // committing to the full searched value while only part-way
            // there would let the camera back out through geometry it hasn't
            // actually cleared yet.
            let frac = step / angle;
            sw.free_distance += (rig.search_free - sw.free_distance) * frac;
            // A rescue is allowed to exceed the deviation cap: the
            // alternative to disagreeing with the player here is a camera
            // inside a wall. The cap re-applies on the next ordinary frame.
            deviation = rig.orientation.angle_between(input.intent);
        }
    }
    let free_distance = sw.free_distance;

    // --- 7. distance: fast in, slow out, and only after a hold ---
    let goal = desired.min(free_distance).max(min_distance);
    // The hold keys off the goal *tightening*, not off the camera being
    // short of it. Keying it off "is there room ahead of me" instead reads a
    // camera sitting exactly at a stable goal as having room to grow, so the
    // timer runs out and the camera pushes past the constraint, gets pulled
    // back, and oscillates -- a 10-frame flicker in a tight space turned
    // that into visible pumping, which is the exact failure the hold exists
    // to prevent.
    if goal < rig.last_goal * 0.98 {
        rig.room_for = 0.0;
    } else {
        rig.room_for += dt;
    }
    rig.last_goal = goal;
    if goal < rig.distance {
        spring(&mut rig.distance, &mut rig.distance_vel, goal, PULL_IN_TAU, dt);
    } else if rig.room_for > PUSH_OUT_HOLD {
        spring(&mut rig.distance, &mut rig.distance_vel, goal, PUSH_OUT_TAU, dt);
    } else {
        // Holding at the pulled-in distance: bleed the spring's velocity so
        // it doesn't resume with stale momentum once the hold expires.
        rig.distance_vel *= (-dt / PUSH_OUT_TAU).exp();
    }
    rig.distance = rig.distance.clamp(min_distance, MAX_DISTANCE);

    // --- 8. last-resort clearance check at the eye itself ---
    // Re-zero the focus's depth error first (step 1 did it against the
    // direction as it was *then*; steps 3-5 have since moved it), so the eye
    // this checks is the eye that renders.
    let forward = rig.orientation * Vec3::NEG_Z;
    let depth_error = (rig.focus - input.marble_pos).dot(forward);
    rig.focus -= forward * depth_error;
    rig.focus_vel -= forward * rig.focus_vel.dot(forward);

    // The sweep bounds clearance along the ray; this checks the one point
    // that actually matters, after every other step has had its say. One
    // `de`, and it catches anything the sweep's own discretisation or a
    // rescue's partial swing left behind.
    //
    // Corrects by *twice* the shortfall rather than exactly it: pulling in
    // by `d` only recovers `d` of clearance when the field's gradient along
    // the view ray is 1, and on the grazing rays where this backstop
    // actually fires it is nowhere near -- so an exact correction leaves the
    // eye a hair inside the surface it was supposed to be pulled out of.
    // Over-correcting is free: the push-out spring gives the distance back
    // as soon as there is room.
    let clearance = sdf.de(rig.eye());
    if clearance < camera_radius {
        rig.distance = (rig.distance - 2.0 * (camera_radius - clearance)).max(min_distance);
    }

    // --- 9. FOV: widen only as far as geometry has forced us in ---
    // Expressed as the ratio between where the camera is and where framing
    // wanted it, so a deliberate zoom-in (which lowers `desired` too) does
    // *not* get silently undone by a widening FOV.
    let focal_goal = (FOCAL_LENGTH * rig.distance / desired.max(1e-6))
        .clamp(MIN_FOCAL_LENGTH, FOCAL_LENGTH);
    rig.focal_length += (focal_goal - rig.focal_length) * approach(FOCAL_TAU, dt);

    rig.debug = RigDebug {
        visibility: sw.visibility,
        free_distance,
        desired_distance: desired,
        screen_fraction: screen_fraction(radius, rig.focal_length, rig.distance, input.aspect),
        deviation,
        eye_clearance: clearance,
        camera_radius,
        steps: sw.steps,
    };
}

/// Which framing target applies. Touch is detected by observing a real touch
/// event rather than by guessing from screen size: a phone in landscape and
/// a small desktop window look identical to the window API, but only one of
/// them has a finger covering a quarter of the screen.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct InputProfile {
    pub touch_seen: bool,
}

impl InputProfile {
    pub fn target_fraction(&self) -> f32 {
        if self.touch_seen {
            TOUCH_TARGET_FRACTION
        } else {
            POINTER_TARGET_FRACTION
        }
    }
}

/// `Update` system: runs one [`solve`] against the live scene tree.
#[allow(clippy::too_many_arguments)] // SystemParam count
///
/// Ordered after the input systems (so this frame's drag is already in
/// `CameraOrbit`/`CameraRig`) and before `render::update_frame_data` (so the
/// solved pose reaches this frame's uniforms rather than landing one frame
/// late) — see `main.rs`'s `Update` chain.
pub fn smart_camera_solve(
    time: Res<Time>,
    mut orbit: ResMut<CameraOrbit>,
    config: Res<crate::config::Config>,
    touches: Res<bevy::input::touch::Touches>,
    marble_state: Res<crate::physics_sys::MarbleState>,
    mp: Res<crate::physics_sys::MultiplayerSession>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    mut profile: ResMut<InputProfile>,
    mut rig: ResMut<CameraRig>,
    mut timings: ResMut<crate::fps_overlay::PhaseTimings>,
) {
    let start = web_time::Instant::now();
    if touches.iter().next().is_some() {
        profile.touch_seen = true;
    }
    let marble = marble_state.local_marble();
    let aspect = windows.single().map(|w| w.width() / w.height().max(1.0)).unwrap_or(1.0);
    let scene = mp.sim.scene();
    let sdf = marble_csg::visibility::SceneSdf { object: &scene.object, params: &scene.params };
    let mut solve_input = SolveInput {
        marble_pos: marble.pos,
        marble_radius: marble.rad,
        intent: orbit.orientation,
        zoom: orbit.zoom,
        aspect,
        target_fraction: profile.target_fraction(),
        dt: time.delta_secs(),
        smart: config.smart_camera,
    };
    solve(&mut rig, &mut solve_input, &sdf);
    // The solver is allowed to drag the intent along when the world has
    // forced the camera past `MAX_CORRECTION` from it (see the cap in
    // `solve`); write that back so the player's next drag starts from where
    // the camera actually is.
    orbit.orientation = solve_input.intent;
    timings.record("camera", start.elapsed());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty space.
    struct Empty;
    impl Sdf for Empty {
        fn de(&self, _p: Vec3) -> f32 {
            1e9
        }
    }

    /// Solid half-space `x > plane_x`.
    struct Wall {
        plane_x: f32,
    }
    impl Sdf for Wall {
        fn de(&self, p: Vec3) -> f32 {
            self.plane_x - p.x
        }
    }

    /// Infinite cylinder of radius `r` along Y at `(cx, _, cz)`.
    struct Pillar {
        cx: f32,
        cz: f32,
        r: f32,
    }
    impl Sdf for Pillar {
        fn de(&self, p: Vec3) -> f32 {
            ((p.x - self.cx).powi(2) + (p.z - self.cz).powi(2)).sqrt() - self.r
        }
    }

    /// Hollow spherical room of inner radius `r` centered at the origin —
    /// the "camera and marble are both inside a tight space" case
    /// (HollowDonut, a Menger tunnel).
    struct Room {
        r: f32,
    }
    impl Sdf for Room {
        fn de(&self, p: Vec3) -> f32 {
            self.r - p.length()
        }
    }

    fn input(intent: Quat, dt: f32) -> SolveInput {
        SolveInput {
            marble_pos: Vec3::ZERO,
            marble_radius: 0.15,
            intent,
            zoom: 1.0,
            aspect: 16.0 / 9.0,
            target_fraction: POINTER_TARGET_FRACTION,
            dt,
            smart: true,
        }
    }

    /// Looking along -Z from +Z (identity orientation): the eye sits at
    /// `focus + Z*distance`.
    fn settled(sdf: &impl Sdf, inp: &SolveInput, frames: usize) -> CameraRig {
        let mut rig = CameraRig::default();
        let mut inp = *inp;
        for _ in 0..frames {
            solve(&mut rig, &mut inp, sdf);
        }
        rig
    }

    #[test]
    fn framing_rule_reproduces_the_hand_tuned_per_scene_distances() {
        // The distances that were previously hand-picked by screenshot, and
        // what the framing rule produces for the same marble at 16:9. Both
        // open-space scenes land close; HollowDonut's 0.6 is deliberately
        // absent -- that one was chosen to fit inside the tube, which is the
        // clearance solver's job, not framing's (rust/CAMERA.md §4.2).
        let f = FOCAL_LENGTH;
        let demo = framing_distance(0.02, f, POINTER_TARGET_FRACTION, 16.0 / 9.0);
        assert!((demo - 0.2).abs() / 0.2 < 0.15, "demo framing {demo} vs hand-tuned 0.2");
        let menger = framing_distance(0.15, f, POINTER_TARGET_FRACTION, 16.0 / 9.0);
        assert!((menger - 1.2).abs() / 1.2 < 0.2, "menger framing {menger} vs hand-tuned 1.2");
    }

    #[test]
    fn framing_hits_the_target_screen_fraction_at_every_aspect_and_radius() {
        for aspect in [0.46, 1.0, 1.78] {
            for radius in [0.02, 0.15, 0.4] {
                for fraction in [POINTER_TARGET_FRACTION, TOUCH_TARGET_FRACTION] {
                    let d = framing_distance(radius, FOCAL_LENGTH, fraction, aspect);
                    let got = screen_fraction(radius, FOCAL_LENGTH, d, aspect);
                    assert!(
                        (got - fraction).abs() < 1e-3,
                        "aspect={aspect} r={radius}: wanted {fraction}, got {got}"
                    );
                }
            }
        }
    }

    #[test]
    fn open_space_settles_at_the_framing_distance_pointing_where_the_player_asked() {
        let intent = CameraOrbit::orientation_from_yaw_pitch(0.7, 0.3);
        let inp = input(intent, 1.0 / 60.0);
        let rig = settled(&Empty, &inp, 240);
        let want = framing_distance(0.15, FOCAL_LENGTH, POINTER_TARGET_FRACTION, 16.0 / 9.0);
        assert!((rig.distance - want).abs() < 1e-3, "settled at {} want {want}", rig.distance);
        assert!(
            rig.orientation.angle_between(intent) < 1e-3,
            "with nothing in the way the camera must end up exactly where the player pointed it"
        );
        assert!((rig.focal_length - FOCAL_LENGTH).abs() < 1e-3, "no reason to widen the FOV here");
    }

    #[test]
    fn a_wall_behind_the_marble_never_lets_the_eye_through_it() {
        // Marble at the origin, framing wants the camera ~1.36 back along
        // +Z, wall at z = 0.5. The camera may resolve this either way --
        // dolly in to stay in front of the wall, or rotate to a direction
        // that has room -- but it must never end up on the far side of it,
        // and must never be inside it.
        struct ZWall;
        impl Sdf for ZWall {
            fn de(&self, p: Vec3) -> f32 {
                0.5 - p.z
            }
        }
        let inp = input(Quat::IDENTITY, 1.0 / 60.0);
        let mut rig = CameraRig::default();
        let mut inp_mut = inp;
        for _ in 0..180 {
            solve(&mut rig, &mut inp_mut, &ZWall);
            assert!(
                rig.eye().z < 0.5,
                "eye crossed the wall plane: {:?}",
                rig.eye()
            );
        }
        assert!(
            rig.debug.eye_clearance > 0.0,
            "eye ended up inside geometry (clearance {})",
            rig.debug.eye_clearance
        );
        assert!(rig.debug.visibility > 0.9, "and it should still be able to see the marble");
    }

    #[test]
    fn a_pillar_between_camera_and_marble_is_resolved_and_the_shot_reopened() {
        // The headline requirement: the marble must not stay hidden, and the
        // camera must not end up jammed against the obstruction either.
        //
        // A pillar sitting squarely on the view ray is first resolved by
        // dollying in (the camera comes to rest in front of it, from where
        // the marble is perfectly visible) -- which is a *clear* shot, just a
        // cramped one, at roughly a quarter of the distance framing wants.
        // The cramped trigger is what then reopens it: the search finds a
        // direction with room and the camera repositions.
        let pillar = Pillar { cx: 0.0, cz: 0.7, r: 0.35 };
        let mut inp = input(Quat::IDENTITY, 1.0 / 60.0);
        let mut rig = CameraRig::default();
        solve(&mut rig, &mut inp, &pillar);
        let desired = framing_distance(0.15, FOCAL_LENGTH, POINTER_TARGET_FRACTION, 16.0 / 9.0);
        assert!(
            rig.debug.free_distance < 0.5 * desired,
            "test setup: the pillar should leave the camera cramped, free was {}",
            rig.debug.free_distance
        );

        for _ in 0..300 {
            solve(&mut rig, &mut inp, &pillar);
        }
        assert!(
            rig.debug.visibility > 0.9,
            "expected a clear view within 5s, got visibility {}",
            rig.debug.visibility
        );
        assert!(
            rig.distance > 0.6 * desired,
            "expected the camera to have found room ({} vs a framing distance of {desired}), \
             instead of staying jammed in front of the pillar",
            rig.distance
        );
        assert!(rig.debug.eye_clearance > 0.0);
    }

    #[test]
    fn the_camera_returns_to_the_players_intent_once_the_obstruction_is_gone() {
        let intent = Quat::IDENTITY;
        let mut inp = input(intent, 1.0 / 60.0);
        let pillar = Pillar { cx: 0.0, cz: 0.7, r: 0.35 };
        let mut rig = CameraRig::default();
        for _ in 0..240 {
            solve(&mut rig, &mut inp, &pillar);
        }
        let deviated = rig.debug.deviation;
        assert!(deviated > 0.1, "test setup: the camera should have deviated, got {deviated}");
        for _ in 0..600 {
            solve(&mut rig, &mut inp, &Empty);
        }
        assert!(
            rig.orientation.angle_between(intent) < 0.02,
            "expected a return to intent once clear, still off by {}rad",
            rig.orientation.angle_between(intent)
        );
    }

    #[test]
    fn deviation_from_intent_is_bounded_even_in_a_hopeless_case() {
        // A marble fully enclosed: no direction is ever clear, so the slide
        // never stops wanting to move. It must still not walk the camera
        // arbitrarily far from where the player pointed it.
        struct Enclosed;
        impl Sdf for Enclosed {
            fn de(&self, p: Vec3) -> f32 {
                // Thin shell just outside the marble: blocked everywhere.
                0.25 - p.length()
            }
        }
        let intent = Quat::IDENTITY;
        let mut inp = input(intent, 1.0 / 60.0);
        let mut rig = CameraRig::default();
        for _ in 0..1200 {
            solve(&mut rig, &mut inp, &Enclosed);
        }
        assert!(
            rig.orientation.angle_between(intent) <= MAX_CORRECTION + 1e-2,
            "deviation {} exceeded the cap",
            rig.orientation.angle_between(intent)
        );
        assert!(rig.distance >= 0.15 * MIN_DISTANCE_MARBLE_RADII - 1e-4);
    }

    #[test]
    fn pulling_in_is_much_faster_than_pushing_back_out() {
        // A wall that closes in, then disappears. Reaction must be prompt;
        // recovery must not be -- that asymmetry is what stops a picket
        // fence of struts from pumping the camera in and out.
        //
        // Measured over short windows deliberately: left running longer, the
        // direction search would (correctly) rotate the camera somewhere
        // roomier, which is a different behavior from the one under test.
        struct ZWall(f32);
        impl Sdf for ZWall {
            fn de(&self, p: Vec3) -> f32 {
                self.0 - p.z
            }
        }
        let mut inp = input(Quat::IDENTITY, 1.0 / 60.0);
        // Settle with a wall already at 0.9 -- far enough not to be cramped,
        // so the camera starts from a legal, stable pose rather than with
        // its eye buried in a wall that appeared around it.
        let mut rig = CameraRig::default();
        for _ in 0..240 {
            solve(&mut rig, &mut inp, &ZWall(0.9));
        }
        let open = rig.distance;
        assert!(open > 0.6, "test setup: expected a roomy start, got {open}");

        let mut frames_to_pull_in = None;
        for i in 0..24 {
            solve(&mut rig, &mut inp, &ZWall(0.45));
            if frames_to_pull_in.is_none() && rig.distance < 0.45 {
                frames_to_pull_in = Some(i);
            }
        }
        let pull_in = frames_to_pull_in.expect("never pulled in from an encroaching wall");
        assert!(pull_in <= 12, "pull-in took {pull_in} frames (>0.2s)");

        let mut frames_to_recover = None;
        for i in 0..600 {
            solve(&mut rig, &mut inp, &Empty);
            if frames_to_recover.is_none() && rig.distance > open * 0.95 {
                frames_to_recover = Some(i);
            }
        }
        let recover = frames_to_recover.expect("never recovered");
        assert!(
            recover > pull_in * 3,
            "recovery ({recover} frames) should be far slower than reaction ({pull_in} frames)"
        );
    }

    #[test]
    fn a_flickering_obstruction_does_not_pump_the_camera() {
        // Menger-strut simulation: the space around the marble opens and
        // closes every 10 frames. A spherical room (rather than a wall) on
        // purpose -- it constrains every direction equally, so the camera
        // cannot sidestep the problem and the test is purely about whether
        // the distance response oscillates.
        //
        // Correct behavior is to pull in to the tight radius and *stay*
        // there: pushing back out requires `PUSH_OUT_HOLD` of uninterrupted
        // room, which a 10-frame flicker never provides. Measured over the
        // second half of the run, after the initial pull-in, so the metric
        // is oscillation rather than the one legitimate move.
        let mut inp = input(Quat::IDENTITY, 1.0 / 60.0);
        let mut rig = CameraRig::default();
        for i in 0..480 {
            let tight = (i / 10) % 2 == 0;
            if tight {
                solve(&mut rig, &mut inp, &Room { r: 0.5 });
            } else {
                solve(&mut rig, &mut inp, &Room { r: 1.6 });
            }
        }
        let mut settled_travel = 0.0;
        let mut prev = rig.distance;
        for i in 0..240 {
            let tight = (i / 10) % 2 == 0;
            solve(&mut rig, &mut inp, &Room { r: if tight { 0.5 } else { 1.6 } });
            settled_travel += (rig.distance - prev).abs();
            prev = rig.distance;
        }
        assert!(
            settled_travel < 0.5,
            "camera pumped {settled_travel} units over 4s of flicker it should have ignored"
        );
    }

    #[test]
    fn a_tight_room_widens_the_fov_instead_of_giving_up_on_framing() {
        // Inner radius 0.5 with a 0.15 marble: framing wants ~1.35, geometry
        // allows well under half that.
        let inp = input(Quat::IDENTITY, 1.0 / 60.0);
        let rig = settled(&Room { r: 0.5 }, &inp, 300);
        assert!(rig.distance < 0.5, "camera must stay inside the room, got {}", rig.distance);
        assert!(
            rig.focal_length < FOCAL_LENGTH - 0.05,
            "expected the FOV to widen when forced close, focal is {}",
            rig.focal_length
        );
        assert!(rig.focal_length >= MIN_FOCAL_LENGTH - 1e-4);
        assert!(rig.debug.eye_clearance > 0.0, "eye must not be inside the room's wall");
    }

    #[test]
    fn the_eye_never_enters_geometry_across_a_long_randomised_run() {
        // I2 as an assertion: a marble moving through a pillar field with the
        // player dragging the camera around the whole time.
        struct Pillars;
        impl Sdf for Pillars {
            fn de(&self, p: Vec3) -> f32 {
                // A grid of vertical pillars every 1.5 units, radius 0.3.
                let gx = (p.x / 1.5).round() * 1.5;
                let gz = (p.z / 1.5).round() * 1.5;
                ((p.x - gx).powi(2) + (p.z - gz).powi(2)).sqrt() - 0.3
            }
        }
        let mut rig = CameraRig::default();
        let mut orbit = CameraOrbit { orientation: Quat::IDENTITY, zoom: 1.0 };
        let mut worst = f32::INFINITY;
        for i in 0..1200 {
            let t = i as f32 / 60.0;
            // Down the corridor between two columns of pillars, weaving
            // gently. Deliberately never *inside* a pillar: a marble buried
            // in solid geometry has no clear camera position at all (every
            // ray out of it starts underground), so requiring one would be
            // testing something no camera can deliver -- the solver's
            // guarantee there is only that it degrades to the minimum
            // distance rather than misbehaving.
            let pos = Vec3::new(0.75 + (t * 1.7).sin() * 0.2, 0.0, 0.75 + t * 0.5);
            // Player dragging continuously. Input writes *intent* only --
            // the realized camera picks the rotation up inside `solve`,
            // under the wall-slide constraint (`constrain_rotation`).
            if let Some(r) = CameraOrbit::drag_rotation(rig.orientation, Vec2::new(1.5, 0.4)) {
                orbit.orientation = (r * orbit.orientation).normalize();
            }
            let mut inp = SolveInput {
                marble_pos: pos,
                intent: orbit.orientation,
                ..input(orbit.orientation, 1.0 / 60.0)
            };
            solve(&mut rig, &mut inp, &Pillars);
            // Skip the first few frames: the rig starts at the framing
            // distance before it has ever seen the geometry.
            if i > 10 {
                worst = worst.min(rig.debug.eye_clearance);
            }
        }
        assert!(worst > 0.0, "the eye entered geometry (worst clearance {worst})");
    }

    /// Solid half-space `x > 0.6`, so the free distance along a direction
    /// `u` is exactly `0.6 / u.x` -- clearance improves monotonically as the
    /// camera turns away from +X, which makes "into the wall" and "away from
    /// it" unambiguous rather than a matter of interpretation.
    struct SideWall;
    impl Sdf for SideWall {
        fn de(&self, p: Vec3) -> f32 {
            0.6 - p.x
        }
    }

    /// A camera direction 45 degrees off the wall's normal: tight enough for
    /// the constraint to be live, with a clear direction of improvement.
    fn beside_the_wall() -> (CameraRig, SolveInput) {
        let u = Vec3::new(1.0, 0.0, 1.0).normalize();
        let mut rig = CameraRig::default();
        let mut inp = input(Quat::from_rotation_arc(Vec3::Z, u), 1.0 / 60.0);
        for _ in 0..90 {
            solve(&mut rig, &mut inp, &SideWall);
        }
        (rig, inp)
    }

    /// The drag delta that moves the *eye* toward `target_dir`. A swipe
    /// rotates `forward` toward its screen direction, and the eye sits
    /// opposite `forward`, hence the negation.
    fn drag_moving_eye_toward(rig: &CameraRig, target_dir: Vec3, pixels: f32) -> Vec2 {
        let u = -(rig.orientation * Vec3::NEG_Z);
        let right = rig.orientation * Vec3::X;
        let up = rig.orientation * Vec3::Y;
        let tangential = (target_dir - u * target_dir.dot(u)).normalize();
        Vec2::new(-tangential.dot(right), -tangential.dot(up)) * pixels
    }

    #[test]
    fn orbiting_into_a_wall_is_resisted_instead_of_diving_at_the_marble() {
        // Reported from play: with a large structure beside the marble,
        // rotating slightly took the distance from 1.411 to 0.279 in one
        // motion -- the camera swinging behind the structure, with the
        // marble fully visible throughout. The dolly was doing all the work,
        // and it is the wrong tool for it. The orbit itself should resist.
        let (mut rig, mut inp) = beside_the_wall();
        let open = rig.debug.desired_distance;
        let mut worst = rig.distance;
        for _ in 0..240 {
            let delta = drag_moving_eye_toward(&rig, Vec3::X, 6.0); // straight at the wall
            if let Some(r) = CameraOrbit::drag_rotation(rig.orientation, delta) {
                inp.intent = (r * inp.intent).normalize();
            }
            solve(&mut rig, &mut inp, &SideWall);
            worst = worst.min(rig.distance);
            assert!(rig.debug.eye_clearance > 0.0, "eye entered the wall");
        }
        assert!(
            worst > 0.9 * WALL_FLOOR_FRACTION * open,
            "four seconds of pushing into a wall cost the camera more than the floor allows: \
             {worst} of a framing distance of {open}"
        );
        assert!(
            rig.debug.visibility > 0.99,
            "and it should never have lost sight of the marble ({})",
            rig.debug.visibility
        );
    }

    #[test]
    fn orbiting_away_from_a_wall_is_never_resisted() {
        // The constraint may only remove the *into the surface* component:
        // turning away from a wall has to track the request exactly, or the
        // player is stuck against it.
        let (mut rig, mut inp) = beside_the_wall();
        let free_before = rig.debug.free_distance;
        let before = rig.orientation;

        let delta = drag_moving_eye_toward(&rig, Vec3::NEG_X, 6.0);
        let rotation = CameraOrbit::drag_rotation(rig.orientation, delta).unwrap();
        inp.intent = (rotation * inp.intent).normalize();
        solve(&mut rig, &mut inp, &SideWall);

        let expected = (rotation * before).normalize();
        assert!(
            rig.orientation.angle_between(expected) < 1e-4,
            "turning away from a wall must be unconstrained (off by {} rad)",
            rig.orientation.angle_between(expected)
        );
        assert!(
            rig.debug.free_distance > free_before,
            "and it should have bought clearance back: {} -> {}",
            free_before,
            rig.debug.free_distance
        );
    }

    #[test]
    fn player_input_is_applied_exactly_in_open_space() {
        // With nothing to run into, a drag must move the realized camera by
        // exactly the arcball rotation, in the same frame, with no damping
        // and no constraint in the way -- the input contract.
        let mut rig = CameraRig::default();
        let mut inp = input(Quat::IDENTITY, 1.0 / 60.0);
        solve(&mut rig, &mut inp, &Empty);
        let before = rig.orientation;

        let rotation = CameraOrbit::drag_rotation(rig.orientation, Vec2::new(12.0, -5.0)).unwrap();
        inp.intent = (rotation * inp.intent).normalize();
        solve(&mut rig, &mut inp, &Empty);

        let expected = (rotation * before).normalize();
        assert!(
            rig.orientation.angle_between(expected) < 1e-5,
            "a drag must apply exactly, with no damping in the way"
        );
    }

    #[test]
    fn behavior_is_frame_rate_independent() {
        // The same 2 seconds of simulated time at 30/60/144 Hz must land in
        // very nearly the same place.
        let pillar = Pillar { cx: 0.0, cz: 0.7, r: 0.35 };
        let mut results = Vec::new();
        for hz in [30.0f32, 60.0, 144.0] {
            let inp = input(Quat::IDENTITY, 1.0 / hz);
            let frames = (2.0 * hz) as usize;
            let rig = settled(&pillar, &inp, frames);
            results.push((rig.distance, rig.debug.deviation));
        }
        for (d, dev) in &results[1..] {
            assert!(
                (d - results[0].0).abs() < 0.05,
                "distance varied with frame rate: {results:?}"
            );
            // Looser than the distance bound above, and deliberately so:
            // the direction path goes through discrete decision points (the
            // search's commit window, the panic-rate threshold) that land on
            // different simulated instants at different frame rates, so a
            // few degrees of spread is expected rather than evidence of a
            // dt-dependent formula. What this catches is the real bug --
            // a fixed per-frame lerp constant, which at 30 vs 144 Hz would
            // differ by a factor, not by 0.1 rad.
            assert!(
                (dev - results[0].1).abs() < 0.2,
                "deviation varied with frame rate: {results:?}"
            );
        }
    }

    #[test]
    fn a_marble_resting_on_a_surface_does_not_collapse_the_shot() {
        // Second half of the reported "camera dives at the marble" bug, and
        // the more serious half: it fired on *approaching* a surface, not
        // just on stopping. Captured on device as
        // `vis 0.00 d 0.225/4.816 (free 0.000) size 2.424 ... steps 1` --
        // a march that gave up on its first sample.
        //
        // Two causes, both fixed: the probe ball was scaled to the framing
        // distance (`0.08 * desired`), which at this scene's `zoom = 3.3`
        // made it four marble radii wide; and the march began at the
        // marble's own surface, so the floor it was resting on sat inside
        // the very first sample. Nothing about a marble touching the ground
        // should move the camera at all -- the camera is up and behind, with
        // an unobstructed view.
        struct Floor;
        impl Sdf for Floor {
            fn de(&self, p: Vec3) -> f32 {
                p.y
            }
        }
        let radius = 0.15;
        let mut rig = CameraRig::default();
        let mut inp = SolveInput {
            marble_pos: Vec3::new(0.0, radius, 0.0), // exactly touching
            marble_radius: radius,
            intent: Quat::from_rotation_x(-0.785), // 45 degrees up and back
            zoom: 3.3,
            aspect: 384.0 / 694.0, // the reporter's phone, in portrait
            target_fraction: POINTER_TARGET_FRACTION,
            dt: 1.0 / 60.0,
            smart: true,
        };
        for _ in 0..120 {
            solve(&mut rig, &mut inp, &Floor);
        }
        let want = rig.debug.desired_distance;
        assert!(
            (rig.distance - want).abs() < 0.01 * want,
            "resting on a floor pulled the camera from {want} to {}",
            rig.distance
        );
        assert_eq!(rig.debug.visibility, 1.0, "nothing is between the camera and the marble");
        assert!(
            rig.debug.screen_fraction < 0.2,
            "marble ballooned to {} of frame",
            rig.debug.screen_fraction
        );
        assert_eq!(rig.focal_length, FOCAL_LENGTH, "no reason to widen the FOV here");
    }

    #[test]
    fn a_marble_that_stops_dead_does_not_pull_the_camera_in() {
        // The reported play bug: fly straight at something, hit it, and the
        // camera lunges toward the marble. Cause was the follow spring's
        // trailing distance being spent along the view axis, so the camera
        // rode ~20% further out while moving and reeled that in the moment
        // the marble stopped. Nothing here touches geometry -- it reproduced
        // in empty space, which is what proves it was never a deocclusion
        // problem.
        let mut rig = CameraRig::default();
        let mut pos = Vec3::ZERO;
        let mut inp = input(Quat::IDENTITY, 1.0 / 60.0); // eye at +Z
        let mut worst_ratio: f32 = 1.0;
        let mut best_ratio = f32::INFINITY;
        for i in 0..180 {
            if i < 60 {
                pos.z -= 3.0 / 60.0; // 3 units/s straight away from the eye
            }
            inp.marble_pos = pos;
            solve(&mut rig, &mut inp, &Empty);
            if i > 5 {
                let ratio = rig.eye().distance(pos) / rig.distance;
                worst_ratio = worst_ratio.max(ratio);
                best_ratio = best_ratio.min(ratio);
            }
        }
        // Pre-fix this ran to 1.20 while moving and unwound to 1.00 over the
        // ~0.3s after the stop. The eye should simply always be at the
        // solved distance, moving or stopped.
        assert!(
            worst_ratio < 1.02 && best_ratio > 0.98,
            "eye-to-marble distance wandered from the solved distance: {best_ratio:.3}..{worst_ratio:.3}"
        );
    }

    #[test]
    fn lateral_smoothing_survives_the_depth_fix() {
        // The other half of the same change: across-frame smoothing is the
        // part worth keeping, so a marble jinking sideways must still be
        // followed smoothly rather than rigidly -- while its *distance* stays
        // pinned exactly (the assertion above, restated on a sideways run).
        let mut rig = CameraRig::default();
        let mut inp = input(Quat::IDENTITY, 1.0 / 60.0);
        let mut max_offset: f32 = 0.0;
        for i in 0..180 {
            let t = i as f32 / 60.0;
            let pos = Vec3::new((t * 4.0).sin() * 0.5, 0.0, 0.0); // brisk lateral weave
            inp.marble_pos = pos;
            solve(&mut rig, &mut inp, &Empty);
            if i > 5 {
                let ratio = rig.eye().distance(pos) / rig.distance;
                assert!(ratio < 1.05, "lateral motion should not change distance much, got {ratio}");
                max_offset = max_offset.max((rig.focus - pos).length());
            }
        }
        assert!(max_offset > 1e-3, "the focus should still trail sideways -- that is the smoothing");
    }

    #[test]
    fn focus_never_lags_further_than_the_clamp_allows() {
        // A marble teleporting away at high speed: the spring alone would
        // leave the focus arbitrarily far behind, putting the marble off
        // frame. The clamp is what prevents that.
        let mut rig = CameraRig::default();
        let mut pos = Vec3::ZERO;
        let mut worst_lag_fraction = 0.0f32;
        for i in 0..300 {
            pos.x += 0.25; // 15 units/s, far faster than the marble ever moves
            let mut inp = SolveInput { marble_pos: pos, ..input(Quat::IDENTITY, 1.0 / 60.0) };
            solve(&mut rig, &mut inp, &Empty);
            if i > 5 {
                worst_lag_fraction =
                    worst_lag_fraction.max((rig.focus - pos).length() / rig.distance);
            }
        }
        assert!(
            worst_lag_fraction <= MAX_FOCUS_LAG_FRACTION + 1e-3,
            "focus lagged {worst_lag_fraction} of the camera distance"
        );
    }

    #[test]
    fn smartcam_off_tracks_intent_exactly_but_still_keeps_the_eye_out_of_walls() {
        let intent = CameraOrbit::orientation_from_yaw_pitch(0.4, -0.2);
        let inp = SolveInput { smart: false, ..input(intent, 1.0 / 60.0) };
        let want = framing_distance(0.15, FOCAL_LENGTH, POINTER_TARGET_FRACTION, 16.0 / 9.0);

        // Open space: exactly the framing distance, exactly the player's
        // orientation, no FOV games, no damping.
        let mut rig = CameraRig::default();
        let mut open = inp;
        solve(&mut rig, &mut open, &Empty);
        assert_eq!(rig.orientation, intent);
        assert_eq!(rig.focal_length, FOCAL_LENGTH);
        assert!((rig.distance - want).abs() < 1e-4);

        // With a pillar on the view ray the direction is still untouched --
        // no sliding, no deviation -- but the distance is capped so the eye
        // does not end up inside it. See the `!input.smart` branch in
        // `solve` for why this one behavior survives the flag being off.
        let pillar = Pillar { cx: 0.0, cz: 0.7, r: 0.35 };
        let rig = settled(&pillar, &inp, 60);
        assert_eq!(rig.orientation, intent, "flag-off must never rotate the camera");
        assert_eq!(rig.focal_length, FOCAL_LENGTH, "flag-off must never touch the FOV");
        assert!(rig.distance < want, "expected the distance to be capped by the pillar");
        assert!(rig.debug.eye_clearance > 0.0, "eye ended up inside the pillar");
    }

    #[test]
    fn zoom_is_a_multiplier_on_the_framed_distance() {
        for zoom in [0.5f32, 1.0, 2.0] {
            let inp = SolveInput { zoom, ..input(Quat::IDENTITY, 1.0 / 60.0) };
            let rig = settled(&Empty, &inp, 300);
            let want =
                framing_distance(0.15, FOCAL_LENGTH, POINTER_TARGET_FRACTION, 16.0 / 9.0) * zoom;
            assert!((rig.distance - want).abs() < 1e-2, "zoom {zoom}: got {} want {want}", rig.distance);
        }
    }

    #[test]
    fn a_wall_dead_ahead_does_not_widen_the_fov_past_its_limit() {
        let inp = input(Quat::IDENTITY, 1.0 / 60.0);
        let rig = settled(&Wall { plane_x: 0.05 }, &inp, 300);
        assert!(rig.focal_length >= MIN_FOCAL_LENGTH - 1e-4);
        assert!(rig.focal_length <= FOCAL_LENGTH + 1e-4);
    }
}

/// Per-scene camera probe: drives the *real* scenes through the *real*
/// physics with a scripted movement + camera-drag script, running this
/// module's solver every tick, and reports what the camera actually did.
///
/// This exists because the questions that matter about a camera ("does the
/// marble stay visible while it moves?", "does the eye ever end up inside
/// the fractal?", "does it stay a sensible size on screen?") can't be
/// answered by a screenshot and can't be answered by analytic-world unit
/// tests either -- fractal geometry is where a camera solver actually gets
/// tested, and this app's real geometry is available to a plain
/// `cargo test` because the whole world is a CPU-evaluable distance field.
///
/// `cargo test -p marble-marcher-bevy scene_probe -- --nocapture` prints the
/// per-scene table.
#[cfg(test)]
mod scene_probe {
    use super::*;
    use crate::render::{build_scene, SceneKind};
    use marble_csg::physics::{step_marbles, Marble, PhysicsConfig, PlayerInput};
    use marble_csg::visibility::SceneSdf;
    use marble_csg::Params;

    struct Report {
        scene: &'static str,
        frames: usize,
        min_visibility: f32,
        mean_visibility: f32,
        frames_blocked: usize,
        min_clearance: f32,
        min_screen_fraction: f32,
        max_screen_fraction: f32,
        mean_steps: f32,
        frames_too_close: usize,
        max_deviation_deg: f32,
        distance_travel: f32,
    }

    /// One scene, 8 simulated seconds at 60 Hz: the marble thrusts around
    /// under camera-relative control (`GravityMode::Flying`, the app's
    /// default) while the player drags the camera for the first two seconds
    /// and then lets go. Deterministic -- no RNG, no wall clock.
    fn probe(kind: SceneKind, smart: bool) -> Report {
        let mut params = Params::new();
        let (object, _handles, _anim) = build_scene(kind, &mut params);
        let spawn = kind.spawn_params();
        let cfg = PhysicsConfig::default();
        let mut marbles = vec![Marble::spawn(spawn.start, spawn.rad)];
        let starts = vec![spawn.start];

        let mut orbit = CameraOrbit::default();
        if matches!(
            kind,
            SceneKind::MengerSponge | SceneKind::MengerSphere | SceneKind::MengerOscillatingSphere
        ) {
            orbit.orientation = CameraOrbit::orientation_from_yaw_pitch(0.8, 0.35);
        }
        if kind == SceneKind::HollowDonut {
            orbit.orientation = CameraOrbit::orientation_from_yaw_pitch(0.5, 0.2);
        }
        let mut rig = CameraRig::default();

        let dt = 1.0 / 60.0;
        let frames = 480;
        let (mut min_vis, mut sum_vis, mut blocked) = (1.0f32, 0.0f32, 0usize);
        let mut min_clear = f32::INFINITY;
        let (mut min_frac, mut max_frac) = (f32::INFINITY, 0.0f32);
        let mut sum_steps = 0.0f32;
        let mut too_close = 0usize;
        let mut max_dev = 0.0f32;
        let mut travel = 0.0f32;
        let mut prev_distance: Option<f32> = None;

        for i in 0..frames {
            let t = i as f32 * dt;
            // Marble input: a wandering thrust, camera-relative (so it
            // interacts with whatever the camera is doing, exactly as a
            // player's would).
            let input = PlayerInput {
                dx: (t * 0.8).sin(),
                dy: (t * 0.53).cos(),
                orientation: rig.orientation,
            };
            step_marbles(&mut marbles, &[input], &object, &params, &cfg, spawn.kill_y, &starts);

            // Player dragging the camera for the first 2 seconds, then idle
            // -- so the run covers both "player is steering" and "player has
            // let go and the solver is on its own".
            if t < 2.0 {
                let delta = Vec2::new(2.0, 0.6);
                if let Some(rotation) = CameraOrbit::drag_rotation(rig.orientation, delta) {
                    orbit.orientation = (rotation * orbit.orientation).normalize();
                }
            }

            let sdf = SceneSdf { object: &object, params: &params };
            let mut inp = SolveInput {
                marble_pos: marbles[0].pos,
                marble_radius: marbles[0].rad,
                intent: orbit.orientation,
                zoom: orbit.zoom,
                aspect: 16.0 / 9.0,
                target_fraction: POINTER_TARGET_FRACTION,
                dt,
                smart,
            };
            solve(&mut rig, &mut inp, &sdf);
            orbit.orientation = inp.intent;

            if let Some(prev) = prev_distance {
                travel += (rig.distance - prev).abs();
            }
            prev_distance = Some(rig.distance);

            // Skip the first few frames: the rig snaps into place on frame 0
            // and hasn't seen the geometry yet.
            if i < 10 {
                continue;
            }
            let d = rig.debug;
            min_vis = min_vis.min(d.visibility);
            sum_vis += d.visibility;
            if d.visibility < 0.5 {
                blocked += 1;
            }
            min_clear = min_clear.min(d.eye_clearance);
            min_frac = min_frac.min(d.screen_fraction);
            max_frac = max_frac.max(d.screen_fraction);
            sum_steps += d.steps as f32;
            if d.screen_fraction > 0.4 {
                too_close += 1;
            }
            max_dev = max_dev.max(d.deviation.to_degrees());
        }

        let counted = (frames - 10) as f32;
        Report {
            scene: kind.name(),
            frames,
            min_visibility: min_vis,
            mean_visibility: sum_vis / counted,
            frames_blocked: blocked,
            min_clearance: min_clear,
            min_screen_fraction: min_frac,
            max_screen_fraction: max_frac,
            mean_steps: sum_steps / counted,
            frames_too_close: too_close,
            max_deviation_deg: max_dev,
            distance_travel: travel,
        }
    }

    #[test]
    fn every_scene_keeps_the_marble_visible_and_the_eye_out_of_the_geometry() {
        let scenes = [
            SceneKind::Demo,
            SceneKind::ClassicOnly,
            SceneKind::MengerSponge,
            SceneKind::MengerSphere,
            SceneKind::MengerOscillatingSphere,
            SceneKind::HollowDonut,
            SceneKind::CubeSphereMorph,
        ];
        println!(
            "{:<26} {:>5} {:>7} {:>7} {:>7} {:>9} {:>9} {:>9} {:>6} {:>7} {:>7}",
            "scene", "mode", "minVis", "meanVis", "blocked", "minClear", "minSize", "maxSize", "close", "maxDev", "travel"
        );
        let mut failures = Vec::new();
        for kind in scenes {
            // The `smart: false` row is the A/B baseline (`?smartcam=0`):
            // the same framing rule, with every geometry-aware behavior
            // switched off. It is printed, never asserted on -- it is
            // *expected* to bury the eye in geometry and lose sight of the
            // marble, and that difference is the measurement.
            let base = probe(kind, false);
            println!(
                "{:<26} {:>5} {:>7.3} {:>7.3} {:>6}/{} {:>9.4} {:>9.3} {:>9.3} {:>6} {:>6.0}d {:>7.2}  steps={:.1}",
                base.scene,
                "off",
                base.min_visibility,
                base.mean_visibility,
                base.frames_blocked,
                base.frames,
                base.min_clearance,
                base.min_screen_fraction,
                base.max_screen_fraction,
                base.frames_too_close,
                base.max_deviation_deg,
                base.distance_travel,
                base.mean_steps,
            );
            let r = probe(kind, true);
            println!(
                "{:<26} {:>5} {:>7.3} {:>7.3} {:>6}/{} {:>9.4} {:>9.3} {:>9.3} {:>6} {:>6.0}d {:>7.2}  steps={:.1}",
                r.scene,
                "on",
                r.min_visibility,
                r.mean_visibility,
                r.frames_blocked,
                r.frames,
                r.min_clearance,
                r.min_screen_fraction,
                r.max_screen_fraction,
                r.frames_too_close,
                r.max_deviation_deg,
                r.distance_travel,
                r.mean_steps,
            );
            // The eye must never be inside geometry (I2). This is the one
            // hard invariant -- everything else here is a quality bar.
            if r.min_clearance <= 0.0 {
                failures.push(format!("{}: eye clearance fell to {}", r.scene, r.min_clearance));
            }
            // The marble must be mostly visible, most of the time. Not
            // "always": a marble that has thrust itself *into* a fractal
            // crevice genuinely cannot be seen from anywhere, and pretending
            // otherwise would be testing an impossibility.
            if r.mean_visibility < 0.6 {
                failures.push(format!("{}: mean visibility only {:.2}", r.scene, r.mean_visibility));
            }
            // On-screen size stays in a sane band around the 1/6 target --
            // except inside HollowDonut's closed tube, where it can't. The
            // tube's interior free radius is 0.85 and the marble's own is
            // 0.15, so the framing rule's 1.36 barely fits across the tube
            // at all, and the marble spends this run pressed against the
            // wall (its `de` sits at exactly its own radius) with the usable
            // directions sweeping around it as it circles. The camera keeps
            // it visible throughout -- which is the requirement -- but sits
            // closer than the framing rule wants, and no camera position
            // exists that would do better while the marble is hugging the
            // wall. Widening the FOV (`MIN_FOCAL_LENGTH`) is what recovers
            // most of the difference; this bound is what's left.
            let size_bound = if kind == SceneKind::HollowDonut { 0.95 } else { 0.35 };
            if r.max_screen_fraction > size_bound {
                failures.push(format!(
                    "{}: marble reached {:.2} of frame (bound {size_bound})",
                    r.scene, r.max_screen_fraction
                ));
            }
        }
        assert!(failures.is_empty(), "camera probe failures:\n{}", failures.join("\n"));
    }
}

/// The exact scenario reported from play: fly the marble head-on into a
/// sphere, with the camera directly behind it looking straight at the
/// sphere. Nothing enters the sightline at any point -- the sphere is
/// *beyond* the marble the whole time -- so a correct camera does not move
/// at all, and any movement is the bug.
#[cfg(test)]
mod head_on_into_a_sphere {
    use super::*;

    struct Ball {
        r: f32,
    }
    impl Sdf for Ball {
        fn de(&self, p: Vec3) -> f32 {
            p.length() - self.r
        }
    }

    /// `(worst deviation of the eye-to-marble distance from the framing
    /// distance, worst visibility, trace)`.
    fn fly_in(zoom: f32, aspect: f32, radius: f32, print: bool) -> (f32, f32) {
        let ball = Ball { r: 1.0 };
        let mut rig = CameraRig::default();
        // Identity orientation looks along -Z, so the eye sits at +Z from
        // the marble: camera behind, sphere dead ahead.
        let mut inp = SolveInput {
            marble_pos: Vec3::new(0.0, 0.0, 3.0),
            marble_radius: radius,
            intent: Quat::IDENTITY,
            zoom,
            aspect,
            target_fraction: POINTER_TARGET_FRACTION,
            dt: 1.0 / 60.0,
            smart: true,
        };
        // Settle first, so the run measures the approach and not the
        // initial snap.
        for _ in 0..120 {
            solve(&mut rig, &mut inp, &ball);
        }
        let settled = rig.distance;
        let (mut worst_move, mut worst_vis) = (0.0f32, 1.0f32);
        for i in 0..300 {
            // Fly straight at the centre at 2 units/s, stopping where the
            // marble's surface meets the sphere's.
            inp.marble_pos.z = (inp.marble_pos.z - 2.0 / 60.0).max(1.0 + radius);
            solve(&mut rig, &mut inp, &ball);
            worst_move = worst_move.max((rig.distance - settled).abs() / settled);
            worst_vis = worst_vis.min(rig.debug.visibility);
            if print && (i % 20 == 0 || (rig.distance - settled).abs() / settled > 0.02) {
                println!(
                    "i={i:3} z={:.3} de(marble)={:+.3} d={:.3}/{:.3} free={:.3} vis={:.2} size={:.3} steps={}",
                    inp.marble_pos.z,
                    ball.de(inp.marble_pos),
                    rig.distance,
                    rig.debug.desired_distance,
                    rig.debug.free_distance,
                    rig.debug.visibility,
                    rig.debug.screen_fraction,
                    rig.debug.steps
                );
            }
        }
        (worst_move, worst_vis)
    }

    #[test]
    fn the_camera_does_not_move_at_all() {
        // The reporter's configuration: `cube_sphere_morph`'s marble radius
        // and zoom, on a phone in portrait.
        let (worst_move, worst_vis) = fly_in(3.3, 384.0 / 694.0, 0.15, true);
        assert!(
            worst_move < 0.01,
            "camera distance moved {:.1}% while flying head-on into a sphere",
            worst_move * 100.0
        );
        assert_eq!(worst_vis, 1.0, "nothing ever came between the camera and the marble");
    }

    #[test]
    fn and_the_same_at_every_zoom_and_aspect() {
        for zoom in [0.5f32, 1.0, 3.3, 6.0] {
            for aspect in [384.0 / 694.0, 1.0, 16.0 / 9.0] {
                let (worst_move, worst_vis) = fly_in(zoom, aspect, 0.15, false);
                assert!(
                    worst_move < 0.01,
                    "zoom {zoom} aspect {aspect:.2}: camera moved {:.1}%",
                    worst_move * 100.0
                );
                assert_eq!(worst_vis, 1.0, "zoom {zoom} aspect {aspect:.2}: view was obstructed");
            }
        }
    }
}

/// The reported scene itself, animated: `cube_sphere_morph`'s shape sweeps
/// between a cube and an inscribed sphere every 12 seconds, so a marble
/// resting against a face can be *engulfed* as the geometry grows back out
/// past it. That is the one situation in which no camera position is
/// correct, and it is worth knowing exactly how the solver degrades.
#[cfg(test)]
mod morphing_geometry {
    use super::*;
    use crate::render::{build_scene, SceneKind};
    use marble_csg::physics::{step_marbles, Marble, PhysicsConfig, PlayerInput};
    use marble_csg::visibility::SceneSdf;
    use marble_csg::Params;

    #[test]
    fn a_marble_swallowed_by_growing_geometry_does_not_slam_the_camera() {
        let kind = SceneKind::CubeSphereMorph;
        let mut params = Params::new();
        let (object, _handles, animations) = build_scene(kind, &mut params);
        let spawn = kind.spawn_params();
        let cfg = PhysicsConfig::default();
        let mut marbles = vec![Marble::spawn(spawn.start, spawn.rad)];
        let starts = vec![spawn.start];

        let mut rig = CameraRig::default();
        let mut orbit = CameraOrbit { orientation: Quat::IDENTITY, zoom: 3.3 };
        let dt = 1.0 / 60.0;
        let mut worst_size: f32 = 0.0;
        let mut frames_collapsed = 0;

        // A full morph period (12s at 60Hz), with the marble thrusting
        // straight at the centre the whole time -- exactly "fly into it and
        // hold".
        for tick in 0..720u64 {
            for (handle, expr) in &animations {
                params.set_scalar(*handle, expr.eval(tick));
            }
            let toward_centre = -marbles[0].pos.normalize_or_zero();
            let (fwd, right) = (rig.orientation * Vec3::NEG_Z, rig.orientation * Vec3::X);
            let input = PlayerInput {
                dx: toward_centre.dot(right),
                dy: toward_centre.dot(fwd),
                orientation: rig.orientation,
            };
            step_marbles(&mut marbles, &[input], &object, &params, &cfg, spawn.kill_y, &starts);

            let sdf = SceneSdf { object: &object, params: &params };
            let mut inp = SolveInput {
                marble_pos: marbles[0].pos,
                marble_radius: marbles[0].rad,
                intent: orbit.orientation,
                zoom: orbit.zoom,
                aspect: 384.0 / 694.0,
                target_fraction: POINTER_TARGET_FRACTION,
                dt,
                smart: true,
            };
            solve(&mut rig, &mut inp, &sdf);
            orbit.orientation = inp.intent;

            if tick > 60 {
                worst_size = worst_size.max(rig.debug.screen_fraction);
                if rig.distance <= marbles[0].rad * MIN_DISTANCE_MARBLE_RADII * 1.01 {
                    frames_collapsed += 1;
                }
                if tick % 60 == 0 || rig.debug.screen_fraction > 0.5 {
                    println!(
                        "tick={tick:3} de(marble)={:+.3} d={:.3}/{:.3} free={:.3} vis={:.2} size={:.3} steps={}",
                        sdf.de(marbles[0].pos),
                        rig.distance,
                        rig.debug.desired_distance,
                        rig.debug.free_distance,
                        rig.debug.visibility,
                        rig.debug.screen_fraction,
                        rig.debug.steps
                    );
                }
            }
        }
        println!("worst size {worst_size:.3}, {frames_collapsed} frames at the distance floor");
        assert!(
            frames_collapsed == 0,
            "camera slammed to its minimum distance on {frames_collapsed} frames"
        );
        assert!(worst_size < 0.6, "marble reached {worst_size:.2} of frame");
    }
}
