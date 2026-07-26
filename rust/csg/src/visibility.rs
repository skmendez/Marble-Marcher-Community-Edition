//! Line-of-sight and clearance queries against a distance field — the
//! geometry half of the smart camera (`rust/CAMERA.md` §4.3).
//!
//! Everything a third-person camera normally needs a physics engine's
//! raycasts for (occlusion probes, swept-sphere "camera radius" collision,
//! whisker rays) is a *sphere trace* here, because this game's world is a
//! distance field that the CPU can evaluate directly ([`crate::Object::de`],
//! the same tree the fragment shader marches). That's not merely a
//! convenient substitution: a sphere trace returns a **continuous**
//! clearance, where a raycast returns a binary hit. Against fractal geometry
//! (thin Menger struts crossing the sightline for one frame at a time) a
//! binary occlusion test drives visible camera jitter no amount of damping
//! fully hides; a continuous one can be used as a *gain* on corrective
//! motion instead, so a barely-clipped shot produces a barely-perceptible
//! correction.
//!
//! One [`sweep`] call answers three questions at once, which is why the
//! camera solver only needs one march per frame in the common case:
//!
//!  1. **How far out can the camera sit?** ([`Sweep::free_distance`]) — the
//!     largest distance a ball of radius `camera_radius` can reach along the
//!     ray without touching geometry. This is the classic "pull camera
//!     forward" deocclusion strategy, computed as an exact swept-sphere test
//!     rather than approximated by a zero-thickness ray (the standard
//!     failure mode: a thin probe reports clear while the near plane is
//!     already inside the wall).
//!  2. **How much of the target can be seen?** ([`Sweep::visibility`]) — a
//!     continuous `[0, 1]` measure, `1` meaning a full target-width of
//!     clearance all along the sightline.
//!  3. **What is blocking it, and which way should the camera slide to get
//!     around?** ([`Sweep::blocker`]).
//!
//! The distance fields in this crate *underestimate* true distance (the
//! marching invariant the whole renderer relies on; `Object::Onion`/`Morph`
//! document their Lipschitz soundness explicitly). Underestimating is the
//! safe direction for a camera: it can pull in slightly more than strictly
//! necessary, never less.

use glam::{Vec3, Vec4};

use crate::{Object, Params};

/// A distance field the camera can query. Implemented for the real scene
/// tree by [`SceneSdf`]; tests implement it with analytic worlds (a plane, a
/// pillar, a slab with a hole) so the camera solver is testable without
/// building a fractal.
pub trait Sdf {
    /// Distance estimate at `p` — non-negative outside the surface,
    /// negative inside, never an *over*estimate of the true distance.
    fn de(&self, p: Vec3) -> f32;

    /// Gradient of [`Self::de`] at `p`, via the standard 4-tap tetrahedral
    /// stencil. `eps` should be small relative to the local feature size but
    /// large enough not to be swamped by `f32` noise -- callers pass a
    /// fraction of the camera radius.
    ///
    /// **The magnitude is meaningful and is the point of returning it.** A
    /// true distance field has `|grad| == 1` wherever it is differentiable;
    /// the fields in this crate only ever *under*estimate, and where they
    /// underestimate badly they do it by going flat (`Object::Morph`'s doc
    /// makes the `|grad| < 1` promise explicit). So the magnitude is a
    /// direct, local, four-`de` answer to "can I trust a linear model of
    /// this field here?" -- which is what [`sweep`] needs to decide whether
    /// a march that ran out of budget learned anything.
    ///
    /// A default rather than a required method so analytic test worlds get
    /// it for free; [`SceneSdf`] does not override it either (the exact
    /// `Object::nearest_point` alternative costs ~40% more than these four
    /// `de` calls on the demo scene and is no more accurate for this use).
    fn gradient(&self, p: Vec3, eps: f32) -> Vec3 {
        // Tetrahedral 4-tap gradient (the standard ray-marching normal):
        // four samples at alternating corners of a tetrahedron, weighted by
        // their own offset directions. The `4 * eps` normalisation is what
        // makes this the gradient rather than merely its direction: for a
        // linear field `de(p) = p . a`, the weighted sum comes to `4 eps a`.
        const K: [Vec3; 4] = [
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];
        let mut g = Vec3::ZERO;
        for k in K {
            g += k * self.de(p + k * eps);
        }
        g / (4.0 * eps.max(1e-9))
    }

    /// Outward direction at `p` (away from the nearest surface) — the
    /// normalised [`Self::gradient`].
    fn outward(&self, p: Vec3, eps: f32) -> Vec3 {
        self.gradient(p, eps).normalize_or_zero()
    }
}

/// The live scene tree as an [`Sdf`] — the adapter between the camera solver
/// and `Object`/`Params`. Borrowed (not owned) so a caller can build one per
/// frame from `RollbackSim::scene()` with no clone.
pub struct SceneSdf<'a> {
    pub object: &'a Object,
    pub params: &'a Params,
}

impl Sdf for SceneSdf<'_> {
    fn de(&self, p: Vec3) -> f32 {
        self.object.de(Vec4::new(p.x, p.y, p.z, 1.0), self.params)
    }
}

/// Tuning for one [`sweep`] call.
#[derive(Clone, Copy, Debug)]
pub struct SweepConfig {
    /// The camera's own collision radius: geometry closer than this to the
    /// ray counts as blocking, so the eye keeps a margin from every surface
    /// instead of grazing it (Cinemachine calls this "camera radius"; here
    /// it's also what keeps the eye out of the near-surface region where
    /// this renderer's normal estimation degenerates into speckle -- see
    /// the per-scene camera-distance comments in `app/src/render.rs`).
    pub camera_radius: f32,
    /// Radius of the thing being looked at (the marble). Sets the angular
    /// scale [`Sweep::visibility`] is measured against: an obstruction
    /// matters in proportion to how much of *this* it covers.
    pub target_radius: f32,
    /// Where the camera's own world begins: the march ignores everything
    /// nearer to the target than this, for both questions it answers.
    ///
    /// This is Cinemachine's "Minimum Distance From Target", and it is not
    /// an optimisation -- it is the difference between a camera that works
    /// and one that collapses the moment the target touches anything. The
    /// camera can never sit closer than this, so geometry inside that radius
    /// is not in its way; it is the surface the target is resting against.
    /// Without it, a marble rolling onto a floor puts a wall inside the
    /// probe's own first sample, the swept test reports "blocked, free
    /// distance zero", and the camera slams to its minimum distance with the
    /// marble filling the screen -- observed in play, reproduced in
    /// `smart_camera`'s `a_marble_resting_on_a_surface_does_not_collapse_
    /// the_shot`.
    pub min_camera_distance: f32,
    /// Hard cap on `de` evaluations. Reaching it returns whatever was found
    /// so far with `exhausted = true` rather than reporting a clear view --
    /// a march that runs out of budget is grazing something, so treating it
    /// as clear would be the unsafe direction.
    pub max_steps: u32,
}

/// How close to `1` the local `|grad de|` must be for a stalled march to
/// finish the job by extrapolation rather than give up (see
/// [`Sweep::exhausted`]).
///
/// This is what distinguishes the two ways [`SweepConfig::max_steps`] gets
/// spent, which call for opposite responses:
///
///  - **Converging onto a surface.** The classic grazing ray: each step is
///    `h - camera_radius`, so a ray closing on a wall at a shallow angle
///    shrinks its steps geometrically and creeps. This is not a rare corner
///    — over a floor a marble is resting on it is *every* shallow angle, and
///    at a 3-unit framing distance it needs 60-130 steps where the budget is
///    40. But the field there is a real distance field, so where the surface
///    is can simply be *computed*: one gradient, and the touchdown point
///    follows.
///  - **A slow field.** An underestimating `de` (`Object::Morph` is the
///    worst offender in this crate) makes every step short even in open air.
///    Here nothing can be concluded: the ray may have kilometres of clear
///    space ahead of it, and inventing a wall dives the camera at an
///    obstruction that does not exist (`rust/CAMERA.md` §13).
///
/// A true distance field has `|grad| == 1`; a field that underestimates
/// badly does it by going flat, so the magnitude separates the two directly
/// and locally. `0.5` is deliberately permissive — the cost of being wrong
/// in the strict direction is only that a stall falls back to reporting
/// nothing, which is what it did before extrapolation existed.
const TRUSTED_GRADIENT: f32 = 0.5;

/// Below this rate of approach, a stalled march does not extrapolate: the
/// ray is running parallel to (or away from) the surface and the touchdown
/// point is both enormously far away and enormously sensitive to the
/// gradient estimate. Reporting nothing is the honest answer there.
const MIN_CLOSING_RATE: f32 = 0.02;

/// Halvings spent recovering when an extrapolated touchdown turns out to be
/// inside geometry. Each one is a `de`; four gets within 6% of the crossing,
/// which is well under the margin `camera_radius` is already reserving.
const EXTRAPOLATION_BISECTIONS: u32 = 4;

/// What one march along the sightline found.
#[derive(Clone, Copy, Debug)]
pub struct Sweep {
    /// How far a ball of radius `camera_radius` can travel along the ray
    /// before touching geometry, capped at the requested `max_dist`.
    pub free_distance: f32,
    /// Fraction of the target's disc that is unobstructed *from the nearest
    /// position the camera can actually reach along this ray*, in `[0, 1]`:
    /// `1.0` = a full target-width of clearance the whole way, `0.0` =
    /// blocked. See [`sweep`]'s doc for the derivation, and for why the
    /// measurement stops at [`Self::free_distance`] rather than running out
    /// to the requested `max_dist`.
    pub visibility: f32,
    /// Where the tightest constriction was (world space), if the view was
    /// less than fully clear. The camera slides *away* from this to open
    /// the shot up (`smart_camera::solve`).
    pub blocker: Option<Vec3>,
    /// `de` evaluations actually spent -- surfaced so the debug overlay can
    /// show what this costs on real geometry rather than guessing.
    pub steps: u32,
    /// The step budget ran out *without establishing anything*:
    /// [`Self::free_distance`] is where the march happened to stop, not a
    /// bound on where the camera can go, and callers must not treat it as an
    /// obstruction (`smart_camera::usable_free_distance`).
    ///
    /// Note this is narrower than "the budget ran out". A march that stalls
    /// while converging onto a surface has spent its budget *and* produced a
    /// usable bound, and reports `false` here -- see
    /// [`GRAZING_STALL_RADII`], which is the distinction. Conflating the two
    /// is what let a camera be dragged through the shallow angles over a
    /// floor (where every march grazes, and every march therefore stalled)
    /// as if the floor were not there at all.
    pub exhausted: bool,
}

/// Marches from `origin` along `dir` (unit) up to `max_dist`, reporting
/// clearance and visibility (see [`Sweep`]).
///
/// Marching *outward from the target* rather than inward from the eye is
/// deliberate, and buys three things:
///
///  - `free_distance` comes out directly as the deocclusion answer ("how far
///    back can the camera sit and still see this?"), rather than needing a
///    second query.
///  - An eye that has somehow ended up inside geometry (a scene param edit,
///    a teleport) still produces a meaningful result, where an eye-origin
///    march would immediately terminate at step 0 with nothing to say.
///  - The near-target steps, where the distance field is smallest and the
///    march therefore steps slowest, are the ones that matter most for
///    framing -- so the step budget gets spent where it counts.
///
/// **The visibility measure.** This is Iñigo Quilez's sphere-traced
/// soft-shadow ratio (`res = min(res, k·h/t)`), with `k` chosen so the
/// number means something physical rather than being a softness knob: treat
/// the target as an area light and ask what fraction of its disc survives.
/// An obstruction leaving clearance `h` at distance `max_dist - t` from the
/// eye subtends `h / (max_dist - t)` there; the target itself subtends
/// `target_radius / max_dist`. Their ratio,
///
/// ```text
/// visibility = min over the march of  h · max_dist / (target_radius · (max_dist - t))
/// ```
///
/// is therefore "how many target-radii of clearance does this constriction
/// leave, measured in the target's own angular units" — clamped to `[0, 1]`
/// because more clearance than the target's own width isn't more visible.
/// Note it correctly ignores geometry immediately in front of the eye
/// (`t → max_dist`): a gap a hand's width from your face doesn't block your
/// view, the same gap across the room does.
pub fn sweep(sdf: &impl Sdf, origin: Vec3, dir: Vec3, max_dist: f32, cfg: SweepConfig) -> Sweep {
    // Everything nearer than this belongs to the target, not to the camera
    // (`SweepConfig::min_camera_distance`).
    let start = cfg.min_camera_distance.max(cfg.target_radius).min(max_dist);
    // Every division below is guarded by this: `max_dist` can legitimately
    // be tiny (a marble wedged in a crevice), and the visibility ratio's
    // denominator vanishes at the eye by construction.
    let eps = 1e-6_f32.max(max_dist * 1e-4);
    // Close enough to the surface to call it a hit and stop marching (a
    // plain sphere trace only converges *towards* a surface, never onto
    // it). Scaled to the *camera's* own radius rather than to `max_dist`:
    // the camera radius is by definition the smallest clearance anything
    // here cares about, whereas a `max_dist`-relative epsilon reads a
    // long-range grazing pass (tiny clearance, huge distance) as a head-on
    // hit. The step budget, not this epsilon, is what bounds a march that
    // crawls without ever quite touching.
    let surface_eps = (cfg.camera_radius * 1e-3).max(1e-7);

    // ---- pass 1: how far along this ray can the camera get? ----
    //
    // Steps by `h - camera_radius`, which is what makes the swept-ball test
    // sound *between* samples and not merely at them: a plain `t += h` step
    // only establishes clearance at the sample points, and a 1-Lipschitz
    // field is free to dip to roughly `h/2` in between two of them -- so a
    // ball of radius `q` gets reported clear along a ray that actually clips
    // a wall, which is how an eye ends up inside geometry the sweep swore
    // was free (observed in HollowDonut's tube before this). Stepping by
    // `h - q` bounds the true distance below by `q` over the whole traversed
    // segment, by the same Lipschitz argument.
    let mut t = start;
    let mut steps = 0;
    let mut exhausted = false;
    let mut free_distance = max_dist;
    loop {
        let h = sdf.de(origin + dir * t);
        steps += 1;
        if h > cfg.camera_radius && steps >= cfg.max_steps {
            // Out of budget short of the goal, with the ball still clear.
            // Marching further is not an option, but *computing* the rest
            // may be: the sample gives clearance `h` here, and a gradient
            // gives the direction the surface lies in and how fast the ray
            // is closing on it. Where the field is a real distance field
            // that is a complete local answer -- exact for a plane, which is
            // what a grazing stall usually is -- and it costs four `de`
            // calls, paid only on a march that has already given up.
            //
            // Where it is *not* (a field that underestimates by going flat)
            // both tests below fail and this reports `exhausted`, i.e. "no
            // information", which is what the whole stall case did before.
            // See `TRUSTED_GRADIENT` for why these are opposite failures and
            // why they must not be conflated.
            let g = sdf.gradient(origin + dir * t, cfg.camera_radius * 0.25);
            let magnitude = g.length();
            let closing = if magnitude > 1e-9 { -dir.dot(g / magnitude) } else { 0.0 };
            if magnitude >= TRUSTED_GRADIENT && closing >= MIN_CLOSING_RATE {
                // The linear model is a *prediction*, so check it before
                // believing it: one `de` at the predicted touchdown. A
                // single flat surface -- the case this exists for -- passes
                // first time. What fails is a scene where the nearest
                // surface changes identity along the ray (marching down a
                // corridor between two rows of pillars, the field switches
                // from one pillar to the next), and there the prediction can
                // land *inside* geometry, which is the one outcome this
                // whole module exists to prevent.
                //
                // On a miss, bisect back toward the last sample known to be
                // clear. Four halvings put the answer within 6% of the
                // crossing and, because only verified-clear midpoints are
                // ever adopted, always on the safe side of it.
                let predicted = (t + (h - cfg.camera_radius) / closing).min(max_dist);
                if sdf.de(origin + dir * predicted) >= cfg.camera_radius {
                    steps += 1;
                    free_distance = predicted;
                } else {
                    let (mut clear, mut blocked_at) = (t, predicted);
                    for _ in 0..EXTRAPOLATION_BISECTIONS {
                        let mid = 0.5 * (clear + blocked_at);
                        if sdf.de(origin + dir * mid) >= cfg.camera_radius {
                            clear = mid;
                        } else {
                            blocked_at = mid;
                        }
                    }
                    steps += 1 + EXTRAPOLATION_BISECTIONS;
                    free_distance = clear;
                }
            } else {
                exhausted = true;
                free_distance = t;
            }
            break;
        }
        if h <= cfg.camera_radius {
            // The ball's surface touches down `camera_radius - h` before
            // this sample (negative when already overlapping, hence the
            // clamp).
            //
            // Backing off *along the ray* by the shortfall is exact for a
            // head-on approach and under-corrects by `1/sin(incidence)` on an
            // oblique one, so a grazing hit can land up to `camera_radius`
            // deep into the margin. That is the margin's job -- the eye is
            // still outside the surface, which is what the margin was
            // reserving room for -- and the alternative costs a gradient
            // evaluation on every march. `smart_camera`'s step-8 backstop
            // reads the eye's own clearance and corrects what is left.
            free_distance = (t - (cfg.camera_radius - h)).max(0.0);
            break;
        }
        t += (h - cfg.camera_radius).max(eps);
        if t >= max_dist {
            break;
        }
    }
    let free_distance = free_distance.clamp(0.0, max_dist);

    // ---- pass 2: from there, how much of the target is visible? ----
    //
    // A separate pass because the answer depends on where the camera ends up
    // (`free_distance`), which pass 1 is what determines. Two things follow
    // from measuring over `[start, free_distance]` with the eye at
    // `free_distance`, and both matter:
    //
    //  - Geometry *beyond* where the camera can get to is not an
    //    obstruction. A camera with its back to a wall sees the target
    //    perfectly well; a single-pass version that marched on to `max_dist`
    //    reported every ordinary corridor as blocked, and had the solver
    //    sliding around for no reason.
    //  - The perspective term uses the real eye distance. Sharing one pass
    //    means using `max_dist` for it, which is wrong by exactly the amount
    //    the camera got pulled in -- i.e. wrong precisely when it matters.
    //
    // This pass cannot hit a surface (pass 1 established clearance of at
    // least `camera_radius` along the whole stretch) and steps by the full
    // `h >= camera_radius`, so it is bounded by `free_distance /
    // camera_radius` steps and is typically a handful.
    let mut visibility = 1.0f32;
    let mut blocker = None;
    if free_distance <= start + eps {
        // Nowhere to stand at all: the camera is jammed against the target.
        visibility = 0.0;
    } else {
        let mut t = start;
        while t < free_distance && steps < cfg.max_steps * 2 {
            let p = origin + dir * t;
            let h = sdf.de(p);
            steps += 1;
            if h <= surface_eps {
                visibility = 0.0;
                blocker = Some(p);
                break;
            }
            // Angular clearance at this depth, in units of the target's own
            // angular radius (see this fn's doc).
            let from_eye = (free_distance - t).max(eps);
            let ratio = h * free_distance / (cfg.target_radius.max(eps) * from_eye);
            if ratio < visibility {
                visibility = ratio;
                blocker = Some(p);
            }
            t += h.max(eps);
        }
    }

    Sweep {
        free_distance,
        visibility: visibility.clamp(0.0, 1.0),
        blocker: if visibility >= 1.0 { None } else { blocker },
        steps,
        exhausted,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Infinite plane at `x = plane_x`, solid for `x > plane_x`.
    struct Wall {
        plane_x: f32,
    }
    impl Sdf for Wall {
        fn de(&self, p: Vec3) -> f32 {
            self.plane_x - p.x
        }
    }

    /// Empty space.
    struct Empty;
    impl Sdf for Empty {
        fn de(&self, _p: Vec3) -> f32 {
            1e9
        }
    }

    /// Infinite cylinder of radius `r` along Y, centered at `(cx, _, cz)`.
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

    fn cfg(camera_radius: f32, target_radius: f32) -> SweepConfig {
        // Tests that predate `min_camera_distance` keep the old behavior by
        // setting it to the target's own radius (i.e. "the camera's world
        // starts at the target's surface").
        SweepConfig {
            camera_radius,
            target_radius,
            min_camera_distance: target_radius,
            max_steps: 64,
        }
    }

    #[test]
    fn empty_space_is_fully_visible_and_fully_free() {
        let s = sweep(&Empty, Vec3::ZERO, Vec3::X, 10.0, cfg(0.1, 0.2));
        assert_eq!(s.free_distance, 10.0);
        assert_eq!(s.visibility, 1.0);
        assert!(s.blocker.is_none());
        assert!(!s.exhausted);
        // One giant step should cover it: the whole point of sphere tracing.
        assert!(s.steps <= 2, "empty space took {} steps", s.steps);
    }

    #[test]
    fn a_wall_caps_free_distance_but_is_not_an_obstruction() {
        // Wall at x = 5, marching from the origin straight at it with a
        // camera ball of radius 0.5: the ball's *center* can reach 4.5.
        let s = sweep(&Wall { plane_x: 5.0 }, Vec3::ZERO, Vec3::X, 10.0, cfg(0.5, 0.2));
        assert!(
            (s.free_distance - 4.5).abs() < 0.05,
            "expected the ball centre to stop 0.5 short of the wall, got {}",
            s.free_distance
        );
        // ...and from 4.5, with nothing between it and the target, the view
        // is clear. A wall the camera has its *back* to is a limit on where
        // it can stand, not something in the way -- the two call for
        // completely different responses (dolly in vs slide around), which
        // is exactly why they are separate numbers here.
        assert_eq!(s.visibility, 1.0, "a wall behind the camera obstructs nothing");
    }

    #[test]
    fn visibility_is_continuous_not_binary_as_a_pillar_slides_across() {
        // The property the whole design rests on: an obstruction easing into
        // the sightline must produce intermediate values, not a 1 -> 0 step.
        // Pillar of radius 0.5 closing in on a sightline of length 10.
        // Swept over the regime where the pillar clips the *sightline* but
        // still leaves the camera ball (radius 0.02) room to pass: gaps from
        // 0.20 down to 0.03. Closer than that and the pillar stops being an
        // occluder and starts being a wall -- the camera can no longer get
        // past it, so `free_distance` collapses and visibility measures the
        // (clear) stretch in front of it instead. Both regimes are checked;
        // conflating them is what the two separate numbers exist to avoid.
        let mut seen_partial = 0;
        let mut last = 1.0;
        for i in 0..=34 {
            let gap = 0.20 - i as f32 * 0.005;
            let s = sweep(
                &Pillar { cx: 5.0, cz: 0.5 + gap, r: 0.5 },
                Vec3::ZERO,
                Vec3::X,
                10.0,
                cfg(0.02, 0.3),
            );
            assert!(
                s.visibility <= last + 1e-3,
                "visibility must fall monotonically as the pillar closes in (gap {gap})"
            );
            assert!(
                s.free_distance > 9.0,
                "the camera can still pass a {gap}-wide gap, so nothing should cap its distance"
            );
            if s.visibility > 0.02 && s.visibility < 0.98 {
                seen_partial += 1;
            }
            last = s.visibility;
        }
        assert!(
            seen_partial >= 5,
            "expected a range of partial-visibility values as the pillar crosses, got {seen_partial}"
        );
        assert!(last < 0.2, "a nearly-touching pillar should be close to fully blocking, got {last}");

        // Other regime: pillar dead centre. The ball cannot pass, so the
        // camera stops in front of it -- with a clear view from there.
        let blocking = sweep(
            &Pillar { cx: 5.0, cz: 0.0, r: 0.5 },
            Vec3::ZERO,
            Vec3::X,
            10.0,
            cfg(0.02, 0.3),
        );
        assert!(blocking.free_distance < 4.6, "expected the camera to stop in front of the pillar");
        assert_eq!(blocking.visibility, 1.0);
    }

    #[test]
    fn clearance_near_the_eye_matters_less_than_the_same_clearance_near_the_target() {
        // Two identical pillars leaving the same absolute gap, one just off
        // the target, one just off the eye. The near-eye one should score
        // far higher (a gap subtends a bigger angle the closer it is).
        let near_target = sweep(
            &Pillar { cx: 1.0, cz: 0.6, r: 0.5 },
            Vec3::ZERO,
            Vec3::X,
            10.0,
            cfg(0.02, 0.3),
        );
        let near_eye = sweep(
            &Pillar { cx: 9.0, cz: 0.6, r: 0.5 },
            Vec3::ZERO,
            Vec3::X,
            10.0,
            cfg(0.02, 0.3),
        );
        assert!(
            near_eye.visibility > 0.98,
            "a gap right in front of the eye barely obstructs, got {}",
            near_eye.visibility
        );
        assert!(
            near_target.visibility < 0.6,
            "the same gap next to the target obstructs plenty, got {}",
            near_target.visibility
        );
    }

    #[test]
    fn blocker_points_at_the_actual_obstruction() {
        // Gap of 0.06: wide enough for the 0.02 camera ball to pass (so this
        // is an occluder, not a wall), narrow enough to clip the target's
        // silhouette.
        let s = sweep(
            &Pillar { cx: 4.0, cz: 0.56, r: 0.5 },
            Vec3::ZERO,
            Vec3::X,
            10.0,
            cfg(0.02, 0.3),
        );
        let b = s.blocker.expect("a partially blocked view must report where");
        assert!((b.x - 4.0).abs() < 1.0, "blocker should sit near the pillar, got {b:?}");
    }

    #[test]
    fn geometry_nearer_than_the_camera_can_get_is_not_an_obstruction() {
        // A target resting on a floor, viewed from 45 degrees above: the
        // floor is within a target-radius of the sightline's start, so a
        // march beginning at the target's own surface reads it as blocking
        // everything. Starting where the camera's world actually begins is
        // what makes this the non-event it should be.
        struct Floor;
        impl Sdf for Floor {
            fn de(&self, p: Vec3) -> f32 {
                p.y
            }
        }
        let r = 0.15;
        let origin = Vec3::new(0.0, r, 0.0); // resting on the floor
        let dir = Vec3::new(0.0, 1.0, 1.0).normalize();
        let blind = SweepConfig {
            camera_radius: 0.3,
            target_radius: r,
            min_camera_distance: r,
            max_steps: 32,
        };
        let collapsed = sweep(&Floor, origin, dir, 8.0, blind);
        assert!(
            collapsed.free_distance < 0.2,
            "test setup: an over-fat probe starting at the surface should collapse to near nothing \
             (against the 8.0 that is actually available), got {}",
            collapsed.free_distance
        );

        let sane = SweepConfig { camera_radius: 0.35 * r, min_camera_distance: 1.5 * r, ..blind };
        let s = sweep(&Floor, origin, dir, 8.0, sane);
        assert_eq!(s.free_distance, 8.0, "nothing is actually in the way of this shot");
        assert_eq!(s.visibility, 1.0);
    }

    #[test]
    fn outward_gradient_points_away_from_the_surface() {
        let wall = Wall { plane_x: 5.0 };
        let n = wall.outward(Vec3::new(4.0, 0.0, 0.0), 0.01);
        assert!(n.distance(Vec3::NEG_X) < 1e-2, "expected -X (away from the wall), got {n:?}");
    }

    #[test]
    fn exhausting_the_step_budget_never_reports_a_clear_view() {
        // A grazing march along a wall: `de` stays tiny, so every step is
        // tiny and the budget runs out. This must not read as "clear".
        struct Grazing;
        impl Sdf for Grazing {
            fn de(&self, p: Vec3) -> f32 {
                // A surface that closes in asymptotically as x grows.
                0.02 + 0.1 / (1.0 + p.x.max(0.0))
            }
        }
        let s = sweep(&Grazing, Vec3::ZERO, Vec3::X, 1000.0, SweepConfig {
            camera_radius: 0.01,
            target_radius: 0.1,
            min_camera_distance: 0.1,
            max_steps: 24,
        });
        assert!(s.exhausted);
        assert!(s.free_distance < 1000.0, "an exhausted march must not claim the full distance");
        // Pass 1's own budget, plus whatever pass 2 spent measuring
        // visibility over the (short) stretch pass 1 established.
        assert!(s.steps >= 24 && s.steps <= 48, "steps = {}", s.steps);
    }

    #[test]
    fn a_ray_grazing_a_floor_reports_a_usable_bound_at_every_angle() {
        // The shallow angles over a plane are exactly where the swept march
        // creeps: each step is `h - camera_radius`, and a ray closing on the
        // floor at 5 degrees shrinks its steps by ~9% each time, so it spends
        // the whole budget without formally touching down.
        //
        // Before `GRAZING_STALL_RADII` those angles came back `exhausted`,
        // i.e. "no information" -- and since the camera solver (correctly)
        // refuses to be constrained by a march that established nothing, a
        // marble resting on a plane had a band of downward angles, right
        // where the floor starts to matter, in which the floor was invisible
        // to it. Dragging through that band put the camera at a third of its
        // framing distance with the marble filling the screen, which is
        // `smart_camera`'s `dragging_the_camera_into_a_floor_...` test and
        // was reported from play.
        //
        // What this asserts is the property the solver actually needs: at
        // *every* angle, the answer is both usable and close to the truth.
        struct Floor;
        impl Sdf for Floor {
            fn de(&self, p: Vec3) -> f32 {
                p.y
            }
        }
        let r = 0.15;
        let origin = Vec3::new(0.0, r, 0.0); // resting on the floor
        let camera_radius = 0.35 * r;
        let c = SweepConfig {
            camera_radius,
            target_radius: r,
            min_camera_distance: 1.5 * r,
            max_steps: 40,
        };
        let max_dist = 1.46; // what the framing rule asks for at these sizes
        for i in 1..=40 {
            let theta = i as f32 * 0.02; // 1.1 to 46 degrees below horizontal
            let dir = Vec3::new(theta.cos(), -theta.sin(), 0.0);
            let s = sweep(&Floor, origin, dir, max_dist, c);
            // The eye's centre must stay `camera_radius` above the plane.
            let exact = ((r - camera_radius) / theta.sin()).min(max_dist);
            assert!(
                !s.exhausted,
                "theta {theta}: a march that has converged onto the floor knows where the floor is"
            );
            // Never past the floor itself. The bound may sit up to a
            // camera-radius into the margin the sweep reserves, because the
            // touchdown correction backs off along the ray rather than along
            // the surface normal (see `sweep`); it may never sit past the
            // surface that margin was protecting.
            assert!(
                (origin + dir * s.free_distance).y >= 0.0,
                "theta {theta}: {} puts the eye through the floor",
                s.free_distance
            );
            assert!(
                s.free_distance <= exact + camera_radius,
                "theta {theta}: claimed {} against the floor's {exact}",
                s.free_distance
            );
            assert!(
                s.free_distance > exact * 0.9,
                "theta {theta}: {} is far short of the {exact} actually available",
                s.free_distance
            );
        }
    }

    #[test]
    fn a_slow_field_in_open_air_still_reports_no_information() {
        // The other half of `GRAZING_STALL_RADII`, and the reason it is a
        // threshold rather than "a stall is always a wall": an
        // underestimating field (this crate's `Object::Morph`) makes every
        // step short even where there is nothing anywhere near. Calling the
        // stall point a wall here is what dives the camera at obstructions
        // that do not exist.
        struct Slow;
        impl Sdf for Slow {
            fn de(&self, _p: Vec3) -> f32 {
                // Sound (it never overestimates) but 50x too small: open
                // space that the march has to crawl across.
                0.02
            }
        }
        let s = sweep(&Slow, Vec3::ZERO, Vec3::X, 100.0, SweepConfig {
            camera_radius: 0.005,
            target_radius: 0.1,
            min_camera_distance: 0.1,
            max_steps: 40,
        });
        assert!(s.exhausted, "a march with room to spare at every sample has found no wall");
        assert!(s.free_distance < 100.0);
    }

    #[test]
    fn works_against_the_real_demo_scene() {
        use crate::scenes::{beware_of_bumps, demo_scene, set_fractal_params};
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
        let sdf = SceneSdf { object: &object, params: &params };
        let origin = beware_of_bumps::START;
        let r = beware_of_bumps::MARBLE_RAD;

        // Straight up from the marble's start is open air (the marble falls
        // to the surface from here, so above it must be free); straight down
        // is the surface it lands on.
        let up = sweep(&sdf, origin, Vec3::Y, 0.2, cfg(r, r));
        let down = sweep(&sdf, origin, Vec3::NEG_Y, 0.2, cfg(r, r));
        assert!(up.free_distance > down.free_distance, "up {up:?} should be freer than down {down:?}");
        assert!(up.steps <= 64 && down.steps <= 64);
    }
}
