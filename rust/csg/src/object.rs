//! M2: the `Object` enum — primitives, CSG combiners, `Fractal`.
//! See rust/DESIGN.md §3–4 and the C++ sources in src/fractals/Object*.hpp, Fractal.hpp.

use glam::{Vec3, Vec4};

use crate::{fold::Fold, Params, ScalarValue, Vec3Value};

/// A primitive or CSG-combined shape. Mirrors the C++ `ObjectBase` hierarchy
/// (src/fractals/Object*.hpp, Fractal.hpp) as a closed enum instead of a
/// virtual class hierarchy (see DESIGN.md §10.4).
#[derive(Clone, Debug)]
pub enum Object {
    /// src/fractals/ObjectSphere.hpp
    Sphere { radius: ScalarValue },
    /// src/fractals/ObjectBox.hpp (renamed: `Box` is the Rust pointer type)
    Cuboid { half_extent: Vec3Value },
    /// src/fractals/Fractal.hpp
    Fractal { fold: Fold, base: Box<Object> },
    /// src/fractals/ObjectClosest.hpp
    Union(Box<Object>, Box<Object>),
    /// src/fractals/ObjectIntersect.hpp
    Intersect(Box<Object>, Box<Object>),
    /// src/fractals/ObjectDifference.hpp
    Difference(Box<Object>, Box<Object>),
    /// Surface offset (no C++ counterpart): `de = base.de - offset`, i.e.
    /// the base inflated outward by `offset` everywhere (the Minkowski sum
    /// with a radius-`offset` ball), or eroded inward for a negative
    /// `offset`. Subtracting a *constant* from a distance field preserves
    /// its gradient exactly (unlike adding another spatially-varying field,
    /// which would need a Lipschitz correction), so an `Offset` of an exact
    /// child DE is itself exact -- and inflating a sharp edge rounds it,
    /// which is the classic "rounded box" construction.
    Offset {
        base: Box<Object>,
        offset: ScalarValue,
    },
    /// Hollow shell (no C++ counterpart): `de = |base.de| - thickness`,
    /// i.e. the set of points within `thickness` of the base's *surface* --
    /// a shell of total wall thickness `2·thickness`, walkable on the
    /// outside and the inside alike. Unconditionally sound: every field in
    /// this crate is 1-Lipschitz (the marching invariant), and for any
    /// 1-Lipschitz `f`, `|f(p)| <= dist(p, {f = 0})` -- so `|de| - t` never
    /// overestimates the distance to the shell, and it's *exact* wherever
    /// the base's own field is exact.
    Onion {
        base: Box<Object>,
        thickness: ScalarValue,
    },
    /// Linear interpolation between two objects: `de = mix(a.de, b.de, t)`,
    /// `t` clamped to `[0, 1]`. A convex combination of 1-Lipschitz fields
    /// is 1-Lipschitz, and any 1-Lipschitz field underestimates the
    /// distance to its own zero set (see `Onion`'s doc), so the morph is a
    /// **sound** DE at every `t` -- safe to march and collide against --
    /// though not an exact SDF mid-blend (`|∇d| < 1` where the children's
    /// gradients disagree; degenerating to near-0 exactly where a thin
    /// feature is vanishing mid-morph). The clamp is load-bearing, not
    /// cosmetic: extrapolating past `[0, 1]` makes the coefficient
    /// magnitudes sum past 1, which breaks the Lipschitz bound and with it
    /// soundness. With `t` as a `ScalarValue::Param` driven by a
    /// tick-deterministic [`crate::expr::Expr`], this gives geometry that
    /// morphs identically on every multiplayer peer through rollback, using
    /// the animation machinery that already exists.
    Morph {
        a: Box<Object>,
        b: Box<Object>,
        t: ScalarValue,
    },
}

impl Object {
    /// Whether every parameter handle referenced anywhere in this tree
    /// (recursively, including any nested `Fold`) is a valid index into a
    /// `Params` table with `slot_count` slots. A decoded `Scene` (`scene_
    /// sync.rs`) is three independently-decoded pieces -- an `Object` tree,
    /// a `Params` table, and an animation list -- each self-delimiting on
    /// its own but never cross-checked against each other; a corrupted-but-
    /// still-parseable buffer could produce a tree whose handles point past
    /// the decoded `Params`'s actual slot count, which panics the moment
    /// anything evaluates it (`Params::scalar`/`vec3`/`mat2`/`int`'s
    /// unguarded `self.slots[h.index()]`). `Scene::from_bytes` calls this
    /// (and the equivalent check on the animation table) before ever
    /// accepting a decoded scene.
    pub(crate) fn handles_valid_for(&self, slot_count: usize) -> bool {
        match self {
            Object::Sphere { radius } => radius.handle_valid_for(slot_count),
            Object::Cuboid { half_extent } => half_extent.handle_valid_for(slot_count),
            Object::Fractal { fold, base } => {
                fold.handles_valid_for(slot_count) && base.handles_valid_for(slot_count)
            }
            Object::Union(left, right) | Object::Intersect(left, right) | Object::Difference(left, right) => {
                left.handles_valid_for(slot_count) && right.handles_valid_for(slot_count)
            }
            Object::Offset { base, offset } => {
                offset.handle_valid_for(slot_count) && base.handles_valid_for(slot_count)
            }
            Object::Onion { base, thickness } => {
                thickness.handle_valid_for(slot_count) && base.handles_valid_for(slot_count)
            }
            Object::Morph { a, b, t } => {
                t.handle_valid_for(slot_count)
                    && a.handles_valid_for(slot_count)
                    && b.handles_valid_for(slot_count)
            }
        }
    }

    /// Distance estimate at `p` (xyz position, `w` = accumulated scale
    /// divisor; callers pass `p.w = 1.0`). Allocation-free.
    pub fn de(&self, p: Vec4, params: &Params) -> f32 {
        match self {
            Object::Sphere { radius } => (p.truncate().length() - radius.get(params)) / p.w,
            Object::Cuboid { half_extent } => {
                let he = half_extent.get(params);
                let a = p.truncate().abs() - he;
                (a.x.max(a.y).max(a.z).min(0.0) + a.max(Vec3::ZERO).length()) / p.w
            }
            Object::Fractal { fold, base } => {
                let mut pp = p;
                fold.fold(&mut pp, params);
                base.de(pp, params)
            }
            Object::Union(left, right) => left.de(p, params).min(right.de(p, params)),
            Object::Intersect(left, right) => left.de(p, params).max(right.de(p, params)),
            Object::Difference(left, right) => left.de(p, params).max(-right.de(p, params)),
            // `offset` is in the same (folded-local) units as `Sphere`'s
            // radius / `Cuboid`'s half-extent, so it scales by `p.w` the same
            // way: the base's returned `de` is already divided by `p.w`, and
            // a local-units inflation shrinks that world-space distance by
            // `offset / p.w` (`Offset(Sphere{r}, c)` == `Sphere{r + c}` at
            // any accumulated scale).
            Object::Offset { base, offset } => base.de(p, params) - offset.get(params) / p.w,
            // `thickness` is in folded-local units, same as `Offset`'s
            // `offset` -- hence the same `/ p.w`.
            Object::Onion { base, thickness } => {
                base.de(p, params).abs() - thickness.get(params) / p.w
            }
            // The `[0, 1]` clamp is a soundness requirement, not input
            // hygiene -- see the variant's doc.
            Object::Morph { a, b, t } => {
                let t = t.get(params).clamp(0.0, 1.0);
                (1.0 - t) * a.de(p, params) + t * b.de(p, params)
            }
        }
    }

    /// Nearest point on the surface to `p`, in the same space as `p.xyz`.
    /// Allocates its own history `Vec` for the `Fractal` case -- fine for
    /// occasional callers, but see [`Self::nearest_point_scratch`] for a
    /// hot-path variant that reuses a caller-owned buffer instead.
    pub fn nearest_point(&self, p: Vec4, params: &Params) -> Vec3 {
        let mut hist = Vec::new();
        self.nearest_point_scratch(p, params, &mut hist)
    }

    /// Same as [`Self::nearest_point`], but takes a caller-owned scratch
    /// buffer for the `Fractal` case's fold history instead of allocating a
    /// fresh `Vec` per call -- for hot paths that call this repeatedly
    /// (`physics::collide`, once per contacting sample per substep; rollback
    /// resimulation multiplies this further), the caller can reuse the same
    /// buffer across calls instead of paying a fresh heap allocation each
    /// time. Requires `hist` to already be empty on entry (the caller's
    /// responsibility -- a fresh `Vec::new()`, or a buffer proven empty by a
    /// prior call's own `debug_assert!` below) and guarantees it's empty
    /// again on return. **Deliberately does not clear `hist` itself**: a
    /// `Fractal` node's `base` can itself be a nested `Fractal`, which must
    /// share (not reset) this same buffer as a proper push/pop stack across
    /// the whole recursion -- clearing on entry would wipe an outer, still
    /// in-progress fold's history before its own `unfold` ever consumes it.
    pub fn nearest_point_scratch(&self, p: Vec4, params: &Params, hist: &mut Vec<Vec4>) -> Vec3 {
        match self {
            Object::Sphere { radius } => p.truncate().normalize() * radius.get(params),
            Object::Cuboid { half_extent } => {
                let he = half_extent.get(params);
                p.truncate().clamp(-he, he)
            }
            Object::Fractal { fold, base } => {
                let mut pp = p;
                fold.fold_with_history(&mut pp, hist, params);
                let mut n = base.nearest_point_scratch(pp, params, hist);
                fold.unfold(hist, &mut n, params);
                debug_assert!(hist.is_empty(), "fold history not fully consumed");
                n
            }
            Object::Union(left, right) => {
                if left.de(p, params) < right.de(p, params) {
                    left.nearest_point_scratch(p, params, hist)
                } else {
                    right.nearest_point_scratch(p, params, hist)
                }
            }
            Object::Intersect(left, right) => {
                if left.de(p, params) > right.de(p, params) {
                    left.nearest_point_scratch(p, params, hist)
                } else {
                    right.nearest_point_scratch(p, params, hist)
                }
            }
            Object::Difference(left, right) => {
                let left_dist = left.de(p, params);
                let right_dist = -right.de(p, params);
                if left_dist > right_dist {
                    left.nearest_point_scratch(p, params, hist)
                } else {
                    right.nearest_point_scratch(p, params, hist)
                }
            }
            // The nearest point on the offset surface lies on the line
            // through `p` and the base's nearest point, shifted along the
            // base's outward direction by `offset`: toward `p` when `p` is
            // outside the base, away from it when inside. Exact for an
            // inflation of an exact base; an erosion (negative offset) can
            // recede past a thin feature's medial axis, where this is only
            // an approximation -- fine for collision resolution, which only
            // needs a push-out direction consistent with `de`.
            Object::Offset { base, offset } => {
                let np = base.nearest_point_scratch(p, params, hist);
                let dir = p.truncate() - np;
                let len = dir.length();
                if len < 1e-9 {
                    // `p` is (numerically) on the base surface: no
                    // well-defined outward direction from the nearest-point
                    // pair alone; the un-offset point is the best answer
                    // available and is within `offset` of correct.
                    return np;
                }
                let side = if base.de(p, params) >= 0.0 { 1.0 } else { -1.0 };
                np + dir * (side * offset.get(params) / len)
            }
            // The nearest point on the shell is always on `p`'s *own* side
            // of the base surface: the same-side face is `||a| - t|` away,
            // the far face `|a| + t` -- so unlike `Offset`, no inside/
            // outside sign flip. One formula covers outside the shell,
            // inside the wall, and inside the base alike.
            Object::Onion { base, thickness } => {
                let np = base.nearest_point_scratch(p, params, hist);
                let dir = p.truncate() - np;
                let len = dir.length();
                if len < 1e-9 {
                    // `p` is (numerically) on the base surface -- the
                    // shell's medial axis, where every direction ties; `np`
                    // is within `thickness` of correct.
                    return np;
                }
                np + dir * (thickness.get(params) / len)
            }
            Object::Morph { a, b, t } => {
                let tv = t.get(params).clamp(0.0, 1.0);
                // At the endpoints the morph *is* that child -- delegate
                // exactly rather than paying (and slightly blurring
                // through) the numerical projection below.
                if tv <= 0.0 {
                    a.nearest_point_scratch(p, params, hist)
                } else if tv >= 1.0 {
                    b.nearest_point_scratch(p, params, hist)
                } else {
                    // Mid-blend there is genuinely no closed form (the
                    // surface is an implicit blend belonging to neither
                    // child), so project onto the zero set numerically --
                    // see `project_to_surface`'s doc for the accuracy
                    // argument.
                    self.project_to_surface(p, params)
                }
            }
        }
    }

    /// Nearest-point fallback for nodes whose surface has no closed form
    /// (currently `Morph` mid-blend): damped Newton projection along the
    /// numerically-estimated gradient, `x -= de(x) · ∇de/|∇de|`, using the
    /// same 4-tap tetrahedral gradient stencil as the shader's
    /// `calc_normal`.
    ///
    /// Accuracy: physics only asks for a nearest point once a sample is in
    /// contact range (`de < marble radius`, `physics::collide`), so the
    /// iteration starts already near the surface, where Newton on a sound
    /// 1-Lipschitz field converges quadratically -- three steps lands far
    /// below the smallest marble radius (0.02) for smooth blends (verified
    /// by the analytic-morph tests below). The degenerate-gradient guard
    /// covers the one place a morph's gradient legitimately collapses
    /// toward zero: where a thin feature is mid-vanish, in which case `p`'s
    /// current projection is returned as-is rather than dividing by ~0.
    fn project_to_surface(&self, p: Vec4, params: &Params) -> Vec3 {
        const K: f32 = 0.577_350_3; // 1/sqrt(3), unit tetrahedron vertex
        let mut x = p.truncate();
        for _ in 0..3 {
            let d = self.de(x.extend(p.w), params);
            // Same order of magnitude as the shader's normal-estimation
            // epsilon; scaled up with distance from the origin so the
            // finite differences stay above f32 noise on large-coordinate
            // scenes.
            let eps = 1e-4 * x.length().max(1.0);
            let e1 = Vec3::new(K, -K, -K) * eps;
            let e2 = Vec3::new(-K, -K, K) * eps;
            let e3 = Vec3::new(-K, K, -K) * eps;
            let e4 = Vec3::new(K, K, K) * eps;
            let g = e1 * self.de((x + e1).extend(p.w), params)
                + e2 * self.de((x + e2).extend(p.w), params)
                + e3 * self.de((x + e3).extend(p.w), params)
                + e4 * self.de((x + e4).extend(p.w), params);
            let glen = g.length();
            if glen < 1e-12 {
                break;
            }
            x -= g * (d / glen);
        }
        x
    }

    /// A world-space `(center, radius)` bounding sphere for this object, or
    /// `None` if it can't be bounded (an unbounded-tiling `Fold::Modulo`/
    /// `Repeat` with no enclosing `Intersect` above it -- see `Union`/
    /// `Intersect` below for how a finite bound still emerges when a scene
    /// actually clips one of those to a finite region, e.g. `creme_spheres`).
    /// Recurses the same shape as `de`, composing a bound instead of a
    /// distance -- see `Fold::unfold_bounding_sphere`'s doc for why every
    /// case here only has to be a *sound* (possibly loose) outer bound, never
    /// an under-approximation: this feeds a ray-clip pre-test, where a bound
    /// that's too small silently culls real geometry instead of erroring.
    pub fn bounding_sphere(&self, params: &Params) -> Option<(Vec3, f32)> {
        match self {
            Object::Sphere { radius } => Some((Vec3::ZERO, radius.get(params))),
            Object::Cuboid { half_extent } => Some((Vec3::ZERO, half_extent.get(params).length())),
            Object::Fractal { fold, base } => {
                fold.unfold_bounding_sphere(base.bounding_sphere(params), params)
            }
            Object::Union(left, right) => {
                let l = left.bounding_sphere(params)?;
                let r = right.bounding_sphere(params)?;
                Some(enclosing_sphere(l, r))
            }
            // Intersection is a subset of both children, so either child's
            // bound is already a valid (if not tightest) bound of it --
            // take whichever is finite, or the smaller if both are.
            Object::Intersect(left, right) => {
                match (left.bounding_sphere(params), right.bounding_sphere(params)) {
                    (Some(l), Some(r)) => Some(if l.1 <= r.1 { l } else { r }),
                    (Some(l), None) => Some(l),
                    (None, Some(r)) => Some(r),
                    (None, None) => None,
                }
            }
            // Subtracting can only shrink the set.
            Object::Difference(left, _right) => left.bounding_sphere(params),
            // Inflation grows the set by exactly `offset` in every
            // direction, so pad the child's radius by it. An erosion
            // (negative offset) only shrinks the set, so the child's own
            // bound already covers it -- clamp rather than subtract, since
            // the child bound's center generally isn't the body's and
            // shrinking the sphere could cut off real geometry.
            Object::Offset { base, offset } => {
                let (c, r) = base.bounding_sphere(params)?;
                Some((c, r + offset.get(params).max(0.0)))
            }
            // The shell reaches `thickness` beyond the base surface, which
            // the inflated-solid bound covers; the inner face only removes
            // material. (Non-positive thickness means an empty shell --
            // clamp so it can't shrink the pad below zero.)
            Object::Onion { base, thickness } => {
                let (c, r) = base.bounding_sphere(params)?;
                Some((c, r + thickness.get(params).max(0.0)))
            }
            // Sound at every `t`: `mix(a, b, t) < 0` requires at least one
            // of `a`/`b` to be negative, so the morph's solid is a subset
            // of the two children's solids' union regardless of `t` --
            // enclose both. Either child unbounded makes the morph
            // potentially unbounded.
            Object::Morph { a, b, t: _ } => {
                let ba = a.bounding_sphere(params)?;
                let bb = b.bounding_sphere(params)?;
                Some(enclosing_sphere(ba, bb))
            }
        }
    }

    /// Serializes to a compact, tag-prefixed byte encoding — same
    /// hand-rolled, self-delimiting, recursive convention as
    /// [`crate::expr::Expr::encode`]/[`Fold::encode`] (see the former's
    /// doc for why). Used by [`crate::Scene`] for
    /// multiplayer's join-time scene sync: the host serializes its live
    /// scene tree once at connect and sends it to the joiner, instead of
    /// the joiner building its own scene from a locally-hardcoded
    /// `SceneKind` constructor that could disagree with whatever the host
    /// actually has loaded.
    pub fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Object::Sphere { radius } => {
                out.push(0);
                radius.encode(out);
            }
            Object::Cuboid { half_extent } => {
                out.push(1);
                half_extent.encode(out);
            }
            Object::Fractal { fold, base } => {
                out.push(2);
                fold.encode(out);
                base.encode(out);
            }
            Object::Union(left, right) => {
                out.push(3);
                left.encode(out);
                right.encode(out);
            }
            Object::Intersect(left, right) => {
                out.push(4);
                left.encode(out);
                right.encode(out);
            }
            Object::Difference(left, right) => {
                out.push(5);
                left.encode(out);
                right.encode(out);
            }
            Object::Offset { base, offset } => {
                out.push(6);
                offset.encode(out);
                base.encode(out);
            }
            Object::Onion { base, thickness } => {
                out.push(7);
                thickness.encode(out);
                base.encode(out);
            }
            Object::Morph { a, b, t } => {
                out.push(8);
                t.encode(out);
                a.encode(out);
                b.encode(out);
            }
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.encode(&mut out);
        out
    }

    /// Inverse of [`Self::encode`]/[`Self::to_bytes`] — `None` on any
    /// malformed/truncated input, or if `bytes` has leftover data after a
    /// complete tree decodes (same reasoning as
    /// [`crate::expr::Expr::from_bytes`]).
    pub fn from_bytes(bytes: &[u8]) -> Option<Object> {
        let (object, consumed) = Self::decode_at(bytes, 0)?;
        if consumed == bytes.len() {
            Some(object)
        } else {
            None
        }
    }

    pub(crate) fn decode_at(bytes: &[u8], pos: usize) -> Option<(Object, usize)> {
        let tag = *bytes.get(pos)?;
        let pos = pos + 1;
        let result = match tag {
            0 => {
                let (radius, pos) = ScalarValue::decode_at(bytes, pos)?;
                (Object::Sphere { radius }, pos)
            }
            1 => {
                let (half_extent, pos) = Vec3Value::decode_at(bytes, pos)?;
                (Object::Cuboid { half_extent }, pos)
            }
            2 => {
                let (fold, pos) = Fold::decode_at(bytes, pos)?;
                let (base, pos) = Object::decode_at(bytes, pos)?;
                (Object::Fractal { fold, base: Box::new(base) }, pos)
            }
            3 => {
                let (left, pos) = Object::decode_at(bytes, pos)?;
                let (right, pos) = Object::decode_at(bytes, pos)?;
                (Object::Union(Box::new(left), Box::new(right)), pos)
            }
            4 => {
                let (left, pos) = Object::decode_at(bytes, pos)?;
                let (right, pos) = Object::decode_at(bytes, pos)?;
                (Object::Intersect(Box::new(left), Box::new(right)), pos)
            }
            5 => {
                let (left, pos) = Object::decode_at(bytes, pos)?;
                let (right, pos) = Object::decode_at(bytes, pos)?;
                (Object::Difference(Box::new(left), Box::new(right)), pos)
            }
            6 => {
                let (offset, pos) = ScalarValue::decode_at(bytes, pos)?;
                let (base, pos) = Object::decode_at(bytes, pos)?;
                (Object::Offset { base: Box::new(base), offset }, pos)
            }
            7 => {
                let (thickness, pos) = ScalarValue::decode_at(bytes, pos)?;
                let (base, pos) = Object::decode_at(bytes, pos)?;
                (Object::Onion { base: Box::new(base), thickness }, pos)
            }
            8 => {
                let (t, pos) = ScalarValue::decode_at(bytes, pos)?;
                let (a, pos) = Object::decode_at(bytes, pos)?;
                let (b, pos) = Object::decode_at(bytes, pos)?;
                (Object::Morph { a: Box::new(a), b: Box::new(b), t }, pos)
            }
            _ => return None,
        };
        Some(result)
    }
}

/// Smallest sphere enclosing two given spheres.
fn enclosing_sphere(a: (Vec3, f32), b: (Vec3, f32)) -> (Vec3, f32) {
    let (ca, ra) = a;
    let (cb, rb) = b;
    let d = ca.distance(cb);
    if ra >= d + rb {
        return a;
    }
    if rb >= d + ra {
        return b;
    }
    let new_r = (d + ra + rb) * 0.5;
    let center = if d > 1e-9 {
        ca + (cb - ca) * ((new_r - ra) / d)
    } else {
        ca
    };
    (center, new_r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Axis;

    fn sphere(r: f32) -> Object {
        Object::Sphere {
            radius: ScalarValue::Const(r),
        }
    }

    fn cuboid(he: Vec3) -> Object {
        Object::Cuboid {
            half_extent: Vec3Value::Const(he),
        }
    }

    #[test]
    fn sphere_de_closed_form() {
        let params = Params::new();
        let s = sphere(2.0);
        assert!((s.de(Vec4::new(5.0, 0.0, 0.0, 1.0), &params) - 3.0).abs() < 1e-6);
        assert!((s.de(Vec4::new(1.0, 0.0, 0.0, 1.0), &params) - (-1.0)).abs() < 1e-6);
        assert!((s.de(Vec4::new(2.0, 0.0, 0.0, 1.0), &params) - 0.0).abs() < 1e-6);
        // Scaled-w: DE divides by p.w.
        assert!((s.de(Vec4::new(5.0, 0.0, 0.0, 2.0), &params) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn cuboid_de_closed_form() {
        let params = Params::new();
        let c = cuboid(Vec3::new(1.0, 2.0, 3.0));
        // Outside, directly off the +x face.
        assert!((c.de(Vec4::new(4.0, 0.0, 0.0, 1.0), &params) - 3.0).abs() < 1e-6);
        // Inside: negative, equal to the (negative) distance to the nearest face.
        let inside = c.de(Vec4::new(0.5, 0.0, 0.0, 1.0), &params);
        assert!(inside < 0.0);
        assert!((inside - (-0.5)).abs() < 1e-6);
        // On the surface.
        assert!(c.de(Vec4::new(1.0, 0.0, 0.0, 1.0), &params).abs() < 1e-6);
        // Scaled-w.
        assert!((c.de(Vec4::new(4.0, 0.0, 0.0, 2.0), &params) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn cuboid_nearest_point_clamps() {
        let params = Params::new();
        let c = cuboid(Vec3::new(1.0, 1.0, 1.0));
        let np = c.nearest_point(Vec4::new(5.0, 0.5, -5.0, 1.0), &params);
        assert_eq!(np, Vec3::new(1.0, 0.5, -1.0));
    }

    #[test]
    fn union_picks_closer_side() {
        let params = Params::new();
        let union = Object::Union(Box::new(sphere(1.0)), Box::new(cuboid(Vec3::splat(1.0))));

        // Union DE is the min of the two DEs.
        let p = Vec4::new(3.0, 0.0, 0.0, 1.0);
        let l = sphere(1.0).de(p, &params);
        let r = cuboid(Vec3::splat(1.0)).de(p, &params);
        assert!((union.de(p, &params) - l.min(r)).abs() < 1e-6);

        // Probe off-axis; Union's nearest_point should match whichever side
        // actually has the smaller DE there.
        let p2 = Vec4::new(0.9, 0.9, 0.9, 1.0);
        let l2 = sphere(1.0).de(p2, &params);
        let r2 = cuboid(Vec3::splat(1.0)).de(p2, &params);
        assert!(l2 > r2, "test setup expects the cuboid to win here");
        let expected_np = cuboid(Vec3::splat(1.0)).nearest_point(p2, &params);
        assert_eq!(union.nearest_point(p2, &params), expected_np);
    }

    #[test]
    fn intersect_picks_farther_side() {
        let params = Params::new();
        let big_sphere = sphere(6.0);
        let small_box = cuboid(Vec3::splat(1.0));
        let inter = Object::Intersect(Box::new(big_sphere), Box::new(small_box));
        let p = Vec4::new(0.5, 0.0, 0.0, 1.0);
        let l = sphere(6.0).de(p, &params);
        let r = cuboid(Vec3::splat(1.0)).de(p, &params);
        assert!((inter.de(p, &params) - l.max(r)).abs() < 1e-6);
        // At this point the box is the binding (max) constraint, so NP should match the box's.
        let np_box = cuboid(Vec3::splat(1.0)).nearest_point(p, &params);
        assert_eq!(inter.nearest_point(p, &params), np_box);
    }

    #[test]
    fn difference_picks_correct_side() {
        let params = Params::new();
        let left = sphere(5.0);
        let right = sphere(1.0);
        let diff = Object::Difference(Box::new(left), Box::new(right));
        // Near the cut-out sphere's surface, -right.de dominates.
        let p = Vec4::new(1.0, 0.0, 0.0, 1.0);
        let l = sphere(5.0).de(p, &params);
        let r = -sphere(1.0).de(p, &params);
        assert!((diff.de(p, &params) - l.max(r)).abs() < 1e-6);
        assert!(l < r); // sanity: right side should be binding here
        let np_right = sphere(1.0).nearest_point(p, &params);
        assert_eq!(diff.nearest_point(p, &params), np_right);
    }

    #[test]
    fn sphere_and_cuboid_bounding_sphere() {
        let params = Params::new();
        assert_eq!(
            sphere(2.0).bounding_sphere(&params),
            Some((Vec3::ZERO, 2.0))
        );
        let (c, r) = cuboid(Vec3::new(1.0, 2.0, 3.0))
            .bounding_sphere(&params)
            .unwrap();
        assert_eq!(c, Vec3::ZERO);
        assert!((r - Vec3::new(1.0, 2.0, 3.0).length()).abs() < 1e-5);
    }

    #[test]
    fn union_bounding_sphere_encloses_both_children() {
        let params = Params::new();
        let union = Object::Union(
            Box::new(Object::Sphere {
                radius: ScalarValue::Const(1.0),
            }),
            Box::new(Object::Fractal {
                fold: Fold::ScaleTranslate {
                    scale: ScalarValue::Const(1.0),
                    shift: Vec3Value::Const(Vec3::new(5.0, 0.0, 0.0)),
                },
                base: Box::new(sphere(1.0)),
            }),
        );
        let (c, r) = union.bounding_sphere(&params).unwrap();
        // Both children (unit sphere at world origin; the second's local
        // sphere at its own origin is reached by world point (-5,0,0), since
        // the fold's forward map is `p' = p + (5,0,0)` and the local sphere
        // sits at `p' = 0`) must lie entirely within the enclosing sphere.
        assert!(c.distance(Vec3::ZERO) + 1.0 <= r + 1e-4);
        assert!(c.distance(Vec3::new(-5.0, 0.0, 0.0)) + 1.0 <= r + 1e-4);
    }

    #[test]
    fn union_with_unbounded_child_is_unbounded() {
        let params = Params::new();
        let unbounded = Object::Fractal {
            fold: Fold::Modulo {
                axis: Axis::X,
                modulus: ScalarValue::Const(1.0),
            },
            base: Box::new(sphere(0.1)),
        };
        let union = Object::Union(Box::new(sphere(1.0)), Box::new(unbounded));
        assert!(union.bounding_sphere(&params).is_none());
    }

    #[test]
    fn bare_modulo_object_is_unbounded() {
        let params = Params::new();
        let obj = Object::Fractal {
            fold: Fold::Modulo {
                axis: Axis::X,
                modulus: ScalarValue::Const(1.0),
            },
            base: Box::new(sphere(0.1)),
        };
        assert!(obj.bounding_sphere(&params).is_none());
    }

    #[test]
    fn creme_spheres_intersect_resolves_infinite_repeat_to_the_outer_sphere() {
        let params = Params::new();
        let object = crate::scenes::creme_spheres();
        // `creme_spheres` is `Intersect(Modulo-repeated spheres [unbounded],
        // outer radius-6 sphere)` -- `Intersect` picks the finite side.
        let (c, r) = object.bounding_sphere(&params).unwrap();
        assert_eq!(c, Vec3::ZERO);
        assert!((r - 6.0).abs() < 1e-5);
    }

    /// Soundness check against real, non-toy scene trees (not just synthetic
    /// examples above): every point the tree itself reports as "inside"
    /// (`de < 0`) must lie within the tree's own computed bounding sphere.
    /// This is the property that actually matters for the ray-clip
    /// pre-test -- an under-approximating bound silently culls real
    /// geometry, so this is checked directly against `de`, not just against
    /// the bound-composition formulas in isolation.
    #[test]
    fn bounding_sphere_is_sound_against_real_scenes() {
        use crate::scenes::*;

        // `steps`/`spacing` are per-scene: `menger_sponge` in particular is
        // a thin-walled shell whose true interior is close to zero-measure
        // (min DE found by a targeted probe was ~-3e-6), so a coarse grid
        // can trivially find zero interior points without that meaning
        // anything about soundness -- give it a finer, narrower scan than
        // the other (thicker/larger) scenes.
        fn assert_sound(name: &str, object: &Object, params: &Params, steps: i32, spacing: f32) {
            let (center, radius) = object
                .bounding_sphere(params)
                .unwrap_or_else(|| panic!("{name}: expected a finite bound"));
            let mut inside_count = 0;
            for xi in -steps..=steps {
                for yi in -steps..=steps {
                    for zi in -steps..=steps {
                        let p = Vec3::new(xi as f32, yi as f32, zi as f32) * spacing;
                        if object.de(p.extend(1.0), params) < 0.0 {
                            inside_count += 1;
                            let d = p.distance(center);
                            assert!(
                                d <= radius + 1e-3,
                                "{name}: inside point {p:?} at distance {d} from bound \
                                 center {center:?}, outside computed radius {radius}"
                            );
                        }
                    }
                }
            }
            assert!(inside_count > 0, "{name}: scan found no inside points at all -- test isn't exercising anything");
        }

        let mut p1 = Params::new();
        let (classic_obj, h1) = classic(&mut p1);
        set_fractal_params(
            &mut p1,
            &h1,
            beware_of_bumps::SCALE,
            beware_of_bumps::ANG1,
            beware_of_bumps::ANG2,
            beware_of_bumps::SHIFT,
            beware_of_bumps::COLOR,
            beware_of_bumps::ITERS,
        );
        assert_sound("classic", &classic_obj, &p1, 20, 0.7);

        let mut p2 = Params::new();
        let (demo, h2) = demo_scene(&mut p2);
        set_fractal_params(
            &mut p2,
            &h2,
            beware_of_bumps::SCALE,
            beware_of_bumps::ANG1,
            beware_of_bumps::ANG2,
            beware_of_bumps::SHIFT,
            beware_of_bumps::COLOR,
            beware_of_bumps::ITERS,
        );
        assert_sound("demo_scene", &demo, &p2, 20, 0.7);

        let p3 = Params::new();
        assert_sound("creme_spheres", &creme_spheres(), &p3, 20, 0.3);

        let mut p4 = Params::new();
        let (sponge, h4) = menger_sponge(&mut p4);
        set_menger_params(&mut p4, &h4, 12, Vec3::new(1.0, 0.5, 0.2));
        assert_sound("menger_sponge", &sponge, &p4, 30, 0.1);
    }

    fn offset(base: Object, c: f32) -> Object {
        Object::Offset {
            base: Box::new(base),
            offset: ScalarValue::Const(c),
        }
    }

    /// `Offset(Sphere{r}, c)` must be indistinguishable from `Sphere{r+c}`
    /// -- `de` at outside, inside, and surface points, including a scaled
    /// `p.w`, and for a negative (eroding) offset too.
    #[test]
    fn offset_sphere_matches_larger_sphere() {
        let params = Params::new();
        let inflated = offset(sphere(1.0), 0.5);
        let eroded = offset(sphere(2.0), -0.5);
        let equivalent = sphere(1.5);
        for p in [
            Vec4::new(5.0, 0.0, 0.0, 1.0),
            Vec4::new(0.3, 0.4, 0.0, 1.0),
            Vec4::new(1.5, 0.0, 0.0, 1.0),
            Vec4::new(3.0, -2.0, 1.0, 2.0), // scaled w
        ] {
            let expected = equivalent.de(p, &params);
            assert!((inflated.de(p, &params) - expected).abs() < 1e-6, "inflated at {p:?}");
            assert!((eroded.de(p, &params) - expected).abs() < 1e-6, "eroded at {p:?}");
        }
    }

    /// The offset is in the same folded-local units as a primitive's own
    /// size parameters, so under a scaling fold `Offset(Sphere{r}, c)` must
    /// still equal `Sphere{r+c}` *inside the same fold* -- this is the
    /// `/ p.w` in the `de` arm (and the `w_save` in codegen's).
    #[test]
    fn offset_de_is_exact_under_a_scaling_fold() {
        let params = Params::new();
        let fold = || Fold::ScaleTranslate {
            scale: ScalarValue::Const(2.0),
            shift: Vec3Value::Const(Vec3::new(0.5, -1.0, 0.25)),
        };
        let offset_in_fold = Object::Fractal {
            fold: fold(),
            base: Box::new(offset(sphere(1.0), 0.5)),
        };
        let equivalent = Object::Fractal {
            fold: fold(),
            base: Box::new(sphere(1.5)),
        };
        for p in [
            Vec4::new(3.0, 1.0, -2.0, 1.0),
            Vec4::new(0.1, 0.2, 0.0, 1.0),
            Vec4::new(-1.0, 4.0, 2.0, 1.0),
        ] {
            let a = offset_in_fold.de(p, &params);
            let b = equivalent.de(p, &params);
            assert!((a - b).abs() < 1e-6, "at {p:?}: offset-in-fold={a} equivalent={b}");
        }
    }

    #[test]
    fn offset_nearest_point_outside_and_inside() {
        let params = Params::new();
        let obj = offset(sphere(1.0), 0.5);
        // Outside: nearest point on the inflated (radius 1.5) sphere.
        let np = obj.nearest_point(Vec4::new(3.0, 0.0, 0.0, 1.0), &params);
        assert!((np - Vec3::new(1.5, 0.0, 0.0)).length() < 1e-6, "outside: {np:?}");
        // Inside the base: surface moved *away*, still at radius 1.5.
        let np = obj.nearest_point(Vec4::new(0.2, 0.0, 0.0, 1.0), &params);
        assert!((np - Vec3::new(1.5, 0.0, 0.0)).length() < 1e-6, "inside: {np:?}");
        // Between the base surface and the inflated surface (inside the
        // inflated solid, outside the base): same surface point.
        let np = obj.nearest_point(Vec4::new(1.2, 0.0, 0.0, 1.0), &params);
        assert!((np - Vec3::new(1.5, 0.0, 0.0)).length() < 1e-6, "in shell: {np:?}");
    }

    #[test]
    fn offset_bounding_sphere_pads_only_for_inflation() {
        let params = Params::new();
        let (c, r) = offset(sphere(2.0), 0.5).bounding_sphere(&params).unwrap();
        assert_eq!(c, Vec3::ZERO);
        assert!((r - 2.5).abs() < 1e-6);
        // Erosion shrinks the set; the child's bound must be kept, not
        // shrunk (see the bounding_sphere arm's comment).
        let (c, r) = offset(sphere(2.0), -0.5).bounding_sphere(&params).unwrap();
        assert_eq!(c, Vec3::ZERO);
        assert!((r - 2.0).abs() < 1e-6);
    }

    #[test]
    fn offset_rounds_a_cuboid_corner() {
        // Inflating is a Minkowski sum with a ball: past a corner, the
        // surface is at `corner + offset` along the diagonal, not at the
        // sharp cuboid's own inflated corner point.
        let params = Params::new();
        let obj = offset(cuboid(Vec3::ONE), 0.5);
        let diag = Vec3::splat(1.0 / 3.0_f32.sqrt());
        let corner = Vec3::ONE;
        // A point along the corner diagonal, 0.5 + 0.25 past the corner:
        // de should be 0.25 (0.5 of the gap is eaten by the rounding).
        let p = corner + diag * 0.75;
        let d = obj.de(p.extend(1.0), &params);
        assert!((d - 0.25).abs() < 1e-6, "rounded-corner de: {d}");
    }

    fn onion(base: Object, t: f32) -> Object {
        Object::Onion {
            base: Box::new(base),
            thickness: ScalarValue::Const(t),
        }
    }

    fn morph(a: Object, b: Object, t: f32) -> Object {
        Object::Morph {
            a: Box::new(a),
            b: Box::new(b),
            t: ScalarValue::Const(t),
        }
    }

    /// A spherical shell has a closed-form CSG equivalent --
    /// `Onion(Sphere{r}, t)` must equal `Difference(Sphere{r+t},
    /// Sphere{r-t})` *everywhere* (their fields agree exactly: both reduce
    /// to `max(|p|-r-t, r-t-|p|)` = `||p|-r| - t`), which pins the whole
    /// `de` arm against an independently-computed exact answer.
    #[test]
    fn onion_sphere_matches_difference_of_spheres_exactly() {
        let params = Params::new();
        let shell = onion(sphere(2.0), 0.3);
        let equivalent = Object::Difference(Box::new(sphere(2.3)), Box::new(sphere(1.7)));
        for p in [
            Vec4::new(5.0, 0.0, 0.0, 1.0),   // outside everything
            Vec4::new(2.1, 0.4, -0.2, 1.0),  // inside the wall
            Vec4::new(0.6, 0.5, 0.2, 1.0),   // inside the hollow
            Vec4::new(2.3, 0.0, 0.0, 1.0),   // on the outer face
            Vec4::new(3.0, -1.0, 2.0, 2.0),  // scaled w
        ] {
            let a = shell.de(p, &params);
            let b = equivalent.de(p, &params);
            assert!((a - b).abs() < 1e-6, "at {p:?}: onion={a} difference={b}");
        }
    }

    #[test]
    fn onion_nearest_point_hits_the_nearer_face_in_every_region() {
        let params = Params::new();
        let shell = onion(sphere(2.0), 0.3);
        // Outside the shell: outer face.
        let np = shell.nearest_point(Vec4::new(5.0, 0.0, 0.0, 1.0), &params);
        assert!((np - Vec3::new(2.3, 0.0, 0.0)).length() < 1e-5, "outside: {np:?}");
        // Inside the hollow: inner face.
        let np = shell.nearest_point(Vec4::new(1.0, 0.0, 0.0, 1.0), &params);
        assert!((np - Vec3::new(1.7, 0.0, 0.0)).length() < 1e-5, "hollow: {np:?}");
        // Inside the wall, nearer the outer face.
        let np = shell.nearest_point(Vec4::new(2.1, 0.0, 0.0, 1.0), &params);
        assert!((np - Vec3::new(2.3, 0.0, 0.0)).length() < 1e-5, "in wall: {np:?}");
    }

    #[test]
    fn onion_de_is_exact_under_a_scaling_fold() {
        let params = Params::new();
        let fold = || Fold::ScaleTranslate {
            scale: ScalarValue::Const(2.0),
            shift: Vec3Value::Const(Vec3::new(0.5, -1.0, 0.25)),
        };
        let onion_in_fold = Object::Fractal {
            fold: fold(),
            base: Box::new(onion(sphere(1.0), 0.25)),
        };
        let equivalent = Object::Fractal {
            fold: fold(),
            base: Box::new(Object::Difference(Box::new(sphere(1.25)), Box::new(sphere(0.75)))),
        };
        for p in [
            Vec4::new(3.0, 1.0, -2.0, 1.0),
            Vec4::new(0.1, 0.6, 0.0, 1.0),
            Vec4::new(-0.3, 0.5, -0.1, 1.0),
        ] {
            let a = onion_in_fold.de(p, &params);
            let b = equivalent.de(p, &params);
            assert!((a - b).abs() < 1e-6, "at {p:?}: onion-in-fold={a} equivalent={b}");
        }
    }

    #[test]
    fn onion_bounding_sphere_pads_by_thickness() {
        let params = Params::new();
        let (c, r) = onion(sphere(2.0), 0.3).bounding_sphere(&params).unwrap();
        assert_eq!(c, Vec3::ZERO);
        assert!((r - 2.3).abs() < 1e-6);
    }

    /// At the endpoints the morph must *be* the corresponding child --
    /// exact `de` and exact (delegated, not projected) `nearest_point`.
    #[test]
    fn morph_endpoints_match_children_exactly() {
        let params = Params::new();
        let probes = [
            Vec4::new(3.0, 0.5, -1.0, 1.0),
            Vec4::new(0.4, 0.2, 0.1, 1.0),
            Vec4::new(-1.5, 2.0, 0.7, 2.0),
        ];
        let at_zero = morph(sphere(1.0), cuboid(Vec3::splat(1.0)), 0.0);
        let at_one = morph(sphere(1.0), cuboid(Vec3::splat(1.0)), 1.0);
        for p in probes {
            assert!((at_zero.de(p, &params) - sphere(1.0).de(p, &params)).abs() < 1e-6);
            assert!((at_one.de(p, &params) - cuboid(Vec3::splat(1.0)).de(p, &params)).abs() < 1e-6);
        }
        let p = Vec4::new(3.0, 0.5, -1.0, 1.0);
        assert_eq!(at_zero.nearest_point(p, &params), sphere(1.0).nearest_point(p, &params));
        assert_eq!(
            at_one.nearest_point(p, &params),
            cuboid(Vec3::splat(1.0)).nearest_point(p, &params)
        );
        // Out-of-range `t` clamps (a soundness requirement, the variant's
        // doc) -- `t = 1.5` behaves exactly like `t = 1`.
        let clamped = morph(sphere(1.0), cuboid(Vec3::splat(1.0)), 1.5);
        assert!((clamped.de(p, &params) - cuboid(Vec3::splat(1.0)).de(p, &params)).abs() < 1e-6);
    }

    /// Concentric spheres are the analytic case: `mix(|p|-r1, |p|-r2, t)`
    /// = `|p| - mix(r1, r2, t)`, an exact sphere SDF -- so both the morph
    /// `de` and the Newton-projected `nearest_point` have a known exact
    /// answer mid-blend, pinning the projection's real accuracy (not just
    /// its self-consistency).
    #[test]
    fn morph_concentric_spheres_is_the_interpolated_sphere() {
        let params = Params::new();
        let m = morph(sphere(1.0), sphere(2.0), 0.3);
        let expected = sphere(1.3);
        for p in [
            Vec4::new(3.0, 0.0, 0.0, 1.0),
            Vec4::new(0.5, 0.4, -0.1, 1.0),
            Vec4::new(-2.0, 1.0, 0.5, 2.0),
        ] {
            assert!((m.de(p, &params) - expected.de(p, &params)).abs() < 1e-6, "de at {p:?}");
        }
        let np = m.nearest_point(Vec4::new(3.0, 0.0, 0.0, 1.0), &params);
        assert!((np - Vec3::new(1.3, 0.0, 0.0)).length() < 1e-3, "outside np: {np:?}");
        let np = m.nearest_point(Vec4::new(0.5, 0.0, 0.0, 1.0), &params);
        assert!((np - Vec3::new(1.3, 0.0, 0.0)).length() < 1e-3, "inside np: {np:?}");
    }

    /// Mid-blend of genuinely different shapes has no closed form to pin
    /// against, so check the two properties collision response actually
    /// relies on: the projected point sits on the morph surface
    /// (`de(np) ~= 0`), and its distance from `p` is consistent with
    /// `de(p)` (the same |p-np|-vs-de sanity the demo-scene test applies).
    #[test]
    fn morph_nearest_point_is_consistent_mid_blend() {
        let params = Params::new();
        let m = morph(sphere(1.0), cuboid(Vec3::splat(1.0)), 0.5);
        for probe in [
            Vec3::new(1.3, 0.2, 0.1),
            Vec3::new(0.8, 0.8, 0.3),
            Vec3::new(0.0, 1.2, 0.0),
            Vec3::new(0.6, 0.6, 0.6),
        ] {
            let p = probe.extend(1.0);
            let d = m.de(p, &params);
            let np = m.nearest_point(p, &params);
            let residual = m.de(np.extend(1.0), &params).abs();
            assert!(residual < 2e-3, "probe {probe:?}: np {np:?} residual {residual}");
            // `de` is a sound *underestimate* mid-blend (`|∇d| < 1` where
            // the children's gradients disagree), so the true travel to the
            // surface legitimately exceeds `|de|` -- assert exactly the
            // soundness direction (never a surface closer than `de`
            // promised), plus the same loose 2x upper factor the
            // demo-scene consistency test uses.
            let travel = (probe - np).length();
            assert!(
                travel + 1e-4 >= d.abs(),
                "probe {probe:?}: found a surface point closer than de promised: \
                 |p-np|={travel} vs |de|={}",
                d.abs()
            );
            assert!(
                travel <= 2.0 * d.abs().max(1e-3),
                "probe {probe:?}: |p-np|={travel} wildly exceeds |de|={}",
                d.abs()
            );
        }
    }

    #[test]
    fn morph_bounding_sphere_encloses_both_children() {
        let params = Params::new();
        let m = morph(sphere(1.0), cuboid(Vec3::splat(2.0)), 0.5);
        let (c, r) = m.bounding_sphere(&params).unwrap();
        let (ca, ra) = sphere(1.0).bounding_sphere(&params).unwrap();
        let (cb, rb) = cuboid(Vec3::splat(2.0)).bounding_sphere(&params).unwrap();
        assert!(c.distance(ca) + ra <= r + 1e-4);
        assert!(c.distance(cb) + rb <= r + 1e-4);
    }

    #[test]
    fn fractal_history_consumed_and_matches_direct_fold() {
        let params = Params::new();
        let fold = Fold::Series(vec![Fold::Abs, Fold::Menger]);
        let obj = Object::Fractal {
            fold: fold.clone(),
            base: Box::new(cuboid(Vec3::new(1.0, 1.0, 1.0))),
        };
        let p = Vec4::new(-2.0, 0.7, -1.3, 1.0);

        // de() matches folding p directly then evaluating the base.
        let mut pp = p;
        fold.fold(&mut pp, &params);
        let expected = cuboid(Vec3::new(1.0, 1.0, 1.0)).de(pp, &params);
        assert!((obj.de(p, &params) - expected).abs() < 1e-6);

        // nearest_point should not panic (debug_assert covers full history
        // consumption) and should return a finite point.
        let np = obj.nearest_point(p, &params);
        assert!(np.is_finite());
    }
}
