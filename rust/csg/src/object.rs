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
}

impl Object {
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
        }
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
