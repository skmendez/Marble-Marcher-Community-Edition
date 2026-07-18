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
    /// `Fractal` allocates one history `Vec` per call (fold → base lookup →
    /// unfold); other variants are allocation-free (aside from recursion).
    pub fn nearest_point(&self, p: Vec4, params: &Params) -> Vec3 {
        match self {
            Object::Sphere { radius } => p.truncate().normalize() * radius.get(params),
            Object::Cuboid { half_extent } => {
                let he = half_extent.get(params);
                p.truncate().clamp(-he, he)
            }
            Object::Fractal { fold, base } => {
                let mut pp = p;
                let mut hist = Vec::new();
                fold.fold_with_history(&mut pp, &mut hist, params);
                let mut n = base.nearest_point(pp, params);
                fold.unfold(&mut hist, &mut n, params);
                debug_assert!(hist.is_empty(), "fold history not fully consumed");
                n
            }
            Object::Union(left, right) => {
                if left.de(p, params) < right.de(p, params) {
                    left.nearest_point(p, params)
                } else {
                    right.nearest_point(p, params)
                }
            }
            Object::Intersect(left, right) => {
                if left.de(p, params) > right.de(p, params) {
                    left.nearest_point(p, params)
                } else {
                    right.nearest_point(p, params)
                }
            }
            Object::Difference(left, right) => {
                let left_dist = left.de(p, params);
                let right_dist = -right.de(p, params);
                if left_dist > right_dist {
                    left.nearest_point(p, params)
                } else {
                    right.nearest_point(p, params)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
