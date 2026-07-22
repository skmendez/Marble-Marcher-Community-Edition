//! Wire codec for [`crate::Scene`] — multiplayer join-time scene sync.
//!
//! **Why this exists**: a joiner's page has no reliable way to know what
//! scene the host is actually running (`?scene=` is a best-effort initial
//! render guess only, since the joiner's own `Startup` systems have to build
//! *something* before any network round trip can possibly complete) — the
//! two could genuinely disagree (host on `menger_oscillating_sphere`, a stale
//! or absent `?scene=` on the joiner) and nothing before this caught it.
//! Sending the host's actual, authoritative `Scene` once at connect and
//! having the joiner adopt it wholesale removes the guess entirely: the
//! joiner ends up physically simulating and rendering against the literal
//! transmitted tree, not a same-named-but-possibly-different one built from
//! its own local `scenes.rs` constructor.
//!
//! **Encoding**: `Object::encode` then `Params::encode` then a `u32` count
//! followed by each `(ScalarParam, Expr)` pair — every piece here is
//! already self-delimiting (`Object`/`Fold`/`Expr` recursively, `Params`
//! length-prefixed), so `Scene` itself needs no extra framing beyond
//! sequencing them back to back, same convention as `net.rs`'s tagged
//! messages.

use crate::expr::Expr;
use crate::scene::Scene;
use crate::{Object, Params, ScalarParam};

impl Scene {
    pub fn encode(&self, out: &mut Vec<u8>) {
        self.object.encode(out);
        self.params.encode(out);
        out.extend_from_slice(&(self.animations.len() as u32).to_le_bytes());
        for (handle, expr) in &self.animations {
            handle.encode(out);
            expr.encode(out);
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.encode(&mut out);
        out
    }

    /// Inverse of [`Self::encode`]/[`Self::to_bytes`] — `None` on any
    /// malformed/truncated input, or leftover bytes after a complete
    /// decode (same reasoning as [`Expr::from_bytes`]).
    pub fn from_bytes(bytes: &[u8]) -> Option<Scene> {
        let (object, pos) = Object::decode_at(bytes, 0)?;
        let (params, pos) = Params::decode_at(bytes, pos)?;
        let count = u32::from_le_bytes(bytes.get(pos..pos + 4)?.try_into().ok()?) as usize;
        let mut pos = pos + 4;
        let mut animations = Vec::with_capacity(count);
        for _ in 0..count {
            let (handle, next) = ScalarParam::decode_at(bytes, pos)?;
            let (expr, next) = Expr::decode_at(bytes, next)?;
            animations.push((handle, expr));
            pos = next;
        }
        if pos == bytes.len() {
            Some(Scene { object, params, animations })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fold::Fold;
    use crate::{scenes, Axis, IntValue, Mat2Value, ScalarValue, Vec3Value};
    use glam::{Mat2, Vec2, Vec3, Vec4};

    fn assert_object_round_trips(object: &Object) {
        let bytes = object.to_bytes();
        let decoded = Object::from_bytes(&bytes).unwrap_or_else(|| panic!("failed to decode {object:?}"));
        assert_eq!(format!("{decoded:?}"), format!("{object:?}"), "round-trip mismatch");
    }

    #[test]
    fn sphere_and_cuboid_round_trip() {
        assert_object_round_trips(&Object::Sphere { radius: ScalarValue::Const(1.5) });
        assert_object_round_trips(&Object::Cuboid { half_extent: Vec3Value::Const(Vec3::new(1.0, 2.0, 3.0)) });
    }

    #[test]
    fn param_referencing_values_round_trip() {
        let mut params = Params::new();
        let s = params.alloc_scalar(1.0);
        assert_object_round_trips(&Object::Sphere { radius: ScalarValue::Param(s) });
        let v = params.alloc_vec3(Vec3::ONE);
        assert_object_round_trips(&Object::Cuboid { half_extent: Vec3Value::Param(v) });
    }

    #[test]
    fn every_fold_variant_round_trips_inside_a_fractal() {
        let variants = vec![
            Fold::Abs,
            Fold::Menger,
            Fold::Rotate { axis: Axis::Y, mat: Mat2Value::Const(Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0))) },
            Fold::ScaleTranslate { scale: ScalarValue::Const(0.5), shift: Vec3Value::Const(Vec3::new(1.0, -1.0, 0.5)) },
            Fold::Plane { normal: Vec3Value::Const(Vec3::new(0.0, 1.0, 0.0)), offset: ScalarValue::Const(2.0) },
            Fold::Modulo { axis: Axis::Z, modulus: ScalarValue::Const(3.0) },
            Fold::Series(vec![Fold::Abs, Fold::Menger, Fold::Abs]),
            Fold::Repeat { count: IntValue::Const(4), inner: Box::new(Fold::Abs) },
            Fold::OrbitInit(Vec3Value::Const(Vec3::new(1.0, 0.0, 0.0))),
            Fold::OrbitMax(Vec3Value::Const(Vec3::new(0.0, 1.0, 0.0))),
        ];
        for fold in variants {
            let object = Object::Fractal { fold: fold.clone(), base: Box::new(Object::Sphere { radius: ScalarValue::Const(1.0) }) };
            assert_object_round_trips(&object);
        }
    }

    #[test]
    fn union_intersect_difference_round_trip() {
        let a = || Box::new(Object::Sphere { radius: ScalarValue::Const(1.0) });
        let b = || Box::new(Object::Cuboid { half_extent: Vec3Value::Const(Vec3::ONE) });
        assert_object_round_trips(&Object::Union(a(), b()));
        assert_object_round_trips(&Object::Intersect(a(), b()));
        assert_object_round_trips(&Object::Difference(a(), b()));
    }

    #[test]
    fn deeply_nested_tree_round_trips() {
        let nested = Object::Union(
            Box::new(Object::Fractal {
                fold: Fold::Series(vec![
                    Fold::Abs,
                    Fold::Menger,
                    Fold::Repeat { count: IntValue::Const(3), inner: Box::new(Fold::ScaleTranslate { scale: ScalarValue::Const(2.0), shift: Vec3Value::Const(Vec3::ZERO) }) },
                ]),
                base: Box::new(Object::Cuboid { half_extent: Vec3Value::Const(Vec3::ONE) }),
            }),
            Box::new(Object::Intersect(
                Box::new(Object::Sphere { radius: ScalarValue::Const(4.0) }),
                Box::new(Object::Difference(
                    Box::new(Object::Sphere { radius: ScalarValue::Const(2.0) }),
                    Box::new(Object::Sphere { radius: ScalarValue::Const(1.0) }),
                )),
            )),
        );
        assert_object_round_trips(&nested);
    }

    #[test]
    fn decode_rejects_truncated_or_malformed_bytes() {
        assert!(Object::from_bytes(&[]).is_none());
        assert!(Object::from_bytes(&[3]).is_none(), "Union with no operands");
        assert!(Object::from_bytes(&[255]).is_none(), "unknown tag");
        let mut bytes = Object::Sphere { radius: ScalarValue::Const(1.0) }.to_bytes();
        bytes.push(0xFF);
        assert!(Object::from_bytes(&bytes).is_none(), "leftover bytes after a complete decode");
    }

    #[test]
    fn params_round_trip_including_empty() {
        let mut out = Vec::new();
        Params::new().encode(&mut out);
        let (decoded, consumed) = Params::decode_at(&out, 0).unwrap();
        assert_eq!(consumed, out.len());
        assert_eq!(decoded.slots(), &[] as &[Vec4]);

        let mut params = Params::new();
        params.alloc_scalar(1.0);
        params.alloc_vec3(Vec3::new(1.0, 2.0, 3.0));
        params.alloc_mat2(Mat2::IDENTITY);
        let mut out = Vec::new();
        params.encode(&mut out);
        let (decoded, consumed) = Params::decode_at(&out, 0).unwrap();
        assert_eq!(consumed, out.len());
        assert_eq!(decoded.slots(), params.slots());
    }

    #[test]
    fn scene_round_trips_a_real_scene_with_animations() {
        let mut params = Params::new();
        let (object, handles) = scenes::menger_oscillating_sphere(&mut params);
        crate::scenes::set_menger_params(&mut params, &handles.menger, 5, Vec3::new(0.9, 0.6, 0.2));
        let scene = Scene {
            object,
            animations: vec![(handles.radius, handles.radius_anim.clone())],
            params,
        };
        let bytes = scene.to_bytes();
        let decoded = Scene::from_bytes(&bytes).expect("failed to decode a real scene");
        assert_eq!(decoded.params.slots(), scene.params.slots());
        assert_eq!(decoded.animations.len(), 1);
        assert_eq!(decoded.animations[0].1, scene.animations[0].1);
        // The tree itself must actually behave the same, not just look the
        // same in `Debug` -- probe `de` at a handful of points.
        for p in [Vec4::new(0.0, 0.0, 0.0, 1.0), Vec4::new(2.0, 1.0, 0.5, 1.0), Vec4::new(-3.0, 2.0, 1.0, 1.0)] {
            assert_eq!(decoded.object.de(p, &decoded.params), scene.object.de(p, &scene.params));
        }
    }

    #[test]
    fn scene_round_trips_the_demo_scene() {
        let mut params = Params::new();
        let (object, handles) = scenes::demo_scene(&mut params);
        crate::scenes::set_fractal_params(
            &mut params,
            &handles,
            scenes::beware_of_bumps::SCALE,
            scenes::beware_of_bumps::ANG1,
            scenes::beware_of_bumps::ANG2,
            scenes::beware_of_bumps::SHIFT,
            scenes::beware_of_bumps::COLOR,
            scenes::beware_of_bumps::ITERS,
        );
        let scene = Scene { object, params, animations: Vec::new() };
        let bytes = scene.to_bytes();
        let decoded = Scene::from_bytes(&bytes).expect("failed to decode demo_scene scene");
        for p in [Vec4::new(0.0, 0.0, 0.0, 1.0), Vec4::new(1.0, 0.5, -0.5, 1.0)] {
            assert_eq!(decoded.object.de(p, &decoded.params), scene.object.de(p, &scene.params));
        }
    }

    #[test]
    fn decode_rejects_truncated_scene() {
        let scene = Scene { object: Object::Sphere { radius: ScalarValue::Const(1.0) }, params: Params::new(), animations: Vec::new() };
        let bytes = scene.to_bytes();
        assert!(Scene::from_bytes(&bytes[..bytes.len() - 1]).is_none());
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(Scene::from_bytes(&extra).is_none(), "leftover trailing byte");
    }
}
