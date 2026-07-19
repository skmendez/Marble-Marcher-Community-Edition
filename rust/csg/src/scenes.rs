//! M2: prebuilt scenes.
//! See rust/DESIGN.md §6 and C++ `Scene::GetInitialObject` (src/Scene.cpp) +
//! src/fractals/StaticFractals.hpp.

use glam::{Mat2, Vec2, Vec3};

use crate::fold::Fold;
use crate::object::Object;
use crate::{
    Axis, IntParam, IntValue, Mat2Param, Mat2Value, Params, ScalarParam, ScalarValue, Vec3Param,
    Vec3Value,
};

/// Parameter handles for the classic Marble Marcher fractal tree, so callers
/// can animate it via [`set_fractal_params`] without a shader/tree rebuild.
#[derive(Clone, Copy, Debug)]
pub struct ClassicHandles {
    pub scale: ScalarParam,
    pub rot1: Mat2Param,
    pub rot2: Mat2Param,
    pub shift: Vec3Param,
    pub color: Vec3Param,
    pub iters: IntParam,
}

/// Builds a rotation matrix for `FoldRotate` from an angle, per the
/// convention fixed in DESIGN.md §4: `M = [[cos, -sin], [sin, cos]]`
/// (column-major `Mat2::from_cols`), giving `x' = c·x + s·y`, `y' = -s·x + c·y`
/// — identical to MMCE's original hard-coded `rotZ`.
pub fn rotation_mat2(angle: f32) -> Mat2 {
    let (s, c) = angle.sin_cos();
    Mat2::from_cols(Vec2::new(c, -s), Vec2::new(s, c))
}

/// The classic Marble Marcher fractal (C++ `Scene::GetInitialObject`'s
/// `fractal`, src/Scene.cpp): a Menger-sponge-like Abs/Rotate/Menger/Rotate/
/// ScaleTranslate loop, repeated `iters` times, folded into a cuboid.
pub fn classic(params: &mut Params) -> (Object, ClassicHandles) {
    let scale = params.alloc_scalar(1.0);
    let rot1 = params.alloc_mat2(Mat2::IDENTITY);
    let rot2 = params.alloc_mat2(Mat2::IDENTITY);
    let shift = params.alloc_vec3(Vec3::ZERO);
    let color = params.alloc_vec3(Vec3::ONE);
    let iters = params.alloc_int(0);

    let handles = ClassicHandles {
        scale,
        rot1,
        rot2,
        shift,
        color,
        iters,
    };

    let inner = Fold::Series(vec![
        Fold::Abs,
        Fold::Rotate {
            axis: Axis::Z,
            mat: Mat2Value::Param(rot1),
        },
        Fold::Menger,
        Fold::Rotate {
            axis: Axis::X,
            mat: Mat2Value::Param(rot2),
        },
        Fold::ScaleTranslate {
            scale: ScalarValue::Param(scale),
            shift: Vec3Value::Param(shift),
        },
        Fold::OrbitMax(Vec3Value::Param(color)),
    ]);

    let fold = Fold::Series(vec![
        Fold::OrbitInit(Vec3Value::Const(Vec3::ZERO)),
        Fold::Repeat {
            count: IntValue::Param(iters),
            inner: Box::new(inner),
        },
    ]);

    let object = Object::Fractal {
        fold,
        base: Box::new(Object::Cuboid {
            half_extent: Vec3Value::Const(Vec3::splat(6.0)),
        }),
    };

    (object, handles)
}

/// "Creme repeating spheres in a sphere" (C++ `BlackRepeatingCubesInSphere`,
/// src/fractals/StaticFractals.hpp — despite the C++ name it repeats small
/// spheres, not cubes).
pub fn creme_spheres() -> Object {
    let modulus = ScalarValue::Const(0.75);
    let fold = Fold::Series(vec![
        Fold::OrbitInit(Vec3Value::Const(Vec3::new(0.90, 0.80, 0.56))),
        Fold::Modulo {
            axis: Axis::X,
            modulus,
        },
        Fold::Modulo {
            axis: Axis::Y,
            modulus,
        },
        Fold::Modulo {
            axis: Axis::Z,
            modulus,
        },
    ]);
    let cubes = Object::Fractal {
        fold,
        base: Box::new(Object::Sphere {
            radius: ScalarValue::Const(0.1),
        }),
    };
    Object::Intersect(
        Box::new(cubes),
        Box::new(Object::Sphere {
            radius: ScalarValue::Const(6.0),
        }),
    )
}

/// The full demo scene (C++ `Scene::GetInitialObject`, src/Scene.cpp):
/// `classic` unioned with `creme_spheres`.
pub fn demo_scene(params: &mut Params) -> (Object, ClassicHandles) {
    let (classic_obj, handles) = classic(params);
    let object = Object::Union(Box::new(classic_obj), Box::new(creme_spheres()));
    (object, handles)
}

/// Parameter handles for [`menger_sponge`]/[`menger_sphere`].
#[derive(Clone, Copy, Debug)]
pub struct MengerHandles {
    pub depth: IntParam,
    pub color: Vec3Param,
}

/// Writes a parameter set for [`menger_sponge`]/[`menger_sphere`].
pub fn set_menger_params(params: &mut Params, handles: &MengerHandles, depth: i32, color: Vec3) {
    params.set_int(handles.depth, depth);
    params.set_vec3(handles.color, color);
}

/// A "true" recursive Menger sponge (C++ `MengerSponge`,
/// src/fractals/StaticFractals.hpp) — distinct from [`classic`]'s
/// fractal: this one folds a `Plane` each iteration (rather than the two
/// `Rotate`s `classic` uses), which is what gives it the classic Menger
/// sponge look rather than the original game's twisted variant. Folded into
/// a unit cuboid, then scaled down by `0.33` as a final step.
pub fn menger_sponge(params: &mut Params) -> (Object, MengerHandles) {
    let depth = params.alloc_int(0);
    let color = params.alloc_vec3(Vec3::ONE);
    let handles = MengerHandles { depth, color };

    let inner = Fold::Series(vec![
        Fold::Abs,
        Fold::Menger,
        Fold::ScaleTranslate {
            scale: ScalarValue::Const(3.0),
            shift: Vec3Value::Const(Vec3::new(-2.0, -2.0, 0.0)),
        },
        Fold::Plane {
            normal: Vec3Value::Const(Vec3::new(0.0, 0.0, -1.0)),
            offset: ScalarValue::Const(-1.0),
        },
        Fold::OrbitMax(Vec3Value::Param(color)),
    ]);

    let loop_fold = Fold::Repeat {
        count: IntValue::Param(depth),
        inner: Box::new(inner),
    };

    let series2 = Fold::Series(vec![
        Fold::OrbitInit(Vec3Value::Const(Vec3::ZERO)),
        loop_fold,
    ]);

    let final_series = Fold::Series(vec![
        Fold::ScaleTranslate {
            scale: ScalarValue::Const(0.33),
            shift: Vec3Value::Const(Vec3::ZERO),
        },
        series2,
    ]);

    let object = Object::Fractal {
        fold: final_series,
        base: Box::new(Object::Cuboid {
            half_extent: Vec3Value::Const(Vec3::ONE),
        }),
    };

    (object, handles)
}

/// [`menger_sponge`] with a radius-3 spherical bite taken out of it (C++
/// `MengerSphere`, src/fractals/StaticFractals.hpp — an `ObjectDifference`,
/// "just for the fun of it" per this repo's commit history).
pub fn menger_sphere(params: &mut Params) -> (Object, MengerHandles) {
    let (sponge, handles) = menger_sponge(params);
    let object = Object::Difference(
        Box::new(sponge),
        Box::new(Object::Sphere {
            radius: ScalarValue::Const(3.0),
        }),
    );
    (object, handles)
}

/// Writes a full parameter set for the classic fractal tree built by
/// [`classic`]/[`demo_scene`]. `ang1`/`ang2` are turned into rotation
/// matrices via [`rotation_mat2`].
#[allow(clippy::too_many_arguments)]
pub fn set_fractal_params(
    params: &mut Params,
    handles: &ClassicHandles,
    scale: f32,
    ang1: f32,
    ang2: f32,
    shift: Vec3,
    color: Vec3,
    iters: i32,
) {
    params.set_scalar(handles.scale, scale);
    params.set_mat2(handles.rot1, rotation_mat2(ang1));
    params.set_mat2(handles.rot2, rotation_mat2(ang2));
    params.set_vec3(handles.shift, shift);
    params.set_vec3(handles.color, color);
    params.set_int(handles.iters, iters);
}

/// Level values for the demo scene, "Beware Of Bumps" (extracted from the
/// binary `.lvl`; DESIGN.md §6).
pub mod beware_of_bumps {
    use glam::Vec3;

    pub const ITERS: i32 = 16;
    pub const SCALE: f32 = 1.66;
    pub const ANG1: f32 = 1.52;
    pub const ANG2: f32 = 0.19;
    pub const SHIFT: Vec3 = Vec3::new(-3.83, -1.94, -1.09);
    pub const COLOR: Vec3 = Vec3::new(0.42, 0.38, 0.19);
    pub const MARBLE_RAD: f32 = 0.02;
    pub const START: Vec3 = Vec3::new(0.681, 2.8, 2.528);
    pub const KILL_Y: f32 = -4.0;
    pub const ORBIT_DIST: f32 = 3.1;
    pub const SUN_COL: Vec3 = Vec3::new(1.0, 0.95, 0.8);
    pub const BG: Vec3 = Vec3::new(0.6, 0.8, 1.0);
    /// Raw (pre-normalization) sun direction from the level file.
    const SUN_DIR_RAW: Vec3 = Vec3::new(0.637, 0.771, 0.017);

    /// Unit sun direction (toward the sun).
    pub fn sun_dir() -> Vec3 {
        SUN_DIR_RAW.normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec4;

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

    #[test]
    fn demo_scene_de_at_marble_start_is_positive_and_finite() {
        let (object, params) = setup_demo();
        let start = beware_of_bumps::START;
        let d = object.de(Vec4::new(start.x, start.y, start.z, 1.0), &params);
        assert!(d.is_finite());
        assert!(
            d > 0.0,
            "expected marble start to be outside the fractal, got {d}"
        );
    }

    #[test]
    fn demo_scene_de_far_away_is_large() {
        let (object, params) = setup_demo();
        let d = object.de(Vec4::new(0.0, 50.0, 0.0, 1.0), &params);
        assert!(d.is_finite());
        assert!(d > 10.0, "expected a large DE far from the scene, got {d}");
    }

    #[test]
    fn demo_scene_nearest_point_is_surface_consistent() {
        let (object, params) = setup_demo();
        let probes = [
            Vec3::new(0.681, 2.8, 2.528),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.13, 0.07, -0.11),
            Vec3::new(-3.0, 1.0, 2.0),
            Vec3::new(5.0, 5.0, 5.0),
        ];
        for probe in probes {
            let p = Vec4::new(probe.x, probe.y, probe.z, 1.0);
            let d = object.de(p, &params).abs();
            let np = object.nearest_point(p, &params);
            assert!(
                np.is_finite(),
                "non-finite nearest_point for probe {probe:?}"
            );
            let actual = (probe - np).length();
            // The DE is a lower bound on the true distance under scaling/folding;
            // require the true distance not to wildly disagree with it either way.
            assert!(
                actual <= 2.0 * d.max(1e-4) && d <= 2.0 * actual.max(1e-4),
                "probe {probe:?}: |p-np|={actual} vs de={d}"
            );
        }
    }

    #[test]
    fn classic_alone_has_reasonable_de() {
        let mut params = Params::new();
        let (object, handles) = classic(&mut params);
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
        let d = object.de(Vec4::new(0.0, 0.0, 0.0, 1.0), &params);
        assert!(d.is_finite());
    }

    #[test]
    fn creme_spheres_is_bounded_by_outer_sphere() {
        let params = Params::new();
        let object = creme_spheres();
        // Well outside the bounding sphere (radius 6): DE should be positive.
        let d = object.de(Vec4::new(20.0, 0.0, 0.0, 1.0), &params);
        assert!(d.is_finite());
        assert!(d > 0.0);
    }

    #[test]
    fn menger_sponge_has_reasonable_de() {
        let mut params = Params::new();
        let (object, handles) = menger_sponge(&mut params);
        set_menger_params(&mut params, &handles, 12, Vec3::new(1.0, 0.5, 0.2));
        let d = object.de(Vec4::new(0.0, 0.0, 0.0, 1.0), &params);
        assert!(d.is_finite());
        // Far outside the (roughly unit-scale, after the final 0.33 shrink)
        // sponge: DE should be positive and reasonably large.
        let d_far = object.de(Vec4::new(20.0, 0.0, 0.0, 1.0), &params);
        assert!(d_far.is_finite());
        assert!(d_far > 5.0, "expected a large DE far away, got {d_far}");
    }

    #[test]
    fn menger_sphere_bites_a_cavity_out_of_the_sponge() {
        let mut params = Params::new();
        let (object, handles) = menger_sphere(&mut params);
        set_menger_params(&mut params, &handles, 12, Vec3::new(1.0, 0.5, 0.2));
        // At the origin, the sponge is solid but it's well inside the
        // radius-3 cavity sphere too, so the difference should read as
        // outside (positive DE) -- the bite removed material here.
        let d = object.de(Vec4::new(0.0, 0.0, 0.0, 1.0), &params);
        assert!(d.is_finite());
        assert!(
            d > 0.0,
            "expected the origin to be inside the carved-out cavity, got de={d}"
        );
    }
}
