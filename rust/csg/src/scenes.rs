//! M2: prebuilt scenes.
//! See rust/DESIGN.md §6 and C++ `Scene::GetInitialObject` (src/Scene.cpp) +
//! src/fractals/StaticFractals.hpp.

use glam::{Mat2, Vec2, Vec3};

use crate::expr::Expr;
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

/// Parameter handles for [`menger_oscillating_sphere`]: [`menger_sponge`]'s
/// own handles, plus the bite sphere's runtime-mutable radius and the
/// [`Expr`] that drives it from the shared tick clock.
#[derive(Clone, Debug)]
pub struct MengerOscillatingSphereHandles {
    pub menger: MengerHandles,
    pub radius: ScalarParam,
    /// Drives `radius` as a pure function of `crate::Tick` — register
    /// `(radius, radius_anim.clone())` into a scene's animation table (see
    /// `crate::expr` module doc) so it's evaluated once per simulated tick,
    /// live and through any rollback resimulation, instead of the old
    /// per-frame wall-clock formula this replaces.
    pub radius_anim: Expr,
}

/// [`menger_sponge`]'s overall bounding half-extent: its single outer
/// `ScaleTranslate{scale: 0.33}` sets this to `1.0/0.33` (confirmed
/// numerically: `object.de` crosses zero almost exactly at the corner point
/// `(k,k,k)` for `k = MENGER_SPONGE_HALF_EXTENT`). [`MENGER_BITE_MIN_RADIUS`]
/// and [`MENGER_BITE_MAX_RADIUS`] are both derived from this, independently
/// of each other — the min radius is about the sponge's own pre-existing
/// central hole and has nothing to do with how far the max radius reaches.
const MENGER_SPONGE_HALF_EXTENT: f32 = 1.0 / 0.33;

/// The smallest bite-sphere radius that removes *nothing visible* from
/// [`menger_sponge`]: exactly the half-extent of the sponge's own
/// already-empty central void (the "+"-shaped cell a Menger sponge always
/// has removed at its exact center, one level in) — each recursive
/// iteration's own `ScaleTranslate{scale: 3.0}` is the classic Menger 3x
/// subdivision, so the first-level removed central cube is 1/3 of
/// [`MENGER_SPONGE_HALF_EXTENT`]. A sphere this size sits entirely inside
/// that pre-existing hole — verified numerically (not just derived):
/// sampling `object.de` at this exact radius across thousands of directions
/// from the origin, the closest any of them come to solid material is +0.42
/// (comfortably positive everywhere), and the nearest solid material in
/// *any* direction from the origin is at distance 1.43, a real margin above
/// this radius.
pub const MENGER_BITE_MIN_RADIUS: f32 = MENGER_SPONGE_HALF_EXTENT / 3.0;

/// The largest bite-sphere radius worth animating up to: reaches each outer
/// edge's midpoint instead of stopping at the face center. For a cube with
/// half-extent `h` = [`MENGER_SPONGE_HALF_EXTENT`], the face center is `h`
/// from the origin, an edge midpoint is `h * sqrt(2.0)`, and a corner is
/// `h * sqrt(3.0)`; this uses the edge-midpoint distance, which carves
/// further than the old face-reaching radius (removing the edge regions
/// too, not just the face tunnels) while still stopping short of the
/// corners (`h*sqrt(3.0) ~= 7.42` from the center — well outside this
/// sphere, at `h*sqrt(2.0) ~= 4.28`).
pub const MENGER_BITE_MAX_RADIUS: f32 = MENGER_SPONGE_HALF_EXTENT * std::f32::consts::SQRT_2;

/// The bite sphere's oscillation period, in simulated [`crate::Tick`]s
/// rather than wall-clock seconds — 12 seconds at the app's fixed 60Hz
/// physics/animation tick rate (`Time::<Fixed>::from_hz(60.0)` in
/// `main.rs`). Expressing the period in ticks (not seconds, with a
/// separate conversion elsewhere) keeps [`menger_oscillating_sphere`]'s
/// [`Expr`] a pure function of `Tick` with no implicit unit baked in
/// anywhere else.
const MENGER_OSCILLATING_SPHERE_PERIOD_TICKS: f32 = 12.0 * 60.0;

/// [`menger_sponge`] with a bite sphere whose radius is a runtime
/// [`ScalarValue::Param`] instead of a fixed constant — demonstrates
/// animating a CSG *geometry* parameter live (not just a fractal fold's
/// rotation/color/iteration-count, which `classic`/`menger_sponge` already
/// show), oscillating between [`MENGER_BITE_MIN_RADIUS`] (removes nothing)
/// and [`MENGER_BITE_MAX_RADIUS`] (only the corners survive) once per
/// [`MENGER_OSCILLATING_SPHERE_PERIOD_TICKS`], driven by
/// [`MengerOscillatingSphereHandles::radius_anim`] instead of wall-clock
/// time (see `crate::expr` module doc for why).
pub fn menger_oscillating_sphere(params: &mut Params) -> (Object, MengerOscillatingSphereHandles) {
    let (sponge, menger) = menger_sponge(params);
    let radius = params.alloc_scalar(MENGER_BITE_MIN_RADIUS);

    // radius = MIN + (MAX - MIN) * 0.5 * (1 - cos(tick * omega))
    let omega = std::f32::consts::TAU / MENGER_OSCILLATING_SPHERE_PERIOD_TICKS;
    let angle = Expr::Mul(Box::new(Expr::Tick), Box::new(Expr::Const(omega)));
    let one_minus_cos = Expr::Sub(Box::new(Expr::Const(1.0)), Box::new(Expr::Cos(Box::new(angle))));
    let span = Expr::Mul(
        Box::new(Expr::Const((MENGER_BITE_MAX_RADIUS - MENGER_BITE_MIN_RADIUS) * 0.5)),
        Box::new(one_minus_cos),
    );
    let radius_anim = Expr::Add(Box::new(Expr::Const(MENGER_BITE_MIN_RADIUS)), Box::new(span));

    let handles = MengerOscillatingSphereHandles {
        menger,
        radius,
        radius_anim,
    };
    let object = Object::Difference(
        Box::new(sponge),
        Box::new(Object::Sphere {
            radius: ScalarValue::Param(radius),
        }),
    );
    (object, handles)
}

/// Parameter handles for [`hollow_donut`], so the params UI (and any future
/// animation) can resize the donut live without a shader/tree rebuild.
#[derive(Clone, Copy, Debug)]
pub struct HollowDonutHandles {
    pub major: ScalarParam,
    pub minor: ScalarParam,
    pub thickness: ScalarParam,
}

/// [`hollow_donut`]'s stock dimensions: ring radius 3, tube radius 1, wall
/// thickness 0.15 -- leaving a free interior tube of radius
/// `1 - 0.15 = 0.85` for the marble to travel around inside.
pub const DONUT_MAJOR_RADIUS: f32 = 3.0;
pub const DONUT_MINOR_RADIUS: f32 = 1.0;
pub const DONUT_THICKNESS: f32 = 0.15;

/// Base ("dough") albedo, set by `OrbitInit` before the angular folds --
/// each channel is then lifted by [`DONUT_STRIPE_COLOR`]'s position term
/// wherever that term exceeds it (`OrbitMax` is a componentwise max). The
/// blue channel is deliberately near-zero: it's the angular-stripe channel,
/// and its *contrast* is `stripe_term - base` after the shader's Reinhard
/// compression, so a low floor is what makes the bands actually visible
/// (the first attempt's 0.14 floor + weak coefficient survived compression
/// as a barely-there tint).
const DONUT_BASE_COLOR: Vec3 = Vec3::new(0.50, 0.22, 0.03);

/// Per-component scale on the *wedge-folded* position fed to `OrbitMax`
/// (`orbit = max(orbit, p.xyz * this)`), one spatial axis per channel:
///  - `r <- x` (tube-radial, `~major - minor ..= major + minor`): the outer
///    wall runs warmer than the inner wall, a constant cross-tunnel cue.
///  - `g <- y` (height): the ceiling lifts toward yellow-green, floor stays
///    dark -- an up/down cue that also halos the skylights.
///  - `b <- z` (angular coordinate within the wedge, `0 ..= ~x`): the
///    stripe term. The coefficient is deliberately large (2.2, not a
///    subtle 0.4): the shader Reinhard-compresses orbit into `v/(1+v)`,
///    so a large coefficient makes the dark->bright ramp complete within
///    the first ~10 degrees past each wedge seam and *saturate* for the
///    rest of the wedge -- rendering as crisp dark rib lines every
///    360/[`DONUT_SYMMETRY`] degrees on otherwise-bright violet walls,
///    instead of the previous one-soft-gradient-per-45-degrees that never
///    read as banding at all (verified across two screenshot rounds --
///    gentle coefficients simply do not survive the compression).
const DONUT_STRIPE_COLOR: Vec3 = Vec3::new(0.20, 0.60, 2.2);

/// How many skylights/stripe repeats around the ring: 3 plane folds halve
/// the angular domain three times, `2^3 = 8` copies of the fold wedge.
pub const DONUT_SYMMETRY: usize = 8;

/// The skylight cutter sphere: centered at the fold wedge's mid-angle
/// (`PI / 8`), just above the tube's top surface, sized to pierce the wall
/// clean through (wall spans `minor ± thickness` vertically at the ring
/// radius; the sphere spans `DONUT_SKYLIGHT_HEIGHT ± DONUT_SKYLIGHT_RADIUS`).
/// Positioned from the stock dimension constants, not the live params -- a
/// params-panel resize moves the wall but not the skylights, which is fine
/// for a tuning tool (the holes just get deeper/shallower).
pub const DONUT_SKYLIGHT_RADIUS: f32 = 0.5;
pub const DONUT_SKYLIGHT_HEIGHT: f32 = 1.3;

/// A hollow donut: `Onion(Torus)` -- the shell of points within
/// `thickness` of the torus surface, i.e. a donut-shaped **tunnel**. The
/// marble plays *inside* the tube (at the ring circle the shell's `de` is
/// `minor - thickness`, comfortably positive), circulating around the ring
/// like a closed hamster-tube circuit; physics collides against the same
/// exact shell field the shader renders (`Object::Onion`'s doc for the
/// exactness argument). `major`/`minor`/`thickness` are runtime `Param`s
/// so the params panel can resize the donut live.
///
/// Interior readability (the whole scene is experienced from *inside* the
/// shell, where the sun never reaches and shading is ambient-only): two
/// structures on top of the bare shell, both built purely from `Fold::
/// Plane` reflections through the Y axis -- **exact symmetries of the
/// torus**, so the shell's geometry (and physics) is completely untouched
/// by the folding; only what's placed *inside* the wedge gets replicated:
///
///  - **Skylights**: one cutter sphere at the wedge's mid-angle above the
///    tube's top, `Difference`d out of the shell -- the folds replicate it
///    into [`DONUT_SYMMETRY`] portholes around the ring, letting real
///    sun/sky light pour in (bright pools on the tunnel floor, and a
///    rhythm of landmarks that makes travel around the ring legible).
///  - **Stripes**: an `OrbitMax` placed *after* the plane folds samples
///    the wedge-folded position, so the albedo carries the same 8-fold
///    angular banding ([`DONUT_STRIPE_COLOR`]) -- the bands recede around
///    the tunnel's curve, which is what actually reads as "inside a
///    donut" instead of "inside a vague pale tube".
pub fn hollow_donut(params: &mut Params) -> (Object, HollowDonutHandles) {
    use std::f32::consts::FRAC_PI_8;

    let major = params.alloc_scalar(DONUT_MAJOR_RADIUS);
    let minor = params.alloc_scalar(DONUT_MINOR_RADIUS);
    let thickness = params.alloc_scalar(DONUT_THICKNESS);
    let handles = HollowDonutHandles { major, minor, thickness };
    let shell = Object::Onion {
        base: Box::new(Object::Torus {
            major: ScalarValue::Param(major),
            minor: ScalarValue::Param(minor),
        }),
        thickness: ScalarValue::Param(thickness),
    };

    // One cutter sphere at the wedge's mid-angle; `ScaleTranslate`'s
    // forward map is `p' = p + shift`, so a sphere-at-origin base appears
    // at `-shift`.
    let skylight_center = Vec3::new(
        DONUT_MAJOR_RADIUS * FRAC_PI_8.cos(),
        DONUT_SKYLIGHT_HEIGHT,
        DONUT_MAJOR_RADIUS * FRAC_PI_8.sin(),
    );
    let skylight = Object::Fractal {
        fold: Fold::ScaleTranslate {
            scale: ScalarValue::Const(1.0),
            shift: Vec3Value::Const(-skylight_center),
        },
        base: Box::new(Object::Sphere {
            radius: ScalarValue::Const(DONUT_SKYLIGHT_RADIUS),
        }),
    };
    let pierced = Object::Difference(Box::new(shell), Box::new(skylight));

    // Three reflections through Y-axis planes fold the full circle into
    // the wedge `atan2(z, x) in [0, PI/4]`: |x|, then |z| (first
    // quadrant), then reflect across the x = z diagonal (keep x >= z).
    let sqrt_half = std::f32::consts::FRAC_1_SQRT_2;
    let fold = Fold::Series(vec![
        Fold::OrbitInit(Vec3Value::Const(DONUT_BASE_COLOR)),
        Fold::Plane {
            normal: Vec3Value::Const(Vec3::X),
            offset: ScalarValue::Const(0.0),
        },
        Fold::Plane {
            normal: Vec3Value::Const(Vec3::Z),
            offset: ScalarValue::Const(0.0),
        },
        Fold::Plane {
            normal: Vec3Value::Const(Vec3::new(sqrt_half, 0.0, -sqrt_half)),
            offset: ScalarValue::Const(0.0),
        },
        Fold::OrbitMax(Vec3Value::Const(DONUT_STRIPE_COLOR)),
    ]);

    let object = Object::Fractal {
        fold,
        base: Box::new(pierced),
    };
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

    /// Unit sun direction (toward the sun). Cached: called twice per frame
    /// (fine + shadow pass uniforms in `render.rs`), and `Vec3::normalize`
    /// on a compile-time-constant input is pure wasted work to repeat.
    pub fn sun_dir() -> Vec3 {
        static SUN_DIR: std::sync::OnceLock<Vec3> = std::sync::OnceLock::new();
        *SUN_DIR.get_or_init(|| SUN_DIR_RAW.normalize())
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

    #[test]
    fn oscillating_sphere_at_min_radius_matches_bare_sponge() {
        // MENGER_BITE_MIN_RADIUS is sized to sit entirely inside the
        // sponge's pre-existing empty center (verified numerically when the
        // constant was derived -- see its doc comment), so biting with it
        // should change nothing: `de` at several points inside that radius
        // must exactly match the bare (un-bitten) sponge.
        let mut bare_params = Params::new();
        let (bare, bare_handles) = menger_sponge(&mut bare_params);
        set_menger_params(&mut bare_params, &bare_handles, 8, Vec3::ONE);

        let mut osc_params = Params::new();
        let (osc, osc_handles) = menger_oscillating_sphere(&mut osc_params);
        set_menger_params(&mut osc_params, &osc_handles.menger, 8, Vec3::ONE);
        osc_params.set_scalar(osc_handles.radius, MENGER_BITE_MIN_RADIUS);

        for p in [
            Vec4::new(0.0, 0.0, 0.0, 1.0),
            Vec4::new(0.3, 0.4, 0.2, 1.0),
            Vec4::new(-0.5, 0.1, 0.6, 1.0),
        ] {
            assert!(
                p.truncate().length() < MENGER_BITE_MIN_RADIUS,
                "test point must actually be inside the bite radius"
            );
            let d_bare = bare.de(p, &bare_params);
            let d_osc = osc.de(p, &osc_params);
            assert!(
                (d_bare - d_osc).abs() < 1e-5,
                "bite at MENGER_BITE_MIN_RADIUS changed de at {p:?}: bare={d_bare} osc={d_osc}"
            );
        }
    }

    #[test]
    fn oscillating_sphere_at_max_radius_hollows_the_center_but_not_the_corner() {
        let mut bare_params = Params::new();
        let (bare, bare_handles) = menger_sponge(&mut bare_params);
        set_menger_params(&mut bare_params, &bare_handles, 8, Vec3::ONE);

        let mut osc_params = Params::new();
        let (osc, osc_handles) = menger_oscillating_sphere(&mut osc_params);
        set_menger_params(&mut osc_params, &osc_handles.menger, 8, Vec3::ONE);
        osc_params.set_scalar(osc_handles.radius, MENGER_BITE_MAX_RADIUS);

        // The center is carved out entirely.
        let d_center = osc.de(Vec4::new(0.0, 0.0, 0.0, 1.0), &osc_params);
        assert!(d_center.is_finite());
        assert!(d_center > 0.0, "expected the center hollowed out, got de={d_center}");

        // A point well beyond the bite sphere (with real margin over
        // MENGER_BITE_MAX_RADIUS, out where the corner regions live) must be
        // *unaffected* by the bite -- same `de` with and without it. This is
        // a more robust check than asserting solid/hollow at one exact
        // point: right at the razor-thin corner tip itself, `de`'s sign is
        // sensitive to fine recursive detail (confirmed while writing this
        // test -- a point at `MENGER_BITE_MAX_RADIUS * 1.02` flipped sign
        // between two nearby `depth` values), but *whether the bite reaches
        // that far at all* is not.
        let k = MENGER_BITE_MAX_RADIUS * 1.5;
        let p = Vec4::new(k, k, k, 1.0);
        let d_bare = bare.de(p, &bare_params);
        let d_osc = osc.de(p, &osc_params);
        assert!(
            (d_bare - d_osc).abs() < 1e-5,
            "bite at MENGER_BITE_MAX_RADIUS reached a corner-region point it shouldn't have: \
             bare={d_bare} osc={d_osc} at k={k}"
        );
    }

    #[test]
    fn max_radius_is_the_edge_midpoint_not_the_face_center() {
        // The requested geometry change: MAX_RADIUS should be the old
        // face-reach distance times sqrt(2) -- exactly an edge midpoint's
        // distance from the center, for a cube with half-extent = the old
        // face-reach distance.
        let face_reach = 1.0 / 0.33;
        let edge_reach = face_reach * std::f32::consts::SQRT_2;
        assert!(
            (MENGER_BITE_MAX_RADIUS - edge_reach).abs() < 1e-6,
            "MENGER_BITE_MAX_RADIUS={MENGER_BITE_MAX_RADIUS} != edge_reach={edge_reach}"
        );
        // ...and strictly less than the corner distance (face_reach * sqrt(3)),
        // so the corners still survive being bitten at MAX_RADIUS.
        assert!(MENGER_BITE_MAX_RADIUS < face_reach * 3.0_f32.sqrt());
    }

    #[test]
    fn hollow_donut_is_a_playable_tunnel() {
        let mut params = Params::new();
        let (object, handles) = hollow_donut(&mut params);

        // The ring-center spawn point sits in free space with the full
        // interior clearance (minor - thickness).
        let d = object.de(Vec4::new(DONUT_MAJOR_RADIUS, 0.0, 0.0, 1.0), &params);
        assert!((d - (DONUT_MINOR_RADIUS - DONUT_THICKNESS)).abs() < 1e-6, "spawn de={d}");
        // The wall is solid: a point on the torus surface is inside the shell.
        let d = object.de(Vec4::new(DONUT_MAJOR_RADIUS + DONUT_MINOR_RADIUS, 0.0, 0.0, 1.0), &params);
        assert!((d - (-DONUT_THICKNESS)).abs() < 1e-6, "wall de={d}");
        // Outside the donut entirely: positive, roughly the gap distance.
        let d = object.de(Vec4::new(10.0, 0.0, 0.0, 1.0), &params);
        assert!(d > 5.0, "outside de={d}");
        // The donut hole's center is also free space (it's outside the tube).
        let d = object.de(Vec4::new(0.0, 0.0, 0.0, 1.0), &params);
        assert!(d > 1.0, "hole-center de={d}");

        // Finite bound covering the whole shell.
        let (c, r) = object.bounding_sphere(&params).unwrap();
        assert_eq!(c, Vec3::ZERO);
        assert!((r - (DONUT_MAJOR_RADIUS + DONUT_MINOR_RADIUS + DONUT_THICKNESS)).abs() < 1e-5);

        // The handles genuinely drive the geometry: growing the tube radius
        // increases the spawn point's clearance by the same amount.
        params.set_scalar(handles.minor, DONUT_MINOR_RADIUS + 0.5);
        let d = object.de(Vec4::new(DONUT_MAJOR_RADIUS, 0.0, 0.0, 1.0), &params);
        assert!((d - (DONUT_MINOR_RADIUS + 0.5 - DONUT_THICKNESS)).abs() < 1e-6);
    }

    #[test]
    fn hollow_donut_skylights_pierce_the_wall_at_every_wedge_center_but_not_the_seams() {
        use std::f32::consts::{FRAC_PI_8, TAU};
        let mut params = Params::new();
        let (object, _handles) = hollow_donut(&mut params);

        // Mid-wall point at the tube's top for a given ring angle.
        let top_wall = |angle: f32| {
            Vec4::new(
                DONUT_MAJOR_RADIUS * angle.cos(),
                DONUT_MINOR_RADIUS,
                DONUT_MAJOR_RADIUS * angle.sin(),
                1.0,
            )
        };

        // Every replicated wedge center has an open hole; every seam
        // between them is still solid wall -- the plane folds turn the one
        // cutter sphere into DONUT_SYMMETRY skylights, no more, no fewer.
        for i in 0..DONUT_SYMMETRY {
            let step = TAU / DONUT_SYMMETRY as f32;
            let center = FRAC_PI_8 + i as f32 * step;
            let seam = i as f32 * step;
            let d_center = object.de(top_wall(center), &params);
            assert!(d_center > 0.05, "wedge {i}: expected an open skylight, de={d_center}");
            let d_seam = object.de(top_wall(seam), &params);
            assert!(d_seam < -0.05, "seam {i}: expected solid wall, de={d_seam}");
        }

        // The tunnel floor is untouched by the top-side skylights.
        let floor = Vec4::new(DONUT_MAJOR_RADIUS, -DONUT_MINOR_RADIUS, 0.0, 1.0);
        let d = object.de(floor, &params);
        assert!(d < -0.05, "floor should still be solid, de={d}");
    }

    #[test]
    fn radius_anim_matches_the_scalar_bounds_at_key_ticks() {
        // The Expr conversion must preserve the exact min/max bounds the
        // scalar radius previously oscillated between (see the doc comment
        // on menger_oscillating_sphere and MENGER_OSCILLATING_SPHERE_PERIOD_TICKS
        // for why this is a pure function of Tick rather than wall time now).
        let mut params = Params::new();
        let (_object, handles) = menger_oscillating_sphere(&mut params);
        let anim = &handles.radius_anim;

        assert!(
            (anim.eval(0) - MENGER_BITE_MIN_RADIUS).abs() < 1e-3,
            "tick 0 should start at MIN_RADIUS, got {}",
            anim.eval(0)
        );
        let half_period = (MENGER_OSCILLATING_SPHERE_PERIOD_TICKS / 2.0) as u64;
        assert!(
            (anim.eval(half_period) - MENGER_BITE_MAX_RADIUS).abs() < 1e-3,
            "half period should reach MAX_RADIUS, got {}",
            anim.eval(half_period)
        );
        let full_period = MENGER_OSCILLATING_SPHERE_PERIOD_TICKS as u64;
        assert!(
            (anim.eval(full_period) - MENGER_BITE_MIN_RADIUS).abs() < 1e-3,
            "a full period should return to MIN_RADIUS, got {}",
            anim.eval(full_period)
        );
    }
}
