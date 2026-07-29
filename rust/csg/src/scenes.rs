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
    /// Barber-pole stripe periods around the ring (toroidal).
    pub ring_count: IntParam,
    /// Barber-pole stripe periods around the tube (poloidal) -- the
    /// stripes' visual tilt is the ratio of the two counts.
    pub twist_count: IntParam,
}

/// [`hollow_donut`]'s stock dimensions: ring radius 3, tube radius 1, wall
/// thickness 0.15 -- leaving a free interior tube of radius
/// `1 - 0.15 = 0.85` for the marble to travel around inside.
pub const DONUT_MAJOR_RADIUS: f32 = 3.0;
pub const DONUT_MINOR_RADIUS: f32 = 1.0;
pub const DONUT_THICKNESS: f32 = 0.15;

/// Barber-pole stripe colors, fed to [`Fold::OrbitBarberPole`] (which
/// replaced the earlier `OrbitInit`/`OrbitMax` wedge-fold scheme -- that
/// algebra could only produce mirror-symmetric triangle-wave bands locked
/// to the geometry's D4 symmetry, 4 ribs at the cardinals, and never a
/// helix; see the fold variant's doc for the expressiveness argument).
/// Classic red/white, with raw values chosen for what survives the
/// shader's albedo pipeline (Reinhard compression, material-gamma
/// squaring, lighting, ACES): white starts well above 1.0 so it still
/// displays near-white after compression+squaring, and the red keeps
/// near-zero green -- the hard-won lesson from this scene's earlier
/// palettes: any meaningful green content reads yellow under direct sun.
const DONUT_POLE_RED: Vec3 = Vec3::new(1.3, 0.05, 0.06);
const DONUT_POLE_WHITE: Vec3 = Vec3::new(2.5, 2.5, 2.5);

/// Default barber-pole stripe counts: 8 periods around the ring (one per
/// skylight) and 3 around the tube; the tilt of the stripes on the
/// surface goes as `(minor * twist) / (major * ring)`. Both are live
/// `IntValue` params ([`HollowDonutHandles`]) -- retune the pattern from
/// the params panel while playing. Any integer values close seamlessly
/// around both loops by construction ([`Fold::OrbitBarberPole`]'s doc).
pub const DONUT_RING_COUNT: i32 = 8;
pub const DONUT_TWIST_COUNT: i32 = 3;

/// How many skylights around the ring: 3 plane folds halve the angular
/// domain three times, `2^3 = 8` copies of the fold wedge, and the cutter
/// sits at the wedge's interior mid-angle, whose orbit is the full 8.
/// NOTE this is the *skylight* count, not the stripe-rib count -- ribs
/// live on wedge *edges* and come in 4s (see [`DONUT_STRIPE_COLOR`]'s
/// orbit-size explanation).
pub const DONUT_SYMMETRY: usize = 8;

/// The skylight cutter sphere: centered at the fold wedge's mid-angle
/// (`PI / 8`) at mid-wall height, sized so the marble can actually pass
/// through. **The sizing constraint is volumetric, not visual**: a hole
/// whose *rim* is wider than the marble can still be impassable, because
/// the marble's center must keep `de >= marble_rad` along a continuous
/// path, and near the inner wall face the free space is the *cutter
/// sphere's* shallow cap, not the rim circle. Along the hole axis the
/// clearance bottoms out at the handoff between tube-interior clearance
/// and cutter-void clearance:
///
/// ```text
/// min de = ((minor - thickness) - (HEIGHT - RADIUS)) / 2
///        = (0.85 - 0.4) / 2 = 0.225
/// ```
///
/// comfortably above the 0.15 marble radius (`render.rs`'s HollowDonut
/// `spawn_params`). The first shipped values (radius 0.5, height 1.3 --
/// centered *above* the wall) failed exactly this: rim aperture 0.218
/// looked passable, but the axis clearance bottomed out at 0.05, a
/// too-tight band the marble could never cross (reported live as "it
/// looks like it should fit, but it doesn't"). Guarded by the
/// `skylights_are_marble_passable` test below. Positioned from the stock
/// dimension constants, not the live params -- a params-panel resize moves
/// the wall but not the skylights, which is fine for a tuning tool (at the
/// thickness slider's max the holes go impassable again; dev knob, dev
/// consequences).
pub const DONUT_SKYLIGHT_RADIUS: f32 = 0.6;
pub const DONUT_SKYLIGHT_HEIGHT: f32 = 1.0;

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
///  - **Stripes**: a [`Fold::OrbitBarberPole`] placed *before* the plane
///    folds (it needs the true, unfolded angles) paints red/white helical
///    stripes that close seamlessly around both loops, with live-param
///    ring/twist counts ([`DONUT_RING_COUNT`]/[`DONUT_TWIST_COUNT`]). The
///    stripes recede around the tunnel's curve, which is what actually
///    reads as "inside a donut" instead of "inside a vague pale tube".
pub fn hollow_donut(params: &mut Params) -> (Object, HollowDonutHandles) {
    use std::f32::consts::FRAC_PI_8;

    let major = params.alloc_scalar(DONUT_MAJOR_RADIUS);
    let minor = params.alloc_scalar(DONUT_MINOR_RADIUS);
    let thickness = params.alloc_scalar(DONUT_THICKNESS);
    let ring_count = params.alloc_int(DONUT_RING_COUNT);
    let twist_count = params.alloc_int(DONUT_TWIST_COUNT);
    let handles = HollowDonutHandles { major, minor, thickness, ring_count, twist_count };
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

    // The barber-pole orbit op must run FIRST, on the *unfolded* query
    // point: it measures the true toroidal/poloidal angles, which the
    // kaleidoscope plane folds below (still needed to replicate the
    // skylight cutter) would destroy. Orbit ops never move `p`, but they
    // are emission-order-sensitive relative to folds that do.
    //
    // Three reflections through Y-axis planes then fold the full circle
    // into the wedge `atan2(z, x) in [0, PI/4]`: |x|, then |z| (first
    // quadrant), then reflect across the x = z diagonal (keep x >= z).
    let sqrt_half = std::f32::consts::FRAC_1_SQRT_2;
    let fold = Fold::Series(vec![
        Fold::OrbitBarberPole {
            major: ScalarValue::Param(major),
            ring_count: IntValue::Param(ring_count),
            twist_count: IntValue::Param(twist_count),
            color_a: Vec3Value::Const(DONUT_POLE_RED),
            color_b: Vec3Value::Const(DONUT_POLE_WHITE),
        },
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
    ]);

    let object = Object::Fractal {
        fold,
        base: Box::new(pierced),
    };
    (object, handles)
}

/// Parameter handles for [`cube_sphere_morph`]: the morph parameter and
/// the [`Expr`] that drives it -- register `(t, t_anim.clone())` into the
/// scene's animation table so it's evaluated once per simulated tick,
/// live and through rollback resimulation alike (same pattern as
/// [`MengerOscillatingSphereHandles`]).
#[derive(Clone, Debug)]
pub struct CubeSphereMorphHandles {
    pub t: ScalarParam,
    pub t_anim: Expr,
}

/// Half-extent of the cube and radius of the sphere -- "the same size".
pub const MORPH_HALF_SIZE: f32 = 1.0;

/// Full cycle length in simulated ticks: hold cube 3s, morph 3s, hold
/// sphere 3s, morph back 3s = 12s at the app's fixed 60Hz tick rate.
const MORPH_PERIOD_TICKS: f32 = 12.0 * 60.0;

/// Deep blue (cube) and crimson (sphere), pre-albedo-pipeline values --
/// same channel logic as the donut palettes: what matters is what
/// survives Reinhard compression + material-gamma squaring + ACES, and
/// near-zero green keeps both colors clean under direct sun.
const MORPH_CUBE_COLOR: Vec3 = Vec3::new(0.10, 0.12, 2.2);
const MORPH_SPHERE_COLOR: Vec3 = Vec3::new(1.3, 0.05, 0.06);

/// A blue cube that periodically melts into a red sphere and back
/// ([`Object::Morph`] with an [`Expr`]-driven `t`):
///
/// ```text
/// t(tick) = clamp(0.5 - cos(2*PI*tick / PERIOD) / sqrt(2), 0, 1)
/// ```
///
/// The clamped cosine is the whole hold/ramp schedule in one closed form:
/// with amplitude `1/sqrt(2)`, the expression sits at/below 0 for a
/// quarter period (3s, held as the cube), rises smoothly for a quarter
/// (cosine-eased morph -- gentler than a linear ramp at both ends),
/// saturates at/above 1 for a quarter (held as the sphere), and eases
/// back. `arccos(0.5 / (1/sqrt(2))) = PI/4` is exactly what makes the
/// hold and ramp windows equal. Tick 0 lands mid-cube-hold (`t = 0`,
/// matching the param's initial value, so the first pre-tick frame isn't
/// briefly wrong -- the `menger_oscillating_sphere` convention).
///
/// The colors crossfade in sync with the geometry for free:
/// `Object::Morph`'s color-pass emission blends `orbit` between the two
/// branches by the same clamped `t` it uses for the distance mix, so the
/// cube's blue `OrbitInit` and the sphere's red one meet mid-morph as a
/// purple in-between -- no separate color animation needed.
pub fn cube_sphere_morph(params: &mut Params) -> (Object, CubeSphereMorphHandles) {
    let t = params.alloc_scalar(0.0);
    let omega = std::f32::consts::TAU / MORPH_PERIOD_TICKS;
    let angle = Expr::Mul(Box::new(Expr::Tick), Box::new(Expr::Const(omega)));
    let wave = Expr::Sub(
        Box::new(Expr::Const(0.5)),
        Box::new(Expr::Mul(
            Box::new(Expr::Const(std::f32::consts::FRAC_1_SQRT_2)),
            Box::new(Expr::Cos(Box::new(angle))),
        )),
    );
    let t_anim = Expr::Clamp(
        Box::new(wave),
        Box::new(Expr::Const(0.0)),
        Box::new(Expr::Const(1.0)),
    );
    let handles = CubeSphereMorphHandles { t, t_anim };

    let cube = Object::Fractal {
        fold: Fold::OrbitInit(Vec3Value::Const(MORPH_CUBE_COLOR)),
        base: Box::new(Object::Cuboid {
            half_extent: Vec3Value::Const(Vec3::splat(MORPH_HALF_SIZE)),
        }),
    };
    let sphere = Object::Fractal {
        fold: Fold::OrbitInit(Vec3Value::Const(MORPH_SPHERE_COLOR)),
        base: Box::new(Object::Sphere {
            radius: ScalarValue::Const(MORPH_HALF_SIZE),
        }),
    };
    let object = Object::Morph {
        a: Box::new(cube),
        b: Box::new(sphere),
        t: ScalarValue::Param(t),
    };
    (object, handles)
}

/// Parameter handles for [`gears`]: the two tooth-phase params and the
/// [`Expr`]s that spin them -- register both `(param, anim)` pairs into
/// the scene's animation table (the [`CubeSphereMorphHandles`] pattern).
#[derive(Clone, Debug)]
pub struct GearsHandles {
    /// Tooth phase of the 6 face-axis gears (radians about each gear's
    /// own outward axis).
    pub face_phase: ScalarParam,
    pub face_anim: Expr,
    /// Tooth phase of the 12 edge-direction gears. Spins **opposite** to
    /// `face_phase` -- see [`gears`] for why the sign is load-bearing.
    pub edge_phase: ScalarParam,
    pub edge_anim: Expr,
}

/// [`gears`] geometry constants, straight from the reference shader
/// (iq's "Gears", Shadertoy 3lBSRK) at full extension `tr = 1`. All the
/// gears live on this sphere: shell radius 0.5, wall +-0.03.
const GEARS_SHELL_RADIUS: f32 = 0.5;
const GEARS_SHELL_THICK: f32 = 0.03;
/// Ring (the gear's rim): cylinder radius 0.155, wall +-0.018.
const GEARS_RING_RADIUS: f32 = 0.155;
const GEARS_RING_THICK: f32 = 0.018;
/// Teeth: 12 per gear, each a rounded box at radial center 0.17 --
/// half-extents (tangential, axial, radial) + rounding. The reference
/// sizes teeth in the plane at shell radius (tangential half-width
/// `0.041 * r` with `r ~= 0.5`); ours are the equivalent constants.
pub const GEARS_TOOTH_COUNT: i32 = 12;
const GEARS_TOOTH_RADIAL_CENTER: f32 = 0.17;
const GEARS_TOOTH_HALF: Vec3 = Vec3::new(0.0155, 9.0, 0.037);
const GEARS_TOOTH_ROUND: f32 = 0.005;
/// Face cross (the 4-spoke cap at the gear's pole, y = 0.485): one
/// rounded bar, replicated into a plus-sign by a 4-fold polar modulo.
const GEARS_CROSS_Y: f32 = 0.485;
const GEARS_CROSS_HALF: Vec3 = Vec3::new(0.039, 0.005, 0.167);
const GEARS_CROSS_ROUND: f32 = 0.003;
/// Axle pivot: radius-0.01 cylinder capped by a radius-0.51 sphere, plus
/// the small knob sphere on the axle at y = 0.12.
const GEARS_PIVOT_RADIUS: f32 = 0.01;
const GEARS_PIVOT_CAP: f32 = 0.51;
const GEARS_KNOB_RADIUS: f32 = 0.025;
const GEARS_KNOB_Y: f32 = 0.12;
/// The stationary center sphere the axles radiate from.
const GEARS_CENTER_RADIUS: f32 = 0.12;

/// Rotation rate: the reference spins gears at 2 rad/s; ticks are 60 Hz.
const GEARS_RATE_PER_TICK: f32 = 2.0 / 60.0;
/// Edge gears' tooth offset: half a tooth sector (`TAU/24`), so their
/// teeth interleave with the face gears' at the contact points.
const GEARS_EDGE_OFFSET: f32 = std::f32::consts::TAU / 24.0;

/// Albedo constants (pre-pipeline values -- see [`hollow_donut`]'s color
/// notes): cool steel for the gears, warm orange for the center sphere.
const GEARS_STEEL: Vec3 = Vec3::new(1.4, 1.4, 1.6);
const GEARS_CENTER_COLOR: Vec3 = Vec3::new(2.0, 0.35, 0.05);

/// One gear *pair* around the template's +-Y axis: every part is built in
/// the upper half only and a `Plane{y}` mirror fold supplies the lower
/// gear. The mirror is not just a dedup -- a mirrored copy of a
/// `phase`-rotating pattern **counter-rotates**, which is exactly what two
/// gears sharing an axle through the center sphere must do (the reference
/// shader gets the same effect from `sign(q.y)` inside its rotation).
///
/// Parts (template coordinates, gear axis = +Y):
///  - ring: radius-0.155 cylinder shell, cut to the 0.5-sphere shell
///    (the gear is the small circle where cylinder meets sphere, near the
///    pole at `y ~= 0.475`);
///  - teeth: one rounded box at radial center 0.17, replicated 12-fold by
///    [`Fold::PolarModulo`] (whose `phase` is the live rotation), cut to
///    the same spherical shell;
///  - cross: one rounded bar at the pole cap, replicated 4-fold by a
///    second `PolarModulo` sharing the same phase param so the spokes
///    spin rigidly with the teeth;
///  - pivot: thin axle cylinder capped by a slightly-larger sphere, plus
///    the knob sphere partway down the axle. These don't rotate visually,
///    but they're inside the phase-free part of the tree anyway.
fn gears_pair(align: Vec<Fold>, phase: ScalarParam) -> Object {
    let shell = || Object::Onion {
        base: Box::new(Object::Sphere {
            radius: ScalarValue::Const(GEARS_SHELL_RADIUS),
        }),
        thickness: ScalarValue::Const(GEARS_SHELL_THICK),
    };

    let ring = Object::Intersect(
        Box::new(Object::Onion {
            base: Box::new(Object::Cylinder {
                radius: ScalarValue::Const(GEARS_RING_RADIUS),
            }),
            thickness: ScalarValue::Const(GEARS_RING_THICK),
        }),
        Box::new(shell()),
    );

    let teeth = Object::Intersect(
        Box::new(Object::Fractal {
            fold: Fold::Series(vec![
                Fold::PolarModulo {
                    axis: Axis::Y,
                    count: IntValue::Const(GEARS_TOOTH_COUNT),
                    phase: ScalarValue::Param(phase),
                },
                Fold::ScaleTranslate {
                    scale: ScalarValue::Const(1.0),
                    shift: Vec3Value::Const(Vec3::new(0.0, 0.0, -GEARS_TOOTH_RADIAL_CENTER)),
                },
            ]),
            base: Box::new(Object::Offset {
                base: Box::new(Object::Cuboid {
                    half_extent: Vec3Value::Const(GEARS_TOOTH_HALF),
                }),
                offset: ScalarValue::Const(GEARS_TOOTH_ROUND),
            }),
        }),
        Box::new(shell()),
    );

    let cross = Object::Fractal {
        fold: Fold::Series(vec![
            Fold::PolarModulo {
                axis: Axis::Y,
                count: IntValue::Const(4),
                phase: ScalarValue::Param(phase),
            },
            Fold::ScaleTranslate {
                scale: ScalarValue::Const(1.0),
                shift: Vec3Value::Const(Vec3::new(0.0, -GEARS_CROSS_Y, 0.0)),
            },
        ]),
        base: Box::new(Object::Offset {
            base: Box::new(Object::Cuboid {
                half_extent: Vec3Value::Const(GEARS_CROSS_HALF),
            }),
            offset: ScalarValue::Const(GEARS_CROSS_ROUND),
        }),
    };

    let pivot = Object::Intersect(
        Box::new(Object::Cylinder {
            radius: ScalarValue::Const(GEARS_PIVOT_RADIUS),
        }),
        Box::new(Object::Sphere {
            radius: ScalarValue::Const(GEARS_PIVOT_CAP),
        }),
    );

    let knob = Object::Fractal {
        fold: Fold::ScaleTranslate {
            scale: ScalarValue::Const(1.0),
            shift: Vec3Value::Const(Vec3::new(0.0, -GEARS_KNOB_Y, 0.0)),
        },
        base: Box::new(Object::Sphere {
            radius: ScalarValue::Const(GEARS_KNOB_RADIUS),
        }),
    };

    let union = Object::Union(
        Box::new(ring),
        Box::new(Object::Union(
            Box::new(teeth),
            Box::new(Object::Union(
                Box::new(cross),
                Box::new(Object::Union(Box::new(pivot), Box::new(knob))),
            )),
        )),
    );

    let mut folds = vec![Fold::OrbitInit(Vec3Value::Const(GEARS_STEEL))];
    folds.extend(align);
    folds.push(Fold::Plane {
        normal: Vec3Value::Const(Vec3::Y),
        offset: ScalarValue::Const(0.0),
    });

    Object::Fractal {
        fold: Fold::Series(folds),
        base: Box::new(union),
    }
}

/// The 9 gear-pair alignments: constant `Rotate` folds mapping each
/// pair's two world axis directions onto the template's +-Y. Verified
/// numerically by `gears_pair_axes_map_to_template_y` below. Which pole
/// lands on +Y vs -Y is irrelevant: the pair is mirror-symmetric, and a
/// rotation about the *outward* axis is itself mirror-invariant, so both
/// choices render identically.
fn gears_alignments() -> Vec<(Vec<Fold>, [Vec3; 2])> {
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};
    let rot = |axis: Axis, angle: f32| Fold::Rotate {
        axis,
        mat: Mat2Value::Const(rotation_mat2(angle)),
    };
    let s = std::f32::consts::FRAC_1_SQRT_2;
    vec![
        // Face pairs: +-X, +-Y, +-Z.
        (vec![rot(Axis::Z, -FRAC_PI_2)], [Vec3::X, -Vec3::X]),
        (vec![], [Vec3::Y, -Vec3::Y]),
        (vec![rot(Axis::X, FRAC_PI_2)], [Vec3::Z, -Vec3::Z]),
        // Edge pairs: the 6 antipodal pairs of cube-edge directions.
        (
            vec![rot(Axis::Z, -FRAC_PI_4)],
            [Vec3::new(s, s, 0.0), Vec3::new(-s, -s, 0.0)],
        ),
        (
            vec![rot(Axis::Z, -3.0 * FRAC_PI_4)],
            [Vec3::new(s, -s, 0.0), Vec3::new(-s, s, 0.0)],
        ),
        (
            vec![rot(Axis::X, FRAC_PI_4)],
            [Vec3::new(0.0, s, s), Vec3::new(0.0, -s, -s)],
        ),
        (
            vec![rot(Axis::X, -FRAC_PI_4)],
            [Vec3::new(0.0, s, -s), Vec3::new(0.0, -s, s)],
        ),
        (
            vec![rot(Axis::Y, -FRAC_PI_4), rot(Axis::Z, -FRAC_PI_2)],
            [Vec3::new(s, 0.0, s), Vec3::new(-s, 0.0, -s)],
        ),
        (
            vec![rot(Axis::Y, FRAC_PI_4), rot(Axis::Z, -FRAC_PI_2)],
            [Vec3::new(s, 0.0, -s), Vec3::new(-s, 0.0, s)],
        ),
    ]
}

/// iq's "Gears" (Shadertoy 3lBSRK) in its fully-extended, constantly
/// rotating state: 18 interlocking gears on the surface of a sphere --
/// one at each face direction and each edge direction of a cube -- their
/// axles meeting a stationary sphere at the center. The reference's
/// extend/contract cycle is deliberately not reproduced; this is the
/// `tr = 1` steady state only.
///
/// Structure: 9 [`gears_pair`] subtrees (each covering two antipodal
/// gears via its internal mirror), explicitly unioned. The reference
/// evaluates only 4 gear calls using octant folds; those folds are
/// mirrors, and a mirrored gear *counter-rotates*, which its shader
/// counteracts with `sign()` tricks inside the rotation -- our fold
/// algebra has no per-branch sign, so the explicit union (cost comparable
/// to the classic fractal's 16 folded iterations) buys exact CPU physics
/// and normals through the standard fold-history machinery instead.
///
/// **Meshing** is a real constraint, not set dressing (the tooth circles
/// of adjacent gears genuinely interpenetrate -- angular radius ~24 deg
/// each, axes 45 deg apart):
///  - The contact graph is bipartite: edge gears touch only face gears
///    (face-face are 90 deg apart, edge-edge 60 deg, both out of reach), so a
///    consistent two-coloring of spin directions exists.
///  - Velocity matching at each contact point forces the two classes to
///    spin at **equal and opposite** rates about their own outward axes;
///    hence `edge_anim = -face rate` (the reference hides this sign in
///    the handedness of its swizzled frames).
///  - Tooth alignment: with these alignment rotations every contact
///    direction sits at azimuth `0 mod TAU/12` in *both* gears' local
///    frames, so face gears (phase 0) present a tooth exactly where edge
///    gears (phase `TAU/24`) present a gap. Matched velocities keep it
///    that way for all time.
pub fn gears(params: &mut Params) -> (Object, GearsHandles) {
    let face_phase = params.alloc_scalar(0.0);
    let edge_phase = params.alloc_scalar(GEARS_EDGE_OFFSET);
    let face_anim = Expr::Mul(
        Box::new(Expr::Tick),
        Box::new(Expr::Const(GEARS_RATE_PER_TICK)),
    );
    let edge_anim = Expr::Add(
        Box::new(Expr::Mul(
            Box::new(Expr::Tick),
            Box::new(Expr::Const(-GEARS_RATE_PER_TICK)),
        )),
        Box::new(Expr::Const(GEARS_EDGE_OFFSET)),
    );
    let handles = GearsHandles {
        face_phase,
        face_anim,
        edge_phase,
        edge_anim,
    };

    let center = Object::Fractal {
        fold: Fold::OrbitInit(Vec3Value::Const(GEARS_CENTER_COLOR)),
        base: Box::new(Object::Sphere {
            radius: ScalarValue::Const(GEARS_CENTER_RADIUS),
        }),
    };

    let mut object = center;
    for (i, (align, _axes)) in gears_alignments().into_iter().enumerate() {
        let phase = if i < 3 { face_phase } else { edge_phase };
        object = Object::Union(Box::new(object), Box::new(gears_pair(align, phase)));
    }
    (object, handles)
}

/// The embedded Stanford-bunny asset (`csg/assets/bunny.mesh`): the
/// **full-resolution** zipper reconstruction (34,834 vertices / 69,664
/// triangles, ~1.2 MB), made watertight offline by fan-filling its five
/// base boundary loops and dropping unreferenced vertices -- after which
/// it verifies as a genuine genus-0 closed manifold (Euler characteristic
/// 2, every edge shared by exactly two consistently-wound faces).
/// Accuracy-first by explicit request: an earlier attempt shipped the
/// res4 decimation repaired via winding-number surface nets, and its
/// ears -- thinner than any affordable extraction cell -- came out
/// visibly mutilated from every angle. The full scan's visible surface
/// is untouched here (only the underside holes gain fill triangles);
/// shrinking the asset again (decimation that respects thin features,
/// or compression) is a later optimization. Normalized to height 1.0
/// standing on `y = 0`. The byte layout is exactly
/// [`crate::trimesh::TriMeshData`]'s serialization, so the asset loader
/// *is* the decoder.
const BUNNY_MESH_BYTES: &[u8] = include_bytes!("../assets/bunny.mesh");

/// Pre-albedo-pipeline colors (the [`hollow_donut`] lessons): warm cream
/// for the bunny, cool slate for the floor it sits on.
const BUNNY_COLOR: Vec3 = Vec3::new(2.2, 1.6, 1.1);
const BUNNY_FLOOR_COLOR: Vec3 = Vec3::new(0.25, 0.35, 1.0);
const BUNNY_FLOOR_HALF: Vec3 = Vec3::new(6.0, 0.15, 6.0);

/// The Stanford bunny standing on a floor slab -- the first
/// [`Object::TriMesh`] scene (`rust/MESH_SDF.md` made real): CPU physics
/// collides against the *exact* mesh field (BVH + pseudonormal sign), the
/// GPU marches the baked grid the app uploads as a `texture_3d`.
pub fn bunny(_params: &mut Params) -> Object {
    let (mesh, _len) = crate::trimesh::TriMeshData::decode_at(BUNNY_MESH_BYTES, 0)
        .expect("embedded bunny asset must decode as a closed manifold");
    let bunny = Object::Fractal {
        fold: Fold::OrbitInit(Vec3Value::Const(BUNNY_COLOR)),
        base: Box::new(Object::TriMesh {
            mesh: std::sync::Arc::new(mesh),
        }),
    };
    // Floor slab with its top face at y = 0, where the bunny's feet are.
    let floor = Object::Fractal {
        fold: Fold::Series(vec![
            Fold::OrbitInit(Vec3Value::Const(BUNNY_FLOOR_COLOR)),
            Fold::ScaleTranslate {
                scale: ScalarValue::Const(1.0),
                shift: Vec3Value::Const(Vec3::new(0.0, BUNNY_FLOOR_HALF.y, 0.0)),
            },
        ]),
        base: Box::new(Object::Cuboid {
            half_extent: Vec3Value::Const(BUNNY_FLOOR_HALF),
        }),
    };
    Object::Union(Box::new(floor), Box::new(bunny))
}

/// [`noise_caverns`]' generation constants. Seed from the reference
/// implementation; sparsity per the explicit request: **70% of space
/// open**, so the solid fraction handed to
/// [`crate::noise3::iso_for_solid_fraction`] is 0.3.
pub const CAVERNS_SEED: u32 = 11;
pub const CAVERNS_SOLID_FRACTION: f32 = 0.3;
/// World scale: the unit noise torus maps to a `6 x 6` arena footprint
/// (fold scale `1/6`), clipped to `y in [0, 2]` -- flat-topped rock
/// formations the marble weaves between.
pub const CAVERNS_SCALE: f32 = 6.0;
pub const CAVERNS_HEIGHT: f32 = 2.0;
const CAVERNS_ROCK_COLOR: Vec3 = Vec3::new(1.7, 0.75, 0.35);
const CAVERNS_FLOOR_COLOR: Vec3 = Vec3::new(0.15, 0.2, 0.8);
const CAVERNS_FLOOR_HALF: Vec3 = Vec3::new(3.4, 0.15, 3.4);

/// Marble spawn (render.rs `spawn_params`): the widest pocket found by a
/// deterministic scan over the arena floor at this seed/iso -- clearance
/// ~0.98 world units, asserted with margin by the scene test below.
pub const CAVERNS_SPAWN: Vec3 = Vec3::new(0.0, 0.35, -0.45);

/// Noise caverns ([`Object::NoiseSolid`], `noise3.rs`): the exact 3-D
/// procedural noise SDF at 70% sparsity, scaled to a 6-unit arena and
/// clipped to a flat-topped box over a floor slab. The rock the marble
/// touches is the *exact* piecewise-linear noise isosurface -- physics
/// collides against |grad d| = 1 geometry with true closest points, and
/// the whole rock formation serializes as 8 bytes (seed + iso).
pub fn noise_caverns(_params: &mut Params) -> Object {
    let iso = crate::noise3::iso_for_solid_fraction(
        CAVERNS_SEED,
        CAVERNS_SOLID_FRACTION,
        0.0,
        CAVERNS_HEIGHT / CAVERNS_SCALE,
    );
    let noise = crate::noise3::NoiseSolidData::new(CAVERNS_SEED, iso)
        .expect("caverns noise field must build");

    // Unit torus -> world: p' = p/6 + (0.5, 0, 0.5) maps world
    // [-3,3] x [0,6] x [-3,3] onto the certified cube (and beyond that
    // the field tiles -- `noise3.rs`'s wrapped queries).
    let scaled = Object::Fractal {
        fold: Fold::ScaleTranslate {
            scale: ScalarValue::Const(1.0 / CAVERNS_SCALE),
            shift: Vec3Value::Const(Vec3::new(0.5, 0.0, 0.5)),
        },
        base: Box::new(Object::NoiseSolid {
            noise: std::sync::Arc::new(noise),
        }),
    };
    // Clip to the arena box: y in [0, CAVERNS_HEIGHT], footprint 6 x 6.
    let arena = Object::Fractal {
        fold: Fold::ScaleTranslate {
            scale: ScalarValue::Const(1.0),
            shift: Vec3Value::Const(Vec3::new(0.0, -CAVERNS_HEIGHT / 2.0, 0.0)),
        },
        base: Box::new(Object::Cuboid {
            half_extent: Vec3Value::Const(Vec3::new(
                CAVERNS_SCALE / 2.0,
                CAVERNS_HEIGHT / 2.0,
                CAVERNS_SCALE / 2.0,
            )),
        }),
    };
    let rock = Object::Fractal {
        fold: Fold::OrbitInit(Vec3Value::Const(CAVERNS_ROCK_COLOR)),
        base: Box::new(Object::Intersect(Box::new(scaled), Box::new(arena))),
    };
    let floor = Object::Fractal {
        fold: Fold::Series(vec![
            Fold::OrbitInit(Vec3Value::Const(CAVERNS_FLOOR_COLOR)),
            Fold::ScaleTranslate {
                scale: ScalarValue::Const(1.0),
                shift: Vec3Value::Const(Vec3::new(0.0, CAVERNS_FLOOR_HALF.y, 0.0)),
            },
        ]),
        base: Box::new(Object::Cuboid {
            half_extent: Vec3Value::Const(CAVERNS_FLOOR_HALF),
        }),
    };
    Object::Union(Box::new(floor), Box::new(rock))
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

    /// The marble (radius 0.15, `render.rs`'s HollowDonut `spawn_params`)
    /// must be able to fly out through a skylight: `physics::collide`
    /// resolves a contact whenever `de < rad`, so passability means a
    /// continuous center path from inside the tube to open air with
    /// `de >= rad` (plus margin) everywhere along it. This is a *stronger*
    /// condition than the hole's rim being wider than the marble -- the
    /// first shipped skylight had a passable-looking 0.218 rim aperture but
    /// a 0.05 clearance pinch along the axis (see
    /// [`DONUT_SKYLIGHT_RADIUS`]'s doc), which this test would have caught.
    #[test]
    fn hollow_donut_skylights_are_marble_passable() {
        use std::f32::consts::FRAC_PI_8;
        const MARBLE_RAD: f32 = 0.15;
        const MARGIN: f32 = 0.05;

        let mut params = Params::new();
        let (object, _handles) = hollow_donut(&mut params);
        let (sin, cos) = FRAC_PI_8.sin_cos();
        // The hole's axis: vertically through the cutter center, from the
        // tube's midline out to clearly-open air above the donut.
        let mut min_de = f32::MAX;
        let mut min_y = 0.0;
        let mut y = 0.0;
        while y <= 2.0 {
            let p = Vec4::new(DONUT_MAJOR_RADIUS * cos, y, DONUT_MAJOR_RADIUS * sin, 1.0);
            let d = object.de(p, &params);
            if d < min_de {
                min_de = d;
                min_y = y;
            }
            y += 0.005;
        }
        assert!(
            min_de >= MARBLE_RAD + MARGIN,
            "skylight pinches to de={min_de} at y={min_y}: the marble (rad {MARBLE_RAD}) \
             cannot pass -- see DONUT_SKYLIGHT_RADIUS's doc for the clearance formula"
        );
    }

    #[test]
    fn cube_sphere_morph_schedule_holds_and_ramps_on_the_12s_cycle() {
        let mut params = Params::new();
        let (object, handles) = cube_sphere_morph(&mut params);
        let anim = &handles.t_anim;

        // Quarter-period landmarks at 60Hz: hold-cube is ticks (-90, 90)
        // around 0, ramp up (90, 270), hold-sphere (270, 450), ramp down
        // (450, 630).
        assert!(anim.eval(0).abs() < 1e-6, "tick 0 must be fully cube");
        assert!(anim.eval(80).abs() < 1e-6, "still held as cube near the hold's edge");
        assert!((anim.eval(180) - 0.5).abs() < 1e-3, "mid-ramp at the eighth-period point");
        assert!((anim.eval(360) - 1.0).abs() < 1e-6, "fully sphere at the half period");
        assert!((anim.eval(430) - 1.0).abs() < 1e-6, "still held as sphere");
        assert!((anim.eval(540) - 0.5).abs() < 1e-3, "mid-ramp back");
        assert!(anim.eval(700).abs() < 1e-6, "back to cube before the period ends");
        // Periodicity (within f32 slack of the big-tick trig argument).
        assert!((anim.eval(123) - anim.eval(123 + 720)).abs() < 1e-3);

        // Ramps must be monotone -- the clamp must never let the cosine
        // wave wiggle t backwards mid-transition.
        let mut prev = anim.eval(90);
        for tick in 91..270 {
            let v = anim.eval(tick);
            assert!(v >= prev - 1e-6, "ramp not monotone at tick {tick}: {v} < {prev}");
            prev = v;
        }

        // Endpoint geometry: with t forced to the extremes, the morph is
        // exactly the cube / exactly the sphere.
        let cube = Object::Cuboid { half_extent: Vec3Value::Const(Vec3::splat(MORPH_HALF_SIZE)) };
        let sphere = Object::Sphere { radius: ScalarValue::Const(MORPH_HALF_SIZE) };
        let probes = [
            Vec4::new(2.0, 0.4, -0.3, 1.0),
            Vec4::new(0.9, 0.9, 0.9, 1.0),
            Vec4::new(0.2, 0.1, 0.0, 1.0),
        ];
        params.set_scalar(handles.t, 0.0);
        for p in probes {
            assert!((object.de(p, &params) - cube.de(p, &params)).abs() < 1e-6);
        }
        params.set_scalar(handles.t, 1.0);
        for p in probes {
            assert!((object.de(p, &params) - sphere.de(p, &params)).abs() < 1e-6);
        }
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

    /// Every alignment in [`gears_alignments`] must send both of its world
    /// gear directions onto the template's +-Y axis -- the whole meshing
    /// argument in [`gears`]'s doc is anchored to these being exact.
    #[test]
    fn gears_pair_axes_map_to_template_y() {
        let params = Params::new();
        for (i, (align, axes)) in gears_alignments().into_iter().enumerate() {
            let fold = Fold::Series(align);
            for axis in axes {
                let mut p = Vec4::new(axis.x, axis.y, axis.z, 1.0);
                fold.fold(&mut p, &params);
                assert!(
                    p.x.abs() < 1e-6 && p.z.abs() < 1e-6 && (p.y.abs() - 1.0).abs() < 1e-6,
                    "pair {i}: axis {axis:?} folded to {p:?}, expected +-Y"
                );
            }
        }
    }

    /// The assembled scene has a pivot axle inside the shell along all 18
    /// gear directions and free space just past the shell -- a coarse
    /// whole-assembly check that catches any alignment fold pointing a
    /// pair the wrong way.
    #[test]
    fn gears_scene_has_all_18_axles_where_expected() {
        let mut params = Params::new();
        let (object, _handles) = gears(&mut params);

        // Stationary center sphere.
        let d = object.de(Vec4::new(0.0, 0.0, 0.0, 1.0), &params);
        assert!((d - (-GEARS_CENTER_RADIUS)).abs() < 1e-5, "center de={d}");

        for (_align, axes) in gears_alignments() {
            for axis in axes {
                // Just under the pivot cap sphere (0.51): inside the axle.
                let p = axis * 0.505;
                let d = object.de(Vec4::new(p.x, p.y, p.z, 1.0), &params);
                assert!(d < 0.0, "no axle at 0.505 * {axis:?}: de={d}");
                // Just past everything (shell ends at 0.53, cap at 0.51).
                let p = axis * 0.56;
                let d = object.de(Vec4::new(p.x, p.y, p.z, 1.0), &params);
                assert!(d > 0.0, "solid past the shell at 0.56 * {axis:?}: de={d}");
            }
        }

        // Finite bound covering the shell.
        let (_c, r) = object.bounding_sphere(&params).unwrap();
        assert!(r.is_finite() && r >= 0.53 && r < 1.5, "bound r={r}");
    }

    /// Phase animations: equal and opposite rates (the meshing condition),
    /// and each param's initial value equals its anim at tick 0 so the
    /// first pre-tick frame isn't briefly wrong.
    #[test]
    fn gears_phase_anims_counter_rotate_from_consistent_starts() {
        let mut params = Params::new();
        let (_object, handles) = gears(&mut params);
        assert!((handles.face_anim.eval(0) - params.scalar(handles.face_phase)).abs() < 1e-6);
        assert!((handles.edge_anim.eval(0) - params.scalar(handles.edge_phase)).abs() < 1e-6);
        let face_rate = handles.face_anim.eval(60) - handles.face_anim.eval(0);
        let edge_rate = handles.edge_anim.eval(60) - handles.edge_anim.eval(0);
        assert!((face_rate - 2.0).abs() < 1e-4, "face rate {face_rate} rad/s != 2");
        assert!((edge_rate + 2.0).abs() < 1e-4, "edge rate {edge_rate} rad/s != -2");
    }

    /// The load-bearing meshing test: march the face gear at +Y and the
    /// edge gear at (0, s, s) through a full tooth period of their
    /// (counter-rotating) phase schedule, densely sampling the contact
    /// lens where their tooth circles interpenetrate, and assert the two
    /// solids never overlap. This catches every failure mode at once: a
    /// wrong alignment azimuth, a wrong edge offset, or a wrong rotation
    /// *sign* (same-sign gears grind within half a sector).
    #[test]
    fn gears_adjacent_gears_never_interpenetrate_through_a_tooth_period() {
        use std::f32::consts::TAU;
        let mut params = Params::new();
        let face_phase = params.alloc_scalar(0.0);
        let edge_phase = params.alloc_scalar(GEARS_EDGE_OFFSET);
        let aligns = gears_alignments();
        assert_eq!(aligns[1].1[0], Vec3::Y);
        let edge_axis = aligns[5].1[0];
        assert!(edge_axis.x.abs() < 1e-6 && edge_axis.y > 0.0 && edge_axis.z > 0.0);
        let face = gears_pair(aligns[1].0.clone(), face_phase);
        let edge = gears_pair(aligns[5].0.clone(), edge_phase);

        // Contact lens around the pitch point between the two axes.
        let m = (Vec3::Y + edge_axis).normalize();
        let u = m.cross(Vec3::X).normalize();
        let v = m.cross(u);
        let center = m * GEARS_SHELL_RADIUS;

        // Tolerance: the *reference geometry itself* lets tooth tips graze
        // the neighbor's solid rim ring by up to ~0.008 (tip reach 0.212+
        // from its own axis vs. the rim band starting 45 deg away -- both
        // DEs capped by the 0.03 shell term), so a shallow overlap is
        // faithful, not a bug. What the phase schedule must prevent is
        // tooth-vs-tooth grinding, which shows up ~0.02 deep (tangential
        // and radial tooth margins both exceed 0.02 at the pitch point);
        // -0.012 sits between the two regimes.
        const OVERLAP: f32 = -0.012;
        let sector = TAU / GEARS_TOOTH_COUNT as f32;
        for step in 0..12 {
            let ang = step as f32 * sector / 12.0;
            params.set_scalar(face_phase, ang);
            params.set_scalar(edge_phase, -ang + GEARS_EDGE_OFFSET);
            let mut probed_inside_face = false;
            let n = 10;
            for i in -n..=n {
                for j in -n..=n {
                    for k in -n..=n {
                        let p = center
                            + u * (i as f32 / n as f32 * 0.06)
                            + v * (j as f32 / n as f32 * 0.06)
                            + m * (k as f32 / n as f32 * 0.06);
                        let p4 = Vec4::new(p.x, p.y, p.z, 1.0);
                        let df = face.de(p4, &params);
                        let de = edge.de(p4, &params);
                        probed_inside_face |= df < OVERLAP;
                        assert!(
                            !(df < OVERLAP && de < OVERLAP),
                            "gears interpenetrate at {p:?} (phase {ang}): \
                             face de={df}, edge de={de}"
                        );
                    }
                }
            }
            // Guard the test itself: the lens must actually contain face-gear
            // material at every phase, or the assert above is vacuous.
            assert!(probed_inside_face, "contact lens missed the face gear at phase {ang}");
        }
    }

    /// At t = 0 the design puts a face-gear *tooth* exactly at the contact
    /// direction and an edge-gear *gap* there (the azimuth-mod-30-degrees
    /// argument in [`gears`]'s doc); spot-check both signs at the pitch
    /// point.
    #[test]
    fn gears_tooth_meets_gap_at_time_zero() {
        let mut params = Params::new();
        let face_phase = params.alloc_scalar(0.0);
        let edge_phase = params.alloc_scalar(GEARS_EDGE_OFFSET);
        let aligns = gears_alignments();
        let face = gears_pair(aligns[1].0.clone(), face_phase);
        let edge = gears_pair(aligns[5].0.clone(), edge_phase);

        let m = (Vec3::Y + aligns[5].1[0]).normalize();
        let p = m * GEARS_SHELL_RADIUS;
        let p4 = Vec4::new(p.x, p.y, p.z, 1.0);
        let df = face.de(p4, &params);
        let de = edge.de(p4, &params);
        assert!(df < -0.005, "expected a face tooth at the pitch point, de={df}");
        assert!(de > 0.005, "expected an edge gap at the pitch point, de={de}");

        // And with the edge gear's built-in offset removed, its tooth lands
        // there instead -- the offset is what interleaves them.
        params.set_scalar(edge_phase, 0.0);
        let de = edge.de(p4, &params);
        assert!(de < -0.005, "offset-less edge gear should present a tooth, de={de}");
    }

    /// The embedded bunny asset must decode -- which *is* the watertight /
    /// consistent-orientation check, since `TriMeshData::new` rejects
    /// anything else (the offline surface-nets repair pipeline's whole
    /// point).
    #[test]
    fn bunny_asset_is_a_closed_manifold() {
        let (mesh, len) = crate::trimesh::TriMeshData::decode_at(BUNNY_MESH_BYTES, 0)
            .expect("bunny.mesh must decode");
        assert_eq!(len, BUNNY_MESH_BYTES.len(), "trailing bytes in the asset");
        assert!(mesh.tri_count() > 500, "suspiciously small bunny");
        // Normalized on export: height 1.0 standing on y = 0.
        let (c, r) = mesh.bounding_sphere();
        assert!(r > 0.4 && r < 1.0, "bunny bound r={r}");
        assert!((c.y - 0.5).abs() < 0.1, "bunny should stand on y=0, center {c:?}");
    }

    #[test]
    fn bunny_scene_fields_read_correctly() {
        let mut params = Params::new();
        let object = bunny(&mut params);

        // Inside the bunny's body.
        let d = object.de(Vec4::new(0.0, 0.45, 0.0, 1.0), &params);
        assert!(d < -0.02, "body should be solid, de={d}");
        // Well above it: free air.
        let d = object.de(Vec4::new(0.0, 1.6, 0.0, 1.0), &params);
        assert!(d > 0.3, "air above de={d}");
        // On the floor top far from the bunny: de is the height above it.
        let d = object.de(Vec4::new(3.0, 0.5, 3.0, 1.0), &params);
        assert!((d - 0.5).abs() < 1e-5, "floor-height de={d}");
        // Inside the floor slab.
        let d = object.de(Vec4::new(0.0, -0.2, 3.0, 1.0), &params);
        assert!(d < -0.05, "floor solid de={d}");
        // Marble spawn (render.rs: (1.1, 0.3, 0.0), rad 0.1) is free with
        // margin.
        let d = object.de(Vec4::new(1.1, 0.3, 0.0, 1.0), &params);
        assert!(d > 0.15, "spawn clearance de={d}");

        // Mesh-branch nearest point is *exact*: for a probe whose nearest
        // scene surface is the bunny, |p - np| == |de| to float precision
        // (the BVH query answers both -- stronger than the generic 2x
        // consistency the approximate nodes get).
        let p = Vec4::new(0.0, 1.2, 0.0, 1.0);
        let d = object.de(p, &params);
        let np = object.nearest_point(p, &params);
        assert!(
            ((p.truncate() - np).length() - d.abs()).abs() < 1e-5,
            "mesh nearest point not exact: |p-np|={} de={d}",
            (p.truncate() - np).length()
        );

        // The tree carries exactly one mesh, and the whole scene stays
        // bounded (floor slab dominates).
        assert!(object.find_trimesh().is_some());
        let (_c, r) = object.bounding_sphere(&params).unwrap();
        assert!(r.is_finite() && r < 15.0);
    }

    #[test]
    fn noise_caverns_is_playable_and_70_percent_sparse() {
        let mut params = Params::new();
        let object = noise_caverns(&mut params);

        // The chosen spawn is genuinely open (marble rad 0.12 + margin).
        let d = object.de(
            Vec4::new(CAVERNS_SPAWN.x, CAVERNS_SPAWN.y, CAVERNS_SPAWN.z, 1.0),
            &params,
        );
        assert!(d > 0.3, "spawn clearance de={d}");

        // Above the arena: open air. Inside the floor: solid.
        assert!(object.de(Vec4::new(0.0, 3.0, 0.0, 1.0), &params) > 0.5);
        assert!(object.de(Vec4::new(0.0, -0.2, 3.0, 1.0), &params) < -0.05);

        // Sparsity: sample the arena volume; the open fraction must be
        // near the requested 70% (the rock is the noise solid clipped to
        // the box, so measure inside the box only).
        let mut open = 0;
        let mut total = 0;
        for i in 0..12 {
            for j in 0..6 {
                for k in 0..12 {
                    let p = Vec4::new(
                        -2.75 + 0.5 * i as f32,
                        0.17 + 0.3 * j as f32,
                        -2.75 + 0.5 * k as f32,
                        1.0,
                    );
                    total += 1;
                    if object.de(p, &params) > 0.0 {
                        open += 1;
                    }
                }
            }
        }
        let frac = open as f32 / total as f32;
        assert!(
            (frac - 0.7).abs() < 0.08,
            "open fraction {frac} != requested 0.70"
        );

        // Nearest-point consistency through the scaled fold (exact for
        // rock-nearest probes below the cap).
        let probe = Vec3::new(CAVERNS_SPAWN.x, CAVERNS_SPAWN.y, CAVERNS_SPAWN.z);
        let p4 = Vec4::new(probe.x, probe.y, probe.z, 1.0);
        let d = object.de(p4, &params);
        let np = object.nearest_point(p4, &params);
        let actual = (probe - np).length();
        assert!(
            actual <= 1.5 * d.abs().max(1e-4) && d.abs() <= 1.5 * actual.max(1e-4),
            "|p-np|={actual} vs de={d}"
        );

        // 8-byte noise payload: the whole scene stays tiny on the wire.
        assert!(object.to_bytes().len() < 200, "encoded scene unexpectedly large");
    }

    /// CPU physics path: nearest-point queries on the gears scene must be
    /// finite and consistent with the DE (the marble collides with the
    /// spinning teeth through the same fold-history machinery as every
    /// other scene).
    #[test]
    fn gears_nearest_point_is_surface_consistent() {
        let mut params = Params::new();
        let (object, _handles) = gears(&mut params);
        let probes = [
            Vec3::new(0.0, 0.6, 0.02),
            Vec3::new(0.19, 0.46, 0.0),
            Vec3::new(0.3, 0.3, 0.3),
            Vec3::new(0.0, 0.2, 0.0),
            Vec3::new(1.0, 0.5, -0.4),
        ];
        for probe in probes {
            let p = Vec4::new(probe.x, probe.y, probe.z, 1.0);
            let d = object.de(p, &params).abs();
            let np = object.nearest_point(p, &params);
            assert!(np.is_finite(), "non-finite nearest_point for probe {probe:?}");
            let actual = (probe - np).length();
            assert!(
                actual <= 2.0 * d.max(1e-4) && d <= 2.0 * actual.max(1e-4),
                "probe {probe:?}: |p-np|={actual} vs de={d}"
            );
        }
    }
}

