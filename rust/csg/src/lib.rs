//! CSG fractal framework, ported from `src/fractals/` (C++).
//!
//! A scene is a tree of `Object`s and `Fold`s. The same tree drives:
//!  - CPU evaluation: `Object::de` (distance estimate) and `Object::nearest_point`
//!    for collision physics,
//!  - GPU code generation: `codegen::generate_shader` emits WGSL with the tree
//!    inlined into `de_scene`/`col_scene`.
//!
//! Parameters are the port of the C++ `GLSLVariable<T>` hierarchy. Instead of
//! named GL uniforms (wgpu has no set-uniform-by-name), every mutable parameter
//! is a slot in a single `array<vec4<f32>>` storage buffer ([`Params`]).
//! `*Value::Const` bakes a literal into the generated WGSL exactly like
//! `GLSLConstant`; `*Value::Param` references a slot exactly like `GLSLUniform`.
//!
//! Module layout (see rust/MILESTONES.md for build-out order):
//!  - `fold`     (M2): the `Fold` enum — space folds + orbit trap ops
//!  - `object`   (M2): the `Object` enum — primitives, CSG combiners, `Fractal`
//!  - `scenes`   (M2): prebuilt scenes (classic Marble Marcher fractal, etc.)
//!  - `codegen`  (M3): WGSL generation from an `Object` tree
//!  - `physics`  (M5): marble/collider simulation against an `Object`
//!  - `rollback` (multiplayer milestone 1): input buffering + snapshot/
//!    rewind/resimulate rollback netcode around `physics::step_marbles`

pub mod codegen;
pub mod fold;
pub mod object;
pub mod physics;
pub mod rollback;
pub mod scenes;

pub use fold::Fold;
pub use object::Object;

use glam::{Mat2, Vec2, Vec3, Vec4};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    pub fn index(self) -> usize {
        match self {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        }
    }
}

/// Typed handles into the shared parameter slot table.
#[derive(Clone, Copy, Debug)]
pub struct ScalarParam(u16);
#[derive(Clone, Copy, Debug)]
pub struct Vec3Param(u16);
#[derive(Clone, Copy, Debug)]
pub struct Mat2Param(u16);
#[derive(Clone, Copy, Debug)]
pub struct IntParam(u16);

/// The parameter slot table: one `Vec4` per parameter, uploaded verbatim as a
/// storage buffer. Ints are stored as floats (converted back in WGSL); a
/// `Mat2` packs column-major into one `Vec4`.
#[derive(Clone, Debug, Default)]
pub struct Params {
    slots: Vec<Vec4>,
}

impl Params {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn slots(&self) -> &[Vec4] {
        &self.slots
    }

    fn alloc(&mut self, v: Vec4) -> u16 {
        let idx = u16::try_from(self.slots.len()).expect("too many params");
        self.slots.push(v);
        idx
    }

    pub fn alloc_scalar(&mut self, v: f32) -> ScalarParam {
        ScalarParam(self.alloc(Vec4::new(v, 0.0, 0.0, 0.0)))
    }

    pub fn alloc_vec3(&mut self, v: Vec3) -> Vec3Param {
        Vec3Param(self.alloc(v.extend(0.0)))
    }

    pub fn alloc_mat2(&mut self, m: Mat2) -> Mat2Param {
        Mat2Param(self.alloc(pack_mat2(m)))
    }

    pub fn alloc_int(&mut self, v: i32) -> IntParam {
        IntParam(self.alloc(Vec4::new(v as f32, 0.0, 0.0, 0.0)))
    }

    pub fn set_scalar(&mut self, h: ScalarParam, v: f32) {
        self.slots[h.0 as usize].x = v;
    }

    pub fn set_vec3(&mut self, h: Vec3Param, v: Vec3) {
        self.slots[h.0 as usize] = v.extend(0.0);
    }

    pub fn set_mat2(&mut self, h: Mat2Param, m: Mat2) {
        self.slots[h.0 as usize] = pack_mat2(m);
    }

    pub fn set_int(&mut self, h: IntParam, v: i32) {
        self.slots[h.0 as usize].x = v as f32;
    }

    pub fn scalar(&self, h: ScalarParam) -> f32 {
        self.slots[h.0 as usize].x
    }

    pub fn vec3(&self, h: Vec3Param) -> Vec3 {
        self.slots[h.0 as usize].truncate()
    }

    pub fn mat2(&self, h: Mat2Param) -> Mat2 {
        unpack_mat2(self.slots[h.0 as usize])
    }

    pub fn int(&self, h: IntParam) -> i32 {
        self.slots[h.0 as usize].x as i32
    }
}

fn pack_mat2(m: Mat2) -> Vec4 {
    // Column-major, matching WGSL's mat2x2<f32>(c0x, c0y, c1x, c1y).
    Vec4::new(m.x_axis.x, m.x_axis.y, m.y_axis.x, m.y_axis.y)
}

fn unpack_mat2(v: Vec4) -> Mat2 {
    Mat2::from_cols(Vec2::new(v.x, v.y), Vec2::new(v.z, v.w))
}

/// WGSL literal for an f32 (always contains a decimal point or exponent).
pub(crate) fn wgsl_f32(v: f32) -> String {
    format!("{v:?}")
}

#[derive(Clone, Copy, Debug)]
pub enum ScalarValue {
    Const(f32),
    Param(ScalarParam),
}

impl ScalarValue {
    pub fn get(&self, params: &Params) -> f32 {
        match self {
            ScalarValue::Const(v) => *v,
            ScalarValue::Param(h) => params.scalar(*h),
        }
    }

    pub fn wgsl(&self) -> String {
        match self {
            ScalarValue::Const(v) => wgsl_f32(*v),
            ScalarValue::Param(h) => format!("params[{}].x", h.0),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Vec3Value {
    Const(Vec3),
    Param(Vec3Param),
}

impl Vec3Value {
    pub fn get(&self, params: &Params) -> Vec3 {
        match self {
            Vec3Value::Const(v) => *v,
            Vec3Value::Param(h) => params.vec3(*h),
        }
    }

    pub fn wgsl(&self) -> String {
        match self {
            Vec3Value::Const(v) => format!(
                "vec3<f32>({}, {}, {})",
                wgsl_f32(v.x),
                wgsl_f32(v.y),
                wgsl_f32(v.z)
            ),
            Vec3Value::Param(h) => format!("params[{}].xyz", h.0),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Mat2Value {
    Const(Mat2),
    Param(Mat2Param),
}

impl Mat2Value {
    pub fn get(&self, params: &Params) -> Mat2 {
        match self {
            Mat2Value::Const(m) => *m,
            Mat2Value::Param(h) => params.mat2(*h),
        }
    }

    pub fn wgsl(&self) -> String {
        match self {
            Mat2Value::Const(m) => format!(
                "mat2x2<f32>({}, {}, {}, {})",
                wgsl_f32(m.x_axis.x),
                wgsl_f32(m.x_axis.y),
                wgsl_f32(m.y_axis.x),
                wgsl_f32(m.y_axis.y)
            ),
            Mat2Value::Param(h) => format!(
                "mat2x2<f32>(params[{i}].x, params[{i}].y, params[{i}].z, params[{i}].w)",
                i = h.0
            ),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum IntValue {
    Const(i32),
    Param(IntParam),
}

impl IntValue {
    pub fn get(&self, params: &Params) -> i32 {
        match self {
            IntValue::Const(v) => *v,
            IntValue::Param(h) => params.int(*h),
        }
    }

    pub fn wgsl(&self) -> String {
        match self {
            IntValue::Const(v) => format!("{v}"),
            IntValue::Param(h) => format!("i32(params[{}].x)", h.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mat2_roundtrip() {
        let mut p = Params::new();
        let m = Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
        let h = p.alloc_mat2(m);
        assert_eq!(p.mat2(h), m);
    }

    #[test]
    fn wgsl_literals_have_decimal_points() {
        assert_eq!(wgsl_f32(1.0), "1.0");
        assert_eq!(wgsl_f32(-0.5), "-0.5");
        assert_eq!(ScalarValue::Const(6.0).wgsl(), "6.0");
    }

    #[test]
    fn param_wgsl_references_slots() {
        let mut p = Params::new();
        let _a = p.alloc_scalar(1.0);
        let b = p.alloc_vec3(Vec3::ONE);
        assert_eq!(Vec3Value::Param(b).wgsl(), "params[1].xyz");
    }
}
