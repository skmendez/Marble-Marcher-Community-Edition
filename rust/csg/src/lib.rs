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
//!  - `expr`     (animated fractals): a small deterministic, serializable
//!    expression tree for driving a `Param` from the shared `Tick` clock —
//!    see its module doc for why this has to share the rollback crate's
//!    tick domain, not just happen to use the same integer type
//!
//! Rollback netcode (input buffering + snapshot/rewind/resimulate around
//! `physics::step_marbles`) lives in the separate `marble-rollback` crate,
//! not here — real-time netcode, not CSG geometry, even though it depends
//! on this crate's public API.

pub mod codegen;
pub mod expr;
pub mod fold;
pub mod object;
pub mod physics;
pub mod scene_sync;
pub mod scenes;

pub use fold::Fold;
pub use object::Object;

use glam::{Mat2, Vec2, Vec3, Vec4};

/// Rollback/animation's shared unit of time — one simulated physics tick,
/// *not* wall-clock time. Lives here (not in the separate `marble-rollback`
/// crate, despite that being where it was first introduced) because `expr`
/// needs it too and `marble-rollback` needs `expr::Expr` — re-exported from
/// `marble_rollback` unchanged so existing `rollback::Tick`-style
/// references (now `marble_rollback::Tick`) don't need to reach into this
/// crate directly.
pub type Tick = u64;

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

    fn encode(self, out: &mut Vec<u8>) {
        out.push(self.index() as u8);
    }

    fn decode_at(bytes: &[u8], pos: usize) -> Option<(Axis, usize)> {
        let axis = match *bytes.get(pos)? {
            0 => Axis::X,
            1 => Axis::Y,
            2 => Axis::Z,
            _ => return None,
        };
        Some((axis, pos + 1))
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

impl ScalarParam {
    fn encode(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0.to_le_bytes());
    }

    fn decode_at(bytes: &[u8], pos: usize) -> Option<(ScalarParam, usize)> {
        let end = pos + 2;
        Some((ScalarParam(u16::from_le_bytes(bytes.get(pos..end)?.try_into().ok()?)), end))
    }
}

impl Vec3Param {
    fn encode(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0.to_le_bytes());
    }

    fn decode_at(bytes: &[u8], pos: usize) -> Option<(Vec3Param, usize)> {
        let end = pos + 2;
        Some((Vec3Param(u16::from_le_bytes(bytes.get(pos..end)?.try_into().ok()?)), end))
    }
}

impl Mat2Param {
    fn encode(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0.to_le_bytes());
    }

    fn decode_at(bytes: &[u8], pos: usize) -> Option<(Mat2Param, usize)> {
        let end = pos + 2;
        Some((Mat2Param(u16::from_le_bytes(bytes.get(pos..end)?.try_into().ok()?)), end))
    }
}

impl IntParam {
    fn encode(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0.to_le_bytes());
    }

    fn decode_at(bytes: &[u8], pos: usize) -> Option<(IntParam, usize)> {
        let end = pos + 2;
        Some((IntParam(u16::from_le_bytes(bytes.get(pos..end)?.try_into().ok()?)), end))
    }
}

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

    /// Serializes the whole slot table as a flat, length-prefixed `f32`
    /// array — multiplayer's join-time scene sync
    /// (`scene_sync::SceneBundle`) sends this alongside the `Object`/`Fold`
    /// tree that indexes into it, since a `*Param` handle is only ever
    /// meaningful together with the slots it indexes.
    pub fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.slots.len() as u32).to_le_bytes());
        for v in &self.slots {
            out.extend_from_slice(&v.x.to_le_bytes());
            out.extend_from_slice(&v.y.to_le_bytes());
            out.extend_from_slice(&v.z.to_le_bytes());
            out.extend_from_slice(&v.w.to_le_bytes());
        }
    }

    fn decode_at(bytes: &[u8], pos: usize) -> Option<(Params, usize)> {
        let mut pos = pos;
        let count = u32::from_le_bytes(bytes.get(pos..pos + 4)?.try_into().ok()?) as usize;
        pos += 4;
        let mut slots = Vec::with_capacity(count);
        for _ in 0..count {
            let end = pos + 16;
            let chunk = bytes.get(pos..end)?;
            let f = |lo: usize| f32::from_le_bytes(chunk[lo..lo + 4].try_into().unwrap());
            slots.push(Vec4::new(f(0), f(4), f(8), f(12)));
            pos = end;
        }
        Some((Params { slots }, pos))
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

    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            ScalarValue::Const(v) => {
                out.push(0);
                out.extend_from_slice(&v.to_le_bytes());
            }
            ScalarValue::Param(h) => {
                out.push(1);
                h.encode(out);
            }
        }
    }

    fn decode_at(bytes: &[u8], pos: usize) -> Option<(ScalarValue, usize)> {
        match *bytes.get(pos)? {
            0 => {
                let end = pos + 1 + 4;
                let v = f32::from_le_bytes(bytes.get(pos + 1..end)?.try_into().ok()?);
                Some((ScalarValue::Const(v), end))
            }
            1 => {
                let (h, end) = ScalarParam::decode_at(bytes, pos + 1)?;
                Some((ScalarValue::Param(h), end))
            }
            _ => None,
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

    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Vec3Value::Const(v) => {
                out.push(0);
                out.extend_from_slice(&v.x.to_le_bytes());
                out.extend_from_slice(&v.y.to_le_bytes());
                out.extend_from_slice(&v.z.to_le_bytes());
            }
            Vec3Value::Param(h) => {
                out.push(1);
                h.encode(out);
            }
        }
    }

    fn decode_at(bytes: &[u8], pos: usize) -> Option<(Vec3Value, usize)> {
        match *bytes.get(pos)? {
            0 => {
                let end = pos + 1 + 12;
                let chunk = bytes.get(pos + 1..end)?;
                let f = |lo: usize| f32::from_le_bytes(chunk[lo..lo + 4].try_into().unwrap());
                Some((Vec3Value::Const(Vec3::new(f(0), f(4), f(8))), end))
            }
            1 => {
                let (h, end) = Vec3Param::decode_at(bytes, pos + 1)?;
                Some((Vec3Value::Param(h), end))
            }
            _ => None,
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

    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Mat2Value::Const(m) => {
                out.push(0);
                let packed = pack_mat2(*m);
                out.extend_from_slice(&packed.x.to_le_bytes());
                out.extend_from_slice(&packed.y.to_le_bytes());
                out.extend_from_slice(&packed.z.to_le_bytes());
                out.extend_from_slice(&packed.w.to_le_bytes());
            }
            Mat2Value::Param(h) => {
                out.push(1);
                h.encode(out);
            }
        }
    }

    fn decode_at(bytes: &[u8], pos: usize) -> Option<(Mat2Value, usize)> {
        match *bytes.get(pos)? {
            0 => {
                let end = pos + 1 + 16;
                let chunk = bytes.get(pos + 1..end)?;
                let f = |lo: usize| f32::from_le_bytes(chunk[lo..lo + 4].try_into().unwrap());
                Some((Mat2Value::Const(unpack_mat2(Vec4::new(f(0), f(4), f(8), f(12)))), end))
            }
            1 => {
                let (h, end) = Mat2Param::decode_at(bytes, pos + 1)?;
                Some((Mat2Value::Param(h), end))
            }
            _ => None,
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

    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            IntValue::Const(v) => {
                out.push(0);
                out.extend_from_slice(&v.to_le_bytes());
            }
            IntValue::Param(h) => {
                out.push(1);
                h.encode(out);
            }
        }
    }

    fn decode_at(bytes: &[u8], pos: usize) -> Option<(IntValue, usize)> {
        match *bytes.get(pos)? {
            0 => {
                let end = pos + 1 + 4;
                let v = i32::from_le_bytes(bytes.get(pos + 1..end)?.try_into().ok()?);
                Some((IntValue::Const(v), end))
            }
            1 => {
                let (h, end) = IntParam::decode_at(bytes, pos + 1)?;
                Some((IntValue::Param(h), end))
            }
            _ => None,
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
