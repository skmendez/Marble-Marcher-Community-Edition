//! M2: the `Fold` enum — space folds + orbit trap ops.
//! See rust/DESIGN.md §3–4 and the C++ sources in src/fractals/Fold*.hpp, Orbit*.hpp.

use glam::{Vec2, Vec3, Vec4};

use crate::{Axis, IntValue, Mat2Value, Params, ScalarValue, Vec3Value};

/// A single space-fold step (or a composite of several). Mirrors the C++
/// `FoldableBase` hierarchy (src/fractals/Fold*.hpp, Orbit*.hpp) as a closed
/// enum instead of a virtual class hierarchy (see DESIGN.md §10.4).
#[derive(Clone, Debug)]
pub enum Fold {
    /// src/fractals/FoldAbs.hpp
    Abs,
    /// src/fractals/FoldMenger.hpp
    Menger,
    /// src/fractals/FoldRotate.hpp
    Rotate { axis: Axis, mat: Mat2Value },
    /// src/fractals/FoldScaleTranslate.hpp
    ScaleTranslate {
        scale: ScalarValue,
        shift: Vec3Value,
    },
    /// src/fractals/FoldPlane.hpp
    Plane {
        normal: Vec3Value,
        offset: ScalarValue,
    },
    /// src/fractals/FoldModulo.hpp
    Modulo { axis: Axis, modulus: ScalarValue },
    /// src/fractals/FoldSeries.hpp
    Series(Vec<Fold>),
    /// src/fractals/FoldRepeat.hpp
    Repeat { count: IntValue, inner: Box<Fold> },
    /// src/fractals/OrbitInit.hpp — CPU no-op, GPU/color-pass only.
    OrbitInit(Vec3Value),
    /// src/fractals/OrbitMax.hpp — CPU no-op, GPU/color-pass only.
    OrbitMax(Vec3Value),
    /// Two-color helical "barber pole" stripes in toroidal coordinates
    /// around the Y axis (no C++ counterpart) — CPU no-op, GPU/color-pass
    /// only, like the other orbit ops. Sets `orbit` outright (no
    /// `OrbitInit` needed before it) from the stripe phase
    /// `ring_count * phi + twist_count * theta`, where `phi = atan2(z, x)`
    /// (toroidal angle around the ring) and `theta = atan2(y, |p.xz| -
    /// major)` (poloidal angle around the tube).
    ///
    /// This op exists because the `OrbitInit`/`OrbitMax` algebra
    /// structurally cannot express it: orbit accumulation is a
    /// componentwise max of *linear* functions of folded coordinates, and
    /// the only periodicity available comes from folds — which also fold
    /// the geometry, so only the object's own symmetries are usable. A
    /// helix is invariant under a *screw* motion (rotate around Y while
    /// advancing the tube angle), which is not a rigid symmetry of
    /// anything, so no fold arrangement can produce it.
    ///
    /// **Seamless closure is automatic for integer counts**: crossing
    /// either `atan2` branch cut jumps the phase by exactly
    /// `count * 2*PI`, invisible to the `sin` the stripe test applies —
    /// the pattern meets itself perfectly around both loops, for any
    /// integer `ring_count`/`twist_count` (including 0: pure poloidal or
    /// pure toroidal bands). Counts are `IntValue`s so they can be
    /// params (live-editable, `Expr`-animatable).
    OrbitBarberPole {
        /// Ring radius the poloidal angle is measured against — pass the
        /// torus's own `major` value (a shared `Param` keeps them in sync
        /// when resized live).
        major: ScalarValue,
        /// Stripe periods around the ring (toroidal).
        ring_count: IntValue,
        /// Stripe periods around the tube cross-section (poloidal); the
        /// stripes' visual tilt is set by the ratio of the two counts.
        twist_count: IntValue,
        color_a: Vec3Value,
        color_b: Vec3Value,
    },
    /// N-fold angular repetition around `axis` (hg_sdf's classic
    /// `pModPolar`; no C++ counterpart in this repo): rotates the query
    /// point's around-the-axis angle into the canonical sector of width
    /// `2*PI / count` centered on angle `phase`. A **piecewise pure
    /// rotation** — every piece is an isometry, so the folded field stays
    /// exact wherever the base is exact, same soundness class as `Rotate`/
    /// `Modulo`. One primitive in the sector unfolds to `count` copies
    /// arranged around the axis: gear teeth, spokes, turbine blades,
    /// flower petals.
    ///
    /// `phase` doubles as a **rotation animation carrier**: the animation
    /// table drives `ScalarParam`s only (not `Mat2Param`s), so a spinning
    /// N-fold assembly is expressed by `Expr`-driving `phase` — exactly
    /// how the gears scene rotates its teeth deterministically across
    /// rollback. The angular pair convention matches `Rotate`'s cyclic
    /// components (`rotate_components`): the canonical sector's center
    /// direction (at `phase = 0`) is the `+c1` axis.
    PolarModulo {
        axis: Axis,
        count: IntValue,
        phase: ScalarValue,
    },
}

/// Component indices `(c1, c2)` rotated by `FoldRotate` for a given axis,
/// cyclic per DESIGN.md §4: X→(y,z), Y→(z,x), Z→(x,y). This intentionally
/// matches the C++ CPU path (`FoldRotate::AccessComponent`) for every axis,
/// including the Y axis where the original GLSL used `p.xz` instead — a
/// known C++ inconsistency we do not replicate (DESIGN.md §10.3).
fn rotate_components(axis: Axis) -> (usize, usize) {
    let i = axis.index();
    ((i + 1) % 3, (i + 2) % 3)
}

fn menger_fold(p: &mut Vec4) {
    let mut a = (p.x - p.y).min(0.0);
    p.x -= a;
    p.y += a;
    a = (p.x - p.z).min(0.0);
    p.x -= a;
    p.z += a;
    a = (p.y - p.z).min(0.0);
    p.y -= a;
    p.z += a;
}

fn menger_unfold(p: Vec4, n: &mut Vec3) {
    let mx = p.x.max(p.y);
    if p.x.min(p.y) < mx.min(p.z) {
        std::mem::swap(&mut n.y, &mut n.z);
    }
    if mx < p.z {
        std::mem::swap(&mut n.x, &mut n.z);
    }
    if p.x < p.y {
        std::mem::swap(&mut n.x, &mut n.y);
    }
}

/// Euclidean modulo (result always in `[0, b)`), matching the C++
/// `FoldModulo::fmodulo` helper. `f32::rem_euclid` has identical semantics.
fn fmodulo(a: f32, b: f32) -> f32 {
    a.rem_euclid(b)
}

impl Fold {
    /// Apply this fold to a point in place, discarding any history needed to
    /// invert it. Mirrors the C++ `FoldableBase::Fold(Vector4f&)` overload.
    pub fn fold(&self, p: &mut Vec4, params: &Params) {
        match self {
            Fold::Abs => {
                *p = Vec4::new(p.x.abs(), p.y.abs(), p.z.abs(), p.w);
            }
            Fold::Menger => menger_fold(p),
            Fold::Rotate { axis, mat } => {
                let m = mat.get(params);
                let (c1, c2) = rotate_components(*axis);
                let v = m * Vec2::new(p[c1], p[c2]);
                p[c1] = v.x;
                p[c2] = v.y;
            }
            Fold::ScaleTranslate { scale, shift } => {
                *p *= scale.get(params);
                let t = shift.get(params);
                p.x += t.x;
                p.y += t.y;
                p.z += t.z;
            }
            Fold::Plane { normal, offset } => {
                let norm = normal.get(params);
                let off = offset.get(params);
                let d = 2.0 * (p.truncate().dot(norm) - off).min(0.0);
                p.x -= d * norm.x;
                p.y -= d * norm.y;
                p.z -= d * norm.z;
            }
            Fold::Modulo { axis, modulus } => {
                let m = modulus.get(params);
                let i = axis.index();
                p[i] = (fmodulo(p[i] - m / 2.0, m) - m / 2.0).abs();
            }
            Fold::Series(folds) => {
                for f in folds {
                    f.fold(p, params);
                }
            }
            Fold::Repeat { count, inner } => {
                for _ in 0..count.get(params) {
                    inner.fold(p, params);
                }
            }
            Fold::OrbitInit(_) | Fold::OrbitMax(_) | Fold::OrbitBarberPole { .. } => {}
            Fold::PolarModulo { axis, count, phase } => {
                let n = count.get(params).max(1) as f32;
                let sector = std::f32::consts::TAU / n;
                let ph = phase.get(params);
                let (c1, c2) = rotate_components(*axis);
                let a = p[c2].atan2(p[c1]) - ph;
                let k = (a / sector).round();
                let d = a - k * sector;
                let rho = (p[c1] * p[c1] + p[c2] * p[c2]).sqrt();
                p[c1] = rho * d.cos();
                p[c2] = rho * d.sin();
            }
        }
    }

    /// Apply this fold, pushing whatever pre-fold state is needed to invert
    /// it later onto `hist`. Push/pop contract (DESIGN.md §4): `Abs`,
    /// `Menger`, `Plane`, `Modulo` push the pre-fold `p`; `Rotate` and
    /// `ScaleTranslate` push nothing (closed-form unfold); `Series` and
    /// `Repeat` recurse.
    pub fn fold_with_history(&self, p: &mut Vec4, hist: &mut Vec<Vec4>, params: &Params) {
        match self {
            Fold::Abs
            | Fold::Menger
            | Fold::Plane { .. }
            | Fold::Modulo { .. }
            | Fold::PolarModulo { .. } => {
                hist.push(*p);
                self.fold(p, params);
            }
            Fold::Rotate { .. } | Fold::ScaleTranslate { .. } => self.fold(p, params),
            Fold::Series(folds) => {
                for f in folds {
                    f.fold_with_history(p, hist, params);
                }
            }
            Fold::Repeat { count, inner } => {
                for _ in 0..count.get(params) {
                    inner.fold_with_history(p, hist, params);
                }
            }
            Fold::OrbitInit(_) | Fold::OrbitMax(_) | Fold::OrbitBarberPole { .. } => {}
        }
    }

    /// Invert this fold's effect on a surface normal, popping history pushed
    /// by `fold_with_history`. Must be called with the exact same `hist`
    /// stack and in the mirror order of the corresponding `fold_with_history`
    /// call (`Series` unfolds in reverse; `Repeat` unfolds `count` times,
    /// which is order-correct because history is a LIFO stack and each call
    /// pops a fixed number of entries — see DESIGN.md §4).
    pub fn unfold(&self, hist: &mut Vec<Vec4>, n: &mut Vec3, params: &Params) {
        match self {
            Fold::Abs => {
                let p = hist.pop().expect("fold history underflow");
                if p.x < 0.0 {
                    n.x = -n.x;
                }
                if p.y < 0.0 {
                    n.y = -n.y;
                }
                if p.z < 0.0 {
                    n.z = -n.z;
                }
            }
            Fold::Menger => {
                let p = hist.pop().expect("fold history underflow");
                menger_unfold(p, n);
            }
            Fold::Rotate { axis, mat } => {
                let m = mat.get(params).transpose();
                let (c1, c2) = rotate_components(*axis);
                let v = m * Vec2::new(n[c1], n[c2]);
                n[c1] = v.x;
                n[c2] = v.y;
            }
            Fold::ScaleTranslate { scale, shift } => {
                *n -= shift.get(params);
                *n /= scale.get(params);
            }
            Fold::Plane { normal, offset } => {
                let p = hist.pop().expect("fold history underflow");
                let norm = normal.get(params);
                let off = offset.get(params);
                if p.truncate().dot(norm) - off < 0.0 {
                    *n -= 2.0 * (n.dot(norm) - off) * norm;
                }
            }
            Fold::Modulo { axis, modulus } => {
                let p = hist.pop().expect("fold history underflow");
                let m = modulus.get(params);
                let i = axis.index();
                let a = fmodulo(p[i] - m / 2.0, m) - m / 2.0;
                if a < 0.0 {
                    n[i] = -n[i];
                }
                n[i] += p[i] - a;
            }
            Fold::Series(folds) => {
                for f in folds.iter().rev() {
                    f.unfold(hist, n, params);
                }
            }
            Fold::Repeat { count, inner } => {
                for _ in 0..count.get(params) {
                    inner.unfold(hist, n, params);
                }
            }
            Fold::OrbitInit(_) | Fold::OrbitMax(_) | Fold::OrbitBarberPole { .. } => {}
            // The fold rotated the (c1, c2) pair from angle `theta0` (the
            // pre-fold angle, recoverable from history) down to the
            // canonical `d = theta0 - ph - k*sector`; the inverse rotation
            // (by `ph + k*sector`) maps the normal back out to the copy
            // the query actually hit.
            Fold::PolarModulo { axis, count, phase } => {
                let p = hist.pop().expect("fold history underflow");
                let n_count = count.get(params).max(1) as f32;
                let sector = std::f32::consts::TAU / n_count;
                let ph = phase.get(params);
                let (c1, c2) = rotate_components(*axis);
                let theta0 = p[c2].atan2(p[c1]);
                let k = ((theta0 - ph) / sector).round();
                let back = ph + k * sector;
                let (sb, cb) = back.sin_cos();
                let (a, b) = (n[c1], n[c2]);
                n[c1] = cb * a - sb * b;
                n[c2] = sb * a + cb * b;
            }
        }
    }

    /// Given a world-space-`Some`-or-unbounded bounding sphere `child` for
    /// whatever this fold's *output* feeds into, returns a bounding sphere
    /// for this fold's *input* (i.e. propagates the bound backward through
    /// the fold, since `fold()`/`de()` apply folds to the query point before
    /// evaluating what's inside). `None` means unbounded -- correct (not a
    /// missing case) for `Modulo`, which tiles infinitely on its axis.
    ///
    /// Every case here only needs to be a *sound* (possibly loose) outer
    /// bound of the true preimage, not tight: `x` maps into `child`'s bound
    /// implies `x` is in the returned bound, nothing stronger. That's all
    /// `Object::bounding_sphere`'s callers need for a ray-clip pre-test --
    /// see its doc for why a loose-but-correct bound is fine and an
    /// under-approximation is the one failure mode that actually matters.
    pub fn unfold_bounding_sphere(
        &self,
        child: Option<(Vec3, f32)>,
        params: &Params,
    ) -> Option<(Vec3, f32)> {
        let (c, r) = child?;
        match self {
            // Abs/Menger are compositions of reflections/permutations through
            // origin-containing planes, so `length(x)` is exactly preserved
            // -- but as *set* maps (not single-point maps) they're many-to-one
            // (every octant/ordering maps onto one canonical region), so the
            // preimage of a sphere not centered at the origin is the union of
            // its images across every octant/ordering, all at the same
            // distance from the origin as `c` itself. Re-centering at the
            // origin with radius `‖c‖ + r` is the exact enclosing sphere of
            // that union (not merely a conservative pad).
            Fold::Abs | Fold::Menger => Some((Vec3::ZERO, c.length() + r)),
            Fold::Rotate { axis, mat } => {
                let m = mat.get(params).transpose();
                let (c1, c2) = rotate_components(*axis);
                let mut center = c;
                let v = m * Vec2::new(c[c1], c[c2]);
                center[c1] = v.x;
                center[c2] = v.y;
                Some((center, r))
            }
            Fold::ScaleTranslate { scale, shift } => {
                let s = scale.get(params);
                Some(((c - shift.get(params)) / s, r / s.abs()))
            }
            Fold::Plane { normal, offset } => {
                // A conditional reflection (isometric, but not necessarily
                // through the origin) -- enclose `child`'s bound *and* its
                // mirror image across the plane, since a preimage point could
                // have come from either the reflected or unreflected branch.
                // Loose (the true preimage only ever needs one branch per
                // point) but always sound.
                let norm = normal.get(params);
                let off = offset.get(params);
                let mirrored_c = c - 2.0 * (c.dot(norm) - off) * norm;
                let center = (c + mirrored_c) * 0.5;
                let spread = (c - center).length();
                Some((center, r + spread))
            }
            // Tiles infinitely on this axis -- genuinely unbounded, not a
            // gap in this function. `Object::Intersect` is what turns this
            // back into a finite bound when a scene actually needs one (see
            // its doc + `creme_spheres`).
            Fold::Modulo { .. } => None,
            Fold::Series(folds) => {
                // Folds apply forward as `folds[0]` then `folds[1]` ...
                // `folds[n-1]`; un-folding a bound walks back from the last
                // fold applied to the first.
                let mut bound = Some((c, r));
                for f in folds.iter().rev() {
                    bound = f.unfold_bounding_sphere(bound, params);
                }
                bound
            }
            Fold::Repeat { count, inner } => {
                let mut bound = Some((c, r));
                for _ in 0..count.get(params) {
                    bound = inner.unfold_bounding_sphere(bound, params);
                }
                bound
            }
            // Color-pass-only no-ops (same as `fold`/`unfold`).
            Fold::OrbitInit(_) | Fold::OrbitMax(_) | Fold::OrbitBarberPole { .. } => Some((c, r)),
            // The preimage is `count` copies of the child bound rotated
            // about the axis: enclose them all with an axis-centered
            // sphere (keep the child's axial offset, pad the radius by its
            // perpendicular distance) -- same shape of argument as
            // `Abs`/`Menger`'s origin-recentering, specialized to one axis.
            Fold::PolarModulo { axis, .. } => {
                let i = axis.index();
                let mut center = Vec3::ZERO;
                center[i] = c[i];
                let mut perp = c;
                perp[i] = 0.0;
                Some((center, r + perp.length()))
            }
        }
    }

    /// Serializes to a compact, tag-prefixed byte encoding, one tag byte
    /// per node then its fields/operands back to back — same hand-rolled,
    /// self-delimiting convention as [`crate::expr::Expr::encode`] (see its
    /// doc), extended here to a tree with more than one field per node.
    /// Used by [`crate::Scene`] for multiplayer's join-time
    /// scene sync.
    pub fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Fold::Abs => out.push(0),
            Fold::Menger => out.push(1),
            Fold::Rotate { axis, mat } => {
                out.push(2);
                axis.encode(out);
                mat.encode(out);
            }
            Fold::ScaleTranslate { scale, shift } => {
                out.push(3);
                scale.encode(out);
                shift.encode(out);
            }
            Fold::Plane { normal, offset } => {
                out.push(4);
                normal.encode(out);
                offset.encode(out);
            }
            Fold::Modulo { axis, modulus } => {
                out.push(5);
                axis.encode(out);
                modulus.encode(out);
            }
            Fold::Series(folds) => {
                out.push(6);
                out.extend_from_slice(&(folds.len() as u32).to_le_bytes());
                for f in folds {
                    f.encode(out);
                }
            }
            Fold::Repeat { count, inner } => {
                out.push(7);
                count.encode(out);
                inner.encode(out);
            }
            Fold::OrbitInit(v) => {
                out.push(8);
                v.encode(out);
            }
            Fold::OrbitMax(v) => {
                out.push(9);
                v.encode(out);
            }
            Fold::OrbitBarberPole { major, ring_count, twist_count, color_a, color_b } => {
                out.push(10);
                major.encode(out);
                ring_count.encode(out);
                twist_count.encode(out);
                color_a.encode(out);
                color_b.encode(out);
            }
            Fold::PolarModulo { axis, count, phase } => {
                out.push(11);
                axis.encode(out);
                count.encode(out);
                phase.encode(out);
            }
        }
    }

    /// Whether every parameter handle in this fold (recursively, for
    /// `Series`/`Repeat`) is valid for a `Params` table with `slot_count`
    /// slots -- see `Object::handles_valid_for`'s doc for why this check
    /// exists at all.
    pub(crate) fn handles_valid_for(&self, slot_count: usize) -> bool {
        match self {
            Fold::Abs | Fold::Menger => true,
            Fold::Rotate { mat, .. } => mat.handle_valid_for(slot_count),
            Fold::ScaleTranslate { scale, shift } => {
                scale.handle_valid_for(slot_count) && shift.handle_valid_for(slot_count)
            }
            Fold::Plane { normal, offset } => {
                normal.handle_valid_for(slot_count) && offset.handle_valid_for(slot_count)
            }
            Fold::Modulo { modulus, .. } => modulus.handle_valid_for(slot_count),
            Fold::Series(folds) => folds.iter().all(|f| f.handles_valid_for(slot_count)),
            Fold::Repeat { count, inner } => {
                count.handle_valid_for(slot_count) && inner.handles_valid_for(slot_count)
            }
            Fold::OrbitInit(v) | Fold::OrbitMax(v) => v.handle_valid_for(slot_count),
            Fold::OrbitBarberPole { major, ring_count, twist_count, color_a, color_b } => {
                major.handle_valid_for(slot_count)
                    && ring_count.handle_valid_for(slot_count)
                    && twist_count.handle_valid_for(slot_count)
                    && color_a.handle_valid_for(slot_count)
                    && color_b.handle_valid_for(slot_count)
            }
            Fold::PolarModulo { count, phase, .. } => {
                count.handle_valid_for(slot_count) && phase.handle_valid_for(slot_count)
            }
        }
    }

    /// Inverse of [`Self::encode`] — see [`crate::expr::Expr::decode_at`]
    /// for the recursion shape this mirrors (`None` on any malformed/
    /// truncated input, `pos` is where the caller should resume reading).
    pub(crate) fn decode_at(bytes: &[u8], pos: usize) -> Option<(Fold, usize)> {
        let tag = *bytes.get(pos)?;
        let pos = pos + 1;
        let fold = match tag {
            0 => (Fold::Abs, pos),
            1 => (Fold::Menger, pos),
            2 => {
                let (axis, pos) = Axis::decode_at(bytes, pos)?;
                let (mat, pos) = Mat2Value::decode_at(bytes, pos)?;
                (Fold::Rotate { axis, mat }, pos)
            }
            3 => {
                let (scale, pos) = ScalarValue::decode_at(bytes, pos)?;
                let (shift, pos) = Vec3Value::decode_at(bytes, pos)?;
                (Fold::ScaleTranslate { scale, shift }, pos)
            }
            4 => {
                let (normal, pos) = Vec3Value::decode_at(bytes, pos)?;
                let (offset, pos) = ScalarValue::decode_at(bytes, pos)?;
                (Fold::Plane { normal, offset }, pos)
            }
            5 => {
                let (axis, pos) = Axis::decode_at(bytes, pos)?;
                let (modulus, pos) = ScalarValue::decode_at(bytes, pos)?;
                (Fold::Modulo { axis, modulus }, pos)
            }
            6 => {
                let count = u32::from_le_bytes(bytes.get(pos..pos + 4)?.try_into().ok()?) as usize;
                let mut pos = pos + 4;
                // Reject before allocating: the smallest possible encoded
                // `Fold` (`Abs`/`Menger`) is a single tag byte, so `count`
                // can never legitimately exceed the bytes actually left --
                // without this bound, a corrupted `count` near `u32::MAX`
                // would immediately attempt a multi-GB `Vec::with_capacity`
                // (an allocation failure aborts the process, worse than a
                // parse error) even against a tiny buffer.
                if count > bytes.len().saturating_sub(pos) {
                    return None;
                }
                let mut folds = Vec::with_capacity(count);
                for _ in 0..count {
                    let (f, next) = Fold::decode_at(bytes, pos)?;
                    folds.push(f);
                    pos = next;
                }
                (Fold::Series(folds), pos)
            }
            7 => {
                let (count, pos) = IntValue::decode_at(bytes, pos)?;
                let (inner, pos) = Fold::decode_at(bytes, pos)?;
                (Fold::Repeat { count, inner: Box::new(inner) }, pos)
            }
            8 => {
                let (v, pos) = Vec3Value::decode_at(bytes, pos)?;
                (Fold::OrbitInit(v), pos)
            }
            9 => {
                let (v, pos) = Vec3Value::decode_at(bytes, pos)?;
                (Fold::OrbitMax(v), pos)
            }
            10 => {
                let (major, pos) = ScalarValue::decode_at(bytes, pos)?;
                let (ring_count, pos) = IntValue::decode_at(bytes, pos)?;
                let (twist_count, pos) = IntValue::decode_at(bytes, pos)?;
                let (color_a, pos) = Vec3Value::decode_at(bytes, pos)?;
                let (color_b, pos) = Vec3Value::decode_at(bytes, pos)?;
                (Fold::OrbitBarberPole { major, ring_count, twist_count, color_a, color_b }, pos)
            }
            11 => {
                let (axis, pos) = Axis::decode_at(bytes, pos)?;
                let (count, pos) = IntValue::decode_at(bytes, pos)?;
                let (phase, pos) = ScalarValue::decode_at(bytes, pos)?;
                (Fold::PolarModulo { axis, count, phase }, pos)
            }
            _ => return None,
        };
        Some(fold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Mat2;

    fn rotation_mat2(angle: f32) -> Mat2 {
        let (s, c) = angle.sin_cos();
        Mat2::from_cols(Vec2::new(c, -s), Vec2::new(s, c))
    }

    #[test]
    fn abs_fold_history_push_pop() {
        let params = Params::new();
        let mut p = Vec4::new(-1.0, 2.0, -3.0, 1.0);
        let mut hist = Vec::new();
        Fold::Abs.fold_with_history(&mut p, &mut hist, &params);
        assert_eq!(p, Vec4::new(1.0, 2.0, 3.0, 1.0));
        assert_eq!(hist.len(), 1);

        let mut n = Vec3::new(1.0, 1.0, 1.0);
        Fold::Abs.unfold(&mut hist, &mut n, &params);
        assert!(hist.is_empty());
        assert_eq!(n, Vec3::new(-1.0, 1.0, -1.0));
    }

    #[test]
    fn rotate_and_scale_translate_push_nothing() {
        let params = Params::new();
        let mut p = Vec4::new(1.0, 2.0, 3.0, 1.0);
        let mut hist = Vec::new();
        let rot = Fold::Rotate {
            axis: Axis::Z,
            mat: Mat2Value::Const(rotation_mat2(0.4)),
        };
        rot.fold_with_history(&mut p, &mut hist, &params);
        assert!(hist.is_empty());

        let st = Fold::ScaleTranslate {
            scale: ScalarValue::Const(2.0),
            shift: Vec3Value::Const(Vec3::new(1.0, 0.0, 0.0)),
        };
        st.fold_with_history(&mut p, &mut hist, &params);
        assert!(hist.is_empty());
    }

    #[test]
    fn menger_fold_orders_components() {
        let params = Params::new();
        let mut p = Vec4::new(3.0, 1.0, 2.0, 1.0);
        Fold::Menger.fold(&mut p, &params);
        assert!(p.x >= p.y);
        assert!(p.y >= p.z);
    }

    #[test]
    fn modulo_fold_is_periodic() {
        let params = Params::new();
        let modulo = Fold::Modulo {
            axis: Axis::X,
            modulus: ScalarValue::Const(1.0),
        };
        for &x in &[-2.7, -0.3, 0.1, 0.9, 1.4, 5.6] {
            let mut p = Vec4::new(x, 0.0, 0.0, 1.0);
            modulo.fold(&mut p, &params);
            assert!(p.x >= 0.0 && p.x <= 0.5 + 1e-5, "x={x} folded={}", p.x);
        }
        // Same fractional offset from any multiple of the modulus folds identically.
        let mut a = Vec4::new(0.2, 0.0, 0.0, 1.0);
        let mut b = Vec4::new(3.2, 0.0, 0.0, 1.0);
        modulo.fold(&mut a, &params);
        modulo.fold(&mut b, &params);
        assert!((a.x - b.x).abs() < 1e-5);
    }

    #[test]
    fn rotate_fold_unfold_roundtrip_single() {
        let params = Params::new();
        let mat = rotation_mat2(0.73);
        let fold = Fold::Rotate {
            axis: Axis::X,
            mat: Mat2Value::Const(mat),
        };
        let orig = Vec4::new(1.0, 2.0, 3.0, 1.0);
        let mut p = orig;
        fold.fold(&mut p, &params);
        assert!((p.truncate() - orig.truncate()).length() > 1e-3); // actually rotated

        let mut n = p.truncate();
        fold.unfold(&mut Vec::new(), &mut n, &params);
        assert!(
            (n - orig.truncate()).length() < 1e-4,
            "roundtrip mismatch: {n:?} vs {:?}",
            orig.truncate()
        );
    }

    #[test]
    fn rotate_fold_unfold_roundtrip_series() {
        // A pure-rotation tree: two Rotates in series, on different axes.
        let params = Params::new();
        let fold = Fold::Series(vec![
            Fold::Rotate {
                axis: Axis::Z,
                mat: Mat2Value::Const(rotation_mat2(0.5)),
            },
            Fold::Rotate {
                axis: Axis::X,
                mat: Mat2Value::Const(rotation_mat2(-1.1)),
            },
        ]);
        let orig = Vec4::new(0.5, -1.5, 2.25, 1.0);
        let mut p = orig;
        let mut hist = Vec::new();
        fold.fold_with_history(&mut p, &mut hist, &params);
        assert!(hist.is_empty()); // Rotate pushes nothing.

        let mut n = p.truncate();
        fold.unfold(&mut hist, &mut n, &params);
        assert!(hist.is_empty());
        assert!(
            (n - orig.truncate()).length() < 1e-4,
            "roundtrip mismatch: {n:?} vs {:?}",
            orig.truncate()
        );
    }

    #[test]
    fn abs_and_menger_unfold_bounding_sphere_recenters_at_origin() {
        let params = Params::new();
        // A child sphere off-center: the preimage under Abs/Menger spans
        // every octant/ordering, so the exact enclosing bound is centered
        // at the origin with radius ||c||+r.
        let child = Some((Vec3::new(1.0, 2.0, 3.0), 0.5));
        let expected_r = Vec3::new(1.0, 2.0, 3.0).length() + 0.5;
        let (c, r) = Fold::Abs.unfold_bounding_sphere(child, &params).unwrap();
        assert_eq!(c, Vec3::ZERO);
        assert!((r - expected_r).abs() < 1e-5);
        let (c, r) = Fold::Menger.unfold_bounding_sphere(child, &params).unwrap();
        assert_eq!(c, Vec3::ZERO);
        assert!((r - expected_r).abs() < 1e-5);
    }

    #[test]
    fn scale_translate_unfold_bounding_sphere_is_exact_affine_inverse() {
        let params = Params::new();
        let fold = Fold::ScaleTranslate {
            scale: ScalarValue::Const(2.0),
            shift: Vec3Value::Const(Vec3::new(1.0, 0.0, -1.0)),
        };
        let child = Some((Vec3::new(3.0, 3.0, 3.0), 1.0));
        let (c, r) = fold.unfold_bounding_sphere(child, &params).unwrap();
        // Forward: p' = p*2 + shift. So p = (p' - shift)/2.
        assert_eq!(c, Vec3::new(1.0, 1.5, 2.0));
        assert!((r - 0.5).abs() < 1e-5);
    }

    #[test]
    fn modulo_unfold_bounding_sphere_is_unbounded() {
        let params = Params::new();
        let fold = Fold::Modulo {
            axis: Axis::X,
            modulus: ScalarValue::Const(1.0),
        };
        assert!(fold
            .unfold_bounding_sphere(Some((Vec3::ZERO, 1.0)), &params)
            .is_none());
    }

    #[test]
    fn repeat_unfold_bounding_sphere_shrinks_by_scale_per_iteration() {
        let params = Params::new();
        // A pure-contraction repeat (no folding to complicate the picture):
        // each iteration divides the radius by `scale`, applied `count`
        // times, so this should compose exactly.
        let inner = Fold::ScaleTranslate {
            scale: ScalarValue::Const(2.0),
            shift: Vec3Value::Const(Vec3::ZERO),
        };
        let repeat = Fold::Repeat {
            count: IntValue::Const(3),
            inner: Box::new(inner),
        };
        let child = Some((Vec3::ZERO, 8.0));
        let (c, r) = repeat.unfold_bounding_sphere(child, &params).unwrap();
        assert_eq!(c, Vec3::ZERO);
        assert!((r - 1.0).abs() < 1e-5); // 8 / 2^3
    }

    #[test]
    fn series_and_repeat_fold_history_roundtrip() {
        let params = Params::new();
        let inner = Fold::Series(vec![
            Fold::Abs,
            Fold::Menger,
            Fold::ScaleTranslate {
                scale: ScalarValue::Const(0.5),
                shift: Vec3Value::Const(Vec3::new(0.1, 0.2, 0.3)),
            },
        ]);
        let repeat = Fold::Repeat {
            count: IntValue::Const(4),
            inner: Box::new(inner),
        };

        let mut p = Vec4::new(1.3, -2.4, 0.7, 1.0);
        let mut hist = Vec::new();
        repeat.fold_with_history(&mut p, &mut hist, &params);
        // Abs + Menger each push once per iteration; 4 iterations * 2 pushes.
        assert_eq!(hist.len(), 8);

        let mut n = Vec3::new(1.0, 0.0, 0.0);
        repeat.unfold(&mut hist, &mut n, &params);
        assert!(hist.is_empty());
        assert!(n.is_finite());
    }

    /// Fix 6 regression test: a `Fold::Series` `count` field claiming far
    /// more nested folds than the buffer could possibly hold (the smallest
    /// possible encoded `Fold` is one tag byte, `Abs`/`Menger`) must be
    /// rejected before ever attempting `Vec::with_capacity(count)`.
    #[test]
    fn decode_at_rejects_a_series_count_that_exceeds_the_buffer() {
        let mut bytes = vec![6u8]; // Fold::Series's tag
        bytes.extend_from_slice(&1_000_000u32.to_le_bytes()); // count -- no fold data follows
        assert!(Fold::decode_at(&bytes, 0).is_none());

        let mut overflow_bytes = vec![6u8];
        overflow_bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(Fold::decode_at(&overflow_bytes, 0).is_none());
    }

    /// `PolarModulo` is an N-fold rotational symmetrizer: folding a point
    /// and folding that same point rotated by any whole number of sectors
    /// about the axis must land on the identical canonical point.
    #[test]
    fn polar_modulo_is_sector_periodic() {
        use std::f32::consts::TAU;
        let params = Params::new();
        let count = 12;
        let fold = Fold::PolarModulo {
            axis: Axis::Y,
            count: IntValue::Const(count),
            phase: ScalarValue::Const(0.4),
        };
        let base = Vec4::new(0.13, 0.46, 0.11, 1.0);
        let mut expect = base;
        fold.fold(&mut expect, &params);
        for k in 1..count {
            let ang = k as f32 * TAU / count as f32;
            let (s, c) = ang.sin_cos();
            // Rotate about +Y (the (z, x) angular pair, matching
            // rotate_components).
            let rotated = Vec4::new(
                s * base.z + c * base.x,
                base.y,
                c * base.z - s * base.x,
                base.w,
            );
            let mut p = rotated;
            fold.fold(&mut p, &params);
            assert!(
                (p - expect).length() < 1e-5,
                "sector {k}: folded {p:?} != {expect:?}"
            );
        }
    }

    /// The canonical sector is centered on +c1 (+Z for a Y axis) rotated
    /// by `phase`, and folded points land within half a sector of it.
    #[test]
    fn polar_modulo_canonicalizes_to_the_phase_centered_sector() {
        use std::f32::consts::TAU;
        let params = Params::new();
        let count = 8;
        let phase = 0.7;
        let fold = Fold::PolarModulo {
            axis: Axis::Y,
            count: IntValue::Const(count),
            phase: ScalarValue::Const(phase),
        };
        let half_sector = TAU / count as f32 / 2.0;
        for i in 0..40 {
            let ang = i as f32 * TAU / 40.0;
            let mut p = Vec4::new(ang.sin() * 2.0, 0.3, ang.cos() * 2.0, 1.0);
            let rho = 2.0;
            fold.fold(&mut p, &params);
            // Radius and y are untouched (it's an isometry per sector)...
            assert!((p.y - 0.3).abs() < 1e-6);
            assert!(((p.x * p.x + p.z * p.z).sqrt() - rho).abs() < 1e-5);
            // ...and the folded azimuth sits within half a sector of 0
            // (the fold already subtracted `phase`).
            let az = p.x.atan2(p.z);
            assert!(
                az.abs() <= half_sector + 1e-5,
                "azimuth {az} escaped the canonical sector"
            );
        }
    }

    /// History/unfold roundtrip: unfolding the canonical nearest point
    /// must produce a world point at the same distance from the query as
    /// the canonical pair -- the property `Object::nearest_point` relies
    /// on (folds are per-sector isometries).
    #[test]
    fn polar_modulo_unfold_preserves_distances() {
        let params = Params::new();
        let fold = Fold::PolarModulo {
            axis: Axis::X,
            count: IntValue::Const(5),
            phase: ScalarValue::Const(-0.9),
        };
        for query in [
            Vec4::new(0.4, -1.2, 0.5, 1.0),
            Vec4::new(-0.1, 0.3, -2.0, 1.0),
            Vec4::new(0.0, 1.7, 0.2, 1.0),
        ] {
            let mut p = query;
            let mut hist = Vec::new();
            fold.fold_with_history(&mut p, &mut hist, &params);
            assert_eq!(hist.len(), 1);
            // Pretend the canonical-space nearest point is this offset spot.
            let np_canonical = p.truncate() + Vec3::new(0.05, -0.1, 0.2);
            let d_canonical = (p.truncate() - np_canonical).length();
            let mut np = np_canonical;
            fold.unfold(&mut hist, &mut np, &params);
            assert!(hist.is_empty());
            let d_world = (query.truncate() - np).length();
            assert!(
                (d_world - d_canonical).abs() < 1e-5,
                "distance not preserved: canonical {d_canonical} vs world {d_world}"
            );
        }
    }
}
