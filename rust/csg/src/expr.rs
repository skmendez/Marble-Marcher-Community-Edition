//! Animated fractals: a small, deterministic, serializable symbolic
//! expression tree for driving a [`crate::ScalarParam`] from the shared
//! simulation clock ([`crate::Tick`]) instead of each client's own local
//! wall clock.
//!
//! **Why this has to be a `Tick`-pure function, not wall-clock time**: the
//! CSG tree's shape (including any [`crate::ScalarValue::Param`] this
//! drives) is exactly what marble collision is computed against
//! (`physics::step_marbles`). `rollback::RollbackSim` already guarantees
//! marble state stays bit-identical between "simulated straight through"
//! and "simulated with rollback" — an animated geometry parameter that
//! isn't *also* a pure function of the same `Tick` domain would silently
//! break that guarantee one layer deeper: two peers' local marbles could
//! collide with visibly different geometry at the same simulated instant,
//! or a rollback resimulation could replay history against the *current*
//! animation phase instead of each historical tick's actual phase. `Expr`
//! exists specifically so `rollback::RollbackSim::advance`/`resim_from`
//! can re-evaluate every animated param at each tick they simulate
//! ([`apply_animations`]), live or replayed, with no other state involved.
//!
//! **Instruction set**: deliberately minimal — constants, the tick clock,
//! the four arithmetic operators, negation, sine/cosine, and `min`/`max`/
//! `clamp` for shaping a curve safely. No `pow`/`sqrt`/general variables/
//! conditionals; add them only when a real scene needs one. A smaller,
//! fully-auditable instruction set is worth more here than premature
//! generality, especially for something that has to stay bit-identical
//! across two independent peers' browsers.
//!
//! **Determinism of `sin`/`cos`**: verified directly against a real
//! `--release` `wasm32-unknown-unknown` build of this workspace (not just
//! assumed), two ways:
//!   1. The wasm-bindgen-generated JS glue (`marble-marcher.js`) contains
//!      zero occurrences of `Math.sin`/`Math.cos`/`Math.*` trig anywhere in
//!      its text.
//!   2. Parsed the compiled `.wasm` module's own binary import section
//!      directly (483 imports total): every single one comes from the
//!      wasm-bindgen glue module (`./marble-marcher_bg.js` — DOM/fetch/GPU
//!      bindings, `web_sys`/`js_sys` calls the app actually makes
//!      elsewhere), none named anything resembling `sin`/`cos`/`math`.
//!      WASM function imports can only ever be *named* host functions
//!      wasm-bindgen chose to wire up — there is no mechanism for a bare
//!      global like `Math.sin` to sneak in unnamed — so an empty search
//!      result here is conclusive, not just suggestive: every `f32::sin`/
//!      `f32::cos` call site in this module resolves to a function
//!      *inside* the `.wasm` binary itself (LLVM-lowered `libm`-equivalent
//!      code), not a host call.
//!
//! This matters because JS `Math.sin`/`Math.cos` genuinely does vary in its
//! last-bit rounding between browser engines (V8, SpiderMonkey, etc. don't
//! guarantee bit-identical transcendental results) and would silently break
//! cross-peer determinism if either trig function ever routed through it —
//! e.g. via a future "just call `js_sys`/`web_sys` for speed" change. If
//! that ever needs to happen, this doc comment is the flag that it would
//! reintroduce exactly the desync class `rollback.rs` exists to prevent.

use crate::{Params, ScalarParam, Tick};

/// A deterministic scalar expression, evaluated purely as a function of
/// the current [`Tick`] — see the module doc for why "purely" is load
/// bearing (no other inputs, no hidden state).
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Const(f32),
    /// The current tick, as `f32` (`tick as f32`) — no implicit unit
    /// conversion to seconds baked in; a scene that wants a per-second
    /// rate multiplies by its own `1.0 / TICK_RATE_HZ` constant, keeping
    /// this primitive minimal and unit-agnostic.
    Tick,
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Sin(Box<Expr>),
    Cos(Box<Expr>),
    Min(Box<Expr>, Box<Expr>),
    Max(Box<Expr>, Box<Expr>),
    /// `Clamp(value, lo, hi)`.
    Clamp(Box<Expr>, Box<Expr>, Box<Expr>),
}

impl Expr {
    pub fn eval(&self, tick: Tick) -> f32 {
        match self {
            Expr::Const(v) => *v,
            Expr::Tick => tick as f32,
            Expr::Add(a, b) => a.eval(tick) + b.eval(tick),
            Expr::Sub(a, b) => a.eval(tick) - b.eval(tick),
            Expr::Mul(a, b) => a.eval(tick) * b.eval(tick),
            Expr::Div(a, b) => a.eval(tick) / b.eval(tick),
            Expr::Neg(a) => -a.eval(tick),
            Expr::Sin(a) => a.eval(tick).sin(),
            Expr::Cos(a) => a.eval(tick).cos(),
            Expr::Min(a, b) => a.eval(tick).min(b.eval(tick)),
            Expr::Max(a, b) => a.eval(tick).max(b.eval(tick)),
            Expr::Clamp(v, lo, hi) => v.eval(tick).clamp(lo.eval(tick), hi.eval(tick)),
        }
    }

    /// Serializes to a compact, tag-prefixed byte encoding (hand-rolled,
    /// matching this codebase's existing wire-format convention —
    /// `net.rs`'s fixed-size `PlayerInput` packing — rather than pulling
    /// in `serde` for one small, tightly-controlled type). Unlike
    /// `PlayerInput`'s fixed 32 bytes, `Expr` is a tree, so this recurses:
    /// one tag byte per node, `Const`'s tag followed by 4 little-endian
    /// bytes, everything else followed by its operands' own encodings back
    /// to back (no length prefixes needed — [`Self::decode`] always knows
    /// exactly how many operands a tag has).
    pub fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Expr::Const(v) => {
                out.push(0);
                out.extend_from_slice(&v.to_le_bytes());
            }
            Expr::Tick => out.push(1),
            Expr::Add(a, b) => {
                out.push(2);
                a.encode(out);
                b.encode(out);
            }
            Expr::Sub(a, b) => {
                out.push(3);
                a.encode(out);
                b.encode(out);
            }
            Expr::Mul(a, b) => {
                out.push(4);
                a.encode(out);
                b.encode(out);
            }
            Expr::Div(a, b) => {
                out.push(5);
                a.encode(out);
                b.encode(out);
            }
            Expr::Neg(a) => {
                out.push(6);
                a.encode(out);
            }
            Expr::Sin(a) => {
                out.push(7);
                a.encode(out);
            }
            Expr::Cos(a) => {
                out.push(8);
                a.encode(out);
            }
            Expr::Min(a, b) => {
                out.push(9);
                a.encode(out);
                b.encode(out);
            }
            Expr::Max(a, b) => {
                out.push(10);
                a.encode(out);
                b.encode(out);
            }
            Expr::Clamp(v, lo, hi) => {
                out.push(11);
                v.encode(out);
                lo.encode(out);
                hi.encode(out);
            }
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.encode(&mut out);
        out
    }

    /// Inverse of [`Self::to_bytes`]/[`Self::encode`] — `None` on any
    /// malformed/truncated input (an unknown tag, or fewer bytes than a
    /// tag's operands need) rather than panicking, since this is meant to
    /// decode a message that in principle arrived over the network from
    /// another peer, not just data this process generated itself. Also
    /// `None` if `bytes` has leftover data after a complete tree decodes
    /// (a truncation/corruption signal in the other direction) — a valid
    /// decode consumes exactly the whole slice.
    pub fn from_bytes(bytes: &[u8]) -> Option<Expr> {
        let (expr, consumed) = Self::decode_at(bytes, 0)?;
        if consumed == bytes.len() {
            Some(expr)
        } else {
            None
        }
    }

    fn decode_at(bytes: &[u8], pos: usize) -> Option<(Expr, usize)> {
        let tag = *bytes.get(pos)?;
        let mut pos = pos + 1;
        let next = |pos: &mut usize| -> Option<Expr> {
            let (e, consumed) = Self::decode_at(bytes, *pos)?;
            *pos = consumed;
            Some(e)
        };
        let expr = match tag {
            0 => {
                let end = pos + 4;
                let bytes4: [u8; 4] = bytes.get(pos..end)?.try_into().ok()?;
                pos = end;
                Expr::Const(f32::from_le_bytes(bytes4))
            }
            1 => Expr::Tick,
            2 => Expr::Add(Box::new(next(&mut pos)?), Box::new(next(&mut pos)?)),
            3 => Expr::Sub(Box::new(next(&mut pos)?), Box::new(next(&mut pos)?)),
            4 => Expr::Mul(Box::new(next(&mut pos)?), Box::new(next(&mut pos)?)),
            5 => Expr::Div(Box::new(next(&mut pos)?), Box::new(next(&mut pos)?)),
            6 => Expr::Neg(Box::new(next(&mut pos)?)),
            7 => Expr::Sin(Box::new(next(&mut pos)?)),
            8 => Expr::Cos(Box::new(next(&mut pos)?)),
            9 => Expr::Min(Box::new(next(&mut pos)?), Box::new(next(&mut pos)?)),
            10 => Expr::Max(Box::new(next(&mut pos)?), Box::new(next(&mut pos)?)),
            11 => {
                let v = next(&mut pos)?;
                let lo = next(&mut pos)?;
                let hi = next(&mut pos)?;
                Expr::Clamp(Box::new(v), Box::new(lo), Box::new(hi))
            }
            _ => return None,
        };
        Some((expr, pos))
    }
}

/// Evaluates every `(handle, expr)` pair in `animations` against `tick`
/// and writes the result into `params` — the one place this actually
/// happens, called both from the offline per-tick path
/// (`physics_sys.rs`) and from inside `rollback::RollbackSim::advance`/
/// `resim_from` for every tick they simulate, live or replayed, so both
/// paths can never disagree about what a given tick's animated params
/// were.
pub fn apply_animations(params: &mut Params, animations: &[(ScalarParam, Expr)], tick: Tick) {
    for (handle, expr) in animations {
        params.set_scalar(*handle, expr.eval(tick));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(v: f32) -> Expr {
        Expr::Const(v)
    }

    #[test]
    fn const_and_tick_eval() {
        assert_eq!(c(3.5).eval(0), 3.5);
        assert_eq!(c(3.5).eval(999), 3.5, "a constant must not depend on tick");
        assert_eq!(Expr::Tick.eval(42), 42.0);
    }

    #[test]
    fn arithmetic_evals_correctly() {
        let e = Expr::Add(Box::new(c(2.0)), Box::new(Expr::Mul(Box::new(c(3.0)), Box::new(c(4.0)))));
        assert_eq!(e.eval(0), 2.0 + 3.0 * 4.0);

        let e = Expr::Div(Box::new(c(10.0)), Box::new(Expr::Sub(Box::new(c(7.0)), Box::new(c(2.0)))));
        assert_eq!(e.eval(0), 10.0 / (7.0 - 2.0));

        assert_eq!(Expr::Neg(Box::new(c(5.0))).eval(0), -5.0);
    }

    #[test]
    fn trig_matches_std_at_a_known_tick() {
        let angle = Expr::Mul(Box::new(Expr::Tick), Box::new(c(0.1)));
        let sin = Expr::Sin(Box::new(angle.clone()));
        let cos = Expr::Cos(Box::new(angle));
        let tick: Tick = 37;
        assert_eq!(sin.eval(tick), (tick as f32 * 0.1).sin());
        assert_eq!(cos.eval(tick), (tick as f32 * 0.1).cos());
    }

    #[test]
    fn min_max_clamp() {
        assert_eq!(Expr::Min(Box::new(c(3.0)), Box::new(c(5.0))).eval(0), 3.0);
        assert_eq!(Expr::Max(Box::new(c(3.0)), Box::new(c(5.0))).eval(0), 5.0);
        let clamp = Expr::Clamp(Box::new(c(10.0)), Box::new(c(0.0)), Box::new(c(1.0)));
        assert_eq!(clamp.eval(0), 1.0);
        let clamp = Expr::Clamp(Box::new(c(-10.0)), Box::new(c(0.0)), Box::new(c(1.0)));
        assert_eq!(clamp.eval(0), 0.0);
    }

    #[test]
    fn eval_is_a_pure_function_of_tick_repeated_calls_agree() {
        // Not a substitute for real cross-peer determinism (that needs the
        // wasm-level check described in the module doc), but a real,
        // cheap regression guard: nothing here should read any ambient
        // state (thread-local RNG, wall-clock time, etc.) that could make
        // two calls with the same tick disagree.
        let e = Expr::Sin(Box::new(Expr::Mul(Box::new(Expr::Tick), Box::new(c(0.037)))));
        let first = e.eval(12345);
        for _ in 0..1000 {
            assert_eq!(e.eval(12345), first);
        }
    }

    fn round_trip(e: Expr) {
        let bytes = e.to_bytes();
        let decoded = Expr::from_bytes(&bytes).unwrap_or_else(|| panic!("failed to decode {e:?}"));
        assert_eq!(decoded, e, "round-trip mismatch for {e:?}");
    }

    #[test]
    fn encode_decode_round_trips() {
        round_trip(c(1.0));
        round_trip(Expr::Tick);
        round_trip(Expr::Add(Box::new(c(1.0)), Box::new(c(2.0))));
        round_trip(Expr::Neg(Box::new(Expr::Sin(Box::new(c(-1.5))))));
        round_trip(Expr::Clamp(Box::new(Expr::Tick), Box::new(c(0.0)), Box::new(c(1.0))));
        // A deeply nested tree, to exercise real recursion in both
        // directions, not just depth-1 trees.
        let nested = Expr::Add(
            Box::new(c(1.0)),
            Box::new(Expr::Mul(
                Box::new(Expr::Sin(Box::new(Expr::Mul(Box::new(Expr::Tick), Box::new(c(0.5)))))),
                Box::new(Expr::Clamp(
                    Box::new(Expr::Cos(Box::new(Expr::Tick))),
                    Box::new(Expr::Neg(Box::new(c(2.0)))),
                    Box::new(c(2.0)),
                )),
            )),
        );
        round_trip(nested);
    }

    #[test]
    fn decode_rejects_truncated_or_malformed_bytes() {
        assert!(Expr::from_bytes(&[]).is_none());
        assert!(Expr::from_bytes(&[2]).is_none(), "Add with no operands");
        assert!(Expr::from_bytes(&[0, 1, 2]).is_none(), "Const with only 2 of 4 payload bytes");
        assert!(Expr::from_bytes(&[255]).is_none(), "unknown tag");
        // Trailing garbage after a complete, otherwise-valid tree.
        let mut bytes = c(1.0).to_bytes();
        bytes.push(0xFF);
        assert!(Expr::from_bytes(&bytes).is_none(), "leftover bytes after a complete decode");
    }

    #[test]
    fn apply_animations_writes_evaluated_values_into_params() {
        let mut params = Params::new();
        let a = params.alloc_scalar(0.0);
        let b = params.alloc_scalar(0.0);
        let animations = vec![
            (a, c(7.0)),
            (b, Expr::Mul(Box::new(Expr::Tick), Box::new(c(2.0)))),
        ];
        apply_animations(&mut params, &animations, 5);
        assert_eq!(params.scalar(a), 7.0);
        assert_eq!(params.scalar(b), 10.0);
        apply_animations(&mut params, &animations, 6);
        assert_eq!(params.scalar(a), 7.0, "a constant-driven param must not change with tick");
        assert_eq!(params.scalar(b), 12.0);
    }
}
