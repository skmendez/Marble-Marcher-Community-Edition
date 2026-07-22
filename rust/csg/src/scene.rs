//! The whole CSG scene as one owned, atomic unit: the `Object`/`Fold` tree,
//! its `Params` slot table, and the `(ScalarParam, Expr)` animation table.
//!
//! **A `*Param` handle is only meaningful together with the `Params` it
//! indexes** (a handle is just a slot index — `lib.rs`'s doc): bundling
//! `object`/`params`/`animations` into one type means a tree and a slot
//! table that don't belong together can't be paired by accident — there's
//! no way to construct or hold one half without the other. This is the
//! type multiplayer's join-time scene sync serializes wholesale
//! (`scene_sync`, this module's sibling), and the type `marble_rollback::
//! RollbackSim` owns internally so resimulation always sees a consistent
//! scene across every replayed tick (same "authoritative push, no
//! independent reconstruction" principle as `RollbackSim::hard_reset_to`).

use crate::expr::Expr;
use crate::{Object, Params, ScalarParam};

/// The whole CSG scene + its live parameters + its animation table, as one
/// unit — see the module doc for why these three travel together.
#[derive(Clone, Debug)]
pub struct Scene {
    pub object: Object,
    pub params: Params,
    pub animations: Vec<(ScalarParam, Expr)>,
}
