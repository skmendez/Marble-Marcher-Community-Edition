//! Multiplayer milestone 1: rollback netcode around
//! [`physics::step_marbles`] — pure logic, no bevy (same reasoning as
//! `physics.rs`: this needs to be unit-testable without spinning up an ECS,
//! and nothing here touches anything bevy-specific).
//!
//! The one property everything below exists to guarantee: **simulating a
//! tick range with rollback (predicting missing inputs, then rewinding and
//! resimulating when a prediction turns out wrong) must produce *exactly*
//! the same final state as simulating the same tick range would have if
//! every input had been known up front.** `step_marbles` is already a pure,
//! deterministic function of `(marbles, inputs, obj, params, cfg, kill_y,
//! starts)` (no hidden state, no `HashMap`/unordered iteration anywhere in
//! its dependency chain — verified by inspection, and `PlayerInput`'s
//! `orientation: Quat` already carries the one piece of per-tick,
//! per-player state that used to be ambient/local — see `PlayerInput`'s own
//! doc), so the only way this module can go wrong is in its own
//! bookkeeping, not in the simulation it wraps.
//!
//! ## Design
//!
//! [`RollbackSim`] holds:
//!  - the *live* marble state (`step_marbles`'s output through
//!    `current_tick`),
//!  - a bounded history of past marble-state snapshots, one per recent
//!    tick, deep enough to rewind into (`window`),
//!  - a bounded per-player log of exactly which [`PlayerInput`] was *used*
//!    for each recent tick, and whether that value was a real confirmed
//!    input or a guess ([`InputStatus::Predicted`]).
//!
//! A `BTreeMap<Tick, _>` backs both the snapshot and input logs rather than
//! a literal ring-buffer array indexed by `tick % window`: same bounded-
//! memory behavior (old entries are pruned every time the window advances),
//! but no wraparound index arithmetic to get subtly wrong — this module's
//! whole job is being trustworthy about exact reproducibility, so it isn't
//! the place to hand-roll ring-buffer indexing when a `BTreeMap` gives the
//! same asymptotic behavior for free.
//!
//! **Predicting a missing input**: the simplest predictor that's still
//! correct-by-construction — repeat that player's most recent *confirmed*
//! input at or before the tick in question ([`RollbackSim::predict`]),
//! falling back to a caller-supplied default only if that player has no
//! confirmed history at all yet (the very first tick(s) of a session).
//! Critically, prediction always searches for the latest *confirmed* entry,
//! never a previously-*predicted* one — otherwise a chain of predictions
//! could compound away from what a confirmed input will eventually turn out
//! to be, and worse, two different resimulation passes could predict
//! *different* things for the same still-unconfirmed tick depending on
//! what stale prediction happened to be sitting in the log, which would
//! break the bit-identical-replay guarantee this module exists for.
//!
//! **Rewind + resim**: [`RollbackSim::receive_inputs`] takes a whole batch
//! of newly-confirmed `(player, tick, input)` arrivals at once (not one at
//! a time) specifically so that a batch touching several different past
//! ticks and/or several different players triggers *one* resimulation pass
//! covering the earliest affected tick through the present, not one pass
//! per arrival — both cheaper and, more importantly, so two arrivals that
//! individually would have picked opposite predictions for some
//! in-between tick can't do that: the whole batch is applied to the input
//! log *before* any resimulation happens, so a single resim pass always
//! sees the fully-updated picture.

use std::collections::BTreeMap;

use glam::Vec3;

use crate::expr::{apply_animations, Expr};
use crate::physics::{step_marbles, Marble, PhysicsConfig, PlayerInput, StepEvent};
use crate::{Object, Params};

/// Rollback's unit of time — one call to [`step_marbles`], not wall-clock
/// time. Starts at `0` (the initial state, before any tick has been
/// simulated) and increments by exactly `1` per [`RollbackSim::advance`]
/// call, regardless of real elapsed time — a caller driving this from a
/// fixed-timestep schedule (`FixedUpdate`, matching `physics_sys.rs`'s
/// existing convention) gets ticks that line up with real 60Hz frames for
/// free, but this module itself has no notion of wall-clock time at all.
/// Re-exported from the crate root (`Tick`'s doc there) — defined at the
/// crate root now, not here, since `expr` needs it too and this module
/// needs `expr::Expr`; kept re-exported under this path unchanged so
/// existing `rollback::Tick` references don't need updating.
pub use crate::Tick;

/// A stable index identifying which [`PlayerInput`]/[`Marble`] slot belongs
/// to which player — the exact same convention [`step_marbles`] already
/// uses (`marbles`/`inputs`/`starts` are parallel slices indexed by player,
/// never a `HashMap`), carried through here for the same determinism
/// reason.
pub type PlayerIndex = usize;

/// Whether a logged [`PlayerInput`] for some `(player, tick)` is a value we
/// actually received, or a guess this module made up because the real one
/// hadn't arrived yet when that tick needed to be simulated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputStatus {
    Confirmed,
    Predicted,
}

/// Snapshot + rewind + resimulate rollback netcode around
/// [`step_marbles`]. See the module doc for the design and the correctness
/// property this exists to guarantee.
pub struct RollbackSim {
    /// Live marble state — always exactly `step_marbles`'s output after
    /// simulating through `current_tick` (whether that simulation happened
    /// via [`Self::advance`] or a resimulation triggered by
    /// [`Self::receive_inputs`]).
    marbles: Vec<Marble>,
    current_tick: Tick,
    /// `snapshots[t]` = marble state immediately *after* tick `t` was
    /// simulated (`snapshots[0]` = the initial state, before any tick).
    /// Pruned to the most recent `window + 1` entries (`+1` since rewinding
    /// *to* tick `t` needs `snapshots[t-1]` as the resim starting point, so
    /// the oldest tick a rewind can target is `current_tick - window`,
    /// which itself needs `current_tick - window - 1` still present).
    snapshots: BTreeMap<Tick, Vec<Marble>>,
    /// `inputs[player][t]` = the `(PlayerInput, InputStatus)` actually used
    /// to simulate tick `t` for that player, for every tick still within
    /// the rewind window. Pruned alongside `snapshots`.
    inputs: Vec<BTreeMap<Tick, (PlayerInput, InputStatus)>>,
    /// Used as a player's predicted input for any tick at or before which
    /// they have no confirmed input at all yet (session startup only, in
    /// practice).
    default_input: PlayerInput,
    /// How many past ticks a rewind can reach back into. Exceeding this
    /// (a confirmed input arrives for a tick that's already been pruned)
    /// is a hard error ([`Self::receive_inputs`]'s doc) rather than a
    /// silent best-effort clamp — a clamp would produce a state that's
    /// *not* what full-knowledge replay would have produced, which is
    /// exactly the property this whole module exists to guarantee, so
    /// silently violating it would be worse than failing loudly.
    window: u64,
    /// Checksums of every tick that has become eligible for cross-peer
    /// comparison (`tick <= current_tick - window` — the same boundary
    /// [`Self::oldest_available`] computes for [`Self::receive_inputs`]),
    /// cached the instant each one is reached rather than recomputed on
    /// demand: `snapshots` itself is pruned to `window + 1` entries, far
    /// tighter than a consumer's own checksum-exchange interval is likely
    /// to be, so the snapshot a given eligible tick needs may already be
    /// gone by the time something asks for its checksum. Retention here is
    /// independent of `window` (see [`CHECKSUM_CACHE_TICKS`]).
    checksum_cache: BTreeMap<Tick, u64>,
}

/// How many past eligible-tick checksums [`RollbackSim`] retains, kept
/// generously larger than `window` on purpose: unlike `snapshots` (which
/// only needs to outlive a rewind), this only needs to outlive however
/// long a caller's own periodic cross-peer checksum-exchange interval
/// turns out to be, a number this module has no reason to know (that's
/// `physics_sys.rs`'s `CHECKSUM_INTERVAL_TICKS`, one layer up) — 300 ticks
/// (5 seconds at this app's 60Hz) comfortably outlives a "once a second"
/// starting-point cadence with room to spare.
const CHECKSUM_CACHE_TICKS: u64 = 300;

/// A confirmed input arrived for a tick this [`RollbackSim`] can no longer
/// rewind to (older than `current_tick - window`) — the caller's rollback
/// window is too shallow for the network conditions it's actually seeing.
/// Not a bug in this module; a configuration/network-conditions problem the
/// caller needs to know about rather than have silently papered over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowExceeded {
    pub requested_tick: Tick,
    pub oldest_available: Tick,
}

/// A cheap, deterministic hash of a whole simulation's marble state —
/// position and velocity only (`rad`/`last_thrust` never change as a
/// function of simulated ticks the way `pos`/`vel` do, so they carry no
/// information relevant to detecting a diverged simulation). Never
/// compares or hashes a raw `f32` directly — `to_bits()` first, always —
/// since this function's entire purpose is catching subtle bit-level
/// simulation divergence between two peers, which is exactly the kind of
/// bug floating-point comparison quirks (`NaN`, signed zero) could
/// otherwise mask or manufacture. Folded through FNV-1a: simple, fast,
/// good-enough avalanche for a lockstep desync check (this is a state
/// fingerprint to compare across two peers who are supposed to agree
/// exactly, not a cryptographic or collision-resistant hash), and — same
/// reasoning as `net.rs`'s hand-rolled wire codecs — one small, auditable
/// function is preferable here to a hashing crate dependency for this one
/// use.
pub fn state_checksum(marbles: &[Marble]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut hash = FNV_OFFSET_BASIS;
    for marble in marbles {
        for component in [marble.pos.x, marble.pos.y, marble.pos.z, marble.vel.x, marble.vel.y, marble.vel.z] {
            for byte in component.to_bits().to_le_bytes() {
                hash = (hash ^ byte as u64).wrapping_mul(FNV_PRIME);
            }
        }
    }
    hash
}

impl RollbackSim {
    /// `initial_marbles` is tick 0's state (before any input is applied —
    /// typically each marble's spawn point). `num_players` must match
    /// `initial_marbles.len()` (asserted) — same "one slot per player"
    /// convention `step_marbles` itself uses. `default_input` is the
    /// fallback prediction for a player with no confirmed history yet.
    pub fn new(initial_marbles: Vec<Marble>, default_input: PlayerInput, window: u64) -> Self {
        let num_players = initial_marbles.len();
        let mut snapshots = BTreeMap::new();
        snapshots.insert(0, initial_marbles.clone());
        Self {
            marbles: initial_marbles,
            current_tick: 0,
            snapshots,
            inputs: vec![BTreeMap::new(); num_players],
            default_input,
            window,
            checksum_cache: BTreeMap::new(),
        }
    }

    pub fn current_tick(&self) -> Tick {
        self.current_tick
    }

    /// The live marble state, after simulating through `current_tick`.
    pub fn marbles(&self) -> &[Marble] {
        &self.marbles
    }

    /// Predicts player `player`'s input for `tick`: their most recent
    /// *confirmed* input at or before `tick`, or `default_input` if none
    /// exists yet. See the module doc for why this only ever searches
    /// confirmed entries, never previously-predicted ones.
    fn predict(&self, player: PlayerIndex, tick: Tick) -> PlayerInput {
        self.inputs[player]
            .range(..=tick)
            .rev()
            .find(|(_, (_, status))| *status == InputStatus::Confirmed)
            .map(|(_, (input, _))| *input)
            .unwrap_or(self.default_input)
    }

    /// Builds the `PlayerInput` slice to feed `step_marbles` for `tick`:
    /// each player's confirmed input if the log already has one, else a
    /// freshly-computed prediction (recorded into the log as `Predicted`,
    /// overwriting any stale prediction already there — see the module doc
    /// on why a fresh prediction, not a cached one, matters for
    /// resimulation correctness).
    fn build_inputs_for_tick(&mut self, tick: Tick) -> Vec<PlayerInput> {
        (0..self.inputs.len())
            .map(|player| {
                if let Some(&(input, InputStatus::Confirmed)) = self.inputs[player].get(&tick) {
                    input
                } else {
                    let predicted = self.predict(player, tick);
                    self.inputs[player].insert(tick, (predicted, InputStatus::Predicted));
                    predicted
                }
            })
            .collect()
    }

    /// Simulates exactly one more tick (`current_tick + 1`), predicting any
    /// player's input that hasn't been confirmed for that tick yet.
    /// `animations` is evaluated against *this* tick and written into
    /// `params` before `step_marbles` runs (`expr::apply_animations`'s
    /// doc — this is the live half of why animated geometry stays in the
    /// same deterministic domain as marble state; [`Self::resim_from`] is
    /// the replayed half). Returns that tick's [`StepEvent`]s (same
    /// meaning as `step_marbles`'s own return — respawns, etc.).
    #[allow(clippy::too_many_arguments)]
    pub fn advance(
        &mut self,
        obj: &Object,
        params: &mut Params,
        animations: &[(crate::ScalarParam, Expr)],
        cfg: &PhysicsConfig,
        kill_y: f32,
        starts: &[Vec3],
    ) -> Vec<StepEvent> {
        let tick = self.current_tick + 1;
        apply_animations(params, animations, tick);
        let inputs = self.build_inputs_for_tick(tick);
        let events = step_marbles(&mut self.marbles, &inputs, obj, params, cfg, kill_y, starts);
        self.snapshots.insert(tick, self.marbles.clone());
        self.current_tick = tick;
        self.prune();
        self.cache_newly_eligible_checksum();
        events
    }

    /// Records a batch of newly-confirmed `(player, tick, input)` arrivals,
    /// and — if any of them changes what was *actually simulated* for a
    /// tick at or before `current_tick` (i.e. a prediction turned out
    /// wrong) — rewinds to just before the earliest such tick and
    /// resimulates forward to `current_tick`, so the live state stays
    /// exactly what full-knowledge replay would have produced.
    ///
    /// Arrivals for ticks *after* `current_tick` just get recorded for
    /// [`Self::advance`] to pick up naturally later — no rewind needed,
    /// nothing's been simulated for them yet. Order within `arrivals`
    /// doesn't matter (the whole batch is applied to the log before any
    /// resimulation happens — see the module doc on why that matters for
    /// batches spanning multiple ticks/players).
    ///
    /// # Errors
    /// [`WindowExceeded`] if a confirmed arrival is for a tick this
    /// instance can no longer rewind to (see `window`'s doc) — the whole
    /// batch is still recorded (so a *later*, in-window correction for the
    /// same tick range still works correctly), but no resimulation happens
    /// for the too-old tick, so the caller must treat the live state as
    /// potentially wrong from that point and handle it (a real
    /// implementation would need a full resync from a peer here, not
    /// something this milestone's local-only scope needs to solve).
    #[allow(clippy::too_many_arguments)]
    pub fn receive_inputs(
        &mut self,
        arrivals: &[(PlayerIndex, Tick, PlayerInput)],
        obj: &Object,
        params: &mut Params,
        animations: &[(crate::ScalarParam, Expr)],
        cfg: &PhysicsConfig,
        kill_y: f32,
        starts: &[Vec3],
    ) -> Result<(), WindowExceeded> {
        let oldest_available = self.oldest_available();
        let mut earliest_mismatch: Option<Tick> = None;
        let mut window_exceeded: Option<WindowExceeded> = None;

        for &(player, tick, input) in arrivals {
            let previous = self.inputs[player].get(&tick).copied();
            self.inputs[player].insert(tick, (input, InputStatus::Confirmed));

            if tick > self.current_tick {
                continue; // not simulated yet -- `advance` will pick this up naturally.
            }
            if tick < oldest_available {
                window_exceeded.get_or_insert(WindowExceeded {
                    requested_tick: tick,
                    oldest_available,
                });
                continue;
            }
            // Mismatch iff the tick was already simulated with a
            // *different* value than what just arrived. A `previous` of
            // `None` at a tick `<= current_tick` shouldn't happen in
            // practice (`build_inputs_for_tick` always records something,
            // confirmed or predicted, for every simulated tick) but is
            // treated as a mismatch defensively rather than silently
            // trusting an untracked tick.
            let differs = match previous {
                Some((prev_input, _)) => prev_input != input,
                None => true,
            };
            if differs {
                earliest_mismatch = Some(earliest_mismatch.map_or(tick, |e| e.min(tick)));
            }
        }

        if let Some(from_tick) = earliest_mismatch {
            self.resim_from(from_tick, obj, params, animations, cfg, kill_y, starts);
        }
        self.prune();

        match window_exceeded {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Restores the snapshot from just before `from_tick` and resimulates
    /// forward through `current_tick`, rebuilding every intermediate
    /// snapshot along the way (not just the final state) — a *later*
    /// correction for some tick in the middle of this range still needs a
    /// valid snapshot to rewind to.
    ///
    /// Re-evaluates `animations` at *each* replayed tick (`expr`'s module
    /// doc) rather than leaving `params` at whatever it held going in —
    /// this is the part that actually makes animated geometry safe to
    /// resimulate: a historical tick being replayed sees that tick's own
    /// animation phase, the same value it would have seen the first time
    /// it was (or would have been) simulated, not the *current* phase.
    /// Skipping this would silently reintroduce a desync risk one layer
    /// below marble state, exactly the class of bug this whole module
    /// exists to rule out.
    #[allow(clippy::too_many_arguments)]
    fn resim_from(
        &mut self,
        from_tick: Tick,
        obj: &Object,
        params: &mut Params,
        animations: &[(crate::ScalarParam, Expr)],
        cfg: &PhysicsConfig,
        kill_y: f32,
        starts: &[Vec3],
    ) {
        debug_assert!(from_tick >= 1, "tick 0 has no prior tick to have mispredicted");
        let base = from_tick - 1;
        let mut marbles = self
            .snapshots
            .get(&base)
            .cloned()
            .expect("resim_from called with a tick outside the retained window");

        for tick in from_tick..=self.current_tick {
            apply_animations(params, animations, tick);
            let inputs = self.build_inputs_for_tick(tick);
            step_marbles(&mut marbles, &inputs, obj, params, cfg, kill_y, starts);
            self.snapshots.insert(tick, marbles.clone());
        }
        self.marbles = marbles;
    }

    /// Drops snapshot/input-log entries older than the rewind window can
    /// reach — keeps this module's memory bounded regardless of session
    /// length, matching the "ring buffer" framing even though the backing
    /// structure is a `BTreeMap` (module doc).
    ///
    /// Uses `window + 1`, not `window`: the oldest tick a correction can
    /// still target is `current_tick - window` (`receive_inputs`'s
    /// `oldest_available`), and resimulating *that* tick needs
    /// `snapshots[current_tick - window - 1]` as its starting point — one
    /// tick further back than `window` alone would keep. Missing this `+1`
    /// is a real bug this module's own test suite caught outright (a panic
    /// in `resim_from`, not a silent wrong answer) the first time a
    /// correction landed right at the edge of the window.
    fn prune(&mut self) {
        let oldest_to_keep = self.current_tick.saturating_sub(self.window + 1);
        self.snapshots.retain(|&t, _| t >= oldest_to_keep);
        for log in &mut self.inputs {
            log.retain(|&t, _| t >= oldest_to_keep);
        }
    }

    /// The oldest tick still safe to rewind/correct — anything before this
    /// has already been pruned. Shared by [`Self::receive_inputs`] (a
    /// confirmed arrival older than this is [`WindowExceeded`]) and, for
    /// exactly the same underlying reason, by checksum eligibility
    /// ([`Self::latest_checksum_tick`]): a tick still inside the window
    /// could yet be resimulated by a late-but-in-window correction, so
    /// comparing its checksum across peers would produce a false-positive
    /// "mismatch" from perfectly ordinary prediction-then-correction, not
    /// real divergence. One computation, reused for both purposes rather
    /// than risking two independently-written copies drifting apart.
    fn oldest_available(&self) -> Tick {
        self.current_tick.saturating_sub(self.window)
    }

    /// Computes and caches the checksum for whichever tick just became
    /// eligible for cross-peer comparison (`current_tick - window`) —
    /// called only from [`Self::advance`], the sole place `current_tick`
    /// ever moves forward, so the tick this caches is always freshly and
    /// *permanently* finalized: nothing can ever resimulate a tick at or
    /// before `current_tick - window` again (a late arrival for one would
    /// report [`WindowExceeded`] instead), so the value cached here is
    /// exactly the final, fully-resolved state's checksum — never a stale
    /// pre-correction one, even if a resim touched this exact tick on its
    /// way to becoming eligible.
    ///
    /// Deliberately uses `checked_sub`, not [`Self::oldest_available`]'s
    /// saturating version: at `current_tick < window` nothing has become
    /// newly eligible yet (tick 0 shouldn't be (re-)cached on every one of
    /// the first `window` ticks), whereas `receive_inputs`'s
    /// `oldest_available` genuinely wants a saturating floor of `0` so
    /// "anything before tick 0" is correctly never rejected as too old.
    ///
    /// The snapshot for `tick` can legitimately be missing, not just at
    /// startup but also for the first `window` ticks right after
    /// [`Self::hard_reset_to`]: that call reseeds history starting fresh
    /// at its own `tick`, so `current_tick - window` can point *before*
    /// that reseed's own genesis until enough ticks have passed since —
    /// there is nothing wrong to report here, simply nothing eligible to
    /// cache yet from this reseeded epoch.
    fn cache_newly_eligible_checksum(&mut self) {
        let Some(tick) = self.current_tick.checked_sub(self.window) else { return };
        let Some(snapshot) = self.snapshots.get(&tick) else { return };
        self.checksum_cache.insert(tick, state_checksum(snapshot));
        let oldest_to_keep = tick.saturating_sub(CHECKSUM_CACHE_TICKS);
        self.checksum_cache.retain(|&t, _| t >= oldest_to_keep);
    }

    /// The most recent tick currently eligible for cross-peer checksum
    /// comparison (`current_tick - window`), or `None` if no tick has
    /// crossed that boundary yet (session just started). A caller
    /// periodically sending its own checksum should send this tick's —
    /// see [`Self::checksum_at`] to get the actual hash.
    pub fn latest_checksum_tick(&self) -> Option<Tick> {
        self.current_tick.checked_sub(self.window)
    }

    /// The cached checksum for `tick`, if it's both eligible (was, at some
    /// point, outside the rewind window) and still within the retained
    /// checksum cache. `None` for a tick that's still in-window (comparing
    /// it would be a false positive — see [`Self::oldest_available`]'s
    /// doc), or one old enough to have been evicted from the cache
    /// ([`CHECKSUM_CACHE_TICKS`]) — a caller should treat either case as
    /// "can't compare, not a mismatch", not guess.
    pub fn checksum_at(&self, tick: Tick) -> Option<u64> {
        self.checksum_cache.get(&tick).copied()
    }

    /// Replaces this session's entire live state with `marbles` — the
    /// authoritative correction after a detected checksum mismatch
    /// (`net.rs`'s `ResyncPayload`) — and reseeds the snapshot/prediction
    /// window fresh starting at `tick`, discarding every confirmed input
    /// and snapshot this side had before it. A clean restart of the window
    /// is the correct move here, not an attempt to reconcile old history:
    /// once a mismatch is confirmed, nothing this side thought it knew
    /// about any tick up to and including `tick` can be trusted anyway
    /// (that's the entire premise of the mismatch), including its own
    /// logged "confirmed" inputs — keeping those around would risk a
    /// resim immediately replaying right back into the same divergence
    /// this call exists to fix, if the divergence's root cause turns out
    /// to be in this side's simulation logic rather than in what input it
    /// thought it had received.
    pub fn hard_reset_to(&mut self, tick: Tick, marbles: Vec<Marble>) {
        assert_eq!(
            marbles.len(),
            self.marbles.len(),
            "a resync must not change how many players this session has"
        );
        self.current_tick = tick;
        self.snapshots.clear();
        self.snapshots.insert(tick, marbles.clone());
        self.marbles = marbles;
        for log in &mut self.inputs {
            log.clear();
        }
        self.checksum_cache.clear();
    }

    pub fn num_players(&self) -> usize {
        self.marbles.len()
    }

    /// Grows this session from N players to N+1 *in place*, inserting the
    /// new player at `index` (any existing players at `index..` shift up by
    /// one) — the always-on-from-scene-load design's answer to "a peer just
    /// connected": single-player is just a session with one confirmed local
    /// player and no remote peers ever arriving, so gaining a second real
    /// player is *growing* that same session, not discarding it and
    /// building a fresh 2-player one — `current_tick`, every already-live
    /// player's simulated state, and their confirmed input history all
    /// carry through completely unbroken (no snapshot/rewind discontinuity,
    /// and critically for [`crate::expr`]-driven animated params, no reset
    /// of the tick an animation is a function of either — the fractal
    /// doesn't visibly jump/restart the moment someone joins).
    ///
    /// `index` is always `0` or `1` in this app (`net::Role`'s doc: host is
    /// always player 0, joiner always player 1, fixed for the session) —
    /// the host inserts the joiner at `1` (after its own existing player
    /// 0), the joiner inserts the host at `0` (before its own existing
    /// player, which shifts from index 0 to 1). Both are the same
    /// operation from this module's point of view, just a different
    /// insertion point, which is why one method handles both instead of
    /// separate "append"/"prepend" ones.
    ///
    /// `spawn_marble` becomes the new player's live marble immediately, and
    /// — since every currently-retained snapshot in the rewind window
    /// necessarily predates them — also their placeholder state in each of
    /// those historical snapshots too. There's no way to know what their
    /// marble "would have been" before they existed; a fixed spawn point is
    /// the same kind of stand-in [`Self::default_input`](struct.RollbackSim.html)
    /// already is for input prediction with no confirmed history. If a
    /// later correction ever triggers a resim reaching back into one of
    /// those pre-join ticks (possible: a *pre-existing* player's own late
    /// input correction can target any in-window tick, join or no join),
    /// the new player simply has no confirmed input there either, so
    /// [`Self::predict`] falls back to `default_input` for them exactly as
    /// it would for any other tick with no confirmed history — consistent,
    /// not a special case.
    pub fn add_player_at(&mut self, index: PlayerIndex, spawn_marble: Marble) -> PlayerIndex {
        assert!(index <= self.marbles.len(), "index out of range for insertion");
        self.marbles.insert(index, spawn_marble);
        self.inputs.insert(index, BTreeMap::new());
        for marbles in self.snapshots.values_mut() {
            marbles.insert(index, spawn_marble);
        }
        index
    }

    /// The reverse of growing: narrows this session down to just `keep`'s
    /// own live state (relabeled to player 0), discarding every other
    /// player. Not used for a real remote player leaving (this milestone
    /// has no graceful-disconnect story — `physics_sys.rs`'s doc) — its one
    /// caller is the rare edge case of a solo session that already had more
    /// than one marble for reasons unrelated to real players (`Demo`'s
    /// decorative extra marbles, spawned purely for a single-player
    /// marble-collision visual demo, `render::setup`'s doc) gaining its
    /// first real player: those decorative extras get dropped at that point
    /// — same as this app already did before this session became always-on
    /// (they were never networked to begin with) — while still preserving
    /// `current_tick` and the real local player's own live state and input
    /// history, the same continuity [`Self::add_player_at`] guarantees for
    /// the common (non-`Demo`) case.
    pub fn narrow_to(&self, keep: PlayerIndex) -> Self {
        let mut snapshots = BTreeMap::new();
        for (&t, marbles) in &self.snapshots {
            snapshots.insert(t, vec![marbles[keep]]);
        }
        Self {
            marbles: vec![self.marbles[keep]],
            current_tick: self.current_tick,
            snapshots,
            inputs: vec![self.inputs[keep].clone()],
            default_input: self.default_input,
            window: self.window,
            // Not carried over: every cached entry was computed over the
            // *old*, wider marble set, a different shape than anything
            // this narrowed session will ever checksum again. Harmless
            // either way in practice (narrowing only ever happens solo,
            // strictly before a peer connects and checksum exchange could
            // start — `physics_sys.rs`'s doc), but starting empty is more
            // obviously correct than carrying over checksums for a shape
            // this session no longer has.
            checksum_cache: BTreeMap::new(),
        }
    }

    /// Out-of-band position/velocity reset (crush/kill-plane manual `R`
    /// respawn — `physics_sys.rs`'s doc on why this is only ever safe to
    /// call while `player` is the *only* player in this session: unlike
    /// every other mutation in this module, a respawn isn't a function of
    /// any `PlayerInput`, so there's no way for [`Self::resim_from`] to
    /// replay it — unreachable in practice because it's only ever called
    /// while solo, and a solo session with only one player's own always-
    /// immediately-confirmed input never triggers a resim at all (nothing
    /// to mismatch against), so there's no resim left to silently "undo"
    /// this by re-deriving `marbles[player]` from a stale snapshot.
    pub fn respawn(&mut self, player: PlayerIndex, start: Vec3) {
        self.marbles[player].respawn(start);
    }

    /// Debug/test-only: nudges `player`'s live position in place, entirely
    /// outside any `PlayerInput`-driven simulation step — lets a manual or
    /// CDP-driven verification pass deliberately force one peer's state to
    /// diverge from the other's, to check that checksum comparison
    /// actually detects it and [`Self::hard_reset_to`] actually recovers
    /// it (this is the "forced-mismatch scenario" this feature's own
    /// verification plan calls for; there's no other way to safely
    /// manufacture a real cross-peer divergence on demand). `cfg`-gated
    /// out of release builds entirely — this must never be reachable from
    /// any real gameplay path, and this way it simply isn't compiled into
    /// the shipped wasm at all (`cargo build --release` disables
    /// `debug_assertions`).
    #[cfg(debug_assertions)]
    pub fn debug_perturb_position(&mut self, player: PlayerIndex, delta: Vec3) {
        self.marbles[player].pos += delta;
    }
}

/// A transport for exchanging per-tick [`PlayerInput`]s with other clients
/// — deliberately minimal so [`RollbackSim`] never needs to change when a
/// real implementation (milestone 2: WebRTC) replaces a local one. Message
/// delivery order/timing is entirely up to the implementation;
/// [`RollbackSim::receive_inputs`] is designed to accept an arbitrarily
/// delayed/reordered/batched set of arrivals correctly (module doc).
pub trait InputTransport {
    fn send_input(&mut self, tick: Tick, input: PlayerInput);
    /// Drains and returns every input received since the last call to this
    /// method, as `(player, tick, input)` triples. No ordering guarantee.
    fn poll_received(&mut self) -> Vec<(PlayerIndex, Tick, PlayerInput)>;
}

#[cfg(test)]
pub mod test_support {
    //! Pure-Rust transport test doubles — not gated behind `#[cfg(test)]`
    //! at the module level (just this submodule) so an app-side integration
    //! test in a different crate could reuse them too if that ever becomes
    //! useful, without duplicating this logic.
    use super::*;
    use std::collections::VecDeque;

    /// A same-process, one-way input channel: `sender_index` is which
    /// player slot everything sent through this end is attributed to on
    /// the receiving side. Two of these, cross-wired, model a genuine
    /// 2-client exchange without any real networking — each client owns
    /// its own [`RollbackSim`] and only ever learns about the other
    /// player's input through `poll_received`, exactly as a real transport
    /// would deliver it.
    pub struct InMemoryTransport {
        sender_index: PlayerIndex,
        /// Messages this end has sent, queued for the paired receiver to
        /// pick up via [`Self::deliver_into`] — deliberately *not* wired
        /// directly to the peer's inbox, so a test can insert artificial
        /// delay/reorder/duplication between "sent" and "received" (see
        /// [`JitteredLink`]).
        outbox: VecDeque<(Tick, PlayerInput)>,
        inbox: VecDeque<(PlayerIndex, Tick, PlayerInput)>,
    }

    impl InMemoryTransport {
        pub fn new(sender_index: PlayerIndex) -> Self {
            Self {
                sender_index,
                outbox: VecDeque::new(),
                inbox: VecDeque::new(),
            }
        }

        /// Test-only hook: injects a message directly into this
        /// transport's inbox, as if it had just arrived from the network
        /// (used by [`JitteredLink`] to control exactly when/in-what-order
        /// a peer's sent messages actually show up).
        pub fn deliver(&mut self, player: PlayerIndex, tick: Tick, input: PlayerInput) {
            self.inbox.push_back((player, tick, input));
        }

        /// Test-only hook: drains this transport's outbox (what it's tried
        /// to send since the last drain), tagged with its own
        /// `sender_index` — the other half of the manual "deliver into the
        /// paired transport, possibly with jitter" wiring `JitteredLink`
        /// does.
        pub fn drain_sent(&mut self) -> Vec<(PlayerIndex, Tick, PlayerInput)> {
            self.outbox
                .drain(..)
                .map(|(tick, input)| (self.sender_index, tick, input))
                .collect()
        }
    }

    impl InputTransport for InMemoryTransport {
        fn send_input(&mut self, tick: Tick, input: PlayerInput) {
            self.outbox.push_back((tick, input));
        }

        fn poll_received(&mut self) -> Vec<(PlayerIndex, Tick, PlayerInput)> {
            self.inbox.drain(..).collect()
        }
    }

    /// Deliberately delays every message sent through it by a fixed number
    /// of ticks (simulating network latency) and reverses the order within
    /// each tick's batch (simulating out-of-order arrival) before handing
    /// them to [`InMemoryTransport::deliver`] — a real network can do far
    /// stranger things than this, but "constant delay + local reordering"
    /// is already enough to force real rewinds in a test, and adversarial
    /// tests build further scenarios (bursts, multi-player simultaneous
    /// corrections) directly on top of `InMemoryTransport::deliver` rather
    /// than needing this to model everything.
    /// One delayed batch: `deliver_at` (the tick it should be released) plus
    /// the arrivals it carries.
    type DelayedBatch = (Tick, Vec<(PlayerIndex, Tick, PlayerInput)>);

    pub struct JitteredLink {
        delay_ticks: Tick,
        pending: VecDeque<DelayedBatch>,
    }

    impl JitteredLink {
        pub fn new(delay_ticks: Tick) -> Self {
            Self {
                delay_ticks,
                pending: VecDeque::new(),
            }
        }

        /// Call once per simulated tick: queues `sent` (this tick's
        /// freshly-sent messages) for delayed delivery, then delivers
        /// anything whose delay has elapsed into `into`.
        pub fn step(
            &mut self,
            now: Tick,
            sent: Vec<(PlayerIndex, Tick, PlayerInput)>,
            into: &mut InMemoryTransport,
        ) {
            if !sent.is_empty() {
                self.pending.push_back((now + self.delay_ticks, sent));
            }
            while let Some((deliver_at, _)) = self.pending.front() {
                if *deliver_at > now {
                    break;
                }
                let (_, mut batch) = self.pending.pop_front().unwrap();
                batch.reverse(); // local reordering within the delayed batch
                for (player, tick, input) in batch {
                    into.deliver(player, tick, input);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenes::{demo_scene, set_fractal_params};
    use crate::{physics::PhysicsConfig, scenes::beware_of_bumps, ScalarParam, ScalarValue};
    use glam::Quat;
    use test_support::{InMemoryTransport, JitteredLink};

    fn setup() -> (Object, Params) {
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

    fn input(dx: f32, dy: f32) -> PlayerInput {
        PlayerInput { dx, dy, orientation: Quat::IDENTITY }
    }

    fn two_player_setup() -> (Object, Params, Vec<Marble>, Vec<Vec3>) {
        let (object, params) = setup();
        let rad = beware_of_bumps::MARBLE_RAD;
        let starts = vec![
            beware_of_bumps::START,
            beware_of_bumps::START + Vec3::new(0.0, 0.3, 0.0),
        ];
        let marbles = starts.iter().map(|s| Marble::spawn(*s, rad)).collect();
        (object, params, marbles, starts)
    }

    /// A fixed, deterministic per-tick input for a player, so straight-
    /// through and rollback runs feed the exact same logical input
    /// sequence and only differ in *when* the rollback run learns about
    /// it.
    fn scripted_input(player: usize, tick: Tick) -> PlayerInput {
        // Arbitrary but deterministic function of (player, tick) so
        // different players and ticks get different (still small, in-range)
        // inputs -- exercises more than just "always press W".
        let t = tick as f32;
        input(((player as f32 * 0.37 + t * 0.05).sin()) * 0.8, ((t * 0.031).cos()) * 0.8)
    }

    /// The core correctness property this whole module exists for: running
    /// a fixed input sequence straight through with every input known up
    /// front must produce *exactly* the same final state as running the
    /// same sequence through `RollbackSim` with those same inputs
    /// deliberately delayed (so most ticks get simulated on a *predicted*
    /// value first, then corrected once the real one "arrives" a few ticks
    /// later).
    #[test]
    fn rollback_replay_matches_full_knowledge_replay_under_constant_delay() {
        let (object, mut params, marbles, starts) = two_player_setup();
        let cfg = PhysicsConfig::default();
        const TICKS: Tick = 200;
        const DELAY: Tick = 4;

        // Straight-through: every input known before its tick is simulated.
        let mut straight = marbles.clone();
        for tick in 1..=TICKS {
            let inputs = vec![scripted_input(0, tick), scripted_input(1, tick)];
            step_marbles(&mut straight, &inputs, &object, &params, &cfg, -1e6, &starts);
        }

        // Rollback: player 0's real input for tick T only "arrives"
        // (via receive_inputs) DELAY ticks late -- every tick in between
        // gets simulated on a *predicted* value first.
        let mut sim = RollbackSim::new(marbles, input(0.0, 0.0), 16);
        for tick in 1..=TICKS {
            // Player 1's own input is "confirmed" the instant it's needed
            // (as if it were the local player) -- only player 0 is
            // deliberately delayed, so every tick still exercises a real
            // misprediction-and-correction, not just "everything's late".
            sim.receive_inputs(
                &[(1, tick, scripted_input(1, tick))],
                &object,
                &mut params,
                &[],
                &cfg,
                -1e6,
                &starts,
            )
            .expect("within window");
            sim.advance(&object, &mut params, &[], &cfg, -1e6, &starts);
            if tick >= DELAY {
                let confirm_tick = tick - DELAY + 1;
                sim.receive_inputs(
                    &[(0, confirm_tick, scripted_input(0, confirm_tick))],
                    &object,
                    &mut params,
                    &[],
                    &cfg,
                    -1e6,
                    &starts,
                )
                .expect("within window");
            }
        }
        // Flush the remaining in-flight delayed confirmations for player 0.
        for confirm_tick in (TICKS - DELAY + 2)..=TICKS {
            sim.receive_inputs(
                &[(0, confirm_tick, scripted_input(0, confirm_tick))],
                &object,
                &mut params,
                &[],
                &cfg,
                -1e6,
                &starts,
            )
            .expect("within window");
        }

        for (i, (a, b)) in straight.iter().zip(sim.marbles().iter()).enumerate() {
            assert_eq!(
                a.pos, b.pos,
                "marble {i} position diverged: straight-through {:?} vs rollback {:?}",
                a.pos, b.pos
            );
            assert_eq!(
                a.vel, b.vel,
                "marble {i} velocity diverged: straight-through {:?} vs rollback {:?}",
                a.vel, b.vel
            );
        }
    }

    /// A minimal scene with exactly one animated geometry parameter --
    /// deliberately not reusing `beware_of_bumps`/`demo_scene` (whose
    /// complex fractal folds make it hard to reason about *which* tick's
    /// animation value would actually be visible in the final marble
    /// state) so this test's only variable is "did resimulation use the
    /// right tick's animation value", not "did some unrelated fractal fold
    /// do something surprising". A single animated sphere a marble rests
    /// directly on top of means the marble's exact position is sensitive
    /// to the sphere's exact radius on essentially every tick -- if a
    /// resim ever evaluated an animation against the wrong tick, this
    /// would show up as an immediate, large position divergence, not a
    /// subtle one.
    #[allow(clippy::type_complexity)]
    fn animated_sphere_setup() -> (Object, Params, Vec<(ScalarParam, Expr)>, Vec<Marble>, Vec<Vec3>)
    {
        let mut params = Params::new();
        let radius = params.alloc_scalar(2.0);
        let object = Object::Sphere { radius: ScalarValue::Param(radius) };
        // Oscillates between 0.5 and 3.5 -- a wide enough swing, on a short
        // enough period, that a marble resting on the surface moves
        // measurably from the animation alone, not just from gravity
        // settling.
        let anim = Expr::Add(
            Box::new(Expr::Const(2.0)),
            Box::new(Expr::Mul(
                Box::new(Expr::Const(1.5)),
                Box::new(Expr::Sin(Box::new(Expr::Mul(
                    Box::new(Expr::Tick),
                    Box::new(Expr::Const(0.13)),
                )))),
            )),
        );
        let animations = vec![(radius, anim)];

        let rad = beware_of_bumps::MARBLE_RAD;
        let starts = vec![Vec3::new(0.0, 2.0 + rad, 0.0), Vec3::new(1.0, 2.0 + rad, 0.0)];
        let marbles = starts.iter().map(|s| Marble::spawn(*s, rad)).collect();
        (object, params, animations, marbles, starts)
    }

    /// The resimulation-correctness property animated params add on top of
    /// [`rollback_replay_matches_full_knowledge_replay_under_constant_delay`]'s:
    /// a resim doesn't just need to replay the right *inputs* per historical
    /// tick, it needs to replay the right *animation value* too --
    /// `RollbackSim::resim_from` re-evaluates `animations` at each tick it
    /// replays rather than leaving `Params` at whatever the animation last
    /// wrote (see its doc comment). If that ever regressed to "resim with
    /// today's animation value instead of tick T's", this test would fail:
    /// the delayed-confirmation path forces every tick to be simulated once
    /// on a prediction and then resimulated after the real input "arrives",
    /// so a resim that gets the animated radius wrong would diverge from
    /// the straight-through run almost immediately, not eventually.
    #[test]
    fn rollback_replay_with_an_animated_param_matches_full_knowledge_replay() {
        let (object, mut params, animations, marbles, starts) = animated_sphere_setup();
        let cfg = PhysicsConfig::default();
        const TICKS: Tick = 200;
        const DELAY: Tick = 4;

        // Straight-through: every input known before its tick is simulated,
        // and the animated radius is (re-)applied fresh each tick too.
        let mut straight_params = params.clone();
        let mut straight = marbles.clone();
        for tick in 1..=TICKS {
            apply_animations(&mut straight_params, &animations, tick);
            let inputs = vec![scripted_input(0, tick), scripted_input(1, tick)];
            step_marbles(&mut straight, &inputs, &object, &straight_params, &cfg, -1e6, &starts);
        }

        // Rollback: player 0's real input for tick T only "arrives" DELAY
        // ticks late, forcing every tick in between to be simulated on a
        // predicted value first and then resimulated once corrected --
        // exactly the same shape as the plain-input test above, but now
        // every `advance`/`receive_inputs` call also carries the animation
        // table, so each resimulated tick re-evaluates the radius too.
        let mut sim = RollbackSim::new(marbles, input(0.0, 0.0), 16);
        for tick in 1..=TICKS {
            sim.receive_inputs(
                &[(1, tick, scripted_input(1, tick))],
                &object,
                &mut params,
                &animations,
                &cfg,
                -1e6,
                &starts,
            )
            .expect("within window");
            sim.advance(&object, &mut params, &animations, &cfg, -1e6, &starts);
            if tick >= DELAY {
                let confirm_tick = tick - DELAY + 1;
                sim.receive_inputs(
                    &[(0, confirm_tick, scripted_input(0, confirm_tick))],
                    &object,
                    &mut params,
                    &animations,
                    &cfg,
                    -1e6,
                    &starts,
                )
                .expect("within window");
            }
        }
        for confirm_tick in (TICKS - DELAY + 2)..=TICKS {
            sim.receive_inputs(
                &[(0, confirm_tick, scripted_input(0, confirm_tick))],
                &object,
                &mut params,
                &animations,
                &cfg,
                -1e6,
                &starts,
            )
            .expect("within window");
        }

        for (i, (a, b)) in straight.iter().zip(sim.marbles().iter()).enumerate() {
            assert_eq!(
                a.pos, b.pos,
                "marble {i} position diverged: straight-through {:?} vs rollback {:?}",
                a.pos, b.pos
            );
            assert_eq!(
                a.vel, b.vel,
                "marble {i} velocity diverged: straight-through {:?} vs rollback {:?}",
                a.vel, b.vel
            );
        }
    }

    /// Same core property, but exercised through the actual
    /// [`InputTransport`] trait boundary and [`test_support::JitteredLink`]
    /// (constant delay + local reorder) rather than the test hand-rolling
    /// delayed `receive_inputs` calls itself — two fully independent
    /// `RollbackSim`s, each only knowing its own player's input a priori,
    /// exchanging the other's input purely through the trait, must still
    /// converge to identical final states.
    ///
    /// The comparison happens only *after* a drain phase that stops
    /// sending new ticks' inputs and lets every still-in-flight message
    /// finish arriving — comparing immediately at the final simulated tick
    /// is *not* a valid test (an earlier version of this test did exactly
    /// that and failed with a tiny, real divergence): with 5-tick latency
    /// on one link, client A genuinely hasn't learned client B's true
    /// input for the last few ticks yet by the time both sides have
    /// simulated tick 150, so A is still resimulating B's marble from a
    /// *predicted* value there — not a bug, just rollback's actual
    /// real-time guarantee (eventual convergence once all inputs are
    /// known, not instant agreement at every tick). The drain phase is
    /// what makes "all inputs are known" true before comparing.
    #[test]
    fn two_independent_clients_over_a_jittered_transport_converge() {
        let (object, mut params, marbles, starts) = two_player_setup();
        let cfg = PhysicsConfig::default();
        const TICKS: Tick = 150;

        let mut sim_a = RollbackSim::new(marbles.clone(), input(0.0, 0.0), 16);
        let mut sim_b = RollbackSim::new(marbles, input(0.0, 0.0), 16);
        let mut transport_a = InMemoryTransport::new(0); // client A owns player 0
        let mut transport_b = InMemoryTransport::new(1); // client B owns player 1
        let mut link_a_to_b = JitteredLink::new(3);
        let mut link_b_to_a = JitteredLink::new(5); // asymmetric latency on purpose

        for tick in 1..=TICKS {
            // Each client "confirms" its own player's input immediately
            // (it's the local player) and sends it to the other.
            let a_input = scripted_input(0, tick);
            let b_input = scripted_input(1, tick);
            transport_a.send_input(tick, a_input);
            transport_b.send_input(tick, b_input);
            sim_a
                .receive_inputs(&[(0, tick, a_input)], &object, &mut params, &[], &cfg, -1e6, &starts)
                .unwrap();
            sim_b
                .receive_inputs(&[(1, tick, b_input)], &object, &mut params, &[], &cfg, -1e6, &starts)
                .unwrap();

            // Ferry each side's outgoing message through its jittered link
            // into the other side's transport.
            link_a_to_b.step(tick, transport_a.drain_sent(), &mut transport_b);
            link_b_to_a.step(tick, transport_b.drain_sent(), &mut transport_a);

            // Each client learns about (some of) the remote player's input
            // for past ticks, whenever the jittered link actually delivers it.
            let arrived_at_a = transport_a.poll_received();
            if !arrived_at_a.is_empty() {
                sim_a
                    .receive_inputs(&arrived_at_a, &object, &mut params, &[], &cfg, -1e6, &starts)
                    .unwrap();
            }
            let arrived_at_b = transport_b.poll_received();
            if !arrived_at_b.is_empty() {
                sim_b
                    .receive_inputs(&arrived_at_b, &object, &mut params, &[], &cfg, -1e6, &starts)
                    .unwrap();
            }

            sim_a.advance(&object, &mut params, &[], &cfg, -1e6, &starts);
            sim_b.advance(&object, &mut params, &[], &cfg, -1e6, &starts);
        }

        // Drain phase (see this test's doc): no new ticks are simulated,
        // just let every message still in flight on either jittered link
        // (worst case, the 5-tick link) finish arriving and get absorbed.
        for extra in 1..=10u64 {
            link_a_to_b.step(TICKS + extra, Vec::new(), &mut transport_b);
            link_b_to_a.step(TICKS + extra, Vec::new(), &mut transport_a);
            let arrived_at_a = transport_a.poll_received();
            if !arrived_at_a.is_empty() {
                sim_a
                    .receive_inputs(&arrived_at_a, &object, &mut params, &[], &cfg, -1e6, &starts)
                    .unwrap();
            }
            let arrived_at_b = transport_b.poll_received();
            if !arrived_at_b.is_empty() {
                sim_b
                    .receive_inputs(&arrived_at_b, &object, &mut params, &[], &cfg, -1e6, &starts)
                    .unwrap();
            }
        }

        for (i, (a, b)) in sim_a.marbles().iter().zip(sim_b.marbles().iter()).enumerate() {
            assert_eq!(
                a.pos, b.pos,
                "client A and B diverged on marble {i} position: {:?} vs {:?}",
                a.pos, b.pos
            );
            assert_eq!(
                a.vel, b.vel,
                "client A and B diverged on marble {i} velocity: {:?} vs {:?}",
                a.vel, b.vel
            );
        }
    }

    /// A batch containing several different players' late corrections for
    /// several different (and non-contiguous) past ticks, delivered all at
    /// once, still converges to the correct full-knowledge result in a
    /// single `receive_inputs` call -- the "one resim pass per batch, not
    /// per arrival" property the module doc describes.
    #[test]
    fn simultaneous_multi_player_corrections_in_one_batch_resolve_correctly() {
        let (object, mut params, marbles, starts) = two_player_setup();
        let cfg = PhysicsConfig::default();
        const TICKS: Tick = 40;

        let mut straight = marbles.clone();
        for tick in 1..=TICKS {
            let inputs = vec![scripted_input(0, tick), scripted_input(1, tick)];
            step_marbles(&mut straight, &inputs, &object, &params, &cfg, -1e6, &starts);
        }

        // Window deliberately >= TICKS: this test is about one batch
        // correcting several players/ticks at once, not about the
        // window-exceeded boundary (that has its own dedicated test) --
        // every tick this batch touches must still be in-window.
        let mut sim = RollbackSim::new(marbles, input(0.0, 0.0), TICKS);
        for _tick in 1..=TICKS {
            sim.advance(&object, &mut params, &[], &cfg, -1e6, &starts); // everything predicted
        }
        // Now deliver a single batch correcting BOTH players across a wide,
        // non-contiguous spread of past ticks, all at once.
        let batch: Vec<_> = (1..=TICKS)
            .flat_map(|t| {
                [(0usize, t, scripted_input(0, t)), (1usize, t, scripted_input(1, t))]
            })
            .collect();
        sim.receive_inputs(&batch, &object, &mut params, &[], &cfg, -1e6, &starts).unwrap();

        for (i, (a, b)) in straight.iter().zip(sim.marbles().iter()).enumerate() {
            assert_eq!(a.pos, b.pos, "marble {i} position diverged after batched correction");
            assert_eq!(a.vel, b.vel, "marble {i} velocity diverged after batched correction");
        }
    }

    /// A burst of several late inputs for the same player arriving
    /// out of order within one batch (tick 5 before tick 3, etc.) is
    /// still applied correctly -- order within `arrivals` must not matter.
    #[test]
    fn out_of_order_arrivals_within_a_batch_still_converge() {
        let (object, mut params, marbles, starts) = two_player_setup();
        let cfg = PhysicsConfig::default();
        const TICKS: Tick = 20;

        let mut straight = marbles.clone();
        for tick in 1..=TICKS {
            let inputs = vec![scripted_input(0, tick), scripted_input(1, tick)];
            step_marbles(&mut straight, &inputs, &object, &params, &cfg, -1e6, &starts);
        }

        let mut sim = RollbackSim::new(marbles, input(0.0, 0.0), 32);
        for _tick in 1..=TICKS {
            sim.advance(&object, &mut params, &[], &cfg, -1e6, &starts);
        }
        // Deliberately shuffled, not ascending, order within the batch.
        let mut batch: Vec<_> = (1..=TICKS).map(|t| (0usize, t, scripted_input(0, t))).collect();
        batch.reverse();
        // Interleave every other one from the front too, for a genuinely
        // non-monotonic order (not just simple reversal).
        let (evens, odds): (Vec<_>, Vec<_>) = batch.into_iter().partition(|(_, t, _)| t % 2 == 0);
        let shuffled: Vec<_> = odds.into_iter().chain(evens).collect();

        sim.receive_inputs(&shuffled, &object, &mut params, &[], &cfg, -1e6, &starts).unwrap();
        for tick in 1..=TICKS {
            sim.receive_inputs(&[(1, tick, scripted_input(1, tick))], &object, &mut params, &[], &cfg, -1e6, &starts)
                .unwrap();
        }

        for (i, (a, b)) in straight.iter().zip(sim.marbles().iter()).enumerate() {
            assert_eq!(a.pos, b.pos, "marble {i} position diverged under out-of-order batch delivery");
        }
    }

    /// A confirmed input that exactly matches what was already predicted
    /// must NOT trigger a resimulation -- verified directly by checking the
    /// snapshot for an untouched later tick is still the exact same
    /// allocation-identical... actually `Vec<Marble>` has no cheap identity
    /// check, so instead verify behaviorally: seed a case where the
    /// predicted value (repeat-last-confirmed) is *known* to equal the real
    /// value (a player holding a perfectly steady input the whole time), and
    /// confirm zero resimulation work happens by checking the live state
    /// never diverges from a straight-through run at any intermediate tick,
    /// not just the end -- if a spurious resim ever ran with a bug that
    /// corrupted state, an intermediate check would catch it even if the
    /// bug happened to cancel out by the final tick.
    #[test]
    fn correct_prediction_never_causes_a_resim_to_produce_wrong_intermediate_state() {
        let (object, mut params, marbles, starts) = two_player_setup();
        let cfg = PhysicsConfig::default();
        let steady = input(0.3, -0.2);
        const TICKS: Tick = 60;

        let mut straight = marbles.clone();
        let mut straight_history = Vec::with_capacity(TICKS as usize);
        for _ in 1..=TICKS {
            step_marbles(&mut straight, &[steady, steady], &object, &params, &cfg, -1e6, &starts);
            straight_history.push(straight.clone());
        }

        let mut sim = RollbackSim::new(marbles, steady, 16);
        for tick in 1..=TICKS {
            // Confirm one tick behind -- the prediction (repeat last
            // confirmed) is *always* exactly `steady` here, i.e. always
            // correct, so this should never trigger a real resim.
            if tick > 1 {
                sim.receive_inputs(
                    &[(0, tick - 1, steady), (1, tick - 1, steady)],
                    &object,
                    &mut params,
                    &[],
                    &cfg,
                    -1e6,
                    &starts,
                )
                .unwrap();
            }
            sim.advance(&object, &mut params, &[], &cfg, -1e6, &starts);
            let expected = &straight_history[(tick - 1) as usize];
            for (i, (a, b)) in expected.iter().zip(sim.marbles().iter()).enumerate() {
                assert_eq!(
                    a.pos, b.pos,
                    "marble {i} diverged at tick {tick} despite input always being correctly predicted"
                );
            }
        }
    }

    /// A confirmed arrival for a tick already evicted from the rewind
    /// window reports [`WindowExceeded`] rather than silently producing a
    /// state that isn't what full-knowledge replay would have given.
    #[test]
    fn arrival_older_than_the_window_reports_window_exceeded() {
        let (object, mut params, marbles, starts) = two_player_setup();
        let cfg = PhysicsConfig::default();
        let mut sim = RollbackSim::new(marbles, input(0.0, 0.0), 4);

        for tick in 1..=20 {
            sim.advance(&object, &mut params, &[], &cfg, -1e6, &starts);
            let _ = tick;
        }
        // Tick 1 is far outside a window-4 rewind horizon at current_tick=20.
        let result =
            sim.receive_inputs(&[(0, 1, input(0.5, 0.5))], &object, &mut params, &[], &cfg, -1e6, &starts);
        assert!(matches!(result, Err(WindowExceeded { requested_tick: 1, .. })));
    }

    /// `RollbackSim::new` with a single marble/player, driven purely by
    /// `advance` (every input predicted-from-default since nothing's ever
    /// confirmed) must match `step_marbles` called directly with the same
    /// always-default input, tick for tick -- sanity-checks that
    /// `RollbackSim` isn't doing anything to the underlying simulation
    /// itself beyond bookkeeping.
    #[test]
    fn advance_with_nothing_ever_confirmed_matches_plain_step_marbles_with_default_input() {
        let (object, mut params) = setup();
        let rad = beware_of_bumps::MARBLE_RAD;
        let start = beware_of_bumps::START + Vec3::new(0.0, 0.3, 0.0);
        let cfg = PhysicsConfig::default();
        let default = input(0.1, -0.2);

        let mut plain = vec![Marble::spawn(start, rad)];
        let mut sim = RollbackSim::new(vec![Marble::spawn(start, rad)], default, 8);

        for _ in 0..100 {
            step_marbles(&mut plain, &[default], &object, &params, &cfg, -1e6, &[start]);
            sim.advance(&object, &mut params, &[], &cfg, -1e6, &[start]);
        }

        assert_eq!(plain[0].pos, sim.marbles()[0].pos);
        assert_eq!(plain[0].vel, sim.marbles()[0].vel);
    }

    /// The core continuity property the always-on, growable-player-count
    /// design exists for: a peer connecting must not reset the tick or
    /// perturb the already-live player's simulated state at all -- not
    /// "close enough", exactly unchanged, the instant before and the
    /// instant after `add_player_at` runs. A discard-and-rebuild design
    /// (the old `MultiplayerSession::live` lazy-init) would fail this
    /// trivially (tick resets to 0, position resets to a fresh spawn
    /// point); this is the regression test for that specific bug class.
    #[test]
    fn add_player_at_preserves_tick_and_existing_players_live_state() {
        let (object, mut params, marbles, starts) = two_player_setup();
        let cfg = PhysicsConfig::default();

        // Starts solo -- only player 0, matching the always-on design
        // (physics_sys.rs's doc: single-player is just a session with one
        // confirmed local player and no remote peers ever arriving).
        let mut sim = RollbackSim::new(vec![marbles[0]], input(0.0, 0.0), 16);
        let solo_start = [starts[0]];
        for tick in 1..=50u64 {
            sim.receive_inputs(
                &[(0, tick, scripted_input(0, tick))],
                &object,
                &mut params,
                &[],
                &cfg,
                -1e6,
                &solo_start,
            )
            .unwrap();
            sim.advance(&object, &mut params, &[], &cfg, -1e6, &solo_start);
        }

        let tick_before_join = sim.current_tick();
        let player0_before_join = sim.marbles()[0];

        // A peer connects: grow the *same* session, exactly what
        // physics_sys.rs does on `NetStatus::Connected`.
        let joiner_marble = marbles[1];
        sim.add_player_at(1, joiner_marble);

        assert_eq!(sim.current_tick(), tick_before_join, "joining must not reset the tick");
        assert_eq!(sim.num_players(), 2);
        assert_eq!(sim.marbles()[0].pos, player0_before_join.pos, "joining moved the existing player");
        assert_eq!(sim.marbles()[0].vel, player0_before_join.vel, "joining changed the existing player's velocity");
        assert_eq!(sim.marbles()[1].pos, joiner_marble.pos, "the new player should start at their spawn point");

        // Simulating onward with both players now just works, no special
        // post-join casing needed anywhere in this module.
        for tick in (tick_before_join + 1)..=(tick_before_join + 50) {
            sim.receive_inputs(
                &[(0, tick, scripted_input(0, tick)), (1, tick, scripted_input(1, tick))],
                &object,
                &mut params,
                &[],
                &cfg,
                -1e6,
                &starts,
            )
            .unwrap();
            sim.advance(&object, &mut params, &[], &cfg, -1e6, &starts);
        }
        assert_eq!(sim.current_tick(), tick_before_join + 50);
    }

    /// The other half of continuity: a *pre-existing* player's own input
    /// can legitimately arrive late enough to still be within the rewind
    /// window but for a tick that predates the join -- must resimulate
    /// through the padded historical snapshots (`add_player_at`'s doc)
    /// without panicking, using the now-2-player `step_marbles` shape for
    /// every replayed tick even though only 1 player actually existed at
    /// some of them.
    #[test]
    fn a_late_correction_for_a_pre_join_tick_still_resimulates_without_panicking() {
        let (object, mut params, marbles, starts) = two_player_setup();
        let cfg = PhysicsConfig::default();

        let mut sim = RollbackSim::new(vec![marbles[0]], input(0.0, 0.0), 16);
        let solo_start = [starts[0]];
        for tick in 1..=10u64 {
            if tick != 5 {
                sim.receive_inputs(
                    &[(0, tick, scripted_input(0, tick))],
                    &object,
                    &mut params,
                    &[],
                    &cfg,
                    -1e6,
                    &solo_start,
                )
                .unwrap();
            }
            sim.advance(&object, &mut params, &[], &cfg, -1e6, &solo_start);
        }

        sim.add_player_at(1, marbles[1]);

        // The real tick-5 input for player 0 finally arrives -- within the
        // window (current_tick=10, window=16), but at a tick that predates
        // the join at tick 10.
        sim.receive_inputs(
            &[(0, 5, scripted_input(0, 5))],
            &object,
            &mut params,
            &[],
            &cfg,
            -1e6,
            &starts,
        )
        .unwrap();

        assert_eq!(sim.current_tick(), 10);
        assert_eq!(sim.num_players(), 2);
    }

    /// [`RollbackSim::narrow_to`]: the rare edge case of a solo session
    /// that already had more than one marble (`Demo`'s decorative extras,
    /// `render::setup`'s doc) gaining a real second player -- narrowing
    /// must keep the given player's exact live state and the session's
    /// current tick, discarding everyone else.
    #[test]
    fn narrow_to_keeps_only_the_given_players_live_state_and_tick() {
        let (object, mut params, marbles, starts) = two_player_setup();
        let cfg = PhysicsConfig::default();
        let mut sim = RollbackSim::new(marbles, input(0.0, 0.0), 16);
        for tick in 1..=20u64 {
            sim.receive_inputs(
                &[(0, tick, scripted_input(0, tick)), (1, tick, scripted_input(1, tick))],
                &object,
                &mut params,
                &[],
                &cfg,
                -1e6,
                &starts,
            )
            .unwrap();
            sim.advance(&object, &mut params, &[], &cfg, -1e6, &starts);
        }
        let player1_before = sim.marbles()[1];
        let tick_before = sim.current_tick();

        let narrowed = sim.narrow_to(1);
        assert_eq!(narrowed.num_players(), 1);
        assert_eq!(narrowed.current_tick(), tick_before);
        assert_eq!(narrowed.marbles()[0].pos, player1_before.pos);
        assert_eq!(narrowed.marbles()[0].vel, player1_before.vel);
    }

    /// [`RollbackSim::respawn`] is only ever safe to call solo (its own
    /// doc) -- this just checks the mechanical reset itself.
    #[test]
    fn respawn_resets_position_and_velocity_in_place() {
        let (object, mut params, marbles, starts) = two_player_setup();
        let cfg = PhysicsConfig::default();
        let mut sim = RollbackSim::new(vec![marbles[0]], input(0.0, 0.0), 16);
        let solo_start = [starts[0]];
        for tick in 1..=10u64 {
            sim.receive_inputs(
                &[(0, tick, scripted_input(0, tick))],
                &object,
                &mut params,
                &[],
                &cfg,
                -1e6,
                &solo_start,
            )
            .unwrap();
            sim.advance(&object, &mut params, &[], &cfg, -1e6, &solo_start);
        }
        assert_ne!(sim.marbles()[0].vel, Vec3::ZERO, "expected some motion before respawn");

        sim.respawn(0, solo_start[0]);
        assert_eq!(sim.marbles()[0].pos, solo_start[0]);
        assert_eq!(sim.marbles()[0].vel, Vec3::ZERO);
    }

    /// [`RollbackSim::debug_perturb_position`]'s mechanical contract: a
    /// direct, immediate position nudge, nothing fancier -- this is the
    /// hook live CDP verification uses to manufacture a real cross-peer
    /// divergence on demand, so it just needs to reliably do the one
    /// simple thing it claims to. `cfg`-gated the same way the method
    /// itself is: a `--release` test run (`debug_assertions` off) doesn't
    /// have this method to call at all.
    #[cfg(debug_assertions)]
    #[test]
    fn debug_perturb_position_is_a_direct_position_nudge() {
        let (_, _, marbles, _) = two_player_setup();
        let other_player_pos_before = marbles[1].pos;
        let mut sim = RollbackSim::new(marbles.clone(), input(0.0, 0.0), 16);
        sim.debug_perturb_position(0, Vec3::new(5.0, 0.0, 0.0));
        assert_eq!(sim.marbles()[0].pos, marbles[0].pos + Vec3::new(5.0, 0.0, 0.0));
        assert_eq!(sim.marbles()[1].pos, other_player_pos_before, "perturbing player 0 must not touch player 1");
    }

    /// The animation half of continuity: since [`Expr::eval`] is a pure
    /// function of `Tick` alone (`crate::expr`'s module doc), and joining
    /// never resets the tick (this test module's `add_player_at_*` test
    /// above), an animated param's value right before and right after a
    /// join must be *identical* if evaluated at the same tick -- there is
    /// no discrete jump for a live scene's fractal to visibly display the
    /// instant a second player connects.
    #[test]
    fn animation_value_is_unaffected_by_a_join_at_the_same_tick() {
        let anim = Expr::Sin(Box::new(Expr::Mul(Box::new(Expr::Tick), Box::new(Expr::Const(0.07)))));
        let (_, mut params, marbles, starts) = two_player_setup();
        let cfg = PhysicsConfig::default();
        let radius = params.alloc_scalar(0.0);
        let animations = vec![(radius, anim.clone())];
        let object = Object::Sphere { radius: ScalarValue::Param(radius) };

        let mut sim = RollbackSim::new(vec![marbles[0]], input(0.0, 0.0), 16);
        let solo_start = [starts[0]];
        for tick in 1..=30u64 {
            sim.advance(&object, &mut params, &animations, &cfg, -1e6, &solo_start);
            assert_eq!(params.scalar(radius), anim.eval(tick), "tick {tick} animation value diverged pre-join");
        }
        let tick_at_join = sim.current_tick();

        sim.add_player_at(1, marbles[1]);
        assert_eq!(sim.current_tick(), tick_at_join, "join must not move the tick the animation is a function of");

        for tick in (tick_at_join + 1)..=(tick_at_join + 30) {
            sim.advance(&object, &mut params, &animations, &cfg, -1e6, &starts);
            assert_eq!(params.scalar(radius), anim.eval(tick), "tick {tick} animation value diverged post-join");
        }
    }

    /// [`state_checksum`] must be a pure function of position/velocity —
    /// identical state hashes identically twice, and perturbing either
    /// position or velocity on any one marble must change the result.
    /// Doesn't prove the hash is collision-free (FNV-1a isn't
    /// cryptographic — this module doesn't need it to be, see
    /// `state_checksum`'s doc), just that it's actually sensitive to the
    /// state it's supposed to be fingerprinting.
    #[test]
    fn state_checksum_is_deterministic_and_sensitive_to_position_and_velocity() {
        let (_, _, marbles, _) = two_player_setup();
        let hash = state_checksum(&marbles);
        assert_eq!(hash, state_checksum(&marbles), "checksum must be deterministic");

        let mut moved = marbles.clone();
        moved[0].pos.x += 0.001;
        assert_ne!(state_checksum(&moved), hash, "checksum must change when a position changes");

        let mut sped_up = marbles.clone();
        sped_up[1].vel.y += 0.001;
        assert_ne!(state_checksum(&sped_up), hash, "checksum must change when a velocity changes");
    }

    /// The core property [`RollbackSim::checksum_at`] exists to guarantee:
    /// once a tick becomes eligible (falls outside the rewind window), its
    /// cached checksum matches what a full-knowledge straight-through
    /// replay would have produced for that exact tick — not a stale value
    /// left over from whatever was first *predicted* for it before player
    /// 0's real, deliberately-delayed input actually arrived and forced a
    /// resim. Mirrors
    /// `rollback_replay_with_an_animated_param_matches_full_knowledge_replay`'s
    /// shape (same delayed-confirmation setup), but checks every eligible
    /// tick's checksum along the way instead of just the final live state.
    #[test]
    fn checksum_of_an_eligible_tick_matches_the_full_knowledge_replay() {
        let (object, mut params, marbles, starts) = two_player_setup();
        let cfg = PhysicsConfig::default();
        const TICKS: Tick = 200;
        const DELAY: Tick = 4;
        const WINDOW: Tick = 16;

        let mut straight = marbles.clone();
        let mut straight_history: BTreeMap<Tick, Vec<Marble>> = BTreeMap::new();
        for tick in 1..=TICKS {
            let inputs = vec![scripted_input(0, tick), scripted_input(1, tick)];
            step_marbles(&mut straight, &inputs, &object, &params, &cfg, -1e6, &starts);
            straight_history.insert(tick, straight.clone());
        }

        let mut sim = RollbackSim::new(marbles, input(0.0, 0.0), WINDOW);
        for tick in 1..=TICKS {
            sim.receive_inputs(
                &[(1, tick, scripted_input(1, tick))],
                &object,
                &mut params,
                &[],
                &cfg,
                -1e6,
                &starts,
            )
            .expect("within window");
            sim.advance(&object, &mut params, &[], &cfg, -1e6, &starts);
            if tick >= DELAY {
                let confirm_tick = tick - DELAY + 1;
                sim.receive_inputs(
                    &[(0, confirm_tick, scripted_input(0, confirm_tick))],
                    &object,
                    &mut params,
                    &[],
                    &cfg,
                    -1e6,
                    &starts,
                )
                .expect("within window");
            }
        }
        for confirm_tick in (TICKS - DELAY + 2)..=TICKS {
            sim.receive_inputs(
                &[(0, confirm_tick, scripted_input(0, confirm_tick))],
                &object,
                &mut params,
                &[],
                &cfg,
                -1e6,
                &starts,
            )
            .expect("within window");
        }

        let last_eligible = sim.latest_checksum_tick().expect("should have eligible ticks by now");
        let mut checked_any = false;
        for tick in 1..=last_eligible {
            if let Some(hash) = sim.checksum_at(tick) {
                checked_any = true;
                assert_eq!(
                    hash,
                    state_checksum(&straight_history[&tick]),
                    "tick {tick}'s cached checksum doesn't match the full-knowledge replay"
                );
            }
        }
        assert!(checked_any, "expected at least one cached checksum to have been produced");
    }

    /// [`RollbackSim::hard_reset_to`]'s mechanical contract in isolation
    /// (no mismatch scenario yet — that's the next test): live state and
    /// `current_tick` are fully replaced, and the reseeded window supports
    /// simulating onward immediately, including a same-tick-as-the-reset
    /// rewind/resim, with no special-casing needed anywhere else in this
    /// module.
    #[test]
    fn hard_reset_to_replaces_live_state_and_reseeds_a_working_window() {
        let (object, mut params, marbles, starts) = two_player_setup();
        let cfg = PhysicsConfig::default();
        let mut sim = RollbackSim::new(marbles, input(0.0, 0.0), 16);
        for tick in 1..=30u64 {
            sim.receive_inputs(
                &[(0, tick, scripted_input(0, tick)), (1, tick, scripted_input(1, tick))],
                &object,
                &mut params,
                &[],
                &cfg,
                -1e6,
                &starts,
            )
            .unwrap();
            sim.advance(&object, &mut params, &[], &cfg, -1e6, &starts);
        }

        let authoritative_tick = 500;
        let authoritative_marbles = vec![
            Marble::spawn(Vec3::new(9.0, 9.0, 9.0), beware_of_bumps::MARBLE_RAD),
            Marble::spawn(Vec3::new(-9.0, -9.0, -9.0), beware_of_bumps::MARBLE_RAD),
        ];
        sim.hard_reset_to(authoritative_tick, authoritative_marbles.clone());

        assert_eq!(sim.current_tick(), authoritative_tick);
        assert_eq!(sim.marbles()[0].pos, authoritative_marbles[0].pos);
        assert_eq!(sim.marbles()[1].pos, authoritative_marbles[1].pos);
        assert_eq!(sim.marbles()[0].vel, Vec3::ZERO);

        // Simulating onward works immediately: a plain advance...
        sim.receive_inputs(
            &[(0, authoritative_tick + 1, scripted_input(0, authoritative_tick + 1))],
            &object,
            &mut params,
            &[],
            &cfg,
            -1e6,
            &starts,
        )
        .unwrap();
        sim.advance(&object, &mut params, &[], &cfg, -1e6, &starts);
        assert_eq!(sim.current_tick(), authoritative_tick + 1);
        sim.advance(&object, &mut params, &[], &cfg, -1e6, &starts);
        assert_eq!(sim.current_tick(), authoritative_tick + 2);

        // ...and a late correction landing right after the reset must
        // still trigger a valid rewind/resim against the *reseeded*
        // snapshot chain, not panic against pre-reset history.
        sim.receive_inputs(
            &[(0, authoritative_tick + 2, scripted_input(0, authoritative_tick + 2 + 1000))],
            &object,
            &mut params,
            &[],
            &cfg,
            -1e6,
            &starts,
        )
        .unwrap();
        assert_eq!(sim.current_tick(), authoritative_tick + 2);
    }

    /// The end-to-end scenario the whole feature exists for: two
    /// independent `RollbackSim`s, fed identical confirmed inputs and
    /// therefore expected to agree exactly, where one has its live state
    /// deliberately corrupted mid-session (standing in for "a real
    /// simulation bug made these two peers silently diverge") — checksum
    /// comparison must detect the resulting mismatch, and
    /// `hard_reset_to`'ing the corrupted side to the authoritative side's
    /// state must restore exact agreement, with both sides continuing to
    /// agree afterward.
    #[test]
    fn a_deliberately_corrupted_peer_is_detected_by_checksum_and_recovered_by_hard_reset() {
        let (object, mut params, marbles, starts) = two_player_setup();
        let cfg = PhysicsConfig::default();
        const WINDOW: Tick = 16;
        let mut authoritative = RollbackSim::new(marbles.clone(), input(0.0, 0.0), WINDOW);
        let mut corrupted = RollbackSim::new(marbles, input(0.0, 0.0), WINDOW);

        let feed = |sim: &mut RollbackSim, params: &mut Params, tick: Tick| {
            sim.receive_inputs(
                &[(0, tick, scripted_input(0, tick)), (1, tick, scripted_input(1, tick))],
                &object,
                params,
                &[],
                &cfg,
                -1e6,
                &starts,
            )
            .unwrap();
            sim.advance(&object, params, &[], &cfg, -1e6, &starts);
        };

        // Both sides agree for a while, exactly like two real peers
        // exchanging identical confirmed input over a healthy connection.
        for tick in 1..=40u64 {
            feed(&mut authoritative, &mut params, tick);
            feed(&mut corrupted, &mut params, tick);
        }
        let agree_tick = authoritative.latest_checksum_tick().unwrap();
        assert_eq!(
            authoritative.checksum_at(agree_tick),
            corrupted.checksum_at(agree_tick),
            "sanity check: both sides must actually agree before the injected corruption"
        );

        // Simulate "a real bug corrupted this peer's simulation" by
        // directly nudging its live marble state -- a test-only
        // privilege, only available because this test lives inside the
        // same module as `RollbackSim`'s private fields (no production
        // code path can or should do this).
        corrupted.marbles[0].pos.x += 2.5;

        // Continue feeding both sides identical input -- the corruption
        // propagates forward through `corrupted`'s own physics from here
        // on, exactly like a real silent divergence would.
        for tick in 41..=(40 + WINDOW + 10) {
            feed(&mut authoritative, &mut params, tick);
            feed(&mut corrupted, &mut params, tick);
        }

        let mismatch_tick = authoritative.latest_checksum_tick().unwrap();
        assert_eq!(authoritative.current_tick(), corrupted.current_tick());
        let authoritative_hash = authoritative.checksum_at(mismatch_tick).expect("must be cached by now");
        let corrupted_hash = corrupted.checksum_at(mismatch_tick).expect("must be cached by now");
        assert_ne!(
            authoritative_hash, corrupted_hash,
            "the injected corruption must be detectable as a checksum mismatch"
        );

        // The non-authoritative side resyncs to the authoritative side's
        // live state.
        corrupted.hard_reset_to(authoritative.current_tick(), authoritative.marbles().to_vec());
        assert_eq!(corrupted.marbles()[0].pos, authoritative.marbles()[0].pos);
        assert_eq!(corrupted.marbles()[1].pos, authoritative.marbles()[1].pos);

        // And, having recovered, the two sides stay in agreement under
        // continued identical input -- the reset didn't just patch the
        // symptom for one tick, it left both sides in a state that keeps
        // agreeing going forward.
        for tick in (corrupted.current_tick() + 1)..=(corrupted.current_tick() + WINDOW + 10) {
            feed(&mut authoritative, &mut params, tick);
            feed(&mut corrupted, &mut params, tick);
        }
        let recovered_tick = authoritative.latest_checksum_tick().unwrap();
        assert_eq!(
            authoritative.checksum_at(recovered_tick),
            corrupted.checksum_at(recovered_tick),
            "both sides must agree again after the resync"
        );
    }
}
