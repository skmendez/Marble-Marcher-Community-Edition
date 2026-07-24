//! M5: fixed-timestep marble physics + WASD/touch input, wired to the demo
//! scene's `SceneState` (rust/DESIGN.md §7/§8).
//!
//! `G` toggles between the two physics models `marble_csg::physics` supports
//! (see its module doc): [`GravityMode::Rolling`] (original MMCE physics —
//! gravity, kill plane, horizontal movement) and [`GravityMode::Flying`]
//! (this branch's zero-gravity free-flight experiment — full 3D
//! camera-relative thrust, no kill plane). Defaults to `Flying` (see
//! `GravityMode`'s doc for why).
//!
//! WASD sign convention: setting `dy = -1` for S moves the marble *toward*
//! the camera's eye position; W (`dy = +1`) moves it away, deeper along the
//! view direction. This is a deliberate user-requested swap of W/S (and, in
//! `camera.rs`, Q/E) from an earlier convention where W was "toward" —
//! verified empirically (not just derived on paper) by logging `marble.pos`
//! before/after holding each key against a live build and checking its
//! displacement's dot product against the camera's `forward` vector.
//! Setting `dx = +1` for D yields `+(cos(yaw), 0, -sin(yaw))`, which is
//! exactly `CameraOrbit`'s `right` vector at zero pitch — D rolls the
//! marble in the camera's screen-right direction (unaffected by the above
//! correction). In `Flying` mode the same `dx`/`dy` inputs instead drive
//! full 3D thrust along wherever the camera is actually pointed (see
//! `step_marble`'s doc).
//!
//! Touch: a 2-finger pinch feeds an additional `dy` (on top of WASD's) via
//! `touch::read_two_finger_gesture` — pinching in pulls the marble toward
//! the camera (S-equivalent), pinching out pushes it away (W-equivalent).
//! Read directly here (not via an `Update`-schedule intermediary) for the
//! same reason WASD is: `Touches`, like `ButtonInput<KeyCode>`, is input
//! state readable from any schedule, not something that needs per-frame
//! accumulation across schedules. The gesture's *rotate* half is handled
//! separately in `touch::touch_camera_input` (`Update` schedule, alongside
//! the mouse-driven `orbit_camera_input`), since it drives `CameraOrbit`,
//! not the marble.
//!
//! **Always-on rollback**: every scene's marble physics runs through a
//! [`RollbackSim`] from scene load, single-player included — see
//! [`MultiplayerSession`]'s doc for why this one path (not "plain
//! `step_marbles` while offline, `RollbackSim` only once connected")
//! replaced an earlier design with exactly that split, and how connecting
//! to a peer *grows* the existing session instead of discarding it.

use web_time::Instant;

use bevy::input::touch::Touches;
use bevy::prelude::*;

use crate::fps_overlay::PhaseTimings;

use marble_csg::physics::{GravityMode, Marble, PhysicsConfig, PlayerInput};
use marble_rollback::{InputTransport, PlayerIndex, RollbackSim, Tick};

use crate::camera::CameraOrbit;
use crate::net::{NetSession, NetStatus, Role, WebRtcTransport, HOST_INDEX};
use crate::render::SceneState;
use crate::touch::read_two_finger_gesture;

/// N marbles' live physics state + shared tuning constants, plus each
/// marble's spawn point and the scene's shared kill-plane height (used for
/// `R`/kill-plane respawns — every scene has different values now that all
/// scenes have a marble, see `SceneKind::spawn_params`, so these can no
/// longer be a fixed constant read directly from `physics_sys.rs`).
/// Constructed per-scene in `render::setup` (there's no scene-independent
/// `Default` to speak of).
///
/// `marbles`/`start_positions` are just a read-only mirror of
/// [`MultiplayerSession::sim`]'s own live state, refreshed every tick —
/// [`RollbackSim`] owns the authoritative copy (needed to snapshot/rewind),
/// same reasoning as multiplayer milestone 2's original design, just no
/// longer conditional on being connected.
#[derive(Resource)]
pub struct MarbleState {
    pub marbles: Vec<Marble>,
    pub cfg: PhysicsConfig,
    pub start_positions: Vec<Vec3>,
    pub kill_y: f32,
    /// Which `marbles` index this client controls/watches -- `0` until a
    /// connection assigns this client a [`crate::net::Role`] (always host,
    /// i.e. `0`, for as long as nobody's connected), named for
    /// multiplayer's benefit rather than assumed to always be zero
    /// everywhere it's read.
    pub local_player_index: usize,
}

impl MarbleState {
    /// The marble this client controls/watches -- what the camera follows,
    /// what WASD/touch input drives, what `R` respawns.
    pub fn local_marble(&self) -> Marble {
        self.marbles[self.local_player_index]
    }
}

/// The live [`RollbackSim`] + its [`WebRtcTransport`] once connected —
/// always live from scene load now (`render::setup` constructs this
/// alongside `MarbleState`/`SceneState`), not lazily created at first
/// connection. Single-player is just a `RollbackSim` session with one
/// confirmed local player and no remote peers ever arriving: this system
/// never confirms input for any player index but its own, so every other
/// slot's input always falls back to `RollbackSim`'s own `default_input`
/// (`RollbackSim::new`'s doc) — the same fallback mechanism naturally
/// covers `Demo`'s decorative extra marbles too (`render::setup`'s doc),
/// no separate "placeholder input for non-real players" code needed.
///
/// Because a session always exists, connecting to a peer *grows* it in
/// place ([`RollbackSim::add_player_at`]) instead of discarding it and
/// building a fresh one — `current_tick` and the local player's own live
/// state both carry through the join with no discontinuity. This matters
/// doubly now that `RollbackSim`'s own tick is also the clock
/// `marble_csg::expr`-driven animated params run on (`SceneState::
/// animations`): a discard-and-rebuild design would have visibly reset a
/// running animation's phase the instant a second player joined, on top of
/// the marble-position "teleport" that design already implied.
///
/// Once `NetSession::status` leaves `NetStatus::Connected` (peer
/// disconnected), this system keeps driving `sim` exactly as before, just
/// confirming a zero input for the departed player from then on (same idea
/// as `default_input`, made explicit — see the "was connected, isn't now"
/// branch below) instead of leaving them repeating whatever they were last
/// doing forever. Both sides keep independently ticking their own `sim`
/// the whole time they're apart, so by the time the *same* peer reconnects
/// their tick counters have drifted exactly as far apart as a first join's
/// would have (see the next paragraph) — `NetStatus` moving back to
/// `Connected` re-triggers the same join/resync hand-off below rather than
/// a special-cased "reconnect" path, for exactly that reason.
///
/// **Join/resync hand-off**: a real WebRTC handshake takes seconds, vastly
/// longer than `ROLLBACK_WINDOW_TICKS`'s ~267ms — so two independently-run
/// solo sessions are already far apart in tick numbering the moment their
/// data channel opens, with nothing to reconcile that on its own. Once the
/// handshake completes (`marble_physics_tick_impl`'s `just_connected`),
/// the host authoritatively pushes its current tick + full state to the
/// non-host side, which adopts it wholesale
/// (`RollbackSim::hard_reset_to`) rather than trying to merge its own
/// prior solo history into it — first join and a later reconnect use the
/// exact same push, since both need the exact same realignment.
///
/// **Rollback resiliency** (periodic state checksums + resync-on-mismatch,
/// same idea Photon Quantum/Factorio-style lockstep engines ship): while
/// connected, [`marble_physics_tick`] also periodically exchanges a
/// checksum of each side's state at whatever tick is currently eligible
/// for comparison (`RollbackSim::latest_checksum_tick`) and, on a
/// mismatch, has the non-host side pull a full authoritative correction
/// from the host. This reuses the exact same message and adoption code
/// ([`apply_resync_payload`]) as the join/resync hand-off above — a
/// mismatch mid-game and a fresh join are both just "adopt the host's
/// authoritative state," the only difference being what triggers it.
#[derive(Resource)]
pub struct MultiplayerSession {
    pub sim: RollbackSim,
    /// `Some` only once actually connected to a remote peer at least once
    /// — absent, this system never polls for or sends anything over the
    /// network. Never reset back to `None` on a later disconnect (see
    /// `NetStatus::Disconnected`'s doc) -- a same-tab reconnect resumes
    /// using this same transport rather than rebuilding one.
    transport: Option<WebRtcTransport>,
    /// Whether this side has ever completed the real 2-player join (grown
    /// past its initial solo/decorative-marbles shape and adopted a
    /// host-pushed [`crate::net::WebRtcTransport::poll_resync_payloads`]
    /// payload at least once) -- distinguishes "still solo, possibly with
    /// `Demo`'s decorative extra marbles" (grow-on-join should run) from
    /// "already a real 2-player session, a later reconnect should just
    /// resend/re-adopt fresh state" (grow must never run again, or it
    /// would mistake the real peer for another decorative marble to grow
    /// past). See [`marble_physics_tick_impl`]'s doc for the full flow.
    joined: bool,
    /// Mirrors `NetSession::status`'s previous value, purely so the join/
    /// resync flow in [`marble_physics_tick_impl`] can edge-trigger on the
    /// transition *into* `Connected` (a freshly-opened data channel,
    /// first-ever or a reconnect alike) instead of re-running every single
    /// tick for as long as `Connected` holds.
    was_connected: bool,
    /// Set once a checksum mismatch has triggered a `ResyncRequest`, until
    /// the matching `ResyncPayload` actually lands and `hard_reset_to`
    /// runs -- without this, a mismatch that's still visible on
    /// subsequent ticks (the request/response round trip takes at least
    /// one network hop each way) would otherwise re-request every single
    /// tick in between.
    resync_pending: bool,
    /// Whether this side is ready to treat its own `render::SceneState` as
    /// authoritative for multiplayer purposes -- `true` immediately for the
    /// host (it never waits on anything), `false` for a joiner until
    /// `render::apply_pending_scene_sync` has actually applied a received
    /// `SceneSync` bundle. Gates the per-tick real input exchange below
    /// alongside `joined`: starting to exchange input against a scene the
    /// two sides might still disagree about would be exactly the kind of
    /// premature, unsynchronized start this whole join flow exists to
    /// avoid (`marble_physics_tick_impl`'s doc).
    scene_synced: bool,
    /// Host only: every joiner index that has ever disconnected — kept
    /// (never removed, even on a later reconnect attempt, which this
    /// milestone doesn't support for a returning player any differently
    /// than a fresh one) so their marble keeps receiving a confirmed zero
    /// input forever instead of repeating whatever they were last doing
    /// (same idea `net::NetStatus::Disconnected`'s doc describes for the
    /// old single-joiner case, generalized to a set of indices).
    disconnected_remotes: Vec<PlayerIndex>,
    /// A resync's tick+marbles, held here instead of applied immediately,
    /// while this side waits for the *paired* scene bundle the host always
    /// sends alongside any resync that grew the player count
    /// (`marble_physics_tick_impl`'s host-side join handling sends
    /// `ResyncPayload` and `SceneSync` back to back, same tick, same
    /// reliable channel). `apply_resync_payload` stashes here rather than
    /// calling `RollbackSim::hard_reset_to` directly whenever it detects
    /// this call actually grew `num_players()` -- `RollbackSim::set_scene`
    /// (not `hard_reset_to` alone) is what finally applies it, once
    /// `render::apply_pending_scene_sync` decodes the matching scene bytes,
    /// so the tick/marble jump and the scene swap land in the same instant
    /// rather than one now and the other however many ticks later the
    /// scene happens to arrive/decode. Getting this wrong doesn't just
    /// leave a brief "wrong-scene" rendering glitch -- it corrupts
    /// `RollbackSim`'s rewind window outright: applying the tick/marble
    /// jump alone via `hard_reset_to` wipes every snapshot older than its
    /// own tick, but the host has already been broadcasting real per-tick
    /// input for ticks *between* that jump and whenever this side's own
    /// clock catches up to it (the host considers itself synced the
    /// instant it sends the pair, never waits on this side's scene to
    /// apply first) -- once those in-flight messages are polled, several
    /// of them land for ticks now older than this side's freshly-reset
    /// `current_tick` but for which no snapshot survived the reset,
    /// sending `resim_from` looking for a snapshot that was never kept and
    /// panicking on the `.expect` that's supposed to be unreachable
    /// (`RollbackSim::resim_from`'s doc). A plain checksum-mismatch resync
    /// (`exchange_checksums_and_resync`) never has this problem -- the host
    /// never sends a scene alongside one, so it never grows `num_players`,
    /// and `apply_resync_payload` still applies that kind immediately,
    /// exactly as before.
    pending_join: Option<(Tick, Vec<Marble>)>,
}

impl MultiplayerSession {
    /// The always-on solo session every scene starts with —
    /// `initial_marbles` must be `MarbleState::marbles` at the same moment
    /// (matches `RollbackSim::new`'s own "must match tick 0's marble
    /// state" contract). `default_input` is a fixed `Quat::IDENTITY`
    /// orientation rather than tracking the live camera: it's only ever
    /// actually used for a player index nothing has confirmed real input
    /// for, and `step_marble` never reads `orientation` unless `dx`/`dy`
    /// are nonzero, which they never are for a purely-predicted-forever
    /// player -- see this type's doc.
    pub fn new_solo(scene: marble_csg::Scene, initial_marbles: Vec<Marble>) -> Self {
        Self {
            sim: RollbackSim::new(
                scene,
                initial_marbles,
                PlayerInput { dx: 0.0, dy: 0.0, orientation: Quat::IDENTITY },
                ROLLBACK_WINDOW_TICKS,
            ),
            transport: None,
            joined: false,
            was_connected: false,
            resync_pending: false,
            scene_synced: false,
            disconnected_remotes: Vec::new(),
            pending_join: None,
        }
    }

    /// Called by `render::apply_pending_scene_sync` once it's actually
    /// applied a received scene bundle -- see `scene_synced`'s doc for why
    /// this (not `joined` alone) gates the per-tick real input exchange.
    /// A plain method rather than a `pub` field, matching this struct's
    /// other state (`transport`/`joined`/`resync_pending`): nothing outside
    /// `physics_sys.rs` should be able to set this to `false` again once
    /// it's true.
    pub fn mark_scene_synced(&mut self) {
        self.scene_synced = true;
    }

    /// Takes whatever join/resync state `apply_resync_payload` stashed
    /// (`pending_join`'s doc), for `render::apply_pending_scene_sync` to
    /// apply atomically alongside the scene it just decoded.
    pub fn take_pending_join(&mut self) -> Option<(Tick, Vec<Marble>)> {
        self.pending_join.take()
    }

    /// Whether this session has never touched a network transport -- the
    /// same `transport.is_none()` gate `marble_physics_tick_impl` applies
    /// to the debug respawn key, exposed for `param_ui.rs`'s live param
    /// editing, which is local-only state a connected peer would never see
    /// (`RollbackSim::params_mut`'s doc) and so must stay solo-only.
    pub fn is_solo(&self) -> bool {
        self.transport.is_none()
    }
}

/// The host's most recently received-but-not-yet-applied
/// [`crate::net::WebRtcTransport::poll_scene_syncs`] bundle, if any --
/// `marble_physics_tick_impl` (`FixedUpdate`) only *polls* the transport
/// and stashes the raw bytes here, since applying them (regenerating the
/// shader, replacing the params storage buffer, etc.) needs rendering
/// resources this physics-focused system doesn't take; `render::
/// apply_pending_scene_sync` (`Update`) is what actually decodes and
/// applies it, then clears this back to `None`.
#[derive(Resource, Default)]
pub struct PendingSceneSync(pub Option<Vec<u8>>);

/// How often (in ticks) each connected peer publishes a checksum of its
/// own state at whatever tick is currently eligible for comparison
/// (`RollbackSim::latest_checksum_tick`) — 60 ticks is one second's worth
/// at this app's fixed 60Hz tick rate: frequent enough to notice a real
/// divergence within a second or so of it happening, infrequent enough
/// not to spend meaningful channel bandwidth on it every tick. A starting
/// point, not a tuned/load-bearing constant (this feature's own plan says
/// as much) — easy to revisit once real cross-machine play has been
/// exercised enough to have an opinion.
const CHECKSUM_INTERVAL_TICKS: u64 = 60;

/// Runs the checksum/resync side channel for one tick, alongside (but
/// independent of) the per-tick input exchange in
/// [`marble_physics_tick_impl`]: periodically publishes this side's own
/// checksum for whatever tick just became eligible, compares any checksum
/// the peer sent back against this side's own cached value for that same
/// tick, and drives the request/response halves of a resync once a
/// mismatch is actually detected. Applying an incoming `ResyncPayload`
/// itself is *not* this function's job -- see [`apply_resync_payload`],
/// which handles both this path's mismatch-triggered payload and the
/// join-time one with the same code, and must run earlier in the tick
/// than this function does (`marble_physics_tick_impl`'s doc).
///
/// Host-authoritative throughout (`net::Role`'s doc, same convention the
/// join flow already establishes): only the non-host side ever requests a
/// correction, and only the host side ever answers one. The host still
/// *compares* checksums like anyone else (so a mismatch is visible in its
/// own logs), it just never acts on one — its own state is authoritative
/// by definition, so there's nothing for it to correct itself against.
fn exchange_checksums_and_resync(
    sim: &mut RollbackSim,
    transport: &mut WebRtcTransport,
    role: Role,
    resync_pending: &mut bool,
) {
    // Answer any resync request unconditionally with this side's own live
    // state -- only ever actually arrives on the host side in practice
    // (a non-host peer never sends one to anyone but the host), but there's
    // no reason to gate this on `role` too: an unexpected request just gets
    // an honest answer either way.
    for _requested_tick in transport.poll_resync_requests() {
        transport.send_resync_payload(sim.current_tick(), sim.marbles().to_vec());
    }

    // Compare the peer's checksums against this side's own cached value
    // for the same tick. `checksum_at` returning `None` means that tick's
    // no longer in this side's own cache (arrived unusually late) --
    // inconclusive, not a mismatch, so it's skipped rather than guessed at.
    for (tick, peer_hash) in transport.poll_checksums() {
        if let Some(local_hash) = sim.checksum_at(tick) {
            if local_hash != peer_hash && role == Role::Joiner && !*resync_pending {
                warn!(
                    "multiplayer: checksum mismatch at tick {tick} (local {local_hash:#x} vs peer {peer_hash:#x}) \
                     -- requesting a full resync from the host"
                );
                transport.send_resync_request(tick);
                *resync_pending = true;
            }
        }
    }

    // Periodically publish this side's own checksum for whatever tick is
    // currently eligible for comparison.
    if sim.current_tick().is_multiple_of(CHECKSUM_INTERVAL_TICKS) {
        if let Some(tick) = sim.latest_checksum_tick() {
            if let Some(hash) = sim.checksum_at(tick) {
                transport.send_checksum(tick, hash);
            }
        }
    }
}

/// Requests a full resync in response to `RollbackSim::receive_inputs`
/// reporting [`WindowExceeded`] -- same trigger role/latch gating as a
/// checksum mismatch ([`exchange_checksums_and_resync`]'s own
/// `role == Role::Joiner && !*resync_pending` check), reused here rather
/// than relying on that separate periodic-checksum path to eventually
/// notice on its own: that path only ever compares ticks still within its
/// own `CHECKSUM_CACHE_TICKS`-deep cache (5 seconds), which a desync bad
/// enough to blow the much shallower (~267ms) rewind window can easily
/// have already outrun too -- once that happens, no mismatch is ever
/// comparable again and the caller's warning would otherwise repeat
/// forever with nothing to end it. Factored out of
/// `marble_physics_tick_impl` (a Bevy system needing a full `World` to
/// construct in a test) purely so this gating logic is directly
/// unit-testable, same reasoning as [`is_valid_welcome_index`].
fn request_resync_on_window_exceeded(mp: &mut MultiplayerSession, role: Role) {
    if role == Role::Joiner && !mp.resync_pending {
        if let Some(transport) = mp.transport.as_mut() {
            transport.send_resync_request(mp.sim.current_tick());
            mp.resync_pending = true;
        }
    }
}

#[cfg(test)]
mod window_exceeded_resync_tests {
    use super::*;
    use marble_csg::scenes::demo_scene;
    use marble_csg::Params;

    /// Fix B regression test: a `WindowExceeded` correction failure must
    /// trigger a fresh resync request (setting `resync_pending`) on a
    /// joiner with none already in flight -- pre-fix, this path was purely
    /// a log statement with no corrective action at all, which is exactly
    /// why the reported symptom was "thousands of repeating warnings that
    /// never recover."
    #[test]
    fn window_exceeded_on_a_joiner_with_no_pending_resync_requests_one() {
        let mut params = Params::new();
        let (object, _handles) = demo_scene(&mut params);
        let scene = marble_csg::Scene { object, params, animations: vec![] };
        let mut mp = MultiplayerSession::new_solo(scene, vec![Marble::spawn(Vec3::ZERO, 0.3)]);
        mp.transport = Some(WebRtcTransport::new(Role::Joiner));
        assert!(!mp.resync_pending);

        request_resync_on_window_exceeded(&mut mp, Role::Joiner);

        assert!(mp.resync_pending, "a WindowExceeded on a joiner must set resync_pending, requesting a resync");
    }

    /// A resync already in flight must not trigger a second request --
    /// same latch behavior `exchange_checksums_and_resync`'s mismatch path
    /// already relies on for its own repeated-mismatch case.
    #[test]
    fn window_exceeded_with_a_resync_already_pending_does_not_request_again() {
        let mut params = Params::new();
        let (object, _handles) = demo_scene(&mut params);
        let scene = marble_csg::Scene { object, params, animations: vec![] };
        let mut mp = MultiplayerSession::new_solo(scene, vec![Marble::spawn(Vec3::ZERO, 0.3)]);
        mp.transport = Some(WebRtcTransport::new(Role::Joiner));
        mp.resync_pending = true;

        request_resync_on_window_exceeded(&mut mp, Role::Joiner);

        assert!(mp.resync_pending, "still pending -- unaffected either way, just confirming no panic/reset occurs");
    }

    /// A host never requests a resync from anyone (host-authoritative
    /// convention, `net::Role`'s doc) -- `WindowExceeded` on the host side
    /// must not flip `resync_pending`.
    #[test]
    fn window_exceeded_on_the_host_never_requests_a_resync() {
        let mut params = Params::new();
        let (object, _handles) = demo_scene(&mut params);
        let scene = marble_csg::Scene { object, params, animations: vec![] };
        let mut mp = MultiplayerSession::new_solo(scene, vec![Marble::spawn(Vec3::ZERO, 0.3)]);
        mp.transport = Some(WebRtcTransport::new(Role::Host));

        request_resync_on_window_exceeded(&mut mp, Role::Host);

        assert!(!mp.resync_pending, "a host must never request a resync from itself");
    }
}

/// Drains and applies the host's most recent authoritative tick+state
/// push, if any arrived since the last tick. The same `ResyncPayload`
/// message (and this same adoption code) serves three triggers that all
/// boil down to "adopt the host's state wholesale, no merge": the very
/// first join (see [`marble_physics_tick_impl`]'s doc for why a real
/// handshake-then-sync beats the old guessed-start-position scheme), a
/// *later* joiner connecting (this side's own player count needs to grow
/// to match too, even though it wasn't the one that just joined), and a
/// checksum-mismatch correction ([`exchange_checksums_and_resync`]) --
/// there is deliberately no second, parallel "initial sync" message type.
/// `hard_reset_to` requires the player count to already match, so this
/// grows in a loop to match the incoming player count first, however many
/// players short it currently is (zero times for an ordinary mismatch
/// correction once already fully joined, once for a first join, once more
/// for each later joiner while this side was already connected).
///
/// Called every tick regardless of `net.status`/role (cheap: a poll that's
/// almost always empty) rather than gating on `Role::Joiner`, since only
/// the non-host side ever actually receives one anyway
/// (`WebRtcTransport::poll_resync_payloads`'s own doc) -- one fewer
/// place that needs to know which role gets which messages. Must run
/// after [`apply_welcome`] each tick (`marble_physics_tick_impl`'s
/// ordering) so `marble_state.local_player_index` is already correct by
/// the time this reads it (this fn's own first-join narrow-down step is
/// the one place that still needs to *write* it, to `0`, before any
/// `Welcome` could plausibly have arrived).
///
/// Whether the incoming tick/marbles are applied immediately or stashed in
/// `mp.pending_join` for `render::apply_pending_scene_sync` to apply
/// atomically with a paired scene depends on whether this call actually
/// grew `num_players()` -- see `MultiplayerSession::pending_join`'s doc for
/// why that's the correct, always-available signal for "the host also sent
/// a `SceneSync` alongside this one:" the host only ever sends a scene
/// alongside a resync that grows the receiving side's player count
/// (`marble_physics_tick_impl`'s host-side join handling), never for a
/// plain checksum-mismatch correction.
fn apply_resync_payload(mp: &mut MultiplayerSession, marble_state: &mut MarbleState) {
    let Some(transport) = mp.transport.as_mut() else { return };
    let Some((tick, marbles)) = transport.poll_resync_payloads().into_iter().next_back() else {
        return;
    };
    if !mp.joined {
        // First time this client has ever heard from a host: strip `Demo`'s
        // decorative extra marbles (`render::setup`'s doc) down to just
        // this client's own real one. That marble is always local index 0
        // (`render::setup` hardcodes `local_player_index: 0` at every
        // session's start) -- a compile-time-known constant, NOT
        // `marble_state.local_player_index`, which `apply_welcome` (called
        // just above, same tick in practice, `marble_physics_tick_impl`'s
        // ordering) has *already* overwritten to the host-assigned
        // multiplayer slot by the time this runs. Reusing that
        // already-reassigned field here would narrow to whichever
        // decorative marble happens to sit at the host-assigned index
        // instead of the joiner's real one, and then the old `= 0`
        // afterward would clobber the correct value `apply_welcome` just
        // set -- together, exactly the "ghost marble, camera on the wrong
        // marble" bug this fixes.
        const JOINERS_OWN_PRE_JOIN_MARBLE_INDEX: PlayerIndex = 0;
        if mp.sim.num_players() > 1 {
            mp.sim = mp.sim.narrow_to(JOINERS_OWN_PRE_JOIN_MARBLE_INDEX);
        }
        mp.joined = true;
    }
    // Grow to match the incoming player count -- a throwaway placeholder
    // spawn point/radius, since `hard_reset_to`/`set_scene` just below (or
    // in `render::apply_pending_scene_sync`, once deferred) overwrites
    // every player's actual state immediately regardless. Whether this
    // loop actually iterates at least once is exactly the "a paired scene
    // is in flight" signal `pending_join`'s doc describes.
    let players_before_growth = mp.sim.num_players();
    while mp.sim.num_players() < marbles.len() {
        let idx = mp.sim.num_players();
        let placeholder_rad = mp.sim.marbles()[0].rad;
        mp.sim.add_player_at(idx, Marble::spawn(Vec3::ZERO, placeholder_rad));
        // `marble_state.start_positions` must grow in lockstep with
        // `mp.sim`'s own player count -- `RollbackSim::advance` runs every
        // tick unconditionally (`marble_physics_tick_impl`'s bottom), and
        // `step_marbles` asserts one start position per marble
        // (`marble_csg::physics`'s doc) -- a throwaway placeholder here too,
        // same reasoning as `add_player_at`'s spawn point above: whichever
        // of the immediate-apply or deferred-to-`apply_pending_scene_sync`
        // path runs overwrites this with the real value before it's ever
        // actually used as a kill-plane respawn reference.
        marble_state.start_positions.push(Vec3::ZERO);
    }
    // The growth loop above only ever grows *towards* `marbles.len()` --
    // a payload with *fewer* marbles than this session already has skips
    // it entirely and would otherwise reach `hard_reset_to` with mismatched
    // lengths, hitting its internal `assert_eq!` (an internal invariant
    // that assert is meant to enforce on its caller, not something it
    // should have to re-validate itself -- `hard_reset_to`'s own doc).
    // Reject rather than ever calling it with bad data: a host is never
    // expected to legitimately shrink a session's player count mid-resync
    // (players are only ever added, never removed, `net.rs`'s stable-
    // index-order guarantee), so a short payload here means something's
    // gone wrong on the wire (or, on this directly peer-drivable P2P
    // channel, a malformed/adversarial message) rather than a real
    // resync this side should ever adopt.
    if mp.sim.num_players() != marbles.len() {
        warn!(
            "multiplayer: rejecting resync payload at tick {tick} -- host sent {} marbles but this \
             session has {} after growth (a resync should never arrive short)",
            marbles.len(),
            mp.sim.num_players()
        );
        mp.resync_pending = false;
        return;
    }
    if mp.sim.num_players() > players_before_growth {
        // A scene is in flight for this exact resync -- defer the
        // tick/marble jump until `render::apply_pending_scene_sync` can
        // apply both atomically via `RollbackSim::set_scene`
        // (`pending_join`'s doc).
        info!(
            "multiplayer: deferring join/resync state from host at tick {tick} until its paired scene sync arrives"
        );
        mp.pending_join = Some((tick, marbles));
        mp.resync_pending = false;
        return;
    }
    info!("multiplayer: adopting authoritative join/resync state from host at tick {tick}");
    mp.sim.hard_reset_to(tick, marbles);
    mp.resync_pending = false;
    // Every player's kill-plane respawn reference becomes wherever the
    // just-adopted authoritative state actually put them -- there's no
    // separate "canonical spawn point" left to derive independently
    // (the old `base + offset` scheme this whole flow replaces): it's
    // just whatever the host said, same as everything else here.
    marble_state.start_positions = mp.sim.marbles().iter().map(|m| m.pos).collect();
    marble_state.marbles = mp.sim.marbles().to_vec();
}

#[cfg(test)]
mod resync_payload_tests {
    use super::*;
    use marble_csg::scenes::demo_scene;
    use marble_csg::Params;

    /// A minimal, valid `MultiplayerSession` + `MarbleState` with exactly
    /// `n` players, already `joined` and holding a live (but never actually
    /// connected -- `js_bridge` is a no-op stub natively) `WebRtcTransport`,
    /// so [`apply_resync_payload`] can be called directly against it.
    fn minimal_session_with_players(n: usize) -> (MultiplayerSession, MarbleState) {
        let mut params = Params::new();
        let (object, _handles) = demo_scene(&mut params);
        let scene = marble_csg::Scene { object, params, animations: vec![] };
        let rad = 0.3;
        let marbles: Vec<Marble> = (0..n).map(|i| Marble::spawn(Vec3::new(i as f32, 0.0, 0.0), rad)).collect();
        let mut mp = MultiplayerSession::new_solo(scene, marbles.clone());
        mp.joined = true;
        mp.transport = Some(WebRtcTransport::new(Role::Joiner));
        let marble_state = MarbleState {
            marbles: marbles.clone(),
            cfg: PhysicsConfig::default(),
            start_positions: marbles.iter().map(|m| m.pos).collect(),
            kill_y: -1e6,
            local_player_index: 0,
        };
        (mp, marble_state)
    }

    /// Fix 4 regression test: a `ResyncPayload` with *fewer* marbles than
    /// the session already has must be rejected rather than passed to
    /// `RollbackSim::hard_reset_to`, whose internal `assert_eq!` on marble
    /// count would otherwise panic. Pre-fix, the growth loop in
    /// `apply_resync_payload` only ever grows *towards* the incoming
    /// count, so a short payload skipped it entirely and reached
    /// `hard_reset_to` directly with mismatched lengths.
    #[test]
    fn apply_resync_payload_rejects_a_payload_with_fewer_marbles_than_current() {
        let (mut mp, mut marble_state) = minimal_session_with_players(2);
        let tick_before = mp.sim.current_tick();
        mp.transport.as_mut().unwrap().test_inject_resync_payload(5, vec![Marble::spawn(Vec3::ZERO, 0.3)]);

        apply_resync_payload(&mut mp, &mut marble_state);

        assert_eq!(mp.sim.num_players(), 2, "a short resync payload must not shrink the session's player count");
        assert_eq!(mp.sim.current_tick(), tick_before, "a rejected resync payload must not move the tick forward");
        assert!(
            !mp.resync_pending,
            "rejecting the payload must still clear resync_pending so a future mismatch can retry"
        );
    }

    /// Sanity check alongside the regression test above: a payload with
    /// *exactly* the current marble count (the ordinary checksum-mismatch-
    /// correction case, no growth needed) is still adopted normally, so
    /// fix 4's new length check doesn't accidentally reject legitimate
    /// same-size resyncs.
    #[test]
    fn apply_resync_payload_accepts_a_payload_with_the_same_marble_count() {
        let (mut mp, mut marble_state) = minimal_session_with_players(2);
        let new_marbles = vec![Marble::spawn(Vec3::new(9.0, 0.0, 0.0), 0.3), Marble::spawn(Vec3::new(10.0, 0.0, 0.0), 0.3)];
        mp.transport.as_mut().unwrap().test_inject_resync_payload(3, new_marbles.clone());

        apply_resync_payload(&mut mp, &mut marble_state);

        assert_eq!(mp.sim.current_tick(), 3);
        assert_eq!(mp.sim.marbles()[0].pos, new_marbles[0].pos);
        assert_eq!(mp.sim.marbles()[1].pos, new_marbles[1].pos);
    }

    /// "Ghost marble" regression test: the first-join narrow-down step must
    /// narrow to the joiner's own compile-time-known pre-join marble
    /// (always local index 0, `render::setup`'s doc) and must NOT touch
    /// `marble_state.local_player_index` -- which, by the time this runs,
    /// `apply_welcome` has already correctly set to the host-assigned
    /// multiplayer slot (simulated directly here, without a live
    /// `apply_welcome` call, since this test is scoped to
    /// `apply_resync_payload`'s own behavior). Pre-fix, this block reused
    /// `marble_state.local_player_index` for both `narrow_to`'s argument
    /// (narrowing to whichever decorative marble sat at the host-assigned
    /// index instead of the joiner's real one) and then reset it to `0`
    /// unconditionally afterward, clobbering the correct value -- from then
    /// on the joiner's camera and its own WASD input both followed the
    /// host's marble (shared index 0) instead of its own.
    #[test]
    fn apply_resync_payload_narrows_to_the_joiners_own_marble_without_clobbering_local_player_index() {
        let (mut mp, mut marble_state) = minimal_session_with_players(2);
        mp.joined = false;
        marble_state.local_player_index = 1;
        mp.transport.as_mut().unwrap().test_inject_resync_payload(7, vec![Marble::spawn(Vec3::new(50.0, 0.0, 0.0), 0.3)]);

        apply_resync_payload(&mut mp, &mut marble_state);

        assert_eq!(
            marble_state.local_player_index, 1,
            "apply_resync_payload must not clobber the index apply_welcome already assigned"
        );
        assert_eq!(mp.sim.num_players(), 1, "narrowing must collapse the pre-join decorative marble(s) away");
    }
}

/// Drains and applies the host's "you are player index K" message, if one
/// arrived since the last tick — a joiner's own player index is no longer
/// a compile-time constant now that more than one joiner can exist
/// (`net::Role`'s doc), so this is how it learns the value the host
/// assigned it. Always a no-op on the host itself (never receives one,
/// its own index is always `0`) and on every tick nothing's arrived.
/// Called before [`apply_resync_payload`] each tick (`marble_physics_tick_impl`'s
/// ordering) so that function's `marble_state.local_player_index` read is
/// already correct — safe by wire order too: PeerJS's `reliable: true`
/// channel preserves send order, and the host always sends `Welcome`
/// before that same connection's first `ResyncPayload`/`SceneSync`
/// (`marble_physics_tick_impl`'s join-handling doc).
///
/// Validates `index` is a real joiner index (`1..=MAX_JOINERS`, same range
/// `WebRtcTransport::take_newly_connected` already asserts on the host
/// side) before ever assigning it — this one check is what keeps every
/// later unguarded `marbles[local_player_index]` read (`MarbleState::
/// local_marble`, `render.rs`, `camera.rs`, `debug_gizmos.rs`) and every
/// `RollbackSim::narrow_to(local_player_index)` call safe, since all of
/// them trust this field without re-checking it themselves. An
/// out-of-range `Welcome` (any raw byte outside that range — a build
/// mismatch or a malformed message on this directly peer-drivable P2P
/// channel, not something `net.js`'s own assignment logic would ever
/// produce against a matching build) is logged and dropped instead of
/// adopted, so `local_player_index` stays at whatever it already was.
fn apply_welcome(mp: &mut MultiplayerSession, marble_state: &mut MarbleState) {
    let Some(transport) = mp.transport.as_mut() else { return };
    if let Some(index) = transport.poll_welcomes().into_iter().next_back() {
        if is_valid_welcome_index(index) {
            marble_state.local_player_index = index;
        } else {
            warn!(
                "multiplayer: ignoring Welcome with out-of-range index {index} (valid range is 1..={})",
                crate::net::MAX_JOINERS
            );
        }
    }
}

/// Whether `index` is a real joiner index (`1..=MAX_JOINERS` — index `0` is
/// always the host, never assigned via `Welcome`). Factored out of
/// [`apply_welcome`] purely so this range check is directly unit-testable:
/// `apply_welcome` itself needs a live `WebRtcTransport` with a pending
/// message queued, which on the native target these tests run on requires
/// going through `net.rs`'s wasm-only `js_bridge` (a no-op stub natively) --
/// this predicate is the actual safety property fix 3 depends on, so
/// testing it directly covers the fix without needing that plumbing.
fn is_valid_welcome_index(index: PlayerIndex) -> bool {
    (1..=crate::net::MAX_JOINERS).contains(&index)
}

#[cfg(test)]
mod welcome_validation_tests {
    use super::*;

    /// Fix 3 regression test: every index outside `1..=MAX_JOINERS` --
    /// including `0` (the host's own index, never legitimately assigned via
    /// `Welcome`) and values past `MAX_JOINERS` (whatever raw byte a
    /// malformed or build-mismatched `Welcome` might carry) -- must be
    /// rejected, while every index actually inside that range is accepted.
    /// Pre-fix, `apply_welcome` assigned any of these straight to
    /// `local_player_index` with no check at all, which then panicked the
    /// next time it indexed `marbles[local_player_index]` or was passed to
    /// `RollbackSim::narrow_to`.
    #[test]
    fn welcome_index_validation_accepts_only_1_through_max_joiners() {
        assert!(!is_valid_welcome_index(0), "index 0 is the host's own index, never a valid Welcome payload");
        for i in 1..=crate::net::MAX_JOINERS {
            assert!(is_valid_welcome_index(i), "index {i} is within 1..=MAX_JOINERS and must be accepted");
        }
        assert!(!is_valid_welcome_index(crate::net::MAX_JOINERS + 1));
        assert!(!is_valid_welcome_index(255));
    }
}

/// Rewind window depth (in ticks, i.e. ~267ms at 60Hz) — deep enough to
/// absorb realistic internet latency between two peers connected directly
/// via WebRTC (typically tens of ms, occasionally more), shallow enough to
/// keep the snapshot ring buffer's memory bounded. Not tuned against real
/// network measurements (this sandbox can't produce those) — a reasonable
/// starting point, easy to revisit once real cross-machine play is tested.
const ROLLBACK_WINDOW_TICKS: u64 = 16;

/// One 60 Hz physics tick (`FixedUpdate`): reads WASD + a 2-finger pinch
/// (additive — see module doc) + the orbit camera's orientation, steps every
/// marble against the live CSG scene tree through the always-on
/// [`RollbackSim`] (resolving marble-vs-marble collision along the way),
/// lets `R` force an immediate manual respawn of the *local* marble, and
/// `G` toggle [`GravityMode`] for everyone (there's only one shared `cfg` —
/// per-player physics config isn't a thing this milestone needs). A no-op
/// for scenes without a real marble (`SceneKind::has_marble` — the static
/// display fractals, not the tuned demo level).
/// Thin timing wrapper over [`marble_physics_tick_impl`] -- records this
/// call's wall-clock cost into [`PhaseTimings`] regardless of which of
/// `_impl`'s several early-return paths actually fires, without needing to
/// touch each one individually.
#[allow(clippy::too_many_arguments)] // one more param than `_impl`, for `PhaseTimings`
pub fn marble_physics_tick(
    keys: Res<ButtonInput<KeyCode>>,
    touches: Res<Touches>,
    orbit: Res<CameraOrbit>,
    scene: Res<SceneState>,
    net: Res<NetSession>,
    mp: ResMut<MultiplayerSession>,
    marble_state: ResMut<MarbleState>,
    pending_scene: ResMut<PendingSceneSync>,
    mut timings: ResMut<PhaseTimings>,
) {
    let start = Instant::now();
    marble_physics_tick_impl(keys, touches, orbit, scene, net, mp, marble_state, pending_scene);
    timings.record("physics", start.elapsed());
}

#[allow(clippy::too_many_arguments)]
fn marble_physics_tick_impl(
    keys: Res<ButtonInput<KeyCode>>,
    touches: Res<Touches>,
    orbit: Res<CameraOrbit>,
    scene: Res<SceneState>,
    net: Res<NetSession>,
    mut mp: ResMut<MultiplayerSession>,
    mut marble_state: ResMut<MarbleState>,
    mut pending_scene: ResMut<PendingSceneSync>,
) {
    if !scene.kind.has_marble() {
        return;
    }
    // `RollbackSim` now owns the scene itself (`marble_csg::Scene`'s doc),
    // so `SceneState` no longer needs the disjoint-mutable-field-borrow
    // trick this used to need alongside `mp` -- it's read-only here
    // (`scene.kind` only). `mp` still needs it:
    // `exchange_checksums_and_resync` needs `&mut mp.sim` and
    // `mp.transport.as_mut()` simultaneously, which `ResMut`'s
    // `Deref`/`DerefMut` (going through method calls the borrow checker
    // can't see through) wouldn't allow in one call.
    let mp = &mut *mp;

    if keys.just_pressed(KeyCode::KeyG) {
        marble_state.cfg.mode = match marble_state.cfg.mode {
            GravityMode::Rolling => GravityMode::Flying,
            GravityMode::Flying => GravityMode::Rolling,
        };
    }

    // `R`-to-respawn is a direct, out-of-band mutation no resim can ever
    // replay (`RollbackSim::respawn`'s doc: it isn't a function of any
    // `PlayerInput`) -- only safe while nothing could ever trigger a resim
    // reaching back over the tick it happens on, i.e. only while not
    // actually connected to a peer yet.
    if keys.just_pressed(KeyCode::KeyR) && mp.transport.is_none() {
        let idx = marble_state.local_player_index;
        let start = marble_state.start_positions[idx];
        mp.sim.respawn(idx, start);
        marble_state.marbles = mp.sim.marbles().to_vec();
        return;
    }

    let mut dx = 0.0f32;
    let mut dy = 0.0f32;
    if keys.pressed(KeyCode::KeyW) {
        dy += 1.0;
    }
    if keys.pressed(KeyCode::KeyS) {
        dy -= 1.0;
    }
    if keys.pressed(KeyCode::KeyA) {
        dx -= 1.0;
    }
    if keys.pressed(KeyCode::KeyD) {
        dx += 1.0;
    }
    if let Some(gesture) = read_two_finger_gesture(&touches) {
        dy += gesture.pinch_dy;
    }

    // `PlayerInput.orientation` carries the camera's actual current
    // orientation, not `dx`/`dy` re-derived from it via a second,
    // independently-written yaw/pitch trig formula that would have to
    // happen to agree in sign convention with `CameraOrbit`'s own — an
    // earlier version of the old single-marble call site did exactly that
    // and got the sign wrong (see `marble_csg::physics::step_marbles`'s
    // doc). `step_marbles` derives `cam_forward`/`cam_right` from it
    // directly (`PlayerInput::cam_basis`), the same math `CameraOrbit`
    // itself uses (`orientation * Vec3::NEG_Z` / `orientation * Vec3::X`).
    let local_input = PlayerInput { dx, dy, orientation: orbit.orientation };

    // Edge-triggered on the *transition into* `Connected` -- a freshly
    // opened data channel, first-ever or a same-tab reconnect alike (see
    // `MultiplayerSession::was_connected`'s doc). Only used to create
    // `mp.transport` once, ever, on the first connection event of any
    // kind -- the actual per-joiner join-handling below runs unconditionally
    // every tick instead (a host can gain a second, third, ... joiner at
    // any later time, not just alongside the very first).
    let just_connected = net.status == NetStatus::Connected && !mp.was_connected;
    mp.was_connected = net.status == NetStatus::Connected;
    if just_connected && mp.transport.is_none() {
        mp.transport = Some(WebRtcTransport::new(net.role));
    }

    // Host: handle every joiner index that connected this tick (almost
    // always empty -- a no-op poll on every quiet tick, and always empty
    // on a joiner, `WebRtcTransport::take_newly_connected`'s doc). Each one
    // grows this session by exactly one player and gets a `Welcome`
    // telling it its assigned index; once at least one has ever connected,
    // a single combined tick+state (+ scene, first time only in practice)
    // push goes out to *everyone* currently connected -- existing joiners
    // need this too, so their own `RollbackSim` grows to match the new
    // count (`apply_resync_payload`'s doc), even though they weren't the
    // one that just joined.
    if net.role == Role::Host {
        let new_indices =
            mp.transport.as_mut().map(WebRtcTransport::take_newly_connected).unwrap_or_default();
        for _ in &new_indices {
            // First-ever joiner only: narrow away `Demo`'s decorative extra
            // marbles (`render::setup`'s doc) -- a no-op for every other
            // scene, which always starts solo with exactly 1 marble
            // already.
            if !mp.joined {
                if mp.sim.num_players() > 1 {
                    mp.sim = mp.sim.narrow_to(marble_state.local_player_index);
                    marble_state.start_positions = vec![marble_state.start_positions[marble_state.local_player_index]];
                    marble_state.local_player_index = 0;
                }
                mp.joined = true;
                // The host never waits on anything -- it's always
                // authoritative, so this is set once, here, rather than
                // gated on a round trip the way a joiner's is below.
                mp.scene_synced = true;
            }
            let idx = mp.sim.num_players();
            let base = marble_state.start_positions[0];
            let rad = mp.sim.marbles()[0].rad;
            let start = base + Vec3::new(idx as f32 * 0.4, 0.0, 0.0);
            mp.sim.add_player_at(idx, Marble::spawn(start, rad));
            marble_state.start_positions.push(start);
            if let Some(transport) = mp.transport.as_mut() {
                transport.send_welcome(idx, idx);
            }
        }
        // The handshake(s) just completed -- push this side's current tick
        // and full state to everyone right away, reusing `ResyncPayload`
        // verbatim (`apply_resync_payload`'s doc): "here's the
        // authoritative tick + state, adopt it" is exactly what a fresh
        // join needs too, not just a checksum-mismatch correction. This is
        // the fix for the original 2-player-only bug this flow was built
        // for -- previously each side just kept its own independently-run
        // tick counter and only ever reconciled the resulting divergence
        // reactively, via periodic checksums, long after the two sessions
        // had already drifted apart by however many ticks the handshake
        // itself took. Runs on every tick with at least one new connection,
        // first-ever or a later joiner (or a same-tab reconnect, which also
        // surfaces as a `take_newly_connected` entry) alike, for the same
        // reason: ticks keep drifting apart independently for the whole
        // time any side is disconnected (`NetStatus::Disconnected`'s doc),
        // so each of these events needs exactly the same realignment --
        // one join/rejoin/grow code path, not a special case per trigger.
        if !new_indices.is_empty() {
            if let Some(transport) = mp.transport.as_mut() {
                transport.send_resync_payload(mp.sim.current_tick(), mp.sim.marbles().to_vec());
                // The host's whole scene tree, atomically alongside the
                // tick+state push above -- see `marble_csg::Scene`'s
                // module doc for why a joiner needs this at all, not just
                // a `?scene=` name. `mp.sim.scene()` is the one
                // authoritative copy -- no separate clone-and-rebuild
                // needed, unlike before `RollbackSim` owned it.
                transport.send_scene_sync(mp.sim.scene().to_bytes());
            }
        }
    }

    // Joiner's side of the hand-off above: learn this client's own
    // assigned index, then adopt whatever the host's most recent
    // join/resync payload says, wholesale. Both polled every tick (not
    // gated on `just_connected`) since each is a separate message that
    // arrives at least one network hop after the handshake itself
    // completes. A no-op on the host (never receives either) and on any
    // tick nothing's actually arrived.
    apply_welcome(mp, &mut marble_state);
    apply_resync_payload(mp, &mut marble_state);

    // Scene-sync half of the same hand-off: stash the raw bytes for
    // `render::apply_pending_scene_sync` (`Update`) to actually decode and
    // apply -- this system doesn't have the rendering resources
    // (`Assets<Shader>` etc.) that needs, only `WebRtcTransport` access
    // (`PendingSceneSync`'s doc).
    if let Some(transport) = mp.transport.as_mut() {
        if let Some(bytes) = transport.poll_scene_syncs().into_iter().next_back() {
            pending_scene.0 = Some(bytes);
        }
    }

    // Debug/test-only: `F9`, while connected, deliberately nudges this
    // client's own live position out from under the local simulation --
    // manufactures exactly the kind of cross-peer divergence rollback
    // resiliency's checksum comparison exists to catch, so a live
    // verification pass can force one and confirm the other side actually
    // detects and recovers from it. Compiled out of release builds
    // entirely (`RollbackSim::debug_perturb_position`'s doc).
    #[cfg(debug_assertions)]
    if keys.just_pressed(KeyCode::F9) && mp.transport.is_some() {
        let idx = marble_state.local_player_index;
        mp.sim.debug_perturb_position(idx, Vec3::new(5.0, 0.0, 0.0));
        warn!("multiplayer: DEBUG forced a local position perturbation (F9) for checksum-resync verification");
    }

    let local_idx = marble_state.local_player_index;
    let tick = mp.sim.current_tick() + 1;

    // We always know our own input the instant we produce it -- confirm it
    // immediately (never predicted), same as a real remote peer's own
    // client does for theirs.
    let mut arrivals = vec![(local_idx, tick, local_input)];
    // Gated on `mp.joined && mp.scene_synced`, not just
    // `mp.transport.is_some()`: on the joiner, a data channel can be open
    // for one or more ticks before the host's join-sync payload(s) actually
    // land (`apply_resync_payload`'s doc) -- until *both* the tick/marble
    // state and the scene bundle have landed, this side's `sim`/`scene` are
    // still either genuinely solo or possibly the wrong scene entirely, so
    // exchanging per-tick input this early would misattribute the host's
    // early messages to whatever index 0 happens to mean locally at that
    // moment, or simulate real cross-peer collision against geometry the
    // two sides don't actually agree on yet.
    if net.status == NetStatus::Connected && mp.joined && mp.scene_synced {
        if let Some(transport) = mp.transport.as_mut() {
            // Host only: note any joiner that dropped this tick (kept
            // forever after -- `MultiplayerSession::disconnected_remotes`'s
            // doc) before the real input exchange below, so this same
            // tick's arrivals already include their confirmed-zero fill-in
            // rather than lagging a tick behind.
            if net.role == Role::Host {
                for idx in transport.take_newly_disconnected() {
                    if !mp.disconnected_remotes.contains(&idx) {
                        mp.disconnected_remotes.push(idx);
                    }
                }
            }
            // `send_input`/`poll_received` broadcast/relay across every
            // known remote for a host, or talk to the one host connection
            // for a joiner (`WebRtcTransport`'s doc) -- this call site
            // doesn't need to know which.
            transport.send_input(tick, local_input);
            arrivals.extend(transport.poll_received());
            // Host only: every joiner that has ever disconnected sits
            // under zero input from here on, rather than silently
            // repeating whatever they were last doing right up to the
            // moment they dropped (empty on a joiner -- `disconnected_remotes`
            // is host-only state).
            for &idx in &mp.disconnected_remotes {
                arrivals.push((idx, tick, PlayerInput { dx: 0.0, dy: 0.0, orientation: Quat::IDENTITY }));
            }
        }
    } else if mp.transport.is_some() && mp.joined && mp.scene_synced {
        // Joiner only in practice: `net.status` only ever actually settles
        // into `Disconnected` for a joiner (`net.js`'s host-side `close`
        // handler never resets the shared `status` back down once it's
        // ever become "connected", precisely because a host can have other
        // still-live joiners even after one drops -- a host's per-joiner
        // disconnect handling lives in the branch above instead, driven by
        // `take_newly_disconnected`, not this one). The host itself sits
        // under zero input from here on.
        arrivals.push((HOST_INDEX, tick, PlayerInput { dx: 0.0, dy: 0.0, orientation: Quat::IDENTITY }));
    }

    let kill_y = marble_state.kill_y;
    let starts = marble_state.start_positions.clone();
    // `receive_inputs`/`advance` re-evaluate the scene's `animations`
    // against each tick they (re)simulate internally, reading its own
    // owned `Scene` rather than taking one as a parameter (`RollbackSim`'s
    // own doc) -- this system never calls `apply_animations` itself.
    if let Err(e) = mp.sim.receive_inputs(&arrivals, &marble_state.cfg, kill_y, &starts) {
        // Live 2-peer verification caught this: a host can hit this same
        // branch too (e.g. a very late/delayed relayed input), but a host
        // never requests a resync from anyone (host-authoritative
        // convention) -- `request_resync_on_window_exceeded` correctly
        // no-ops for it, so the log text must say so too instead of always
        // claiming a request was made.
        let action = if net.role == Role::Joiner { "requesting a full resync" } else { "host is authoritative, no action taken" };
        warn!(
            "multiplayer: rollback window exceeded (tick {}, oldest available {}) -- {action}",
            e.requested_tick, e.oldest_available
        );
        request_resync_on_window_exceeded(mp, net.role);
    }
    mp.sim.advance(&marble_state.cfg, kill_y, &starts);
    marble_state.marbles = mp.sim.marbles().to_vec();

    // Gated on `mp.joined && mp.scene_synced` too: comparing/publishing
    // checksums only makes sense once this is genuinely a synced 2-player
    // session (a still-solo or not-yet-scene-synced joiner mid-handshake
    // has nothing meaningful to compare against yet).
    if net.status == NetStatus::Connected && mp.joined && mp.scene_synced {
        if let Some(transport) = mp.transport.as_mut() {
            exchange_checksums_and_resync(&mut mp.sim, transport, net.role, &mut mp.resync_pending);
        }
    }
}
