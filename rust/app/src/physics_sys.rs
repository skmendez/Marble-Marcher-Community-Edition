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
use marble_csg::rollback::{InputTransport, RollbackSim};

use crate::camera::CameraOrbit;
use crate::net::{NetSession, NetStatus, Role, WebRtcTransport};
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
/// Known, deliberately-unaddressed limitation (per the multiplayer plan:
/// "graceful reconnect is a further refinement, not required"): once
/// `NetSession::status` leaves `NetStatus::Connected` (peer disconnected),
/// this system keeps driving `sim` exactly as before, just confirming a
/// zero input for the departed player from then on (same idea as
/// `default_input`, made explicit — see the "was connected, isn't now"
/// branch below) instead of leaving them repeating whatever they were last
/// doing forever. If the *same* peer later reconnects, `NetStatus` moving
/// back to `Connected` resumes exchanging real input on the *same* `sim`
/// (nothing was ever torn down) rather than rebuilding — simpler and more
/// correct than a lazy-init design's reconnect story, though still not a
/// full resync from a peer's own state if the two have actually diverged
/// during the gap.
///
/// **Rollback resiliency** (periodic state checksums + resync-on-mismatch,
/// same idea Photon Quantum/Factorio-style lockstep engines ship): while
/// connected, [`marble_physics_tick`] also periodically exchanges a
/// checksum of each side's state at whatever tick is currently eligible
/// for comparison (`RollbackSim::latest_checksum_tick`) and, on a
/// mismatch, has the non-host side pull a full authoritative correction
/// from the host (`RollbackSim::hard_reset_to`) — this is the "actually
/// diverged" case the paragraph above calls out as unaddressed by the
/// plain reconnect story, now handled.
#[derive(Resource)]
pub struct MultiplayerSession {
    pub sim: RollbackSim,
    /// `Some` only once actually connected to a remote peer at least once
    /// — absent, this system never polls for or sends anything over the
    /// network.
    transport: Option<WebRtcTransport>,
    /// Set once a checksum mismatch has triggered a `ResyncRequest`, until
    /// the matching `ResyncPayload` actually lands and `hard_reset_to`
    /// runs -- without this, a mismatch that's still visible on
    /// subsequent ticks (the request/response round trip takes at least
    /// one network hop each way) would otherwise re-request every single
    /// tick in between.
    resync_pending: bool,
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
    pub fn new_solo(initial_marbles: Vec<Marble>) -> Self {
        Self {
            sim: RollbackSim::new(
                initial_marbles,
                PlayerInput { dx: 0.0, dy: 0.0, orientation: Quat::IDENTITY },
                ROLLBACK_WINDOW_TICKS,
            ),
            transport: None,
            resync_pending: false,
        }
    }
}

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
/// mismatch is actually detected.
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

    // Accept the most recent resync payload the host has sent, if any --
    // unconditionally overwriting local state. This is the one thing that
    // actually corrects a detected divergence.
    if let Some((tick, marbles)) = transport.poll_resync_payloads().into_iter().next_back() {
        info!("multiplayer: applying full-state resync from host at tick {tick}");
        sim.hard_reset_to(tick, marbles);
        *resync_pending = false;
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
    scene: ResMut<SceneState>,
    net: Res<NetSession>,
    mp: ResMut<MultiplayerSession>,
    marble_state: ResMut<MarbleState>,
    mut timings: ResMut<PhaseTimings>,
) {
    let start = Instant::now();
    marble_physics_tick_impl(keys, touches, orbit, scene, net, mp, marble_state);
    timings.record("physics", start.elapsed());
}

fn marble_physics_tick_impl(
    keys: Res<ButtonInput<KeyCode>>,
    touches: Res<Touches>,
    orbit: Res<CameraOrbit>,
    mut scene: ResMut<SceneState>,
    net: Res<NetSession>,
    mut mp: ResMut<MultiplayerSession>,
    mut marble_state: ResMut<MarbleState>,
) {
    if !scene.kind.has_marble() {
        return;
    }
    // A real `&mut SceneState`/`&mut MultiplayerSession` (not `ResMut`'s
    // `Deref`/`DerefMut`, which go through method calls the borrow checker
    // can't see through) so their individual fields below can be borrowed
    // as the disjoint fields they actually are, in one call --
    // `exchange_checksums_and_resync` needs `&mut mp.sim` and
    // `mp.transport.as_mut()` simultaneously, same reasoning as `scene`.
    let scene = &mut *scene;
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

    // Newly connected this tick: grow the *existing* session instead of
    // discarding it (`MultiplayerSession`'s doc).
    if net.status == NetStatus::Connected && mp.transport.is_none() {
        // Multiplayer is always exactly 2 *real* players (`net::Role`'s
        // doc) -- narrow away any non-real extra marbles first (`Demo`'s
        // decorative ones, `render::setup`'s doc); a no-op for every other
        // scene, which always starts solo with exactly 1 marble already.
        if mp.sim.num_players() > 1 {
            mp.sim = mp.sim.narrow_to(marble_state.local_player_index);
            marble_state.start_positions = vec![marble_state.start_positions[marble_state.local_player_index]];
            marble_state.local_player_index = 0;
        }

        // The remote player's starting position has to be a value *both*
        // sides independently compute identically, without knowing
        // anything real about each other yet (no state-sync protocol
        // exists -- see this fork's report on `net.rs`) -- deriving it
        // from `start_positions.get(remote_index)` (an earlier version of
        // this code) is wrong: for the joiner, `remote_index` is `0`, which
        // *already has an entry* pre-connection (the joiner's own solo
        // start, not the host's), so that lookup would silently reuse the
        // joiner's own position for where it thinks the host is standing.
        // Instead, both sides derive the same canonical `base` (this
        // scene's spawn point, identical on both clients since it comes
        // from the same `?scene=`) and offset purely by `remote_index`
        // itself (not by "am I the host or the joiner"), so a fresh host
        // inserting the joiner at `1` and a fresh joiner inserting the
        // host at `0` agree exactly: the host's own real, unmodified
        // position is `base` (offset 0), and that's exactly what the
        // joiner's `remote_index = 0` guess also computes for it.
        let remote_index = net.role.remote_index();
        let base = marble_state.start_positions[marble_state.local_player_index];
        let rad = mp.sim.marbles()[marble_state.local_player_index].rad;
        let remote_start = base + Vec3::new(remote_index as f32 * 0.4, 0.0, 0.0);
        mp.sim.add_player_at(remote_index, Marble::spawn(remote_start, rad));
        marble_state.start_positions.insert(remote_index, remote_start);
        marble_state.local_player_index = net.role.local_index();
        mp.transport = Some(WebRtcTransport::new(net.role));
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
    if net.status == NetStatus::Connected {
        if let Some(transport) = mp.transport.as_mut() {
            transport.send_input(tick, local_input);
            arrivals.extend(transport.poll_received());
        }
    } else if mp.transport.is_some() {
        // Was connected, isn't anymore -- the departed peer's marble sits
        // under zero input from here on (`MultiplayerSession`'s doc),
        // rather than silently repeating whatever they were last doing
        // right up to the moment they dropped.
        let remote_idx = net.role.remote_index();
        arrivals.push((remote_idx, tick, PlayerInput { dx: 0.0, dy: 0.0, orientation: Quat::IDENTITY }));
    }

    let kill_y = marble_state.kill_y;
    let starts = marble_state.start_positions.clone();
    // `receive_inputs`/`advance` re-evaluate `scene.animations` against
    // each tick they (re)simulate internally (`RollbackSim`'s own doc) --
    // this system never calls `apply_animations` itself.
    if let Err(e) = mp.sim.receive_inputs(
        &arrivals,
        &scene.object,
        &mut scene.params,
        &scene.animations,
        &marble_state.cfg,
        kill_y,
        &starts,
    ) {
        // Network conditions outran the rewind window. Milestone 2's
        // minimum bar (per the plan) is "don't crash or corrupt the
        // simulation" -- log and keep going with whatever state the
        // sim already has, rather than panicking. A real fix needs a
        // full state resync from the peer, out of scope here.
        warn!(
            "multiplayer: rollback window exceeded (tick {}, oldest available {}) -- \
             possible desync until the next full resync",
            e.requested_tick, e.oldest_available
        );
    }
    mp.sim.advance(&scene.object, &mut scene.params, &scene.animations, &marble_state.cfg, kill_y, &starts);
    marble_state.marbles = mp.sim.marbles().to_vec();

    if net.status == NetStatus::Connected {
        if let Some(transport) = mp.transport.as_mut() {
            exchange_checksums_and_resync(&mut mp.sim, transport, net.role, &mut mp.resync_pending);
            // `exchange_checksums_and_resync` may have just replaced
            // `mp.sim`'s live state wholesale (`hard_reset_to`) -- refresh
            // the read-only mirror so it can't go a whole extra frame
            // showing stale, pre-resync marble state.
            marble_state.marbles = mp.sim.marbles().to_vec();
        }
    }
}
