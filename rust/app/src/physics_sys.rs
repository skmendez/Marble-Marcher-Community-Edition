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
//! WASD sign convention: setting `dy = -1` for W moves the marble *toward*
//! the camera's eye position; S (`dy = +1`) moves it away, deeper along the
//! view direction. Verified empirically (not just derived on paper) by
//! logging `marble.pos` before/after holding each key against a live build
//! and checking its displacement's dot product against the camera's
//! `forward` vector — an earlier version of this comment claimed the
//! opposite ("W rolls the marble away from the orbiting camera... S rolls
//! it back toward the camera"), derived purely algebraically from
//! `CameraOrbit`'s basis and never actually checked against a running
//! build; that derivation had a sign error, which is exactly why the touch
//! pinch gesture built on top of it (below) felt backwards to begin with.
//! Setting `dx = +1` for D yields `+(cos(yaw), 0, -sin(yaw))`, which is
//! exactly `CameraOrbit`'s `right` vector at zero pitch — D rolls the
//! marble in the camera's screen-right direction (unaffected by the above
//! correction). In `Flying` mode the same `dx`/`dy` inputs instead drive
//! full 3D thrust along wherever the camera is actually pointed (see
//! `step_marble`'s doc).
//!
//! Touch: a 2-finger pinch feeds an additional `dy` (on top of WASD's) via
//! `touch::read_two_finger_gesture` — pinching in pulls the marble toward
//! the camera (W-equivalent), pinching out pushes it away (S-equivalent).
//! Read directly here (not via an `Update`-schedule intermediary) for the
//! same reason WASD is: `Touches`, like `ButtonInput<KeyCode>`, is input
//! state readable from any schedule, not something that needs per-frame
//! accumulation across schedules. The gesture's *rotate* half is handled
//! separately in `touch::touch_camera_input` (`Update` schedule, alongside
//! the mouse-driven `orbit_camera_input`), since it drives `CameraOrbit`,
//! not the marble.

use bevy::input::touch::Touches;
use bevy::prelude::*;

use marble_csg::physics::{step_marbles, GravityMode, Marble, PhysicsConfig, PlayerInput};
use marble_csg::rollback::{InputTransport, RollbackSim, Tick};

use crate::camera::CameraOrbit;
use crate::net::{NetSession, NetStatus, WebRtcTransport};
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
/// Multiplayer milestone 0: was a single `Marble` before this — `marbles`/
/// `start_positions` are parallel `Vec`s (same index = same marble,
/// [`step_marbles`]'s own convention) rather than a `HashMap` keyed by some
/// player id, since stable index order is what determinism (a future
/// rollback engine's whole point) depends on.
#[derive(Resource)]
pub struct MarbleState {
    pub marbles: Vec<Marble>,
    pub cfg: PhysicsConfig,
    pub start_positions: Vec<Vec3>,
    pub kill_y: f32,
    /// Which `marbles` index this client controls/watches -- always `0`
    /// today (no networking exists yet), named for milestone 2's benefit
    /// rather than assumed to always be zero everywhere it's read.
    pub local_player_index: usize,
}

impl MarbleState {
    /// The marble this client controls/watches -- what the camera follows,
    /// what WASD/touch input drives, what `R` respawns.
    pub fn local_marble(&self) -> Marble {
        self.marbles[self.local_player_index]
    }
}

/// Multiplayer milestone 2: the live 2-player [`RollbackSim`] + its
/// [`WebRtcTransport`], once a peer actually connects — `None` until then,
/// so single-player behavior (the direct `step_marbles` path below) is
/// completely unaffected by any of this existing. Lives in its own
/// resource rather than folded into [`MarbleState`] since `RollbackSim`
/// owns its *own* copy of the marble list (needed to snapshot/rewind) —
/// `MarbleState::marbles` becomes just a read-only mirror of
/// `RollbackSim::marbles()`, refreshed every tick, once this exists.
///
/// Known, deliberately-unaddressed limitation (per the multiplayer plan:
/// "graceful reconnect is a further refinement, not required"): once
/// `NetSession::status` leaves `NetStatus::Connected` (peer disconnected),
/// `marble_physics_tick` falls back to the plain local `step_marbles` path,
/// still driving `MarbleState::marbles` (whatever 2-marble state the
/// rollback session left behind) — the departed player's marble just sits
/// wherever it last was, driven by zero input, exactly like milestone 0's
/// original "placeholder marbles" behavior. If the *same* peer later
/// reconnects, this resource (still holding the pre-disconnect
/// `RollbackSim`) resumes from where it left off rather than resetting —
/// since `MarbleState::marbles` kept moving locally during the gap while
/// `RollbackSim`'s internal state didn't, resuming can produce a visible
/// position "snap" back to the rollback session's last known state. Rare in
/// practice (this milestone's join flow doesn't support reconnecting to a
/// *different* link anyway) and explicitly out of scope to fully reconcile.
#[derive(Resource, Default)]
pub struct MultiplayerSession {
    live: Option<LiveSession>,
}

struct LiveSession {
    sim: RollbackSim,
    transport: WebRtcTransport,
    /// Rollback's own tick counter (`marble_csg::rollback::Tick`), *not*
    /// wall-clock time or this app's `FixedUpdate` frame count directly —
    /// happens to line up 1:1 with `FixedUpdate` calls in practice (this
    /// system only ever increments it once per call), but keeping it as
    /// its own field rather than reading it back out of `sim` every time
    /// keeps "whose tick counter is authoritative" unambiguous.
    tick: Tick,
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
/// marble against the live CSG scene tree (`step_marbles`, resolving
/// marble-vs-marble collision along the way), lets `R` force an immediate
/// manual respawn of the *local* marble, and `G` toggle [`GravityMode`] for
/// everyone (there's only one shared `cfg` — per-player physics config isn't
/// a thing this milestone needs). A no-op for scenes without a real marble
/// (`SceneKind::has_marble` — the static display fractals, not the tuned
/// demo level).
pub fn marble_physics_tick(
    keys: Res<ButtonInput<KeyCode>>,
    touches: Res<Touches>,
    orbit: Res<CameraOrbit>,
    scene: Res<SceneState>,
    net: Res<NetSession>,
    mut mp: ResMut<MultiplayerSession>,
    mut marble_state: ResMut<MarbleState>,
) {
    if !scene.kind.has_marble() {
        return;
    }

    if keys.just_pressed(KeyCode::KeyG) {
        marble_state.cfg.mode = match marble_state.cfg.mode {
            GravityMode::Rolling => GravityMode::Flying,
            GravityMode::Flying => GravityMode::Rolling,
        };
    }

    // `R`-to-respawn is a direct, out-of-band mutation of marble state --
    // fine for the offline path (nothing else needs to agree on it), but
    // NOT safe to allow while a `RollbackSim` is live: it owns its own
    // marble state independently and isn't aware of this mutation, so
    // `marble_state.marbles` (a plain mirror once connected, see
    // `MultiplayerSession`'s doc) would get silently overwritten back to
    // the pre-respawn position on the very next tick, and the two peers'
    // simulations would disagree about whether a respawn ever happened at
    // all -- a real desync, not just a visual glitch. Disabled while
    // networked rather than half-implemented; a real fix would need
    // respawn to become an actual networked, deterministic input (a button
    // bit on `PlayerInput`, replayable like everything else), which is
    // future work.
    if keys.just_pressed(KeyCode::KeyR) && mp.live.is_none() {
        let idx = marble_state.local_player_index;
        let start = marble_state.start_positions[idx];
        marble_state.marbles[idx].respawn(start);
        return;
    }

    let mut dx = 0.0f32;
    let mut dy = 0.0f32;
    if keys.pressed(KeyCode::KeyW) {
        dy -= 1.0;
    }
    if keys.pressed(KeyCode::KeyS) {
        dy += 1.0;
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

    // Multiplayer milestone 2: once a peer is actually connected, drive
    // the session through `RollbackSim` instead of calling `step_marbles`
    // directly -- see `MultiplayerSession`'s doc for why this lazily
    // (re)initializes a fresh 2-marble session the first tick we're
    // connected, and for the known reconnect-snap limitation.
    if net.status == NetStatus::Connected {
        if mp.live.is_none() {
            let rad = marble_state.marbles.first().map_or(0.15, |m| m.rad);
            let starts: Vec<Vec3> = (0..2)
                .map(|i| {
                    marble_state
                        .start_positions
                        .get(i)
                        .copied()
                        .unwrap_or(Vec3::new(i as f32 * 0.4, 0.0, 0.0))
                })
                .collect();
            let initial: Vec<Marble> = starts.iter().map(|s| Marble::spawn(*s, rad)).collect();
            marble_state.marbles = initial.clone();
            marble_state.start_positions = starts;
            marble_state.local_player_index = net.role.local_index();
            mp.live = Some(LiveSession {
                sim: RollbackSim::new(
                    initial,
                    PlayerInput { dx: 0.0, dy: 0.0, orientation: Quat::IDENTITY },
                    ROLLBACK_WINDOW_TICKS,
                ),
                transport: WebRtcTransport::new(net.role),
                tick: 0,
            });
        }

        let live = mp.live.as_mut().unwrap();
        live.tick += 1;
        let local_idx = net.role.local_index();

        // We always know our own input the instant we produce it -- send
        // it to the peer and confirm it locally in the same batch as
        // whatever the peer has sent us that's arrived since last tick
        // (one `receive_inputs` call, not two, per its own doc on why a
        // combined batch matters for correctness, not just efficiency).
        live.transport.send_input(live.tick, local_input);
        let mut arrivals = vec![(local_idx, live.tick, local_input)];
        arrivals.extend(live.transport.poll_received());

        let kill_y = marble_state.kill_y;
        let starts = marble_state.start_positions.clone();
        if let Err(e) = live.sim.receive_inputs(
            &arrivals,
            &scene.object,
            &scene.params,
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
        live.sim.advance(&scene.object, &scene.params, &marble_state.cfg, kill_y, &starts);
        marble_state.marbles = live.sim.marbles().to_vec();
        return;
    }

    // Offline path (unchanged from milestone 0/1): every marble gets *some*
    // `PlayerInput` this tick, since `step_marbles` needs one per marble --
    // the local player's is built from real input above; every other
    // marble gets a placeholder zero-input one (they just sit under
    // gravity/collision).
    let local_idx = marble_state.local_player_index;
    let inputs: Vec<PlayerInput> = (0..marble_state.marbles.len())
        .map(|i| if i == local_idx { local_input } else { PlayerInput { dx: 0.0, dy: 0.0, orientation: orbit.orientation } })
        .collect();

    let kill_y = marble_state.kill_y;
    let MarbleState { marbles, cfg, start_positions, .. } = &mut *marble_state;
    let _events = step_marbles(
        marbles,
        &inputs,
        &scene.object,
        &scene.params,
        cfg,
        kill_y,
        start_positions,
    );
}
