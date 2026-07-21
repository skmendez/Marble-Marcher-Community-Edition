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

use crate::camera::CameraOrbit;
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

    if keys.just_pressed(KeyCode::KeyR) {
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
    //
    // Every marble gets *some* `PlayerInput` this tick, since `step_marbles`
    // needs one per marble — the local player's is built from real input
    // above; every other marble gets a placeholder zero-input one (they just
    // sit under gravity/collision) until milestone 2's real networking
    // supplies a real one per remote player.
    let local_idx = marble_state.local_player_index;
    let inputs: Vec<PlayerInput> = (0..marble_state.marbles.len())
        .map(|i| PlayerInput {
            dx: if i == local_idx { dx } else { 0.0 },
            dy: if i == local_idx { dy } else { 0.0 },
            orientation: orbit.orientation,
        })
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
