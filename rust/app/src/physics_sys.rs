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
//! WASD sign convention (`Rolling` mode; verified algebraically against
//! `CameraOrbit`'s basis — rust/DESIGN.md §7): given `step_marble`'s
//! camera-yaw-relative movement formula and `CameraOrbit`'s basis (forward
//! points from the eye toward the target, i.e. roughly `-(sin(yaw), 0,
//! cos(yaw))`), setting `dy = -1` for W yields velocity `+(sin(yaw), 0,
//! cos(yaw))` — the direction from the target *away* from the eye, i.e. W
//! rolls the marble away from the orbiting camera (and S rolls it back
//! toward the camera). Setting `dx = +1` for D yields `+(cos(yaw), 0,
//! -sin(yaw))`, which is exactly `CameraOrbit`'s `right` vector at zero
//! pitch — D rolls the marble in the camera's screen-right direction. In
//! `Flying` mode the same `dx`/`dy` inputs instead drive full 3D thrust
//! along wherever the camera is actually pointed (see `step_marble`'s doc).
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

use bevy::input::touch::Touches;
use bevy::prelude::*;

use marble_csg::physics::{step_marble, GravityMode, Marble, PhysicsConfig};

use crate::camera::CameraOrbit;
use crate::render::SceneState;
use crate::touch::read_two_finger_gesture;

/// The marble's live physics state + tuning constants, plus the current
/// scene's spawn point and kill-plane height (used for `R`/kill-plane
/// respawns — every scene has different values now that all scenes have a
/// marble, see `SceneKind::spawn_params`, so these can no longer be a fixed
/// constant read directly from `physics_sys.rs`). Constructed per-scene in
/// `render::setup` (there's no scene-independent `Default` to speak of).
#[derive(Resource)]
pub struct MarbleState {
    pub marble: Marble,
    pub cfg: PhysicsConfig,
    pub start_pos: Vec3,
    pub kill_y: f32,
}

/// One 60 Hz physics tick (`FixedUpdate`): reads WASD + a 2-finger pinch
/// (additive — see module doc) + the orbit camera's yaw/pitch, steps the
/// marble against the live CSG scene tree, lets `R` force an immediate
/// manual respawn, and `G` toggle [`GravityMode`]. A no-op for scenes
/// without a real marble (`SceneKind::has_marble` — the static display
/// fractals, not the tuned demo level).
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
        let start = marble_state.start_pos;
        marble_state.marble.respawn(start);
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

    let kill_y = marble_state.kill_y;
    let start = marble_state.start_pos;
    let MarbleState { marble, cfg, .. } = &mut *marble_state;
    let _event = step_marble(
        marble,
        &scene.object,
        &scene.params,
        Vec2::new(dx, dy),
        orbit.yaw,
        orbit.pitch,
        cfg,
        kill_y,
        start,
    );
}
