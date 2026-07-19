//! M5: fixed-timestep marble physics + WASD input, wired to the demo scene's
//! `SceneState` (rust/DESIGN.md §7/§8).
//!
//! `G` toggles between the two physics models `marble_csg::physics` supports
//! (see its module doc): [`GravityMode::Rolling`] (original MMCE physics —
//! gravity, kill plane, horizontal movement) and [`GravityMode::Flying`]
//! (this branch's zero-gravity free-flight experiment — full 3D
//! camera-relative thrust, no kill plane). Defaults to `Rolling`.
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

use bevy::prelude::*;

use marble_csg::physics::{step_marble, GravityMode, Marble, PhysicsConfig};
use marble_csg::scenes::beware_of_bumps;

use crate::camera::CameraOrbit;
use crate::render::SceneState;

/// The marble's live physics state + tuning constants. Spawns at the demo
/// scene's start position/radius (DESIGN.md §6/§7).
#[derive(Resource)]
pub struct MarbleState {
    pub marble: Marble,
    pub cfg: PhysicsConfig,
}

impl Default for MarbleState {
    fn default() -> Self {
        Self {
            marble: Marble::spawn(beware_of_bumps::START, beware_of_bumps::MARBLE_RAD),
            cfg: PhysicsConfig::default(),
        }
    }
}

/// One 60 Hz physics tick (`FixedUpdate`): reads WASD + the orbit camera's
/// yaw/pitch, steps the marble against the live CSG scene tree, lets `R`
/// force an immediate manual respawn, and `G` toggle [`GravityMode`]. A no-op
/// for scenes without a real marble (`SceneKind::has_marble` — the static
/// display fractals, not the tuned demo level).
pub fn marble_physics_tick(
    keys: Res<ButtonInput<KeyCode>>,
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
        marble_state.marble.respawn(beware_of_bumps::START);
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

    let MarbleState { marble, cfg } = &mut *marble_state;
    let _event = step_marble(
        marble,
        &scene.object,
        &scene.params,
        Vec2::new(dx, dy),
        orbit.yaw,
        orbit.pitch,
        cfg,
        beware_of_bumps::KILL_Y,
        beware_of_bumps::START,
    );
}
