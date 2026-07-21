//! Live thrust-direction debug overlay: draws the marble's actual applied
//! force (`Marble::last_thrust`, set by `step_marble` every physics tick)
//! plus the camera's own right/up reference axes, so "is the pushing force
//! actually toward/away from the camera in every setting" can be checked by
//! eye instead of re-derived on paper.
//!
//! This renderer's "camera" is virtual — the actual picture is a fullscreen
//! `Material2d` quad ray-marched by a generated WGSL shader
//! (`render.rs`/`codegen.rs`), not a real Bevy 3D camera with a
//! `Transform`/`Projection` matching what's on screen. So a normal 3D gizmo
//! (drawn through a `Camera3d`) would not line up with the ray-marched
//! output at all. Instead, [`project_to_screen`] below manually inverts the
//! *exact* perspective convention `codegen.rs`'s `fragment` ray setup uses
//! (`rd = right*ndc.x*aspect + up*ndc.y + forward*f`) and draws 2D arrows on
//! the same `Camera2d` the marcher quad and FPS overlay already render
//! through — a second, independently-written projection here could drift
//! out of sync with the shader's own camera math exactly the way the
//! original thrust-direction bug this overlay exists to verify did.
//!
//! Key geometric fact this overlay leans on: `CameraOrbit::eye_and_basis`
//! always targets the marble (`update_material` passes `marble.pos` as
//! `target`), so the marble is *always* exactly at the center of the screen
//! (`eye = target - forward*distance` puts `target` at zero lateral offset
//! and depth exactly `distance`, regardless of roll — `right`/`up` are
//! always orthogonal to `forward` by construction). That means a pure
//! `cam_forward`-aligned direction (straight toward/away from the camera)
//! literally has no distinguishable on-screen 2D heading — it's the
//! vanishing point itself — so this overlay doesn't try to draw a "forward"
//! reference arrow at all; instead it draws `right`/`up` (which *do* have a
//! well-defined, always-horizontal/always-vertical screen direction by the
//! same construction) as fixed references, plus the actual thrust vector.
//! In `Flying` mode (`v = dy*FOCAL_DIST*cam_forward + dx*cam_right`, no
//! `up` component at all), a bug-free pure W/S thrust must draw as a
//! near-zero-length arrow collapsed at screen center, and a bug-free pure
//! A/D thrust must draw exactly parallel to the horizontal `right`
//! reference arrow with no vertical component — any visible deviation from
//! either is a real bug, not a projection artifact.

use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::camera::{CameraOrbit, FOCAL_LENGTH};
use crate::physics_sys::MarbleState;

/// Fixed on-screen arrow length (logical pixels) — a screen-space debug
/// overlay, deliberately not scaled by world distance from the camera.
const ARROW_LENGTH_PX: f32 = 90.0;

/// World-space offset used to probe a direction's on-screen heading (see
/// [`draw_thrust_debug`]) — small enough that even a fully camera-ward unit
/// direction (`-forward`) stays in front of the eye at `CameraOrbit`'s
/// `MIN_DISTANCE` (0.12): worst-case remaining depth `0.12 - 0.03 = 0.09 >
/// 0`. Only matters for directions with a `forward` component (`right`/`up`
/// are exactly depth-preserving regardless of this value, since they're
/// orthogonal to `forward` by construction).
const PROBE_OFFSET: f32 = 0.03;

/// Projects `point` into this frame's `Camera2d` screen space: logical
/// pixels, origin at the window center, +x right, +y up (Bevy's default 2D
/// camera convention — matches how `fps_overlay.rs`'s UI nodes and the
/// marcher quad's own `Transform` are already positioned relative to the
/// window). `eye`/`right`/`up`/`forward` are the virtual ray-marcher
/// camera's basis (`CameraOrbit::eye_and_basis`); `aspect` and
/// `window_size` match what `update_material` feeds the shader.
///
/// Inverts `codegen.rs`'s ray-setup formula: a ray through NDC `(nx, ny)` is
/// `rd ∝ right*nx*aspect + up*ny + forward*f`. Given a world point at
/// camera-relative depth `z_cam = dot(point - eye, forward)`, solving for
/// the `(nx, ny)` whose ray passes through it yields `nx = x_cam*f /
/// (z_cam*aspect)`, `ny = y_cam*f / z_cam`. NDC `[-1, 1]` maps to logical
/// pixels the same way `codegen.rs`'s `ndc = (uv*2-1, 1-uv*2)` maps to `uv`
/// (`[0, 1]`, y-down) and thence to a `[0, window]` pixel — composing that
/// with Bevy 2D gizmos' own window-center-origin, y-up convention collapses
/// to the plain `ndc * window_size / 2` below.
///
/// Returns `None` if `point` is behind the camera (`z_cam <= 0`), where a
/// perspective projection has no on-screen answer.
fn project_to_screen(
    point: Vec3,
    eye: Vec3,
    right: Vec3,
    up: Vec3,
    forward: Vec3,
    aspect: f32,
    window_size: Vec2,
) -> Option<Vec2> {
    let rel = point - eye;
    let z_cam = rel.dot(forward);
    if z_cam <= 1e-4 {
        return None;
    }
    let x_cam = rel.dot(right);
    let y_cam = rel.dot(up);
    let ndc_x = x_cam * FOCAL_LENGTH / (z_cam * aspect);
    let ndc_y = y_cam * FOCAL_LENGTH / z_cam;
    Some(Vec2::new(
        ndc_x * window_size.x * 0.5,
        ndc_y * window_size.y * 0.5,
    ))
}

/// `Update` system: draws `right`/`up` reference arrows (always) and the
/// actual applied thrust direction (`MarbleState::marble::last_thrust`,
/// only while nonzero) from the screen center — see the module doc for why
/// the marble is always exactly there, and why "forward" isn't drawn as a
/// reference.
pub fn draw_thrust_debug(
    orbit: Res<CameraOrbit>,
    marble_state: Res<MarbleState>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut gizmos: Gizmos,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let window_size = Vec2::new(window.width(), window.height());
    let aspect = window_size.x / window_size.y.max(1.0);

    let marble = marble_state.local_marble();
    let (eye, right, up, forward) = orbit.eye_and_basis(marble.pos);

    // The marble is always exactly at screen center by construction (module
    // doc) -- use that directly rather than reprojecting it every frame.
    let origin = Vec2::ZERO;

    let mut draw_ray = |dir: Vec3, color: Srgba| {
        if dir.length_squared() < 1e-8 {
            return;
        }
        let Some(tip) = project_to_screen(
            marble.pos + dir.normalize() * PROBE_OFFSET,
            eye,
            right,
            up,
            forward,
            aspect,
            window_size,
        ) else {
            return;
        };
        let screen_dir = tip - origin;
        if screen_dir.length_squared() < 1.0 {
            // Collapsed to (near) the center point -- expected for a pure
            // forward/backward direction (module doc), not drawable as a
            // heading.
            return;
        }
        let end = origin + screen_dir.normalize() * ARROW_LENGTH_PX;
        gizmos.arrow_2d(origin, end, color);
    };

    // Fixed references: always exactly horizontal/vertical on screen.
    draw_ray(right, bevy::color::palettes::basic::AQUA);
    draw_ray(up, bevy::color::palettes::basic::LIME);
    // The actual thrust `step_marble` applied this tick.
    draw_ray(marble.last_thrust, bevy::color::palettes::basic::RED);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_along_forward_projects_to_center() {
        let (eye, right, up, forward) = (Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z);
        let got = project_to_screen(forward * 10.0, eye, right, up, forward, 1.0, Vec2::new(200.0, 100.0))
            .unwrap();
        assert!(got.distance(Vec2::ZERO) < 1e-3, "expected center, got {got:?}");
    }

    #[test]
    fn point_offset_along_right_projects_with_known_offset() {
        // eye=0, basis = standard axes, f = 2.0 (not FOCAL_LENGTH -- this
        // test constructs its own basis/point to check the formula directly,
        // independent of whatever FOCAL_LENGTH currently is).
        // point = forward*10 + right*5 => z_cam=10, x_cam=5.
        // ndc_x = x_cam*f/(z_cam*aspect) = 5*2/(10*1) = 1.0
        // sx = ndc_x * window_x/2 = 1.0 * 100 = 100
        let (eye, right, up, forward) = (Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z);
        let point = forward * 10.0 + right * 5.0;
        // Can't override FOCAL_LENGTH per-call, so pick window/aspect that
        // make the expected answer easy to state given the real constant:
        // sx = 5 * FOCAL_LENGTH / 10 / aspect * window_x/2, with aspect=1,
        // window_x=2 => sx = FOCAL_LENGTH/2.
        let got = project_to_screen(point, eye, right, up, forward, 1.0, Vec2::new(2.0, 2.0)).unwrap();
        assert!((got.x - FOCAL_LENGTH / 2.0).abs() < 1e-4, "got {got:?}");
        assert!(got.y.abs() < 1e-4, "got {got:?}");
    }

    #[test]
    fn point_behind_camera_has_no_projection() {
        let (eye, right, up, forward) = (Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z);
        let behind = -forward * 5.0;
        assert!(project_to_screen(behind, eye, right, up, forward, 1.0, Vec2::new(2.0, 2.0)).is_none());
    }

    #[test]
    fn marble_target_is_always_screen_center_regardless_of_roll() {
        // Regression check for the module doc's key invariant: eye_and_basis
        // always targets `marble.pos`, so it must project to exactly (0,0)
        // for any orientation (twist included -- composed directly into
        // `orientation` via `Quat::from_rotation_z`, matching what
        // `CameraOrbit::roll` itself does) /distance.
        for roll in [0.0_f32, 0.7, 2.9, -1.4] {
            let orbit = CameraOrbit {
                orientation: CameraOrbit::orientation_from_yaw_pitch(0.8, 0.35) * Quat::from_rotation_z(roll),
                distance: 0.6,
            };
            let target = Vec3::new(1.0, 2.0, 3.0);
            let (eye, right, up, forward) = orbit.eye_and_basis(target);
            let got = project_to_screen(target, eye, right, up, forward, 1.3, Vec2::new(1280.0, 720.0))
                .unwrap();
            assert!(got.distance(Vec2::ZERO) < 1e-2, "roll={roll}: expected center, got {got:?}");
        }
    }
}
