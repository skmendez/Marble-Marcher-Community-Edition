//! The ray-march material: a fullscreen `Material2d` whose fragment shader is
//! generated from a `marble_csg::Object` tree (see `rust/DESIGN.md` §5/§8).
//!
//! Bind group layout (must match `marble_csg::codegen::BINDINGS`, group 2):
//!   binding 0: `scene: SceneUniforms` (uniform)
//!   binding 1: `params: array<vec4<f32>>` (read-only storage)
//!
//! Deviation from DESIGN.md §8: the design sketches `params` as a bare
//! `Vec<Vec4>` field with `#[storage(1, read_only)]`. In bevy_render
//! 0.16.1 the `AsBindGroup` derive's non-`buffer` `storage(..)` arm only
//! accepts a `Handle<ShaderStorageBuffer>` field (it looks up a GPU buffer
//! asset by handle; see `bevy_render_macros::as_bind_group` and
//! `bevy_render::storage::ShaderStorageBuffer`) — there is no derive path
//! that turns an inline `Vec<Vec4>` field into a storage binding directly.
//! So `params` here is a `Handle<ShaderStorageBuffer>`, and the per-frame
//! system writes `Params::slots()` into that asset's bytes via
//! `ShaderStorageBuffer::set_data` (still a pure buffer write, no shader
//! recompile — the design's actual intent, just via an asset indirection).

use bevy::asset::weak_handle;
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderRef};
use bevy::render::storage::ShaderStorageBuffer;
use bevy::sprite::Material2d;
use bevy::window::PrimaryWindow;

use marble_csg::codegen::generate_shader;
use marble_csg::scenes::{
    beware_of_bumps, classic, demo_scene, menger_sphere, menger_sponge, set_fractal_params,
    set_menger_params, ClassicHandles, MengerHandles,
};
use marble_csg::{Object, Params};

use crate::camera::CameraOrbit;
use crate::physics_sys::MarbleState;

/// Which scene to build, selected via `MM_SCENE=demo|menger_sponge|menger_sphere`
/// (default `demo`). Only `Demo` has an actual marble/level tuning (start
/// position, kill plane, `ang1` animation support) — the other two are
/// `StaticFractals.hpp`'s standalone display fractals (C++ `MengerSponge`/
/// `MengerSphere`), ported for their own sake, not as playable levels; the
/// marble is hidden and its physics tick skipped for them (see
/// `physics_sys::marble_physics_tick` and `update_material` below).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneKind {
    Demo,
    /// `classic()` alone, without the `creme_spheres()` union — a debugging
    /// aid for isolating whether an issue comes from the decorative
    /// "repeating spheres" clutter or the classic fractal itself.
    ClassicOnly,
    MengerSponge,
    MengerSphere,
}

impl SceneKind {
    pub fn from_env() -> Self {
        match std::env::var("MM_SCENE").as_deref() {
            Ok("classic_only") => Self::ClassicOnly,
            Ok("menger_sponge") => Self::MengerSponge,
            Ok("menger_sphere") => Self::MengerSphere,
            _ => Self::Demo,
        }
    }

    /// Whether this scene has a real marble to simulate/render (the tuned
    /// demo level, in full or with the decorative clutter stripped out).
    pub fn has_marble(self) -> bool {
        matches!(self, Self::Demo | Self::ClassicOnly)
    }
}

/// The fractal-tree-specific parameter handles, so [`update_material`] can
/// animate `ang1` for [`SceneKind::Demo`] without needing to know about
/// [`MengerHandles`] (the static display fractals don't have an `ang1`).
pub enum SceneHandles {
    Classic(ClassicHandles),
    /// Not read anywhere yet — kept for a future depth/color animation
    /// toggle on the static display fractals, symmetric to Classic's ang1.
    #[allow(dead_code)]
    Menger(MengerHandles),
}

/// Fixed weak handle for the generated ray-marcher shader. A startup system
/// inserts the WGSL source into `Assets<Shader>` under this id; regenerating
/// the shader later (structural tree edits) is the same `insert` call again.
pub const MARCHER_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("ecd867ad-15d0-4d9f-8108-2f83d1101f00");

// `bevy_encase_derive`'s `ShaderType` derive emits, per field, a free
// `const _: fn() = || { fn check() { .. } };` item whose sole purpose is to
// force a trait-bound check at the field's span (for a nice compile error
// location) — `check` is deliberately never called, so rustc's `dead_code`
// lint flags it once per field. This is upstream macro-hygiene noise, not
// dead code in this crate; scope the allow to just this module so it
// doesn't hide real dead code elsewhere.
#[allow(dead_code)]
mod scene_uniforms_impl {
    use super::Vec4;
    use bevy::render::render_resource::ShaderType;

    /// Field-for-field match of the WGSL `SceneUniforms` struct emitted by
    /// `marble_csg::codegen` (DESIGN.md §5) — nine `vec4<f32>`s, same order.
    #[derive(Clone, Copy, Debug, ShaderType)]
    pub struct SceneUniforms {
        /// xyz eye position.
        pub cam_pos: Vec4,
        /// xyz unit right.
        pub cam_right: Vec4,
        /// xyz unit up.
        pub cam_up: Vec4,
        /// xyz unit forward, w = focal length (1 / tan(fov/2)).
        pub cam_forward: Vec4,
        /// xyz center, w radius (r <= 0 -> hidden). Unused until M5.
        pub marble: Vec4,
        /// xyz unit direction toward the sun.
        pub sun: Vec4,
        /// rgb.
        pub sun_col: Vec4,
        /// rgb.
        pub bg_col: Vec4,
        /// x aspect (w/h), y time seconds, z/w reserved.
        pub misc: Vec4,
    }

    impl Default for SceneUniforms {
        fn default() -> Self {
            Self {
                cam_pos: Vec4::ZERO,
                cam_right: Vec4::X,
                cam_up: Vec4::Y,
                cam_forward: Vec4::new(0.0, 0.0, -1.0, 1.5),
                marble: Vec4::ZERO,
                sun: Vec4::new(0.0, 1.0, 0.0, 0.0),
                sun_col: Vec4::ONE,
                bg_col: Vec4::ONE,
                misc: Vec4::new(1.0, 0.0, 0.0, 0.0),
            }
        }
    }
}
pub use scene_uniforms_impl::SceneUniforms;

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct MarcherMaterial {
    #[uniform(0)]
    pub scene: SceneUniforms,
    #[storage(1, read_only)]
    pub params: Handle<ShaderStorageBuffer>,
}

impl Material2d for MarcherMaterial {
    fn fragment_shader() -> ShaderRef {
        MARCHER_SHADER_HANDLE.into()
    }
}

/// Default depth/color for the two static display fractals ([`SceneKind::MengerSponge`]/
/// [`SceneKind::MengerSphere`]) — these aren't level data (unlike
/// `beware_of_bumps`), just reasonable-looking values.
///
/// `depth = 14` (this scene's original value) produced visibly speckled/
/// grainy rendering. Diagnosed as two separate, additive problems rather
/// than one (confirmed by probing `Object::de` directly, replicating this
/// exact camera, off the shader/GPU entirely -- see git history / session
/// notes for the standalone probe):
///  1. The default camera distance for these scenes (previously 3.0, see
///     `setup` below) put the camera embedded in/hugging the fractal
///     surface, so hits landed at a tiny `t` -- too tiny for `calc_normal`'s
///     `t`-scaled epsilon to give a well-conditioned finite-difference
///     normal (central differences of a chaotic fold at that scale are
///     dominated by float32 noise). This alone caused most of the
///     speckle and was fixed by moving the camera back (see `distance`
///     below) -- verified depth-independent: the degenerate-normal count
///     was ~85% of sampled pixels across `depth` 2..14 alike at the old
///     distance, and ~0% at every depth once the camera cleared the
///     surface.
///  2. Independently, `depth` beyond about 6-7 keeps folding but stops
///     changing anything an on-screen pixel can resolve: the recursive
///     squares shrink by 3x per level, and at this camera distance a level
///     is sub-pixel by around depth 6. Past that point, adjacent pixels'
///     marched hit points land in different recursive "cells" (each with
///     its own `OrbitMax`-accumulated color) for what should be the same
///     surface point, so the *color* (not the geometry -- confirmed by
///     screenshotting with lighting/shadow/AO forced flat, which still
///     showed the speckle) flickers per-pixel: classic sub-pixel-detail
///     aliasing from point-sampled marching with no supersampling. `depth
///     = 5` keeps 3-4 clearly nested levels of the characteristic Menger
///     cross-void pattern on screen with no visible speckle; `6` already
///     shows it faintly, growing with depth.
const MENGER_DEPTH: i32 = 5;
const MENGER_SPONGE_COLOR: Vec3 = Vec3::new(0.9, 0.65, 0.15);
const MENGER_SPHERE_COLOR: Vec3 = Vec3::new(0.25, 0.65, 0.9);

/// The CSG scene + its live parameter table, kept around so per-frame systems
/// can animate params and (M5) run the marble's CPU distance/nearest-point
/// collision queries against `object` without rebuilding the tree or
/// regenerating the shader.
#[derive(Resource)]
pub struct SceneState {
    pub kind: SceneKind,
    /// The scene tree, queried each physics tick by
    /// `physics_sys::marble_physics_tick` (for [`SceneKind::Demo`] only).
    pub object: Object,
    pub params: Params,
    pub handles: SceneHandles,
    pub material: Handle<MarcherMaterial>,
    pub params_buffer: Handle<ShaderStorageBuffer>,
}

/// Startup system: builds the selected scene (`MM_SCENE`, [`SceneKind::from_env`]),
/// generates its WGSL, and spawns the fullscreen quad that renders it
/// (DESIGN.md §8).
pub fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<MarcherMaterial>>,
    mut shaders: ResMut<Assets<Shader>>,
    mut storage_buffers: ResMut<Assets<ShaderStorageBuffer>>,
    mut camera_orbit: ResMut<CameraOrbit>,
) {
    let kind = SceneKind::from_env();
    let mut params = Params::new();

    // CameraOrbit::default() is tuned for the Demo scene specifically (aimed
    // at the marble's actual resting-surface normal, at a marble_rad-scaled
    // distance — see its doc comment). The static display fractals are a
    // completely different world scale (menger_sponge/menger_sphere are
    // roughly unit-scale after their final 0.33 shrink) and have no marble
    // to aim at, so that default puts the camera embedded inside the
    // fractal geometry (visibly: ray-marching noise/grain from being too
    // close to/inside thin recursive struts). Use a plain, un-embedded view
    // instead for those.
    //
    // `distance = 3.0` (the first value tried here) is *still* embedded:
    // measured with a probe replicating this exact yaw/pitch/fov against
    // `Object::de` (see git history / session notes), at distance 3.0 the
    // camera sits close enough that most of the 32x18 sample grid hits the
    // surface within `t ~= 0.4` world units, giving a normal-estimation
    // epsilon (`1e-4 * max(t, 0.05)`) so tiny that `calc_normal`'s central
    // difference is dominated by float32 noise -- ~85% of sampled pixels had
    // a near-zero (degenerate/undefined-direction) raw normal, which is
    // exactly the speckle/static look. Critically, this was reproduced
    // identically across `MENGER_DEPTH` 2..14 -- the recursion depth is not
    // the cause (deeper folds stop changing anything observable past about
    // depth 7, since the coordinate pattern is by then self-similar). Once
    // `distance` is large enough that hits land at `t` on the order of a
    // few world units, degenerate normals drop to zero at every depth
    // tried, confirming this is purely a "camera embedded in/hugging the
    // geometry" problem, not a marcher-precision-vs-recursion-depth one.
    if !kind.has_marble() {
        camera_orbit.yaw = 0.8;
        camera_orbit.pitch = 0.35;
        camera_orbit.distance = 6.0;
    }

    let (object, handles) = match kind {
        SceneKind::Demo => {
            let (object, classic_handles) = demo_scene(&mut params);
            set_fractal_params(
                &mut params,
                &classic_handles,
                beware_of_bumps::SCALE,
                beware_of_bumps::ANG1,
                beware_of_bumps::ANG2,
                beware_of_bumps::SHIFT,
                beware_of_bumps::COLOR,
                beware_of_bumps::ITERS,
            );
            (object, SceneHandles::Classic(classic_handles))
        }
        SceneKind::ClassicOnly => {
            let (object, classic_handles) = classic(&mut params);
            set_fractal_params(
                &mut params,
                &classic_handles,
                beware_of_bumps::SCALE,
                beware_of_bumps::ANG1,
                beware_of_bumps::ANG2,
                beware_of_bumps::SHIFT,
                beware_of_bumps::COLOR,
                beware_of_bumps::ITERS,
            );
            (object, SceneHandles::Classic(classic_handles))
        }
        SceneKind::MengerSponge => {
            let (object, menger_handles) = menger_sponge(&mut params);
            set_menger_params(&mut params, &menger_handles, MENGER_DEPTH, MENGER_SPONGE_COLOR);
            (object, SceneHandles::Menger(menger_handles))
        }
        SceneKind::MengerSphere => {
            let (object, menger_handles) = menger_sphere(&mut params);
            set_menger_params(&mut params, &menger_handles, MENGER_DEPTH, MENGER_SPHERE_COLOR);
            (object, SceneHandles::Menger(menger_handles))
        }
    };

    let wgsl = generate_shader(&object);
    shaders.insert(
        MARCHER_SHADER_HANDLE.id(),
        Shader::from_wgsl(wgsl, "generated://marcher.wgsl"),
    );

    let params_buffer = storage_buffers.add(ShaderStorageBuffer::from(params.slots().to_vec()));
    let material = materials.add(MarcherMaterial {
        scene: SceneUniforms::default(),
        params: params_buffer.clone(),
    });

    commands.spawn(Camera2d);
    commands.spawn((
        Mesh2d(meshes.add(Rectangle::new(1.0, 1.0).mesh())),
        MeshMaterial2d(material.clone()),
        Transform::default(),
    ));

    commands.insert_resource(SceneState {
        kind,
        object,
        params,
        handles,
        material,
        params_buffer,
    });
}

/// Keeps the fullscreen quad's world size equal to the window's pixel size
/// every frame (cheap, robust across resizes — DESIGN.md §8).
pub fn sync_quad_scale(
    windows: Query<&Window, With<PrimaryWindow>>,
    mut quads: Query<&mut Transform, With<Mesh2d>>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    for mut transform in &mut quads {
        transform.scale.x = window.width();
        transform.scale.y = window.height();
    }
}

/// Set `MM_ANIMATE_FRACTAL=1` to continuously nudge `ang1` over time — a
/// demo of live parameter updates with no shader recompile (the params
/// buffer is just a buffer write; DESIGN.md §3). **Off by default**: this
/// was an M4 proof-of-concept added before the marble had real physics, and
/// it actively breaks gameplay — the marble rests on a strut only ~1x its
/// own radius wide, and animating the fractal's rotation angle shifts that
/// strut out from under it, so the marble slides and falls even with zero
/// player input (confirmed by tracing the exact scenario headless: with the
/// animation on, a motionless marble starts falling within a few seconds).
/// The capability itself is still exercised by other means (codegen tests,
/// this same buffer-write path running every frame regardless for the
/// marble's position/radius) without needing to visibly wreck the level.
fn animate_fractal() -> bool {
    std::env::var("MM_ANIMATE_FRACTAL").as_deref() == Ok("1")
}

/// Per-frame system: writes the orbit camera basis (now following the
/// marble), timing, and the marble uniform into the material (DESIGN.md
/// §7/§8); re-syncs fractal params only if `animate_fractal()` is enabled
/// (otherwise the static params `setup()` wrote stay untouched, matching
/// what the marble's physics collides against).
pub fn update_material(
    time: Res<Time>,
    orbit: Res<CameraOrbit>,
    marble_state: Res<MarbleState>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut scene_state: ResMut<SceneState>,
    mut materials: ResMut<Assets<MarcherMaterial>>,
    mut storage_buffers: ResMut<Assets<ShaderStorageBuffer>>,
) {
    let t = time.elapsed_secs();

    // ang1 animation only applies to (and only makes sense for) the Classic
    // fractal tree the Demo scene builds; the static display fractals have
    // no such parameter.
    if animate_fractal() {
        if let SceneHandles::Classic(handles) = &scene_state.handles {
            let ang1 = beware_of_bumps::ANG1 + 0.02 * (0.5 * t).sin();
            let handles = *handles;
            set_fractal_params(
                &mut scene_state.params,
                &handles,
                beware_of_bumps::SCALE,
                ang1,
                beware_of_bumps::ANG2,
                beware_of_bumps::SHIFT,
                beware_of_bumps::COLOR,
                beware_of_bumps::ITERS,
            );
            if let Some(buffer) = storage_buffers.get_mut(&scene_state.params_buffer) {
                buffer.set_data(scene_state.params.slots().to_vec());
            }
        }
    }

    let aspect = windows
        .single()
        .map(|w| w.width() / w.height().max(1.0))
        .unwrap_or(1.0);

    // Only the Demo scene has a real marble: follow it with the camera and
    // render it; the static display fractals get a fixed origin-centered
    // camera and a hidden marble (marble.w <= 0 -> the shader skips it).
    let has_marble = scene_state.kind.has_marble();
    let marble = marble_state.marble;
    let target = if has_marble { marble.pos } else { Vec3::ZERO };
    let (eye, right, up, forward) = orbit.eye_and_basis(target);
    let marble_uniform = if has_marble {
        marble.pos.extend(marble.rad)
    } else {
        Vec4::ZERO
    };

    if let Some(mat) = materials.get_mut(&scene_state.material) {
        mat.scene = SceneUniforms {
            cam_pos: eye.extend(0.0),
            cam_right: right.extend(0.0),
            cam_up: up.extend(0.0),
            cam_forward: forward.extend(1.5),
            marble: marble_uniform,
            sun: beware_of_bumps::sun_dir().extend(0.0),
            sun_col: beware_of_bumps::SUN_COL.extend(0.0),
            bg_col: beware_of_bumps::BG.extend(0.0),
            misc: Vec4::new(aspect, t, 0.0, 0.0),
        };
    }
}
