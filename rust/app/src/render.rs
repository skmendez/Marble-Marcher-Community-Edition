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
use marble_csg::scenes::{beware_of_bumps, demo_scene, set_fractal_params, ClassicHandles};
use marble_csg::{Object, Params};

use crate::camera::CameraOrbit;
use crate::physics_sys::MarbleState;

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

/// The CSG scene + its live parameter table, kept around so per-frame systems
/// can animate params and (M5) run the marble's CPU distance/nearest-point
/// collision queries against `object` without rebuilding the tree or
/// regenerating the shader.
#[derive(Resource)]
pub struct SceneState {
    /// The demo scene tree, queried each physics tick by
    /// `physics_sys::marble_physics_tick`.
    pub object: Object,
    pub params: Params,
    pub handles: ClassicHandles,
    pub material: Handle<MarcherMaterial>,
    pub params_buffer: Handle<ShaderStorageBuffer>,
}

/// Startup system: builds the demo scene, generates its WGSL, and spawns the
/// fullscreen quad that renders it (DESIGN.md §8).
pub fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<MarcherMaterial>>,
    mut shaders: ResMut<Assets<Shader>>,
    mut storage_buffers: ResMut<Assets<ShaderStorageBuffer>>,
) {
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

/// Per-frame system: writes the orbit camera basis (now following the
/// marble), timing, the marble uniform, and animated fractal params into the
/// material (DESIGN.md §7/§8). Animating `ang1` here (rather than at startup
/// only) is the proof that parameter changes are buffer writes with no
/// shader recompile.
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
    let ang1 = beware_of_bumps::ANG1 + 0.02 * (0.5 * t).sin();

    let SceneState {
        params,
        handles,
        material,
        params_buffer,
        ..
    } = &mut *scene_state;

    set_fractal_params(
        params,
        handles,
        beware_of_bumps::SCALE,
        ang1,
        beware_of_bumps::ANG2,
        beware_of_bumps::SHIFT,
        beware_of_bumps::COLOR,
        beware_of_bumps::ITERS,
    );

    if let Some(buffer) = storage_buffers.get_mut(&*params_buffer) {
        buffer.set_data(params.slots().to_vec());
    }

    let aspect = windows
        .single()
        .map(|w| w.width() / w.height().max(1.0))
        .unwrap_or(1.0);

    let marble = marble_state.marble;
    let (eye, right, up, forward) = orbit.eye_and_basis(marble.pos);

    if let Some(mat) = materials.get_mut(&*material) {
        mat.scene = SceneUniforms {
            cam_pos: eye.extend(0.0),
            cam_right: right.extend(0.0),
            cam_up: up.extend(0.0),
            cam_forward: forward.extend(1.5),
            marble: marble.pos.extend(marble.rad),
            sun: beware_of_bumps::sun_dir().extend(0.0),
            sun_col: beware_of_bumps::SUN_COL.extend(0.0),
            bg_col: beware_of_bumps::BG.extend(0.0),
            misc: Vec4::new(aspect, t, 0.0, 0.0),
        };
    }
}
