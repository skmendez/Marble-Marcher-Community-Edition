//! The ray-march material: a fullscreen `Material2d` whose fragment shader is
//! generated from a `marble_csg::Object` tree (see `rust/DESIGN.md` §5/§8).
//!
//! Bind group layout (must match `marble_csg::codegen::BINDINGS`, group 2):
//!   binding 0: `scene: SceneUniforms` (uniform)
//!   binding 1: `params: array<vec4<f32>>` (read-only storage)
//!   binding 2: `coarse_tex: texture_2d<f32>` (MRRM coarse pre-pass's cached
//!   hit-distance render target -- `mrrm.rs`; fine pass
//!   ([`FineMarcherMaterial`]) only. The coarse pass's own material
//!   ([`crate::mrrm::CoarseMarcherMaterial`]) has no binding 2 -- see that
//!   module's doc for why coarse/fine are two separate `Material2d`s with
//!   two separate generated shader modules instead of one shared one.
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
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, ShaderRef, TextureDimension, TextureFormat, TextureViewDescriptor,
    TextureViewDimension,
};
use bevy::render::storage::ShaderStorageBuffer;
use bevy::sprite::Material2d;
use bevy::ui::IsDefaultUiCamera;
use bevy::window::PrimaryWindow;

use marble_csg::codegen::generate_shader;
use marble_csg::expr::Expr;
use marble_csg::physics::{Marble, PhysicsConfig};
use marble_csg::scenes::{
    beware_of_bumps, classic, demo_scene, menger_oscillating_sphere, menger_sphere, menger_sponge,
    set_fractal_params, set_menger_params, ClassicHandles, MengerHandles,
    MengerOscillatingSphereHandles, MENGER_BITE_MIN_RADIUS,
};
use marble_csg::scene_sync::SceneBundle;
use marble_csg::{Object, Params, ScalarParam};

use crate::camera::CameraOrbit;
use crate::physics_sys::{MarbleState, MultiplayerSession, PendingSceneSync};

/// Which scene to build, selected via
/// `MM_SCENE=demo|classic_only|menger_sponge|menger_sphere|menger_oscillating_sphere`
/// on native, or `?scene=<same value>` in the URL on the deployed web build
/// (`std::env::var` always returns `Err` on wasm32-unknown-unknown -- there's
/// no OS environment in a browser -- so `MM_SCENE` alone only ever had any
/// effect for local/native testing; `web_config::query_param` is the
/// browser-reachable equivalent, same value vocabulary, so the two can't
/// drift apart). Defaults to `MengerOscillatingSphere` either way if
/// unset/unrecognized.
///
/// `Demo`/`ClassicOnly` have authored level data (`beware_of_bumps`: a
/// start position tuned to rest on a surface, a kill plane, `ang1`
/// animation support); the Menger scenes don't (C++ `StaticFractals.hpp`'s
/// `MengerSponge`/`MengerSphere` were never actual levels, just standalone
/// shape-generator functions), so their marble spawn point/radius/kill
/// plane are reasonable-looking placeholders (`spawn_params` below), not
/// tuned level data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneKind {
    Demo,
    /// `classic()` alone, without the `creme_spheres()` union — a debugging
    /// aid for isolating whether an issue comes from the decorative
    /// "repeating spheres" clutter or the classic fractal itself.
    ClassicOnly,
    MengerSponge,
    MengerSphere,
    /// [`menger_oscillating_sphere`] — same shape as `MengerSphere`, but the
    /// bite sphere's radius is a runtime `Param` driven by a
    /// `marble_csg::expr::Expr`, evaluated once per physics tick against
    /// the shared `Tick` clock (`physics_sys::MarbleState::offline_tick` or
    /// `RollbackSim`'s own tick once networked — see `expr`'s module doc
    /// for why wall-clock time isn't safe here), oscillating between
    /// `MENGER_BITE_MIN_RADIUS` (removes nothing) and
    /// `MENGER_BITE_MAX_RADIUS` (only the corners survive) — a live demo of
    /// the CSG tree's existing runtime-uniform support
    /// (`marble_csg`'s `*Value::Param`/`Params`) animating actual geometry,
    /// not just a fractal fold's rotation/color/iteration-count.
    MengerOscillatingSphere,
}

/// Marble spawn parameters for a scene: start position, radius, kill-plane
/// height (only actually reachable in [`marble_csg::physics::GravityMode::Rolling`],
/// since `Flying` — the default — has no kill plane at all, but every scene
/// needs *some* value to hand `step_marble` regardless of mode).
pub struct MarbleSpawn {
    pub start: Vec3,
    pub rad: f32,
    pub kill_y: f32,
}

impl SceneKind {
    /// Reads `MM_SCENE` (native) or `?scene=` (web) -- see this type's doc.
    /// `query_param` is always `None` on native (`web_config`'s doc), so the
    /// `.or_else` falls straight through to `std::env::var` there exactly as
    /// before; on wasm `std::env::var` is always `Err`, so the query param is
    /// the only branch that can ever match.
    pub fn from_config() -> Self {
        let value = crate::web_config::query_param("scene")
            .or_else(|| std::env::var("MM_SCENE").ok());
        match value.as_deref() {
            Some("demo") => Self::Demo,
            Some("classic_only") => Self::ClassicOnly,
            Some("menger_sponge") => Self::MengerSponge,
            Some("menger_sphere") => Self::MengerSphere,
            Some("menger_oscillating_sphere") => Self::MengerOscillatingSphere,
            _ => Self::MengerOscillatingSphere,
        }
    }

    /// Every scene has a marble now (matching the original MMCE, where
    /// every playable level does) — kept as a named method rather than
    /// inlining `true` everywhere in case a genuinely marble-less preview
    /// scene is ever added later.
    pub fn has_marble(self) -> bool {
        true
    }

    /// Where/how big the marble starts, and where the (`Rolling`-mode-only)
    /// kill plane sits, for this scene.
    pub fn spawn_params(self) -> MarbleSpawn {
        match self {
            Self::Demo | Self::ClassicOnly => MarbleSpawn {
                start: beware_of_bumps::START,
                rad: beware_of_bumps::MARBLE_RAD,
                kill_y: beware_of_bumps::KILL_Y,
            },
            // `Vec3::ZERO` (the original spawn point, deep inside the
            // sponge's internal void network) has de(origin) ~= 1.4 -- not
            // embedded, but *occluded*: from the exterior corner view
            // `setup`'s camera override actually uses, the origin sits
            // behind the sponge's own outer walls, so the marble was
            // invisible there no matter how large it was (confirmed: even
            // `rad = 0.3` at the origin was still fully hidden). This spawn
            // point instead sits in the open-air pocket just in front of
            // that same exterior corner, along the camera's default view
            // ray -- found by probing `Object::de` at points between the
            // camera's eye and the origin (`rust/csg/examples/spawn_probe.rs`
            // in git history) for a spot with a comfortable safety margin
            // (de ~= 0.34, well clear of `rad` below) that's still close
            // to the visible surface, not floating off in empty sky.
            // `rad = 0.15` (up from the original 0.05) is sized to read
            // clearly at the correspondingly closer orbit distance
            // (`setup`'s Menger camera override, below) rather than for
            // fitting through the sponge's recursive tunnels, since a
            // marble spawned outside the structure doesn't need to.
            // `kill_y = -50.0` is a generous "fell way out of the
            // structure" bound for if `G` is toggled to `Rolling` mode
            // while in one of these scenes. Reused as-is for
            // `MengerOscillatingSphere`: this spawn point's distance from
            // the origin (~4.92) stays well clear of even
            // `MENGER_BITE_MAX_RADIUS` (~3.03), so the marble is never
            // engulfed by the oscillating bite sphere.
            Self::MengerSponge | Self::MengerSphere | Self::MengerOscillatingSphere => {
                MarbleSpawn {
                    start: Vec3::new(3.32, 1.69, 3.22),
                    rad: 0.15,
                    kill_y: -50.0,
                }
            }
        }
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
    /// [`SceneKind::MengerOscillatingSphere`]'s handles -- the actual
    /// animation lives in [`SceneState::animations`] now (populated from
    /// [`MengerOscillatingSphereHandles::radius_anim`] once, in `setup`),
    /// not read back out of here per frame; kept for parity with the other
    /// variants and in case a future feature needs the raw handles again.
    #[allow(dead_code)]
    MengerOscillatingSphere(MengerOscillatingSphereHandles),
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
    /// `marble_csg::codegen` (DESIGN.md §5) — ten `vec4<f32>`s, same order.
    /// The marble list moved out to its own storage buffer (multiplayer
    /// milestone 0, `FineMarcherMaterial::marbles`/`MARBLE_BUFFER_BINDING`'s
    /// doc) -- N marbles don't fit a fixed uniform field the way one did.
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
        /// xyz unit direction toward the sun.
        pub sun: Vec4,
        /// rgb.
        pub sun_col: Vec4,
        /// rgb.
        pub bg_col: Vec4,
        /// x aspect (w/h), y time seconds, z *this pass's own* render target
        /// height in physical pixels (drives the marcher's distance-scaled
        /// hit threshold -- see `codegen.rs`'s `MARCH_CORE`; deliberately
        /// read from a uniform each frame rather than a shader constant so
        /// adaptive-render-resolution doesn't need this touched -- and
        /// "this pass's own": each of the three passes' materials write
        /// their *own* render target's height here, not another's, since
        /// each pass's cone-angle threshold must match its own resolution),
        /// w MRRM on/off flag (fine pass only -- `mrrm::mrrm_enabled`, see
        /// `update_material`'s doc; unused/always 0 on the coarse/shadow
        /// passes' own materials, which don't read it).
        pub misc: Vec4,
        /// x the shadow/AO pass's own render-target height (fine pass's
        /// material only -- feeds `sample_shadow`'s depth-tolerance `sz`
        /// computation, mirroring how `misc.z` threads the coarse pass's own
        /// resolution into the fine pass's MRRM back-off), y `MM_SHADOW_LOD`
        /// on/off flag (same uniform-flag convention as `misc.w`'s MRRM
        /// flag -- see `shadow_pass::shadow_lod_enabled`), z the `?perfprobe=`
        /// diagnostic's runtime override for the fine pass's march step
        /// budget (`perfprobe.rs`; `0.0` means "no override, use the
        /// compile-time `MAX_STEPS`" -- the value every non-probe run ever
        /// sees), w unused.
        pub misc2: Vec4,
        /// xyz world-space bounding-sphere center, w radius (`<= 0.0` means
        /// "no bound" -- either the scene is genuinely unbounded or this was
        /// never populated -- see `ray_sphere_clip`'s doc, `MARCH_CORE`).
        /// Same value on every pass's material (computed once from the
        /// scene tree at setup time, `SceneState::bounding_sphere`).
        pub bounding: Vec4,
    }

    impl Default for SceneUniforms {
        fn default() -> Self {
            Self {
                cam_pos: Vec4::ZERO,
                cam_right: Vec4::X,
                cam_up: Vec4::Y,
                cam_forward: Vec4::new(0.0, 0.0, -1.0, 1.5),
                sun: Vec4::new(0.0, 1.0, 0.0, 0.0),
                sun_col: Vec4::ONE,
                bg_col: Vec4::ONE,
                misc: Vec4::new(1.0, 0.0, 0.0, 0.0),
                misc2: Vec4::ZERO,
                // radius 0.0 -- "no bound" until `SceneState::bounding_sphere`
                // is actually written in.
                bounding: Vec4::ZERO,
            }
        }
    }
}
pub use scene_uniforms_impl::SceneUniforms;

/// Marker for the ray-marcher's own camera -- distinguishes it from
/// `mrrm.rs`'s `CoarseCamera` so marcher-only systems don't need to guess
/// which `Camera2d` entity is which.
#[derive(Component)]
pub struct MarcherCamera;

/// Marker for the ray-marcher's fullscreen quad -- distinguishes it from
/// `mrrm.rs`'s coarse-pass quad, both of which are `Mesh2d` entities, so
/// `sync_quad_scale` only scales this one.
#[derive(Component)]
pub struct MarcherQuad;

/// The fine (full-resolution) marcher's material. Two bindings unchanged
/// from pre-MRRM (`scene`/`params`), the MRRM coarse pre-pass's cached
/// hit-distance render target (`mrrm.rs`) and the half-resolution shadow/AO
/// pass's cached visibility (`shadow_pass.rs`) as a third and fourth -- no
/// paired `#[sampler]` bindings: the generated shader only ever reads these
/// textures with `textureLoad` (exact texel; `sample_shadow` does its own
/// hand-rolled depth-aware 4-tap blend for the shadow one), never
/// `textureSample`, so no sampler is needed at all -- and a fifth, the live
/// marble list (multiplayer milestone 0, `marble_csg::codegen::MARBLE_BUFFER_BINDING`),
/// mirroring `params`' own storage-buffer pattern rather than a fixed-size
/// uniform array. See `marble_csg::codegen::COARSE_TEXTURE_BINDING`'s doc
/// for a known, environment-specific (llvmpipe-only) crash this exact data
/// flow triggers in this project's native test sandbox -- not a bug in this
/// binding/shader.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct FineMarcherMaterial {
    #[uniform(0)]
    pub scene: SceneUniforms,
    #[storage(1, read_only)]
    pub params: Handle<ShaderStorageBuffer>,
    #[texture(2)]
    pub coarse: Handle<Image>,
    #[texture(3)]
    pub shadow: Handle<Image>,
    #[storage(4, read_only)]
    pub marbles: Handle<ShaderStorageBuffer>,
    /// The marble's cubemap texture (`MarbleCubemap`'s doc) -- unlike
    /// `coarse`/`shadow` this genuinely needs a paired `#[sampler]`: the
    /// shader samples it with `textureSample` (a filtered, direction-vector
    /// lookup), not `textureLoad`, so a real sampler binding is required,
    /// not optional (`marble_csg::codegen::MARBLE_TEXTURE_BINDING`'s doc).
    #[texture(5, dimension = "cube")]
    #[sampler(6)]
    pub marble_texture: Handle<Image>,
}

impl Material2d for FineMarcherMaterial {
    fn fragment_shader() -> ShaderRef {
        MARCHER_SHADER_HANDLE.into()
    }
}

/// A minimal (1x1 per face) placeholder cube texture, already correctly
/// `Cube`-dimensioned -- used as `marble_texture`'s initial value in `setup`
/// while the real `marble_cubemap.png` is still loading. `Handle::default()`
/// (what `coarse`/`shadow` use for their own startup placeholders) is *not*
/// substitutable here: it resolves to Bevy's generic placeholder image,
/// which is a plain 1x1 **2D** texture -- dimensionally incompatible with a
/// binding declared `dimension = "cube"` (confirmed live: wgpu rejects the
/// bind group with "Dimension (e2D) ... doesn't match the expected
/// dimension (Cube)" the moment that mismatch is actually bound). `coarse`/
/// `shadow` get away with the generic default only because their own
/// bindings don't override the dimension at all (plain `texture_2d<f32>`),
/// so a plain-2D placeholder is exactly what they expect.
fn make_placeholder_cubemap() -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 6,
        },
        TextureDimension::D2,
        &[255, 255, 255, 255],
        TextureFormat::Rgba8UnormSrgb,
        bevy::asset::RenderAssetUsages::default(),
    );
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::Cube),
        ..default()
    });
    image
}

/// The marble's cubemap texture: loaded from `assets/marble_cubemap.png`, a
/// vertical strip of 6 square faces in `+X, -X, +Y, -Y, +Z, -Z` order --
/// exactly wgpu/WebGPU's own cube-array layer order, so treating the 6
/// stacked regions as sequential array layers (below) needs no manual face
/// remapping.
///
/// `AssetServer::load` returns a usable-immediately `Handle<Image>`, but the
/// actual pixel data doesn't exist until it finishes loading asynchronously
/// (a local file read natively, an HTTP fetch on wasm) -- so `setup` can't
/// build the final cube-shaped image in the same frame it starts the load.
/// `finalize_marble_cubemap` (`Update`) does that the first frame the data
/// actually exists, gated by `done` so it only ever runs once.
///
/// `loading: Handle<Image>` is the raw 2D-strip asset from `AssetServer::load`
/// -- never bound to any material, never rendered, exists only so
/// `finalize_marble_cubemap` has something to poll for "has the PNG finished
/// downloading/decoding yet." The *real*, cube-shaped image the shader
/// actually samples is a **separate** `Image` built fresh once that data
/// exists (see that fn's doc for why this indirection is load-bearing, not
/// incidental complexity).
#[derive(Resource)]
pub(crate) struct MarbleCubemap {
    loading: Handle<Image>,
    done: bool,
}

/// `Update` system: the other half of `MarbleCubemap`'s doc. Once the PNG's
/// pixel data actually exists, this builds a **new** `Image` already
/// correctly cube-shaped from that data and redirects
/// [`FineMarcherMaterial::marble_texture`] to it.
///
/// This is deliberate, not incidental: an earlier version of this function
/// instead reinterpreted `MarbleCubemap::loading`'s own `Image` in place
/// (`Image::reinterpret_stacked_2d_as_array` plus overriding
/// `texture_view_descriptor`), and that broke production (commit `86ebb1d`,
/// reverted in `d5785ed`). Chrome's real WebGPU implementation rejected the
/// generated shader module outright at `CreateRenderPipeline` time, even
/// though naga's own offline parse+validate considered it fully valid.
/// Root-caused via a live diagnostic build: the render pipeline is already
/// created (against `FallbackImage::cube`, a real, correctly-dimensioned
/// fallback Bevy provides for exactly this still-loading-handle window) for
/// the several frames before the PNG finishes loading. The crash appeared
/// immediately after mutating the already-bound, already-extracted
/// `loading` image's dimension in place from 2D to a 6-layer cube — the
/// same "mutating an actively-bound `Image` corrupts rendering" bug class
/// this codebase already hit and fixed twice before
/// (`mrrm::resize_coarse_render_target`, `shadow_pass::resize_shadow_render_target`,
/// both of which build a new `Image` and redirect handles rather than
/// resize in place, for the same underlying reason). Building an entirely
/// separate, never-previously-bound `Image` here avoids ever presenting
/// Bevy's render extraction with a resource that changes shape after it's
/// already live, exactly like those two fixes.
pub fn finalize_marble_cubemap(
    mut cubemap: ResMut<MarbleCubemap>,
    mut images: ResMut<Assets<Image>>,
    fine_quads: Query<&MeshMaterial2d<FineMarcherMaterial>, With<MarcherQuad>>,
    mut fine_materials: ResMut<Assets<FineMarcherMaterial>>,
) {
    if cubemap.done {
        return;
    }
    let Some(loaded) = images.get(&cubemap.loading) else {
        return;
    };

    let mut cube_image = loaded.clone();
    cube_image.reinterpret_stacked_2d_as_array(6);
    cube_image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::Cube),
        ..default()
    });
    let cube_handle = images.add(cube_image);

    for fine_quad in &fine_quads {
        if let Some(mat) = fine_materials.get_mut(&fine_quad.0) {
            mat.marble_texture = cube_handle.clone();
        }
    }

    // The raw 2D-strip loading handle is never bound to anything and never
    // will be again -- drop it now rather than leaking it for the rest of
    // the session. `cubemap.done` (not removing the resource entirely) is
    // what stops this system doing any of this a second time -- `main.rs`
    // schedules it unconditionally every frame, so removing the resource
    // outright would panic the next time it runs.
    images.remove(&cubemap.loading);
    cubemap.done = true;
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
const MENGER_OSCILLATING_SPHERE_COLOR: Vec3 = Vec3::new(0.75, 0.25, 0.85);

/// The CSG scene + its live parameter table, kept around so per-frame systems
/// can animate params and (M5) run the marble's CPU distance/nearest-point
/// collision queries against `object` without rebuilding the tree or
/// regenerating the shader.
#[derive(Resource)]
pub struct SceneState {
    pub kind: SceneKind,
    /// The scene tree, queried each physics tick by
    /// `physics_sys::marble_physics_tick` against every scene's marble.
    pub object: Object,
    pub params: Params,
    /// Every `(param, expr)` this scene wants re-evaluated once per
    /// simulated tick (`crate::expr` module doc) -- written by
    /// `physics_sys::marble_physics_tick_impl` (offline path directly,
    /// connected path via `RollbackSim::advance`/`receive_inputs`), read
    /// here only to decide whether to re-upload `params` to the GPU this
    /// frame. Empty for every scene except
    /// [`SceneKind::MengerOscillatingSphere`] today, but any future scene
    /// that wants an animated param just needs to populate this instead of
    /// adding another one-off `SceneHandles` match arm to `update_material`.
    pub animations: Vec<(ScalarParam, Expr)>,
    pub handles: SceneHandles,
    pub material: Handle<FineMarcherMaterial>,
    pub params_buffer: Handle<ShaderStorageBuffer>,
    /// The live marble list (`FineMarcherMaterial::marbles`/
    /// `MARBLE_BUFFER_BINDING`'s doc), rewritten every frame by
    /// `update_material` from `MarbleState::marbles`.
    pub marbles_buffer: Handle<ShaderStorageBuffer>,
    /// The scene's world-space bounding sphere (`object.bounding_sphere`),
    /// packed as `xyz = center, w = radius` (`w <= 0.0` means "no bound" --
    /// `bounding_sphere` returned `None`), computed once here at setup
    /// rather than every frame: it only depends on `params`, which this
    /// scene's own tree either never changes after setup or (Demo's
    /// opt-in `MM_ANIMATE_FRACTAL`) only nudges `ang1` -- a `Fold::Rotate`
    /// angle, which `Fold::unfold_bounding_sphere` handles as an exact
    /// isometry regardless of angle, so the bound stays valid without
    /// needing to be recomputed as that animates. Every pass's material
    /// writes this same value into its own `SceneUniforms::bounding` each
    /// frame (`update_material`/`mrrm::update_coarse_material`/
    /// `shadow_pass::update_shadow_material`).
    pub bounding_sphere: Vec4,
}

/// `object.bounding_sphere(params)` packed for `SceneUniforms::bounding`/
/// `ray_sphere_clip` (`marble_csg::codegen`): `xyz = center, w = radius`,
/// or an all-zero (`radius <= 0.0`, "no bound") vector on `None`.
fn pack_bounding_sphere(object: &Object, params: &Params) -> Vec4 {
    match object.bounding_sphere(params) {
        Some((center, radius)) => center.extend(radius),
        None => Vec4::ZERO,
    }
}

/// `marbles` packed for [`FineMarcherMaterial::marbles`]/
/// `marble_csg::codegen::MARBLE_BUFFER_BINDING`: one `vec4<f32>` per marble
/// (`xyz = center, w = radius`), same order as `MarbleState::marbles` (so
/// `marble_idx` in the shader is directly `local_player_index`-comparable if
/// a future feature needs that).
fn pack_marbles(marbles: &[Marble]) -> Vec<Vec4> {
    marbles.iter().map(|m| m.pos.extend(m.rad)).collect()
}

/// Startup system: builds the selected scene ([`SceneKind::from_config`]),
/// generates its WGSL, and spawns the fullscreen quad that renders it
/// (DESIGN.md §8).
#[allow(clippy::too_many_arguments)]
pub fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<FineMarcherMaterial>>,
    mut shaders: ResMut<Assets<Shader>>,
    mut storage_buffers: ResMut<Assets<ShaderStorageBuffer>>,
    mut camera_orbit: ResMut<CameraOrbit>,
    asset_server: Res<AssetServer>,
    mut images: ResMut<Assets<Image>>,
) {
    let kind = SceneKind::from_config();
    let mut params = Params::new();

    // CameraOrbit::default() is tuned for the Demo scene specifically (aimed
    // at the marble's actual resting-surface normal, at a marble_rad-scaled
    // distance — see its doc comment). The static display fractals are a
    // completely different world scale (menger_sponge/menger_sphere are
    // roughly unit-scale after their final 0.33 shrink), so that default's
    // yaw/pitch/distance don't make sense here — use a view tuned for these
    // scenes' own scale instead.
    //
    // `yaw`/`pitch` were originally chosen (at `distance = 3.0`, then later
    // `6.0`) to find an *un-embedded* view of the sponge's exterior corner:
    // an early attempt at `distance = 3.0` put the camera close enough that
    // most of a 32x18 probe grid hit the surface within `t ~= 0.4` world
    // units, giving a normal-estimation epsilon (`1e-4 * max(t, 0.05)`) so
    // tiny that `calc_normal`'s central difference was dominated by float32
    // noise -- ~85% of sampled pixels had a degenerate raw normal, the exact
    // speckle/static look. This was reproduced identically across
    // `MENGER_DEPTH` 2..14 (recursion depth wasn't the cause), confirming a
    // "camera embedded in/hugging the geometry" problem specific to being
    // too close along *this* yaw/pitch's view ray -- the same yaw/pitch is
    // kept here since it's a verified-safe viewing angle of the corner.
    //
    // `distance = 1.2` is tuned relative to the marble's spawn point
    // (`spawn_params`, not the origin): since the camera always orbits
    // `marble.pos` (`update_material`), and the marble now spawns in the
    // open-air pocket just in front of the corner (see `spawn_params`'s doc
    // for why, and how that point was found) rather than deep inside the
    // sponge, this is a close, marble-centric framing -- similar in spirit
    // to the Demo scene's own close default -- verified visually to show
    // the marble prominently with the corner as a backdrop, no
    // embedded-camera speckle.
    let is_menger = matches!(
        kind,
        SceneKind::MengerSponge | SceneKind::MengerSphere | SceneKind::MengerOscillatingSphere
    );
    if is_menger {
        camera_orbit.orientation = CameraOrbit::orientation_from_yaw_pitch(0.8, 0.35);
        camera_orbit.distance = if kind == SceneKind::MengerOscillatingSphere {
            // The other two Menger scenes are static, so `distance = 1.2`'s
            // tight corner-detail framing (see the doc above) is the right
            // shot. This scene's entire point is a *global* structural
            // change -- the center hollowing out while the corners survive
            // -- which that framing can't show at all: verified visually
            // (screenshot) that at `distance = 1.2` the same close-up patch
            // of corner surface renders identically at both bite-radius
            // extremes, since everything in frame sits well outside even
            // `MENGER_BITE_MAX_RADIUS`. Pulled back enough to fit the whole
            // ~6-unit sponge (bounding half-extent `MENGER_BITE_MAX_RADIUS
            // ~= 3.03`) in frame with margin.
            9.0
        } else {
            1.2
        };
    }

    let mut animations: Vec<(ScalarParam, Expr)> = Vec::new();
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
        SceneKind::MengerOscillatingSphere => {
            let (object, osc_handles) = menger_oscillating_sphere(&mut params);
            set_menger_params(
                &mut params,
                &osc_handles.menger,
                MENGER_DEPTH,
                MENGER_OSCILLATING_SPHERE_COLOR,
            );
            // The physics tick overwrites this every tick once it starts
            // running; the initial value just needs to match what
            // `radius_anim` evaluates to at tick 0 (MENGER_BITE_MIN_RADIUS)
            // so the very first rendered frame, before any tick has run,
            // isn't briefly wrong.
            params.set_scalar(osc_handles.radius, MENGER_BITE_MIN_RADIUS);
            animations.push((osc_handles.radius, osc_handles.radius_anim.clone()));
            (object, SceneHandles::MengerOscillatingSphere(osc_handles))
        }
    };

    let wgsl = generate_shader(&object);
    shaders.insert(
        MARCHER_SHADER_HANDLE.id(),
        Shader::from_wgsl(wgsl, "generated://marcher.wgsl"),
    );

    let params_buffer = storage_buffers.add(ShaderStorageBuffer::from(params.slots().to_vec()));
    let bounding_sphere = pack_bounding_sphere(&object, &params);

    let spawn = kind.spawn_params();
    // Multiplayer milestone 0: spawn a few extra marbles in `Demo` (the one
    // scene with a real, sizeable resting platform, not the Menger scenes'
    // narrow open-air pocket) purely so marble-vs-marble collision is
    // visually verifiable without inventing per-scene spawn layouts this
    // milestone doesn't need. Stacked above the primary spawn (not spread
    // out in X/Z): `beware_of_bumps::START` rests on a ledge only about 1x
    // the marble's own radius wide (`animate_fractal`'s doc below), so they
    // fall onto/into each other and the ledge together instead of each
    // needing their own independently-verified flat spot to land on.
    // Exact same X/Z, staggered only in Y: guarantees they actually collide
    // with each other on the way down (the lowest one lands first, the ones
    // still falling above it are on the same vertical line and *must* pass
    // through its combined-radius zone), rather than each independently
    // landing on unrelated nearby terrain with an X/Z offset large enough
    // to never truly overlap.
    let extra_marbles = if kind == SceneKind::Demo { 2 } else { 0 };
    let start_positions: Vec<Vec3> = (0..=extra_marbles)
        .map(|i| spawn.start + Vec3::new(0.0, spawn.rad * 4.0 * i as f32, 0.0))
        .collect();
    let marbles: Vec<Marble> = start_positions
        .iter()
        .map(|&start| Marble::spawn(start, spawn.rad))
        .collect();
    let marbles_buffer = storage_buffers.add(ShaderStorageBuffer::from(pack_marbles(&marbles)));

    // `coarse`/`shadow: Handle::default()` are placeholders -- `mrrm::setup_mrrm_pipeline`
    // and `shadow_pass::setup_shadow_pipeline` (Startup systems chained to
    // run after this one) correct them to the real render targets once
    // those `Image`s exist. Every Startup system finishes before the first
    // frame is ever rendered, so these placeholders are never actually read
    // by the GPU. `marble_texture` gets exactly the same placeholder
    // treatment, for the same reason -- `finalize_marble_cubemap` (`Update`)
    // redirects it to a freshly-built, already-cube-shaped `Image` once the
    // PNG finishes loading (see that fn's doc for why it builds a *new*
    // `Image` rather than reinterpreting the loading handle's own image in
    // place: binding the raw loading handle here directly, even as a
    // temporary stand-in, hits the exact same "already-bound `Image` that
    // later changes shape" issue that broke production once already).
    let marble_cubemap_handle: Handle<Image> = asset_server.load("marble_cubemap.png");
    let material = materials.add(FineMarcherMaterial {
        scene: SceneUniforms::default(),
        params: params_buffer.clone(),
        coarse: Handle::default(),
        shadow: Handle::default(),
        marbles: marbles_buffer.clone(),
        marble_texture: images.add(make_placeholder_cubemap()),
    });
    commands.insert_resource(MarbleCubemap { loading: marble_cubemap_handle, done: false });

    // Renders straight to the primary window (no adaptive-resolution
    // render-to-texture indirection -- removed after it turned out to cause
    // visible jitter even throttled way down, not worth the complexity for
    // now). `IsDefaultUiCamera` pins the FPS overlay (`fps_overlay.rs`) to
    // this camera explicitly rather than relying on bevy_ui's "highest-order
    // camera targeting the window" fallback -- there's only one
    // window-targeting camera now, so the fallback would pick this one
    // anyway, but being explicit doesn't depend on that staying true.
    commands.spawn((Camera2d, MarcherCamera, IsDefaultUiCamera));
    commands.spawn((
        Mesh2d(meshes.add(Rectangle::new(1.0, 1.0).mesh())),
        MeshMaterial2d(material.clone()),
        Transform::default(),
        MarcherQuad,
    ));

    commands.insert_resource(SceneState {
        kind,
        object,
        params,
        animations,
        handles,
        material,
        params_buffer,
        marbles_buffer,
        bounding_sphere,
    });

    commands.insert_resource(MultiplayerSession::new_solo(marbles.clone()));

    commands.insert_resource(MarbleState {
        marbles,
        cfg: PhysicsConfig::default(),
        start_positions,
        kill_y: spawn.kill_y,
        local_player_index: 0,
    });
}

/// `Update` system: applies the host's most recently received scene-sync
/// bundle, if `physics_sys::marble_physics_tick_impl`'s `FixedUpdate` polling
/// stashed one this frame (`PendingSceneSync`'s doc) -- decodes it, replaces
/// `SceneState`'s tree/params/animations wholesale, and pushes the two
/// GPU-visible consequences of that (the generated shader, the params
/// storage buffer) so rendering picks up the new scene with no further
/// special-casing, exactly as if it had been built by `setup` in the first
/// place. Runs before `update_material` in the `Update` chain (`main.rs`) so
/// this frame's material sync already sees the new `bounding_sphere`, not a
/// stale one.
///
/// A no-op decode failure (`SceneBundle::from_bytes` returning `None`) is
/// swallowed with a `warn!` rather than a panic -- same defensive posture as
/// every other "this arrived over the network from another peer" decode in
/// this codebase (`net.rs`'s `decode_messages`): a malformed bundle can't
/// happen between two clients running the same build, but silently ignoring
/// one is a safer failure mode than trusting corrupt geometry.
pub fn apply_pending_scene_sync(
    mut pending_scene: ResMut<PendingSceneSync>,
    mut scene_state: ResMut<SceneState>,
    mut mp: ResMut<MultiplayerSession>,
    mut shaders: ResMut<Assets<Shader>>,
    mut storage_buffers: ResMut<Assets<ShaderStorageBuffer>>,
) {
    let Some(bytes) = pending_scene.0.take() else { return };
    let Some(bundle) = SceneBundle::from_bytes(&bytes) else {
        warn!("multiplayer: received an undecodable scene-sync bundle -- ignoring it");
        return;
    };
    let SceneBundle { object, params, animations } = bundle;

    let wgsl = generate_shader(&object);
    shaders.insert(MARCHER_SHADER_HANDLE.id(), Shader::from_wgsl(wgsl, "generated://marcher.wgsl"));
    if let Some(buffer) = storage_buffers.get_mut(&scene_state.params_buffer) {
        buffer.set_data(params.slots().to_vec());
    }
    scene_state.bounding_sphere = pack_bounding_sphere(&object, &params);
    scene_state.object = object;
    scene_state.params = params;
    scene_state.animations = animations;

    mp.mark_scene_synced();
    info!("multiplayer: applied the host's authoritative scene sync");
}

/// Keeps the fullscreen quad's world size equal to the window's pixel size
/// every frame (cheap, robust across resizes — DESIGN.md §8).
pub fn sync_quad_scale(
    windows: Query<&Window, With<PrimaryWindow>>,
    mut quads: Query<&mut Transform, With<MarcherQuad>>,
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
/// local player's marble), timing, and the live marble list into the
/// material (DESIGN.md §7/§8); re-syncs fractal params only if
/// `animate_fractal()` is enabled (otherwise the static params `setup()`
/// wrote stay untouched, matching what the marbles' physics collides
/// against).
/// Thin timing wrapper -- see `fps_overlay::PhaseTimings`'s doc for exactly
/// what this measures (CPU-side uniform computation, not GPU execution).
/// Uses `web_time::Instant`, not `std::time::Instant`: the latter panics
/// unconditionally on `wasm32-unknown-unknown` ("time not implemented on
/// this platform" -- no OS clock on that target), which took production
/// down the first time this readout shipped (`bcb4d50`/`d5785ed`)
/// -- `web_time` is a drop-in replacement that routes through
/// `Performance.now()` via `web_sys` on wasm, and is a transparent
/// pass-through to real `std::time` everywhere else.
#[allow(clippy::too_many_arguments)]
pub fn update_material(
    time: Res<Time>,
    orbit: Res<CameraOrbit>,
    marble_state: Res<MarbleState>,
    windows: Query<&Window, With<PrimaryWindow>>,
    shadow_render_target: Res<crate::shadow_pass::ShadowRenderTarget>,
    perfprobe: Res<crate::perfprobe::PerfProbeState>,
    scene_state: ResMut<SceneState>,
    materials: ResMut<Assets<FineMarcherMaterial>>,
    storage_buffers: ResMut<Assets<ShaderStorageBuffer>>,
    mut timings: ResMut<crate::fps_overlay::PhaseTimings>,
) {
    let start = web_time::Instant::now();
    update_material_impl(
        time,
        orbit,
        marble_state,
        windows,
        shadow_render_target,
        perfprobe,
        scene_state,
        materials,
        storage_buffers,
    );
    timings.record("fine", start.elapsed());
}

#[allow(clippy::too_many_arguments)] // SystemParam count, unchanged from before the timing wrapper split
fn update_material_impl(
    time: Res<Time>,
    orbit: Res<CameraOrbit>,
    marble_state: Res<MarbleState>,
    windows: Query<&Window, With<PrimaryWindow>>,
    shadow_render_target: Res<crate::shadow_pass::ShadowRenderTarget>,
    perfprobe: Res<crate::perfprobe::PerfProbeState>,
    mut scene_state: ResMut<SceneState>,
    mut materials: ResMut<Assets<FineMarcherMaterial>>,
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

    // Scene-agnostic animated-param upload: `physics_sys::
    // marble_physics_tick_impl` already wrote this tick's evaluated values
    // straight into `scene_state.params` (offline path directly, connected
    // path via `RollbackSim` -- see `SceneState::animations`'s doc for why
    // evaluation doesn't happen here). This system just has to get the
    // result onto the GPU every frame; skipped entirely for scenes with no
    // animations (the common case) rather than uploading an unchanged
    // buffer every frame for no reason.
    if !scene_state.animations.is_empty() {
        if let Some(buffer) = storage_buffers.get_mut(&scene_state.params_buffer) {
            buffer.set_data(scene_state.params.slots().to_vec());
        }
    }

    let (aspect, resolution_height) = windows
        .single()
        .map(|w| (w.width() / w.height().max(1.0), w.physical_height() as f32))
        .unwrap_or((1.0, 1.0));

    // Every scene has a real marble now (`SceneKind::has_marble`): the
    // camera always follows the local player's.
    let target = marble_state.local_marble().pos;
    let (eye, right, up, forward) = orbit.eye_and_basis(target);

    // Live marble list -> its storage buffer, every frame (multiplayer
    // milestone 0 -- mirrors the params-buffer per-frame write pattern just
    // above, same reasoning: a pure buffer write, no shader recompile).
    if let Some(buffer) = storage_buffers.get_mut(&scene_state.marbles_buffer) {
        buffer.set_data(pack_marbles(&marble_state.marbles));
    }

    if let Some(mat) = materials.get_mut(&scene_state.material) {
        mat.scene = SceneUniforms {
            cam_pos: eye.extend(0.0),
            cam_right: right.extend(0.0),
            cam_up: up.extend(0.0),
            cam_forward: forward.extend(1.5),
            sun: beware_of_bumps::sun_dir().extend(0.0),
            sun_col: beware_of_bumps::SUN_COL.extend(0.0),
            bg_col: beware_of_bumps::BG.extend(0.0),
            // w: MRRM on/off (`crate::mrrm::mrrm_enabled`) -- a uniform flag
            // rather than skipping the coarse pass/texture binding entirely,
            // so every frame's cameras/passes are identical whether MRRM is
            // on or off and an `MM_MRRM=0` vs `MM_MRRM=1` A/B screenshot
            // comparison at a fixed camera state only ever differs in this
            // one value (see `mrrm::mrrm_enabled`'s doc).
            misc: Vec4::new(aspect, t, resolution_height, if crate::mrrm::mrrm_enabled() { 1.0 } else { 0.0 }),
            // x: the shadow pass's own render-target height (`sample_shadow`'s
            // `sz` computation), y: `MM_SHADOW_LOD` on/off (same
            // A/B-comparability reasoning as MRRM's `misc.w` above).
            misc2: Vec4::new(
                shadow_render_target.size.y as f32,
                if crate::shadow_pass::shadow_lod_enabled() { 1.0 } else { 0.0 },
                perfprobe.fine_max_steps_override,
                0.0,
            ),
            bounding: scene_state.bounding_sphere,
        };
    }
}
