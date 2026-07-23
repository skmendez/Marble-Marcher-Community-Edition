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
//! `Vec<Vec4>` field with `#[storage(1, read_only)]`. Neither that nor the
//! `Handle<ShaderStorageBuffer>` indirection this used before are used
//! anymore: all per-frame data (scene uniforms, params, marbles) now lives
//! in `gpu::MarcherGpuBuffers` -- persistent GPU buffers written in place
//! each frame with `queue.write_buffer` instead of round-tripping through
//! `Assets` mutation (see `gpu.rs`'s module doc for why: the old
//! `Assets`-mutation path was a confirmed, unbounded GPU resource leak --
//! ~179 buffers/sec, ~359 bind groups/sec, zero destroys, measured directly
//! against production). The materials here hand-implement `AsBindGroup` to
//! bind those shared buffers, so a material asset is only ever *mutated*
//! (and its bind group rebuilt) when a render-target texture handle changes
//! (resize) or a storage buffer's element count genuinely changes (a
//! multiplayer join growing the marble count, or a scene resync) -- see
//! `update_frame_data`'s doc.

use bevy::asset::weak_handle;
use bevy::ecs::system::lifetimeless::SRes;
use bevy::ecs::system::SystemParamItem;
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::camera::{RenderTarget, Viewport};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    sampler, storage_buffer_read_only, texture_2d, texture_cube, uniform_buffer,
};
use bevy::render::render_resource::{
    AsBindGroup, AsBindGroupError, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
    BindGroupLayoutEntry, BindingResources, Extent3d, PreparedBindGroup, SamplerBindingType,
    ShaderRef, ShaderStages, TextureDimension, TextureFormat, TextureSampleType,
    TextureUsages, TextureViewDescriptor, TextureViewDimension, UnpreparedBindGroup,
};
use bevy::render::renderer::RenderDevice;
use bevy::render::texture::GpuImage;
use bevy::render::view::RenderLayers;
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
use marble_csg::{Object, Params, ScalarParam, Scene};

use crate::camera::CameraOrbit;
use crate::gpu::{MarcherFrameData, MarcherGpuBuffers};
use crate::mrrm::{CoarseMarcherMaterial, CoarseQuad};
use crate::physics_sys::{MarbleState, MultiplayerSession, PendingSceneSync};
use crate::shadow_pass::{ShadowMarcherMaterial, ShadowQuad};

/// Which scene to build, selected via
/// `MM_SCENE=demo|classic_only|menger_sponge|menger_sphere|menger_oscillating_sphere`
/// on native, or `?scene=<same value>` in the URL on the deployed web build
/// -- see `config::Config`'s doc for why this is read once into `Config`
/// rather than each caller re-parsing it. Defaults to `MengerOscillatingSphere`
/// either way if unset/unrecognized.
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
    /// Pure parse from an already-resolved `?scene=`/`MM_SCENE` value --
    /// reading the query param/env var itself is `config::Config`'s job now
    /// (`Config::from_env`), not this type's; kept as a pure function
    /// (rather than folded directly into `Config::from_env`) so it stays
    /// independently unit-testable without needing a real URL/environment.
    pub fn from_value(value: Option<&str>) -> Self {
        match value {
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
            // `Vec3::ZERO` (deep inside the sponge's internal void network)
            // has de(origin) ~= 1.4 -- not embedded, but *occluded*: from
            // the exterior corner view `setup`'s camera override uses for
            // these two static scenes, the origin sits behind the sponge's
            // own outer walls, so the marble was invisible there no matter
            // how large it was (confirmed: even `rad = 0.3` at the origin
            // was still fully hidden). This spawn point instead sits in the
            // open-air pocket just in front of that same exterior corner,
            // along the camera's default view ray -- found by probing
            // `Object::de` at points between the camera's eye and the
            // origin (`rust/csg/examples/spawn_probe.rs` in git history)
            // for a spot with a comfortable safety margin (de ~= 0.34, well
            // clear of `rad` below) that's still close to the visible
            // surface, not floating off in empty sky. `rad = 0.15` (up from
            // the original 0.05) is sized to read clearly at the
            // correspondingly closer orbit distance (`setup`'s Menger
            // camera override, below) rather than for fitting through the
            // sponge's recursive tunnels, since a marble spawned outside
            // the structure doesn't need to. `kill_y = -50.0` is a generous
            // "fell way out of the structure" bound for if `G` is toggled
            // to `Rolling` mode while in one of these scenes.
            Self::MengerSponge | Self::MengerSphere => MarbleSpawn {
                start: Vec3::new(3.32, 1.69, 3.22),
                rad: 0.15,
                kill_y: -50.0,
            },
            // Center spawn, unlike the two static scenes above: the whole
            // point of this scene is the oscillating bite sphere hollowing
            // out the center, so the marble sits right where that's
            // actually happening rather than off at a corner untouched by
            // it (verified this session -- de(origin) ~= 1.4 regardless of
            // bite phase, since the bite sphere only ever *removes*
            // material, never fills in the sponge's own existing internal
            // voids, so the origin is safe at every point in the
            // oscillation, not just at rest). Paired with `setup`'s camera
            // override pulling in to `distance = 1.2` for this scene too
            // (was `9.0`, framing the whole exterior instead) -- close and
            // centered puts the player inside the hollowing-out effect
            // itself instead of watching it from outside.
            Self::MengerOscillatingSphere => MarbleSpawn { start: Vec3::ZERO, rad: 0.15, kill_y: -50.0 },
        }
    }
}

/// The fractal-tree-specific parameter handles, so a future per-scene
/// animation toggle can animate `ang1` for [`SceneKind::Demo`] without
/// needing to know about [`MengerHandles`] (the static display fractals
/// don't have an `ang1`).
pub enum SceneHandles {
    /// The `Demo`/`ClassicOnly` scenes' old wall-clock-driven `ang1` wobble
    /// (`MM_ANIMATE_FRACTAL`) that used to read this back every frame was
    /// removed (dead on the deployed web build, and wall-clock-driven
    /// geometry animation is exactly the determinism anti-pattern this
    /// session's `Expr`/tick-driven work has been eliminating elsewhere) --
    /// not read anywhere now, kept for the same reason as the other two
    /// variants below: a future, properly deterministic per-scene animation
    /// toggle.
    #[allow(dead_code)]
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
        /// height in pixels (drives the marcher's distance-scaled
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
        /// sees), w the marble cubemap's current Y-axis rotation angle in
        /// radians (fine pass's material only, same as the shadow-LOD/
        /// perfprobe lanes above -- coarse/shadow don't shade the marble at
        /// all) -- see `update_frame_data_impl`'s doc for why this is a
        /// deterministic function of the shared simulation tick, not
        /// wall-clock time.
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

#[cfg(test)]
mod scene_uniforms_abi_tests {
    // Rust has no runtime struct-field reflection, so this can't generate
    // its expected list *from* the `SceneUniforms` struct above -- it's a
    // manually-kept mirror of that struct's field order, checked against
    // `marble_csg::codegen::SCENE_UNIFORMS_FIELD_NAMES` (the single
    // authoritative list the WGSL-side struct is actually *generated*
    // from, see that constant's doc). If either side adds, removes, or
    // reorders a field without updating the other, this fails loudly at
    // `cargo test` time instead of silently misrendering.
    const RUST_FIELD_ORDER: [&str; 10] = [
        "cam_pos", "cam_right", "cam_up", "cam_forward", "sun", "sun_col", "bg_col", "misc",
        "misc2", "bounding",
    ];

    #[test]
    fn rust_field_order_matches_the_wgsl_codegen_source_of_truth() {
        assert_eq!(RUST_FIELD_ORDER, marble_csg::codegen::SCENE_UNIFORMS_FIELD_NAMES);
    }
}

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

/// The fine (full-resolution) marcher's material. Bindings 0/1/4 (scene
/// uniforms, params, marble list) come from `gpu::MarcherGpuBuffers`'s
/// persistent shared buffers, not from fields on this struct -- see the
/// hand-written [`AsBindGroup`] impl below and `gpu.rs`'s module doc. The
/// remaining fields are the MRRM coarse pre-pass's cached hit-distance
/// render target (`mrrm.rs`), the half-resolution shadow/AO pass's cached
/// visibility (`shadow_pass.rs`), and the marble's cubemap texture
/// (`MarbleCubemap`'s doc). `coarse`/`shadow` need no paired sampler: the
/// generated shader only ever reads them with `textureLoad` (exact texel;
/// `sample_shadow` does its own hand-rolled depth-aware 4-tap blend for the
/// shadow one), never `textureSample`. `marble_texture` does need one --
/// the shader samples it with `textureSample` (a filtered, direction-vector
/// lookup), not `textureLoad` (`marble_csg::codegen::MARBLE_TEXTURE_BINDING`'s
/// doc) -- bound from its own `GpuImage::sampler` below. Mutating any of
/// these three handles (render-target resize, or the cubemap finishing its
/// async load) is what re-prepares this material -- which is exactly the
/// case where the bind group genuinely must be rebuilt (it holds the old
/// texture's view). See `marble_csg::codegen::COARSE_TEXTURE_BINDING`'s doc
/// for a known, environment-specific (llvmpipe-only) crash this exact data
/// flow triggers in this project's native test sandbox -- not a bug in this
/// binding/shader.
#[derive(Asset, TypePath, Clone)]
pub struct FineMarcherMaterial {
    pub coarse: Handle<Image>,
    pub shadow: Handle<Image>,
    pub marble_texture: Handle<Image>,
}

impl AsBindGroup for FineMarcherMaterial {
    type Data = ();
    type Param = (SRes<MarcherGpuBuffers>, SRes<RenderAssets<GpuImage>>);

    fn label() -> Option<&'static str> {
        Some("fine_marcher_material")
    }

    fn as_bind_group(
        &self,
        layout: &BindGroupLayout,
        render_device: &RenderDevice,
        (buffers, images): &mut SystemParamItem<'_, '_, Self::Param>,
    ) -> Result<PreparedBindGroup<Self::Data>, AsBindGroupError> {
        // `RetryNextUpdate` until `gpu::write_marcher_buffers`'s first run
        // has allocated the shared buffers and the render targets/cubemap
        // exist as GPU images -- the material prepare machinery re-attempts
        // next frame, so startup order isn't load-bearing here.
        let scene = buffers.fine_scene.binding().ok_or(AsBindGroupError::RetryNextUpdate)?;
        let params = buffers.params.binding().ok_or(AsBindGroupError::RetryNextUpdate)?;
        let marbles = buffers.marbles.binding().ok_or(AsBindGroupError::RetryNextUpdate)?;
        let coarse = images.get(&self.coarse).ok_or(AsBindGroupError::RetryNextUpdate)?;
        let shadow = images.get(&self.shadow).ok_or(AsBindGroupError::RetryNextUpdate)?;
        let marble_texture =
            images.get(&self.marble_texture).ok_or(AsBindGroupError::RetryNextUpdate)?;
        let bind_group = render_device.create_bind_group(
            Self::label(),
            layout,
            &BindGroupEntries::with_indices((
                (0, scene),
                (1, params),
                (2, &coarse.texture_view),
                (3, &shadow.texture_view),
                (4, marbles),
                (5, &marble_texture.texture_view),
                (6, &marble_texture.sampler),
            )),
        );
        Ok(PreparedBindGroup { bindings: BindingResources(Vec::new()), bind_group, data: () })
    }

    fn unprepared_bind_group(
        &self,
        _layout: &BindGroupLayout,
        _render_device: &RenderDevice,
        _param: &mut SystemParamItem<'_, '_, Self::Param>,
        _force_no_bindless: bool,
    ) -> Result<UnpreparedBindGroup<Self::Data>, AsBindGroupError> {
        // The shared-buffer bindings can't be expressed as owned resources;
        // `as_bind_group` above is the real implementation.
        Err(AsBindGroupError::CreateBindGroupDirectly)
    }

    fn bind_group_layout_entries(
        _render_device: &RenderDevice,
        _force_no_bindless: bool,
    ) -> Vec<BindGroupLayoutEntry> {
        BindGroupLayoutEntries::with_indices(
            ShaderStages::FRAGMENT,
            (
                (0, uniform_buffer::<SceneUniforms>(false)),
                (1, storage_buffer_read_only::<Vec<Vec4>>(false)),
                (2, texture_2d(TextureSampleType::Float { filterable: true })),
                (3, texture_2d(TextureSampleType::Float { filterable: true })),
                (4, storage_buffer_read_only::<Vec<Vec4>>(false)),
                (5, texture_cube(TextureSampleType::Float { filterable: true })),
                (6, sampler(SamplerBindingType::Filtering)),
            ),
        )
        .to_vec()
    }
}

impl Material2d for FineMarcherMaterial {
    fn fragment_shader() -> ShaderRef {
        MARCHER_SHADER_HANDLE.into()
    }
}

/// Adaptive-resolution plumbing (GPU perf plan milestone 5) -- stage 1:
/// the tier-switching decision logic doesn't exist yet (a later, separate
/// stage), so `active_size` is pinned to `max_size` forever here, but the
/// full render-to-texture/present-pass architecture is real and live so
/// that stage can land as a pure logic change with no new Bevy-render
/// plumbing risk.
///
/// A *previous* attempt at this (now fully reverted, see `MarcherCamera`'s
/// doc) failed for a specific, now-avoided reason: every time its
/// resolution scale changed, it rebuilt its offscreen render target as a
/// **new** GPU `Image` -- a heavy operation, done several times over a few
/// seconds under sustained load, which is what actually read as "visible
/// jitter" (not a flaw in its hysteresis logic, which was fine). This
/// design sidesteps that entirely: [`FineRenderTarget`]'s backing `Image`
/// is allocated once, at the window's native size, and only ever rebuilt
/// on a genuine window resize (as rare as `mrrm.rs`'s `CoarseRenderTarget`
/// resize, which uses the identical "build new `Image`, redirect handles"
/// pattern for the same underlying Bevy render-target-freeze reason). A
/// resolution *tier* change becomes a `Camera.viewport` mutation -- a
/// plain component write, not a GPU resource recreation -- confirmed
/// against Bevy 0.16.1's own source to genuinely restrict fragment-shader
/// invocation to the viewport's sub-rectangle (a real perf win), not just
/// apply a cosmetic scissor.
///
/// **Load-bearing correctness detail**: Bevy's default `OrthographicProjection`
/// (`ScalingMode::WindowSize`, `Camera2d`'s default) sizes the visible
/// world-space area from the camera's *viewport* size, not the render
/// target's own size. If `MarcherQuad`'s `Transform.scale` and
/// `MarcherCamera`'s `Camera.viewport.physical_size` ever disagree, even
/// for one frame, the result is a zoomed-in crop of the frame's center,
/// not a correctly-framed downsample. `sync_fine_render_target_and_present`
/// writes both (and the present material's UV crop) together, atomically,
/// from this resource's own fields, so they can never drift apart.
#[derive(Resource)]
pub struct FineRenderTarget {
    /// The fixed backing texture, allocated at `max_size` and never
    /// resized except on a genuine window-size change.
    pub image: Handle<Image>,
    /// The backing texture's own fixed pixel size (the window's native
    /// *logical* size, deliberately not physical -- `setup`'s doc on why)
    /// -- NOT the currently-active tier.
    pub max_size: UVec2,
    /// The currently-active tier's pixel size -- what `Camera.viewport`
    /// restricts rendering to, what `MarcherQuad`'s `Transform.scale`
    /// must match, and what the fine pass's own `SceneUniforms::misc.z`
    /// (this pass's own render-target height) should reflect. Stage 1:
    /// always equal to `max_size`.
    pub active_size: UVec2,
}

/// Builds the fine pass's offscreen backing texture. `TextureFormat::
/// bevy_default()` (`Rgba8UnormSrgb`, confirmed via `bevy_image`'s
/// `BevyDefault` impl -- 4 bytes/pixel, hence the `[0u8; 4]` fill below),
/// not `Rgba16Float` like `mrrm.rs`'s coarse target: that HDR format
/// exists specifically so the coarse pass's raw (negative-capable,
/// unbounded) hit-distance value survives Bevy's intermediate-texture
/// blit uncompressed (`CoarseMarcherMaterial`'s doc) -- an entirely
/// different problem from this one. The fine pass already produces
/// final, tonemapped display color (`codegen.rs`'s `tonemap()`), so its
/// target just needs a normal 8-bit-per-channel format, matching what
/// this pass rendered into (the window's own swapchain format) before
/// this offscreen indirection existed at all.
fn make_fine_render_target_image(size: UVec2) -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0u8; 4],
        TextureFormat::bevy_default(),
        bevy::asset::RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::RENDER_ATTACHMENT;
    image
}

/// All present-pass entities (camera + quad) live on this layer -- layer
/// `1`, unused since the original (reverted) adaptive-resolution attempt's
/// own present pass was deleted; distinct from the fine marcher's
/// (implicit, unmarked) layer 0 so the present camera doesn't also render
/// the fine quad on top of its own output.
const PRESENT_LAYER: usize = 1;

/// Fixed weak handle for the present pass's (non-CSG-tree-derived, plain
/// Bevy blit/crop) shader.
const PRESENT_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("9f3c7b5e-2d4a-4e1f-8a6c-1b5d9e7f4c32");

const PRESENT_SHADER_WGSL: &str = r#"
#import bevy_sprite::mesh2d_vertex_output::VertexOutput

struct PresentUniform {
    // xy: the active tier's fractional UV coverage of the fixed backing
    // texture (half-texel-inset against the max size -- see
    // `sync_fine_render_target_and_present`'s doc for why this exact
    // formula, not a naive `active/max` ratio). zw unused padding.
    active_uv_scale: vec4<f32>,
}

@group(2) @binding(0) var<uniform> present: PresentUniform;
@group(2) @binding(1) var fine_tex: texture_2d<f32>;
@group(2) @binding(2) var fine_sampler: sampler;

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let uv = mesh.uv * present.active_uv_scale.xy;
    return textureSample(fine_tex, fine_sampler, uv);
}
"#;

/// The present pass's material: samples the fine pass's fixed-size
/// backing texture, cropped to the currently-active tier's sub-rectangle
/// (`active_uv_scale`) and stretched (bilinear) to fill the real window --
/// the actual "adaptive resolution" visual effect. A standard derived
/// `AsBindGroup` (not hand-rolled like `FineMarcherMaterial`'s persistent-
/// buffer bindings, `gpu.rs`'s module doc): this material only ever
/// mutates on a resolution-tier change or a window resize, both rare,
/// throttled events by design -- not the every-single-frame case that
/// made a hand-rolled persistent-buffer binding necessary for the fine/
/// coarse/shadow materials' own uniforms.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct PresentMaterial {
    #[uniform(0)]
    pub active_uv_scale: Vec4,
    #[texture(1)]
    #[sampler(2)]
    pub image: Handle<Image>,
}

impl Material2d for PresentMaterial {
    fn fragment_shader() -> ShaderRef {
        PRESENT_SHADER_HANDLE.into()
    }
}

/// Marker for the present pass's own camera -- the one and only
/// window-targeting camera now that `MarcherCamera` renders into
/// [`FineRenderTarget`] instead (see that resource's doc).
#[derive(Component)]
pub struct PresentCamera;

/// Marker for the present pass's fullscreen quad.
#[derive(Component)]
pub struct PresentQuad;

/// Computes `PresentMaterial::active_uv_scale`'s xy from the fine render
/// target's current `(active_size, max_size)`: a half-texel inset against
/// `max_size`, not a naive `active_size / max_size` ratio, so the maximum
/// sampled UV lands exactly on the last valid rendered texel's center
/// rather than bilinear-blending in whatever's just past the active
/// tier's crop boundary (stale content from a previous, larger tier, or
/// the clear color) -- `ClampToEdge` (the default sampler address mode)
/// doesn't help here since it only engages *outside* `[0,1]`, and this
/// crop boundary sits strictly inside it for any real downscale.
fn fine_active_uv_scale(active_size: UVec2, max_size: UVec2) -> Vec2 {
    Vec2::new(
        (active_size.x as f32 - 0.5) / max_size.x as f32,
        (active_size.y as f32 - 0.5) / max_size.y as f32,
    )
}

/// `Update` system: whenever [`FineRenderTarget`] changes (a real window
/// resize, or -- once a later stage adds tier-switching logic -- a
/// resolution-tier change), writes `MarcherCamera`'s `Camera.viewport`,
/// `MarcherQuad`'s `Transform.scale`, and the present material's
/// `active_uv_scale` together, atomically, all three derived from the
/// same `(active_size, max_size)` pair -- see [`FineRenderTarget`]'s doc
/// for why these must never disagree even for one frame. Guarded on
/// `Res<FineRenderTarget>::is_changed()` (true the frame it's first
/// inserted, and again only on a genuine subsequent write, not every
/// frame) rather than a manually-cached previous value, since every write
/// to this resource already only happens behind its own writer's "did
/// this actually change" gate (`resize_fine_render_target`'s own guard).
pub fn sync_fine_render_target_and_present(
    render_target: Res<FineRenderTarget>,
    mut fine_cameras: Query<&mut Camera, (With<MarcherCamera>, Without<PresentCamera>)>,
    mut fine_quads: Query<&mut Transform, (With<MarcherQuad>, Without<PresentQuad>)>,
    present_quads: Query<&MeshMaterial2d<PresentMaterial>, With<PresentQuad>>,
    mut present_materials: ResMut<Assets<PresentMaterial>>,
) {
    if !render_target.is_changed() {
        return;
    }
    let (w, h) = (render_target.active_size.x, render_target.active_size.y);
    for mut camera in &mut fine_cameras {
        camera.viewport = Some(Viewport {
            physical_position: UVec2::ZERO,
            physical_size: UVec2::new(w, h),
            depth: 0.0..1.0,
        });
    }
    for mut transform in &mut fine_quads {
        transform.scale.x = w as f32;
        transform.scale.y = h as f32;
    }
    let uv_scale = fine_active_uv_scale(render_target.active_size, render_target.max_size);
    for mesh_material in &present_quads {
        if let Some(mat) = present_materials.get_mut(&mesh_material.0) {
            mat.active_uv_scale = uv_scale.extend(0.0).extend(0.0);
        }
    }
}

/// `Update` system: keeps [`FineRenderTarget`]'s backing texture sized to
/// the window's current native *logical* size (`setup`'s doc on why
/// logical, not physical) -- only touches the GPU resource (a new `Image`)
/// when that size actually changed (a real window resize, or a
/// devicePixelRatio change on wasm, e.g. dragging the window to a
/// different-DPR display), mirroring `mrrm.rs`'s `resize_coarse_render_target`
/// exactly, including the "build a new `Image` and redirect handles
/// rather than resize in place" freeze-avoidance (that function's doc).
/// Stage 1: `active_size` tracks `max_size` 1:1 here too (tier pinned at
/// full resolution) -- a later stage's tier-switching logic will decide
/// `active_size` independently (never larger than `max_size`).
pub fn resize_fine_render_target(
    windows: Query<&Window, With<PrimaryWindow>>,
    mut render_target: ResMut<FineRenderTarget>,
    mut images: ResMut<Assets<Image>>,
    mut fine_cameras: Query<&mut Camera, With<MarcherCamera>>,
    present_quads: Query<&MeshMaterial2d<PresentMaterial>, With<PresentQuad>>,
    mut present_materials: ResMut<Assets<PresentMaterial>>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let native_size =
        UVec2::new(window.width().round().max(1.0) as u32, window.height().round().max(1.0) as u32);
    if native_size == render_target.max_size {
        return;
    }

    let new_handle = images.add(make_fine_render_target_image(native_size));

    for mut camera in &mut fine_cameras {
        camera.target = RenderTarget::from(new_handle.clone());
    }
    for mesh_material in &present_quads {
        if let Some(mat) = present_materials.get_mut(&mesh_material.0) {
            mat.image = new_handle.clone();
        }
    }

    images.remove(&render_target.image);
    render_target.image = new_handle;
    render_target.max_size = native_size;
    render_target.active_size = native_size;
}

/// Full down-and-up period for `oscillate_fine_resolution_tier`'s sweep.
const RES_OSCILLATE_PERIOD_SECS: f32 = 6.0;

/// The tier fraction (of `FineRenderTarget::max_size`) `oscillate_fine_
/// resolution_tier` never goes below -- 1/10th linear, i.e. 100x fewer
/// pixels than full resolution, per this tool's own purpose (a smoothness
/// check deliberately more extreme than any real tier the eventual
/// hysteresis-based tier-switching logic would ever pick, so a human
/// watching it can clearly see whether the transition itself is smooth
/// across the widest range this architecture supports, not just confirm
/// a realistic in-range change looks fine).
const RES_OSCILLATE_MIN_FRACTION: f32 = 0.1;

/// `Update` system, gated on `?res_oscillate=1`/`MM_RES_OSCILLATE=1`: a
/// manual, always-visible smoothness check for the adaptive-resolution
/// plumbing (`FineRenderTarget`'s doc) -- continuously sweeps
/// `active_size` between `max_size` (full resolution) and 1/10th of it in
/// each dimension (100x fewer total pixels) and back, via a plain
/// triangle wave over wall-clock time (`Time`, not the deterministic
/// simulation tick -- this is a debug-only visual tool with no physics/
/// rollback involvement, matching `perfprobe.rs`'s own wall-clock-based
/// timing for the same reason). Writes only through `FineRenderTarget`'s
/// `active_size` field, same as the (not-yet-built) real tier-switching
/// logic will -- `sync_fine_render_target_and_present` reacts atomically
/// either way, so this exercises the exact same downstream path a real
/// resolution change will use, not a separate toy code path.
pub fn oscillate_fine_resolution_tier(
    config: Res<crate::config::Config>,
    time: Res<Time>,
    mut render_target: ResMut<FineRenderTarget>,
) {
    if !config.res_oscillate_enabled {
        return;
    }
    let phase = (time.elapsed_secs() % RES_OSCILLATE_PERIOD_SECS) / RES_OSCILLATE_PERIOD_SECS;
    // Triangle wave: 0..0.5 sweeps down (1.0 -> MIN), 0.5..1.0 sweeps back up.
    let down = phase < 0.5;
    let leg_phase = if down { phase * 2.0 } else { (phase - 0.5) * 2.0 };
    let span = 1.0 - RES_OSCILLATE_MIN_FRACTION;
    let fraction = if down {
        1.0 - leg_phase * span
    } else {
        RES_OSCILLATE_MIN_FRACTION + leg_phase * span
    };
    let target = UVec2::new(
        ((render_target.max_size.x as f32 * fraction).round() as u32).max(1),
        ((render_target.max_size.y as f32 * fraction).round() as u32).max(1),
    );
    if render_target.active_size != target {
        render_target.active_size = target;
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
    asset_server: Res<AssetServer>,
    fine_quads: Query<&MeshMaterial2d<FineMarcherMaterial>, With<MarcherQuad>>,
    mut fine_materials: ResMut<Assets<FineMarcherMaterial>>,
) {
    if cubemap.done {
        return;
    }
    let Some(loaded) = images.get(&cubemap.loading) else {
        // A still-loading asset and a *failed* load both leave
        // `Assets<Image>` empty for this handle forever -- without this
        // check, a missing/404'd `marble_cubemap.png` silently leaves the
        // marble on its white placeholder cubemap with no indication
        // anything is wrong (exactly how it ships when serving `rust/web/`
        // without the README's `cp -r rust/app/assets rust/web/assets` step
        // -- `web/assets/` is gitignored -- or when running the native
        // binary directly instead of via `cargo run`, which resolves
        // `assets/` next to the executable; `BEVY_ASSET_ROOT` fixes that,
        // see scripts/headless_screenshot.sh).
        if let Some(bevy::asset::LoadState::Failed(err)) =
            asset_server.get_load_state(&cubemap.loading)
        {
            error!(
                "marble cubemap failed to load; marble will keep its plain white \
                 placeholder texture: {err} (if serving rust/web/, did you copy \
                 rust/app/assets in? if running the binary directly, is \
                 BEVY_ASSET_ROOT set?)"
            );
            // Stop polling -- the load will never complete.
            cubemap.done = true;
        }
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
    /// Not read anywhere now (see [`SceneHandles`]'s own doc on why) --
    /// kept alongside its variants for the same future-per-scene-toggle
    /// reason.
    #[allow(dead_code)]
    pub handles: SceneHandles,
    pub material: Handle<FineMarcherMaterial>,
    /// The scene's world-space bounding sphere (`object.bounding_sphere`),
    /// packed as `xyz = center, w = radius` (`w <= 0.0` means "no bound" --
    /// `bounding_sphere` returned `None`), computed once here at setup
    /// rather than every frame: even [`SceneKind::MengerOscillatingSphere`],
    /// the one scene whose params do animate post-setup, never needs this
    /// recomputed -- its bite sphere is strictly subtractive and its
    /// radius never exceeds `MENGER_BITE_MAX_RADIUS`, already accounted for
    /// in the Menger sponge's own (static) outer extent, so removing more
    /// material as the bite radius oscillates can never enlarge the bound.
    /// [`update_frame_data`] writes this same value into every pass's
    /// `SceneUniforms::bounding` each frame.
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

/// Startup system: builds the selected scene (`config.scene`), generates
/// its WGSL, and spawns the fullscreen quad that renders it (DESIGN.md §8).
#[allow(clippy::too_many_arguments)]
pub fn setup(
    mut commands: Commands,
    config: Res<crate::config::Config>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<FineMarcherMaterial>>,
    mut present_materials: ResMut<Assets<PresentMaterial>>,
    mut shaders: ResMut<Assets<Shader>>,
    mut camera_orbit: ResMut<CameraOrbit>,
    asset_server: Res<AssetServer>,
    mut images: ResMut<Assets<Image>>,
    windows: Query<&Window, With<PrimaryWindow>>,
) {
    let kind = config.scene;
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
        // Same close, marble-centric `distance = 1.2` for all three Menger
        // scenes now (see the doc above) -- `MengerOscillatingSphere` used
        // to pull back to `9.0` to frame the whole ~6-unit sponge from
        // outside, since its old corner spawn point never went anywhere
        // near the bite sphere's effect. Now that it spawns at the center
        // instead (`spawn_params`'s doc), close framing is the right shot
        // again: the marble sits right where the hollowing-out is actually
        // happening, so there's no need to zoom out to see it happen.
        camera_orbit.distance = 1.2;
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
    shaders.insert(
        PRESENT_SHADER_HANDLE.id(),
        Shader::from_wgsl(PRESENT_SHADER_WGSL, "generated://present.wgsl"),
    );

    let bounding_sphere = pack_bounding_sphere(&object, &params);

    let spawn = kind.spawn_params();
    // Multiplayer milestone 0: spawn a few extra marbles in `Demo` (the one
    // scene with a real, sizeable resting platform, not the Menger scenes'
    // narrow open-air pocket) purely so marble-vs-marble collision is
    // visually verifiable without inventing per-scene spawn layouts this
    // milestone doesn't need. Stacked above the primary spawn (not spread
    // out in X/Z): `beware_of_bumps::START` rests on a ledge only about 1x
    // the marble's own radius wide, so they fall onto/into each other and
    // the ledge together instead of each needing their own
    // independently-verified flat spot to land on.
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
        coarse: Handle::default(),
        shadow: Handle::default(),
        marble_texture: images.add(make_placeholder_cubemap()),
    });
    commands.insert_resource(MarbleCubemap { loading: marble_cubemap_handle, done: false });

    // Initial frame data: `update_frame_data` overwrites the uniforms every
    // frame; params/marbles just need to carry the real initial contents so
    // the very first `gpu::write_marcher_buffers` run allocates the storage
    // buffers at their final starting sizes (see `gpu.rs`'s module doc for
    // why growing past this later, e.g. a multiplayer join, is still safe).
    commands.insert_resource(MarcherFrameData {
        fine: SceneUniforms::default(),
        coarse: SceneUniforms::default(),
        shadow: SceneUniforms::default(),
        params: params.slots().to_vec(),
        marbles: pack_marbles(&marbles),
    });

    // Renders into `FineRenderTarget`'s fixed-size offscreen texture,
    // *not* straight to the window -- adaptive-resolution plumbing
    // (`FineRenderTarget`'s doc). A previous, since-fully-reverted
    // attempt at this rendered straight to the window with no offscreen
    // indirection at all; this replaces that.
    //
    // Logical, not physical, pixels: on a scaled-HiDPI display (e.g. a
    // native 4K panel set to "look like" 2560x1440, `devicePixelRatio:
    // 1.5`), physical pixels are `dpr^2` more ray-marched samples than the
    // display can actually distinguish once the compositor scales this
    // pass's output back down to fit the window -- real GPU cost for a
    // per-pixel-expensive shader, no perceptible benefit. A DPR=1 display
    // (the common case, and this dev environment's) has logical ==
    // physical, so this is a no-op there.
    let native_size = windows
        .single()
        .map(|w| UVec2::new(w.width().round().max(1.0) as u32, w.height().round().max(1.0) as u32))
        .unwrap_or(UVec2::new(1280, 720));
    let fine_image_handle = images.add(make_fine_render_target_image(native_size));

    commands.spawn((
        Camera2d,
        Camera {
            target: RenderTarget::from(fine_image_handle.clone()),
            ..default()
        },
        MarcherCamera,
        // GPU-timestamp-query profiling (`gpu_profile.rs`).
        crate::gpu_profile::GpuProfiledPass(crate::gpu_profile::FINE_PASS_NAME),
    ));
    commands.spawn((
        Mesh2d(meshes.add(Rectangle::new(1.0, 1.0).mesh())),
        MeshMaterial2d(material.clone()),
        Transform::default(),
        MarcherQuad,
    ));
    commands.insert_resource(FineRenderTarget {
        image: fine_image_handle.clone(),
        max_size: native_size,
        // Stage 1: tier pinned at full resolution -- see `FineRenderTarget`'s
        // doc.
        active_size: native_size,
    });

    // Present pass: the one and only window-targeting camera now,
    // stretching `FineRenderTarget`'s (possibly cropped-to-a-smaller-tier)
    // content to fill the real window (`PresentMaterial`'s doc).
    // `IsDefaultUiCamera` pins the FPS overlay (`fps_overlay.rs`) to this
    // camera explicitly rather than relying on bevy_ui's "highest-order
    // camera targeting the window" fallback -- there's only one
    // window-targeting camera, so the fallback would pick this one
    // anyway, but being explicit doesn't depend on that staying true.
    let present_material = present_materials.add(PresentMaterial {
        active_uv_scale: fine_active_uv_scale(native_size, native_size).extend(0.0).extend(0.0),
        image: fine_image_handle,
    });
    commands.spawn((
        Camera2d,
        Camera {
            // After the fine pass (default order 0) so this frame's fine
            // render actually exists by the time this pass samples it.
            order: 1,
            ..default()
        },
        RenderLayers::layer(PRESENT_LAYER),
        PresentCamera,
        IsDefaultUiCamera,
        // GPU-timestamp-query profiling (`gpu_profile.rs`).
        crate::gpu_profile::GpuProfiledPass(crate::gpu_profile::PRESENT_PASS_NAME),
    ));
    commands.spawn((
        Mesh2d(meshes.add(Rectangle::new(1.0, 1.0).mesh())),
        MeshMaterial2d(present_material),
        Transform::default(),
        RenderLayers::layer(PRESENT_LAYER),
        PresentQuad,
    ));

    commands.insert_resource(SceneState { kind, handles, material, bounding_sphere });

    // `RollbackSim` (inside `MultiplayerSession`) owns the one
    // authoritative `Scene` from here on -- see `marble_csg::Scene`'s doc
    // for why that's the actual correctness precondition rollback
    // determinism depends on. `object`/`params`/`animations` were only
    // ever borrowed above (shader generation, `pack_bounding_sphere`,
    // `MarcherFrameData`'s initial upload), so moving them here doesn't
    // disturb anything already done with them.
    let scene = Scene { object, params, animations };
    commands.insert_resource(MultiplayerSession::new_solo(scene, marbles.clone()));

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
/// A no-op decode failure (`Scene::from_bytes` returning `None`) is
/// swallowed with a `warn!` rather than a panic -- same defensive posture as
/// every other "this arrived over the network from another peer" decode in
/// this codebase (`net.rs`'s `decode_messages`): a malformed scene can't
/// happen between two clients running the same build, but silently ignoring
/// one is a safer failure mode than trusting corrupt geometry.
///
/// Also force-touches all three materials (one no-op `Assets::get_mut` each)
/// so their bind groups rebuild against `gpu::MarcherGpuBuffers::params`'s
/// buffer object, in case the new scene's param count differs from the old
/// one and `write_marcher_buffers` reallocated it underneath them (`gpu.rs`'s
/// module doc) -- cheap and correct to do unconditionally here since a scene
/// resync already forces a full shader regen (pipeline recompile) regardless,
/// unlike the per-frame path this would be wasteful on.
///
/// Applies `mp.pending_join`'s stashed tick/marbles atomically alongside the
/// scene via `RollbackSim::set_scene`, rather than reusing whatever tick/
/// marbles `mp.sim` currently happens to hold (`MultiplayerSession::
/// pending_join`'s doc on why the latter corrupts the rewind window). A
/// scene bundle without a paired `pending_join` shouldn't happen on the
/// current wire protocol (the host only ever sends one alongside a
/// growing resync, `apply_resync_payload`'s doc) -- logged loudly rather
/// than silently guessed at if it ever does.
#[allow(clippy::too_many_arguments)]
pub fn apply_pending_scene_sync(
    mut pending_scene: ResMut<PendingSceneSync>,
    mut scene_state: ResMut<SceneState>,
    mut mp: ResMut<MultiplayerSession>,
    mut marble_state: ResMut<MarbleState>,
    mut shaders: ResMut<Assets<Shader>>,
    mut frame: ResMut<MarcherFrameData>,
    mut fine_materials: ResMut<Assets<FineMarcherMaterial>>,
    mut coarse_materials: ResMut<Assets<CoarseMarcherMaterial>>,
    mut shadow_materials: ResMut<Assets<ShadowMarcherMaterial>>,
    coarse_quads: Query<&MeshMaterial2d<CoarseMarcherMaterial>, With<CoarseQuad>>,
    shadow_quads: Query<&MeshMaterial2d<ShadowMarcherMaterial>, With<ShadowQuad>>,
) {
    let Some(bytes) = pending_scene.0.take() else { return };
    let Some(scene) = Scene::from_bytes(&bytes) else {
        warn!("multiplayer: received an undecodable scene-sync payload -- ignoring it");
        return;
    };

    let wgsl = generate_shader(&scene.object);
    shaders.insert(MARCHER_SHADER_HANDLE.id(), Shader::from_wgsl(wgsl, "generated://marcher.wgsl"));
    frame.params.clear();
    frame.params.extend_from_slice(scene.params.slots());
    scene_state.bounding_sphere = pack_bounding_sphere(&scene.object, &scene.params);
    match mp.take_pending_join() {
        Some((tick, marbles)) => {
            mp.sim.set_scene(tick, scene, marbles.clone());
            marble_state.start_positions = marbles.iter().map(|m| m.pos).collect();
            marble_state.marbles = marbles;
        }
        None => {
            error!(
                "multiplayer: scene sync arrived with no paired join/resync state pending -- \
                 applying it to the current tick/marbles as a fallback, but this should be unreachable"
            );
            let current_tick = mp.sim.current_tick();
            let current_marbles = mp.sim.marbles().to_vec();
            mp.sim.set_scene(current_tick, scene, current_marbles);
        }
    }

    fine_materials.get_mut(&scene_state.material);
    for mesh_material in &coarse_quads {
        coarse_materials.get_mut(&mesh_material.0);
    }
    for mesh_material in &shadow_quads {
        shadow_materials.get_mut(&mesh_material.0);
    }

    mp.mark_scene_synced();
    info!("multiplayer: applied the host's authoritative scene sync");
}

/// Keeps the present pass's fullscreen quad world size equal to the
/// window's pixel size (robust across resizes -- DESIGN.md §8). This used
/// to scale `MarcherQuad` directly, back when the fine pass rendered
/// straight to the window; now that it renders into `FineRenderTarget`
/// instead (that resource's own doc), `MarcherQuad`'s scale is driven by
/// the active tier size instead (`sync_fine_render_target_and_present`),
/// and the present quad is the one that actually needs to match the
/// window 1:1. Deref-muts the `Transform` only when the size actually
/// differs, so change detection (and the transform propagation +
/// re-extraction it triggers) stays quiet on the vast majority of frames
/// where the window hasn't resized.
pub fn sync_quad_scale(
    windows: Query<&Window, With<PrimaryWindow>>,
    mut quads: Query<&mut Transform, With<PresentQuad>>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    for mut transform in &mut quads {
        if transform.scale.x != window.width() || transform.scale.y != window.height() {
            transform.scale.x = window.width();
            transform.scale.y = window.height();
        }
    }
}

/// One full marble-cubemap revolution every 2 seconds at the 60Hz physics
/// tick (`marble_rollback`'s tick rate) -- `update_frame_data_impl`'s
/// doc on why this drives the rotation angle instead of wall-clock time.
const ROTATION_PERIOD_TICKS: u64 = 120;

/// Per-frame system: writes the orbit camera basis (following the local
/// player's marble), timing, the live marble list, and each pass's own
/// target-resolution/flag lanes into [`MarcherFrameData`] -- plain CPU data
/// that `gpu::write_marcher_buffers` uploads in place into persistent
/// buffers (see `gpu.rs`). Replaces the three per-pass `update_material`/
/// `update_coarse_material`/`update_shadow_material` systems that used to
/// rewrite material assets (and thereby recreate buffers + bind groups)
/// every frame; re-uploads `params` to the GPU only if `scene_state.
/// animations` is non-empty (otherwise the static params `setup()` wrote
/// stay untouched, matching what the marbles' physics collides against).
///
/// All three passes share one camera basis (they render the same scene from
/// the same eye) and differ only in their `misc`/`misc2` lanes:
///  - `misc.z` is *each pass's own* render-target height (drives that
///    pass's distance-scaled hit threshold/cone angle -- `codegen.rs`'s
///    `MARCH_CORE`), so the coarse pass gets its coarse height, not the
///    window's.
///  - `misc.x` (aspect) is the *window's* aspect for every pass, not each
///    target's own -- see `mrrm::update_coarse_material`'s old doc (kept
///    below on the coarse branch) for why.
///  - Only the fine pass gets sun/bg colors, the MRRM/shadow-LOD/perfprobe
///    flags; the shadow pass gets the sun direction it marches toward.
///
/// Also detects a marble-count change (a multiplayer join growing the
/// player count -- `physics_sys.rs`'s `RollbackSim::add_player_at`) and
/// force-touches [`FineMarcherMaterial`] exactly then, so its bind group
/// rebuilds against `gpu::MarcherGpuBuffers::marbles`'s buffer object once
/// `write_marcher_buffers` reallocates it for the new (larger) marble list
/// -- see `gpu.rs`'s module doc for why this is the one per-frame-reachable
/// case the persistent-buffer design has to handle instead of asserting
/// against.
///
/// Also computes the marble cubemap's Y-axis rotation angle
/// (`SceneUniforms::misc2.w`) from `MultiplayerSession::sim.current_tick()`
/// -- *not* `time.elapsed_secs()` -- since this app has a real deterministic
/// rollback simulation (`marble_rollback::RollbackSim`) and every peer
/// must render the same marble at the same rotation phase for the same
/// tick; a wall-clock-driven angle would desync visually the moment two
/// clients' clocks disagree even slightly (`MultiplayerSession` is always
/// live, online or offline -- `physics_sys.rs`'s "always-on rollback"
/// doc -- so `current_tick()` is a single source of truth in both cases).
/// The tick is reduced modulo the rotation period *before* converting to
/// `f32` (`ROTATION_PERIOD_TICKS`), not after, so a session running for
/// hours never loses precision converting an ever-growing tick count --
/// the visual result is exactly periodic anyway, so only the phase within
/// one period ever needs representing.
///
/// Uses `web_time::Instant`, not `std::time::Instant`: the latter panics
/// unconditionally on `wasm32-unknown-unknown` ("time not implemented on
/// this platform" -- no OS clock on that target), which took production
/// down the first time this readout shipped (`bcb4d50`/`d5785ed`)
/// -- `web_time` is a drop-in replacement that routes through
/// `Performance.now()` via `web_sys` on wasm, and is a transparent
/// pass-through to real `std::time` everywhere else.
#[allow(clippy::too_many_arguments)]
pub fn update_frame_data(
    time: Res<Time>,
    orbit: Res<CameraOrbit>,
    marble_state: Res<MarbleState>,
    mp: Res<MultiplayerSession>,
    config: Res<crate::config::Config>,
    windows: Query<&Window, With<PrimaryWindow>>,
    fine_render_target: Res<FineRenderTarget>,
    coarse_render_target: Res<crate::mrrm::CoarseRenderTarget>,
    shadow_render_target: Res<crate::shadow_pass::ShadowRenderTarget>,
    perfprobe: Res<crate::perfprobe::PerfProbeState>,
    scene_state: Res<SceneState>,
    frame: ResMut<MarcherFrameData>,
    materials: ResMut<Assets<FineMarcherMaterial>>,
    mut timings: ResMut<crate::fps_overlay::PhaseTimings>,
) {
    let start = web_time::Instant::now();
    update_frame_data_impl(
        time,
        orbit,
        marble_state,
        mp,
        config,
        windows,
        fine_render_target,
        coarse_render_target,
        shadow_render_target,
        perfprobe,
        scene_state,
        frame,
        materials,
    );
    // One combined system now (three per-pass systems merged, `gpu.rs`'s
    // module doc), but the debug overlay's existing "fine=/coarse=/shadow="
    // readout stays meaningful by recording the same elapsed time under all
    // three labels -- CPU-side uniform computation for all three passes
    // genuinely does happen together now, so a per-pass split would just be
    // reporting the same number three ways with extra bookkeeping.
    let elapsed = start.elapsed();
    timings.record("fine", elapsed);
    timings.record("coarse", elapsed);
    timings.record("shadow", elapsed);
}

#[allow(clippy::too_many_arguments)] // SystemParam count, one more for the marble-rotation tick source
fn update_frame_data_impl(
    time: Res<Time>,
    orbit: Res<CameraOrbit>,
    marble_state: Res<MarbleState>,
    mp: Res<MultiplayerSession>,
    config: Res<crate::config::Config>,
    windows: Query<&Window, With<PrimaryWindow>>,
    fine_render_target: Res<FineRenderTarget>,
    coarse_render_target: Res<crate::mrrm::CoarseRenderTarget>,
    shadow_render_target: Res<crate::shadow_pass::ShadowRenderTarget>,
    perfprobe: Res<crate::perfprobe::PerfProbeState>,
    scene_state: Res<SceneState>,
    mut frame: ResMut<MarcherFrameData>,
    mut materials: ResMut<Assets<FineMarcherMaterial>>,
) {
    // Scene-agnostic animated-param upload: `physics_sys::
    // marble_physics_tick_impl` already wrote this tick's evaluated values
    // straight into `mp.sim.scene().params` (offline path directly,
    // connected path via `RollbackSim::advance`/`receive_inputs`, which
    // read/write the same owned `Scene` -- `marble_csg::Scene`'s doc).
    // This system just has to get the result into `frame.params`; skipped
    // entirely for scenes with no animations (the common case) rather
    // than re-copying an unchanged slice every frame for no reason. (The
    // `Demo` scene's older wall-clock-driven `ang1` wobble that used to
    // live here was removed -- it was already dead on the deployed web
    // build (native-`env::var`-only, no query-param path), and reviving
    // it would've meant giving a non-deterministic, desync-prone
    // animation mechanism a live path to the actual deployed multiplayer
    // game, the opposite direction from this session's `Expr`/tick-driven
    // animation work. `menger_oscillating_sphere` is the real, shipped,
    // deterministic way to get animated geometry now.)
    let params_changed = !mp.sim.scene().animations.is_empty();
    if params_changed {
        frame.params.clear();
        frame.params.extend_from_slice(mp.sim.scene().params.slots());
    }

    let t = time.elapsed_secs();
    // `aspect` stays the *window's* aspect for every pass regardless of
    // each pass's own target size (see this fn's own doc, and
    // `frame.coarse`'s comment below on why) -- only the fine pass's own
    // render-target height changed source: `fine_render_target.active_size`
    // (the active tier -- `FineRenderTarget`'s doc), not the window's
    // physical height, now that the fine pass renders into its own
    // fixed-size offscreen target instead of straight to the window.
    let aspect = windows.single().map(|w| w.width() / w.height().max(1.0)).unwrap_or(1.0);
    let resolution_height = fine_render_target.active_size.y as f32;

    // Every scene has a real marble now (`SceneKind::has_marble`): the
    // camera always follows the local player's.
    let target = marble_state.local_marble().pos;
    let (eye, right, up, forward) = orbit.eye_and_basis(target);

    // Marble cubemap Y-axis rotation, 1 revolution per `ROTATION_PERIOD_TICKS`
    // (2 seconds at the 60Hz physics tick) -- deterministic function of
    // `mp.sim.current_tick()`, not wall-clock time, so every peer renders the
    // same marble at the same phase for the same tick (this fn's own doc).
    // Reduced modulo the period *before* the `f32` cast, not after, so this
    // stays numerically exact no matter how long a session runs.
    let tick_in_period = mp.sim.current_tick() % ROTATION_PERIOD_TICKS;
    let marble_rotation =
        tick_in_period as f32 * (std::f32::consts::TAU / ROTATION_PERIOD_TICKS as f32);

    let base = SceneUniforms {
        cam_pos: eye.extend(0.0),
        cam_right: right.extend(0.0),
        cam_up: up.extend(0.0),
        cam_forward: forward.extend(1.5),
        bounding: scene_state.bounding_sphere,
        ..SceneUniforms::default()
    };

    frame.fine = SceneUniforms {
        sun: beware_of_bumps::sun_dir().extend(0.0),
        sun_col: beware_of_bumps::SUN_COL.extend(0.0),
        bg_col: beware_of_bumps::BG.extend(0.0),
        // w: MRRM on/off (`crate::mrrm::mrrm_enabled`) -- a uniform flag
        // rather than skipping the coarse pass/texture binding entirely,
        // so every frame's cameras/passes are identical whether MRRM is
        // on or off and an `MM_MRRM=0` vs `MM_MRRM=1` A/B screenshot
        // comparison at a fixed camera state only ever differs in this
        // one value (see `mrrm::mrrm_enabled`'s doc).
        misc: Vec4::new(aspect, t, resolution_height, if config.mrrm_enabled { 1.0 } else { 0.0 }),
        // x: the shadow pass's own render-target height (`sample_shadow`'s
        // `sz` computation), y: `MM_SHADOW_LOD` on/off (same
        // A/B-comparability reasoning as MRRM's `misc.w` above), z: the
        // `?perfprobe=` diagnostic's fine-step-budget override, w: the
        // marble cubemap's current Y-axis rotation angle (see above).
        misc2: Vec4::new(
            shadow_render_target.size.y as f32,
            if config.shadow_lod_enabled { 1.0 } else { 0.0 },
            perfprobe.fine_max_steps_override,
            marble_rotation,
        ),
        ..base
    };

    // `misc.x` (aspect) is the *window's* current size, not this pass's own
    // (slightly different after integer-dividing by `COARSE_SCALE_DIVISOR`,
    // e.g. a 540-tall window gives a 67-tall coarse target, a ~1%
    // aspect-ratio drift from 540/8=67.5) -- using the window's aspect
    // keeps every pass casting rays in matching directions for the same UV
    // coordinate, which is what makes the coarse pass's hit distance a
    // meaningful guess for the fine pixel at all; the resolution mismatch
    // itself (many fine pixels per coarse texel) is already what the
    // backed-off starting-`t` guess accounts for, so it doesn't need this
    // aspect drift piled on top.
    frame.coarse = SceneUniforms {
        misc: Vec4::new(aspect, t, coarse_render_target.size.y as f32, 0.0),
        ..base
    };

    frame.shadow = SceneUniforms {
        sun: beware_of_bumps::sun_dir().extend(0.0),
        // w: the same MRRM flag the fine pass gets -- the shadow shader's
        // own coarse warm-start read is gated on it too (`marble_csg::codegen`'s
        // `SHADOW_MARCHER`), so `?mrrm=0` disables the coarse-guess data
        // flow in *every* consumer, not just the fine pass.
        misc: Vec4::new(
            aspect,
            t,
            shadow_render_target.size.y as f32,
            if config.mrrm_enabled { 1.0 } else { 0.0 },
        ),
        ..base
    };

    // Live marble list, every frame (multiplayer milestone 0) -- detect a
    // count change (a join growing the player count) before overwriting, so
    // the one material whose bind group references this buffer
    // (`FineMarcherMaterial`; `Coarse`/`ShadowMarcherMaterial` don't bind
    // marbles) gets force-touched exactly on that rare frame (see this fn's
    // doc and `gpu.rs`'s module doc).
    let marbles_len_changed = frame.marbles.len() != marble_state.marbles.len();
    frame.marbles.clear();
    frame.marbles.extend(marble_state.marbles.iter().map(|m| m.pos.extend(m.rad)));
    if marbles_len_changed {
        materials.get_mut(&scene_state.material);
    }
}
