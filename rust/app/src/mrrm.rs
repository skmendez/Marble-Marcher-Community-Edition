//! MRRM (multi-resolution ray marching, named after the original C++ Marble
//! Marcher's `MRRM1`/`MRRM2` compute passes): a coarse, low-resolution
//! ray-march pre-pass whose cached hit distance the fine (full-resolution)
//! pass in `render.rs` starts its own march from, instead of from the
//! camera (`t=0`). This is the single biggest remaining performance lever
//! for this renderer -- it cuts *per-pixel* march-step cost (every fine
//! pixel skips almost all of the empty-space traversal its neighbors in the
//! same coarse texel already paid for).
//!
//! Three passes per frame now (a second-round-perf addition,
//! `shadow_pass.rs`, sits between this one and the fine pass), ordered by
//! `Camera.order` (lowest first):
//!  1. **This module's coarse pass** (`CoarseCamera`/`CoarseQuad`): marches
//!     `de_scene` from `t=0` into a small offscreen `Rgba16Float` render
//!     target (the hit distance in R, `-1.0` sentinel on a miss, G/B/A
//!     unused), using its *own* resolution's cone angle
//!     (`update_coarse_material`'s `misc.z`) -- skips all shading (no
//!     normals/color/shadow).
//!  2. **`shadow_pass.rs`'s half-resolution shadow/AO pass**: also
//!     warm-starts from this pass's output, marches to a real hit, and
//!     caches a shadow-visibility value the fine pass resamples instead of
//!     marching a fresh shadow ray per full-res pixel.
//!  3. **The fine pass** (`render.rs`'s `MarcherCamera`/`MarcherQuad`,
//!     `FineMarcherMaterial`, renders straight to the window): reads this
//!     pass's render target back as a starting-`t` guess (backed off by one
//!     coarse-texel's angular footprint) and does the full march + shading
//!     from there.
//!
//! Sized as a fixed fraction of the window's current physical size
//! (`coarse_target_size`), so it automatically tracks window resizes
//! without this module needing a dedicated resize-detection mechanism of
//! its own.
//!
//! `CoarseCamera`/`CoarseQuad` get their own `RenderLayers`: entities
//! without a `RenderLayers` component default to layer 0 (the fine marcher
//! quad's layer), so without an explicit, distinct layer here, the fine
//! camera would also render the coarse quad on top of its own output (and
//! vice versa).

use bevy::asset::weak_handle;
use bevy::ecs::system::lifetimeless::SRes;
use bevy::ecs::system::SystemParamItem;
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::camera::RenderTarget;
use bevy::render::render_resource::binding_types::{storage_buffer_read_only, uniform_buffer};
use bevy::render::render_resource::{
    AsBindGroup, AsBindGroupError, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
    BindGroupLayoutEntry, BindingResources, Extent3d, PreparedBindGroup, ShaderRef, ShaderStages,
    TextureDimension, TextureFormat, TextureUsages, UnpreparedBindGroup,
};
use bevy::render::renderer::RenderDevice;
use bevy::render::view::RenderLayers;
use bevy::sprite::Material2d;
use bevy::window::PrimaryWindow;

use marble_csg::codegen::generate_coarse_shader;

use crate::gpu::MarcherGpuBuffers;
use crate::physics_sys::MultiplayerSession;
use crate::render::{SceneState, SceneUniforms};

/// All MRRM coarse-pass entities (camera + quad) live on this layer --
/// distinct from the fine marcher's (implicit, unmarked) layer 0 -- see
/// module doc.
const COARSE_LAYER: usize = 2;

/// The coarse render target is `1/COARSE_SCALE_DIVISOR` of the fine
/// target's size in each dimension (so ~1/64 the pixel count) -- coarse
/// enough that the pre-pass march is cheap relative to the fine pass it's
/// speeding up, while still fine-grained enough that neighboring fine
/// pixels within one coarse texel are actually looking at similar depths
/// (thin/close geometry a texel straddles is exactly what the backed-off
/// starting guess and the fine march's own overrelax/backtrack logic are
/// for -- see `march_scene`'s doc in `marble_csg::codegen`).
const COARSE_SCALE_DIVISOR: u32 = 8;

// `?mrrm=0`/`MM_MRRM=0` disables the fine pass's use of this module's
// coarse guess (falls back to always starting its march at `t=0`, i.e.
// exactly the pre-MRRM behavior) -- a per-frame *shader* toggle
// (`SceneUniforms::misc.w`, read from `config::Config::mrrm_enabled` and
// written by `render::update_frame_data`) rather than an entity/system-level
// one that would stop this module's camera/pass from running at all: every
// frame's cameras/passes are identical whether MRRM is on or off, so an
// `?mrrm=0` vs `?mrrm=1` A/B screenshot comparison at a fixed camera state
// only ever differs in this one shader value, not in *what ran* -- which is
// what makes that comparison trustworthy as a regression check.

/// Rounds the window's current physical pixel size down by
/// `COARSE_SCALE_DIVISOR` in each dimension, flooring at 1px (a 0-sized
/// `Image`/texture is invalid) -- pure and unit-tested.
pub fn coarse_target_size(window_size: UVec2) -> UVec2 {
    UVec2::new(
        (window_size.x / COARSE_SCALE_DIVISOR).max(1),
        (window_size.y / COARSE_SCALE_DIVISOR).max(1),
    )
}

/// Fixed weak handle for the generated MRRM coarse-pass shader, following
/// the same pattern as `render::MARCHER_SHADER_HANDLE`.
const COARSE_MARCHER_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("6f1a2c3d-6a10-4f9a-8b7d-2a6b0c9f5e21");

/// The MRRM coarse pass's own material: bindings 0/1 (scene/params) come
/// from `gpu::MarcherGpuBuffers`'s persistent shared buffers (see
/// `gpu.rs`) -- no `coarse_tex` binding (this pass doesn't read its own
/// output), so this is a genuinely different bind-group layout from the
/// fine pass's, hence a separate `Material2d` struct/generated shader
/// module rather than a second entry point sharing one (see
/// `marble_csg::codegen::COARSE_MARCHER`'s doc for why the module itself
/// is also separate, not just the entry point).
///
/// Fieldless: both bindings come from the shared buffers, so this material
/// is never mutated after setup and its bind group is created exactly
/// once.
#[derive(Asset, TypePath, Clone)]
pub struct CoarseMarcherMaterial {}

impl AsBindGroup for CoarseMarcherMaterial {
    type Data = ();
    type Param = SRes<MarcherGpuBuffers>;

    fn label() -> Option<&'static str> {
        Some("coarse_marcher_material")
    }

    fn as_bind_group(
        &self,
        layout: &BindGroupLayout,
        render_device: &RenderDevice,
        buffers: &mut SystemParamItem<'_, '_, Self::Param>,
    ) -> Result<PreparedBindGroup<Self::Data>, AsBindGroupError> {
        // `RetryNextUpdate` until `gpu::write_marcher_buffers`'s first run
        // has allocated the shared buffers (render.rs's fine impl doc).
        let scene = buffers.coarse_scene.binding().ok_or(AsBindGroupError::RetryNextUpdate)?;
        let params = buffers.params.binding().ok_or(AsBindGroupError::RetryNextUpdate)?;
        let bind_group = render_device.create_bind_group(
            Self::label(),
            layout,
            &BindGroupEntries::with_indices(((0, scene), (1, params))),
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
            ),
        )
        .to_vec()
    }
}

impl Material2d for CoarseMarcherMaterial {
    fn fragment_shader() -> ShaderRef {
        COARSE_MARCHER_SHADER_HANDLE.into()
    }

    // Deliberately *no* `specialize()` override here: `Mesh2dPipeline`'s
    // base `specialize` (bevy_sprite 0.16.1, `mesh2d/mesh.rs`) always draws
    // into `ViewTarget::TEXTURE_FORMAT_HDR` (`Rgba16Float`) when the camera
    // has `hdr: true` (`CoarseCamera` below), or `TextureFormat::bevy_default()`
    // otherwise -- *never* whatever format the camera's actual target
    // `Image` is; that's handled entirely by a later
    // (`bevy_core_pipeline::upscaling`) blit pass that reads the
    // intermediate and writes it into the real target at its real format
    // (`view_target.out_texture_format()`). An earlier version of this
    // material tried to override the pipeline's color target format to
    // match the destination `Image` directly (`R32Float`) -- that's wrong:
    // it draws into the *intermediate*, not the destination, so it just
    // made this pipeline's declared format disagree with the render pass it
    // actually runs in, and wgpu rejected it (\"RenderPipeline ... uses
    // attachments with formats [Some(R32Float)]\" vs. the pass's
    // `[Some(Rgba8UnormSrgb)]`). Setting `hdr: true` on the camera and
    // matching the destination `Image`'s format to `Rgba16Float`
    // (`make_coarse_render_target_image`) is the actually-correct way to
    // get a wide-range, non-8-bit-unorm value through this pipeline intact.
}

/// Marker for the MRRM coarse pass's own camera -- lowest `Camera.order` of
/// the three passes (renders first: both the shadow pass and the fine pass
/// warm-start their own march from this pass's output, so it must finish
/// before either reads it back; see module doc). `pub(crate)`: it appears in
/// `resize_coarse_render_target`'s public query type, which `main.rs` (a
/// different module) needs to be able to name when registering that system
/// -- same reasoning as `CoarseQuad`'s doc.
#[derive(Component)]
pub(crate) struct CoarseCamera;

/// Marker for the MRRM coarse pass's fullscreen quad. `pub(crate)`: it
/// appears in `sync_coarse_quad_scale`/`update_coarse_material`'s public
/// query types, which `main.rs` (a different module) needs to be able to
/// name when registering those systems.
#[derive(Component)]
pub(crate) struct CoarseQuad;

/// The coarse pass's offscreen render target.
#[derive(Resource)]
pub struct CoarseRenderTarget {
    pub image: Handle<Image>,
    pub size: UVec2,
}

/// Builds the coarse pass's offscreen hit-distance `Image`. `Rgba16Float`,
/// not a single-channel format: this is what the camera actually has to
/// render into given how Bevy's 2D pipeline works (see
/// `CoarseMarcherMaterial`'s doc) -- the camera's intermediate "main
/// texture" is `Rgba16Float` whenever `hdr: true` (`CoarseCamera` below),
/// and the final blit into this destination `Image` is a plain
/// `textureSample` copy (`bevy_core_pipeline`'s `blit.wgsl`) with no
/// tonemapping/gamma step, so matching the destination format to the
/// intermediate's is what lets an arbitrary (negative-capable, > 1.0-capable)
/// hit-distance value survive that copy exactly instead of being clamped/
/// gamma-encoded the way an 8-bit UNORM destination would. Verified against
/// wgpu's `TextureFormat::guaranteed_format_features` (wgpu-types 24.0.0,
/// matching this workspace's locked wgpu version): `Rgba16Float` needs zero
/// optional device features and its guaranteed flags include
/// `RENDER_ATTACHMENT` + `TEXTURE_BINDING` unconditionally -- it's also
/// exactly the format Bevy's own HDR camera path already relies on, so
/// there's no real question of it being supported.
///
/// Initial fill is all-zero bytes (decodes to `(0.0, 0.0, 0.0, 0.0)`, not
/// the shader's own `-1.0` miss sentinel) -- deliberately: the fine pass's
/// `coarse_t > 0.0` check treats exactly `0.0` as a miss too, so an all-zero
/// fill is just as safe a placeholder and avoids needing to hand-encode an
/// `f16` bit pattern here. Only matters for an instant at startup anyway --
/// every `Startup` system (including this one) finishes before the first
/// frame is rendered, and the coarse camera fully redraws this texture every
/// frame after that.
fn make_coarse_render_target_image(size: UVec2) -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0u8; 8],
        TextureFormat::Rgba16Float,
        bevy::asset::RenderAssetUsages::default(),
    );
    // `Image::new_fill` only sets a plain sampled-texture usage by default,
    // and this image is also a camera render target, so it additionally
    // needs `RENDER_ATTACHMENT`.
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::RENDER_ATTACHMENT;
    image
}

/// `Startup` system -- chained to run after `render::setup` (needs
/// `SceneState`'s scene tree/params buffer to build the coarse shader and
/// material, and corrects `FineMarcherMaterial`'s placeholder `coarse`
/// handle to point at the real image once it exists).
///
/// Bevy systems commonly need one `SystemParam` per distinct resource/query
/// they touch, and this one legitimately touches eight (two material asset
/// stores, the shader/image/mesh asset stores, `Commands`, and two
/// resources) -- splitting it into multiple chained systems just to dodge
/// clippy's argument-count lint would need an intermediate resource purely
/// to shuttle values between them, which is more indirection, not less.
#[allow(clippy::too_many_arguments)]
pub fn setup_mrrm_pipeline(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut coarse_materials: ResMut<Assets<CoarseMarcherMaterial>>,
    mut fine_materials: ResMut<Assets<crate::render::FineMarcherMaterial>>,
    mut shaders: ResMut<Assets<Shader>>,
    mut images: ResMut<Assets<Image>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    scene_state: Res<SceneState>,
    mp: Res<MultiplayerSession>,
) {
    let wgsl = generate_coarse_shader(&mp.sim.scene().object);
    shaders.insert(
        COARSE_MARCHER_SHADER_HANDLE.id(),
        Shader::from_wgsl(wgsl, "generated://marcher_coarse.wgsl"),
    );

    let native_size = windows
        .single()
        .map(|w| UVec2::new(w.physical_width().max(1), w.physical_height().max(1)))
        .unwrap_or(UVec2::new(1280, 720));
    let size = coarse_target_size(native_size);
    let image_handle = images.add(make_coarse_render_target_image(size));

    if let Some(fine_material) = fine_materials.get_mut(&scene_state.material) {
        fine_material.coarse = image_handle.clone();
    }

    let coarse_material = coarse_materials.add(CoarseMarcherMaterial {});

    commands.spawn((
        Camera2d,
        Camera {
            target: RenderTarget::from(image_handle.clone()),
            // Lowest order of the three passes -- must finish before the
            // shadow pass (`shadow_pass.rs`, order -1) and fine pass
            // (`render.rs`, order 0 by default) both read this texture back
            // (module doc).
            order: -2,
            // Required so this camera's intermediate "main texture" is
            // `Rgba16Float` (matching this pass's destination `Image`),
            // rather than the default 8-bit-UNORM `bevy_default()` --
            // see `CoarseMarcherMaterial`'s doc.
            hdr: true,
            ..default()
        },
        // MSAA would resolve-blend a hit distance with the `-1.0` miss
        // sentinel at any edge between a hit and a miss, producing a
        // meaningless intermediate "distance" instead of either real value
        // -- forced off regardless of format support, since blending
        // distances is wrong even where it'd be technically possible.
        Msaa::Off,
        RenderLayers::layer(COARSE_LAYER),
        CoarseCamera,
    ));

    commands.spawn((
        Mesh2d(meshes.add(Rectangle::new(1.0, 1.0).mesh())),
        MeshMaterial2d(coarse_material),
        Transform::default(),
        RenderLayers::layer(COARSE_LAYER),
        CoarseQuad,
    ));

    commands.insert_resource(CoarseRenderTarget {
        image: image_handle,
        size,
    });
}

/// `Update` system: keeps the coarse render target sized at
/// `coarse_target_size` of the window's current physical size -- only
/// touches the GPU resource when that rounded size actually changed (a
/// real window resize, not every frame).
///
/// Builds a **new** `Image` and redirects the coarse camera's `Camera.target`
/// and the fine pass's `FineMarcherMaterial.coarse` binding to it, rather
/// than resizing the *same* `Image` asset in place
/// (`images.get_mut(..).resize(..)`): doing that while a camera is actively
/// rendering into it is a known upstream Bevy behavior that permanently
/// freezes that camera's output (this hit the fine render target hard
/// enough, back when one existed for adaptive resolution, to be worth
/// avoiding here too even though this system fires far less often now that
/// it only reacts to real window resizes).
pub fn resize_coarse_render_target(
    windows: Query<&Window, With<PrimaryWindow>>,
    mut coarse_render_target: ResMut<CoarseRenderTarget>,
    mut images: ResMut<Assets<Image>>,
    mut coarse_cameras: Query<&mut Camera, With<CoarseCamera>>,
    fine_quads: Query<&MeshMaterial2d<crate::render::FineMarcherMaterial>, With<crate::render::MarcherQuad>>,
    mut fine_materials: ResMut<Assets<crate::render::FineMarcherMaterial>>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let native_size = UVec2::new(window.physical_width().max(1), window.physical_height().max(1));
    let desired = coarse_target_size(native_size);
    if desired == coarse_render_target.size {
        return;
    }

    let new_handle = images.add(make_coarse_render_target_image(desired));

    for mut camera in &mut coarse_cameras {
        camera.target = RenderTarget::from(new_handle.clone());
    }
    for mesh_material in &fine_quads {
        if let Some(mat) = fine_materials.get_mut(&mesh_material.0) {
            mat.coarse = new_handle.clone();
        }
    }

    images.remove(&coarse_render_target.image);
    coarse_render_target.image = new_handle;
    coarse_render_target.size = desired;
}

/// Keeps the coarse quad's world size equal to its own render target's
/// pixel size -- mirrors `render::sync_quad_scale` for the fine
/// quad/target, including its "deref-mut only on a real change" guard (the
/// target size only changes on a window resize).
pub fn sync_coarse_quad_scale(
    render_target: Res<CoarseRenderTarget>,
    mut quads: Query<&mut Transform, With<CoarseQuad>>,
) {
    let (w, h) = (render_target.size.x as f32, render_target.size.y as f32);
    for mut transform in &mut quads {
        if transform.scale.x != w || transform.scale.y != h {
            transform.scale.x = w;
            transform.scale.y = h;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coarse_target_size_divides_down() {
        assert_eq!(coarse_target_size(UVec2::new(1280, 720)), UVec2::new(160, 90));
    }

    #[test]
    fn coarse_target_size_floors_at_one_pixel() {
        assert_eq!(coarse_target_size(UVec2::new(4, 4)), UVec2::new(1, 1));
    }

    #[test]
    fn coarse_target_size_truncates_not_rounds() {
        // 67 * 8 = 536, not 540 -- integer division truncates, doesn't
        // round to nearest, matching plain `/` semantics (no surprise
        // off-by-one on odd sizes).
        assert_eq!(coarse_target_size(UVec2::new(960, 540)), UVec2::new(120, 67));
    }
}
