//! Half-resolution shadow/AO pre-pass (second-round-perf addition, mirrors
//! `mrrm.rs`'s architecture): a soft-shadow march is far more expensive per
//! pixel than the primary march it sits next to (`SHADOW_STEPS = 24` fixed
//! iterations, `marble_csg::codegen`'s `shadow()`, run at *every* full-res
//! pixel today), so this decouples its resolution from color resolution --
//! ported from MMCE's own `Illumination_step` pass
//! (`game_folder/shaders/shader_documentation.md`: "half res shadows and
//! AO"), which does the same thing.
//!
//! Sits between MRRM's coarse pass and the fine pass in `Camera.order`
//! (`mrrm.rs`'s module doc): warm-starts its own march from MRRM's existing
//! coarse hit-distance buffer (same technique the fine pass uses), then the
//! fine pass resamples *this* pass's cached shadow visibility
//! (`marble_csg::codegen`'s `sample_shadow`, a depth-aware 4-tap blend --
//! ported from MMCE's `bilinear_surface`) instead of marching a fresh shadow
//! ray per full-res pixel.
//!
//! `ShadowCamera`/`ShadowQuad` get their own `RenderLayers` (distinct from
//! both the fine marcher's implicit layer 0 and MRRM's `COARSE_LAYER`), same
//! reasoning as `mrrm.rs`'s module doc.

use bevy::asset::weak_handle;
use bevy::ecs::system::lifetimeless::SRes;
use bevy::ecs::system::SystemParamItem;
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::camera::RenderTarget;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only, texture_2d, uniform_buffer,
};
use bevy::render::render_resource::{
    AsBindGroup, AsBindGroupError, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
    BindGroupLayoutEntry, BindingResources, Extent3d, PreparedBindGroup, ShaderRef, ShaderStages,
    TextureDimension, TextureFormat, TextureSampleType, TextureUsages, UnpreparedBindGroup,
};
use bevy::render::renderer::RenderDevice;
use bevy::render::texture::GpuImage;
use bevy::render::view::RenderLayers;
use bevy::sprite::Material2d;
use bevy::window::PrimaryWindow;

use marble_csg::codegen::generate_shadow_shader;

use crate::gpu::MarcherGpuBuffers;
use crate::mrrm::CoarseRenderTarget;
use crate::render::{SceneState, SceneUniforms};

/// All shadow-pass entities (camera + quad) live on this layer -- distinct
/// from the fine marcher's (implicit) layer 0 and MRRM's `COARSE_LAYER`.
const SHADOW_LAYER: usize = 3;

/// The shadow render target is `1/SHADOW_SCALE_DIVISOR` of the fine target's
/// size in each dimension -- a much gentler downscale than MRRM's coarse
/// pass (`COARSE_SCALE_DIVISOR = 8`): shadow/AO detail matters more visually
/// than raw hit-distance does, and the depth-aware resample
/// (`sample_shadow`) is specifically what makes even this gentle a downscale
/// safe at silhouette edges. Mirrors MMCE's own `Illumination_step`, whose
/// shader-config doc calls half resolution the standard case.
const SHADOW_SCALE_DIVISOR: u32 = 2;

// `?shadowlod=0`/`MM_SHADOW_LOD=0` disables the fine pass's use of this
// module's cached shadow value (falls back to marching a fresh shadow ray
// per full-res pixel, i.e. exactly the pre-this-change behavior) -- a
// per-frame *shader* toggle (`SceneUniforms::misc2.y`, read from
// `config::Config::shadow_lod_enabled` and written by
// `render::update_frame_data`), not an entity/system-level one, for the
// same A/B-comparability reason as `mrrm.rs`'s doc: every frame's
// cameras/passes are identical whether this is on or off, so an
// `?shadowlod=0` vs `=1` screenshot comparison at a fixed camera state
// only ever differs in this one value.

/// Rounds the window's current physical pixel size down by
/// `SHADOW_SCALE_DIVISOR` in each dimension, flooring at 1px -- pure and
/// unit-tested, mirrors `mrrm::coarse_target_size`.
pub fn shadow_target_size(window_size: UVec2) -> UVec2 {
    UVec2::new(
        (window_size.x / SHADOW_SCALE_DIVISOR).max(1),
        (window_size.y / SHADOW_SCALE_DIVISOR).max(1),
    )
}

/// Fixed weak handle for the generated shadow-pass shader, following the
/// same pattern as `render::MARCHER_SHADER_HANDLE`/`mrrm`'s coarse handle.
const SHADOW_MARCHER_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("9d3e5a71-2f84-4b6c-9a1d-7c5e8f0b2d4a");

/// The shadow pass's own material: bindings 0/1 (scene/params) come from
/// `gpu::MarcherGpuBuffers`'s persistent shared buffers (see `gpu.rs`); the
/// only field left is MRRM's coarse hit-distance texture (binding 2, this
/// pass's own warm-start source -- same bind-group shape `FineMarcherMaterial`
/// had before it also grew a `shadow` binding, see `generate_shadow_shader`'s
/// doc for why this still needs its own `Material2d`/shader module rather
/// than reusing either the coarse or fine one). Mutating `coarse` (MRRM's
/// render target resizing) is the only thing that re-prepares this
/// material now.
#[derive(Asset, TypePath, Clone)]
pub struct ShadowMarcherMaterial {
    pub coarse: Handle<Image>,
}

impl AsBindGroup for ShadowMarcherMaterial {
    type Data = ();
    type Param = (SRes<MarcherGpuBuffers>, SRes<RenderAssets<GpuImage>>);

    fn label() -> Option<&'static str> {
        Some("shadow_marcher_material")
    }

    fn as_bind_group(
        &self,
        layout: &BindGroupLayout,
        render_device: &RenderDevice,
        (buffers, images): &mut SystemParamItem<'_, '_, Self::Param>,
    ) -> Result<PreparedBindGroup<Self::Data>, AsBindGroupError> {
        // `RetryNextUpdate` until `gpu::write_marcher_buffers`'s first run
        // has allocated the shared buffers and the coarse render target
        // exists as a GPU image (render.rs's fine impl doc).
        let scene = buffers.shadow_scene.binding().ok_or(AsBindGroupError::RetryNextUpdate)?;
        let params = buffers.params.binding().ok_or(AsBindGroupError::RetryNextUpdate)?;
        let coarse = images.get(&self.coarse).ok_or(AsBindGroupError::RetryNextUpdate)?;
        let bind_group = render_device.create_bind_group(
            Self::label(),
            layout,
            &BindGroupEntries::with_indices(((0, scene), (1, params), (2, &coarse.texture_view))),
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
                (2, texture_2d(TextureSampleType::Float { filterable: true })),
            ),
        )
        .to_vec()
    }
}

impl Material2d for ShadowMarcherMaterial {
    fn fragment_shader() -> ShaderRef {
        SHADOW_MARCHER_SHADER_HANDLE.into()
    }

    // No `specialize()` override -- same reasoning as `CoarseMarcherMaterial`'s
    // doc (`mrrm.rs`): `hdr: true` on the camera below is what makes this
    // pipeline draw into an `Rgba16Float` intermediate, matching this pass's
    // destination `Image` format, without needing to fight the base
    // `Mesh2dPipeline`'s own color-target-format choice.
}

/// Marker for the shadow pass's own camera -- middle `Camera.order` of the
/// three passes (after MRRM's coarse pass, whose output it warm-starts from;
/// before the fine pass, which reads this pass's output back). `pub(crate)`:
/// appears in `resize_shadow_render_target`'s public query type, which
/// `main.rs` needs to name when registering that system.
#[derive(Component)]
pub(crate) struct ShadowCamera;

/// Marker for the shadow pass's fullscreen quad. `pub(crate)`: appears in
/// `sync_shadow_quad_scale`/`update_shadow_material`'s public query types,
/// which `main.rs` needs to name when registering those systems.
#[derive(Component)]
pub(crate) struct ShadowQuad;

/// The shadow pass's offscreen render target.
#[derive(Resource)]
pub struct ShadowRenderTarget {
    pub image: Handle<Image>,
    pub size: UVec2,
}

/// Builds the shadow pass's offscreen `Image` -- `Rgba16Float`, same
/// reasoning as `mrrm::make_coarse_render_target_image`'s doc (this pass's
/// output needs an arbitrary, not-clamped-to-[0,1], not-gamma-encoded `A`
/// channel for the traveled-distance value `sample_shadow` reads back).
/// Initial fill is all-zero bytes, decoding to `(0.0, 0.0, 0.0, 0.0)`:
/// harmless (every `Startup` system finishes before the first frame renders,
/// and this pass fully redraws its target every frame after that), and even
/// a stale zero read (`shadow = 0.0`, `td = 0.0`) would just read as
/// "fully shadowed at the camera" rather than corrupting anything.
fn make_shadow_render_target_image(size: UVec2) -> Image {
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
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::RENDER_ATTACHMENT;
    image
}

/// `Startup` system -- chained to run after `render::setup` (needs
/// `SceneState`) and `mrrm::setup_mrrm_pipeline` (needs `CoarseRenderTarget`'s
/// image handle to bind as this pass's own warm-start source, and corrects
/// `FineMarcherMaterial`'s placeholder `shadow` handle to point at the real
/// image once it exists).
#[allow(clippy::too_many_arguments)]
pub fn setup_shadow_pipeline(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut shadow_materials: ResMut<Assets<ShadowMarcherMaterial>>,
    mut fine_materials: ResMut<Assets<crate::render::FineMarcherMaterial>>,
    mut shaders: ResMut<Assets<Shader>>,
    mut images: ResMut<Assets<Image>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    scene_state: Res<SceneState>,
    coarse_render_target: Res<CoarseRenderTarget>,
) {
    let wgsl = generate_shadow_shader(&scene_state.object);
    shaders.insert(
        SHADOW_MARCHER_SHADER_HANDLE.id(),
        Shader::from_wgsl(wgsl, "generated://marcher_shadow.wgsl"),
    );

    let native_size = windows
        .single()
        .map(|w| UVec2::new(w.physical_width().max(1), w.physical_height().max(1)))
        .unwrap_or(UVec2::new(1280, 720));
    let size = shadow_target_size(native_size);
    let image_handle = images.add(make_shadow_render_target_image(size));

    if let Some(fine_material) = fine_materials.get_mut(&scene_state.material) {
        fine_material.shadow = image_handle.clone();
    }

    let shadow_material =
        shadow_materials.add(ShadowMarcherMaterial { coarse: coarse_render_target.image.clone() });

    commands.spawn((
        Camera2d,
        Camera {
            target: RenderTarget::from(image_handle.clone()),
            // Middle order of the three passes (module doc): after MRRM's
            // coarse pass (-2), before the fine pass (0 by default).
            order: -1,
            hdr: true,
            ..default()
        },
        // Same reasoning as `mrrm.rs`'s `CoarseCamera`: blending a shadow
        // value with the `MAX_DIST` miss sentinel at a hit/miss edge would
        // produce a meaningless intermediate value.
        Msaa::Off,
        RenderLayers::layer(SHADOW_LAYER),
        ShadowCamera,
    ));

    commands.spawn((
        Mesh2d(meshes.add(Rectangle::new(1.0, 1.0).mesh())),
        MeshMaterial2d(shadow_material),
        Transform::default(),
        RenderLayers::layer(SHADOW_LAYER),
        ShadowQuad,
    ));

    commands.insert_resource(ShadowRenderTarget {
        image: image_handle,
        size,
    });
}

/// `Update` system: keeps the shadow render target sized at
/// `shadow_target_size` of the window's current physical size, and keeps
/// its material's `coarse` binding pointed at MRRM's coarse render target
/// (which itself gets rebuilt as a new `Image` on resize, `mrrm.rs`) --
/// only touches GPU resources when something actually changed. Builds a
/// **new** `Image` and redirects handles rather than resizing in place, same
/// known-Bevy-freeze-bug avoidance as `mrrm::resize_coarse_render_target`'s
/// doc.
#[allow(clippy::too_many_arguments)]
pub fn resize_shadow_render_target(
    windows: Query<&Window, With<PrimaryWindow>>,
    mut shadow_render_target: ResMut<ShadowRenderTarget>,
    coarse_render_target: Res<CoarseRenderTarget>,
    mut images: ResMut<Assets<Image>>,
    mut shadow_cameras: Query<&mut Camera, With<ShadowCamera>>,
    shadow_quads: Query<&MeshMaterial2d<ShadowMarcherMaterial>, With<ShadowQuad>>,
    fine_quads: Query<&MeshMaterial2d<crate::render::FineMarcherMaterial>, With<crate::render::MarcherQuad>>,
    mut shadow_materials: ResMut<Assets<ShadowMarcherMaterial>>,
    mut fine_materials: ResMut<Assets<crate::render::FineMarcherMaterial>>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let native_size = UVec2::new(window.physical_width().max(1), window.physical_height().max(1));
    let desired = shadow_target_size(native_size);

    // Keep the `coarse` binding current (MRRM's own render target is
    // rebuilt independently on its own resize schedule) -- but check with
    // an immutable `get` first: an unconditional `get_mut` marks the
    // material asset modified every frame, which re-runs its whole
    // bind-group preparation (`gpu.rs`'s module doc) for what is a no-op
    // write on every frame that isn't an actual resize.
    for shadow_quad in &shadow_quads {
        let stale = shadow_materials
            .get(&shadow_quad.0)
            .is_some_and(|mat| mat.coarse != coarse_render_target.image);
        if stale {
            if let Some(mat) = shadow_materials.get_mut(&shadow_quad.0) {
                mat.coarse = coarse_render_target.image.clone();
            }
        }
    }

    if desired == shadow_render_target.size {
        return;
    }

    let new_handle = images.add(make_shadow_render_target_image(desired));

    for mut camera in &mut shadow_cameras {
        camera.target = RenderTarget::from(new_handle.clone());
    }
    for mesh_material in &fine_quads {
        if let Some(mat) = fine_materials.get_mut(&mesh_material.0) {
            mat.shadow = new_handle.clone();
        }
    }

    images.remove(&shadow_render_target.image);
    shadow_render_target.image = new_handle;
    shadow_render_target.size = desired;
}

/// Keeps the shadow quad's world size equal to its own render target's pixel
/// size -- mirrors `mrrm::sync_coarse_quad_scale`, including its
/// "deref-mut only on a real change" guard.
pub fn sync_shadow_quad_scale(
    render_target: Res<ShadowRenderTarget>,
    mut quads: Query<&mut Transform, With<ShadowQuad>>,
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
    fn shadow_target_size_divides_down() {
        assert_eq!(shadow_target_size(UVec2::new(1280, 720)), UVec2::new(640, 360));
    }

    #[test]
    fn shadow_target_size_floors_at_one_pixel() {
        assert_eq!(shadow_target_size(UVec2::new(1, 1)), UVec2::new(1, 1));
    }
}
