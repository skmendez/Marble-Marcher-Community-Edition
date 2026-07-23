//! Estimates the *cumulative* ray-march step count across the whole fine
//! pass's frame, as a companion to `?stepheat=1` (`codegen.rs`'s per-pixel
//! heatmap) and `?gpuprofile=1` (`gpu_profile.rs`'s real per-pass GPU
//! timing) -- lets the two be correlated (does total step count actually
//! predict fine-pass GPU time?) instead of only ever eyeballing a heatmap.
//!
//! ## Why not an atomic counter
//!
//! The direct approach -- every one of the fine pass's ~1M+ fragment
//! invocations does an `atomicAdd` into one shared counter -- is expressible
//! in WGSL/WebGPU (atomics on storage buffers are supported from the
//! fragment stage), but that many invocations all serializing on one memory
//! location risks real GPU contention that would distort the very
//! measurement being taken. Rejected for that reason, not attempted here.
//!
//! ## What this actually does instead
//!
//! A fifth, tiny auxiliary pass, architecturally identical in shape to the
//! coarse/shadow/fine/present passes (`mrrm.rs`/`shadow_pass.rs`/`render.rs`):
//! its own `Camera2d` + `RenderLayers` + offscreen `Image` + `Material2d`.
//! Its shader (`marble_csg::codegen::generate_stepdata_shader`) runs the
//! *exact same* terrain march the real fine pass does -- same MRRM
//! warm-start (reads the *same* `coarse_tex`, the *same* `misc.w` flag),
//! same `fine_max_steps` budget -- so its output is a statistically
//! faithful, just spatially-downsampled, sample of the real fine pass's
//! per-pixel step-count distribution. It differs from the real fine pass
//! in exactly two ways: its render target is small and fixed
//! ([`STEPDATA_TARGET_SIZE`]) rather than full window resolution, and its
//! fragment shader's final output is the *raw* `f32(iters)` value (an
//! `Rgba16Float` render target's R channel) rather than a shaded color or
//! the `?stepheat=1` heatmap's stylized, non-invertible color ramp.
//!
//! Crucially, its bind group reads the *same* `MarcherGpuBuffers::fine_scene`
//! uniform buffer the real fine pass's own material reads (`render.rs`'s
//! `FineMarcherMaterial`) -- not a second uniform buffer this module has to
//! keep in sync itself. Every value this pass's warm-start needs (camera
//! vectors, `misc.w`'s MRRM flag, `misc.z`'s pixel-angle basis, the bounding
//! sphere) is already written there once a frame by `render.rs`'s
//! `update_frame_data_impl`; this module adds zero new per-frame CPU-side
//! uniform writes.
//!
//! Readback uses Bevy's own public `bevy_render::gpu_readback` machinery
//! directly (`Readback`/`ReadbackComplete`), not a hand-rolled
//! `map_async`/channel pipeline the way `gpu_profile.rs` had to build:
//! `gpu_profile.rs`'s hand-rolled plumbing exists only because Bevy has no
//! way to hook a *mid-render-pass* timestamp write at all (a genuine, hard
//! Bevy limitation). Reading back a texture's contents *after* it's
//! finished rendering is a completely ordinary, already-supported
//! operation with no such limitation -- `Readback::texture(handle)`
//! attached to an entity re-triggers every frame on its own, no extra
//! per-frame system needed on this module's part beyond an `Observer` to
//! receive [`ReadbackComplete`].
//!
//! Averaging happens on the CPU side once the (tiny) readback lands: sum
//! every texel's R value, divide by texel count, multiply by the real fine
//! pass's *actual current* pixel count (`render::FineRenderTarget::active_size`
//! -- not the raw window size, since `?res_oscillate=1`/adaptive resolution
//! can shrink the fine pass below the window's native size) to get an
//! estimated cumulative step count for the whole real frame.
//!
//! Gated end-to-end on `Config.gpu_profile_enabled` -- the same flag
//! `?gpuprofile=1` already uses for GPU timestamps, not a second flag for
//! what's conceptually the same "understand perf" feature bundle. When it's
//! off, [`setup_step_data_pipeline`] never spawns this pass's camera/quad at
//! all, so there's no render-graph node, no readback, no per-frame cost --
//! a true no-op, the same standard every other profiling feature in this
//! app already holds itself to.

use bevy::asset::weak_handle;
use bevy::ecs::system::lifetimeless::SRes;
use bevy::ecs::system::SystemParamItem;
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::camera::RenderTarget;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{storage_buffer_read_only, texture_2d, uniform_buffer};
use bevy::render::render_resource::{
    AsBindGroup, AsBindGroupError, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
    BindGroupLayoutEntry, BindingResources, Extent3d, PreparedBindGroup, ShaderRef, ShaderStages,
    TextureDimension, TextureFormat, TextureSampleType, TextureUsages, UnpreparedBindGroup,
};
use bevy::render::renderer::RenderDevice;
use bevy::render::texture::GpuImage;
use bevy::render::view::RenderLayers;
use bevy::sprite::Material2d;

use marble_csg::codegen::generate_stepdata_shader;

use crate::config::Config;
use crate::gpu::MarcherGpuBuffers;
use crate::mrrm::CoarseRenderTarget;
use crate::physics_sys::MultiplayerSession;
use crate::render::{FineRenderTarget, SceneUniforms};

/// This pass's own `RenderLayers` -- distinct from the fine marcher's
/// (implicit layer 0), MRRM coarse's (`mrrm::COARSE_LAYER = 2`), shadow's
/// (`shadow_pass::SHADOW_LAYER = 3`), and present's (`render::PRESENT_LAYER
/// = 1`).
const STEPDATA_LAYER: usize = 4;

/// Fixed, small render-target size for the step-count-data pass. Width `64`
/// with `Rgba16Float` (8 bytes/pixel) gives exactly 512 bytes/row -- already
/// a multiple of `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT` (256), so the actual
/// GPU→CPU readback needs no row-padding stripped out in practice (this
/// module still computes the real aligned stride via
/// `RenderDevice::align_copy_bytes_per_row` rather than assuming that,
/// so it stays correct even if this constant is ever changed to a width
/// that isn't already aligned). `36` rows is a modest, cheap-to-read-back
/// sample size that's still large enough to be statistically representative
/// given how smoothly step count tends to vary across most of a frame (see
/// this session's `?stepheat=1` heatmap screenshots) -- not a derived
/// value, just a reasonable, cheap default.
const STEPDATA_TARGET_SIZE: UVec2 = UVec2::new(64, 36);

/// Fixed weak handle for the generated step-data shader, following the same
/// pattern as `render::MARCHER_SHADER_HANDLE`/`mrrm::COARSE_MARCHER_SHADER_HANDLE`.
const STEPDATA_SHADER_HANDLE: Handle<Shader> = weak_handle!("2c9f4e6a-8b1d-4f7c-9a3e-5d8f1c6b2e94");

/// The step-count-data pass's own material: bindings 0/1 (scene/params)
/// come from `gpu::MarcherGpuBuffers`'s persistent shared buffers -- the
/// *same* `fine_scene` uniform the real fine pass's own material reads
/// (module doc's "Crucially" paragraph) -- binding 2 is the MRRM coarse
/// pass's cached hit-distance texture (needed to warm-start identically to
/// the fine pass). No shadow-texture/marble-buffer/marble-cubemap bindings:
/// none of those affect step count, so this material's bind-group layout is
/// deliberately smaller than `FineMarcherMaterial`'s.
#[derive(Asset, TypePath, Clone)]
pub struct StepDataMaterial {
    pub coarse: Handle<Image>,
}

impl AsBindGroup for StepDataMaterial {
    type Data = ();
    type Param = (SRes<MarcherGpuBuffers>, SRes<RenderAssets<GpuImage>>);

    fn label() -> Option<&'static str> {
        Some("step_data_material")
    }

    fn as_bind_group(
        &self,
        layout: &BindGroupLayout,
        render_device: &RenderDevice,
        (buffers, images): &mut SystemParamItem<'_, '_, Self::Param>,
    ) -> Result<PreparedBindGroup<Self::Data>, AsBindGroupError> {
        // `RetryNextUpdate` until `gpu::write_marcher_buffers`'s first run
        // has allocated the shared buffers and the coarse render target
        // exists as a GPU image -- same reasoning as `FineMarcherMaterial`'s
        // own `as_bind_group`.
        let scene = buffers.fine_scene.binding().ok_or(AsBindGroupError::RetryNextUpdate)?;
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

impl Material2d for StepDataMaterial {
    fn fragment_shader() -> ShaderRef {
        STEPDATA_SHADER_HANDLE.into()
    }
}

/// Marker for the step-count-data pass's own camera.
#[derive(Component)]
struct StepDataCamera;

/// Marker for the step-count-data pass's fullscreen quad. `pub(crate)`: also
/// queried from `mrrm::resize_coarse_render_target`, which must repoint this
/// material's `coarse` handle (same as it already does for
/// `FineMarcherMaterial`) whenever the coarse render target is rebuilt on a
/// real window resize -- otherwise it goes stale, pointing at a removed
/// `Image` asset.
#[derive(Component)]
pub(crate) struct StepDataQuad;

/// Main-world resource: the last cumulative-step-count estimate, updated
/// whenever a readback lands (`on_step_data_readback`). `None` until the
/// first successful readback arrives, or forever if `?gpuprofile=1` is off
/// (this whole pass never spawns in that case).
#[derive(Resource, Clone, Copy, Default)]
pub struct EstimatedSteps {
    /// Average `iters` value sampled across this pass's small render
    /// target -- a direct statistic, not scaled by anything.
    pub avg_iters_per_px: Option<f32>,
    /// `avg_iters_per_px` multiplied by the real fine pass's current
    /// active pixel count (`FineRenderTarget::active_size`) -- the actual
    /// cumulative-step-count estimate for the whole frame.
    pub estimated_total_steps: Option<f64>,
}

/// Builds the step-count-data pass's offscreen `Image`. `Rgba16Float`, same
/// reasoning as `mrrm.rs`'s coarse target (`make_coarse_render_target_image`'s
/// doc): Bevy's Material2d pipeline always renders into the HDR intermediate
/// format when the camera has `hdr: true`, regardless of the destination
/// `Image`'s own declared format, so the destination must match that
/// intermediate (`Rgba16Float`) rather than a tighter format like
/// `R32Float` -- an earlier attempt at exactly this mismatch, documented on
/// `CoarseMarcherMaterial`, is why this isn't attempted again here. Step
/// counts (typically 0-128, `FINE_MAX_STEPS`) are small integers `f16`
/// represents exactly, so no precision is lost.
fn make_stepdata_render_target_image(size: UVec2) -> Image {
    let mut image = Image::new_fill(
        Extent3d { width: size.x, height: size.y, depth_or_array_layers: 1 },
        TextureDimension::D2,
        &[0u8; 8],
        TextureFormat::Rgba16Float,
        bevy::asset::RenderAssetUsages::default(),
    );
    // `COPY_SRC` (missing from `mrrm.rs`'s coarse target, which is never
    // read back to the CPU) is load-bearing here specifically: `Readback::
    // texture`'s own `copy_texture_to_buffer` call requires it on the
    // *source* texture, and its absence isn't a loud error -- caught live
    // during this feature's own verification, the missing flag makes that
    // one copy operation invalid, which (per this session's established
    // `RESOLVE_BUFFER_STRIDE` lesson in `gpu_profile.rs`) silently no-ops
    // Bevy's *entire* batched `queue.submit()` for the frame, not just this
    // pass's own copy -- the whole canvas renders black with no console
    // error at all, not a narrowly-scoped failure.
    image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_DST
        | TextureUsages::COPY_SRC
        | TextureUsages::RENDER_ATTACHMENT;
    image
}

/// `Startup` system, chained after `mrrm::setup_mrrm_pipeline` (needs
/// `CoarseRenderTarget` to already exist for this pass's own `coarse_tex`
/// binding). Only spawns this pass's camera/quad/readback at all when
/// `Config.gpu_profile_enabled` is true at startup -- true no-op otherwise:
/// no camera, no render-graph subgraph, no readback, matching the standard
/// every other profiling feature in this app already holds itself to.
///
/// Eight `SystemParam`s is one over clippy's default `too_many_arguments`
/// threshold -- same situation, same reasoning, as `mrrm::setup_mrrm_pipeline`:
/// splitting this into multiple chained systems just to dodge the lint
/// would need an intermediate resource purely to shuttle values between
/// them, which is more indirection, not less.
#[allow(clippy::too_many_arguments)]
pub fn setup_step_data_pipeline(
    mut commands: Commands,
    config: Res<Config>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StepDataMaterial>>,
    mut shaders: ResMut<Assets<Shader>>,
    mut images: ResMut<Assets<Image>>,
    coarse_render_target: Res<CoarseRenderTarget>,
    mp: Res<MultiplayerSession>,
) {
    if !config.gpu_profile_enabled {
        return;
    }

    let wgsl = generate_stepdata_shader(&mp.sim.scene().object);
    shaders.insert(STEPDATA_SHADER_HANDLE.id(), Shader::from_wgsl(wgsl, "generated://marcher_stepdata.wgsl"));

    let image_handle = images.add(make_stepdata_render_target_image(STEPDATA_TARGET_SIZE));
    let material = materials.add(StepDataMaterial { coarse: coarse_render_target.image.clone() });

    commands
        .spawn((
            Camera2d,
            Camera {
                target: RenderTarget::from(image_handle.clone()),
                // Only depends on the coarse pass's output (order -2), not
                // on the shadow (-1) or fine (0, default) passes at all --
                // ordered last (after present's order 1) purely so it's
                // obviously "extra"/independent at a glance, not because
                // anything downstream depends on it running at a
                // particular time.
                order: 10,
                hdr: true,
                ..default()
            },
            // Same reasoning as `mrrm.rs`'s coarse camera: this target holds
            // a raw data value (step count), not a color to blend.
            Msaa::Off,
            RenderLayers::layer(STEPDATA_LAYER),
            StepDataCamera,
            // Re-triggers every frame on its own as long as this component
            // stays attached (`bevy_render::gpu_readback`'s doc) -- no
            // per-frame re-request system needed on this module's part.
            Readback::texture(image_handle.clone()),
        ))
        .observe(on_step_data_readback);

    commands.spawn((
        Mesh2d(meshes.add(Rectangle::new(1.0, 1.0).mesh())),
        MeshMaterial2d(material),
        // Must match the render target's own pixel size, same as every
        // other pass's quad (`mrrm::sync_coarse_quad_scale`, `shadow_pass::
        // sync_shadow_quad_scale`, `render::sync_quad_scale`) -- Bevy 2D's
        // default `OrthographicProjection`/`ScalingMode::WindowSize` maps
        // world units to pixels 1:1, so a `Transform::default()` quad
        // (`scale` left at `(1, 1, 1)`) only covers a single-pixel corner
        // of this target, leaving the rest at its all-zero initial fill --
        // caught live during this feature's own verification (a near-zero
        // average that made no sense given the real fine-pass GPU time).
        // Set once here, not synced every frame like the other passes': this
        // target's size ([`STEPDATA_TARGET_SIZE`]) is a fixed constant, not
        // window-size-derived, so it never needs to change after this.
        Transform::from_scale(Vec3::new(STEPDATA_TARGET_SIZE.x as f32, STEPDATA_TARGET_SIZE.y as f32, 1.0)),
        RenderLayers::layer(STEPDATA_LAYER),
        StepDataQuad,
    ));
}

/// Observer, fires once per frame this pass is active (`ReadbackComplete`
/// delivered into the main world by `bevy_render::gpu_readback`'s own
/// `sync_readbacks`, `ExtractSchedule`-side). Averages every texel's R
/// (step-count) value out of the raw bytes -- accounting for WebGPU's
/// row-byte-alignment padding via the same `RenderDevice::
/// align_copy_bytes_per_row` Bevy's own readback machinery uses internally,
/// not an assumption that `STEPDATA_TARGET_SIZE`'s width happens to avoid
/// padding -- then scales by the real fine pass's current active pixel
/// count for the cumulative estimate.
fn on_step_data_readback(
    trigger: Trigger<ReadbackComplete>,
    mut estimated: ResMut<EstimatedSteps>,
    fine_render_target: Option<Res<FineRenderTarget>>,
) {
    const BYTES_PER_PIXEL: usize = 8; // Rgba16Float: 4 * f16.
    let bytes = &trigger.event().0;
    let width = STEPDATA_TARGET_SIZE.x as usize;
    let height = STEPDATA_TARGET_SIZE.y as usize;
    let bytes_per_row = RenderDevice::align_copy_bytes_per_row(width * BYTES_PER_PIXEL);

    let mut sum = 0.0f64;
    let mut count = 0usize;
    for y in 0..height {
        let row_start = y * bytes_per_row;
        for x in 0..width {
            let px_start = row_start + x * BYTES_PER_PIXEL;
            let Some(r_bytes) = bytes.get(px_start..px_start + 2) else { continue };
            let r = half::f16::from_le_bytes([r_bytes[0], r_bytes[1]]).to_f32();
            sum += r as f64;
            count += 1;
        }
    }
    if count == 0 {
        return;
    }
    let avg = (sum / count as f64) as f32;
    estimated.avg_iters_per_px = Some(avg);
    if let Some(fine_render_target) = fine_render_target {
        let active = fine_render_target.active_size;
        estimated.estimated_total_steps = Some(avg as f64 * active.x as f64 * active.y as f64);
    }
}

/// `Update` system (main world): reports the current `EstimatedSteps` to
/// the JS overlay (`index.html`) via `net.rs`'s `js_bridge`, once per frame
/// -- same "-1.0 sentinel for not-yet-available" convention as
/// `gpu_profile::report_pass_timings_to_js`. Only runs at all while
/// `gpuprofile` is on -- true no-op otherwise, matching every other
/// profiling system in this app.
pub fn report_step_data_to_js(config: Res<Config>, estimated: Res<EstimatedSteps>) {
    if !config.gpu_profile_enabled {
        return;
    }
    crate::net::js_bridge::report_step_data(
        estimated.avg_iters_per_px.map(|v| v as f64).unwrap_or(-1.0),
        estimated.estimated_total_steps.unwrap_or(-1.0),
    );
}

/// Registers the `EstimatedSteps` resource and its JS-reporting system.
/// `bevy_render`'s own `RenderPlugin` already adds `GpuReadbackPlugin`
/// unconditionally as part of its own default plugin group (confirmed by
/// reading `bevy_render-0.16.1/src/lib.rs:409` directly -- adding it a
/// second time here panics at startup with "plugin was already added in
/// application", caught live during this feature's own verification), so
/// this `Plugin` doesn't add it itself. The pass's own camera/quad/material
/// and the `Startup` system that (conditionally) spawns them are registered
/// directly in `main.rs`, matching this codebase's existing convention for
/// the coarse/shadow/fine passes (plain functions `main.rs` wires up, not a
/// second layer of plugin structs) -- this `Plugin` exists only for the
/// `Update`-schedule registration a plain function can't do on its own.
pub struct StepDataPlugin;

impl Plugin for StepDataPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EstimatedSteps>().add_systems(Update, report_step_data_to_js);
    }
}
