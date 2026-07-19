//! The "present" pass: draws the (adaptively-scaled) offscreen marcher
//! render target onto the screen, upscaled to fill the actual window.
//!
//! Two-camera / two-quad split (the marcher camera + quad live in
//! `render.rs`, DESIGN.md §8):
//!  - The **marcher** camera (`render.rs::setup`, tagged [`MarcherCamera`])
//!    renders the expensive ray-march quad into an offscreen `Image` sized
//!    `AdaptiveResolution::scale * window pixel size` instead of straight to
//!    the window -- this is the entire point of the adaptive-resolution
//!    feature (`adaptive_res.rs`): the per-pixel marcher's cost is roughly
//!    proportional to pixel count, so shrinking the *internal* render
//!    target (not the window) is the cheapest way to buy back frame time.
//!  - This **present** camera + quad (below) samples that offscreen image
//!    with linear filtering and draws it stretched across the real window,
//!    so the visible result is always the full window size regardless of
//!    the marcher's current internal resolution -- just blurrier/blockier
//!    the lower the scale goes.
//!
//! `RenderLayers` keeps each camera seeing only its own quad: the marcher
//! quad stays on the default layer 0 (no component needed -- per
//! `RenderLayers`'s own doc, "Entities without this component belong to
//! layer 0"), while the present camera + quad are both explicitly on the
//! next layer up, so the marcher camera (implicitly layer-0-only) never
//! sees the present quad, and the present camera (explicitly on the other
//! layer only) never sees the marcher quad -- without this, both cameras
//! would render both quads on top of each other.
//!
//! The present camera also carries `IsDefaultUiCamera` so the existing
//! `bevy_ui`-based FPS overlay (`fps_overlay.rs`) deterministically attaches
//! to *this* camera (full window resolution, not the offscreen target).
//! Without an explicit marker, bevy_ui's fallback rule ("the highest-order
//! camera targeting the primary window") would currently happen to pick
//! this camera anyway, since it targets the window and the marcher camera
//! doesn't -- but relying on that fallback is fragile against future camera
//! additions/reordering, so it's pinned explicitly.

use bevy::asset::weak_handle;
use bevy::image::{BevyDefault, Image, ImageSampler};
use bevy::prelude::*;
use bevy::render::camera::RenderTarget;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, ShaderRef, TextureDimension, TextureFormat, TextureUsages,
};
use bevy::render::view::RenderLayers;
use bevy::sprite::Material2d;
use bevy::ui::IsDefaultUiCamera;
use bevy::window::PrimaryWindow;

use crate::adaptive_res::{target_pixel_size, AdaptiveResolution};
use crate::render::MarcherCamera;

/// All present-pass entities (camera + quad) live on this layer, kept
/// separate from the default layer 0 the marcher camera/quad use -- see
/// module doc.
const PRESENT_LAYER: usize = 1;

/// Fixed weak handle for the present pass's blit shader, following the same
/// pattern as `render::MARCHER_SHADER_HANDLE`.
const PRESENT_SHADER_HANDLE: Handle<Shader> = weak_handle!("2f6a8f7a-6f5a-4e0a-9d9a-9a0f5b8d6a11");

/// Trivial passthrough fragment shader: samples the offscreen marcher
/// render target and writes it straight to the present quad's pixel. The
/// upscale filtering comes entirely from the sampled `Image`'s own
/// `ImageSampler` (set to `ImageSampler::linear()` in
/// `make_render_target_image` below), not from anything in this shader --
/// linear so upscaling a lower-than-window-resolution internal render looks
/// smoothly blurred rather than blocky/nearest-neighbor.
const PRESENT_SHADER_WGSL: &str = r#"
#import bevy_sprite::mesh2d_vertex_output::VertexOutput

@group(2) @binding(0) var marcher_output: texture_2d<f32>;
@group(2) @binding(1) var marcher_sampler: sampler;

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(marcher_output, marcher_sampler, mesh.uv);
}
"#;

/// Minimal `Material2d` that just blits a texture -- see `PRESENT_SHADER_WGSL`.
/// `pub(crate)` only so `main.rs` can register `Material2dPlugin::<PresentMaterial>`.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub(crate) struct PresentMaterial {
    #[texture(0)]
    #[sampler(1)]
    image: Handle<Image>,
}

impl Material2d for PresentMaterial {
    fn fragment_shader() -> ShaderRef {
        PRESENT_SHADER_HANDLE.into()
    }
}

/// Marker for the present pass's camera (targets the window, samples the
/// marcher's offscreen render target) -- see module doc.
#[derive(Component)]
struct PresentCamera;

/// Marker for the present pass's fullscreen quad. `pub(crate)`: it appears
/// in `sync_present_quad_scale`'s public query type, which `main.rs` (a
/// different module) needs to be able to name when registering the system.
#[derive(Component)]
pub(crate) struct PresentQuad;

/// The marcher's offscreen render target: the `Image` asset the marcher
/// camera renders into and the present quad samples, plus its current pixel
/// size -- kept alongside the handle so `resize_marcher_render_target` can
/// cheaply tell whether the desired size actually changed (and only then
/// touch the `Image` asset, a GPU resource) instead of comparing against
/// the asset's own (borrow-requiring) current `texture_descriptor.size`
/// every frame.
#[derive(Resource)]
pub struct MarcherRenderTarget {
    pub image: Handle<Image>,
    pub size: UVec2,
}

/// Builds the offscreen `Image` the marcher camera renders into.
fn make_render_target_image(size: UVec2) -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::bevy_default(),
        bevy::asset::RenderAssetUsages::default(),
    );
    // `Image::new_fill` only sets `TEXTURE_BINDING | COPY_DST` by default
    // (a plain sampled texture); this image is also a camera render
    // target, so the GPU additionally needs `RENDER_ATTACHMENT` usage.
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::RENDER_ATTACHMENT;
    // Explicit, not left to `ImagePlugin`'s global default sampler setting:
    // this image's whole purpose is to be upscaled to the window's size by
    // the present quad, and that must look smoothly blurred, not blocky,
    // regardless of what the global default happens to be.
    image.sampler = ImageSampler::linear();
    image
}

/// `Startup` system -- must run *after* `render::setup` (chained in
/// `main.rs`), since it looks up the already-spawned [`MarcherCamera`]
/// entity to redirect its render target. Creates the marcher's initial
/// offscreen render target at native window resolution (`AdaptiveResolution`
/// only scales it down once real, sustained frame-time evidence justifies
/// it -- see `adaptive_res.rs`) and spawns the present camera + quad that
/// blit it to the screen.
pub fn setup_present_pipeline(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<PresentMaterial>>,
    mut shaders: ResMut<Assets<Shader>>,
    mut images: ResMut<Assets<Image>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut marcher_cameras: Query<&mut Camera, With<MarcherCamera>>,
) {
    shaders.insert(
        PRESENT_SHADER_HANDLE.id(),
        Shader::from_wgsl(PRESENT_SHADER_WGSL, "generated://present.wgsl"),
    );

    let native_size = windows
        .single()
        .map(|w| UVec2::new(w.physical_width().max(1), w.physical_height().max(1)))
        .unwrap_or(UVec2::new(1280, 720));

    let image_handle = images.add(make_render_target_image(native_size));

    for mut camera in &mut marcher_cameras {
        camera.target = RenderTarget::from(image_handle.clone());
    }

    commands.insert_resource(MarcherRenderTarget {
        image: image_handle.clone(),
        size: native_size,
    });

    let present_material = materials.add(PresentMaterial { image: image_handle });

    commands.spawn((
        Camera2d,
        Camera {
            // Higher order than the (default order 0) marcher camera --
            // renders after it, so the present quad (sampling the marcher's
            // completed frame) is never a frame behind (module doc).
            order: 1,
            ..default()
        },
        RenderLayers::layer(PRESENT_LAYER),
        PresentCamera,
        IsDefaultUiCamera,
    ));

    commands.spawn((
        Mesh2d(meshes.add(Rectangle::new(1.0, 1.0).mesh())),
        MeshMaterial2d(present_material),
        Transform::default(),
        RenderLayers::layer(PRESENT_LAYER),
        PresentQuad,
    ));
}

/// `Update` system: recomputes the marcher's desired target pixel size from
/// the window's current physical size and `AdaptiveResolution::scale`
/// (`adaptive_res::target_pixel_size`), and only actually touches the
/// render target -- a GPU resource -- when the *rounded* target size
/// genuinely differs from what's already there.
/// `AdaptiveResolution::scale` itself is already throttled to a few times a
/// second (`adaptive_res::adjust_resolution_scale`), so in practice this
/// system's own per-frame cost is just a cheap size comparison on most
/// frames, with a real resize only happening exactly when the scale
/// controller actively adjusts.
///
/// Builds a **new** `Image` asset and redirects every handle that points at
/// the render target (the marcher camera's own `Camera.target` and the
/// present quad's `PresentMaterial.image`) to it, rather than calling
/// `images.get_mut(&render_target.image).resize(...)` in place on the
/// *same* asset the marcher camera is actively rendering into that frame --
/// confirmed via a live, cache-disabled A/B test against this project's
/// real-GPU/WebGPU deploy target (not just this sandbox) that in-place
/// resize permanently freezes that camera's output from the very next
/// frame onward (the present pass and UI keep updating normally, so the
/// symptom reads as "the page is responsive but nothing you do ever moves
/// anything," not a crash) -- this is a known upstream Bevy behavior, not a
/// bug in this app's own logic: see
/// <https://github.com/bevyengine/bevy/issues/16159> ("Changing backing
/// image of a `RenderTarget::Image` causes rendering to that image to
/// stop") and <https://github.com/bevyengine/bevy/issues/20445> ("Calling
/// `get_mut` on an image being used as a render target causes it to not
/// get drawn on that frame"). The old `Image` asset is explicitly removed
/// (rather than left for the asset server's own GC) so a resize doesn't
/// leak a full-resolution GPU texture every time the scale changes.
pub fn resize_marcher_render_target(
    windows: Query<&Window, With<PrimaryWindow>>,
    adaptive: Res<AdaptiveResolution>,
    mut render_target: ResMut<MarcherRenderTarget>,
    mut images: ResMut<Assets<Image>>,
    mut marcher_cameras: Query<&mut Camera, With<MarcherCamera>>,
    present_quads: Query<&MeshMaterial2d<PresentMaterial>, With<PresentQuad>>,
    mut present_materials: ResMut<Assets<PresentMaterial>>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let native_size = UVec2::new(window.physical_width().max(1), window.physical_height().max(1));
    let desired = target_pixel_size(native_size, adaptive.scale);
    if desired == render_target.size {
        return;
    }

    let new_handle = images.add(make_render_target_image(desired));

    for mut camera in &mut marcher_cameras {
        camera.target = RenderTarget::from(new_handle.clone());
    }
    for mesh_material in &present_quads {
        if let Some(mat) = present_materials.get_mut(&mesh_material.0) {
            mat.image = new_handle.clone();
        }
    }

    images.remove(&render_target.image);
    render_target.image = new_handle;
    render_target.size = desired;
}

/// Keeps the present quad's world size equal to the window's (logical)
/// pixel size every frame -- same pattern as (pre-adaptive-resolution)
/// `render::sync_quad_scale`, just scoped to [`PresentQuad`]: this quad
/// always fills the actual window regardless of the marcher's current
/// internal render-target size, since it's the thing the player actually
/// sees on screen.
pub fn sync_present_quad_scale(
    windows: Query<&Window, With<PrimaryWindow>>,
    mut quads: Query<&mut Transform, With<PresentQuad>>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    for mut transform in &mut quads {
        transform.scale.x = window.width();
        transform.scale.y = window.height();
    }
}
