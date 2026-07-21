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
use bevy::image::Image;
use bevy::prelude::*;
use bevy::render::camera::RenderTarget;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, ShaderRef, TextureDimension, TextureFormat, TextureUsages,
};
use bevy::render::storage::ShaderStorageBuffer;
use bevy::render::view::RenderLayers;
use bevy::sprite::Material2d;
use bevy::window::PrimaryWindow;

use marble_csg::codegen::generate_shadow_shader;
use marble_csg::scenes::beware_of_bumps;

use crate::camera::CameraOrbit;
use crate::mrrm::CoarseRenderTarget;
use crate::physics_sys::MarbleState;
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

/// `?shadowlod=0` (web) / `MM_SHADOW_LOD=0` (native) disables the fine
/// pass's use of this module's cached shadow value (falls back to marching
/// a fresh shadow ray per full-res pixel, i.e. exactly the pre-this-change
/// behavior) -- a per-frame *shader* toggle (`SceneUniforms::misc2.y`,
/// written by `render::update_material`), not an entity/system-level one,
/// for the same A/B-comparability reason as `mrrm::mrrm_enabled`'s doc:
/// every frame's cameras/passes are identical whether this is on or off, so
/// an `?shadowlod=0` vs `=1` screenshot comparison at a fixed camera state
/// only ever differs in this one value. Matches `mrrm_enabled`'s
/// query-param-then-env-var layering -- this now actually has an effect on
/// the deployed web build, not just native. Cached in a `OnceLock` rather
/// than re-parsing the URL every frame.
pub fn shadow_lod_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let value =
            crate::web_config::query_param("shadowlod").or_else(|| std::env::var("MM_SHADOW_LOD").ok());
        value.as_deref() != Some("0")
    })
}

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

/// The shadow pass's own material: `scene`/`params` (bindings 0/1, same as
/// every pass) plus MRRM's coarse hit-distance texture (binding 2, this
/// pass's own warm-start source -- same bind-group shape `FineMarcherMaterial`
/// had before it also grew a `shadow` binding, see `generate_shadow_shader`'s
/// doc for why this still needs its own `Material2d`/shader module rather
/// than reusing either the coarse or fine one).
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct ShadowMarcherMaterial {
    #[uniform(0)]
    pub scene: SceneUniforms,
    #[storage(1, read_only)]
    pub params: Handle<ShaderStorageBuffer>,
    #[texture(2)]
    pub coarse: Handle<Image>,
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

    let shadow_material = shadow_materials.add(ShadowMarcherMaterial {
        scene: SceneUniforms::default(),
        params: scene_state.params_buffer.clone(),
        coarse: coarse_render_target.image.clone(),
    });

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

    // Always keep the `coarse` binding current (MRRM's own render target is
    // rebuilt independently on its own resize schedule) -- cheap (a handle
    // clone + possible no-op `Assets` write), and simpler than trying to
    // coordinate two independent resize systems' ordering.
    for shadow_quad in &shadow_quads {
        if let Some(mat) = shadow_materials.get_mut(&shadow_quad.0) {
            mat.coarse = coarse_render_target.image.clone();
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
/// size every frame -- mirrors `mrrm::sync_coarse_quad_scale`.
pub fn sync_shadow_quad_scale(
    render_target: Res<ShadowRenderTarget>,
    mut quads: Query<&mut Transform, With<ShadowQuad>>,
) {
    for mut transform in &mut quads {
        transform.scale.x = render_target.size.x as f32;
        transform.scale.y = render_target.size.y as f32;
    }
}

/// `Update` system: writes the shadow pass's own `SceneUniforms` each frame
/// -- same camera basis as `render::update_material`/`mrrm::update_coarse_material`
/// (every pass renders the same scene from the same camera), but `misc.z`
/// is *this* pass's own render-target height, and `bounding` is the scene's
/// bounding sphere (`ray_sphere_clip`'s pre-test, same value every pass
/// writes -- `SceneState::bounding_sphere`'s doc).
#[allow(clippy::too_many_arguments)]
/// Thin timing wrapper -- see `fps_overlay::PhaseTimings`'s doc for exactly
/// what this measures (CPU-side uniform computation, not GPU execution).
pub fn update_shadow_material(
    time: Res<Time>,
    orbit: Res<CameraOrbit>,
    marble_state: Res<MarbleState>,
    scene_state: Res<SceneState>,
    windows: Query<&Window, With<PrimaryWindow>>,
    shadow_render_target: Res<ShadowRenderTarget>,
    quads: Query<&MeshMaterial2d<ShadowMarcherMaterial>, With<ShadowQuad>>,
    materials: ResMut<Assets<ShadowMarcherMaterial>>,
    mut timings: ResMut<crate::fps_overlay::PhaseTimings>,
) {
    let start = web_time::Instant::now();
    update_shadow_material_impl(
        time, orbit, marble_state, scene_state, windows, shadow_render_target, quads, materials,
    );
    timings.record("shadow", start.elapsed());
}

#[allow(clippy::too_many_arguments)] // SystemParam count, unchanged from before the timing wrapper split
fn update_shadow_material_impl(
    time: Res<Time>,
    orbit: Res<CameraOrbit>,
    marble_state: Res<MarbleState>,
    scene_state: Res<SceneState>,
    windows: Query<&Window, With<PrimaryWindow>>,
    shadow_render_target: Res<ShadowRenderTarget>,
    quads: Query<&MeshMaterial2d<ShadowMarcherMaterial>, With<ShadowQuad>>,
    mut materials: ResMut<Assets<ShadowMarcherMaterial>>,
) {
    let t = time.elapsed_secs();
    let aspect = windows
        .single()
        .map(|w| w.width() / w.height().max(1.0))
        .unwrap_or(1.0);
    let shadow_height = shadow_render_target.size.y as f32;

    let marble = marble_state.local_marble();
    let (eye, right, up, forward) = orbit.eye_and_basis(marble.pos);

    for mesh_material in &quads {
        if let Some(mat) = materials.get_mut(&mesh_material.0) {
            mat.scene = SceneUniforms {
                cam_pos: eye.extend(0.0),
                cam_right: right.extend(0.0),
                cam_up: up.extend(0.0),
                cam_forward: forward.extend(1.5),
                sun: beware_of_bumps::sun_dir().extend(0.0),
                misc: Vec4::new(aspect, t, shadow_height, 0.0),
                bounding: scene_state.bounding_sphere,
                ..SceneUniforms::default()
            };
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
