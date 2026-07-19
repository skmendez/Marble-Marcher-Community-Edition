//! MRRM (multi-resolution ray marching, named after the original C++ Marble
//! Marcher's `MRRM1`/`MRRM2` compute passes): a coarse, low-resolution
//! ray-march pre-pass whose cached hit distance the fine (full-resolution)
//! pass in `render.rs` starts its own march from, instead of from the
//! camera (`t=0`). This is the single biggest remaining performance lever
//! for this renderer -- it cuts *per-pixel* march-step cost (every fine
//! pixel skips almost all of the empty-space traversal its neighbors in the
//! same coarse texel already paid for), on top of (not instead of) the
//! adaptive internal *resolution* scaling `adaptive_res.rs`/`present.rs`
//! already do, which only cuts pixel *count*.
//!
//! Three passes per frame now, ordered by `Camera.order` (lowest first):
//!  1. **This module's coarse pass** (`CoarseCamera`/`CoarseQuad`): marches
//!     `de_scene` from `t=0` into a small offscreen `Rgba16Float` render
//!     target (the hit distance in R, `-1.0` sentinel on a miss, G/B/A
//!     unused), using its *own* resolution's cone angle
//!     (`update_coarse_material`'s `misc.z`) -- skips all shading (no
//!     normals/color/shadow).
//!  2. **The fine pass** (`render.rs`'s `MarcherCamera`/`MarcherQuad`,
//!     `FineMarcherMaterial`): reads this pass's render target back as a
//!     starting-`t` guess (backed off by one coarse-texel's angular
//!     footprint) and does the full march + shading from there.
//!  3. **The present pass** (`present.rs`, unchanged): blits the fine
//!     pass's result to the window.
//!
//! Sized as a fixed fraction of the *fine* render target's current size
//! (`coarse_target_size`), not the window's, so it automatically tracks both
//! window resizes and adaptive-resolution scale changes without this module
//! needing to know about either -- mirrors `present.rs`'s
//! `resize_marcher_render_target` doc for the same reasoning, just with the
//! fine target playing the role the window plays there.
//!
//! `CoarseCamera`/`CoarseQuad` get their own `RenderLayers` (like
//! `present.rs`'s present-pass layer): entities without a `RenderLayers`
//! component default to layer 0 (the fine marcher quad's layer), so without
//! an explicit, distinct layer here, the fine camera would also render the
//! coarse quad on top of its own output (and vice versa).

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

use marble_csg::codegen::generate_coarse_shader;

use crate::camera::CameraOrbit;
use crate::physics_sys::MarbleState;
use crate::present::MarcherRenderTarget;
use crate::render::{SceneState, SceneUniforms};

/// All MRRM coarse-pass entities (camera + quad) live on this layer --
/// distinct from the fine marcher's (implicit, unmarked) layer 0 and the
/// present pass's layer 1 (`present::PRESENT_LAYER`) -- see module doc.
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

/// `MM_MRRM=0` disables the fine pass's use of this module's coarse guess
/// (falls back to always starting its march at `t=0`, i.e. exactly the
/// pre-MRRM behavior) -- a per-frame *shader* toggle (`SceneUniforms::misc.w`,
/// written by `render::update_material`) rather than an entity/system-level
/// one that would stop this module's camera/pass from running at all: every
/// frame's cameras/passes are identical whether MRRM is on or off, so an
/// `MM_MRRM=0` vs `MM_MRRM=1` A/B screenshot comparison at a fixed camera
/// state only ever differs in this one shader value, not in *what ran* --
/// which is what makes that comparison trustworthy as a regression check.
/// Matches this codebase's other `MM_*` testing-hook convention
/// (`render.rs`'s `MM_SCENE`, `adaptive_res.rs`'s `MM_FORCE_RES_SCALE`).
/// `std::env::var` always errors on wasm32-unknown-unknown, so this has no
/// effect (MRRM always on) in the deployed web build.
pub fn mrrm_enabled() -> bool {
    std::env::var("MM_MRRM").as_deref() != Ok("0")
}

/// Rounds the fine render target's current pixel size down by
/// `COARSE_SCALE_DIVISOR` in each dimension, flooring at 1px (a 0-sized
/// `Image`/texture is invalid) -- pure and unit-tested like
/// `adaptive_res::target_pixel_size`, which this mirrors.
pub fn coarse_target_size(fine_size: UVec2) -> UVec2 {
    UVec2::new(
        (fine_size.x / COARSE_SCALE_DIVISOR).max(1),
        (fine_size.y / COARSE_SCALE_DIVISOR).max(1),
    )
}

/// Fixed weak handle for the generated MRRM coarse-pass shader, following
/// the same pattern as `render::MARCHER_SHADER_HANDLE`.
const COARSE_MARCHER_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("6f1a2c3d-6a10-4f9a-8b7d-2a6b0c9f5e21");

/// The MRRM coarse pass's own material: just `scene`/`params` (identical
/// bindings 0/1 to `FineMarcherMaterial`) -- no `coarse_tex` binding (this
/// pass doesn't read its own output), so this is a genuinely different
/// bind-group layout from the fine pass's, hence a separate `Material2d`
/// struct/generated shader module rather than a second entry point sharing
/// one (see `marble_csg::codegen::COARSE_MARCHER`'s doc for why the module
/// itself is also separate, not just the entry point).
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct CoarseMarcherMaterial {
    #[uniform(0)]
    pub scene: SceneUniforms,
    #[storage(1, read_only)]
    pub params: Handle<ShaderStorageBuffer>,
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
/// the three (renders first, so the fine pass can read its finished output
/// later in the same frame; see module doc).
#[derive(Component)]
struct CoarseCamera;

/// Marker for the MRRM coarse pass's fullscreen quad. `pub(crate)`: it
/// appears in `sync_coarse_quad_scale`/`update_coarse_material`'s public
/// query types, which `main.rs` (a different module) needs to be able to
/// name when registering those systems -- same reasoning as
/// `present::PresentQuad`'s doc.
#[derive(Component)]
pub(crate) struct CoarseQuad;

/// The coarse pass's offscreen render target, mirroring
/// `present::MarcherRenderTarget`'s shape/role for the fine target.
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
    // Same reasoning as `present::make_render_target_image`: `Image::new_fill`
    // only sets a plain sampled-texture usage by default, and this image is
    // also a camera render target, so it additionally needs
    // `RENDER_ATTACHMENT`.
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::RENDER_ATTACHMENT;
    image
}

/// `Startup` system -- chained to run after both `render::setup` (needs
/// `SceneState`'s scene tree/params buffer to build the coarse shader and
/// material) and `present::setup_present_pipeline` (needs `MarcherRenderTarget`
/// to already exist, to size the coarse target off the fine target's actual
/// initial size rather than independently re-deriving it from the window).
/// Corrects `FineMarcherMaterial`'s placeholder `coarse` handle
/// (`render::setup`) to point at the real image, same "spawn now, redirect
/// once the real asset exists" pattern `present.rs` already uses for
/// `MarcherCamera`'s render target.
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
    fine_render_target: Res<MarcherRenderTarget>,
    scene_state: Res<SceneState>,
) {
    let wgsl = generate_coarse_shader(&scene_state.object);
    shaders.insert(
        COARSE_MARCHER_SHADER_HANDLE.id(),
        Shader::from_wgsl(wgsl, "generated://marcher_coarse.wgsl"),
    );

    let size = coarse_target_size(fine_render_target.size);
    let image_handle = images.add(make_coarse_render_target_image(size));

    if let Some(fine_material) = fine_materials.get_mut(&scene_state.material) {
        fine_material.coarse = image_handle.clone();
    }

    let coarse_material = coarse_materials.add(CoarseMarcherMaterial {
        scene: SceneUniforms::default(),
        params: scene_state.params_buffer.clone(),
    });

    commands.spawn((
        Camera2d,
        Camera {
            target: RenderTarget::from(image_handle.clone()),
            // Lowest order of the three passes -- must finish before the
            // fine pass (`render.rs`, order 0 by default) reads this
            // texture back (module doc).
            order: -1,
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
/// `coarse_target_size` of the *fine* target's current size (`present.rs`'s
/// `MarcherRenderTarget`, already itself adaptively resized) -- same
/// touch-the-GPU-resource-only-on-real-change guard as
/// `present::resize_marcher_render_target`. No re-fill of the resized
/// texture's contents is needed: the coarse camera (lower `Camera.order`
/// than the fine camera reading it) fully redraws every pixel of it every
/// frame before the fine pass ever samples it, resize or not.
pub fn resize_coarse_render_target(
    fine_render_target: Res<MarcherRenderTarget>,
    mut coarse_render_target: ResMut<CoarseRenderTarget>,
    mut images: ResMut<Assets<Image>>,
) {
    let desired = coarse_target_size(fine_render_target.size);
    if desired == coarse_render_target.size {
        return;
    }
    if let Some(image) = images.get_mut(&coarse_render_target.image) {
        image.resize(Extent3d {
            width: desired.x,
            height: desired.y,
            depth_or_array_layers: 1,
        });
        coarse_render_target.size = desired;
    }
}

/// Keeps the coarse quad's world size equal to its own render target's
/// pixel size every frame -- mirrors `render::sync_quad_scale` for the fine
/// quad/target.
pub fn sync_coarse_quad_scale(
    render_target: Res<CoarseRenderTarget>,
    mut quads: Query<&mut Transform, With<CoarseQuad>>,
) {
    for mut transform in &mut quads {
        transform.scale.x = render_target.size.x as f32;
        transform.scale.y = render_target.size.y as f32;
    }
}

/// `Update` system: writes the coarse pass's own `SceneUniforms` each frame
/// -- same camera basis as `render::update_material` (both passes render
/// the same scene from the same camera), but `misc.z` is *this* pass's own
/// render-target height (`coarse_render_target.size.y`), not the fine
/// pass's, so its shader's cone-angle threshold (`marble_csg::codegen`'s
/// `COARSE_MARCHER`) matches its own, much coarser, resolution.
///
/// `misc.x` (aspect) is still taken from the *fine* target's size, not this
/// pass's own (slightly different after integer-dividing by
/// `COARSE_SCALE_DIVISOR`, e.g. a 540-tall fine target gives a 67-tall
/// coarse one, a ~1% aspect-ratio drift from 540/8=67.5) -- using the fine
/// aspect keeps both passes casting rays in matching directions for the
/// same UV coordinate, which is what makes the coarse pass's hit distance a
/// meaningful guess for the fine pixel at all; the resolution mismatch
/// itself (many fine pixels per coarse texel) is already what the backed-off
/// starting-`t` guess in `render.rs`'s `fragment` accounts for, so it
/// doesn't need this aspect drift piled on top.
pub fn update_coarse_material(
    time: Res<Time>,
    orbit: Res<CameraOrbit>,
    marble_state: Res<MarbleState>,
    fine_render_target: Res<MarcherRenderTarget>,
    coarse_render_target: Res<CoarseRenderTarget>,
    quads: Query<&MeshMaterial2d<CoarseMarcherMaterial>, With<CoarseQuad>>,
    mut materials: ResMut<Assets<CoarseMarcherMaterial>>,
) {
    let t = time.elapsed_secs();
    let fine_size = fine_render_target.size;
    let aspect = fine_size.x as f32 / (fine_size.y.max(1)) as f32;
    let coarse_height = coarse_render_target.size.y as f32;

    let marble = marble_state.marble;
    let (eye, right, up, forward) = orbit.eye_and_basis(marble.pos);

    for mesh_material in &quads {
        if let Some(mat) = materials.get_mut(&mesh_material.0) {
            mat.scene = SceneUniforms {
                cam_pos: eye.extend(0.0),
                cam_right: right.extend(0.0),
                cam_up: up.extend(0.0),
                cam_forward: forward.extend(1.5),
                misc: Vec4::new(aspect, t, coarse_height, 0.0),
                ..SceneUniforms::default()
            };
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
