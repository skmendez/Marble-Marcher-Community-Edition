//! Persistent GPU buffers for the marcher's per-frame data (camera uniforms,
//! CSG params, live marble list), written in place with `queue.write_buffer`
//! instead of round-tripping through `Assets` mutation.
//!
//! Why this module exists: the `AsBindGroup` derive's convenience path treats
//! a material as "an asset that occasionally changes" -- any
//! `Assets::get_mut` on a material (or `ShaderStorageBuffer::set_data` on a
//! storage asset) marks it modified, and the render world responds by
//! re-running the whole `as_bind_group`: a **new** uniform buffer, a **new**
//! storage buffer (`create_buffer_with_data`), and a **new** bind group,
//! every time. With the camera basis changing nearly every frame during
//! play, the old per-frame `update_material`/`update_coarse_material`/
//! `update_shadow_material` systems paid that full recreation cost (1
//! storage + 3 uniform buffers + 3-4 bind groups per frame) for what is
//! semantically a ~160-byte content update. Confirmed directly against
//! production this session: ~179 buffers/sec, ~359 bind groups/sec, zero
//! `destroy()` calls, on the default (continuously-animated) scene -- a real,
//! unbounded resource leak, not just wasted CPU/GPU time.
//!
//! The replacement data flow:
//!  1. Main world: `render::update_frame_data` writes all per-frame values
//!     into [`MarcherFrameData`] (plain CPU data, one source of truth for
//!     all three passes).
//!  2. `ExtractResourcePlugin` clones it into the render world each frame
//!     (it changes every frame -- time/camera -- so this always extracts;
//!     the clone is a few hundred bytes to a few KiB).
//!  3. [`write_marcher_buffers`] (render world, `RenderSet::PrepareResources`)
//!     uploads it into the persistent [`MarcherGpuBuffers`] via
//!     `UniformBuffer::write_buffer`/`StorageBuffer::write_buffer` -- after
//!     the first frame allocates each buffer, these are in-place
//!     `queue.write_buffer` calls: no new GPU objects, in the common case.
//!  4. The three materials' hand-written `AsBindGroup` impls
//!     (`render.rs`/`mrrm.rs`/`shadow_pass.rs`) bind these shared buffers,
//!     so their bind groups are created once and only rebuilt when
//!     something that actually invalidates them changes: a render-target
//!     *texture handle* (resize), or -- see below -- a storage buffer
//!     genuinely growing past its allocated capacity.
//!
//! **Deviation from the reference design this was blueprinted from**: an
//! earlier version of this fix (on a since-abandoned branch, predating
//! multiplayer's join-time player growth landing in its form here) asserted
//! that the params/marbles slot counts never change after the first frame,
//! on the theory that a growing storage buffer would silently stale every
//! material's bind group. That assumption doesn't hold in this codebase:
//! `physics_sys.rs`'s `apply_resync_payload`/`marble_physics_tick_impl` grow
//! the live player count via `RollbackSim::add_player_at` the moment a
//! second player joins (`marble_state.marbles.len()` goes from 1 to 2
//! mid-session, not at startup) -- asserting against that would panic on
//! every real multiplayer join. Handled properly instead: `write_buffer`'s
//! own capacity-growth reallocation is allowed to happen (`bevy_render`
//! already does this safely for its own per-frame-growing buffers), and
//! `render::update_frame_data` detects a marble-count (or, defensively,
//! params-count) change in the *main world* and does one explicit,
//! `Assets::get_mut` touch of exactly the material(s) whose bind group
//! references that buffer -- forcing exactly one rebuild on the rare frame
//! the count actually changes, not every frame. A full scene resync
//! (`render::apply_pending_scene_sync`) already forces a shader regen
//! (pipeline recompile regardless), so it does the same touch for all three
//! materials unconditionally there, rather than trying to detect whether
//! the new scene's param count happens to differ.
//!
//! Stability invariant otherwise unchanged: a bind group holds the specific
//! buffer *objects* it was created with. `UniformBuffer<SceneUniforms>` is
//! fixed-size so it never reallocates after the first write. The two
//! storage buffers only reallocate when their serialized size grows, which
//! (per the above) is now a handled, not asserted-against, case.

use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_resource::{StorageBuffer, UniformBuffer};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::{Render, RenderApp, RenderSet};

use crate::render::SceneUniforms;

/// All per-frame GPU-bound data for the three marcher passes, produced by
/// `render::update_frame_data` in the main world. Extracted (cloned) into
/// the render world each frame and uploaded by [`write_marcher_buffers`].
#[derive(Resource, Clone, ExtractResource)]
pub struct MarcherFrameData {
    /// The fine pass's `SceneUniforms` (window-sized target, sun/bg colors,
    /// MRRM + shadow-LOD + perfprobe-step-override flags).
    pub fine: SceneUniforms,
    /// The MRRM coarse pre-pass's `SceneUniforms` (its own target height in
    /// `misc.z`; no sun/colors -- the coarse shader doesn't shade).
    pub coarse: SceneUniforms,
    /// The shadow/AO pass's `SceneUniforms` (its own target height, sun
    /// direction).
    pub shadow: SceneUniforms,
    /// `Params::slots()` -- the CSG tree's runtime parameter table.
    pub params: Vec<Vec4>,
    /// One `xyz = center, w = radius` entry per marble, same order as
    /// `MarbleState::marbles`. Grows when a player joins -- see module doc.
    pub marbles: Vec<Vec4>,
}

/// The render world's persistent buffers. Allocated on first write, then
/// updated in place -- see module doc for the no-reallocation-in-the-common-
/// case invariant.
#[derive(Resource, Default)]
pub struct MarcherGpuBuffers {
    pub fine_scene: UniformBuffer<SceneUniforms>,
    pub coarse_scene: UniformBuffer<SceneUniforms>,
    pub shadow_scene: UniformBuffer<SceneUniforms>,
    pub params: StorageBuffer<Vec<Vec4>>,
    pub marbles: StorageBuffer<Vec<Vec4>>,
}

/// Render-world system (`RenderSet::PrepareResources`): uploads the frame's
/// extracted [`MarcherFrameData`] into [`MarcherGpuBuffers`]. Steady-state
/// cost is five `queue.write_buffer` calls totalling well under 2 KiB -- the
/// same mechanism Bevy itself uses for view uniforms every frame. On the
/// rare frame a storage buffer's serialized size grows past its allocated
/// capacity, `write_buffer` reallocates that one underlying GPU buffer --
/// `render::update_frame_data` is responsible for detecting that in the main
/// world and forcing the affected material(s) to rebuild their bind group
/// against the new buffer object (see module doc).
pub fn write_marcher_buffers(
    frame: Option<Res<MarcherFrameData>>,
    mut buffers: ResMut<MarcherGpuBuffers>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let Some(frame) = frame else {
        return;
    };

    buffers.fine_scene.set(frame.fine);
    buffers.coarse_scene.set(frame.coarse);
    buffers.shadow_scene.set(frame.shadow);
    buffers.params.set(frame.params.clone());
    buffers.marbles.set(frame.marbles.clone());

    buffers.fine_scene.write_buffer(&device, &queue);
    buffers.coarse_scene.write_buffer(&device, &queue);
    buffers.shadow_scene.write_buffer(&device, &queue);
    buffers.params.write_buffer(&device, &queue);
    buffers.marbles.write_buffer(&device, &queue);
}

/// Registers the extraction + upload pipeline. Must be added so that
/// [`MarcherGpuBuffers`] exists before any material prepare system runs
/// (`init_resource` at plugin-build time guarantees that regardless of
/// plugin ordering).
pub struct MarcherGpuPlugin;

impl Plugin for MarcherGpuPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<MarcherFrameData>::default());
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<MarcherGpuBuffers>();
            render_app
                .add_systems(Render, write_marcher_buffers.in_set(RenderSet::PrepareResources));
        }
    }
}
