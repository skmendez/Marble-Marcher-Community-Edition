//! Bypasses per-frame `Assets<M>::get_mut` / `Assets<ShaderStorageBuffer>::get_mut`
//! for GPU content that changes every frame but never changes size, writing
//! straight into the already-allocated GPU buffer instead.
//!
//! ## Why this exists
//!
//! `render.rs`'s module doc previously assumed `Assets<ShaderStorageBuffer>::
//! get_mut` + `ShaderStorageBuffer::set_data` was "a pure buffer write, no
//! shader recompile" -- true of the *shader*, but not of the underlying GPU
//! resource. `Assets::get_mut` marks the asset `Modified`, and Bevy's generic
//! `RenderAsset` extract/prepare pipeline (`bevy_render::render_asset::
//! prepare_assets`) reacts to *any* Modified event by calling
//! `RenderAsset::prepare_asset` completely from scratch. For
//! `GpuShaderStorageBuffer` that's `render_device.create_buffer_with_data(..)`
//! (confirmed by reading `bevy_render-0.16.1/src/storage.rs`) -- a brand-new
//! wgpu buffer every time, discarding (not `.destroy()`-ing -- just
//! dropping, relying on eventual GC) the old one. The exact same thing
//! happens to `Material2d`s: `PreparedMaterial2d<M>` is *itself* a
//! `RenderAsset` (`bevy_sprite-0.16.1/src/mesh2d/material.rs`), so
//! `Assets<M>::get_mut` on a material triggers a fresh
//! `material.as_bind_group(..)` call -- a new uniform buffer *and* a new
//! bind group, every single Modified event.
//!
//! This app's per-frame render-update systems (`render.rs::update_material`,
//! `mrrm.rs::update_coarse_material`, `shadow_pass.rs::update_shadow_material`)
//! called `.get_mut()` on all three materials (camera uniforms) *every
//! frame* unconditionally, plus the marbles storage buffer every frame and
//! the params storage buffer whenever the scene has any animated params
//! (true for the default `MengerOscillatingSphere` scene) -- so this was
//! allocating a fresh GPU buffer and bind group for every one of those,
//! every single frame, forever. Caught live via a WebGPU call-tracing
//! extension (github.com/brendan-duncan/webgpu_inspector) showing the live
//! buffer/bind-group count grow without bound.
//!
//! The fix: for *same-size* content-only updates, skip the `Assets<T>`
//! machinery entirely and write straight into the already-prepared GPU
//! resource via `RenderQueue::write_buffer` -- a plain in-place GPU copy
//! that touches neither the buffer's nor the bind group's identity, so nothing
//! new is ever created and nothing needs collecting. `Assets<T>::get_mut`
//! remains the correct, unavoidable path for the rare *structural* cases
//! where a buffer's byte length actually needs to change (a new scene loaded
//! via `apply_pending_scene_sync`, or growing the marbles buffer when a new
//! player joins) -- reallocating there is genuinely necessary and
//! infrequent, not the problem this module exists to fix.
//!
//! ## Mechanism
//!
//! [`FrameGpuWrites`] is a plain (non-asset) `Resource` populated every
//! frame in the main world by the same systems that used to call
//! `Assets::get_mut`, then copied into the render world every frame by
//! Bevy's own [`ExtractResourcePlugin`] (a cheap `Clone`, not asset
//! machinery -- doesn't touch `RenderAssets<A>` at all). [`apply_frame_gpu_writes`]
//! runs once per frame in the render world, after `RenderSet::PrepareAssets`
//! (so the target buffer/bind group from the *first* time each asset was
//! created already exists), and reaches directly into
//! `RenderAssets<PreparedMaterial2d<M>>`/`RenderAssets<GpuShaderStorageBuffer>`
//! to find the already-live wgpu buffer for binding 0 (`SceneUniforms`) or
//! the whole storage buffer, and calls `RenderQueue::write_buffer` on it
//! directly. `PreparedMaterial2d::bindings`/`GpuShaderStorageBuffer::buffer`
//! are public exactly to support this kind of direct-write pattern.

use bevy::asset::AssetId;
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::encase;
use bevy::render::render_resource::OwnedBindingResource;
use bevy::render::renderer::RenderQueue;
use bevy::render::storage::{GpuShaderStorageBuffer, ShaderStorageBuffer};
use bevy::render::{Render, RenderApp, RenderSet};
use bevy::sprite::{Material2d, PreparedMaterial2d};

use crate::mrrm::CoarseMarcherMaterial;
use crate::render::{FineMarcherMaterial, SceneUniforms};
use crate::shadow_pass::ShadowMarcherMaterial;

/// This frame's raw content for every GPU binding that should be
/// direct-written rather than routed through `Assets<T>::get_mut` -- see
/// module doc. `None` means "nothing to write this frame" (e.g. a
/// size-changing marbles update took the ordinary `Assets::get_mut` path
/// instead, so there's nothing left for the direct-write system to do for
/// that binding this frame).
#[derive(Resource, Clone, Default)]
pub struct FrameGpuWrites {
    pub fine_scene: Option<(AssetId<FineMarcherMaterial>, SceneUniforms)>,
    pub coarse_scene: Option<(AssetId<CoarseMarcherMaterial>, SceneUniforms)>,
    pub shadow_scene: Option<(AssetId<ShadowMarcherMaterial>, SceneUniforms)>,
    /// Raw `Vec4` slots, not pre-encoded bytes -- encoded on the render-world
    /// side via `encase::StorageBuffer`, matching exactly what
    /// `ShaderStorageBuffer::set_data`/`From<Vec<Vec4>>` already do for the
    /// ordinary (occasional, size-changing) path, so a caller can freely mix
    /// this direct-write path and that one across frames without the two
    /// ever disagreeing about layout.
    pub params: Option<(AssetId<ShaderStorageBuffer>, Vec<Vec4>)>,
    pub marbles: Option<(AssetId<ShaderStorageBuffer>, Vec<Vec4>)>,
}

impl ExtractResource for FrameGpuWrites {
    type Source = Self;

    fn extract_resource(source: &Self) -> Self {
        source.clone()
    }
}

pub struct DirectGpuWritesPlugin;

impl Plugin for DirectGpuWritesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FrameGpuWrites>()
            .add_plugins(ExtractResourcePlugin::<FrameGpuWrites>::default());
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_systems(Render, apply_frame_gpu_writes.in_set(RenderSet::Prepare));
        }
    }
}

/// Render-world system: consumes this frame's [`FrameGpuWrites`] (already
/// copied over by `ExtractResourcePlugin`) and writes each present field
/// straight into its already-live GPU buffer. Runs in `RenderSet::Prepare`,
/// which is ordered after `RenderSet::PrepareAssets` -- the set that runs
/// `prepare_assets::<PreparedMaterial2d<M>>`/`prepare_assets::
/// <GpuShaderStorageBuffer>` -- so every asset id referenced here is
/// guaranteed to already have a `RenderAssets` entry from at least its first
/// (ordinary, `Assets::add`-driven) creation.
fn apply_frame_gpu_writes(
    writes: Res<FrameGpuWrites>,
    queue: Res<RenderQueue>,
    fine_materials: Res<RenderAssets<PreparedMaterial2d<FineMarcherMaterial>>>,
    coarse_materials: Res<RenderAssets<PreparedMaterial2d<CoarseMarcherMaterial>>>,
    shadow_materials: Res<RenderAssets<PreparedMaterial2d<ShadowMarcherMaterial>>>,
    storage_buffers: Res<RenderAssets<GpuShaderStorageBuffer>>,
) {
    if let Some((id, scene)) = &writes.fine_scene {
        write_scene_uniform(&fine_materials, *id, &queue, scene);
    }
    if let Some((id, scene)) = &writes.coarse_scene {
        write_scene_uniform(&coarse_materials, *id, &queue, scene);
    }
    if let Some((id, scene)) = &writes.shadow_scene {
        write_scene_uniform(&shadow_materials, *id, &queue, scene);
    }
    if let Some((id, slots)) = &writes.params {
        write_storage_vec4(&storage_buffers, *id, &queue, slots);
    }
    if let Some((id, slots)) = &writes.marbles {
        write_storage_vec4(&storage_buffers, *id, &queue, slots);
    }
}

/// Writes `value` into binding 0 (`#[uniform(0)] scene: SceneUniforms` on
/// every one of this app's three materials) of `id`'s already-prepared bind
/// group, encoded exactly the way `AsBindGroup`'s own derive would (WGSL
/// uniform-address-space layout via `encase`) -- if `id` has no prepared
/// bind group yet (the very first frame or two, before `Startup`'s
/// `Assets::add` has been extracted/prepared), this frame's write is simply
/// skipped; the material's own `SceneUniforms::default()` initial value
/// (`render.rs::setup`) is a harmless placeholder for that handful of
/// frames, and the next frame retries.
fn write_scene_uniform<M: Material2d>(
    materials: &RenderAssets<PreparedMaterial2d<M>>,
    id: AssetId<M>,
    queue: &RenderQueue,
    value: &SceneUniforms,
) {
    let Some(prepared) = materials.get(id) else {
        return;
    };
    let Some((_, OwnedBindingResource::Buffer(buffer))) =
        prepared.bindings.iter().find(|(index, _)| *index == 0)
    else {
        return;
    };
    let mut scratch = encase::UniformBuffer::new(Vec::new());
    scratch
        .write(value)
        .expect("SceneUniforms is a fixed, already-uniform-compatible layout");
    queue.write_buffer(buffer, 0, scratch.as_ref());
}

/// Encodes `slots` exactly as `ShaderStorageBuffer::set_data`/
/// `From<Vec<Vec4>>` would (`encase::StorageBuffer`, matching `params`'s/
/// `marbles`' `var<storage, read> array<vec4<f32>>` WGSL binding) and writes
/// it into `id`'s already-allocated storage buffer. Caller's responsibility
/// to only call this when the encoded length matches what the buffer was
/// already sized for (see module doc) -- a mismatched length would either
/// silently truncate or panic deep in wgpu's validation, neither of which
/// this function can safely paper over.
fn write_storage_vec4(
    buffers: &RenderAssets<GpuShaderStorageBuffer>,
    id: AssetId<ShaderStorageBuffer>,
    queue: &RenderQueue,
    slots: &[Vec4],
) {
    let Some(gpu_buffer) = buffers.get(id) else {
        return;
    };
    let mut scratch = encase::StorageBuffer::new(Vec::new());
    scratch
        .write(&slots.to_vec())
        .expect("Vec<Vec4> is always a valid WGSL storage-buffer layout");
    queue.write_buffer(&gpu_buffer.buffer, 0, scratch.as_ref());
}
