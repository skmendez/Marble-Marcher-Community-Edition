//! GPU timestamp-query profiling for all 4 named render passes (coarse
//! warm-start marcher, shadow-LOD marcher, fine marcher, present/blit).
//!
//! ## Why this needs a hand-authored render-graph node
//!
//! Bevy's own `bevy_render::diagnostic::RenderDiagnosticsPlugin` cannot be
//! used for this: it's GPU-inert on WebGPU/WebGL2/Metal by its own
//! documented design (CPU time only there -- this app's primary target is
//! WebGPU), and even where GPU-functional its per-pass span names are
//! hardcoded literals per node *type*, not per camera/view, so this app's
//! multiple `Camera2d`s sharing the same stock node would all collapse
//! into one indistinguishable diagnostic path regardless of platform.
//!
//! The underlying WebGPU mechanism is real and does work on this app's
//! actual deployed target, though: `wgpu`'s WebGPU backend forwards the
//! *pass-descriptor* form of GPU timestamps (`RenderPassDescriptor::
//! timestamp_writes`, `resolve_query_set`) to real browser calls. What does
//! NOT work on WebGPU is an arbitrary mid-pass/mid-encoder
//! `write_timestamp()` call (which is what Bevy's own diagnostics use) --
//! that panics on this backend. So timestamps have to be attached at the
//! moment a render pass is *constructed*.
//!
//! Bevy's stock `MainOpaquePass2dNode` (`bevy_core_pipeline::core_2d`)
//! hardcodes `timestamp_writes: None` with no config surface, and this
//! codebase has no existing custom render-graph node to extend. Rather
//! than vendoring a patch against a specific Bevy point release (fragile,
//! silently breakable on a future `cargo update`), [`MarcherTimingNode`]
//! below is a from-scratch replacement for that stock node, registered
//! under the *same* `Node2d::MainOpaquePass` label -- `RenderGraph::
//! add_node` is a plain `HashMap` insert (confirmed by reading
//! `bevy_render`'s source directly), so re-registering under an
//! already-used label silently replaces the stock node rather than
//! erroring, while every existing graph edge (which references the label,
//! not the concrete node type) keeps working unchanged. Since `Camera2d`'s
//! own `#[require(...)]` pins every 2D camera to the `Core2d` sub-graph,
//! this one replacement node's `run()` gets invoked once per camera/view
//! per frame -- all 4 of this app's passes go through the exact same node
//! instance, distinguished only by which one (if any) carries a
//! [`GpuProfiledPass`] marker naming it.
//!
//! Trade-off accepted deliberately: the stock node uses
//! `RenderContext::add_command_buffer_generation_task` to record each
//! view's pass on a fresh, independent `CommandEncoder` (enabling
//! multi-threaded command-buffer generation across cameras). This
//! replacement instead uses the frame's single shared encoder
//! (`RenderContext::command_encoder`/`begin_tracked_render_pass`) for
//! every view, trading that parallelism for much simpler, easier-to-audit
//! code -- acceptable here since this app only has 4 cameras total, not
//! the many-view workloads that optimization targets.
//!
//! ## Four passes sharing one query set
//!
//! All 4 named passes share a single 8-slot `wgpu::QuerySet` (2 slots --
//! begin/end -- per pass) and a single 64-byte resolve buffer, rather than
//! one query set per pass: simpler resource lifecycle (one lazy-init
//! check, not four), and every pass already resolves its own 2-slot
//! sub-range independently the moment its own render pass ends (see
//! [`slot_pair`]), so sharing the underlying set/buffer costs nothing --
//! there's no cross-pass synchronization concern since each pass only
//! ever touches its own disjoint byte range.
//!
//! ## Readback
//!
//! Mirrors `bevy_render::gpu_readback::GpuReadbackPlugin`'s own shape
//! (confirmed by reading its source): a plain `async_channel` (not a Bevy
//! async task -- works identically on native and wasm), `map_async`
//! registered from a system that runs *after* `render_system` has actually
//! submitted the command buffer containing the corresponding
//! `resolve_query_set`/copy (same ordering `gpu_readback.rs`'s own
//! `map_buffers` system uses), and each resolved `(pass_name, duration)`
//! pair drained into the main world's [`PassTimings`] resource via a
//! system in `ExtractSchedule` (which has `ResMut<MainWorld>` access
//! despite running on the `RenderApp`, same as `gpu_readback.rs`'s
//! `sync_readbacks`). Still no buffer pool (a handful of small buffers
//! allocated and dropped per frame, 4 passes worth, isn't worth pooling
//! at this scale -- revisit only if profiling shows this actually matters).

use async_channel::{Receiver, Sender};

use bevy::core_pipeline::core_2d::graph::{Core2d, Node2d};
use bevy::core_pipeline::core_2d::{AlphaMask2d, Opaque2d};
use bevy::ecs::query::QueryItem;
use bevy::ecs::schedule::IntoScheduleConfigs;
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_graph::{
    NodeRunError, RenderGraphApp, RenderGraphContext, ViewNode, ViewNodeRunner,
};
use bevy::render::render_phase::ViewBinnedRenderPhases;
use bevy::render::render_resource::{Buffer, BufferDescriptor, BufferUsages, MapMode, RenderPassDescriptor, StoreOp};
use bevy::render::renderer::{render_system, RenderAdapter, RenderContext, RenderDevice, RenderQueue};
use bevy::render::view::{ExtractedView, ViewDepthTexture, ViewTarget};
use bevy::render::{ExtractSchedule, MainWorld, Render, RenderApp, RenderSet};

use crate::config::Config;

/// Marker for a camera whose Material2d pass should get real GPU
/// timestamp queries when profiling is active. Attached to each of the 4
/// named-pass cameras: the coarse camera in `mrrm.rs`'s
/// `setup_mrrm_pipeline`, the shadow camera in `shadow_pass.rs`'s
/// `setup_shadow_pipeline`, and the fine + present cameras in
/// `render.rs`'s `setup`.
#[derive(Component, Clone, Copy, ExtractComponent)]
pub struct GpuProfiledPass(pub &'static str);

pub const COARSE_PASS_NAME: &str = "coarse";
pub const SHADOW_PASS_NAME: &str = "shadow";
pub const FINE_PASS_NAME: &str = "fine";
pub const PRESENT_PASS_NAME: &str = "present";

/// Fixed order backing both the shared 8-slot query set's layout and
/// [`slot_pair`]'s lookup -- each pass owns a disjoint 2-slot sub-range
/// (index `i` in this array owns slots `2*i`/`2*i + 1`), so all 4 passes
/// can safely share one `wgpu::QuerySet`/resolve buffer.
const PASS_NAMES: [&str; 4] = [COARSE_PASS_NAME, SHADOW_PASS_NAME, FINE_PASS_NAME, PRESENT_PASS_NAME];

/// WebGPU requires `resolve_query_set`'s *destination buffer* offset to be
/// a multiple of 256 bytes (confirmed the hard way live: a tightly-packed
/// 16-bytes-per-pass layout produced a "destination buffer offset is not
/// a multiple of 256" validation warning -- which, worse, silently
/// invalidates that *entire* frame's command-buffer submission, not just
/// this one pass's resolve, since Bevy batches every view's command
/// buffer into one `queue.submit()` call and a single invalid buffer in
/// that list is a no-op for the whole call per the WebGPU spec). Each pass
/// therefore gets its own 256-byte-aligned region of the resolve buffer,
/// even though only the first 16 bytes of each region are ever actually
/// read -- the query *set* itself has no such alignment constraint, so
/// its slots stay tightly packed (2 per pass, [`slot_pair`]'s shape).
const RESOLVE_BUFFER_STRIDE: u64 = 256;

/// This pass's (begin, end) slot indices within the shared query set, or
/// `None` for an unrecognized name (never expected in practice -- every
/// [`GpuProfiledPass`] this app ever constructs uses one of the 4 named
/// constants above).
fn slot_pair(name: &str) -> Option<(u32, u32)> {
    PASS_NAMES.iter().position(|&n| n == name).map(|i| (i as u32 * 2, i as u32 * 2 + 1))
}

/// This pass's 256-byte-aligned byte offset into the shared resolve
/// buffer (see [`RESOLVE_BUFFER_STRIDE`]'s doc for why this is a
/// *separate* index space from the query set's own tightly-packed slots).
fn resolve_byte_offset(name: &str) -> Option<u64> {
    PASS_NAMES.iter().position(|&n| n == name).map(|i| i as u64 * RESOLVE_BUFFER_STRIDE)
}

/// Main-world resource, set once at `Startup`: whether this adapter
/// actually supports `wgpu::Features::TIMESTAMP_QUERY`. Bevy's default
/// `WgpuSettingsPriority::Functionality` already auto-requests every
/// adapter-supported feature (confirmed by reading `bevy_render`'s own
/// device-creation code), and this app doesn't override that priority
/// (`main.rs`'s `DefaultPlugins` setup has no `RenderPlugin`/
/// `WgpuSettings` override at all) -- so this is purely a capability
/// check and a clear log, not a feature *request*.
#[derive(Resource, Clone, Copy)]
pub struct GpuProfileCapability {
    pub supported: bool,
}

/// `Startup` system (main world): logs whether GPU profiling can ever
/// produce real numbers on this adapter, and records the answer for
/// [`update_gpu_profile_active`] to combine with `Config.gpu_profile_enabled`.
pub fn log_gpu_profile_capability(mut commands: Commands, adapter: Res<RenderAdapter>) {
    let supported = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    if supported {
        info!("gpu_profile: this adapter supports wgpu::Features::TIMESTAMP_QUERY");
    } else {
        warn!(
            "gpu_profile: this adapter does NOT support wgpu::Features::TIMESTAMP_QUERY -- \
             GPU profiling (?gpuprofile=1) will report 'unsupported' rather than real timings"
        );
    }
    commands.insert_resource(GpuProfileCapability { supported });
}

/// Main-world resource, recomputed every frame by [`update_gpu_profile_active`]
/// and extracted into the render world -- the one flag [`MarcherTimingNode`]
/// actually gates on, so it doesn't need to separately reach for both
/// `Config` and [`GpuProfileCapability`] itself.
#[derive(Resource, Clone, Copy, Default, ExtractResource)]
pub struct GpuProfileActive {
    pub active: bool,
}

/// `Update` system (main world): combines `Config.gpu_profile_enabled`
/// with the adapter capability check into the one flag the render world
/// actually reads.
pub fn update_gpu_profile_active(
    config: Res<Config>,
    capability: Option<Res<GpuProfileCapability>>,
    mut active: ResMut<GpuProfileActive>,
) {
    active.active = config.gpu_profile_enabled && capability.is_some_and(|c| c.supported);
}

/// Main-world resource: each named pass's last resolved GPU duration, in
/// nanoseconds. `None` until that pass's first successful readback
/// arrives (GPU readback is inherently a frame or more behind).
#[derive(Resource, Clone, Copy, Default)]
pub struct PassTimings {
    pub coarse_pass_ns: Option<i64>,
    pub shadow_pass_ns: Option<i64>,
    pub fine_pass_ns: Option<i64>,
    pub present_pass_ns: Option<i64>,
}

impl PassTimings {
    fn set_by_name(&mut self, name: &str, ns: i64) {
        match name {
            COARSE_PASS_NAME => self.coarse_pass_ns = Some(ns),
            SHADOW_PASS_NAME => self.shadow_pass_ns = Some(ns),
            FINE_PASS_NAME => self.fine_pass_ns = Some(ns),
            PRESENT_PASS_NAME => self.present_pass_ns = Some(ns),
            _ => {}
        }
    }
}

/// `Update` system (main world): reports the current `PassTimings` (plus
/// whether this adapter even supports GPU profiling at all) to the JS
/// overlay (`index.html`) via `net.rs`'s `js_bridge`, once per frame.
/// `-1.0` is the "not yet available" sentinel for a pass whose first
/// readback hasn't landed yet (distinct from `supported == false`, which
/// means it never will). Only runs at all while `gpuprofile` is on --
/// true no-op otherwise, matching every other profiling system here.
pub fn report_pass_timings_to_js(
    config: Res<Config>,
    capability: Option<Res<GpuProfileCapability>>,
    timings: Res<PassTimings>,
) {
    if !config.gpu_profile_enabled {
        return;
    }
    let supported = capability.is_some_and(|c| c.supported);
    let to_ms = |ns: Option<i64>| ns.map(|n| n as f64 / 1_000_000.0).unwrap_or(-1.0);
    crate::net::js_bridge::report_pass_timings(
        supported,
        to_ms(timings.coarse_pass_ns),
        to_ms(timings.shadow_pass_ns),
        to_ms(timings.fine_pass_ns),
        to_ms(timings.present_pass_ns),
    );
}

/// Render-world resource: the timestamp query set + its resolve buffer,
/// created lazily (by [`ensure_gpu_profile_resources`]) the first time
/// they're actually needed -- true no-op allocation-wise whenever
/// profiling is off. Sized for all 4 named passes at once (see module
/// doc's "Four passes sharing one query set").
#[derive(Resource)]
struct GpuProfileResources {
    query_set: wgpu::QuerySet,
    resolve_buffer: Buffer,
}

impl GpuProfileResources {
    fn new(device: &RenderDevice) -> Self {
        let slot_count = (PASS_NAMES.len() * 2) as u32;
        let query_set = device.wgpu_device().create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("gpu_profile_query_set"),
            ty: wgpu::QueryType::Timestamp,
            count: slot_count,
        });
        let resolve_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("gpu_profile_resolve_buffer"),
            // One `RESOLVE_BUFFER_STRIDE`-sized (256-byte-aligned) region
            // per pass, not `slot_count * 8` -- see that constant's doc.
            size: (PASS_NAMES.len() as u64) * RESOLVE_BUFFER_STRIDE,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self { query_set, resolve_buffer }
    }
}

/// A `Mutex`-guarded queue so [`MarcherTimingNode::run`] (which only has
/// shared `&World` access) can still hand freshly-created readback
/// buffers off to [`poll_gpu_profile_readback`] (an ordinary system that
/// runs later the same frame, with normal `Res`/`ResMut` access). Each
/// entry is tagged with which named pass it belongs to, since up to 4
/// views can each queue one per frame.
#[derive(Resource, Default)]
struct PendingReadbackQueue(std::sync::Mutex<Vec<(&'static str, Buffer)>>);

impl PendingReadbackQueue {
    fn push(&self, name: &'static str, buffer: Buffer) {
        self.0.lock().unwrap().push((name, buffer));
    }
}

/// Render-world resource: the channel [`poll_gpu_profile_readback`]'s
/// `map_async` callbacks send a `(pass_name, resolved_raw_tick_duration)`
/// pair through, drained each frame by [`sync_gpu_profile_timings`] in
/// `ExtractSchedule` -- mirrors `gpu_readback.rs`'s own channel-per-
/// readback shape, except this one channel is created once and shared by
/// all 4 passes (each message self-identifies via its pass name) rather
/// than one channel per request.
#[derive(Resource)]
struct GpuProfileReadbackChannel {
    tx: Sender<(&'static str, i64)>,
    rx: Receiver<(&'static str, i64)>,
}

impl Default for GpuProfileReadbackChannel {
    fn default() -> Self {
        let (tx, rx) = async_channel::unbounded();
        Self { tx, rx }
    }
}

/// Replacement for Bevy's stock `MainOpaquePass2dNode`, registered under
/// the same `Node2d::MainOpaquePass` render-graph label (see module doc
/// for why this is safe and how it composes with the existing graph).
/// Behaves identically to stock for every view except one tagged
/// [`GpuProfiledPass`] while [`GpuProfileActive`] is also true.
#[derive(Default)]
struct MarcherTimingNode;

impl ViewNode for MarcherTimingNode {
    type ViewQuery = (
        &'static ExtractedCamera,
        &'static ExtractedView,
        &'static ViewTarget,
        &'static ViewDepthTexture,
        Option<&'static GpuProfiledPass>,
    );

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (camera, view, target, depth, profiled_pass): QueryItem<'w, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let (Some(opaque_phases), Some(alpha_mask_phases)) = (
            world.get_resource::<ViewBinnedRenderPhases<Opaque2d>>(),
            world.get_resource::<ViewBinnedRenderPhases<AlphaMask2d>>(),
        ) else {
            return Ok(());
        };
        let (Some(opaque_phase), Some(alpha_mask_phase)) = (
            opaque_phases.get(&view.retained_view_entity),
            alpha_mask_phases.get(&view.retained_view_entity),
        ) else {
            return Ok(());
        };

        let active = world.get_resource::<GpuProfileActive>().is_some_and(|a| a.active);
        // (name, begin_slot, end_slot, resolve_byte_offset) for this view,
        // only if it's both named and profiling is currently active --
        // `None` for the other 3 passes (all the time) or any pass while
        // profiling is off. The query-set slot pair and the resolve-buffer
        // byte offset are deliberately separate index spaces -- see
        // `RESOLVE_BUFFER_STRIDE`'s doc for why.
        let pass_id: Option<(&'static str, u32, u32, u64)> = active.then_some(profiled_pass).flatten().and_then(|p| {
            let (begin, end) = slot_pair(p.0)?;
            let offset = resolve_byte_offset(p.0)?;
            Some((p.0, begin, end, offset))
        });
        let profile_resources = pass_id.is_some().then(|| world.get_resource::<GpuProfileResources>()).flatten();

        let color_attachments = [Some(target.get_color_attachment())];
        let depth_stencil_attachment = Some(depth.get_attachment(StoreOp::Store));
        let timestamp_writes = match (&profile_resources, pass_id) {
            (Some(res), Some((_, begin, end, _))) => Some(wgpu::RenderPassTimestampWrites {
                query_set: &res.query_set,
                beginning_of_pass_write_index: Some(begin),
                end_of_pass_write_index: Some(end),
            }),
            _ => None,
        };

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("main_opaque_pass_2d"),
            color_attachments: &color_attachments,
            depth_stencil_attachment,
            timestamp_writes,
            occlusion_query_set: None,
        });

        if let Some(viewport) = camera.viewport.as_ref() {
            render_pass.set_camera_viewport(viewport);
        }

        let view_entity = graph.view_entity();
        if !opaque_phase.is_empty() {
            if let Err(err) = opaque_phase.render(&mut render_pass, world, view_entity) {
                error!("Error encountered while rendering the 2d opaque phase {err:?}");
            }
        }
        if !alpha_mask_phase.is_empty() {
            if let Err(err) = alpha_mask_phase.render(&mut render_pass, world, view_entity) {
                error!("Error encountered while rendering the 2d alpha mask phase {err:?}");
            }
        }
        drop(render_pass);

        // Resolve + copy this view's own 2-slot sub-range out to a fresh
        // readback buffer, still within the same shared command encoder
        // used above -- ordering within one encoder's recorded command
        // list is exactly what makes this safe without any cross-view
        // synchronization, and each pass only ever touches its own
        // disjoint byte range of the shared resolve buffer.
        if let (Some(res), Some((name, begin, end, byte_offset))) = (&profile_resources, pass_id) {
            if let (Some(device), Some(pending)) =
                (world.get_resource::<RenderDevice>(), world.get_resource::<PendingReadbackQueue>())
            {
                let readback_buffer = device.create_buffer(&BufferDescriptor {
                    label: Some("gpu_profile_readback_buffer"),
                    size: 16,
                    usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                let encoder = render_context.command_encoder();
                encoder.resolve_query_set(&res.query_set, begin..(end + 1), &res.resolve_buffer, byte_offset);
                encoder.copy_buffer_to_buffer(&res.resolve_buffer, byte_offset, &readback_buffer, 0, 16);
                pending.push(name, readback_buffer);
            }
        }

        Ok(())
    }
}

/// `RenderSet::Prepare` system: ensures [`GpuProfileResources`] exists
/// before [`MarcherTimingNode`] runs this frame, whenever profiling is
/// active -- the lazy lookup itself has to happen here (an ordinary
/// system with `ResMut`/`Commands`), not inside the node's `run()`, since
/// that only gets shared `&World` access.
fn ensure_gpu_profile_resources(
    mut commands: Commands,
    active: Option<Res<GpuProfileActive>>,
    device: Res<RenderDevice>,
    existing: Option<Res<GpuProfileResources>>,
) {
    if active.is_some_and(|a| a.active) && existing.is_none() {
        commands.insert_resource(GpuProfileResources::new(&device));
    }
}

/// `RenderSet::Render`, after `render_system` (mirrors `gpu_readback.rs`'s
/// `map_buffers` ordering exactly): only now, after the command buffer
/// containing this frame's `resolve_query_set`/copy calls has actually
/// been submitted to the queue, register the `map_async` reads -- mapping
/// a buffer any earlier risks racing the copy that fills it.
fn poll_gpu_profile_readback(pending: Res<PendingReadbackQueue>, channel: Res<GpuProfileReadbackChannel>) {
    let buffers: Vec<(&'static str, Buffer)> = std::mem::take(&mut *pending.0.lock().unwrap());
    for (name, buffer) in buffers {
        let tx = channel.tx.clone();
        let buffer_for_callback = buffer.clone();
        buffer.slice(..).map_async(MapMode::Read, move |result| {
            if result.is_err() {
                return;
            }
            let (begin, end) = {
                let data = buffer_for_callback.slice(..).get_mapped_range();
                (
                    i64::from_le_bytes(data[0..8].try_into().unwrap()),
                    i64::from_le_bytes(data[8..16].try_into().unwrap()),
                )
            };
            buffer_for_callback.unmap();
            let duration = end - begin;
            if duration > 0 {
                let _ = tx.try_send((name, duration));
            }
        });
    }
}

/// `ExtractSchedule` system (mirrors `gpu_readback.rs`'s `sync_readbacks`):
/// drains whatever the readback channel has produced since last frame,
/// across all 4 passes, into the main world's [`PassTimings`].
/// `RenderQueue::get_timestamp_period` converts raw GPU ticks to
/// nanoseconds -- required for correctness on native backends
/// (Vulkan/Metal/DX12 ticks aren't always 1ns); the WebGPU spec
/// guarantees timestamps are already in nanoseconds, where this returns
/// `1.0` and the multiply is a no-op.
fn sync_gpu_profile_timings(
    mut main_world: ResMut<MainWorld>,
    channel: Res<GpuProfileReadbackChannel>,
    queue: Res<RenderQueue>,
) {
    let period = queue.get_timestamp_period();
    while let Ok((name, raw_ticks)) = channel.rx.try_recv() {
        let ns = (raw_ticks as f64 * period as f64) as i64;
        if let Some(mut timings) = main_world.get_resource_mut::<PassTimings>() {
            timings.set_by_name(name, ns);
        }
    }
}

pub struct GpuProfilePlugin;

impl Plugin for GpuProfilePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<GpuProfiledPass>::default())
            .add_plugins(ExtractResourcePlugin::<GpuProfileActive>::default())
            .init_resource::<GpuProfileActive>()
            .init_resource::<PassTimings>()
            .add_systems(Startup, log_gpu_profile_capability)
            .add_systems(Update, (update_gpu_profile_active, report_pass_timings_to_js).chain());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<PendingReadbackQueue>()
            .init_resource::<GpuProfileReadbackChannel>()
            .add_systems(ExtractSchedule, sync_gpu_profile_timings)
            .add_systems(
                Render,
                (
                    ensure_gpu_profile_resources.in_set(RenderSet::Prepare),
                    poll_gpu_profile_readback.in_set(RenderSet::Render).after(render_system),
                ),
            )
            .add_render_graph_node::<ViewNodeRunner<MarcherTimingNode>>(Core2d, Node2d::MainOpaquePass);
    }
}
