//! GPU timestamp-query profiling (Stage 3 proof of concept -- the "fine"
//! marcher pass only; the other 3 named passes are a later stage, not this
//! one).
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
//! silently breakable on a future `cargo update`), [`FineTimingNode`]
//! below is a from-scratch replacement for that stock node, registered
//! under the *same* `Node2d::MainOpaquePass` label -- `RenderGraph::
//! add_node` is a plain `HashMap` insert (confirmed by reading
//! `bevy_render`'s source directly), so re-registering under an
//! already-used label silently replaces the stock node rather than
//! erroring, while every existing graph edge (which references the label,
//! not the concrete node type) keeps working unchanged. Since `Camera2d`'s
//! own `#[require(...)]` pins every 2D camera to the `Core2d` sub-graph,
//! this one replacement node's `run()` gets invoked once per camera/view
//! per frame (all 4 of this app's passes go through it) -- it behaves
//! exactly like stock for every view except the one tagged
//! [`GpuProfiledPass`], where it also attaches real timestamp writes.
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
//! ## Readback
//!
//! Mirrors `bevy_render::gpu_readback::GpuReadbackPlugin`'s own shape
//! (confirmed by reading its source): a plain `async_channel` (not a Bevy
//! async task -- works identically on native and wasm), `map_async`
//! registered from a system that runs *after* `render_system` has actually
//! submitted the command buffer containing the corresponding
//! `resolve_query_set`/copy (same ordering `gpu_readback.rs`'s own
//! `map_buffers` system uses), and the resolved `[begin_ns, end_ns]` pair
//! drained into the main world's [`PassTimings`] resource via a system in
//! `ExtractSchedule` (which has `ResMut<MainWorld>` access despite running
//! on the `RenderApp`, same as `gpu_readback.rs`'s `sync_readbacks`).
//! Deliberately no buffer pool yet at this proof-of-concept stage (a
//! single 16-byte buffer allocated and dropped per frame is not worth the
//! complexity here); add one, mirroring `gpu_readback.rs`'s own pool, if
//! generalizing to all 4 passes in a later stage makes it worth it.

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
/// timestamp queries when profiling is active. Attached to the fine
/// camera in `render.rs`'s `setup` -- the other 3 named passes are a
/// later stage.
#[derive(Component, Clone, Copy, ExtractComponent)]
// Unread for now: with only one profiled pass (Stage 3), `FineTimingNode`
// only ever needs `Option::is_some()` on this, not the name itself --
// becomes load-bearing (choosing which query-set slot pair a view gets)
// once a later stage generalizes to all 4 named passes.
#[allow(dead_code)]
pub struct GpuProfiledPass(pub &'static str);

pub const FINE_PASS_NAME: &str = "fine";

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
/// and extracted into the render world -- the one flag [`FineTimingNode`]
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

/// Main-world resource: the fine pass's last resolved GPU duration, in
/// nanoseconds. `None` until the first successful readback arrives (GPU
/// readback is inherently a frame or more behind).
#[derive(Resource, Clone, Copy, Default)]
pub struct PassTimings {
    pub fine_pass_ns: Option<i64>,
}

/// `Update` system (main world): temporary Stage-3 verification signal --
/// logs the fine pass's GPU duration whenever it changes, so this can be
/// confirmed live via the browser console without waiting for the real
/// overlay (a later stage). Safe to remove once Stage 5 lands.
pub fn log_pass_timings_on_change(timings: Res<PassTimings>) {
    if timings.is_changed() {
        if let Some(ns) = timings.fine_pass_ns {
            info!("gpu_profile: fine pass = {:.3} ms", ns as f64 / 1_000_000.0);
        }
    }
}

/// Render-world resource: the timestamp query set + its resolve buffer,
/// created lazily (by [`ensure_gpu_profile_resources`]) the first time
/// they're actually needed -- true no-op allocation-wise whenever
/// profiling is off.
#[derive(Resource)]
struct GpuProfileResources {
    query_set: wgpu::QuerySet,
    resolve_buffer: Buffer,
}

impl GpuProfileResources {
    fn new(device: &RenderDevice) -> Self {
        let query_set = device.wgpu_device().create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("gpu_profile_query_set"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("gpu_profile_resolve_buffer"),
            size: 2 * 8,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self { query_set, resolve_buffer }
    }
}

/// A `Mutex`-guarded queue so [`FineTimingNode::run`] (which only has
/// shared `&World` access) can still hand a freshly-created readback
/// buffer off to [`poll_gpu_profile_readback`] (an ordinary system that
/// runs later the same frame, with normal `Res`/`ResMut` access).
#[derive(Resource, Default)]
struct PendingReadbackQueue(std::sync::Mutex<Vec<Buffer>>);

impl PendingReadbackQueue {
    fn push(&self, buffer: Buffer) {
        self.0.lock().unwrap().push(buffer);
    }
}

/// Render-world resource: the channel [`poll_gpu_profile_readback`]'s
/// `map_async` callback sends a resolved raw tick duration through,
/// drained each frame by [`sync_gpu_profile_timings`] in `ExtractSchedule`
/// -- mirrors `gpu_readback.rs`'s own channel-per-readback shape, except
/// this channel is created once and reused every frame rather than
/// per-request, since there's only ever one in-flight fine-pass readback
/// at a time at this proof-of-concept stage.
#[derive(Resource)]
struct GpuProfileReadbackChannel {
    tx: Sender<i64>,
    rx: Receiver<i64>,
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
struct FineTimingNode;

impl ViewNode for FineTimingNode {
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
        let profile_resources =
            (active && profiled_pass.is_some()).then(|| world.get_resource::<GpuProfileResources>()).flatten();

        let color_attachments = [Some(target.get_color_attachment())];
        let depth_stencil_attachment = Some(depth.get_attachment(StoreOp::Store));
        let timestamp_writes = profile_resources.map(|res| wgpu::RenderPassTimestampWrites {
            query_set: &res.query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        });

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

        // Resolve + copy this view's own timestamp pair out to a fresh
        // readback buffer, still within the same shared command encoder
        // used above -- ordering within one encoder's recorded command
        // list is exactly what makes this safe without any cross-view
        // synchronization.
        if let Some(res) = profile_resources {
            if let (Some(device), Some(pending)) =
                (world.get_resource::<RenderDevice>(), world.get_resource::<PendingReadbackQueue>())
            {
                let readback_buffer = device.create_buffer(&BufferDescriptor {
                    label: Some("gpu_profile_readback_buffer"),
                    size: 2 * 8,
                    usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                let encoder = render_context.command_encoder();
                encoder.resolve_query_set(&res.query_set, 0..2, &res.resolve_buffer, 0);
                encoder.copy_buffer_to_buffer(&res.resolve_buffer, 0, &readback_buffer, 0, 16);
                pending.push(readback_buffer);
            }
        }

        Ok(())
    }
}

/// `RenderSet::Prepare` system: ensures [`GpuProfileResources`] exists
/// before [`FineTimingNode`] runs this frame, whenever profiling is
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
/// containing this frame's `resolve_query_set`/copy has actually been
/// submitted to the queue, register the `map_async` read -- mapping a
/// buffer any earlier risks racing the copy that fills it.
fn poll_gpu_profile_readback(pending: Res<PendingReadbackQueue>, channel: Res<GpuProfileReadbackChannel>) {
    let buffers: Vec<Buffer> = std::mem::take(&mut *pending.0.lock().unwrap());
    for buffer in buffers {
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
                let _ = tx.try_send(duration);
            }
        });
    }
}

/// `ExtractSchedule` system (mirrors `gpu_readback.rs`'s `sync_readbacks`):
/// drains whatever the readback channel has produced since last frame
/// into the main world's [`PassTimings`]. `RenderQueue::get_timestamp_period`
/// converts raw GPU ticks to nanoseconds -- required for correctness on
/// native backends (Vulkan/Metal/DX12 ticks aren't always 1ns); the
/// WebGPU spec guarantees timestamps are already in nanoseconds, where
/// this returns `1.0` and the multiply is a no-op.
fn sync_gpu_profile_timings(mut main_world: ResMut<MainWorld>, channel: Res<GpuProfileReadbackChannel>, queue: Res<RenderQueue>) {
    let period = queue.get_timestamp_period();
    while let Ok(raw_ticks) = channel.rx.try_recv() {
        let ns = (raw_ticks as f64 * period as f64) as i64;
        if let Some(mut timings) = main_world.get_resource_mut::<PassTimings>() {
            timings.fine_pass_ns = Some(ns);
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
            .add_systems(Update, (update_gpu_profile_active, log_pass_timings_on_change).chain());

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
            .add_render_graph_node::<ViewNodeRunner<FineTimingNode>>(Core2d, Node2d::MainOpaquePass);
    }
}
