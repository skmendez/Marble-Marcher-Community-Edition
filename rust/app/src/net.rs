//! Multiplayer milestone 2: real cross-machine transport for
//! [`marble_rollback::InputTransport`], over WebRTC via PeerJS.
//!
//! All the actual `Peer`/`DataConnection` object/callback handling lives in
//! `rust/web/net.js` (loaded before the wasm module — see `index.html`),
//! not here. This module only ever calls net.js's narrow, poll-based API
//! (`js_bridge`) and packs/unpacks a fixed-size 32-byte wire record per
//! [`PlayerInput`] — deliberately not modeling PeerJS's JS class hierarchy
//! from Rust at all (no `Closure`s, no JS object method calls beyond the
//! handful of free functions in `js_bridge`), since getting WebRTC's
//! callback lifetimes right from `wasm-bindgen` is exactly the kind of
//! thing worth keeping in plain JS where it's simpler to write and read.
//!
//! **Up to 4 players, host-relay topology**: the host is always player
//! index `0`; each joiner gets the next sequential index (`1`, `2`, `3`) in
//! connection order, capped at `MAX_JOINERS` additional players (matches
//! `codegen.rs`'s per-marble tint, which already wraps at 4 colors — a
//! natural, already-hinted-at scope boundary). PeerJS gives every joiner a
//! direct connection to the host for free (a star topology), but never a
//! direct joiner-to-joiner connection — so the host relays every `Input`
//! message it receives from one joiner to every *other* connected joiner
//! (`WebRtcTransport::sort_message`), while a joiner only ever talks to the
//! host directly, exactly as before. This matches this codebase's existing
//! host-authoritative convention throughout (`Role`'s doc, resync always
//! flows from the host) rather than a full mesh.
//!
//! **Networking is strictly additive**: nothing here changes single-player
//! behavior. A session always starts hosting in the background (creating a
//! `Peer` costs nothing if nobody ever uses the resulting link), but
//! `physics_sys.rs` only switches from its existing direct `step_marbles`
//! call to a [`RollbackSim`](marble_rollback::RollbackSim)-driven one
//! once a real `DataConnection` actually opens — never just because a
//! `Peer` exists.

use bevy::prelude::*;

use marble_csg::physics::{Marble, PlayerInput};
use marble_rollback::{InputTransport, PlayerIndex, Tick};

use crate::web_config::query_param;

/// Additional players beyond the host — matches `net.js`'s own
/// `MAX_JOINERS` (kept in sync by convention, not a shared constant,
/// since one lives in Rust and the other in plain JS): a connection
/// beyond this cap is closed immediately on the JS side, so Rust never
/// even hears about it.
pub const MAX_JOINERS: usize = 3;

/// Wire size of one packed [`PlayerInput`] record: `tick: u64` (8) +
/// `dx`/`dy: f32` (4 each) + `orientation: Quat` as 4×`f32` (16) = 32
/// bytes. Fixed-size, and always immediately preceded by [`TAG_INPUT`] on
/// the wire (see [`NetMessage`]) — this constant is just this one
/// variant's own body length, not the whole message's.
const RECORD_LEN: usize = 32;

fn pack(tick: Tick, input: PlayerInput) -> [u8; RECORD_LEN] {
    let mut out = [0u8; RECORD_LEN];
    out[0..8].copy_from_slice(&tick.to_le_bytes());
    out[8..12].copy_from_slice(&input.dx.to_le_bytes());
    out[12..16].copy_from_slice(&input.dy.to_le_bytes());
    let q = input.orientation;
    out[16..20].copy_from_slice(&q.x.to_le_bytes());
    out[20..24].copy_from_slice(&q.y.to_le_bytes());
    out[24..28].copy_from_slice(&q.z.to_le_bytes());
    out[28..32].copy_from_slice(&q.w.to_le_bytes());
    out
}

fn unpack(bytes: &[u8]) -> (Tick, PlayerInput) {
    debug_assert_eq!(bytes.len(), RECORD_LEN);
    let tick = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    let dx = f32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let dy = f32::from_le_bytes(bytes[12..16].try_into().unwrap());
    let x = f32::from_le_bytes(bytes[16..20].try_into().unwrap());
    let y = f32::from_le_bytes(bytes[20..24].try_into().unwrap());
    let z = f32::from_le_bytes(bytes[24..28].try_into().unwrap());
    let w = f32::from_le_bytes(bytes[28..32].try_into().unwrap());
    (tick, PlayerInput { dx, dy, orientation: Quat::from_xyzw(x, y, z, w) })
}

/// One packed [`Marble`] record inside a [`NetMessage::ResyncPayload`]:
/// `pos`/`vel` (3×`f32` each) + `rad` (`f32`) = 28 bytes. `last_thrust` is
/// deliberately not carried over the wire — `step_marbles` overwrites it
/// unconditionally on the very next tick regardless of what it held
/// before (`physics.rs`'s own field doc: it's `v` before that tick's
/// force-scale multiply, always freshly computed), so it holds no state a
/// resync actually needs to restore; the receiving side just zeroes it,
/// same as [`Marble::spawn`] already does.
const MARBLE_LEN: usize = 12 + 12 + 4;

fn pack_marble(m: &Marble) -> [u8; MARBLE_LEN] {
    let mut out = [0u8; MARBLE_LEN];
    out[0..4].copy_from_slice(&m.pos.x.to_le_bytes());
    out[4..8].copy_from_slice(&m.pos.y.to_le_bytes());
    out[8..12].copy_from_slice(&m.pos.z.to_le_bytes());
    out[12..16].copy_from_slice(&m.vel.x.to_le_bytes());
    out[16..20].copy_from_slice(&m.vel.y.to_le_bytes());
    out[20..24].copy_from_slice(&m.vel.z.to_le_bytes());
    out[24..28].copy_from_slice(&m.rad.to_le_bytes());
    out
}

fn unpack_marble(bytes: &[u8]) -> Marble {
    debug_assert_eq!(bytes.len(), MARBLE_LEN);
    let f = |lo: usize, hi: usize| f32::from_le_bytes(bytes[lo..hi].try_into().unwrap());
    Marble {
        pos: Vec3::new(f(0, 4), f(4, 8), f(8, 12)),
        vel: Vec3::new(f(12, 16), f(16, 20), f(20, 24)),
        rad: f(24, 28),
        last_thrust: Vec3::ZERO,
    }
}

const TAG_INPUT: u8 = 0;
const TAG_CHECKSUM: u8 = 1;
const TAG_RESYNC_REQUEST: u8 = 2;
const TAG_RESYNC_PAYLOAD: u8 = 3;
const TAG_SCENE_SYNC: u8 = 4;
const TAG_WELCOME: u8 = 5;

/// One message on the wire, tag-prefixed so [`WebRtcTransport`] can share
/// a single channel between the original per-tick input exchange and the
/// later-added checksum/resync-on-mismatch flow (rollback resiliency).
/// `Input` is exactly [`pack`]/[`unpack`]'s existing 32-byte record with a
/// 1-byte tag in front; the other three variants are new.
enum NetMessage {
    Input {
        tick: Tick,
        input: PlayerInput,
    },
    /// A peer's checksum of its own state at `tick` — only ever sent for
    /// a tick outside that peer's own rewind window
    /// (`RollbackSim::latest_checksum_tick`'s doc on why an in-window tick
    /// would be a false positive).
    Checksum {
        tick: Tick,
        hash: u64,
    },
    /// Sent by the non-host side on a detected checksum mismatch — "send
    /// me your authoritative state as of `tick`". Host-authoritative
    /// throughout (`Role`'s doc): only ever sent host-ward, never the
    /// reverse.
    ResyncRequest {
        tick: Tick,
    },
    /// The host's answer to a [`Self::ResyncRequest`]: its own live state.
    /// The one genuinely variable-length message on the wire —
    /// length-prefixed by `marbles.len()` as a `u32`.
    ResyncPayload {
        tick: Tick,
        marbles: Vec<Marble>,
    },
    /// The host's whole CSG scene tree + params + animation table
    /// (`marble_csg::Scene::to_bytes`), sent once at
    /// connect (join or reconnect) alongside `ResyncPayload` so the joiner
    /// renders/simulates against the host's actual scene instead of
    /// guessing one from its own `?scene=` (`physics_sys.rs`'s doc on the
    /// join flow). Opaque here — `net.rs` only ever carries these bytes,
    /// it doesn't decode them (`Scene` is `marble_csg`'s concern, not
    /// the transport's), length-prefixed as a `u32` same as
    /// `ResyncPayload`'s marbles.
    SceneSync {
        bytes: Vec<u8>,
    },
    /// Host -> exactly one newly-connected joiner: "you are player index
    /// K" (`net.js` assigns joiner indices sequentially at connect time,
    /// but only the host's side of the connection knows the assignment —
    /// this is how the joiner learns it). Sent once per connection,
    /// always before that same connection's first `ResyncPayload`/
    /// `SceneSync` (PeerJS's `reliable: true` channel preserves send
    /// order), never relayed to other joiners.
    Welcome {
        index: PlayerIndex,
    },
}

fn encode_message(msg: &NetMessage) -> Vec<u8> {
    match msg {
        NetMessage::Input { tick, input } => {
            let mut out = Vec::with_capacity(1 + RECORD_LEN);
            out.push(TAG_INPUT);
            out.extend_from_slice(&pack(*tick, *input));
            out
        }
        NetMessage::Checksum { tick, hash } => {
            let mut out = Vec::with_capacity(1 + 8 + 8);
            out.push(TAG_CHECKSUM);
            out.extend_from_slice(&tick.to_le_bytes());
            out.extend_from_slice(&hash.to_le_bytes());
            out
        }
        NetMessage::ResyncRequest { tick } => {
            let mut out = Vec::with_capacity(1 + 8);
            out.push(TAG_RESYNC_REQUEST);
            out.extend_from_slice(&tick.to_le_bytes());
            out
        }
        NetMessage::ResyncPayload { tick, marbles } => {
            let mut out = Vec::with_capacity(1 + 8 + 4 + marbles.len() * MARBLE_LEN);
            out.push(TAG_RESYNC_PAYLOAD);
            out.extend_from_slice(&tick.to_le_bytes());
            out.extend_from_slice(&(marbles.len() as u32).to_le_bytes());
            for m in marbles {
                out.extend_from_slice(&pack_marble(m));
            }
            out
        }
        NetMessage::SceneSync { bytes } => {
            let mut out = Vec::with_capacity(1 + 4 + bytes.len());
            out.push(TAG_SCENE_SYNC);
            out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(bytes);
            out
        }
        NetMessage::Welcome { index } => {
            vec![TAG_WELCOME, *index as u8]
        }
    }
}

/// Decodes every complete message packed back-to-back in `bytes` — exactly
/// what `net.js`'s `takeReceived` hands back: zero or more whole `send()`
/// calls' worth of bytes, concatenated in arrival order with no gaps
/// (module doc). Each variant's own encoding carries enough information (a
/// fixed size, or a length prefix for `ResyncPayload`) to know exactly
/// where it ends, so no *additional* outer framing is needed even though
/// messages now vary in length between kinds.
///
/// Every body read goes through `bytes.get(range)` rather than direct
/// index/slice syntax, and a `None` (not enough bytes remaining, or — for
/// `ResyncPayload`'s attacker/bug-controlled `count` — an overflowing
/// `count * MARBLE_LEN`) stops decoding immediately and returns whatever
/// full messages were already parsed. A truncated or malformed tail on an
/// otherwise-trusted peer's stream is still possible (a dropped byte, a
/// build mismatch, or genuinely hostile input on this P2P WebRTC channel —
/// nothing here assumes the remote is running the exact same build) and
/// this is the difference between silently losing the tail of one
/// `takeReceived` batch and a hard `panic = "abort"` process crash on the
/// very next `bytes[i]`.
fn decode_messages(bytes: &[u8]) -> Vec<NetMessage> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let tag = bytes[i];
        i += 1;
        match tag {
            TAG_INPUT => {
                let Some(body) = bytes.get(i..i + RECORD_LEN) else { break };
                let (tick, input) = unpack(body);
                i += RECORD_LEN;
                out.push(NetMessage::Input { tick, input });
            }
            TAG_CHECKSUM => {
                let Some(body) = bytes.get(i..i + 16) else { break };
                let tick = u64::from_le_bytes(body[0..8].try_into().unwrap());
                let hash = u64::from_le_bytes(body[8..16].try_into().unwrap());
                i += 16;
                out.push(NetMessage::Checksum { tick, hash });
            }
            TAG_RESYNC_REQUEST => {
                let Some(body) = bytes.get(i..i + 8) else { break };
                let tick = u64::from_le_bytes(body.try_into().unwrap());
                i += 8;
                out.push(NetMessage::ResyncRequest { tick });
            }
            TAG_RESYNC_PAYLOAD => {
                let Some(header) = bytes.get(i..i + 12) else { break };
                let tick = u64::from_le_bytes(header[0..8].try_into().unwrap());
                let count = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;
                i += 12;
                let Some(needed) = count.checked_mul(MARBLE_LEN) else { break };
                let Some(end) = i.checked_add(needed) else { break };
                let Some(body) = bytes.get(i..end) else { break };
                let marbles = body.chunks_exact(MARBLE_LEN).map(unpack_marble).collect();
                i = end;
                out.push(NetMessage::ResyncPayload { tick, marbles });
            }
            TAG_SCENE_SYNC => {
                let Some(header) = bytes.get(i..i + 4) else { break };
                let len = u32::from_le_bytes(header.try_into().unwrap()) as usize;
                i += 4;
                let Some(end) = i.checked_add(len) else { break };
                let Some(body) = bytes.get(i..end) else { break };
                let payload = body.to_vec();
                i = end;
                out.push(NetMessage::SceneSync { bytes: payload });
            }
            TAG_WELCOME => {
                let Some(&b) = bytes.get(i) else { break };
                let index = b as PlayerIndex;
                i += 1;
                out.push(NetMessage::Welcome { index });
            }
            // An unrecognized tag means the stream is malformed -- can't
            // happen in practice (both peers always run the exact same
            // build, so every tag either side ever writes is one of the
            // six above), but stopping rather than guessing how many
            // bytes to skip past garbage is the safe failure mode.
            _ => break,
        }
    }
    out
}

#[cfg(target_arch = "wasm32")]
pub(crate) mod js_bridge {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = mmNet, js_name = host)]
        pub fn host();
        #[wasm_bindgen(js_namespace = mmNet, js_name = join)]
        pub fn join(host_id: &str);
        #[wasm_bindgen(js_namespace = mmNet, js_name = send)]
        pub fn send(bytes: &[u8]);
        #[wasm_bindgen(js_namespace = mmNet, js_name = sendTo)]
        pub fn send_to(idx: u8, bytes: &[u8]);
        #[wasm_bindgen(js_namespace = mmNet, js_name = status)]
        pub fn status() -> i32;
        #[wasm_bindgen(js_namespace = mmNet, js_name = hostedId)]
        pub fn hosted_id() -> String;
        #[wasm_bindgen(js_namespace = mmNet, js_name = takeError)]
        pub fn take_error() -> String;
        #[wasm_bindgen(js_namespace = mmNet, js_name = takeReceived)]
        pub fn take_received() -> Vec<u8>;
        #[wasm_bindgen(js_namespace = mmNet, js_name = takeReceivedFrom)]
        pub fn take_received_from(idx: u8) -> Vec<u8>;
        #[wasm_bindgen(js_namespace = mmNet, js_name = newlyConnectedIndices)]
        pub fn newly_connected_indices() -> Vec<u8>;
        #[wasm_bindgen(js_namespace = mmNet, js_name = newlyDisconnectedIndices)]
        pub fn newly_disconnected_indices() -> Vec<u8>;
        #[wasm_bindgen(js_namespace = mmNet, js_name = copyToClipboard)]
        pub fn copy_to_clipboard(text: &str);
        #[wasm_bindgen(js_namespace = mmNet, js_name = takeClipboardStatus)]
        pub fn take_clipboard_status() -> i32;
        // GPU-timestamp-query profiling (`gpu_profile.rs`) -- reports the
        // last-resolved duration (milliseconds, `-1.0` sentinel for "not
        // yet available") for each of the 4 named passes, plus whether
        // this adapter supports GPU profiling at all (distinct from "no
        // reading yet" -- `supported == false` means it never will).
        // Namespaced under `mmNet` like every other bridge call here, even
        // though this isn't strictly networking -- that's just this app's
        // one established Rust->JS bridge, not a second mechanism.
        #[wasm_bindgen(js_namespace = mmNet, js_name = reportPassTimings)]
        pub fn report_pass_timings(supported: bool, coarse_ms: f64, shadow_ms: f64, fine_ms: f64, present_ms: f64);
        // Cumulative ray-march step-count estimate (`step_data.rs`) --
        // `avg_iters_per_px`/`estimated_total_steps` are `-1.0` sentinels
        // for "no reading yet" (same convention as `report_pass_timings`'s
        // `-1.0`), not "unsupported" -- this estimate has no adapter
        // capability question of its own (it's an ordinary render + texture
        // readback, not a `wgpu::Features::TIMESTAMP_QUERY` dependency), so
        // there's no separate `supported` flag to thread through here.
        #[wasm_bindgen(js_namespace = mmNet, js_name = reportStepData)]
        pub fn report_step_data(avg_iters_per_px: f64, estimated_total_steps: f64);
    }
}

/// Native build stub: networking is a wasm/browser-only concept here (there
/// is no PeerJS/WebRTC on native at all), so every one of these is a
/// no-op/idle value — native always behaves as if it's permanently
/// offline, which is exactly today's (pre-milestone-2) native behavior.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod js_bridge {
    pub fn host() {}
    pub fn join(_host_id: &str) {}
    pub fn send(_bytes: &[u8]) {}
    pub fn send_to(_idx: u8, _bytes: &[u8]) {}
    pub fn status() -> i32 {
        0
    }
    pub fn hosted_id() -> String {
        String::new()
    }
    pub fn take_error() -> String {
        String::new()
    }
    pub fn take_received() -> Vec<u8> {
        Vec::new()
    }
    pub fn take_received_from(_idx: u8) -> Vec<u8> {
        Vec::new()
    }
    pub fn newly_connected_indices() -> Vec<u8> {
        Vec::new()
    }
    pub fn newly_disconnected_indices() -> Vec<u8> {
        Vec::new()
    }
    pub fn copy_to_clipboard(_text: &str) {}
    pub fn take_clipboard_status() -> i32 {
        0
    }
    pub fn report_pass_timings(_supported: bool, _coarse_ms: f64, _shadow_ms: f64, _fine_ms: f64, _present_ms: f64) {}
    pub fn report_step_data(_avg_iters_per_px: f64, _estimated_total_steps: f64) {}
}

/// This client's role in the current (at most 2-player) session, mirroring
/// `net.js`'s `status()` codes (kept as a proper enum on the Rust side
/// rather than passing the raw `i32` around everywhere).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NetStatus {
    /// `host()`/`join()` hasn't been called yet, or `net.js` hasn't run
    /// its first callback yet.
    Idle,
    /// This client is hosting and waiting for a peer to connect.
    Hosting,
    /// A `DataConnection` is open — [`WebRtcTransport`] is live and
    /// `physics_sys.rs` should be driving the session through
    /// `RollbackSim`.
    Connected,
    /// Was connected, isn't anymore (peer left/dropped). `physics_sys.rs`'s
    /// minimum bar (per the plan) is to not crash or corrupt the local
    /// simulation for the remaining player — it does this by simply
    /// continuing to predict the departed player's input forever (the same
    /// prediction path `RollbackSim` already always has), not by trying to
    /// remove them from the marble list (which would break the stable-
    /// index-order guarantee every other part of this system depends on).
    Disconnected,
    Error,
}

impl NetStatus {
    fn from_js(code: i32) -> Self {
        match code {
            1 => Self::Hosting,
            2 => Self::Connected,
            3 => Self::Disconnected,
            4 => Self::Error,
            _ => Self::Idle,
        }
    }
}

/// The host's own player index — always `0`, host-relay topology (module
/// doc).
pub const HOST_INDEX: PlayerIndex = 0;

/// This client's role: host (player `0`, accepting up to [`MAX_JOINERS`]
/// joiners) or joiner (connecting to a host's shared link, sequentially
/// assigned indices `1..=MAX_JOINERS`) — fixed for the whole session from
/// the very first frame (`?join=` present or not), never renegotiated.
///
/// Unlike the old exactly-2-players design, a joiner's own player index is
/// no longer a pure function of `Role` — it's assigned by the host at
/// connect time and learned via [`NetMessage::Welcome`]
/// (`WebRtcTransport::poll_welcomes`), since two different joiners
/// connecting to the same host land at different indices. `physics_sys.rs`
/// tracks the learned value directly on `MarbleState::local_player_index`
/// (defaulting to `0`, correct for the host and a reasonable placeholder
/// for a joiner until its `Welcome` arrives, which happens before any real
/// input exchange starts — `MultiplayerSession`'s doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Host,
    Joiner,
}

/// Whole-session networking state — one per app, inserted at [`Startup`].
#[derive(Resource)]
pub struct NetSession {
    pub role: Role,
    pub status: NetStatus,
    /// The shareable link's query-string suffix (`?join=<id>`), populated
    /// once `net.js`'s `host()` callback reports an assigned id. `None`
    /// for a joiner (they never host) or before hosting's `open` event has
    /// fired yet.
    pub hosted_join_suffix: Option<String>,
    pub last_error: String,
}

impl NetSession {
    /// The full, shareable invite URL, if one exists yet (hosting, with an
    /// id already assigned) -- shared by [`sync_net_ui_text`] (displays it)
    /// and [`handle_copy_button_click`] (copies it), so there's exactly one
    /// place that assembles a link from `page_url_base` + the join suffix.
    fn current_link(&self) -> Option<String> {
        let suffix = self.hosted_join_suffix.as_ref()?;
        Some(page_url_base().map_or_else(|| suffix.clone(), |base| format!("{base}{suffix}")))
    }
}

/// `Startup` system: reads `?join=` (via the same `web_config::query_param`
/// helper the `?scene=` work already built and proved out) to decide this
/// client's [`Role`], then calls `net.js`'s `host()`/`join()` exactly once.
/// Unconditional — every session hosts by default, per the module doc.
pub fn setup_networking(mut commands: Commands) {
    let (role, hosted_join_suffix) = match query_param("join") {
        Some(host_id) => {
            js_bridge::join(&host_id);
            (Role::Joiner, None)
        }
        None => {
            js_bridge::host();
            (Role::Host, None)
        }
    };
    info!("multiplayer: starting as {role:?}");
    commands.insert_resource(NetSession {
        role,
        status: NetStatus::Idle,
        hosted_join_suffix,
        last_error: String::new(),
    });
}

/// `Update` system: polls `net.js`'s current status/hosted-id/error every
/// frame — cheap (a handful of `extern "C"` calls into already-marshalled
/// JS values, no allocation on the happy path) so polling unconditionally
/// is simpler than trying to event-drive this from Rust's side of the
/// boundary.
pub fn poll_net_status(mut session: ResMut<NetSession>) {
    session.status = NetStatus::from_js(js_bridge::status());
    if session.role == Role::Host && session.hosted_join_suffix.is_none() {
        let id = js_bridge::hosted_id();
        if !id.is_empty() {
            // Deliberately just `?join=<id>` -- no `&scene=` hint. The
            // host's actual scene (tree + params + animations) is sent
            // over WebRTC itself (`NetMessage::SceneSync`, applied by
            // `render::apply_pending_scene_sync`), which is the only
            // copy that's ever authoritative; a URL-carried scene name
            // would just be a second, redundant source that could disagree
            // with it (a joiner opening a stale/shared link after the host
            // switched scenes, for instance) with no way to ever be
            // "wrong" about it -- not worth that divergence risk for what
            // was only ever a cosmetic pre-sync guess.
            session.hosted_join_suffix = Some(format!("?join={id}"));
        }
    }
    let err = js_bridge::take_error();
    if !err.is_empty() {
        session.last_error = err;
    }
}

/// The current page's origin + path, so [`sync_net_ui_text`] can show a
/// complete, clickable/copyable URL rather than a bare `?join=<id>`
/// suffix the user would have to know to append themselves. `None` on
/// native (no browser location to read) or if the browser APIs are
/// unavailable for some reason — callers fall back to showing the raw
/// suffix in that case.
#[cfg(target_arch = "wasm32")]
fn page_url_base() -> Option<String> {
    let location = web_sys::window()?.location();
    Some(format!("{}{}", location.origin().ok()?, location.pathname().ok()?))
}

#[cfg(not(target_arch = "wasm32"))]
fn page_url_base() -> Option<String> {
    None
}

/// Marker for the on-screen networking status/invite-link text. `pub(crate)`:
/// it appears in `sync_net_ui_text`'s public query type, which `main.rs` (a
/// different module) needs to be able to name when registering that system
/// — same reasoning as this codebase's other UI marker components
/// (`mrrm.rs`'s `CoarseCamera`/`CoarseQuad`, `touch.rs`'s debug markers).
#[derive(Component)]
pub(crate) struct NetStatusText;

/// Marker for the "Copy Link" button itself (the `Button`-required `Node`,
/// not its label) -- `pub(crate)` for the same reason as [`NetStatusText`].
#[derive(Component)]
pub(crate) struct CopyLinkButton;

/// Marker for the button's `Text` child, so [`handle_copy_button_click`]/
/// [`update_copy_feedback`] can update the label without also touching the
/// button `Node` it's attached to.
#[derive(Component)]
pub(crate) struct CopyLinkButtonLabel;

const COPY_BUTTON_IDLE_TEXT: &str = "Copy Link";

/// How long "Copied!"/"Copy failed" stays on the button before it reverts
/// to [`COPY_BUTTON_IDLE_TEXT`] -- long enough to actually read, short
/// enough that a second click shortly after isn't confusingly stuck on
/// stale feedback from the first.
const COPY_FEEDBACK_SECONDS: f64 = 1.5;

/// Tracks the transient "just clicked" feedback state so
/// [`update_copy_feedback`] knows what the button should say right now and
/// when to revert it back to [`COPY_BUTTON_IDLE_TEXT`] -- separate from
/// [`NetSession`] since this is pure UI-feedback timing, nothing else in
/// the app cares about it. `None` means the button is showing its idle
/// label; `Some((shown_at, message))` means it's showing `message` until
/// `shown_at + COPY_FEEDBACK_SECONDS`.
#[derive(Resource, Default)]
pub(crate) struct CopyFeedback(Option<(f64, String)>);

/// `Startup` system: spawns the always-present networking status readout
/// (top-right, so it doesn't collide with the debug overlays' top-left
/// stack — `fps_overlay.rs`, `touch.rs`). Chained after `setup_networking`
/// (`main.rs`) purely for readability (it doesn't actually read
/// `NetSession` yet — [`sync_net_ui_text`] does that every frame).
pub fn spawn_net_ui(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(6.0),
                right: Val::Px(6.0),
                max_width: Val::Px(420.0),
                padding: UiRect::axes(Val::Px(6.0), Val::Px(3.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.55)),
        ))
        .with_child((
            Text::new("connecting..."),
            TextFont { font_size: 16.0, ..default() },
            TextColor(Color::srgb(0.6, 0.85, 1.0)),
            NetStatusText,
        ));

    // Separate node from the status text above (not a child of it): bevy_ui
    // text is drawn straight to the canvas, not real DOM text, so there is
    // nothing for the browser's normal copy/selection handling to grab --
    // this button is the actual way to get the invite link onto the
    // clipboard. Hidden (`Display::None`) until `update_copy_button_visibility`
    // finds a real link to copy; clicking before then would just copy
    // nothing useful, so there's no point showing it yet.
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(46.0),
                right: Val::Px(6.0),
                padding: UiRect::axes(Val::Px(10.0), Val::Px(4.0)),
                display: Display::None,
                ..default()
            },
            BackgroundColor(Color::srgba(0.2, 0.4, 0.6, 0.85)),
            Button,
            CopyLinkButton,
        ))
        .with_child((
            Text::new(COPY_BUTTON_IDLE_TEXT),
            TextFont { font_size: 16.0, ..default() },
            TextColor(Color::WHITE),
            CopyLinkButtonLabel,
        ));
}

/// `Update` system: shows the copy button only once there's an actual link
/// to copy (hosting, with a broker-assigned id) — a joiner, or a host that
/// hasn't received its id yet, has nothing useful to copy.
pub fn update_copy_button_visibility(session: Res<NetSession>, mut node: Query<&mut Node, With<CopyLinkButton>>) {
    let Ok(mut node) = node.single_mut() else { return };
    let display = if session.current_link().is_some() { Display::Flex } else { Display::None };
    if node.display != display {
        node.display = display;
    }
}

/// `Update` system: the actual copy action. `Interaction` is driven by
/// mouse *and* touch alike (`bevy_ui::focus::ui_focus_system` reads
/// `Res<Touches>` directly, not just `ButtonInput<MouseButton>`), so this
/// needs no separate touch-handling path — the whole point of this button,
/// since this app targets phones as much as desktop.
///
/// Calls `copy_to_clipboard` synchronously from inside this click-triggered
/// system, in the same frame the `Interaction::Pressed` transition is
/// observed — the browser Clipboard API only grants permission when
/// invoked synchronously within a real user-gesture callstack, so this
/// can't be deferred (e.g. queued and handled a frame later) without
/// silently losing that permission.
pub fn handle_copy_button_click(
    session: Res<NetSession>,
    mut feedback: ResMut<CopyFeedback>,
    time: Res<Time>,
    interactions: Query<&Interaction, (Changed<Interaction>, With<CopyLinkButton>)>,
) {
    let Ok(interaction) = interactions.single() else { return };
    if *interaction != Interaction::Pressed {
        return;
    }
    let Some(link) = session.current_link() else { return };
    js_bridge::copy_to_clipboard(&link);
    // Immediate feedback that the click registered, ahead of the async
    // write actually resolving (typically near-instant, but the click and
    // its result land on different frames regardless — see
    // `update_copy_feedback`, which overwrites this once the real result
    // arrives).
    feedback.0 = Some((time.elapsed_secs_f64(), "Copying...".to_string()));
}

/// `Update` system: polls the async clipboard-write result
/// (`takeClipboardStatus`) and drives the button label between its idle
/// text and "Copied!"/"Copy failed" feedback, reverting automatically
/// after `COPY_FEEDBACK_SECONDS` — separate from `handle_copy_button_click`
/// since the result arrives on a later frame than the click that triggered
/// it (the clipboard write is async even though *invoking* it must be
/// synchronous — see that system's doc).
pub fn update_copy_feedback(
    time: Res<Time>,
    mut feedback: ResMut<CopyFeedback>,
    mut label: Query<&mut Text, With<CopyLinkButtonLabel>>,
) {
    let now = time.elapsed_secs_f64();
    match js_bridge::take_clipboard_status() {
        1 => feedback.0 = Some((now, "Copied!".to_string())),
        2 => feedback.0 = Some((now, "Copy failed".to_string())),
        _ => {}
    }

    let Ok(mut label) = label.single_mut() else { return };
    match &feedback.0 {
        Some((shown_at, message)) if now - shown_at < COPY_FEEDBACK_SECONDS => {
            label.0 = message.clone();
        }
        Some(_) => {
            feedback.0 = None;
            label.0 = COPY_BUTTON_IDLE_TEXT.to_string();
        }
        None => {}
    }
}

/// `Update` system: renders [`NetSession`]'s current state into
/// [`NetStatusText`] — the one place a human (not just `net.rs` itself)
/// finds out there's an invite link to share, or that a peer connected.
///
/// Compares before assigning: writing through `Text`'s `DerefMut` marks the
/// component `Changed` unconditionally, regardless of whether the new value
/// actually differs (Bevy's change detection triggers on the mutable
/// access itself, not a value-equality check) — this system runs every
/// frame, and `session.status` sits at the same value (`Idle`, i.e.
/// "connecting...") for the entire time a peer hasn't connected, so an
/// unconditional write here was re-triggering bevy_text/bevy_ui's glyph
/// reshaping and GPU bind-group upload every single frame for no reason.
/// Confirmed directly this session: with the marcher's own (unrelated,
/// separately fixed) GPU buffer leak eliminated and `as_bind_group` on all
/// three marcher materials verified to fire zero times after startup, this
/// was the actual remaining source of ongoing `createBindGroup` growth
/// (measured in production, present even with the debug overlay disabled
/// — this UI panel is always visible, unlike that overlay).
pub fn sync_net_ui_text(session: Res<NetSession>, mut text: Query<&mut Text, With<NetStatusText>>) {
    let Ok(mut text) = text.single_mut() else { return };
    let new_text = match session.status {
        NetStatus::Idle => "connecting...".to_string(),
        NetStatus::Hosting => match session.current_link() {
            Some(link) => format!("Share this link to play together:\n{link}"),
            None => "starting host...".to_string(),
        },
        NetStatus::Connected => "Player connected!".to_string(),
        NetStatus::Disconnected => "Other player disconnected — still playing solo.".to_string(),
        NetStatus::Error => format!("Network error: {}", session.last_error),
    };
    if text.0 != new_text {
        text.0 = new_text;
    }
}

/// [`InputTransport`] over the live WebRTC data channel(s), extended to
/// also carry the checksum/resync side channel (rollback resiliency) and
/// the host-relay fan-out (module doc) — see the module doc for the wire
/// format.
///
/// A joiner has exactly one remote (the host, implicitly [`HOST_INDEX`])
/// and talks to it via `net.js`'s single-connection `send`/`takeReceived`.
/// A host has up to [`MAX_JOINERS`] remotes, tracked in `known_remotes`
/// (grows via [`Self::take_newly_connected`], never shrinks — a departed
/// joiner's index is never reused, same stable-index-order guarantee
/// `MultiplayerSession` already relies on) and talks to them via
/// `net.js`'s per-index `sendTo`/`takeReceivedFrom`.
pub struct WebRtcTransport {
    role: Role,
    /// Host only: every joiner index that has ever connected. Empty and
    /// unused for a joiner.
    known_remotes: Vec<PlayerIndex>,
    /// Demultiplexed per-kind inboxes, filled by [`Self::drain_and_demux`]
    /// and drained by each kind's own `poll_*` method — separate from
    /// `net.js`'s own receive queue(s) so that `poll_received` (the
    /// [`InputTransport`] trait method) and the other `poll_*` methods
    /// below can each be called independently, in any order, any number of
    /// times per tick, without one silently stealing messages meant for
    /// another.
    pending_inputs: Vec<(PlayerIndex, Tick, PlayerInput)>,
    /// Origin-less: a joiner's only source is unambiguously the host; a
    /// host never *acts* on a checksum comparison regardless of which
    /// joiner it came from (only the non-host side ever requests a
    /// correction, `net::Role`'s host-authoritative convention), so
    /// there's nothing origin-specific to do with this at either role.
    pending_checksums: Vec<(Tick, u64)>,
    /// Origin-less for the same reason: a host answers *any* resync
    /// request by broadcasting its current state to every connected
    /// joiner (not just the one that asked) — simpler and just as correct
    /// as a targeted answer, since an authoritative resync is never wrong
    /// for a peer that didn't actually need it, and this avoids needing
    /// per-joiner request tracking for what should be a rare event.
    pending_resync_requests: Vec<Tick>,
    /// Joiner-only: always from the host.
    pending_resync_payloads: Vec<(Tick, Vec<Marble>)>,
    /// Joiner-only: always from the host.
    pending_scene_syncs: Vec<Vec<u8>>,
    /// Joiner-only: "you are player index K" from the host, once per
    /// connection.
    pending_welcomes: Vec<PlayerIndex>,
}

impl WebRtcTransport {
    pub fn new(role: Role) -> Self {
        Self {
            role,
            known_remotes: Vec::new(),
            pending_inputs: Vec::new(),
            pending_checksums: Vec::new(),
            pending_resync_requests: Vec::new(),
            pending_resync_payloads: Vec::new(),
            pending_scene_syncs: Vec::new(),
            pending_welcomes: Vec::new(),
        }
    }

    /// Host-only: player indices whose connection opened since the last
    /// call — also folds them into `known_remotes` so `Self::broadcast`/
    /// `Self::relay_input_except` immediately include them. A no-op,
    /// always-empty poll for a joiner (`js_bridge::newly_connected_indices`
    /// is host-only in `net.js` too).
    pub fn take_newly_connected(&mut self) -> Vec<PlayerIndex> {
        if self.role != Role::Host {
            return Vec::new();
        }
        let indices: Vec<PlayerIndex> =
            js_bridge::newly_connected_indices().into_iter().map(|b| b as PlayerIndex).collect();
        for &idx in &indices {
            debug_assert!(
                (1..=MAX_JOINERS).contains(&idx),
                "net.js should never report a joiner index outside 1..=MAX_JOINERS (got {idx})"
            );
            if !self.known_remotes.contains(&idx) {
                self.known_remotes.push(idx);
            }
        }
        indices
    }

    /// Host-only: player indices whose connection closed since the last
    /// call. Stays in `known_remotes` regardless (a send to a closed
    /// connection is already a harmless no-op on the JS side) — this is
    /// purely for `physics_sys.rs` to know which indices need a confirmed
    /// zero input from here on (`MultiplayerSession`'s doc).
    pub fn take_newly_disconnected(&mut self) -> Vec<PlayerIndex> {
        if self.role != Role::Host {
            return Vec::new();
        }
        js_bridge::newly_disconnected_indices().into_iter().map(|b| b as PlayerIndex).collect()
    }

    /// Sends `bytes` to every currently-known remote (a joiner has
    /// exactly one, the host; a host sends to every joiner that has ever
    /// connected). Used for this side's *own* outgoing messages (its own
    /// input, its own checksum, an authoritative resync/scene push) —
    /// distinct from [`Self::relay_input_except`], which forwards a
    /// *received* message on to others.
    fn broadcast(&self, bytes: &[u8]) {
        match self.role {
            Role::Joiner => js_bridge::send(bytes),
            Role::Host => {
                for &idx in &self.known_remotes {
                    js_bridge::send_to(idx as u8, bytes);
                }
            }
        }
    }

    /// Host-only: forwards one joiner's `Input` message on to every
    /// *other* connected joiner, so every peer's `RollbackSim` ends up with
    /// every player's input each tick (module doc) — PeerJS gives a star
    /// topology, joiners never connect to each other directly, so this
    /// relay is the only way an input reaches anyone but the host.
    fn relay_input_except(&self, origin: PlayerIndex, bytes: &[u8]) {
        for &idx in &self.known_remotes {
            if idx != origin {
                js_bridge::send_to(idx as u8, bytes);
            }
        }
    }

    /// Sends `Welcome { index }` directly to exactly one connection (the
    /// joiner that just connected as that index) — never broadcast, since
    /// every other connection already knows its own index.
    pub fn send_welcome(&self, to: PlayerIndex, index: PlayerIndex) {
        js_bridge::send_to(to as u8, &encode_message(&NetMessage::Welcome { index }));
    }

    /// Drains whatever `net.js` has received since the last poll (of
    /// *any* kind), from every known channel, and sorts each decoded
    /// message into its own buffer — the one place that actually calls
    /// `js_bridge::take_received`/`take_received_from`, so every `poll_*`
    /// method (including [`InputTransport::poll_received`]) can call this
    /// unconditionally: whichever runs first each tick does the real
    /// draining, the rest just find the queue(s) already empty.
    fn drain_and_demux(&mut self) {
        match self.role {
            Role::Joiner => {
                let bytes = js_bridge::take_received();
                for msg in decode_messages(&bytes) {
                    self.sort_message(HOST_INDEX, msg);
                }
            }
            Role::Host => {
                for idx in self.known_remotes.clone() {
                    let bytes = js_bridge::take_received_from(idx as u8);
                    for msg in decode_messages(&bytes) {
                        self.sort_message(idx, msg);
                    }
                }
            }
        }
    }

    /// Files one decoded message into its kind's buffer, tagged with
    /// `origin` where that's meaningful ([`Self::pending_inputs`] only —
    /// see the other `pending_*` fields' docs for why they don't need it).
    /// Relays a host-received `Input` to every other joiner inline, here,
    /// so relay happens exactly once per real message received, not as a
    /// separate pass.
    fn sort_message(&mut self, origin: PlayerIndex, msg: NetMessage) {
        if self.role == Role::Host {
            if let NetMessage::Input { .. } = &msg {
                let bytes = encode_message(&msg);
                self.relay_input_except(origin, &bytes);
            }
        }
        match msg {
            NetMessage::Input { tick, input } => self.pending_inputs.push((origin, tick, input)),
            NetMessage::Checksum { tick, hash } => self.pending_checksums.push((tick, hash)),
            NetMessage::ResyncRequest { tick } => self.pending_resync_requests.push(tick),
            NetMessage::ResyncPayload { tick, marbles } => self.pending_resync_payloads.push((tick, marbles)),
            NetMessage::SceneSync { bytes } => self.pending_scene_syncs.push(bytes),
            NetMessage::Welcome { index } => self.pending_welcomes.push(index),
        }
    }

    pub fn send_scene_sync(&mut self, bytes: Vec<u8>) {
        self.broadcast(&encode_message(&NetMessage::SceneSync { bytes }));
    }

    /// Drains every scene-sync bundle the peer has sent since the last call
    /// (only ever populated on the non-host side, same as
    /// [`Self::poll_resync_payloads`] -- host-authoritative throughout).
    pub fn poll_scene_syncs(&mut self) -> Vec<Vec<u8>> {
        self.drain_and_demux();
        std::mem::take(&mut self.pending_scene_syncs)
    }

    pub fn send_checksum(&mut self, tick: Tick, hash: u64) {
        self.broadcast(&encode_message(&NetMessage::Checksum { tick, hash }));
    }

    pub fn send_resync_request(&mut self, tick: Tick) {
        // Only ever called when `role == Role::Joiner` (its one caller in
        // `physics_sys.rs` is gated on that) -- `broadcast` degrades to a
        // plain single send for a joiner regardless, so this is correct
        // either way.
        self.broadcast(&encode_message(&NetMessage::ResyncRequest { tick }));
    }

    pub fn send_resync_payload(&mut self, tick: Tick, marbles: Vec<Marble>) {
        self.broadcast(&encode_message(&NetMessage::ResyncPayload { tick, marbles }));
    }

    /// Drains every checksum the peer has sent since the last call.
    pub fn poll_checksums(&mut self) -> Vec<(Tick, u64)> {
        self.drain_and_demux();
        std::mem::take(&mut self.pending_checksums)
    }

    /// Drains every resync request received since the last call (only
    /// ever populated on the host side — a non-host peer never receives
    /// one, `Role`'s host-authoritative convention).
    pub fn poll_resync_requests(&mut self) -> Vec<Tick> {
        self.drain_and_demux();
        std::mem::take(&mut self.pending_resync_requests)
    }

    /// Drains every resync payload the peer has sent since the last call
    /// (only ever populated on the non-host side).
    pub fn poll_resync_payloads(&mut self) -> Vec<(Tick, Vec<Marble>)> {
        self.drain_and_demux();
        std::mem::take(&mut self.pending_resync_payloads)
    }

    /// Drains every `Welcome` received since the last call (joiner-only in
    /// practice — a host never receives one).
    pub fn poll_welcomes(&mut self) -> Vec<PlayerIndex> {
        self.drain_and_demux();
        std::mem::take(&mut self.pending_welcomes)
    }

    /// Test-only: injects a `ResyncPayload` directly into the pending
    /// queue, bypassing `js_bridge` entirely -- `physics_sys.rs`'s fix-4
    /// regression test needs to simulate "a resync payload arrived" for a
    /// `WebRtcTransport` under test, but `js_bridge` is a no-op stub on the
    /// native target these tests run on (there's no real WebRTC channel to
    /// receive anything from), so there's no way to exercise the real
    /// `drain_and_demux` path here.
    #[cfg(test)]
    pub(crate) fn test_inject_resync_payload(&mut self, tick: Tick, marbles: Vec<Marble>) {
        self.pending_resync_payloads.push((tick, marbles));
    }
}

impl InputTransport for WebRtcTransport {
    fn send_input(&mut self, tick: Tick, input: PlayerInput) {
        self.broadcast(&encode_message(&NetMessage::Input { tick, input }));
    }

    fn poll_received(&mut self) -> Vec<(PlayerIndex, Tick, PlayerInput)> {
        self.drain_and_demux();
        std::mem::take(&mut self.pending_inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrips_exactly() {
        let input = PlayerInput {
            dx: 0.37,
            dy: -0.81,
            orientation: Quat::from_xyzw(0.1, -0.2, 0.3, 0.9).normalize(),
        };
        let bytes = pack(123_456_789, input);
        let (tick, got) = unpack(&bytes);
        assert_eq!(tick, 123_456_789);
        assert_eq!(got.dx, input.dx);
        assert_eq!(got.dy, input.dy);
        assert_eq!(got.orientation, input.orientation);
    }

    fn sample_marble(x: f32) -> Marble {
        Marble { pos: Vec3::new(x, x + 1.0, x + 2.0), vel: Vec3::new(-x, x * 0.5, 0.25), rad: 0.3, last_thrust: Vec3::ZERO }
    }

    /// Each [`NetMessage`] kind round-trips through [`encode_message`]/
    /// [`decode_messages`] exactly — the tagged-protocol analogue of
    /// `pack_unpack_roundtrips_exactly` for the three new message kinds
    /// (`Input`'s own body encoding is already covered by that test).
    #[test]
    fn encode_decode_roundtrips_each_message_kind() {
        let input = PlayerInput { dx: 0.42, dy: -0.17, orientation: Quat::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize() };
        let cases = vec![
            NetMessage::Input { tick: 42, input },
            NetMessage::Checksum { tick: 100, hash: 0xdead_beef_1234_5678 },
            NetMessage::ResyncRequest { tick: 7 },
            NetMessage::ResyncPayload { tick: 55, marbles: vec![sample_marble(1.0), sample_marble(2.0)] },
            NetMessage::SceneSync { bytes: vec![1, 2, 3, 4, 5] },
            NetMessage::Welcome { index: 2 },
        ];
        for msg in cases {
            let decoded = decode_messages(&encode_message(&msg));
            assert_eq!(decoded.len(), 1, "one encoded message must decode back to exactly one message");
            match (&msg, &decoded[0]) {
                (NetMessage::Input { tick: t1, input: i1 }, NetMessage::Input { tick: t2, input: i2 }) => {
                    assert_eq!(t1, t2);
                    assert_eq!(i1.dx, i2.dx);
                    assert_eq!(i1.dy, i2.dy);
                    assert_eq!(i1.orientation, i2.orientation);
                }
                (NetMessage::Checksum { tick: t1, hash: h1 }, NetMessage::Checksum { tick: t2, hash: h2 }) => {
                    assert_eq!(t1, t2);
                    assert_eq!(h1, h2);
                }
                (NetMessage::ResyncRequest { tick: t1 }, NetMessage::ResyncRequest { tick: t2 }) => assert_eq!(t1, t2),
                (
                    NetMessage::ResyncPayload { tick: t1, marbles: m1 },
                    NetMessage::ResyncPayload { tick: t2, marbles: m2 },
                ) => {
                    assert_eq!(t1, t2);
                    assert_eq!(m1.len(), m2.len());
                    for (a, b) in m1.iter().zip(m2.iter()) {
                        assert_eq!(a.pos, b.pos);
                        assert_eq!(a.vel, b.vel);
                        assert_eq!(a.rad, b.rad);
                    }
                }
                (NetMessage::SceneSync { bytes: b1 }, NetMessage::SceneSync { bytes: b2 }) => assert_eq!(b1, b2),
                (NetMessage::Welcome { index: i1 }, NetMessage::Welcome { index: i2 }) => assert_eq!(i1, i2),
                _ => panic!("decoded message kind doesn't match the encoded kind"),
            }
        }
    }

    /// Several different message kinds, encoded back-to-back with no
    /// external framing between them, must decode back into exactly the
    /// same sequence in order — this is the property that makes the
    /// tagged protocol safe to share `net.js`'s single flat-concatenated
    /// `takeReceived` buffer with no length-prefixing at the transport
    /// level (module doc): every message's own encoding says exactly how
    /// many bytes it occupies.
    #[test]
    fn decode_messages_parses_several_concatenated_messages_of_different_kinds_in_order() {
        let mut bytes = Vec::new();
        bytes.extend(encode_message(&NetMessage::Input {
            tick: 1,
            input: PlayerInput { dx: 0.0, dy: 0.0, orientation: Quat::IDENTITY },
        }));
        bytes.extend(encode_message(&NetMessage::Checksum { tick: 2, hash: 999 }));
        bytes.extend(encode_message(&NetMessage::ResyncRequest { tick: 3 }));
        bytes.extend(encode_message(&NetMessage::ResyncPayload { tick: 4, marbles: vec![sample_marble(0.0)] }));
        bytes.extend(encode_message(&NetMessage::SceneSync { bytes: vec![9, 8, 7] }));
        bytes.extend(encode_message(&NetMessage::Welcome { index: 3 }));
        bytes.extend(encode_message(&NetMessage::Input {
            tick: 5,
            input: PlayerInput { dx: 0.5, dy: 0.5, orientation: Quat::IDENTITY },
        }));

        let decoded = decode_messages(&bytes);
        assert_eq!(decoded.len(), 7);
        assert!(matches!(decoded[0], NetMessage::Input { tick: 1, .. }));
        assert!(matches!(decoded[1], NetMessage::Checksum { tick: 2, hash: 999 }));
        assert!(matches!(decoded[2], NetMessage::ResyncRequest { tick: 3 }));
        assert!(matches!(decoded[3], NetMessage::ResyncPayload { tick: 4, .. }));
        assert!(matches!(&decoded[4], NetMessage::SceneSync { bytes } if bytes == &[9, 8, 7]));
        assert!(matches!(decoded[5], NetMessage::Welcome { index: 3 }));
        assert!(matches!(decoded[6], NetMessage::Input { tick: 5, .. }));
    }

    /// [`WebRtcTransport`]'s demultiplexing: a fabricated byte stream
    /// carrying every message kind at once, run through the same decode
    /// path `poll_*` uses internally, must sort each message into its own
    /// bucket without losing or misattributing any of them. Exercised at
    /// the `decode_messages` level rather than truly through
    /// `WebRtcTransport` itself, since the latter's `send`/`take_received`
    /// go through the wasm-only `js_bridge` (a no-op stub on the native
    /// target these tests run on) -- this still fully covers the
    /// demultiplexing logic itself, just not the native-stub plumbing
    /// around it.
    #[test]
    fn decode_messages_preserves_every_kind_when_all_are_mixed_together() {
        let mut bytes = Vec::new();
        bytes.extend(encode_message(&NetMessage::Checksum { tick: 10, hash: 1 }));
        bytes.extend(encode_message(&NetMessage::Input {
            tick: 11,
            input: PlayerInput { dx: 1.0, dy: 0.0, orientation: Quat::IDENTITY },
        }));
        bytes.extend(encode_message(&NetMessage::ResyncPayload {
            tick: 12,
            marbles: vec![sample_marble(3.0), sample_marble(4.0), sample_marble(5.0)],
        }));
        bytes.extend(encode_message(&NetMessage::ResyncRequest { tick: 13 }));
        bytes.extend(encode_message(&NetMessage::SceneSync { bytes: vec![42; 37] }));
        bytes.extend(encode_message(&NetMessage::Checksum { tick: 14, hash: 2 }));

        let decoded = decode_messages(&bytes);
        let checksum_count = decoded.iter().filter(|m| matches!(m, NetMessage::Checksum { .. })).count();
        let input_count = decoded.iter().filter(|m| matches!(m, NetMessage::Input { .. })).count();
        let payload_marble_count = decoded
            .iter()
            .find_map(|m| match m {
                NetMessage::ResyncPayload { marbles, .. } => Some(marbles.len()),
                _ => None,
            })
            .unwrap();
        let scene_sync_len = decoded
            .iter()
            .find_map(|m| match m {
                NetMessage::SceneSync { bytes } => Some(bytes.len()),
                _ => None,
            })
            .unwrap();
        assert_eq!(checksum_count, 2);
        assert_eq!(input_count, 1);
        assert_eq!(payload_marble_count, 3);
        assert_eq!(scene_sync_len, 37);
        assert!(decoded.iter().any(|m| matches!(m, NetMessage::ResyncRequest { tick: 13 })));
    }

    /// Fix 2 regression test: a valid message followed by a truncated tail
    /// (fewer bytes than the tag promises) must not panic — the truncated
    /// tail is silently dropped and everything decoded before it is kept.
    /// Pre-fix, this would panic on the out-of-bounds slice for whichever
    /// tag's fixed body length exceeded what's left in `bytes`.
    #[test]
    fn decode_messages_stops_cleanly_on_a_truncated_trailing_message_instead_of_panicking() {
        let mut bytes = Vec::new();
        bytes.extend(encode_message(&NetMessage::Checksum { tick: 10, hash: 1 }));
        // A `TAG_INPUT` byte with only 3 of the required 32 body bytes
        // actually present.
        bytes.push(TAG_INPUT);
        bytes.extend_from_slice(&[1, 2, 3]);

        let decoded = decode_messages(&bytes);
        assert_eq!(decoded.len(), 1);
        assert!(matches!(decoded[0], NetMessage::Checksum { tick: 10, hash: 1 }));
    }

    /// Fix 2 regression test: a `ResyncPayload` whose wire-supplied `count`
    /// claims far more marbles than actually follow in the buffer must not
    /// panic. Pre-fix, `bytes[i..i + count * MARBLE_LEN]` would either
    /// panic on an out-of-bounds slice (small buffers) or attempt to
    /// multiply-overflow / index with a huge `count`; both are unreachable
    /// after the fix, which validates the needed length against what's
    /// actually left before ever slicing.
    #[test]
    fn decode_messages_rejects_a_resync_payload_whose_count_exceeds_the_buffer() {
        let mut bytes = Vec::new();
        bytes.push(TAG_RESYNC_PAYLOAD);
        bytes.extend_from_slice(&55u64.to_le_bytes()); // tick
        bytes.extend_from_slice(&1000u32.to_le_bytes()); // count -- wildly more than 0 marbles follow
        // No marble bytes actually present.
        let decoded = decode_messages(&bytes);
        assert!(decoded.is_empty(), "a malformed ResyncPayload must decode to nothing, not panic");

        // Also confirm a `count` near `u32::MAX` (which would overflow
        // `count * MARBLE_LEN` in a naive `usize` multiply on a 32-bit
        // target, and is a many-terabyte request regardless) is rejected
        // the same way rather than panicking or attempting to allocate.
        let mut overflow_bytes = Vec::new();
        overflow_bytes.push(TAG_RESYNC_PAYLOAD);
        overflow_bytes.extend_from_slice(&1u64.to_le_bytes());
        overflow_bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        let decoded_overflow = decode_messages(&overflow_bytes);
        assert!(decoded_overflow.is_empty());
    }

    /// Fix 2 regression test: same as the `ResyncPayload` case above but for
    /// `SceneSync`'s wire-supplied `len` prefix.
    #[test]
    fn decode_messages_rejects_a_scene_sync_whose_len_exceeds_the_buffer() {
        let mut bytes = Vec::new();
        bytes.push(TAG_SCENE_SYNC);
        bytes.extend_from_slice(&9999u32.to_le_bytes()); // len -- nothing close to this follows
        bytes.extend_from_slice(&[1, 2, 3]);
        let decoded = decode_messages(&bytes);
        assert!(decoded.is_empty(), "a malformed SceneSync must decode to nothing, not panic");
    }

    /// Fix 2 regression test: a stream that ends exactly after a tag byte
    /// (no body bytes at all, including `Welcome`'s single-byte index) must
    /// not panic on `bytes[i]`.
    #[test]
    fn decode_messages_stops_cleanly_when_stream_ends_right_after_a_tag_byte() {
        for tag in [TAG_INPUT, TAG_CHECKSUM, TAG_RESYNC_REQUEST, TAG_RESYNC_PAYLOAD, TAG_SCENE_SYNC, TAG_WELCOME] {
            let decoded = decode_messages(&[tag]);
            assert!(decoded.is_empty(), "tag {tag} with zero body bytes must decode to nothing, not panic");
        }
    }
}
