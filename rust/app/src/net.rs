//! Multiplayer milestone 2: real cross-machine transport for
//! [`marble_csg::rollback::InputTransport`], over WebRTC via PeerJS.
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
//! **Exactly 2 players, always**: the host is always player index `0`, the
//! joiner always `1` — matches `rollback.rs`'s own test setups
//! (`two_player_setup`) and keeps the join-link flow simple (one incoming
//! `DataConnection`, not a mesh/star topology decision). N > 2 networked
//! players is future work, not this milestone.
//!
//! **Networking is strictly additive**: nothing here changes single-player
//! behavior. A session always starts hosting in the background (creating a
//! `Peer` costs nothing if nobody ever uses the resulting link), but
//! `physics_sys.rs` only switches from its existing direct `step_marbles`
//! call to a [`RollbackSim`](marble_csg::rollback::RollbackSim)-driven one
//! once a real `DataConnection` actually opens — never just because a
//! `Peer` exists.

use bevy::prelude::*;

use marble_csg::physics::PlayerInput;
use marble_csg::rollback::{InputTransport, PlayerIndex, Tick};

use crate::web_config::query_param;

/// Wire size of one packed [`PlayerInput`] record: `tick: u64` (8) +
/// `dx`/`dy: f32` (4 each) + `orientation: Quat` as 4×`f32` (16) = 32
/// bytes. Fixed-size and never varies, so `net.js`'s `takeReceived` can
/// just concatenate messages with no length-prefixing — the receiver
/// always knows to chunk by exactly this many bytes.
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

#[cfg(target_arch = "wasm32")]
mod js_bridge {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = mmNet, js_name = host)]
        pub fn host();
        #[wasm_bindgen(js_namespace = mmNet, js_name = join)]
        pub fn join(host_id: &str);
        #[wasm_bindgen(js_namespace = mmNet, js_name = send)]
        pub fn send(bytes: &[u8]);
        #[wasm_bindgen(js_namespace = mmNet, js_name = status)]
        pub fn status() -> i32;
        #[wasm_bindgen(js_namespace = mmNet, js_name = hostedId)]
        pub fn hosted_id() -> String;
        #[wasm_bindgen(js_namespace = mmNet, js_name = takeError)]
        pub fn take_error() -> String;
        #[wasm_bindgen(js_namespace = mmNet, js_name = takeReceived)]
        pub fn take_received() -> Vec<u8>;
    }
}

/// Native build stub: networking is a wasm/browser-only concept here (there
/// is no PeerJS/WebRTC on native at all), so every one of these is a
/// no-op/idle value — native always behaves as if it's permanently
/// offline, which is exactly today's (pre-milestone-2) native behavior.
#[cfg(not(target_arch = "wasm32"))]
mod js_bridge {
    pub fn host() {}
    pub fn join(_host_id: &str) {}
    pub fn send(_bytes: &[u8]) {}
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

/// This client's role: host (player 0, waiting for/hosting a joiner) or
/// joiner (player 1, connecting to a host's shared link) — fixed for the
/// whole session from the very first frame (`?join=` present or not),
/// never renegotiated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Host,
    Joiner,
}

impl Role {
    /// This client's own player index in the (always exactly 2 players)
    /// session — see the module doc.
    pub fn local_index(self) -> PlayerIndex {
        match self {
            Self::Host => 0,
            Self::Joiner => 1,
        }
    }

    pub fn remote_index(self) -> PlayerIndex {
        match self {
            Self::Host => 1,
            Self::Joiner => 0,
        }
    }
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
}

/// `Update` system: renders [`NetSession`]'s current state into
/// [`NetStatusText`] — the one place a human (not just `net.rs` itself)
/// finds out there's an invite link to share, or that a peer connected.
pub fn sync_net_ui_text(session: Res<NetSession>, mut text: Query<&mut Text, With<NetStatusText>>) {
    let Ok(mut text) = text.single_mut() else { return };
    text.0 = match session.status {
        NetStatus::Idle => "connecting...".to_string(),
        NetStatus::Hosting => match &session.hosted_join_suffix {
            Some(suffix) => {
                let link = page_url_base().map_or_else(|| suffix.clone(), |base| format!("{base}{suffix}"));
                format!("Share this link to play together:\n{link}")
            }
            None => "starting host...".to_string(),
        },
        NetStatus::Connected => "Player connected!".to_string(),
        NetStatus::Disconnected => "Other player disconnected — still playing solo.".to_string(),
        NetStatus::Error => format!("Network error: {}", session.last_error),
    };
}

/// [`InputTransport`] over the live WebRTC data channel — see the module
/// doc for the wire format. `remote` is fixed for the whole session
/// (exactly one other player, always the same index — [`Role::remote_index`]).
pub struct WebRtcTransport {
    remote: PlayerIndex,
}

impl WebRtcTransport {
    pub fn new(role: Role) -> Self {
        Self { remote: role.remote_index() }
    }
}

impl InputTransport for WebRtcTransport {
    fn send_input(&mut self, tick: Tick, input: PlayerInput) {
        js_bridge::send(&pack(tick, input));
    }

    fn poll_received(&mut self) -> Vec<(PlayerIndex, Tick, PlayerInput)> {
        let bytes = js_bridge::take_received();
        // `net.js`'s `takeReceived` only ever concatenates whole
        // `RECORD_LEN`-byte records (this module is the only thing that
        // ever calls `send`, always with exactly one `pack`ed record) —
        // `chunks_exact` silently dropping a trailing partial chunk is
        // therefore not a real failure mode, just defensive.
        bytes
            .chunks_exact(RECORD_LEN)
            .map(|chunk| {
                let (tick, input) = unpack(chunk);
                (self.remote, tick, input)
            })
            .collect()
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

    #[test]
    fn role_indices_are_always_the_opposite_pair() {
        assert_eq!(Role::Host.local_index(), 0);
        assert_eq!(Role::Host.remote_index(), 1);
        assert_eq!(Role::Joiner.local_index(), 1);
        assert_eq!(Role::Joiner.remote_index(), 0);
    }
}
