//! Dev-only "snapshot" feature (`web/index.html`'s `?debug=1`-gated dat.gui
//! panel): serializes the *exact* current frozen state -- the scene's live
//! CSG tree/params/animation table, the [`marble_rollback::Tick`] that pins
//! whatever phase an [`marble_csg::expr::Expr`]-driven animated param is
//! currently at, every marble's live position/velocity/radius, and the
//! orbit camera's orientation/zoom -- into one URL-embeddable string,
//! so pasting the resulting `?snapshot=<value>` URL later reproduces the
//! identical still frame deterministically (same scene, same camera angle,
//! same marble position, same animated-fold phase).
//!
//! ## Wire format
//! `[u8 version][u64 tick, LE][u32 scene_len, LE][scene_len bytes:
//! marble_csg::Scene::to_bytes()][u32 marble_count, LE][marble_count *
//! (pos.x pos.y pos.z vel.x vel.y vel.z rad, each an LE f32)][camera
//! orientation as 4 LE f32 (x, y, z, w)][camera zoom, LE f32]`,
//! then the whole buffer is URL-safe-base64 (no padding) encoded (`b64`
//! below).
//!
//! Deliberately reuses [`marble_csg::Scene::to_bytes`]/[`marble_csg::Scene::from_bytes`]
//! wholesale for the scene half, rather than a separate "scene-identity
//! string + `Params` only" pair: capturing the literal object tree this way
//! is exactly what multiplayer's own join-time scene sync already does for
//! the analogous "adopt a wholesale different scene, atomically with a
//! tick + marble list" problem (`render::apply_pending_scene_sync`,
//! `marble_rollback::RollbackSim::set_scene`) -- reusing that existing wire
//! format outright means this feature's correctness never depends on a
//! `render::SceneKind` match arm independently reconstructing
//! byte-identical geometry from a scene-identity string. The one thing this
//! does *not* capture is which [`crate::render::SceneKind`] variant built
//! the scene in the first place -- unneeded structurally (the object tree
//! is fully self-describing, same as multiplayer's scene sync), and the
//! "copy snapshot" button (`report_snapshot_state`, `web/index.html`)
//! builds its URL by preserving the page's own current `?scene=`, so the
//! common case (reload the same page's own snapshot) already has the two
//! agreeing without this format needing to carry it.
//!
//! No re-derivation of animated params from `tick` is needed at load time
//! (unlike a fresh `Scene` built from scratch, whose slots start at
//! whatever `render::setup`'s per-scene tick-0 default is): the captured
//! `Params` are read directly off `RollbackSim::scene()`, which
//! [`marble_csg::expr::apply_animations`] already overwrites in place, in
//! step with `tick`, every time [`marble_rollback::RollbackSim::advance`]/
//! `resim_from` runs -- so what's captured here is already that tick's
//! fully-resolved values, not a pre-animation base that would need
//! re-deriving. Loading just has to put `Params`/`tick` back exactly as
//! captured; the *next* `advance()` call naturally continues the
//! animation's phase forward from there, using the shared `Tick` clock
//! exactly as it would have if this session had simply kept running.
//!
//! Solo-only end to end ([`crate::physics_sys::MultiplayerSession::is_solo`]'s
//! doc): [`report_snapshot_state`] only ever reports a real (non-empty)
//! value while solo -- same reasoning the live params panel already
//! established (`RollbackSim::params_mut`'s doc) -- injecting arbitrary
//! frozen state bypasses the deterministic input-replay model multiplayer
//! depends on. Loading one (`render::load_snapshot_from_url`) is inherently
//! solo too: it only ever runs once, at `Startup`, strictly before any
//! network connection can exist.

use bevy::prelude::*;

use marble_csg::physics::Marble;
use marble_csg::{Scene, Tick};

use crate::camera::CameraOrbit;
use crate::physics_sys::MultiplayerSession;

/// One marble's wire size: `pos` (3), `vel` (3), `rad` (1) LE `f32`s.
const MARBLE_STRIDE_BYTES: usize = 7 * 4;
/// Camera wire size: `orientation` (4, xyzw) + `zoom` (1) LE `f32`s.
const CAMERA_BYTES: usize = 5 * 4;

/// See the module doc for the exact byte layout.
#[derive(Debug, Clone)]
pub struct SceneSnapshot {
    pub tick: Tick,
    pub scene: Scene,
    pub marbles: Vec<Marble>,
    pub camera_orientation: Quat,
    /// [`CameraOrbit::zoom`] -- the player's zoom *preference* (a multiplier
    /// on the automatically-framed distance), not an absolute world
    /// distance. Restoring this alone is enough to reproduce the identical
    /// rendered eye position: [`crate::render::load_snapshot_from_url`] also
    /// resets [`crate::smart_camera::CameraRig`] (the realized camera,
    /// derived from this every frame) so the very next solve snaps straight
    /// to it instead of springing there over several frames.
    pub camera_zoom: f32,
}

impl SceneSnapshot {
    /// Bumped any time the wire layout changes -- a `?snapshot=` value from
    /// an old build is rejected outright (`decode`'s first check) rather
    /// than risk misreading a shifted-but-still-parseable byte layout, the
    /// same defensive posture `marble_csg`'s own decoders take toward any
    /// input that didn't originate from a `Scene::from_bytes`-validated
    /// tree.
    const VERSION: u8 = 1;

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(Self::VERSION);
        out.extend_from_slice(&self.tick.to_le_bytes());

        let scene_bytes = self.scene.to_bytes();
        out.extend_from_slice(&(scene_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&scene_bytes);

        out.extend_from_slice(&(self.marbles.len() as u32).to_le_bytes());
        for m in &self.marbles {
            for v in [m.pos.x, m.pos.y, m.pos.z, m.vel.x, m.vel.y, m.vel.z, m.rad] {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }

        let q = self.camera_orientation;
        for v in [q.x, q.y, q.z, q.w, self.camera_zoom] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Inverse of [`Self::encode`] -- `None` on any malformed/truncated/
    /// version-mismatched input, or leftover bytes after a complete decode
    /// (same "reject rather than silently under-read" posture as
    /// `marble_csg::scene_sync::Scene::from_bytes`).
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let mut pos = 0usize;

        if *bytes.first()? != Self::VERSION {
            return None;
        }
        pos += 1;

        let tick = Tick::from_le_bytes(bytes.get(pos..pos + 8)?.try_into().ok()?);
        pos += 8;

        let scene_len = u32::from_le_bytes(bytes.get(pos..pos + 4)?.try_into().ok()?) as usize;
        pos += 4;
        let scene_end = pos.checked_add(scene_len)?;
        let scene = Scene::from_bytes(bytes.get(pos..scene_end)?)?;
        pos = scene_end;

        let marble_count = u32::from_le_bytes(bytes.get(pos..pos + 4)?.try_into().ok()?) as usize;
        pos += 4;
        // Reject before allocating: same reasoning as every other
        // length-prefixed decoder in this codebase (`marble_csg::Params::
        // decode_at`'s doc) -- a corrupted `marble_count` near `u32::MAX`
        // would otherwise attempt a multi-GB `Vec::with_capacity`, an
        // allocation failure that aborts the process rather than a
        // catchable parse error.
        if marble_count > bytes.len().saturating_sub(pos) / MARBLE_STRIDE_BYTES {
            return None;
        }
        let mut marbles = Vec::with_capacity(marble_count);
        for _ in 0..marble_count {
            let end = pos + MARBLE_STRIDE_BYTES;
            let chunk = bytes.get(pos..end)?;
            let f = |lo: usize| f32::from_le_bytes(chunk[lo..lo + 4].try_into().unwrap());
            marbles.push(Marble {
                pos: Vec3::new(f(0), f(4), f(8)),
                vel: Vec3::new(f(12), f(16), f(20)),
                rad: f(24),
                // Not captured -- purely a debug-gizmo visualization value,
                // recomputed by the very next `step_marble` call regardless
                // (`Marble::last_thrust`'s doc); zero is exactly what a
                // freshly-`spawn`ed marble also starts at.
                last_thrust: Vec3::ZERO,
            });
            pos = end;
        }

        let cam_end = pos + CAMERA_BYTES;
        let chunk = bytes.get(pos..cam_end)?;
        let f = |lo: usize| f32::from_le_bytes(chunk[lo..lo + 4].try_into().unwrap());
        // Renormalized defensively, matching `CameraOrbit::drag`'s own
        // per-step renormalization -- a hand-edited or bit-flipped-in-
        // transit `?snapshot=` value could otherwise hand `eye_and_basis` a
        // non-unit quaternion.
        let camera_orientation = Quat::from_xyzw(f(0), f(4), f(8), f(12)).normalize();
        let camera_zoom = f(16);
        pos = cam_end;

        if pos != bytes.len() {
            return None;
        }

        Some(Self { tick, scene, marbles, camera_orientation, camera_zoom })
    }

    pub fn to_url_param(&self) -> String {
        b64::encode(&self.encode())
    }

    pub fn from_url_param(s: &str) -> Option<Self> {
        Self::decode(&b64::decode(s)?)
    }
}

/// `Update` system (gated on `Config::debug_enabled` at the `main.rs`
/// call site, matching every other dev-only system's convention, e.g.
/// `draw_thrust_debug`): pushes the current frame's snapshot -- base64,
/// ready to embed straight into a `?snapshot=` URL -- to JS every frame via
/// `js_bridge::report_snapshot`, so the dat.gui "copy snapshot" button
/// (`web/index.html`) always has a fresh value to hand `navigator.clipboard`
/// on click without needing a separate JS-calls-into-Rust round trip
/// (matching `gpu_profile.rs`/`step_data.rs`'s existing "Rust pushes,
/// JS holds the latest" convention, not the getter-style `live_mrrm_enabled`/
/// `live_view_mode` polling convention -- there's no plain-JS-object
/// equivalent of `RollbackSim`'s state for JS to hold itself).
///
/// Reports an empty string while connected to a peer (`MultiplayerSession::
/// is_solo`'s doc) -- the JS side treats that as "not available right now"
/// rather than copying a broken/misleading link.
pub fn report_snapshot_state(mp: Res<MultiplayerSession>, camera_orbit: Res<CameraOrbit>) {
    let encoded = if mp.is_solo() {
        let snapshot = SceneSnapshot {
            tick: mp.sim.current_tick(),
            scene: mp.sim.scene().clone(),
            marbles: mp.sim.marbles().to_vec(),
            camera_orientation: camera_orbit.orientation,
            camera_zoom: camera_orbit.zoom,
        };
        snapshot.to_url_param()
    } else {
        String::new()
    };
    crate::net::js_bridge::report_snapshot(&encoded);
}

/// Hand-rolled URL-safe (RFC 4648 §5), unpadded base64 -- the same
/// "one small, auditable function beats a crate dependency for this"
/// reasoning `marble_rollback::state_checksum`'s doc gives for its own
/// hand-rolled FNV-1a, and `net.rs`'s hand-rolled wire codecs give more
/// generally: this repo has no `serde`/`base64` dependency anywhere, and a
/// `?snapshot=` payload is exactly the kind of small, already-versioned,
/// already-length-checked byte blob those existing codecs are built to
/// carry, so pulling in a whole crate just for this one call site would be
/// out of proportion. URL-safe (`-`/`_` instead of `+`/`/`) so the result
/// never needs percent-encoding to sit directly in a query string value;
/// unpadded since trailing `=` has no query-string-safety benefit here and
/// this module always knows the exact byte length up front (no streaming
/// concatenation that would need padding to stay unambiguous).
mod b64 {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

    pub fn encode(bytes: &[u8]) -> String {
        let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
        for chunk in bytes.chunks(3) {
            let b0 = chunk[0] as u32;
            let b1 = *chunk.get(1).unwrap_or(&0) as u32;
            let b2 = *chunk.get(2).unwrap_or(&0) as u32;
            let n = (b0 << 16) | (b1 << 8) | b2;
            out.push(ALPHABET[((n >> 18) & 0x3f) as usize] as char);
            out.push(ALPHABET[((n >> 12) & 0x3f) as usize] as char);
            if chunk.len() > 1 {
                out.push(ALPHABET[((n >> 6) & 0x3f) as usize] as char);
            }
            if chunk.len() > 2 {
                out.push(ALPHABET[(n & 0x3f) as usize] as char);
            }
        }
        out
    }

    fn value(c: u8) -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a') as u32 + 26,
            b'0'..=b'9' => (c - b'0') as u32 + 52,
            b'-' => 62,
            b'_' => 63,
            _ => return None,
        })
    }

    /// `None` on any character outside the alphabet above, or a final
    /// group of exactly 1 leftover character (not a valid base64 length --
    /// every encoded group is 2, 3, or 4 characters, `encode`'s doc).
    pub fn decode(s: &str) -> Option<Vec<u8>> {
        if !s.is_ascii() {
            return None;
        }
        let chars = s.as_bytes();
        let mut out = Vec::with_capacity(chars.len() * 3 / 4);
        for group in chars.chunks(4) {
            if group.len() == 1 {
                return None;
            }
            let mut n = 0u32;
            for (i, &c) in group.iter().enumerate() {
                n |= value(c)? << (18 - 6 * i);
            }
            out.push((n >> 16) as u8);
            if group.len() >= 3 {
                out.push((n >> 8) as u8);
            }
            if group.len() == 4 {
                out.push(n as u8);
            }
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use marble_csg::scenes::{menger_oscillating_sphere, set_menger_params};
    use marble_csg::Params;

    mod b64_tests {
        use super::super::b64;

        #[test]
        fn round_trips_every_length_mod_3() {
            for bytes in [
                Vec::<u8>::new(),
                vec![0u8],
                vec![1u8, 2],
                vec![1u8, 2, 3],
                vec![1u8, 2, 3, 4],
                vec![1u8, 2, 3, 4, 5],
                vec![1u8, 2, 3, 4, 5, 6],
                (0..=255u8).collect::<Vec<u8>>(),
            ] {
                let encoded = b64::encode(&bytes);
                // Unpadded, URL-safe: no `=`, `+`, or `/` ever appears.
                assert!(!encoded.contains(['=', '+', '/']));
                let decoded = b64::decode(&encoded).expect("a freshly-encoded string must decode");
                assert_eq!(decoded, bytes, "round-trip mismatch for {bytes:?}");
            }
        }

        #[test]
        fn rejects_invalid_characters_and_lone_leftover_char() {
            assert!(b64::decode("not valid base64!").is_none());
            // "AAAAA" is 5 chars -- one full group of 4, plus a lone
            // leftover 5th, which no valid encoding ever produces.
            assert!(b64::decode("AAAAA").is_none());
        }
    }

    fn sample_scene() -> Scene {
        let mut params = Params::new();
        let (object, handles) = menger_oscillating_sphere(&mut params);
        set_menger_params(&mut params, &handles.menger, 5, Vec3::new(0.9, 0.6, 0.2));
        Scene {
            object,
            animations: vec![(handles.radius, handles.radius_anim.clone())],
            params,
        }
    }

    fn sample_marbles() -> Vec<Marble> {
        vec![
            Marble { pos: Vec3::new(1.0, 2.0, 3.0), vel: Vec3::new(-0.5, 0.25, 0.1), rad: 0.15, last_thrust: Vec3::ZERO },
            Marble { pos: Vec3::new(-4.0, 0.5, 2.0), vel: Vec3::ZERO, rad: 0.2, last_thrust: Vec3::new(1.0, 0.0, 0.0) },
        ]
    }

    /// The core round-trip property this whole module exists for: encoding
    /// a known scene/tick/marble/camera state and decoding it back must
    /// reproduce every field exactly -- geometry, params, animation table,
    /// tick, every marble's position/velocity/radius, and the camera's
    /// orientation/zoom (`last_thrust` deliberately excluded, its own
    /// doc above).
    #[test]
    fn scene_snapshot_round_trips_a_real_animated_scene() {
        let scene = sample_scene();
        let marbles = sample_marbles();
        let snapshot = SceneSnapshot {
            tick: 123_456,
            scene: scene.clone(),
            marbles: marbles.clone(),
            camera_orientation: Quat::from_rotation_y(0.8) * Quat::from_rotation_x(0.35),
            camera_zoom: 1.2,
        };

        let bytes = snapshot.encode();
        let decoded = SceneSnapshot::decode(&bytes).expect("a freshly-encoded snapshot must decode");

        assert_eq!(decoded.tick, snapshot.tick);
        assert_eq!(decoded.camera_zoom, snapshot.camera_zoom);
        assert!(decoded.camera_orientation.abs_diff_eq(snapshot.camera_orientation, 1e-6));

        assert_eq!(decoded.marbles.len(), marbles.len());
        for (got, want) in decoded.marbles.iter().zip(marbles.iter()) {
            assert_eq!(got.pos, want.pos);
            assert_eq!(got.vel, want.vel);
            assert_eq!(got.rad, want.rad);
        }

        assert_eq!(decoded.scene.params.slots(), scene.params.slots());
        assert_eq!(decoded.scene.animations.len(), scene.animations.len());
        assert_eq!(decoded.scene.animations[0].1, scene.animations[0].1);
        // The tree itself must actually behave the same, not just decode
        // without error -- probe `de` at a handful of points (same
        // technique `scene_sync`'s own real-scene round-trip test uses).
        for p in [Vec4::new(0.0, 0.0, 0.0, 1.0), Vec4::new(2.0, 1.0, 0.5, 1.0), Vec4::new(-3.0, 2.0, 1.0, 1.0)] {
            assert_eq!(decoded.scene.object.de(p, &decoded.scene.params), scene.object.de(p, &scene.params));
        }
    }

    /// `to_url_param`/`from_url_param` (the actual `?snapshot=` value the
    /// dat.gui button copies and `render::load_snapshot_from_url` reads)
    /// round-trip the same way, end to end through base64 -- and the
    /// result contains no characters that would need percent-encoding in a
    /// URL query string.
    #[test]
    fn url_param_round_trips_and_is_query_string_safe() {
        let snapshot = SceneSnapshot {
            tick: 42,
            scene: sample_scene(),
            marbles: sample_marbles(),
            camera_orientation: Quat::IDENTITY,
            camera_zoom: 0.2,
        };
        let encoded = snapshot.to_url_param();
        assert!(encoded.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'));

        let decoded = SceneSnapshot::from_url_param(&encoded).expect("a freshly-encoded url param must decode");
        assert_eq!(decoded.tick, snapshot.tick);
        assert_eq!(decoded.scene.params.slots(), snapshot.scene.params.slots());
    }

    #[test]
    fn decode_rejects_wrong_version_and_truncated_or_trailing_bytes() {
        let snapshot =
            SceneSnapshot { tick: 1, scene: sample_scene(), marbles: sample_marbles(), camera_orientation: Quat::IDENTITY, camera_zoom: 1.0 };
        let bytes = snapshot.encode();

        let mut wrong_version = bytes.clone();
        wrong_version[0] = 0xFF;
        assert!(SceneSnapshot::decode(&wrong_version).is_none());

        assert!(SceneSnapshot::decode(&bytes[..bytes.len() - 1]).is_none(), "truncated");

        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(SceneSnapshot::decode(&trailing).is_none(), "trailing byte");
    }

    #[test]
    fn decode_rejects_a_marble_count_that_exceeds_the_buffer() {
        // Same "reject before allocating" property `marble_csg::Params::
        // decode_at`'s own regression test checks, but for this module's
        // marble-count field.
        let scene = sample_scene();
        let mut bytes = Vec::new();
        bytes.push(SceneSnapshot::VERSION);
        bytes.extend_from_slice(&0u64.to_le_bytes());
        let scene_bytes = scene.to_bytes();
        bytes.extend_from_slice(&(scene_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&scene_bytes);
        bytes.extend_from_slice(&1_000_000u32.to_le_bytes()); // no marble data follows
        assert!(SceneSnapshot::decode(&bytes).is_none());
    }
}
