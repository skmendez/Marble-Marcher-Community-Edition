// Minimal PeerJS wrapper exposing a narrow, poll-based API to the wasm
// build (rust/app/src/net.rs), so all WebRTC/PeerJS event-callback
// complexity (open/connection/data/close/error) lives here in plain JS,
// and Rust only ever calls a handful of functions and polls plain
// values/byte arrays once per frame -- the same "poll what arrived since
// last time" shape `InputTransport::poll_received` already has, so the
// Rust side is a thin, boring adapter rather than a second place that has
// to get WebRTC's callback lifetimes right.
//
// PeerJS (https://peerjs.com) is loaded via a plain <script> tag in
// index.html before this file and before the wasm module -- it wraps
// browser WebRTC plus a free public signaling broker (0.peerjs.com by
// default) purely for the initial SDP/ICE handshake; once connected, data
// flows directly peer-to-peer, not through the broker.
//
// N-player topology: host-relay star, not a mesh. Every joiner connects
// only to the host (PeerJS gives this for free); joiners never connect to
// each other directly. The host assigns each incoming connection the next
// sequential player index (1, 2, 3, ...; host is always 0) and keeps that
// assignment for the session -- a departed joiner's index is never reused
// (matches net.rs's "stable index order" convention). Capped at
// MAX_JOINERS additional players (4 total) -- a connection beyond that is
// closed immediately.
window.mmNet = (function () {
  const MAX_JOINERS = 3;

  let peer = null;
  // Joiner role only: this client's single connection to the host.
  let hostConn = null;
  let received = []; // joiner only: array of Uint8Array, arrival order.

  // Host role only: player index (1..MAX_JOINERS) -> PeerJS DataConnection.
  let joinerConns = {};
  let receivedByIndex = {}; // player index -> array of Uint8Array.
  let connectedSet = new Set(); // indices currently open.
  let newlyConnected = []; // indices that opened since the last poll.
  let newlyDisconnected = []; // indices that closed since the last poll.
  let nextJoinerIndex = 1;

  let hostedId = "";
  let lastError = "";
  // 0 idle, 1 hosting (waiting for a peer to connect), 2 connected,
  // 3 disconnected (was connected, now isn't), 4 error. For the host this
  // reflects "at least one joiner has ever connected" -- per-joiner detail
  // is what newlyConnected/newlyDisconnected are for.
  let status = 0;
  let clipboardStatus = 0;

  function setupJoinerConnection(c) {
    hostConn = c;
    hostConn.on("open", () => {
      status = 2;
    });
    hostConn.on("data", (data) => {
      received.push(new Uint8Array(data));
    });
    hostConn.on("close", () => {
      if (status === 2) status = 3;
    });
    hostConn.on("error", (err) => {
      lastError = String(err && err.message ? err.message : err);
      status = 4;
    });
  }

  function setupHostConnection(idx, c) {
    joinerConns[idx] = c;
    receivedByIndex[idx] = [];
    c.on("open", () => {
      connectedSet.add(idx);
      newlyConnected.push(idx);
      status = 2;
    });
    c.on("data", (data) => {
      receivedByIndex[idx].push(new Uint8Array(data));
    });
    c.on("close", () => {
      if (connectedSet.has(idx)) {
        connectedSet.delete(idx);
        newlyDisconnected.push(idx);
      }
    });
    c.on("error", (err) => {
      lastError = String(err && err.message ? err.message : err);
    });
  }

  function concatAndClear(arr) {
    if (arr.length === 0) return new Uint8Array(0);
    const total = arr.reduce((n, r) => n + r.length, 0);
    const out = new Uint8Array(total);
    let offset = 0;
    for (const r of arr) {
      out.set(r, offset);
      offset += r.length;
    }
    arr.length = 0;
    return out;
  }

  return {
    // Creates this client's Peer with a broker-assigned id and accepts
    // incoming connections, one per joiner, up to MAX_JOINERS -- call
    // once, at startup, unconditionally (see net.rs's doc: every session
    // hosts by default, whether or not anyone ever uses the resulting
    // link is irrelevant to single-player behavior).
    host: function () {
      peer = new Peer();
      peer.on("open", (id) => {
        hostedId = id;
        status = 1;
      });
      peer.on("connection", (c) => {
        if (nextJoinerIndex > MAX_JOINERS) {
          c.close(); // session already at the player cap
          return;
        }
        const idx = nextJoinerIndex++;
        setupHostConnection(idx, c);
      });
      peer.on("error", (err) => {
        lastError = String(err && err.message ? err.message : err);
        status = 4;
      });
    },
    // Creates this client's own (unshared) Peer, then connects to
    // `hostId` once it's ready -- call once, at startup, only when a
    // `?join=` id was present in the URL.
    join: function (hostId) {
      peer = new Peer();
      peer.on("open", () => {
        setupJoinerConnection(peer.connect(hostId, { reliable: true, serialization: "binary" }));
      });
      peer.on("error", (err) => {
        lastError = String(err && err.message ? err.message : err);
        status = 4;
      });
    },
    // Joiner-only: send to the one host connection. `bytes` is a
    // Uint8Array from the wasm side. No-op if not currently connected.
    send: function (bytes) {
      if (hostConn && hostConn.open) {
        hostConn.send(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
      }
    },
    // Host-only: send to one specific joiner by its assigned index.
    // No-op if that index isn't currently connected.
    sendTo: function (idx, bytes) {
      const c = joinerConns[idx];
      if (c && c.open) {
        c.send(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
      }
    },
    // `text` is the full invite link. Fire-and-forget from the wasm side --
    // must be called synchronously from within a real click/tap's event
    // handler (the Clipboard API only grants permission inside an actual
    // user-gesture callstack; calling this after an `await`/async hop, or
    // from a plain per-frame poll, would silently fail every time even
    // though the code looks identical). The write itself is async, so the
    // result is reported through `takeClipboardStatus`, not a return value.
    copyToClipboard: function (text) {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(
          () => { clipboardStatus = 1; },
          () => { clipboardStatus = 2; },
        );
      } else {
        // No Clipboard API at all (non-secure context, or an embedded
        // webview that doesn't expose it) -- report failure immediately
        // rather than leaving the caller waiting on a status that will
        // never arrive.
        clipboardStatus = 2;
      }
    },
    takeClipboardStatus: function () {
      const s = clipboardStatus;
      clipboardStatus = 0;
      return s;
    },
    status: function () {
      return status;
    },
    hostedId: function () {
      return hostedId;
    },
    takeError: function () {
      const e = lastError;
      lastError = "";
      return e;
    },
    // Joiner-only: drains every message received from the host since the
    // last call, concatenated into one flat Uint8Array -- each message
    // self-describes its own length (net.rs's tagged protocol), so no
    // extra framing is needed here.
    takeReceived: function () {
      return concatAndClear(received);
    },
    // Host-only: drains every message received from joiner `idx` since
    // the last call, same flat-concatenation convention as `takeReceived`.
    takeReceivedFrom: function (idx) {
      return concatAndClear(receivedByIndex[idx] || []);
    },
    // Host-only: player indices whose connection opened since the last
    // call to this function (each index reported exactly once, the tick
    // its connection actually opens).
    newlyConnectedIndices: function () {
      const out = new Uint8Array(newlyConnected);
      newlyConnected = [];
      return out;
    },
    // Host-only: player indices whose connection closed since the last
    // call to this function.
    newlyDisconnectedIndices: function () {
      const out = new Uint8Array(newlyDisconnected);
      newlyDisconnected = [];
      return out;
    },
  };
})();
