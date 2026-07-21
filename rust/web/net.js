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
window.mmNet = (function () {
  let peer = null;
  let conn = null;
  let hostedId = "";
  let lastError = "";
  // 0 idle, 1 hosting (waiting for a peer to connect), 2 connected,
  // 3 disconnected (was connected, now isn't), 4 error.
  let status = 0;
  let received = []; // array of Uint8Array, one per message, in arrival order

  function setupConnection(c) {
    conn = c;
    conn.on("open", () => {
      status = 2;
    });
    conn.on("data", (data) => {
      // Always sent as a raw ArrayBuffer from the Rust side (net.rs packs
      // fixed-size binary records) -- PeerJS hands it back as whatever
      // arrived, normalize to Uint8Array here so `takeReceived` never has
      // to branch on type.
      received.push(new Uint8Array(data));
    });
    conn.on("close", () => {
      if (status === 2) status = 3;
    });
    conn.on("error", (err) => {
      lastError = String(err && err.message ? err.message : err);
      status = 4;
    });
  }

  return {
    // Creates this client's Peer with a broker-assigned id and waits for
    // an incoming connection -- call once, at startup, unconditionally
    // (see net.rs's doc: every session hosts by default, whether or not
    // anyone ever uses the resulting link is irrelevant to single-player
    // behavior).
    host: function () {
      peer = new Peer();
      peer.on("open", (id) => {
        hostedId = id;
        status = 1;
      });
      peer.on("connection", (c) => setupConnection(c));
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
        setupConnection(peer.connect(hostId, { reliable: true, serialization: "binary" }));
      });
      peer.on("error", (err) => {
        lastError = String(err && err.message ? err.message : err);
        status = 4;
      });
    },
    // `bytes` is a Uint8Array from the wasm side. No-op if not currently
    // connected -- callers are expected to check `status()` themselves,
    // this is just a safe fallback against a stray call during a
    // connect/disconnect race.
    send: function (bytes) {
      if (conn && conn.open) {
        conn.send(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
      }
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
    // Drains every message received since the last call, concatenated
    // into one flat Uint8Array -- safe because net.rs only ever sends
    // fixed-size 32-byte records, so no length-prefixing is needed on
    // either side; the Rust side just chunks the result by 32.
    takeReceived: function () {
      if (received.length === 0) return new Uint8Array(0);
      const total = received.reduce((n, r) => n + r.length, 0);
      const out = new Uint8Array(total);
      let offset = 0;
      for (const r of received) {
        out.set(r, offset);
        offset += r.length;
      }
      received = [];
      return out;
    },
  };
})();
