# Marble Marcher CSG renderer (Rust/Bevy port)

A port of this repo's C++ CSG fractal framework (`src/fractals/`) to Rust,
rendering via runtime-generated WGSL on Bevy. See `DESIGN.md` for the full
design and `MILESTONES.md` for the build-out plan and status.

Workspace layout:

- `csg/` — `marble-csg`: pure logic (glam only). The `Fold`/`Object` tree,
  CPU distance/nearest-point evaluation, WGSL codegen, and marble physics —
  no Bevy dependency, so `cargo test -p marble-csg` is fast.
- `app/` — `marble-marcher-bevy`: the Bevy 0.16 app (rendering, camera,
  input, fixed-timestep physics wiring).

## Native

```
cargo run -p marble-marcher-bevy
```

WASD moves the marble; `R` forces a manual respawn; `G` toggles between the
two physics models `marble_csg::physics` supports (see its module doc for
exactly what each one is a faithful port of):

- **Rolling** (default) — original MMCE physics: gravity, kill plane,
  horizontal camera-yaw-relative rolling.
- **Flying** — this branch's in-progress zero-gravity experiment ("new
  camera/movement mechanics"): no gravity, no kill plane, full 3D
  camera-relative thrust (WASD flies wherever the camera is actually
  pointed, including up/down).

Left-drag orbits the camera, scroll wheel zooms.

`MM_WINDOW_SIZE=WxH` overrides the starting window resolution — useful on a
software (CPU) Vulkan/GL fallback (e.g. llvmpipe, when no hardware GPU ICD
is available), where this per-pixel ray marcher is far more expensive than
on real GPU hardware; a small window renders far fewer pixels.

## Headless verification (`MM_SCREENSHOT`)

Set `MM_SCREENSHOT=path.png` to capture the primary window and exit —
useful for CI or confirming a change actually renders without a human at
the keyboard:

```
MM_SCREENSHOT=/tmp/shot.png MM_SCREENSHOT_DELAY_SECS=10 cargo run -p marble-marcher-bevy
```

`MM_SCREENSHOT_DELAY_SECS` (default 5) is the wait before capturing. This
matters more than it sounds: an entity whose material's render pipeline
hasn't finished compiling yet is simply skipped for that frame (not an
error — just absent), and on a software Vulkan fallback, compiling this
shader can itself take minutes, so a screenshot taken too early just shows
the window's plain clear color with no indication anything is wrong. Raise
the delay if you're on a software renderer.

## Web (WebGPU)

Build for `wasm32-unknown-unknown` with the `web` feature (enables
`bevy/webgpu`):

```
rustup target add wasm32-unknown-unknown          # one-time
cargo install wasm-bindgen-cli --version <X>       # match the locked
                                                    # `wasm-bindgen` version
                                                    # in rust/Cargo.lock
cargo build -p marble-marcher-bevy --target wasm32-unknown-unknown --features web
wasm-bindgen --no-typescript --target web \
  --out-dir rust/web --out-name marble-marcher \
  rust/target/wasm32-unknown-unknown/debug/marble-marcher-bevy.wasm
cp -r rust/app/assets rust/web/assets   # image assets (e.g. the marble's
                                         # cubemap texture) -- Bevy's
                                         # AssetPlugin fetches these via HTTP
                                         # relative to the served page on
                                         # wasm, they are not bundled into
                                         # the wasm binary by cargo; native
                                         # builds find `rust/app/assets`
                                         # directly and don't need this copy
```

Then serve `rust/web/` (it has a hand-written `index.html` loader) and open
it in a WebGPU-capable browser:

```
cd rust/web && python3 -m http.server 8080
```

**The page must be served from a secure context** — `https://`,
`http://localhost`, or `http://127.0.0.1` — or `navigator.gpu`/
`canvas.getContext("webgpu")` will not behave as a real WebGPU context and
the app panics on startup (`wgpu`'s `backend::webgpu::create_surface_from_context`,
"canvas context is not a GPUCanvasContext"). A plain LAN/VM IP address
(e.g. testing from WSL2, where the browser runs on the Windows host but the
server runs in the Linux guest) does **not** count as secure, even though
the port is otherwise reachable and the page loads — this looks like a
networking problem but isn't; forward the port to the *browser host's own*
loopback address instead of navigating to the server's LAN IP.
