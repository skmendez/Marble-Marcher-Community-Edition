# Milestones

Build-out plan for the Bevy port. Each milestone is a self-contained chunk
of work with acceptance criteria; read `DESIGN.md` first — it fixes the
APIs and semantics. Milestones are executed by implementation agents;
sequencing matters (M3 depends on M2; M4 on M3; M5 on M2+M4).

## M1 — Scaffold (DONE)

Workspace (`csg` + `app`), `Params`/`*Value` slot table in
`csg/src/lib.rs` with passing unit tests, design docs.

## M2 — CSG core: `fold.rs`, `object.rs`, `scenes.rs`

- Implement the `Fold`/`Object` enums and CPU eval exactly per DESIGN.md §4,
  porting from `src/fractals/*.hpp`.
- `scenes.rs` per §6 (`classic`, `creme_spheres`, `demo_scene`,
  `ClassicHandles`, `set_fractal_params`, Beware-Of-Bumps constants).
- Re-export `Fold`/`Object` from lib.rs.
- Tests (all in `#[cfg(test)]`, must pass with `cargo test -p marble-csg`):
  sphere/cuboid DE closed-form checks (inside/outside/scaled-w);
  menger fold produces x≥y≥z ordering; nearest_point of cuboid clamps;
  Union/Intersect/Difference pick correct side for DE and NP;
  modulo fold periodicity; rotate fold/unfold roundtrip (fold then unfold
  of a normal returns original for pure-rotation tree);
  demo scene: DE at marble start (0.681, 2.8, 2.528) is positive and finite,
  DE far away (0,50,0) is large; NP: for several probe points p,
  `de(NP-surface sanity)` — |p − np| within ~2× of de(p) and
  fold history fully consumed (the debug_assert covers it).

## M3 — WGSL codegen: `codegen.rs`

- Per DESIGN.md §5: CodeWriter, per-variant emission, helper library,
  `generate_library` / `generate_scene_functions` / `generate_shader`.
- Tests: naga 24 parse+validate of the full generated shader for the demo
  scene (with the VertexOutput struct substituted for the #import line);
  nested-Union and nested-Repeat trees validate (regression tests for the
  two C++ codegen bugs); repeated generation is byte-identical; snippet
  assertions for a small tree.

## M4 — Bevy app: render the scene (DONE)

- Per DESIGN.md §8: MarcherMaterial, shader handle + startup generation,
  fullscreen quad, per-frame uniform/param sync, free-orbit camera around
  the scene origin (marble not required yet; set `marble.w = 0`).
- Animate the classic fractal params slightly over time (e.g. ang1 +=
  0.1·sin(t)) via `set_fractal_params` to prove param-only updates work
  without recompiles.
- Deviation from DESIGN.md §8 (bevy_render 0.16's `AsBindGroup` derive has
  no inline-`Vec<Vec4>` storage path): `params` is a
  `Handle<ShaderStorageBuffer>`, written via `set_data` each frame — still a
  pure buffer write, no recompile. See `app/src/render.rs`'s module doc.
- Verified rendering correctly on both native (llvmpipe software Vulkan)
  and in-browser WebGPU (real GPU, via Chrome/CDP) — see M6.

## M5 — Marble physics + game camera (DONE)

- `csg/src/physics.rs` per DESIGN.md §7 (pure logic + tests: marble falls
  under gravity onto a flat-ish region and comes to rest; crush respawn;
  kill-plane respawn; collision pushes out to |de| ≥ rad).
- App systems: fixed 60 Hz update (`physics_sys.rs`), WASD camera-relative
  control, orbit camera following the marble, marble uniform fed to the
  shader.
- **Found and fixed a real bug during verification, not just a spec gap**:
  the original single-substep-per-tick design (DESIGN.md §7's first draft)
  let the marble tunnel through thin fractal struts and, worse, left a
  resting marble's tangential velocity uncorrected for a whole tick —
  friction alone decays it far too slowly, so it crept sideways off its
  starting ledge and fell through the kill plane within ~100 ticks. Fixed
  by porting the C++'s actual substep structure faithfully (`NUM_PHYS_STEPS
  = 6`, gravity/integration split across substeps with a full collision
  resolution after each one — DESIGN.md §7 updated, `physics.rs` module
  doc explains in detail). Caught by tracing the exact "Beware Of Bumps"
  start scenario tick-by-tick, not by the original (too-strict) test
  assertions, which had to be corrected too.

## M6 — Web build + polish (DONE)

- `rustup target add wasm32-unknown-unknown`;
  `cargo build -p marble-marcher-bevy --target wasm32-unknown-unknown --features web`
  compiles clean; `rust/README.md` has native + web (wasm-bindgen) run
  instructions; `rust/web/index.html` loader.
- Verified rendering for real: connected to a Windows-side Chrome instance
  over CDP (`--remote-debugging-port`) from this Linux dev environment and
  screenshotted the live page — confirms the WebGPU path end-to-end on
  real GPU hardware, not just "compiles."
- **Real gotcha, not a networking bug**: WebGPU requires a *secure
  context* (`https://`, `localhost`, or `127.0.0.1`); serving the page from
  a plain LAN/VM IP causes `canvas.getContext("webgpu")` to misbehave and
  `wgpu` panics on startup even though the page loads fine and the port is
  reachable. Documented in `rust/README.md`'s web section so this doesn't
  get re-diagnosed as a connectivity problem next time.
- Added an opt-in `MM_SCREENSHOT`/`MM_SCREENSHOT_DELAY_SECS` debug-only
  capture-and-exit mode (`app/src/debug_screenshot.rs`) for headless/CI
  verification — zero-cost (zero systems added) when unset. The delay
  matters: an entity whose pipeline hasn't finished compiling is simply
  skipped for that frame (no error), and a software Vulkan fallback can
  take minutes to compile this shader, so an early capture just shows the
  clear color with nothing indicating a problem.

## Later (unscheduled ambitions)

- Renderer quality: soft shadows/AO from MMCE's shading pipeline, fog,
  glow, temporal effects; eventually evaluate multi-pass compute (MRRM,
  PTGI) on WebGPU.
- Interpreter fallback tier for instant structural edits (see conversation
  notes: compiled-with-async-swap stays the steady state).
- CSG-vs-CSG collision via point-shell sampling on the `SamplePoint` hook.
- Serde for trees; level loading.
