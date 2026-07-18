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

## M4 — Bevy app: render the scene

- Per DESIGN.md §8: MarcherMaterial, shader handle + startup generation,
  fullscreen quad, per-frame uniform/param sync, free-orbit camera around
  the scene origin (marble not required yet; set `marble.w = 0`).
- Animate the classic fractal params slightly over time (e.g. ang1 +=
  0.1·sin(t)) via `set_fractal_params` to prove param-only updates work
  without recompiles.
- Acceptance: `cargo build -p marble-marcher-bevy` succeeds; `cargo run`
  opens a window and renders the fractal (needs a GPU — in a headless
  environment, compile + shader-validation tests are the bar).

## M5 — Marble physics + game camera

- `csg/src/physics.rs` per DESIGN.md §7 (pure logic + tests: marble falls
  under gravity onto a flat-ish region and comes to rest; crush respawn;
  kill-plane respawn; collision pushes out to |de| ≥ rad).
- App systems: fixed 60 Hz update, WASD camera-relative control, orbit
  camera following the marble, marble uniform fed to the shader.

## M6 — Web build + polish

- `rustup target add wasm32-unknown-unknown`;
  `cargo check -p marble-marcher-bevy --target wasm32-unknown-unknown --features web`
  compiles clean; add `rust/README.md` with native + web (trunk or
  wasm-bindgen) run instructions; `rust/web/index.html` loader.
- Fix anything wasm-incompatible (no threads assumptions, etc.).

## Later (unscheduled ambitions)

- Renderer quality: soft shadows/AO from MMCE's shading pipeline, fog,
  glow, temporal effects; eventually evaluate multi-pass compute (MRRM,
  PTGI) on WebGPU.
- Interpreter fallback tier for instant structural edits (see conversation
  notes: compiled-with-async-swap stays the steady state).
- CSG-vs-CSG collision via point-shell sampling on the `SamplePoint` hook.
- Serde for trees; level loading.
