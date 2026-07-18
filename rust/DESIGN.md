# Design: Rust/Bevy port of the CSG ray-marched renderer

This document is the source of truth for the port. It records every decision
and every C++ ↔ Rust correspondence. If you are implementing a milestone
(see `MILESTONES.md`), read this fully first. When this doc and the C++
disagree without an explicit note here, the C++ (`src/fractals/*.hpp`,
`src/Scene.cpp`) is the behavioral reference.

## 1. Goals and constraints

- Port the CSG fractal framework (`src/fractals/`) so ONE tree definition
  drives (a) CPU distance/nearest-point queries for physics and (b) GPU
  ray marching via generated WGSL. Single source of truth — no hand-kept
  duplicate of the distance estimator.
- Runs natively and in the browser via **WebGPU** (`--features web`,
  `wasm32-unknown-unknown`). No WebGL2 fallback required.
- Physics on the **CPU** (GPU readback has ≥1 frame latency in browsers;
  collision needs a handful of DE evaluations per frame — trivial on CPU).
- Renderer v1 is a simple sphere-tracing fragment shader; architecture must
  leave room to grow toward MMCE's quality (shadows, fog, glow, GI) later.
- Structural tree changes regenerate the shader (async pipeline compile,
  swap when ready). Parameter changes are buffer writes only — no recompile.

## 2. Workspace layout

```
rust/
  csg/   marble-csg          — pure logic, deps: glam 0.29 (+ naga 24 dev-dep)
  app/   marble-marcher-bevy — Bevy 0.16 integration (thin)
```

`marble-csg` must not depend on bevy: fast `cargo test -p marble-csg`
cycles, and glam 0.29 is exactly what bevy 0.16 re-exports as `bevy::math`,
so types unify across the crates.

## 3. Parameters (`csg/src/lib.rs` — DONE)

C++ `GLSLVariable<T>` (named GL uniforms) becomes a **slot table**:
`Params` holds `Vec<Vec4>`, uploaded verbatim as one read-only storage
buffer. Typed handles (`ScalarParam`, `Vec3Param`, `Mat2Param`, `IntParam`)
index it. Each `*Value` enum is `Const(T)` (bakes a WGSL literal, like
`GLSLConstant`) or `Param(handle)` (emits `params[i].x` / `.xyz` / a
`mat2x2` constructor, like `GLSLUniform`).

Packing conventions (already implemented, do not change):
- scalar → slot `.x`; int → slot `.x` as f32, WGSL reads `i32(params[i].x)`
- vec3 → slot `.xyz`
- mat2 → column-major `(c0x, c0y, c1x, c1y)`; WGSL
  `mat2x2<f32>(v.x, v.y, v.z, v.w)` reconstructs the same matrix.
- f32 WGSL literals via `format!("{v:?}")` (always has `.` or exponent).

## 4. The tree: `Fold` and `Object` enums (M2)

Closed enum sets (not trait objects): pattern matching, serde-ability later,
and codegen/CPU-eval live next to each other per variant.

```rust
pub enum Fold {
    Abs,
    Menger,
    Rotate { axis: Axis, mat: Mat2Value },
    ScaleTranslate { scale: ScalarValue, shift: Vec3Value },
    Plane { normal: Vec3Value, offset: ScalarValue },
    Modulo { axis: Axis, modulus: ScalarValue },
    Series(Vec<Fold>),
    Repeat { count: IntValue, inner: Box<Fold> },
    OrbitInit(Vec3Value),
    OrbitMax(Vec3Value),
}

pub enum Object {
    Sphere { radius: ScalarValue },
    Cuboid { half_extent: Vec3Value },       // C++ ObjectBox ("Box" is the Rust pointer type)
    Fractal { fold: Fold, base: Box<Object> },
    Union(Box<Object>, Box<Object>),         // C++ ObjectClosest
    Intersect(Box<Object>, Box<Object>),
    Difference(Box<Object>, Box<Object>),
}
```

CPU API (signatures fixed; M2 implements):

```rust
impl Fold {
    pub fn fold(&self, p: &mut Vec4, params: &Params);
    pub fn fold_with_history(&self, p: &mut Vec4, hist: &mut Vec<Vec4>, params: &Params);
    pub fn unfold(&self, hist: &mut Vec<Vec4>, n: &mut Vec3, params: &Params);
}
impl Object {
    pub fn de(&self, p: Vec4, params: &Params) -> f32;
    pub fn nearest_point(&self, p: Vec4, params: &Params) -> Vec3;
}
```

Semantics, ported 1:1 from C++ (file map at bottom):

- Points are `Vec4`: xyz position, `w` = accumulated scale divisor. Every
  primitive DE divides by `p.w` (`(length(p.xyz) - r) / p.w`, etc.).
  Callers pass `p.w = 1.0`.
- **Fold history contract**: `Abs`, `Menger`, `Plane`, `Modulo` push the
  *pre-fold* `p` onto `hist` in `fold_with_history`; their `unfold` pops it.
  `Rotate`, `ScaleTranslate` push nothing (their unfold is closed-form).
  `Series` folds in order / unfolds in **reverse** order. `Repeat` runs
  `count` iterations both ways (inner unfolds pop from the back, so forward
  iteration order in `unfold` is correct — matches C++ `FoldRepeat`).
  `OrbitInit`/`OrbitMax` are no-ops on the CPU (color-pass only).
- `Fractal::nearest_point`: fold with history → `base.nearest_point` →
  `fold.unfold` → assert history is empty (`debug_assert!`).
- **Rotation convention**: the mat2 acts on the component pair as a
  column-vector multiply `v' = M v`. For angle `a`,
  `M = Mat2::from_cols(Vec2::new(cos a, -sin a), Vec2::new(sin a, cos a))`
  which gives `x' = c·x + s·y, y' = −s·x + c·y` — identical to the original
  hard-coded `rotZ` in MMCE. Unfold uses `M.transpose()`.
  Axis → component pairs are **cyclic**: X→(y,z), Y→(z,x), Z→(x,y).
  ⚠ Known C++ inconsistency: `FoldRotate` for axis Y used (z,x) on CPU but
  `p.xz` in GLSL. We use cyclic (z,x) on BOTH sides. (No current scene
  rotates around Y, so no behavior change.)
- **Modulo** uses euclidean fmod (result ≥ 0):
  `fold: p[a] = |fmodulo(p[a] − m/2, m) − m/2|`;
  `unfold: let r = fmodulo(p[a] − m/2, m) − m/2; if r < 0 { n[a] = −n[a]; } n[a] += p[a] − r;`
- **Menger fold** (and its unfold swap sequence) ports verbatim from
  `FoldMenger.hpp`.
- `Union`/`Intersect`/`Difference` pick min / max / max(left, −right); their
  `nearest_point` evaluates both DEs and recurses into the winning side
  (Difference: right side wins when `−de_right > de_left`; its nearest point
  is on the right object's surface).

## 5. WGSL codegen (M3) — `csg/src/codegen.rs`

`CodeWriter { out: String, indent: usize, color_pass: bool, next_id: u32 }`
with `fresh(&mut self, base) -> String` returning `{base}_{id}`.
⚠ Fixes two C++ bugs — do not replicate them: `FoldRepeat` used a
function-`static` depth counter (names leak across generations), and the
combiners used fixed local names (`original_p_union` — nested unions
collide). Every generated local/loop variable gets a `fresh` name.

Emission per variant (`p`, `d`, `orbit` are `var`s in the generated fn):

| Node | WGSL |
|---|---|
| Abs | `p = vec4<f32>(abs(p.xyz), p.w);` |
| Menger | `menger_fold(&p);` |
| Rotate | `p = rot_xy(p, M);` (axis X→`rot_yz`, Y→`rot_zx`, Z→`rot_xy`) |
| ScaleTranslate | `p = p * S;` then `p = vec4<f32>(p.xyz + T, p.w);` |
| Plane | `p = plane_fold(p, N, O);` |
| Modulo (axis x) | `p.x = mod_fold(p.x, M);` |
| Repeat | `for (var it_N: i32 = 0; it_N < C; it_N++) { … }` |
| OrbitInit | *(color pass only)* `orbit = V;` |
| OrbitMax | *(color pass only)* `orbit = max(orbit, p.xyz * V);` |
| Sphere | `d = de_sphere(p, R);` |
| Cuboid | `d = de_box(p, B);` |
| Union | save `p`/`d`(/`orbit`) to fresh `let`s, emit left, save left `d`, restore `p`, emit right, `if (dl_N < d) { d = dl_N; orbit = ol_N; }` |
| Intersect | same shape, `if (dl_N > d)` |
| Difference | same shape, right side then `d = -d;` before `if (dl_N > d)` |

Careful with Union/Intersect/Difference: the saved-`p` `let` must be taken
*before* emitting the left child, and left's `d` saved *after* it. Orbit
save/restore only when `color_pass`.

WGSL constraints (naga will reject violations):
- **No swizzle assignment** (`p.xy = …` is invalid; `p.x = …` is fine) —
  hence the `rot_*`/vector-rebuild patterns above.
- No `1.0/0.0`: initialize `d` to `1e20`.
- Function params are immutable: generated fns start `var p = p_in;`.

Generated functions:

```wgsl
fn de_scene(p_in: vec4<f32>) -> f32 {
    var p = p_in;
    var d = 1e20;
    …tree (color_pass=false)…
    return d;
}
fn col_scene(p_in: vec4<f32>) -> vec4<f32> {
    var p = p_in;
    var d = 1e20;
    var orbit = vec3<f32>(0.0);
    …tree (color_pass=true)…
    return vec4<f32>(orbit, d);
}
```

Static helper library (a `const &str` in codegen.rs): `de_sphere`, `de_box`
(port from `game_folder/shaders/compute/utility/distance_estimators.glsl`),
`menger_fold(p: ptr<function, vec4<f32>>)`, `plane_fold`, `mod_fold` +
euclidean `fmodulo`, `rot_xy`/`rot_yz`/`rot_zx`.

Public API:
- `generate_library() -> String` — bindings decl + helpers + nothing else
- `generate_scene_functions(obj: &Object) -> String` — `de_scene` + `col_scene`
- `generate_shader(obj: &Object) -> String` — full fragment shader for the app
  (import line + bindings + library + scene fns + marcher + `fragment` entry)

Bindings (verified against bevy_sprite 0.16.1 — material is **bind group 2**):

```wgsl
#import bevy_sprite::mesh2d_vertex_output::VertexOutput

@group(2) @binding(0) var<uniform> scene: SceneUniforms;
@group(2) @binding(1) var<storage, read> params: array<vec4<f32>>;
```

`SceneUniforms` (must match the Rust `ShaderType` struct in the app crate,
field-for-field, all `vec4<f32>`):

```wgsl
struct SceneUniforms {
    cam_pos: vec4<f32>,     // xyz eye position
    cam_right: vec4<f32>,   // xyz unit right
    cam_up: vec4<f32>,      // xyz unit up
    cam_forward: vec4<f32>, // xyz unit forward, w = focal length (1/tan(fov/2))
    marble: vec4<f32>,      // xyz center, w radius (r<=0 → hidden)
    sun: vec4<f32>,         // xyz unit direction toward sun
    sun_col: vec4<f32>,     // rgb
    bg_col: vec4<f32>,      // rgb
    misc: vec4<f32>,        // x aspect (w/h), y time seconds, z/w reserved
}
```

Marcher/shading in the generated shader (v1 — keep simple, it will be
replaced when we chase MMCE quality):
- Ray from `VertexOutput.uv`: `ndc = vec2(uv.x*2-1, 1-uv.y*2)`,
  `rd = normalize(right*ndc.x*aspect + up*ndc.y + forward*focal)`.
- Sphere tracing: up to 256 steps, `t += d`, hit when `d < 2e-4 * max(t, 0.05)`,
  miss when `t > 30.0`. Track iteration count.
- Normal: central differences of `de_scene`, step `1e-4 * max(t, 0.05)`.
- Marble: analytic ray-sphere intersection; if closer than the fractal hit,
  shade as glossy: `mix(bg tint, sky(reflect(rd,n)), 0.04 + 0.96*fresnel)`
  (fresnel weights the *reflection* — more reflective at grazing angles).
- Fractal shading: `base = clamp(col_scene(hit).rgb, vec3(0), vec3(1))`;
  diffuse `max(dot(n, sun), 0)`; one shadow ray (march toward sun from
  `hit + n*2*eps`, soft factor `min(1, 8*min_ratio)`); ambient
  `0.3 + 0.4*max(dot(n, up), 0)` (clamped — a negative ambient can push the
  color negative and `pow` a negative base to NaN in the tonemapper); AO from
  iteration count
  `1 - iters/256`; combine, then fog `mix(color, bg, smoothstep(0, 30, t))`.
- Sky: vertical gradient of `bg_col` + sun disc glow. Tonemap: `x/(1+x)`,
  gamma 1/2.2.

Testing (M3): dev-dep `naga = { version = "24", features = ["wgsl-in"] }`.
Parse + validate (`naga::front::wgsl::parse_str` then
`naga::valid::Validator::validate`) the generated shader for the demo scene
and for adversarial trees (nested Unions, nested Repeats — the C++ bugs).
For the full-shader test, replace the `#import …VertexOutput` line with the
real struct (verified from bevy_sprite 0.16.1):

```wgsl
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}
```

Also: golden-ish assertions that generated code for a known small tree
contains expected fragments, and that two successive generations of the same
tree are byte-identical (no leaking name counter).

## 6. Demo scene (M2, `csg/src/scenes.rs`)

Reproduce `Scene::GetInitialObject()` (src/Scene.cpp): the classic Marble
Marcher fractal **Union**'d with the "creme repeating spheres in a sphere".

```
classic(params) -> (Object, ClassicHandles):
  Fractal {
    fold: Series [
      OrbitInit(Const(0,0,0)),
      Repeat { count: Param(iters), inner: Series [
        Abs,
        Rotate { Z, Param(rot1) },
        Menger,
        Rotate { X, Param(rot2) },
        ScaleTranslate { Param(scale), Param(shift) },
        OrbitMax(Param(color)),
      ]},
    ],
    base: Cuboid { Const(6,6,6) },
  }

creme_spheres() -> Object:            # StaticFractals.hpp
  Intersect(
    Fractal {
      fold: Series [ OrbitInit(Const(0.90, 0.80, 0.56)),
                     Modulo{X, Const(0.75)}, Modulo{Y, Const(0.75)}, Modulo{Z, Const(0.75)} ],
      base: Sphere { Const(0.1) },
    },
    Sphere { Const(6.0) })

demo_scene(params) = Union(classic, creme_spheres)
```

`ClassicHandles { scale: ScalarParam, rot1: Mat2Param, rot2: Mat2Param,
shift: Vec3Param, color: Vec3Param, iters: IntParam }` plus a
`set_fractal_params(&mut Params, &ClassicHandles, scale, ang1, ang2, shift, color, iters)`
helper that builds the two rotation matrices from angles (convention in §4).

Level values for the demo — "Beware Of Bumps" (extracted from the binary
`.lvl`): iters=16, scale=1.66, ang1=1.52, ang2=0.19,
shift=(−3.83, −1.94, −1.09), color=(0.42, 0.38, 0.19), marble rad=0.02,
start=(0.681, 2.8, 2.528), kill_y=−4.0, orbit_dist=3.1,
sun dir=normalize(0.637, 0.771, 0.017), sun col=(1.0, 0.95, 0.8),
bg=(0.6, 0.8, 1.0).

## 7. Physics (M5, `csg/src/physics.rs` + app systems)

Pure-logic module in marble-csg (glam only), driven by a Bevy fixed-update
system in the app. Direct port of `Scene::UpdateMarble`/`MarbleCollision`
(src/Scene.cpp — use the ORIGINAL upstream logic; ignore the experimental
zero-gravity/camera-fly edits on this branch, i.e. gravity stays on and the
kill plane stays on).

Frame-rate-locked constants (per 60 Hz tick, from Scene.cpp/Level.cpp):
`ground_force=0.008, air_force=0.004, ground_friction=0.99,
air_friction=0.995, gravity=0.005, ground_ratio=1.15, marble_bounce=1.2`.
Run physics in a fixed 60 Hz loop (`Time<Fixed>`), forces scaled by
`marble_rad` as in C++: `f = marble_rad * (on_ground ? ground_force : air_force)`.

Per tick: `vel.y -= gravity/steps` (C++ uses num_phys_steps sub-steps; use
steps=1 to start) → collision → input force (camera-yaw-relative:
`v = (dx·cos−dy·sin, 0, −dy·cos−dx·sin)`) → friction → `pos += vel` →
kill-plane respawn (`pos.y < kill_y` → reset to start, zero vel).

`MarbleCollision` exact port (Scene.cpp:1075, minus debug prints):

```
de = obj.de(vec4(pos, 1))
if de >= rad { return on_ground = de < rad * ground_ratio }
if de < rad * 0.001 { crushed → respawn; return false }
np = obj.nearest_point(vec4(pos, 1))
d = np - pos; dn = normalize(d); dv = dot(vel, dn)
pos -= dn * rad - d          // = np - dn*rad
vel -= dn * (dv * marble_bounce)
return true
```

**Collider abstraction** (forward-looking, cheap now): collision queries take
a `&[SamplePoint]` (point + radius) rather than hard-coding one sphere. The
marble is the 1-element case. This is the hook for future CSG-vs-CSG
collision (point-shell sampling of one body against the other's DE) without
reworking the physics API.

Camera: orbit around the marble. Yaw (drag / arrow keys), pitch clamped to
(−π/2, π/2), zoom (wheel) around `orbit_dist·marble_rad/0.035` scale. Eye =
`marble_pos + R(yaw,pitch) · (0, 0, orbit)`; basis vectors passed straight
to `SceneUniforms`.

## 8. App wiring (M4, `app/src/`)

- `MarcherMaterial`: `#[derive(Asset, TypePath, AsBindGroup, Clone)]` with
  `#[uniform(0)] scene: SceneUniforms` (a `ShaderType` struct of 9 Vec4s,
  §5) and `#[storage(1, read_only)] params: Vec<Vec4>`.
- `impl Material2d for MarcherMaterial { fn fragment_shader() -> ShaderRef
  { MARCHER_SHADER_HANDLE.into() } }` where
  `const MARCHER_SHADER_HANDLE: Handle<Shader> = weak_handle!("<uuid4>");`
  (macro exists in bevy_asset 0.16.1). A startup system generates WGSL via
  `marble_csg::codegen::generate_shader` and does
  `shaders.insert(MARCHER_SHADER_HANDLE.id(), Shader::from_wgsl(source, "generated://marcher.wgsl"))`.
  Regenerating later = same insert (asset modification re-specializes the
  pipeline; async compile means the old pipeline draws until the new one is
  ready — good enough for v1).
- Fullscreen: `Camera2d` + `Mesh2d(Rectangle)` + `MeshMaterial2d`, transform
  scale synced to window size every frame (cheap, robust on resize).
- Per-frame system: copy camera basis, marble state, time into
  `materials.get_mut(&handle)` (`scene` uniform + `params` from
  `Params::slots()`).
- Keep bevy features minimal (already set in app/Cargo.toml). Web:
  `--features web` → `bevy/webgpu`.

## 9. C++ → Rust file map

| C++ (behavioral reference) | Rust |
|---|---|
| fractals/GLSLVariable.hpp | csg/src/lib.rs (`Params`, `*Value`) — DONE |
| fractals/GLSLBase.hpp, GLSLCodeFactory.hpp | csg/src/codegen.rs (M3) |
| fractals/Fold*.hpp, Orbit*.hpp | csg/src/fold.rs (M2) |
| fractals/Object*.hpp, Fractal.hpp | csg/src/object.rs (M2) |
| fractals/StaticFractals.hpp, Scene::GetInitialObject | csg/src/scenes.rs (M2) |
| Scene.cpp DE/NP/MarbleCollision/UpdateMarble | csg/src/physics.rs (M5) |
| Shaders.cpp preprocessing / `#here` | superseded by codegen.rs templates |
| utility/distance_estimators.glsl helpers | WGSL library const in codegen.rs (M3) |

## 10. Deliberate deviations from the C++

1. Named GL uniforms → param slot table (§3).
2. Fresh-name codegen counter; fixes FoldRepeat static-depth bug and
   nested-combiner name collisions (§5).
3. `FoldRotate` axis-Y component pair: cyclic (z,x) on both CPU and GPU (§4).
4. Enums instead of virtual class hierarchy.
5. GLSL → WGSL; `#here` preprocessor directive → whole-shader generation.
6. This branch's experimental Scene.cpp gameplay edits (gravity=0, disabled
   kill plane, fly-camera movement) are NOT ported; we use upstream marble
   behavior.
