# Noise SDFs — brainstorm

Design exploration for adding random-noise-driven geometry to the CSG
framework (`rust/csg/`): both **standalone** noise objects (terrain, blobby
volumes) and **modifiers** that perturb an existing `Object` (surface
displacement, domain warping). Nothing here is implemented yet; this is the
option space, the constraints the existing architecture imposes, and a
recommended path.

## 1. What the architecture demands of any noise node

Every `Object`/`Fold` node in this crate is *four* things at once, and a
noise node has to be all four too:

1. **A CPU distance estimate** — `Object::de` drives marble collision
   (`physics::collide` calls `de` per sample per substep). A noise DE that
   overestimates distance lets the marble tunnel into geometry.
2. **A CPU nearest-point query** — `physics::collide` calls
   `nearest_point_scratch` on every contacting sample to build the push-out
   vector and bounce normal. The demo marble radius is **0.02**, so "just
   return the base object's nearest point" is not accurate enough for any
   noise amplitude worth having.
3. **WGSL codegen** — `codegen::emit_object`/`emit_fold` must emit matching
   shader code, and the coarse/shadow passes want a *cheaper* variant (the
   existing `iteration_divisor` mechanism).
4. **A serializable, parameterized tree node** — tag-prefixed
   `encode`/`decode_at` (next free `Object` tag: 6; next free `Fold` tag:
   10), `handles_valid_for`, and parameters as `ScalarValue`/`Vec3Value`/
   `IntValue` so they live in the `Params` slot table and can be driven by
   tick-deterministic `Expr` animations and synced to multiplayer joiners.

Plus two cross-cutting constraints:

- **Soundness of `bounding_sphere`**: bounds may be loose but never
  under-approximate (they feed the ray-clip pre-test). Any noise
  perturbation with max amplitude `A` can be handled by padding the child
  bound's radius by `A` — easy, but it must not be forgotten.
- **Determinism**: physics is CPU-only but must replay identically through
  rollback resimulation, and ideally identically *across peers on different
  platforms*. That argues for noise built from **integer hashing + pure
  polynomial interpolation** (u32 arithmetic and mul/add are IEEE-exact and
  identical in Rust and WGSL), and against `sin`-based or libm-dependent
  noise, whose last-bit behavior varies by platform/driver. GPU-vs-CPU
  bit-parity is *not* required (rendering is cosmetic; physics never reads
  the GPU), but the closer they match, the less the marble visibly floats
  or sinks — hash-based noise gives near-parity for free.

## 2. The core correctness problem: Lipschitz bounds

Sphere tracing and `physics::collide` both assume `de(p)` never exceeds the
true distance to the surface. Raw noise breaks this: `fbm(p) - iso` is a
*density*, not a distance — its gradient magnitude (Lipschitz constant `L`)
can be ≫ 1, so the zero-crossing can be much closer than the value suggests.

The fix is the standard one: **divide by a bound on the gradient**.

- One octave of value noise with frequency `f` and amplitude 1 has
  `|∇n| ≤ K·f` for a constant `K` determined by the interpolant (for the
  quintic fade, `K ≈ 1.875` per axis; a safe conservative choice is
  `K = 3`, tightened later by measurement).
- fBm with `k` octaves, lacunarity 2, gain 0.5 has
  `L ≤ K·f·Σ(2·0.5)^i = K·f·k` — octaves stop helping smoothness, so
  bounding by `K·f·k` (or the exact geometric sum for other gain choices)
  is simple and sound.

Each proposal below says where this factor goes. The over-relaxed marcher's
backtracking (`MARCH_CORE`) tolerates *mild* overestimates, but physics has
no such safety net — the divisor is load-bearing, not a rendering nicety.

## 3. The noise primitive itself (shared by everything below)

One shared building block, implemented twice with identical algorithms:

- `csg/src/noise.rs` — `fn value_noise(p: Vec3, seed: u32) -> f32`,
  `fn fbm(p: Vec3, octaves: i32, seed: u32) -> f32`, plus a `Vec3`-valued
  variant (three seeds) for domain warping, and *analytic gradients*
  (`fbm_grad`) — the interpolant is polynomial, so gradients are exact and
  cheap, which `nearest_point` will want (§6).
- New entries in `codegen::HELPERS` — `hash_u32`, `value_noise`, `fbm`,
  `fbm_vec3` in WGSL, same integer hash (e.g. PCG-style `u32` mixing),
  same quintic fade `t³(t(6t−15)+10)`.

Design choices:

- **Value noise over Perlin/simplex** for v1: fewest moving parts, easiest
  to make bit-identical CPU/GPU, cheapest per eval (8 hashes + trilinear
  blend). Gradient noise is a drop-in upgrade later if the "plateau at
  lattice points" look bothers us.
- **Seed as an `IntValue`** (or a `Vec3Value` domain offset): per-level
  "randomness" without new machinery, animatable (a tick-driven `Expr` on a
  domain-offset param gives scrolling/boiling noise that replays correctly
  through rollback).
- **Octaves as an `IntValue`** — deliberately mirrors `Fold::Repeat`'s
  count, so the coarse/shadow passes can reduce it (§7).
- A parity test: golden-value tests in Rust, plus a structural codegen test
  like the existing ones; optionally a naga-validation test already exists
  to catch WGSL syntax slips.

## 4. Standalone noise objects

### 4a. `Object::NoiseTerrain` — heightfield

```
d = (p.y − amplitude · fbm(frequency · p.xz…)) / lip / p.w
lip = 1 + amplitude · frequency · K · octaves   (slope bound → distance bound)
```

- Rolling-marble-friendly: an infinite rolling landscape is the single most
  gameplay-relevant noise object for a marble game.
- Unbounded in x/z → `bounding_sphere` returns `None`, exactly like
  `Fold::Modulo`; scenes clip it with `Object::Intersect` (the
  `creme_spheres` precedent) or just accept an unclipped march.
- `nearest_point`: vertical projection is a good starting guess; refine
  with 1–2 Newton steps along the analytic gradient (§6).

### 4b. `Object::NoiseVolume` — iso-surface ("blob caves")

```
d = (iso − fbm(frequency · p.xyz)) / L / p.w      (inside where density > iso)
```

- Gives organic cave/asteroid masses; intersect with a sphere for a bounded
  blob, or use as the *right* operand of `Difference` to eat noisy holes
  out of any existing object — that last form is a "manipulate an existing
  object" capability that falls out **for free from existing CSG combiners**
  with zero new combiner code, and is the cheapest thing to ship first.
- Marching pure fBm everywhere is expensive (§8); most useful intersected
  with something that bounds it.

## 5. Noise as a modifier of an existing object

### 5a. `Object::Displace { base, amplitude, frequency, octaves, seed }`

Surface displacement, the classic IQ construction, as a new single-child
combiner node (like `Fractal` but wrapping a base *after* evaluation):

```
d = (base.de(p) + amplitude · fbm(frequency · p.xyz)) / (1 + amplitude·frequency·K·octaves)
```

- Bumps/ridges/erosion on *any* subtree: a gnarled Menger sponge, a rocky
  sphere planet, a roughened `creme_spheres` field.
- `bounding_sphere`: child bound padded by `amplitude`.
- Far-field skip (both an optimization and a soundness aid): when
  `base.de(p) > amplitude / lip`-ish, return `base.de(p) − amplitude`
  without evaluating noise at all — sound, and makes empty-space marching
  cost the same as the undisplaced scene.
- `nearest_point`: no closed form; Newton projection (§6).
- Codegen is trivial: emit base, then
  `d = (d + A * fbm(F * p.xyz, …)) * INV_LIP;` — note `d` here is already
  `/ p.w`-scaled by the base emission, and the noise term must be scaled by
  `1/p.w` too (displacement is defined in folded space; the same subtlety
  the C++ DEs handle with their `/p.w` convention — needs a deliberate test
  under `ScaleTranslate`).

### 5b. `Fold::DomainWarp { amplitude, frequency, seed }`

Warp the *query point* before whatever comes next — melted, flow-like
distortion rather than surface bumps:

```
p.xyz += amplitude · fbm_vec3(frequency · p.xyz)
p.w   *= 1 + amplitude · frequency · K · 3      // ← the nice part
```

- This slots into the existing `Fold` machinery *exactly* where it belongs,
  and the Lipschitz correction has a beautiful home: the warp map has
  Lipschitz constant `≤ 1 + A·F·K·√3-ish`, and the framework already
  divides every downstream DE by the accumulated `p.w` — so the warp just
  multiplies `p.w`, precisely how `ScaleTranslate` already accounts for
  scale. No changes to any other node.
- Composes with everything: warp a whole fractal, warp only the base inside
  a `Fractal`, put it inside `Repeat` (each iteration warps again —
  probably chaotic, but legal).
- **The hard part is `unfold`.** The warp is not an isometry and has no
  closed-form inverse. Options, in increasing fidelity:
  1. Push the pre-warp `p` to history; `unfold` maps the normal through the
     inverse-transpose Jacobian *approximated as identity* (treat the warp
     locally as a translation, like `Modulo`'s unfold does for position).
     Normals err by O(A·F) — acceptable for small warps.
  2. Same, but apply the analytic Jacobian `I + A·F·∇fbm_vec3` (we have
     analytic gradients) — transpose-multiply the normal, cheap and much
     better.
  3. Sidestep entirely: see §6's "numerical fallback" — may be the honest
     answer for warped geometry.
- Serialization: `Fold` tag 10; `unfold_bounding_sphere` pads by
  `amplitude` and un-scales like `ScaleTranslate`.

### 5c. Rejected/deferred variants

- **Texture-based noise** (sample a 3D texture): breaks CPU physics parity
  and multiplayer scene sync (would need to ship the texture); procedural
  hash gives the same look without any of that.
- **`Fold::Noise` as an orbit-style color-only op**: trivial and safe
  (orbit ops are already GPU-only no-ops on CPU), a nice cheap add-on for
  visual variety, but it's not geometry — worth doing opportunistically,
  not as the goal.

## 6. `nearest_point` for noisy surfaces

The one genuinely new algorithmic requirement. Existing nodes have exact
answers; noise doesn't. Recommended: **Newton/gradient projection**:

```
x = p
repeat 2–3×:  x -= de(x) · normalize(∇de(x))     // ∇ analytic where possible
return x
```

- For `Displace`/terrain, `∇de` is `∇base.de` (available in closed form for
  sphere/cuboid; numerical 4-tap tetrahedral otherwise, mirroring
  `calc_normal`) plus the analytic fbm gradient — cheap and convergent,
  since collision only queries when `de < 0.02`-ish, i.e. already near the
  surface where one or two steps land within a tiny fraction of the marble
  radius.
- This also suggests a longer-term simplification the codebase already
  hints at (`SamplePoint` doc, "future CSG-vs-CSG collision"): a generic
  *fallback* `nearest_point` via gradient projection for any node, with the
  exact per-node answers kept as fast paths. A noise milestone could
  introduce it as a private helper used only by noise nodes and let it
  graduate later.
- Physics-facing test to gate on: marble dropped onto a `Displace`d plane
  and a `NoiseTerrain` must come to rest (`on_ground`, no tunneling, no
  crush) across a grid of start positions and a sweep of
  amplitude/frequency — the noise analog of the existing
  `bounding_sphere_is_sound_against_real_scenes` scan-style test, plus a
  direct soundness scan: sample many points, assert
  `de(p) ≤ true distance` (estimated by fine local search) everywhere.

## 7. Codegen & pass-tiering integration

- New helpers in `HELPERS` (hash / value noise / fbm / fbm_vec3 / fbm_grad
  if the shader ever wants it); emission for the new nodes is a handful of
  `writeln`s using `CodeWriter::fresh` for temporaries.
- **Octave tiering**: reuse the `iteration_divisor` idea — the coarse and
  shadow shaders emit `max(1, octaves / divisor)` octaves. Same rationale,
  same mechanism, works for `Param`-driven octave counts for the same
  reason the `Repeat` divisor had to be an emission-time division. A
  floor of 1 octave (vs `MIN_REPEAT_ITERATIONS = 2`) is fine: one octave
  still reads as "the same terrain, approximately", which is all the
  warm-start/occlusion passes need. Coarse-pass warm-starting stays sound
  automatically (a wrong `t0` only costs steps — `march_scene`'s doc).
- The Lipschitz divisor must use the *fine* octave count even in reduced
  passes (dividing by more than necessary is sound; by less is not) —
  easiest is to just always use the full-count bound.

## 8. Performance reality check

`map()` cost multiplies everywhere: march steps × 4-tap normals ×
`SHADOW_STEPS`. An 8-octave fbm is ~64 hashes per eval — noticeable.
Mitigations, roughly in order of value:

1. Far-field skip on `Displace` (§5a) — makes cost proportional to
   proximity, not scene coverage.
2. Low default octaves (3–4 reads fine for terrain at marble scale).
3. Octave tiering for coarse/shadow passes (§7).
4. The bounding-sphere clip already avoids marching sky pixels.
5. If still hot: cheaper interpolant (cubic fade) for reduced passes only.

## 9. Suggested build order

Mirrors the repo's milestone style; each step lands with tests and is
independently useful:

- **N0 — Zero-code prototype**: build a scene using *existing* nodes only —
  `Difference(sponge, NoiseVolume)` needs N1 first, but
  pseudo-noise-via-folds (a few incommensurate `Modulo`+`Rotate` layers)
  can visually validate appetite for the feature before any new node
  exists. Optional, skippable.
- **N1 — noise core**: `csg/src/noise.rs` + WGSL helpers + golden/parity
  tests. No tree nodes yet.
- **N2 — `Object::NoiseVolume`** (tag 6): smallest node, exercises the full
  four-role checklist (§1) end to end; ship with a demo scene
  (`Difference(menger_sponge, noise_volume)` — "worm-eaten sponge").
- **N3 — `Object::Displace`** (tag 7) + Newton `nearest_point` + the
  physics rest-and-roll test. This is the headline "manipulate an existing
  object" feature.
- **N4 — `Object::NoiseTerrain`** (tag 8) + a rolling-mode terrain scene.
- **N5 — `Fold::DomainWarp`** (Fold tag 10) with the `p.w` Lipschitz trick
  and Jacobian-transpose unfold.
- **N6 — polish**: octave tiering in coarse/shadow shaders, Expr-driven
  animated noise (scrolling offset), orbit-noise coloring.

## 10. Open questions

1. **Gameplay target**: rolling terrain (favors N4 early) vs. weirder
   fractal variety (favors N2/N3/N5)? Ordering above assumes the latter.
2. **Cross-platform peer determinism**: is bit-identical physics across
   different peer platforms a hard requirement? If yes, hash+polynomial
   noise is mandatory (as assumed); if no, gradient/simplex noise opens up
   sooner.
3. **Animated geometry noise**: is boiling/scrolling noise (domain offset
   driven by an `Expr`) in scope? It's cheap given N1 params, but every
   animated-geometry param re-runs through rollback resim — fine in
   principle (the oscillating bite sphere already does this), worth a perf
   glance at higher octave counts.
4. **How much `nearest_point` accuracy is enough?** The Newton scheme's
   error should be validated against the smallest marble (0.02) — if 2
   steps aren't enough at plausible amplitudes, the answer is more steps
   only near-contact, not globally.
