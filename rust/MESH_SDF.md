# Triangle meshes as an `Object` — feasibility analysis

Could `Object::TriMesh` exist, with *every* part of the `Object` API
implemented efficiently? Short answer: **yes — and the CPU half is easy;
the design question is entirely on the GPU side.** The right shape is a
two-representation object: exact BVH queries on the CPU (physics), a
baked signed-distance grid in a `texture_3d` on the GPU (rendering),
with a zero-infrastructure brute-force tier for small meshes. Companion
to `NOISE_SDF.md` / `COMPOSITE_OBJECTS.md` / `COLORING.md`.

## 0. What the API actually demands

An `Object` is four roles plus bookkeeping (every node in `object.rs`
implements all of them):

| Role | Contract | Mesh difficulty |
|---|---|---|
| `de(p)` (CPU) | sound (never overestimates), signed | easy — exact, even |
| `nearest_point(p)` (CPU) | surface point for collision push-out | easy — exact |
| WGSL emission | evaluable ~10²x per pixel per frame | **the hard part** |
| encode/decode | self-delimiting bytes, peer-syncable | easy, size-bounded |
| `bounding_sphere` | sound outer bound | trivial |
| `Params`/animation | scalar slots driven per tick | rigid only (see §6) |

The asymmetry drives the whole design: **physics does a handful of
queries per tick; rendering does millions per frame.** They do not need
the same representation, and nothing in the API says they must — only
that both fields are sound and agree to within the coarser one's error
(the `Morph` precedent: its CPU `nearest_point` is already a Newton
approximation against the exact-on-both-sides `de`).

## 1. CPU: exact signed distance, exactly solved (this part is free)

Point-to-mesh distance is a solved problem with a classic exact
algorithm:

- **BVH over triangles** (built once at construction/decode time,
  ~µs-scale queries for thousands of triangles): traverse
  nearest-first, prune nodes whose AABB distance exceeds the best
  found. Point-triangle distance is a closed-form 10-line kernel.
- **Sign via angle-weighted pseudonormals** (Bærentzen & Aanæs 2005):
  precompute, per face/edge/vertex, the angle-weighted average of
  incident face normals; `sign = sign(dot(p - np, pseudonormal(feature
  hit)))`. **Provably correct for any closed 2-manifold mesh** — the
  edge/vertex cases are exactly what naive `dot(p - np, face_normal)`
  gets wrong.
- `nearest_point` falls out of the same query — the argmin triangle's
  closest point *is* the answer, exact. Better than `Morph` (Newton
  approximation) and `Offset`-of-thin-features (medial-axis caveat)
  already ship with.

So the physics-facing half is not merely feasible — a mesh would have
*higher-quality* CPU fields than some existing composite nodes. Marble
rolls over a statue with exact contact normals.

Determinism (rollback/multiplayer): BVH construction and traversal are
deterministic given deterministic construction order; f32 arithmetic is
IEEE-identical across native and wasm. Same discipline the physics tick
already lives by.

## 2. GPU: three tiers

### Tier 0 — brute force, baked into the shader (works **today**)

The codegen already emits per-scene WGSL with constants inlined and
helper functions in `HELPERS` (`de_torus`). A small mesh is just more
constants: emit a `const` triangle array and a loop-min helper —
`d = de_mesh_<id>(p)`. No new bindings, no pipeline changes, exact
distance, sharp edges preserved.

Budget math: the gears scene's `de` costs ~73 primitive evaluations
plus ~30 folds and renders fine. A point-triangle kernel is a few times
a primitive eval, so **~32–64 triangles is the comfortable ceiling** —
enough for icons, dice, low-poly props, and for proving out the whole
node (serialization, physics, folds) before any infrastructure work.

### Tier 1 — baked SDF grid in a `texture_3d` (the real answer)

Bake the mesh (using §1's exact CPU query) into an N³ signed-distance
grid over its bounding box, upload as a 3D texture, and `de` becomes
one filtered texture sample plus a soundness margin (§4). Cost per
query: **O(1), independent of triangle count** — a 100k-triangle
sculpture costs the same per march step as a sphere.

- Memory: 64³ f16 = 512 KB; 128³ = 4 MB. A per-mesh resolution knob.
- Bake cost: 64³ = 262k exact queries against the BVH — well under a
  second at load, deterministic (serial loop, no parallel reduction
  order), so **peers re-bake identical grids from the synced mesh
  bytes** rather than shipping the grid.
- Fidelity: features smaller than a cell round off; sharp edges round
  at radius ~h. Acceptable for props; not for razor geometry (that is
  what Tier 0 and analytic primitives are for).
- Infrastructure: this is the honest cost of the feature. The fine
  material already hand-implements `AsBindGroup` with two render
  textures, a cubemap + sampler, and a storage buffer, and the
  generated WGSL already declares those bindings (`render.rs`,
  `MARBLE_TEXTURE_BINDING`) — adding `texture_3d<f32>` + sampler
  bindings to all four shader variants follows an existing, well-worn
  path. New but not novel. One binding array (or texture atlas) serves
  multiple meshes.

### Tier 2 — BVH traversal in-shader: **rejected**

Exact and unbounded, and people have done it, but: a stack-based
traversal loop per `de` call × ~100 steps per pixel is the wrong cost
center for a fragment-shader marcher; and it is architecturally alien
to the codegen, which emits straight-line expressions per node. The
grid gives O(1) queries for a controllable, pre-payable error. Not
worth it while Tiers 0–1 cover the range.

## 3. Signedness on the GPU

The pseudonormal trick needs the *feature* argmin plus stored
pseudonormals — heavy in-shader. Three practical options:

1. **Grid tier: sign is free** — bake it (CPU query is signed). This is
   another argument for the grid as the default.
2. **Brute-force tier: go unsigned** and accept a restriction. An
   unsigned distance field is still 1-Lipschitz and sound to march from
   outside, and a watertight solid viewed from outside is
   indistinguishable from its shell. Interior sign only matters to (a)
   CPU physics — which uses §1's exact signed query anyway — and (b)
   CSG uses that *negate or invert* the field: `Difference(x, mesh)`
   and `Intersect` need real sign. So: Tier-0 meshes are valid as
   solids and `Union` members; using one as a cutter is rejected at
   construction (or upgrades it to a grid).
3. Generalized winding numbers (Jacobson et al. 2013) if we ever care
   about non-watertight scans — robust but expensive; out of scope for
   v1, which should simply *validate closedness at construction* (every
   edge shared by exactly two triangles, consistently wound).

## 4. Soundness of the sampled grid (the part worth being pedantic about)

Storing exact distances at grid points does **not** make the trilinear
interpolant sound: the interpolant of a 1-Lipschitz function can
overestimate between samples (error up to ~h·√3/2 near the surface
diagonal). Two remedies, both cheap:

- **Rigorous:** from any corner sample, `d_true(p) ≥ d_corner - |p -
  corner|` (1-Lipschitz cone bound); take the max over the cell's 8
  corners. Sound by construction, needs raw texel loads
  (`textureLoad`) instead of hardware filtering.
- **Pragmatic:** `d = trilinear_sample(p) - h·√3/2`. One hardware
  sample, provably sound (the interpolant is a convex combination of
  corner values, each within `h·√3/2·...` of a cone bound), costs one
  subtract; the constant margin only softens the march near the
  surface, where steps are small anyway. **Recommended.**

Outside the grid's box: for `q = clamp(p, box)`, the projection
inequality on a convex box gives `|p - m|² ≥ |p - q|² + |q - m|²` for
any mesh point `m`, so

```text
de(p) = max(dist_to_box(p), de_grid(q))     -- sound
```

(the tempting `dist_to_box + de_grid` **overestimates** — max, not
sum). This also makes far-field marching exact: rays skip to the box
at full speed, the classic bounded-object behavior every other
primitive has via `bounding_sphere`.

## 5. Serialization and multiplayer

The tree encoding gains one tag: `TriMesh { verts, indices, grid_res }`
— indexed mesh bytes, self-delimiting with the same
count-validation discipline the decoder already applies (Fix 6). A
1000-triangle prop ≈ 500 verts × 12 B + 3000 indices × 2–4 B ≈ **15–18
KB** — a one-time scene-sync payload comparable to a big param table,
entirely reasonable. The grid is *derived state*, re-baked
deterministically on each peer (§2), so it never crosses the wire, and
`COLORING.md` §5's objection to texture *assets* (an unserializable
external reference) does not apply: the mesh **is** tree data.

Content-hash caching (bake once per mesh hash) keeps repeated scene
edits cheap.

## 6. What composes for free (the actual payoff)

Because `de` composes through the existing algebra, a mesh node
inherits the entire system with zero extra code:

- **Folds**: `Rotate`/`ScaleTranslate` animate it rigidly with `Expr`
  determinism; `Modulo` tiles one statue into an infinite field;
  `PolarModulo` makes a ring of 12 of them — all at **one grid sample
  per query** regardless of instance count, because folds map the query
  into the canonical cell. This is where mesh-as-SDF crushes
  mesh-as-rasterized-geometry.
- **CSG**: `Union` with fractals, `Onion` for a hollow shell of the
  mesh, `Offset` to inflate/erode it (grid sign makes both valid),
  `Morph` between two meshes — a convex combination of two sound
  fields is sound, so *mesh morphing is free and correct*.
- **Color**: `OrbitInit` and every pattern op in `COLORING.md` apply
  unchanged.
- Shadows/AO/MRRM coarse pass: all just call `de_scene`; nothing new.

Out of scope, honestly: per-vertex animation (skinning). The `Params`
table is Vec4 slots — thousands of animated vertices don't fit the
model, and re-baking the grid per tick is a non-starter. Rigid folds +
`Morph` between a few baked poses covers a surprising amount of what
a marble game would ever want.

## 7. Verdict and order of work

Feasible, efficiently, for every part of the API — with one honest
infrastructure cost (3D-texture bindings through the four shader
variants) and one honest fidelity trade (grid rounds sub-cell detail;
choose resolution accordingly).

1. **CPU core first**: point-triangle kernel, BVH, pseudonormal sign,
   closedness validation, encode/decode + handle checks. Unit-test
   against analytic shapes (icosphere mesh vs `Sphere`'s exact field —
   the crate's existing test style).
2. **Tier 0** WGSL emission (const-array brute force, ≤64 tris,
   no-cutter restriction) — proves the node end-to-end with zero
   pipeline changes; ship a low-poly-prop scene.
3. **Tier 1** grid bake + `texture_3d` bindings + sound sampling
   (§4's margins, boxed exterior) — lifts the triangle ceiling to
   "any prop", signed everywhere.
4. Later, on demand: winding-number sign for scans, multi-mesh atlas,
   content-hash bake cache.
