# Noise SDFs — brainstorm (v2: exact unit-gradient noise)

Design exploration for adding random-noise geometry to the CSG framework
(`rust/csg/`), both **standalone** and as a way to **manipulate existing
objects**.

**Requirement (v2):** the noise field must have `|∇d| = 1` almost
everywhere — a genuine distance function, not a density divided by a
Lipschitz bound. An earlier draft of this doc explored fbm-style value
noise with a `1/(1 + A·F·K·octaves)` correction; that approach is
*rejected* by this requirement (it is sound but everywhere-underestimating:
wasted march steps, mushy physics margins, and `|∇d| ≪ 1` at typical
parameters). See §9 for what the fbm path uniquely offered and what we
give up.

## 1. What "gradient 1 almost everywhere" forces

Two mathematical facts pin down the whole design space:

- A function with `|∇f| = 1` a.e. (and 1-Lipschitz) **is** the signed
  distance to some set, locally. So we are not looking for a "noise
  function" at all — we are looking for a **random surface family whose
  exact distance is cheap to evaluate locally**.
- **Eikonal rigidity**: any C² function with `|∇f| = 1` on an open set is
  locally an affine distance function. Smooth random hills with exactly
  unit gradient *do not exist*. Creases (the medial axis, where two
  nearest-surface candidates tie) are unavoidable — but they're measure
  zero, which is exactly the "almost everywhere" in the requirement, and
  the framework already lives happily with creased fields (`min` in
  `Union` creases too).

Consequence: the resulting geometry has a **pebbly / bubbly / faceted**
character (unions of primitives, lattices of planes), not smooth rolling
fBm hills. That is the honest trade of this requirement, and for a marble
game it's arguably the *better* aesthetic anyway (readable contact
geometry).

## 2. The eikonal algebra — and how much of it this crate already has

Operations that **preserve** `|∇| = 1` a.e.:

| op | result | in-crate today |
|---|---|---|
| `min(f, g)` | exact outside, ≥-bound in overlaps, unit-gradient a.e. | `Object::Union` |
| `max(f, g)`, `max(f, −g)` | bound, unit-gradient a.e. | `Intersect` / `Difference` |
| `−f`, `\|f\|` | unit gradient a.e. | (`\|f\|` = shell/onion, not yet a node) |
| `f − c` (offset/inflate) | unit gradient | trivial new node (§6) |
| isometries of the domain | unit gradient | `Fold::Abs`, `Menger`, `Rotate`, `Plane`, `Modulo` — all piecewise isometries |
| uniform scale | handled | `ScaleTranslate` via the `p.w` convention |

Operations that **destroy** it: sums (displacement!), products, smooth-min
(sub-unit gradient throughout the blend band, which has positive measure).

This is the big realization: **the existing `Object`/`Fold` machinery is
already an eikonal-preserving algebra.** `creme_spheres` (`Modulo`³ over a
sphere) is *already* an exact unit-gradient "noise-like" field — a periodic
one. The only missing ingredient is **randomness**, and the only genuinely
new node needed is a random point process with exact distance. Everything
else — manipulating existing objects included — falls out of combiners we
already have.

## 3. The core new node: `Object::Scatter` (hashed Worley / random sphere field)

Distance to a random point process, i.e. Worley F1 recast as an exact SDF:

```
d(p) = min over feature points i of ( |p − c_i| − r_i )
```

- One feature point per lattice cell of size `s`, position jittered inside
  the cell by an integer hash of the cell coordinates (+ a seed), radius
  `r_i` hashed into `[r_min, r_max]`.
- Evaluation looks at the 3×3×3 neighboring cells: 27 hashes + 27
  sphere distances, `min`-reduced. Cost is comparable to ~3 octaves of the
  rejected fbm — and unlike fbm it needs **no transcendentals and no
  interpolation**, so CPU/GPU/cross-platform determinism is even easier
  than the v1 plan (pure u32 mixing + `length`).
- `|∇d| = 1` everywhere except the crease set where two spheres tie —
  exactly the requirement.

Fields/params (all animatable, all in the `Params` table):

```rust
Object::Scatter {
    cell:   ScalarValue,   // lattice cell size s
    r_min:  ScalarValue,
    r_max:  ScalarValue,
    seed:   IntValue,      // also enables per-level randomization
}
```

### Soundness corner case (worth getting right on day one)

With one point per cell and a 3³ search, a query point sitting in a corner
of its cell can theoretically be nearer (in SDF terms) to a sphere just
outside the searched block than to any of the 27 found (worst case
`d_27 ≤ √3·s − r` vs. an unsearched sphere at `> s − r_max`). Two sound
fixes:

1. **Cap**: return `min(d_27, s − r_max)`. The cap is itself sound (every
   unsearched sphere is provably at least that far) and is a *constant*
   only in the rare emptiest-corner regions where it binds — flat gradient
   there, but still a valid underestimate. Cheapest.
2. **5³ search**: the equivalent cap `2s − r_max` then never binds for any
   `r_max < (2−√3)·s ≈ 0.27s`… which is a real constraint on `r_max`. More
   hashes, no flat spots.

Recommendation: 3³ + cap, with `r_max ≤ 0.5s` enforced at construction;
document that the cap region is tiny and empty of surface anyway.

### Exact `nearest_point` — the physics win

This is where the eikonal approach beats fbm decisively: `nearest_point`
is **closed form**. Re-run the same 27-cell scan, keep the argmin sphere,
project: `c + normalize(p − c) · r`. No Newton iteration, no numerical
gradients, no accuracy question against the 0.02-radius marble. The
collision path (`physics::collide`) gets the same exactness guarantees the
hand-written `Sphere`/`Cuboid` cases have today.

### The rest of the §1-checklist for `Scatter`

- **`bounding_sphere`**: `None` — infinite lattice, same story as
  `Fold::Modulo`; scenes clip it with `Intersect` (the `creme_spheres`
  precedent) when they need a finite bound.
- **Serialization**: `Object` tag 6; `handles_valid_for` over its four
  values.
- **Codegen**: a WGSL triple loop over `[-1, 1]³` with `CodeWriter::fresh`
  locals; the hash helper goes into `HELPERS`. No divisor/tiering games
  needed — there are no octaves. (The coarse/shadow passes can share the
  full-fidelity emission; 27 hashes is cheap enough not to tier, and
  `march_scene`'s warm-start tolerates nothing less anyway.)

## 4. Variants on the same skeleton (each still exactly eikonal)

- **Random boxes**: swap the per-feature primitive for `de_box` with a
  hashed half-extent (and optionally a hashed 90°-multiple orientation to
  stay exact without rotation params). Box SDFs are exact ⇒ the min-field
  is unit-gradient a.e. Rubble/debris look instead of pebbles.
- **Mixed primitives per cell**: hash a primitive *kind* per feature.
  Still exact. This is "stochastic CSG": a random union of exact
  primitives is itself an exact object.
- **Multi-scale (fractal) scatter**: `Union(Scatter(s), Scatter(s/3),
  Scatter(s/9))` — `min` preserves the property, so a fractal
  pebbles-on-boulders field is *still* exactly unit-gradient. This is the
  eikonal-legal replacement for fbm octaves. (Sums of octaves are what's
  illegal, not multi-scale itself.)
- **Faceted / crystalline noise, zero new nodes**: `mod_fold` is already
  the exact distance-to-a-plane-lattice triangle wave. `Fractal { Series[
  Rotate(random), Modulo(x)], base }` stacks randomly-oriented plane
  lattices — piecewise isometries all the way down, so any base object
  stays exact. Randomness enters through hashed rotation *params* chosen
  at build time (or `Expr`-driven).
- **Cell walls (F2 − F1)** — the classic Voronoi-crack look: **flag: not
  eikonal.** `(F2 − F1)/2` has `|∇| ≤ 1` with equality only where the two
  feature directions are antiparallel; it's a sound bound but sub-unit
  over positive measure. Include only if the look is wanted, clearly
  labeled as a bound-not-distance node; it does not meet this doc's
  requirement.
- **Jittered-`Modulo` fold** (jitter the cell contents inside the existing
  fold instead of a new object): tempting, **rejected** — the per-cell
  translation is discontinuous at cell walls, so the folded field
  *overestimates* distance to geometry jittered toward a neighboring wall
  (tunneling risk). Neighbor-aware search is unavoidable, which is exactly
  what `Scatter` is.

## 5. Manipulating existing objects = plain CSG (no new combiner nodes)

Because the noise is a first-class exact object, "noise as a modifier"
stops being a new mechanism and becomes composition:

- **Pockmarks / erosion**: `Difference(base, Scatter{…})` — random bites
  wherever the scatter field intersects the body. No placement problem, no
  Lipschitz factor, exact `nearest_point` on both branches. This is the
  cheapest compelling demo (e.g. a worm-eaten `menger_sponge`).
- **Welded bumps**: `Union(base, Scatter{…})` — but a bare union sprinkles
  spheres through all of space, not just near the surface. Fix via §6's
  `Offset`: `Union(base, Intersect(Scatter, Offset(base, ε)))` keeps only
  the spheres within `ε` of the base surface. Barnacles on the sponge.
- **Pebble-ification**: `Intersect(base, Scatter{dense, overlapping})` —
  carves the body *into* the scatter's bubble texture.
- Displacement in the literal `d + noise` sense is **off the table** by
  the requirement (sums break unit gradient; see §2) — and these three
  cover its use cases with strictly better physics.

## 6. Small supporting node: `Object::Offset { base, offset: ScalarValue }`

`d = base.de(p) − offset` (positive inflates). Preserves unit gradient,
three lines per role, `Object` tag 7. Wanted by §5's shell trick and
independently useful (rounded/inflated variants of any object; a `|d| − t`
"onion" sibling is equally trivial if shells become interesting).
`bounding_sphere` pads the child radius by `offset`; `nearest_point` of an
inflated object is the base's nearest point pushed out along the normal —
or, simpler and sound for v1, restrict `Offset` to wrap exact-`de`
subtrees and reuse the base `nearest_point` + `offset·normalize(p − np)`.

## 7. Determinism & randomness plumbing

- All randomness flows from **integer hashes of (cell coords, seed,
  channel)** — u32 mul/xor/shift mixing is bit-identical in Rust and WGSL
  and across platforms; nothing transcendental anywhere in the pipeline.
  This is *stronger* determinism than the v1 fbm plan (which still needed
  polynomial fades).
- `seed` as an `IntValue` param ⇒ per-level variation, multiplayer scene
  sync for free (it rides in the `Params` table), and even animation:
  an `Expr`-driven *domain offset* param (add a `shift: Vec3Value` to
  `Scatter` or just wrap it in `Fractal{ScaleTranslate}`) gives drifting
  pebble fields that replay correctly through rollback. Note animating the
  *seed* teleports geometry discontinuously — offset animation is the
  usable knob.

## 8. Costs and pass integration

- ~27 hashes + 27 `length()` per `de` eval; multiplied as ever by march
  steps × 4-tap normals × `SHADOW_STEPS`. Comparable to the scenes'
  existing 16-iteration fold loops. Multi-scale unions multiply by the
  number of scales — 2–3 scales is the sane budget.
- An early-out helps: cells are scanned nearest-first only with effort;
  simpler is the far-field skip — if the *cap* value (§3) is already ≤ the
  running min, later cells can't win. Probably unnecessary for v1.
- MRRM/shadow passes: no reduced-detail variant needed (no octaves to
  tier); if profiling ever demands one, "scan 1 cell + cap" is the natural
  degraded mode — sound because of the cap, just flatter.

## 9. What fbm offered that this gives up (recorded for honesty)

Smooth rolling terrain and stacked-octave spectral richness require sums,
which the unit-gradient requirement forbids. If smooth hills ever become a
gameplay need, that feature *cannot* meet this requirement — it would come
back as a clearly-labeled bound-not-distance node (heightfield with slope
`≤ 1` enforced, divided by `√2`), coexisting with the eikonal nodes. The
two families compose fine through `min`/`max` (a bound `min` an exact
field is a bound).

## 10. Build order

- **N1** — hash + `Scatter` core in `csg/src/noise.rs` (or straight into
  `object.rs`): `de`, exact `nearest_point`, cap soundness, encode tag 6,
  unit tests incl. a numerical `|∇d| = 1` a.e. probe (sample random
  points, central-difference gradient, assert `≈ 1` away from crease
  detection) and a `de ≤ true distance` scan in the spirit of
  `bounding_sphere_is_sound_against_real_scenes`.
- **N2** — codegen: WGSL hash + scatter emission, structural tests, naga
  validation; demo scene `Difference(menger_sponge, Scatter)` clipped like
  `creme_spheres`.
- **N3** — `Object::Offset` (tag 7) + the `Union(base,
  Intersect(Scatter, Offset(base, ε)))` barnacle construction.
- **N4** — physics test: marble rests/rolls on a scatter field and on a
  pockmarked sponge; crush/tunnel regression sweep over `r_min/r_max/s`.
- **N5** — multi-scale scatter union scene + hashed-rotation faceted
  variant (pure existing folds); `Expr`-driven drifting offset demo.

## 11. Open questions

1. **Look check**: is pebbly/bubbly/faceted the desired aesthetic, or was
   smooth-hills terrain the actual goal? (If the latter, §9 applies and
   the requirement needs renegotiating — the two are mathematically
   exclusive.)
2. **Primitive set for v1 `Scatter`**: spheres only (simplest, fully
   exact `nearest_point`) vs. hashed sphere/box mix?
3. **Cap vs. 5³** (§3): accept rare flat-gradient cap regions, or pay 125
   hashes for none? (Recommendation: cap.)
4. **Inside behavior**: overlapping scatter spheres make `min` an
   underestimate *inside* overlaps (same as `Union` today — fine for
   physics, which only needs outside/near-surface exactness). Any use case
   for marble-inside-noise (caves via `Difference`) should get its own
   test coverage since it exercises the `−de` branch.
