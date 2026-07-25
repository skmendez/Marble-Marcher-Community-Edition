# Coloring objects — brainstorm

Where the coloring system could go next. Companion to `NOISE_SDF.md` and
`COMPOSITE_OBJECTS.md`; grounded in what this session actually built and
the walls it hit doing so.

## 0. What exists, and the lessons already paid for

The vocabulary today: `OrbitInit` (set a constant), `OrbitMax`
(componentwise `max(orbit, p_folded * c)`), `OrbitBarberPole` (helical
two-color stripes in toroidal coordinates), plus structural behavior —
CSG combiners pick the winning branch's orbit, `Morph` blends it by `t`.
Everything is color-pass-only (physics never pays), serialized in the
`Fold` tag space, and parameterized via `Params` (live-editable,
`Expr`-animatable).

Lessons that should shape everything below:

- **`OrbitMax` is axis-locked and monotone**: `r<-x, g<-y, b<-z`, max
  accumulation only, and the only periodicity comes from folds — which
  also fold the geometry, so only an object's own symmetries are usable
  (the "4 ribs, not 8" orbit-size lesson). Patterns that aren't functions
  of the folded coordinates need their own ops (the barber-pole lesson).
- **Ops are placement-order-sensitive**: an op before the folds sees true
  world coordinates; after, wedge coordinates. Both are useful; the
  ordering is a real part of a scene's design.
- **Design colors for the *pipeline*, not the raw values**: orbit passes
  through Reinhard compression, material-gamma squaring, lighting, and
  ACES before reaching the screen. Mid-range "tasteful" values crush to
  near-black; any meaningful green content reads yellow under direct sun
  (two rounds of yellow-band reports). Predict display values through the
  whole chain before rendering — the 10-line simulation script from the
  palette round is worth keeping somewhere permanent (see §6).

## 1. New pattern-generator ops (the barber-pole recipe, repeated)

Each of these is the same shape of change as `OrbitBarberPole`: one enum
variant, CPU no-op, one WGSL emission arm, a serialization tag. Ordered
roughly by value-per-effort:

- **`OrbitPalette`** — IQ's cosine palette: `color = a + b*cos(2*PI*(c*s
  + d))` with vec3 `a,b,c,d` and a scalar source `s` (orbit-trap value,
  a coordinate, distance from origin...). One op buys the entire classic
  fractal-coloring aesthetic — smooth, wraparound-safe, infinitely
  tunable gradients — and subsumes most "I want a nicer ramp" requests
  that `OrbitMax` can't express. The strongest single addition on this
  list.
- **`OrbitMin` trap** — sibling of `OrbitMax` accumulating the *minimum*
  (classically: distance to a point/axis per fold iteration). Fractal
  surfaces get thin bright veins where folds pass near the trap instead
  of broad max-washes; pairs beautifully with `OrbitPalette` (trap value
  as the palette's `s`).
- **Noise albedo** (`OrbitNoise`/fbm color) — the color-only half of
  `NOISE_SDF.md`, and the constraint analysis there evaporates for
  color: soundness/Lipschitz/eikonal concerns were all about *geometry*;
  an albedo pattern has no physics implications at all. Hash + value
  noise + fbm in WGSL gives marble veining, wood grain (fbm-warped
  rings), granite speckle — the looks people actually mean by "make it
  prettier" — with zero risk. The `NOISE_SDF.md` N1 work (hash/fbm
  helpers) becomes shippable purely for color even if noise geometry
  never lands.
- **`OrbitChecker` / `OrbitStripes`** — parity of `floor(p/cell)` (3D
  checkerboard) and planar stripes along an arbitrary direction (the
  linear generalization of the barber pole). Cheap, useful for scale
  reference in test scenes if nothing else.
- **Barber-pole `phase` param** — not a new op: add a `phase:
  ScalarValue` to the existing one, and an `Expr` on it makes the stripes
  *rotate* around the ring, deterministic across peers. Probably the
  highest fun-per-line change available (a candy donut whose stripes
  crawl).

## 2. Surface-property-driven color (shader-level, not tree ops)

These read geometry the tree ops can't see, because they're computed in
`MARCHER` where the normal/AO already exist:

- **Slope tinting** — blend toward a second color by `n.y` (up-facing =
  one material, walls = another): the terrain-palette classic, and inside
  tunnels it distinguishes floor/ceiling — the cue the donut lost when
  the `g <- y` term died (it was the right *idea* aimed at the wrong
  mechanism; the normal is per-pixel and lighting-independent, so it
  can't go yellow the way the albedo term did).
- **Cavity/edge tint** — approximate curvature from a few extra DE taps
  (or reuse the AO term's structure): darken concavities toward a "dirt"
  color, lighten sharp convex edges toward a "wear" color. This is the
  single most effective realism trick per instruction in SDF renderers.
- **Colored ambient** — already on the wash-out list (sky-tinted ambient
  instead of flat gray); listed here because it's *the* fix for
  interior scenes reading flat, donut included.

These want a small uniform block (a couple of vec4 lanes or params
slots) rather than per-node serialization — they're global material
behavior, not per-object structure. `misc3.w` is still free.

## 3. Fractal-native depth coloring

- **Iteration-graded color** — an orbit op *inside* a `Repeat` whose
  contribution decays with iteration (e.g. `orbit = mix(orbit, c,
  1/(1+i))`) colors by recursion depth: outer structure one hue, fine
  detail another. Needs the emitted loop to expose its induction variable
  to orbit ops — a small `CodeWriter` extension (the loop var name is
  already `fresh`-generated; pass it down).
- **Trap-value remap** — `OrbitPalette` over `length(orbit)` after the
  existing traps run; no new trap machinery, immediate payoff on the
  Menger scenes.

## 4. Material properties beyond albedo

Orbit is one vec3. The C++ original's `scene_material` returns albedo
*plus* `pbr` (metallic/roughness) *plus* `emission`:

- **Emissive channel** — an `OrbitEmissive` op + an `emission` term added
  after lighting in `MARCHER`. Enables neon/glow looks (and is the
  prerequisite for bloom meaning anything). Modest ABI change: either a
  second accumulator (`var emissive: vec3f`) threaded through `col_scene`
  , or pack a shared exponent scheme into the existing vec4's `w`.
- **Roughness/metallic channels** — only worth it once Cook-Torrance
  specular lands (wash-out tier 2); at that point per-region shininess
  (glazed vs matte donut icing) becomes expressible. Design note: a
  second `col_scene` return vec means every combiner needs a rule for
  blending it (`Union` picks, `Morph` mixes — same shapes as orbit).

## 5. Texture-like approaches (flagged, not endorsed)

Triplanar-mapped real textures are the industry answer to "SDFs are hard
to UV" — sample a texture from three axes, blend by normal. The marble
cubemap proves the asset plumbing exists. But: it breaks the
"scene = serializable tree" property multiplayer sync relies on (a
texture is an asset reference, not tree data), adds bandwidth-shaped
look variation, and most of what this app wants from textures is
achievable procedurally (§1's noise + patterns) with none of that. Only
worth revisiting if a concrete scene demands photographic detail.

## 6. Cross-cutting infrastructure worth doing once

- **A pipeline-aware palette helper**: a tiny `#[cfg(test)]` (or doc'd)
  Rust function replicating orbit->display math (compress, square,
  light, ACES, gamma) so scene authors can assert "this constant
  displays as approximately this sRGB value" *in a unit test* instead of
  rediscovering the crush-and-yellow lessons by screenshot round-trip.
- **Aliasing policy**: high-frequency patterns (fine stripes, checker,
  noise) shimmer at distance; the ops can't see the cone angle today.
  Cheap mitigation: smoothstep edge widths chosen for the typical view
  distance (barber pole already does this); real fix someday: pass a
  filter width into `col_scene` (it's `t * pixel_angle`, already
  computed in `MARCHER`) and let ops fade their contrast toward the
  pattern's mean — SDF-land's analog of mipmapping.
- **Tag space**: `Fold` tags 0–10 used; pattern ops will consume several
  — no scarcity concern (u8), just keep the registry obvious.

## 7. Suggested order

1. `OrbitPalette` + `OrbitMin` — the fractal scenes' look-ceiling raiser,
   two small ops.
2. Barber-pole `phase` param + an `Expr` on the donut — animated stripes,
   nearly free.
3. Slope tinting + colored ambient (one shader feature, shared uniforms) —
   fixes interior flatness properly.
4. Noise albedo (`NOISE_SDF.md` N1, color-only scope).
5. Emissive channel (pre-bloom).
6. Cavity/edge tint; iteration-graded color; PBR channels alongside the
   specular work; textures only on concrete demand.
