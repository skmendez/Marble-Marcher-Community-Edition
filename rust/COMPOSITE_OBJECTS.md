# Composite objects — brainstorm

What else the `Object`/`Fold` algebra could grow, now that `Offset` (tag 6)
exists. Same ground rules as `NOISE_SDF.md`: every node is four things at
once (CPU `de` for physics, CPU `nearest_point` for collision response,
WGSL emission, tag-prefixed serialization + `handles_valid_for` +
`bounding_sphere`), and candidates are sorted by how honest their distance
field is:

- **Exact**: `|∇d| = 1` a.e., closed-form `nearest_point` possible.
- **Sound bound**: never overestimates distance (safe to march and collide
  against), but `|∇d| < 1` somewhere; `nearest_point` needs approximation.
- **Unsound without correction**: can overestimate; needs a Lipschitz
  divisor or the `p.w` trick before it's shippable at all.

A recurring dependency shows up below: several of the best candidates have
no closed-form `nearest_point`. Rather than blocking on that per-node, one
shared piece of infrastructure unblocks all of them at once — see §7.

## 1. Exact single-child nodes (cheapest wins)

### `Object::Onion { base, thickness }` — hollow shell
`d = |base.de(p)| − t / p.w`. The sibling `Offset`'s doc already promises
this one. Turns any solid into a shell of thickness `2t` — and for a marble
game that's not a rendering trick, it's a *level archetype*: rolling around
the **inside** of a hollow sphere, or inside a hollowed Menger shell, works
immediately because physics collides against the same field. Exact for an
exact base (`|·|` preserves unit gradient a.e.). `nearest_point`: the
base's nearest point pushed `±t` along the same line `Offset` already
computes — same code shape. Caveat to document: for a base whose interior
`de` is only a bound (inside `Union` overlaps), `|d|` inherits that
looseness; fine in practice, worth a sentence. Bounding sphere: child
radius `+ t`.

### `Object::Negate { base }` — complement
`d = −base.de(p)`. Three lines per role, exact, and `nearest_point` is
*identical* to the base's (nearest surface point doesn't move when you flip
inside/outside). What it buys: cave/arena worlds — `Negate(sphere)` is an
infinite solid with a spherical playable void; intersect with anything for
carved-out interiors without needing an "everything" primitive to subtract
from. `bounding_sphere`: `None` (the complement of a bounded thing is
unbounded) — same `Intersect`-resolves-it story as `Modulo`.

## 2. Exact folds (domain ops with closed-form unfold)

### `Fold::Elongate { half_extent }` — stretch along axes
`q = p − clamp(p, −h, h)` (IQ's exact elongation). One fold turns the
existing primitives into a whole family: sphere → capsule, sphere →
rounded slab, cuboid → longer cuboid without touching its params. Exact,
and the unfold is closed-form: push the pre-fold `p`, pop and add back
`clamp(p, −h, h)` to the nearest point (it's a per-region translation —
same push/pop contract `Modulo` already follows). This is the cheapest way
to get capsules/rails/beams for level-building without new primitives.

### `Fold::FiniteRepeat { axis, spacing, count }` — bounded tiling
The bounded sibling of `Modulo`: `q = p − s·clamp(round(p/s), −n, n)`.
Fixes a real wart: `Modulo` is infinite, so every tiled scene needs an
`Intersect` wrapper just to have a bounding sphere (`creme_spheres`), and
the ray-clip pre-test degrades to nothing without one. Finite repeat keeps
a finite bound natively (`(2n+1)·s/2 + child_radius` along that axis,
composed per-axis). **Soundness constraint to enforce at construction**:
plain repetition is only exact when the base fits inside half a cell —
otherwise a query near a cell wall can be closer to the neighbor cell's
copy than to its own, and the field overestimates (the same neighbor-cell
issue `NOISE_SDF.md` §3 documents for jittered lattices). A
`debug_assert`/doc contract ("base bounding radius ≤ s/2") is enough for
v1; the alternative (min over the 2 adjacent cells) doubles cost for a
case scenes shouldn't ship anyway. Unfold: closed-form translation, like
`Modulo`'s.

## 3. Sound-bound combiners (the headline visual upgrades)

### `Object::SmoothUnion { left, right, k }` (+ Intersect/Difference duals)
The single most-requested CSG op in any SDF system: the polynomial
smooth-min fillet,

```
h = clamp(0.5 + 0.5·(dr − dl)/k, 0, 1)
d = mix(dr, dl, h) − k·h·(1 − h)
```

Welds two shapes with a rounded fillet instead of a crease — this is what
makes composed scenes stop looking like boolean arithmetic and start
looking sculpted. Correctness status: `smin ≤ min`, so it's a **sound
underestimate** of its own (filleted) surface — safe for marching and for
physics contact tests — with sub-unit gradient only inside the blend band
(width ~`k`). Bounding sphere: the fillet only adds material where a point
is within `k` of *both* children, so the plain enclosing-sphere-plus-`k`
is sound. Two real design costs, both solvable:

- **`nearest_point` has no closed form.** In the blend band, "which child"
  is genuinely the wrong question. Needs §7's shared fallback; outside the
  band (h saturated at 0/1), delegate to the winning child exactly like
  `Union` does today.
- **The color pass needs a blend, not a pick.** `emit_combine` snapshots
  `orbit` and picks one side's; a smooth union should `mix` the two
  orbits by the same `h` or the fillet region's color pops. This is the
  first node that makes the emitted combiner shape genuinely different
  from the existing three, not just a different comparison operator.

Serialization note: `k` as a `ScalarValue` means the fillet radius is
param-driven and `Expr`-animatable (a weld that tightens over time) for
free.

### `Object::Morph { a, b, t }` — animated interpolation
`d = mix(a.de(p), b.de(p), t)`, `t ∈ [0,1]` a `ScalarValue`. The
convexity argument makes this **sound**: a convex combination of
1-Lipschitz functions is 1-Lipschitz, and any 1-Lipschitz field is a valid
conservative DE of its own zero set — so marching and physics are both
safe at every `t`, even though mid-morph shapes are not exact SDFs.
What makes it special *here* specifically: this repo already has
deterministic tick-driven `Expr` animation feeding `ScalarParam`s through
rollback (`menger_oscillating_sphere`), so `Morph` immediately gives
**geometry that morphs identically on every multiplayer peer** —
sphere-world melting into cube-world on a 30-second cycle — with zero new
animation machinery. Bounding sphere: enclosing sphere of both children's
bounds (sound at every `t`). `nearest_point`: §7 again.

## 4. Corrected distortions (need the `p.w` trick, flagged honestly)

`Fold::Twist { axis, rate }` / `Fold::Bend { rate }` — the classic
sculpting warps. Both stretch space, so raw emission **overestimates** —
unsound, full stop. The fix is the same one `NOISE_SDF.md` §5b identified
for domain warping: the framework already divides every DE by accumulated
`p.w`, so a warp with Lipschitz constant `L` just multiplies `p.w` by `L`
(`L = 1 + |rate|·max_radius` for twist, needing the child's bounding
radius at build/emission time — a new but tractable wrinkle: the fold
needs to know its subtree's bound). Twisted Menger columns are the
signature look this buys. Unfold for `nearest_point` is approximate
(inverse-rotate by the twist angle at the *folded* height — good, not
exact). Worth doing *after* the exact tiers; listed here so the ordering
is a decision, not an accident. The chamfer/columns ops from hg_sdf
(`min(min(a,b), (a+b−r)/√2)`) belong in this tier too: the diagonal term's
gradient reaches √2 when the children's gradients align, so they'd need a
√2 divisor (or acceptance of unsoundness) — the polynomial smooth-min
family in §3 gives the same visual role without that problem, so chamfer
is probably not worth it.

## 5. New primitives that punch above their weight

Not composites, but three exact primitives that multiply what the
combiners above can express, each a handful of lines in all four roles:

- **`HalfSpace { normal, offset }`** — an actual ground plane. Every
  "marble rolls on terrain" experiment currently fakes ground with a huge
  cuboid face; exact, free, unbounded (`bounding_sphere: None`).
- **`Torus { major, minor }`** — exact, and it's a *ring track* for a
  marble; `Onion(Torus)` is a half-pipe the moment `Intersect` cuts it.
- **`Capsule { a, b, radius }`** — exact segment-distance primitive;
  rails/pillars/bridges. (Partially redundant with `Elongate(Sphere)` —
  pick one; `Elongate` is more general, `Capsule` reads better in scene
  code.)

## 6. Deliberately not proposed

- **2D extrusion/revolution sub-language** (IQ's `opExtrusion`/
  `opRevolution`): real power, but it drags in a whole second tree type
  (2D primitives, their own codegen and serialization). Only worth it when
  a concrete level design demands profiles that the 3D algebra can't fake.
- **Displacement / additive blends**: sums of fields — ruled out by the
  unit-gradient requirement established in `NOISE_SDF.md` (v2 §2); their
  use cases are covered by `SmoothUnion` + `Offset` + CSG.
- **Engrave/groove/tongue ops**: already expressible as compositions of
  existing nodes plus `Onion` (`Difference(base, Intersect(shell, tool))`
  etc.) — adding named nodes for them would grow the enum without growing
  the algebra.

## 7. The shared unblocker: a generic `nearest_point` fallback

`SmoothUnion` and `Morph` (and any future blend-family node) all stall on
the same missing piece: a `nearest_point` for fields where "delegate to
one child" is wrong. The fix is one shared helper, not per-node cleverness:
project along the numerical gradient,

```
x = p;  repeat 2–3×:  x -= de(x) · ∇de(x)/|∇de(x)|   (4-tap tetrahedral ∇, mirroring calc_normal)
```

- Physics only calls `nearest_point` when `de < marble_rad` (contact), so
  the iteration starts near-converged; 2 steps generally lands well within
  the smallest marble radius (0.02) for 1-Lipschitz sound fields.
- Costs ~10–15 `de` evaluations per *contacting* sample — bounded, and
  only on the new node types; every existing node keeps its exact path.
- `physics.rs`'s `SamplePoint` doc already anticipates exactly this
  ("point-shell sampling... without reworking the physics API").

Build it once (private helper in `object.rs`, property-tested against the
exact nodes where the true answer is known), and the whole §3 tier opens
up.

## 8. Suggested order

1. **`Onion`** — trivial, exact, and unlocks a genuinely new level
   archetype (playing inside geometry). Pairs with existing `Intersect`
   for windows/cutaways. Tag 7.
2. **`Negate`** — near-free, enables cave worlds. Tag 8.
3. **§7's gradient-projection `nearest_point` fallback** — infrastructure,
   no new nodes, tested against exact ground truth.
4. **`SmoothUnion`/`SmoothIntersect`/`SmoothDifference`** — the big visual
   payoff, now unblocked. Tags 9–11 (or one tag + a mode byte, mirroring
   how `Combine` is one code path in codegen).
5. **`Morph`** — small once §7 exists; showcase scene: tick-driven
   world-morph, multiplayer-deterministic by construction. Tag 12.
6. **`Fold::Elongate` + `Fold::FiniteRepeat`** — exact folds, tags 10–11
   in the fold space; `FiniteRepeat` retires the `Intersect`-for-bounding
   workaround for new scenes.
7. **`HalfSpace`/`Torus`** primitives alongside, as level design wants
   them.
8. **`Twist`/`Bend`** last, once the subtree-bound-at-emission wrinkle is
   worth solving for the look.
