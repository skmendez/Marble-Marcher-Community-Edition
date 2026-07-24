# Smart camera design

Research + design notes for turning the current free-orbit camera
(`app/src/camera.rs`) into a *directed* game camera — one that keeps a clear
view of the marble, frames it at a sensible size, moves like a drone operator
rather than a rigidly-attached boom, and still does exactly what the player
asks it to.

This is a design document, not an implementation record: nothing in
`app/src/camera.rs` has changed yet. Everything below is written against the
code as it exists today, with file/line references so each integration point
is checkable.

---

## 1. What the camera is today

`CameraOrbit` (`app/src/camera.rs:82`) is two fields:

```rust
pub struct CameraOrbit { pub orientation: Quat, pub distance: f32 }
```

and one derivation (`camera.rs:146`):

```rust
eye = target - forward * distance          // forward = orientation * NEG_Z
right = orientation * X,  up = orientation * Y
```

* **Target**: not stored here. `render::update_frame_data_impl` passes the
  local player's marble position every frame (`render.rs:1592`), so the eye is
  recomputed from scratch each frame. There is no camera *state* between
  frames beyond the orientation quaternion and the scalar distance.
* **Input**: `orbit_camera_input` (`camera.rs:245`, mouse drag / wheel / Q-E
  roll) and `touch_camera_input` (`touch.rs:201`, swipe / two-finger twist)
  write `orientation`/`distance` directly, 1:1, with no smoothing. The
  arcball construction in `drag` (`camera.rs:189`) is careful and hard-won —
  a swipe is a single rotation about `forward × screen_dir`, so it can never
  inject twist and its gain is pitch-independent. **This part is good and
  should be preserved verbatim.**
* **Output**: the basis goes straight into `SceneUniforms`
  (`render.rs:1592-1608`), with focal length hard-coded as
  `forward.extend(1.5)`; the shader builds rays as
  `rd = right·ndc.x·aspect + up·ndc.y + forward·f` (`csg/src/codegen.rs:1131-1134`).
* **Side effect**: the camera orientation is also the marble's control frame.
  `physics_sys.rs:869` puts `orbit.orientation` into `PlayerInput`, and
  `marble_csg::physics::step_marble` derives thrust from it. Anything the
  camera does automatically therefore steers the marble too (§4.9).
* **Per-scene tuning**: `render::setup` overrides orientation and distance per
  scene (`render.rs:1062-1085`) — `0.2` for Demo, `1.2` for the Menger scenes,
  `0.6` for HollowDonut — with comments explaining that these are hand-picked
  distances that avoid embedding the eye in geometry.

What it does **not** do: nothing checks whether geometry sits between the eye
and the marble; nothing checks whether the eye is inside a wall; nothing
adapts distance to marble size, screen size, or aspect; nothing damps or
constrains motion. Every "smart" behavior in this document is new, and every
one of the per-scene magic distances above exists because these behaviors are
missing.

---

## 2. What makes this game's camera problem unusual

Five constraints that rule out simply porting a stock third-person rig:

1. **The world is a distance field, on the CPU, for free.**
   `Object::de` (`csg/src/object.rs:121`) and `Object::nearest_point`
   (`csg/src/object.rs:169`) already evaluate the exact same tree the shader
   marches, and the live tree is reachable from any system via
   `MultiplayerSession::sim().scene()`. Every classic camera technique that
   normally needs physics raycasts against triangle soup — occlusion probes,
   swept-sphere collision, whisker rays — becomes a sphere trace, which is
   *cheaper and more informative* than a raycast: it returns a continuous
   clearance, not a binary hit. §4.3 leans on this hard.

2. **The camera has no "up".** `CameraOrbit` is a full 3D orientation
   quaternion with no pitch clamp and no world-up reference (deliberately —
   see the module doc's history of gimbal-lock bugs), and `GravityMode::Flying`
   (the default) has no gravity to define one. So every stock formulation that
   says "yaw around world Y, pitch around local right, keep the horizon level"
   is unusable. Every behavior below is expressed frame-free: as rotations
   about axes derived from the camera's own basis and the geometry, never
   about a world axis.

3. **The geometry is a fractal.** Menger sponges and the classic MMCE fractal
   have thin struts, recursive tunnels, and self-similar detail at every
   scale. "Just pull the camera in until nothing blocks it" degenerates badly
   here: a strut one pixel wide can technically block the sightline every few
   frames. Continuous, hysteretic, time-gated responses are mandatory; binary
   ones will jitter.

4. **Perf: wasm on phones, GPU-bound.** The per-pixel ray march dominates
   frame time. CPU camera work is cheap by comparison but not free (§6 has
   measured numbers).

5. **Rollback multiplayer.** The camera is local-only state, *except* that its
   orientation rides in `PlayerInput` over the wire. That's fine — each client
   sends its own — but it means auto-camera motion changes the control basis
   under the player's feet, which needs rate limits (§4.9).

---

## 3. Research digest: what good third-person cameras do

Condensed from the sources in §10; the parts that actually bear on this game.

**Separate intent from realization.** Every good rig has an "ideal" pose
driven by player input plus a "realized" pose that respects the world.
Cinemachine's Deoccluder is literally a post-process on the ideal pose. The
Little Polygon breakdown goes further and parameterizes the camera as
`(trackingPosition, framing, distance, pitch, yaw)` — treating the camera as a
*picture-plane operator* rather than a world-space actor, and blending in
parameter space so the target provably stays on screen during transitions.
This is the single most useful structural idea for us (§4.1).

**Deocclusion strategies, in order of preference.** Cinemachine offers *pull
camera forward* (dolly in along the view ray until the target is visible),
*preserve camera height*, and *preserve camera distance* (orbit instead of
dolly). Ancillary parameters matter as much as the strategy: **camera radius**
(a fattened probe so the near plane never clips), **minimum occlusion time**
(don't react to a strut you're whipping past), **smoothing time** (hold at the
pulled-in position briefly), and **asymmetric damping** — `Damping` for
returning to normal vs `Damping When Occluded` for reacting, with the reaction
much faster than the recovery. Cinemachine also scores candidate poses by
*shot quality*: distance-from-optimal plus obstruction, which is exactly the
shape of the whisker search in §4.5.

**Thin probes lie.** The classic single ray from target to camera lets the
near plane clip through walls (a ray has no thickness) and produces abrupt
jumps between hit points. The standard fix is a thick probe (spherecast, or
two casts — one thin, one thick). With an SDF we get the thick probe exactly
and for free: stop the march at `de ≤ q` instead of `de ≤ 0`.

**Mario Odyssey specifics** (from reimplementation breakdowns): distance is a
function of pitch (closer looking up, farther looking down); the camera tracks
*ground* height rather than the player's exact position so jumps don't bob the
frame; camera input is velocity-based with damping and a stop threshold; and —
notably — when the player is obscured in tight spots, Odyssey does **not**
always fight the geometry: it draws the player's silhouette through the wall
instead. There's a recenter button (~0.3s) plus a ~5s idle auto-recenter.

**Player authority.** The recurring failure in the "common camera problems"
literature is a camera that fights the player: auto-rotation that resists
input, recentering that fires while the player is steering, cameras that get
"stuck" on geometry so input feels dead. The rule that falls out: player input
is applied immediately and undamped; automatic behaviors are suspended while
the player is touching the controls and for a hold time afterward; hard safety
constraints (don't be inside a wall) are the only thing that overrides intent,
and they should prefer changing *distance* (which reads as necessary) over
changing *direction* (which reads as the camera disobeying).

**Motion sickness.** Excessive automatic rotation, rapid FOV changes, and
frame-perfect tracking of a jittery target all induce nausea. Deliberate lag
in the follow, rate-limited auto-rotation, and slow FOV changes are the
mitigations.

**Smoothing math.** Frame-rate-independent exponential smoothing
(`x ← lerp(target, x, exp(−dt/τ))`) for scalars, and critically damped springs
(ζ = 1 — settles as fast as possible with no overshoot) where velocity
continuity matters. Never a fixed per-frame lerp constant: that silently
changes feel with frame rate, which on a wasm build ranging 30–120 Hz is a
real bug, not a purity concern.

---

## 4. The design

### 4.1 Split: intent vs. realized

```
CameraIntent  (player's wishes — today's CameraOrbit, unchanged math)
  orientation: Quat          // arcball, from drag/roll, applied 1:1
  zoom: f32                  // multiplier on the auto-framed distance, not an absolute
  last_input_at: f32         // for the authority gate

CameraRig     (what actually renders — new)
  orientation: Quat          // realized; forward points at the focus
  distance: f32, distance_vel: f32
  focus: Vec3, focus_vel: Vec3   // smoothed marble position (+ screen-space lead)
  focal_length: f32          // replaces the hard-coded 1.5, for tight-space FOV widening
  occluded_for: f32, clear_for: f32, hold_until: f32
  probe_cursor: usize        // round-robin whisker index
```

`render.rs` reads `CameraRig` (eye = `focus − forward·distance`) instead of
`CameraOrbit::eye_and_basis(marble.pos)`; `cam_forward.w` becomes
`rig.focal_length` instead of the four independently-written `1.5` literals
the code already flags as duplication (`camera.rs:34-40`).

Two structural invariants make requirement 1 ("always a clear view") a
property of the representation rather than something that has to emerge from
tuning:

* **I1 — the eye is always on a ray from the focus.** The realized pose is
  `(u, d)` with `eye = focus + u·d`, never a free-floating position. So "the
  camera is looking at the marble" is true by construction; corrections can
  only change *which* ray and *how far along it*.
* **I2 — `d ≤ t_free(u)`**, where `t_free` is the swept-sphere free distance
  along `u` (§4.3). The eye therefore always lives inside the star-shaped
  region of space visible from the marble. It cannot be inside geometry, and
  it cannot be on the far side of a wall from the marble — not because a
  solver pushed it out after the fact, but because no other state is
  representable.

### 4.2 Framing: distance from screen size (requirement 2)

The shader's projection (`codegen.rs:1131-1134`) is
`ndc_y = y_cam·f / z_cam`, `ndc_x = x_cam·f / (z_cam·aspect)`, with NDC spanning
`[−1, 1]` over the full window. A marble of radius `r` at distance `d`
therefore covers, as a fraction of **screen height**:

```
s_h = f·r / sqrt(d² − r²)   ≈  f·r / d
```

and `s_w = s_h / aspect` of screen width. Taking the *shorter* screen
dimension as the reference (height in landscape, width in portrait) unifies
desktop and mobile into one rule:

```
s_min = f·r / (d · min(1, aspect))          ⟹      d_framing = f·r / (s_target · min(1, aspect))
```

Targets, from the brief:

| profile | target `s_min` | resulting distance |
|---|---|---|
| desktop / pointer | `1/6` of height | `d = 9·r` |
| touch / small screen | `0.28` (between ¼ and ⅓ of width) | `d ≈ 11·r` at portrait aspect 0.46 |

**This rule reproduces the values that were hand-tuned by screenshot**, which
is the main evidence it's the right rule:

| scene | `r` | tuned `d` | `d/r` | rule (`1/6` of height) |
|---|---|---|---|---|
| Demo (`render.rs` default `0.2`) | 0.02 | 0.20 | 10.0 | 0.18 |
| Menger ×3 (`render.rs:1072`) | 0.15 | 1.20 | 8.0 | 1.35 |
| HollowDonut (`render.rs:1084`) | 0.15 | 0.60 | 4.0 | 1.35 → clamped by clearance |

Two of three land within ~12% of the hand-picked value, and the third is
exactly the tight-space case (`0.6` was chosen because the tube's interior
free radius is `0.85`) that §4.4/§4.8 exist to handle — the framing rule wants
`1.35`, the geometry says `0.6`, and the solver is what reconciles them. All
three per-scene distance overrides can then be deleted, and scenes stop
needing a magic number when someone adds one.

Consequences:

* **Zoom becomes a multiplier**, `d_goal = d_framing · zoom`, `zoom ∈ [0.4, 4]`,
  changed multiplicatively (`zoom *= exp(−lines·ZOOM_RATE)`) so a wheel notch
  feels the same at any distance. The existing absolute `MIN_DISTANCE`/
  `MAX_DISTANCE`/`MIN_DISTANCE_MARBLE_RADII` clamps (`camera.rs:41-65`) stay as
  a final safety clamp.
* **Input modality picks the target**: touch profile once any touch event has
  ever been seen (`Touches` is already read in `touch.rs`), else pointer;
  optionally also switch on `min(window.width, window.height) < ~700` logical
  px so a small desktop window behaves sensibly.
* Because `d` now tracks `r`, a future scene with a differently-sized marble
  frames correctly with no tuning at all.

### 4.3 Visibility from the SDF: one march, three answers

March from the marble's surface toward the eye:
`u` = unit vector focus→eye, `t ∈ [r, d]`, `h(t) = de(focus + u·t)`.

Three quantities come out of that single march:

**(a) Swept free distance `t_free`** — the largest `t` reached before
`h ≤ q`, where `q` is the *camera radius* (Cinemachine's "distance to maintain
from any obstacle"; here also the guard against the "embedded camera speckle"
the existing per-scene tuning comments describe). This is exactly "pull camera
forward", computed as a swept-sphere test rather than approximated by a thin
ray: step `t += max(h − q, min_step)`, stop when `h ≤ q`.

Suggested `q = clamp(0.08·d_framing, 0.5·r, 3·r)` — scale-free, so it works
for `r = 0.02` and `r = 0.15` alike.

**(b) Continuous visibility `κ`** — Iñigo Quilez's sphere-traced soft-shadow
ratio, `res = min(res, k·h/t)`, adapted with the physically meaningful `k`:
treat the marble as an area light and ask what fraction of its disc is
unobstructed. An obstruction with clearance `h` at distance `t` from the
marble subtends `h/(d−t)` as seen from the eye; the marble subtends `r/d`. So

```
κ = clamp( min over the march of [ h·d / (r·(d − t)) ], 0, 1 )
```

`κ = 1` means the sightline has a full marble-width of clearance everywhere —
a comfortably clear shot. `κ = 0.3` means a strut is cutting off ~70% of the
marble's silhouette. `κ = 0` means blocked. This is the single most important
piece of the design: it replaces a binary "occluded?" test — which in a
fractal flickers frame to frame and drives visible camera jitter — with a
smooth signal that can be damped, thresholded with hysteresis, and used as a
*gain* on corrective motion. iq's improved variant (tracking `ph`, the
previous `h`, and correcting for closest approach between samples) removes
banding at grazing angles and costs two extra multiplies per step; worth
using.

**(c) The blocking point and its normal** — remember the `t*` where the
minimum was attained. `x* = focus + u·t*`; the outward direction is
`n = normalize(x* − nearest_point(x*))` (exact, one `nearest_point` call at
~0.9 µs) or a 4-tap `de` gradient. §4.5 uses `n` to decide *which way* to slide
so the shot opens up.

These DEs are *estimators* that underestimate true distance (the invariant the
whole renderer already depends on; `Onion`/`Morph` document their Lipschitz
soundness explicitly in `object.rs`). Underestimating is the safe direction
here: we may pull in slightly more than strictly necessary, never less.

### 4.4 Distance solve: fast in, slow out

```
d_goal = min( d_framing · zoom , t_free(u) )   clamped to [max(MIN_DISTANCE, 1.5r), MAX_DISTANCE]
```

with asymmetric damping (the Cinemachine `Damping` vs `Damping When Occluded`
split, and the single most feel-critical pair of constants in the system):

* **Pulling in** (`d_goal < d`): τ ≈ 0.05 s, i.e. near-immediate. Lagging here
  means the wall is visibly in front of the marble, or the eye is inside it.
* **Pushing back out** (`d_goal > d`): τ ≈ 0.35 s **and** only after the
  obstruction has been gone for `PUSH_OUT_HOLD ≈ 0.4 s` (Cinemachine's
  "smoothing time": hold at the near point so a picket fence of struts doesn't
  pump the camera in and out).

Critically damped spring for the scalar (exact ζ=1 solution, frame-rate
independent, `ω = 2/τ`):

```
dx = x − target;  B = v + ω·dx;  e = exp(−ω·dt)
x' = target + (dx + B·dt)·e
v' = (v − ω·B·dt)·e
```

Clamp `dt` to ≤ 1/20 s so a wasm hitch can't produce one enormous step.

### 4.5 Direction solve: slide first, search only if stuck

Requirement 1's hard case is when pulling in isn't enough — the marble rolls
behind a pillar, and no distance along `u` gives a clear shot. Two mechanisms,
in order:

**Tangential slide (continuous, the common case).** With the blocking normal
`n` from §4.3(c), project it perpendicular to the sightline:
`n_t = normalize(n − (n·u)·u)`; rotate about `a = u × n_t` by

```
θ = SLIDE_GAIN · (1 − κ) · dt        (rate-limited to ≤ 90°/s, ≤ 180°/s in "panic")
```

composed onto the realized orientation exactly the way `CameraOrbit::drag`
composes a swipe (`camera.rs:189`) — same axis-⊥-to-forward construction, so
it provably introduces no twist and needs no roll compensation. Visually: the
camera slides sideways around the pillar to peek past it, continuously,
proportional to how badly the shot is blocked. Because the gain is `(1 − κ)`,
a mostly-clear shot produces a barely perceptible drift and a fully blocked
one produces decisive motion — no threshold to jitter across.

**Whisker search (discrete, the fallback).** If `κ ≈ 0` persists for
`> 0.35 s` while sliding (i.e. sliding isn't finding an opening — a dead-end
pocket, or a concave corner), sample candidate directions
`u_i = R(axis_i, angle_i) · u` with `angle_i ∈ {20°, 40°, 70°}` about axes in
the camera's own right/up plane and their diagonals (frame-free — no world up).
Score each with a step-capped march:

```
S_i = w_κ·κ_i + w_d·min(1, t_free_i / d_framing) + w_intent·(u_i · u_intent) + w_cont·(u_i · u)
```

Take the best only if `S_best > S_current + margin` (hysteresis), then commit
to it for ≥ 0.3 s (Cinemachine's minimum-occlusion / smoothing times), and
*rotate toward it at the same rate limit as the slide* — never snap. Evaluate
4 candidates per frame round-robin (`probe_cursor`): the camera turns at ≤ 90°/s,
so a 3-frame-old search result is not observable, and the worst-case per-frame
cost drops by 3–4× (§6).

**Never block the intent.** The user can always keep dragging into the wall:
their `CameraIntent` rotates freely, and the realized camera stays where
geometry allows. When they drag back out, the realized pose springs back to
intent with no state to unwind. To keep drag from feeling dead when the two
diverge a lot, soft-clamp the *intent* toward the realized direction once the
angle between them exceeds ~60°.

### 4.6 Motion: how it moves, not just where it ends up (requirement 3)

* **Focus point.** `focus` is a critically damped spring toward
  `marble.pos` with τ ≈ 0.10 s — short, because the marble *is* the gameplay —
  plus a **screen-space dead zone**: the marble may move within a small
  central box before the focus starts following, which kills micro-jitter
  during vibration/contact chatter without adding perceptible lag. Optionally
  a small **velocity lead** (`focus += clamp(vel·0.12)`), clamped so the
  marble never leaves the central box; in a rolling game this shows the player
  where they're going. (The Odyssey-style "track the ground, not the player"
  trick doesn't port: there's no world up, and in `Flying` mode no ground.)
* **Orientation.** Realized orientation springs toward intent with τ ≈ 0.15 s
  (tight enough that dragging feels 1:1 — see the authority rule below —
  loose enough that corrections blend), with corrective rotations rate-limited
  as in §4.5.
* **Nothing tunnels.** Even with I1/I2, successive eye positions are joined by
  a straight segment, and the visible region is non-convex, so add one short
  swept check per frame: sphere-trace `[eye_prev, eye_new]` with radius `q`; on
  a hit, stop at the hit and project the residual motion onto the surface
  tangent (standard collide-and-slide). This is the "moves like a drone
  operator, doesn't clip through the level" behavior, and it costs ~8 steps.
* **Reuse note.** `marble_csg::physics::collide` (`csg/src/physics.rs:185`)
  already implements "push a radius-`q` sphere out of the geometry with a
  `SamplePoint` list". A fully physical alternative — simulate the camera as a
  second, gravity-free collider springing toward the ideal eye — is tempting
  and would reuse tested code, but it gives up I1/I2: a physical camera can
  end up behind a wall and stay there. **Recommendation: keep the radial
  formulation, and reuse `collide` only as the final push-out safety net.**

### 4.7 Player authority (requirement 4)

Concrete rules:

1. Drag / twist / pinch / wheel write `CameraIntent` **immediately and
   undamped**, exactly as today. Only *corrections* are damped. This is what
   keeps the "if I move the camera right, it moves right" contract literal.
2. Build the drag's `screen_dir` from the **realized** basis (what the player
   sees), then compose that same world-space rotation onto *both* realized and
   intent. If drag used the intent basis while the screen showed the realized
   one, a divergence would make swipes come out crooked — a bug worth
   designing out up front.
3. All *elective* automatic behaviors (idle re-follow, whisker re-framing) are
   suspended while any camera input is active and for `AUTO_RESUME ≈ 1.2 s`
   afterward.
4. *Safety* behaviors (pull-in, clearance push-out, tangential slide) never
   suspend — but they act on distance first and direction only when distance
   can't solve it, because players read a dolly-in as "the camera had to" and
   an unrequested orbit as "the camera disobeyed".
5. Roll (`Q`/`E`, two-finger twist) is never touched by any automatic
   behavior. There's no world up to level against, so there is no
   auto-leveling to do — and the arcball construction guarantees no correction
   leaks twist.

### 4.8 When geometry wins: two fallbacks

Inside a Menger tunnel or the HollowDonut tube, *no* camera pose satisfies
both framing and clearance. Two levers, in order:

* **Widen the FOV instead of backing up.** `f` (`cam_forward.w`) is already a
  per-frame uniform; the shader derives its cone angle and step budget from it
  (`codegen.rs:1196-1205`). When `t_free < ~0.6·d_framing`, reduce `f` toward
  `f_min = 1.0` (90° vertical FOV, from the default 67°), which restores the
  marble's screen fraction while physically closer. Rate-limit hard (τ ≈ 0.5 s,
  and only widen, never narrow below default) — FOV pumping is a top-tier
  nausea trigger. This lever is nearly free here and unavailable to most
  engines' camera code.
* **X-ray the marble.** The fine shader already resolves the marble as an
  analytic sphere and depth-compares it against the terrain march
  (`codegen.rs:1233-1253`). When the terrain wins *and* `κ` has been ~0 for
  longer than the solver's recovery time, blend a silhouette/ghost of the
  marble over the terrain instead of continuing to fight the geometry — one
  extra branch plus a uniform. This is precisely what Odyssey does in tight
  spots, and it's the only mechanism that can *guarantee* requirement 1 in a
  fractal where a genuinely enclosed pocket may admit no clear view at all.

### 4.9 What feeds marble control

`physics_sys.rs:869` should send the **realized** orientation, not the intent:
the player steers by what they see, and a mismatch between the rendered view
and the control frame is worse than slow drift in either. The rate limits in
§4.5 bound how fast the control basis can rotate on its own (≤ 90°/s, and only
while the player isn't steering the camera). No determinism concern —
`PlayerInput.orientation` is transmitted per-client, and rollback resimulates
from the transmitted values.

### 4.10 Per-frame pipeline

```
Update (after orbit_camera_input / touch_camera_input, before update_frame_data):

  1. dt = min(real_dt, 1/20)
  2. focus  ← spring(focus, marble.pos, dead-zone, τ=0.10) [+ velocity lead]
  3. d_framing ← f·r / (s_target(profile) · min(1, aspect))
  4. march once along u_realized  →  t_free, κ, blocking normal n      [§4.3]
  5. direction:  u_desired ← intent  (+ tangential slide if κ<1)
                 whisker search only if κ≈0 sustained                  [§4.5]
                 slerp realized → u_desired, rate-limited
  6. distance:   d_goal = min(d_framing·zoom, t_free); asymmetric spring [§4.4]
  7. eye = focus + u·d;  swept check + clearance push-out               [§4.6]
  8. focal_length ← widen if t_free ≪ d_framing                         [§4.8]
  9. write CameraRig; render.rs and physics_sys.rs read it
```

---

## 5. Code shape and integration points

```
app/src/camera/
  mod.rs        Bevy resources + systems; the only file that knows about Bevy
  intent.rs     today's CameraOrbit, renamed CameraIntent — math and tests unchanged
  framing.rs    screen-fraction ↔ distance; shares FOCAL_LENGTH with debug_gizmos
  solver.rs     pure solve(state, goal, &impl Sdf, dt) -> CameraRig     (glam only)
  smoothing.rs  critically damped spring for f32 / Vec3 / Quat, dt-correct
csg/src/visibility.rs   (new, marble-csg)
  trait Sdf { fn de(&self, p: Vec3) -> f32; }   + impl for (&Object, &Params)
  fn sweep(...) -> Sweep { t_free, kappa, block_point, block_t }
```

Putting the sphere-trace queries in `marble-csg` next to `physics.rs` keeps
them Bevy-free and unit-testable with analytic worlds; putting the solver
behind an `Sdf` trait means the whole camera can be tested against a plane, a
slab with a hole, and a pillar — no fractal required, no Bevy `App` required.

Touch points, all small:

* `main.rs:281-282` — add `smart_camera_solve` to the `Update` chain right
  after `touch_camera_input`, before `update_frame_data`.
* `render.rs:1592` — read `CameraRig` instead of `orbit.eye_and_basis(target)`;
  `render.rs:1608` — `forward.extend(rig.focal_length)`.
* `render.rs:1062-1085` — delete the per-scene distance overrides (framing rule
  supersedes them); keep the per-scene *orientation* presets.
* `physics_sys.rs:869` — `orientation: rig.orientation`.
* `config.rs` — `?smartcam=0` / `MM_SMARTCAM=0` reverts to today's behavior,
  following the existing flag convention, so A/B comparison and bisecting a
  feel regression stay possible.
* `fps_overlay.rs` — add `κ`, `t_free`, `d/d_framing`, screen fraction, and
  auto/idle state to the `?debug=1` readout; add a `"camera"` phase to
  `PhaseTimings` (the harness in `perfprobe.rs` already reports per-phase cost).

---

## 6. Performance budget (measured)

`rust/csg/examples/de_bench.rs` (added with this document; run with
`cargo run --release -p marble-csg --example de_bench`). Native, this
container's CPU, `opt-level = 3` — the profile `marble-csg` gets in release
builds:

| scene | `de` | `nearest_point` | 32-step march |
|---|---|---|---|
| Demo (classic, iters=16, ∪ creme spheres) | **649 ns** | 920 ns | **20.5 µs** |
| Menger sphere, depth 5 | 236 ns | 571 ns | 7.9 µs |

Per-frame worst case for the design above, on the most expensive scene:

| work | steps | cost |
|---|---|---|
| primary sweep (§4.3) | ≤ 24 | 15.6 µs |
| swept move check (§4.6) | ≤ 8 | 5.2 µs |
| 4 whiskers (only while blocked) | ≤ 16 each | 41.5 µs |
| `nearest_point` for the blocking normal | 1 | 0.9 µs |
| **total, blocked frame** | | **≈ 63 µs** |
| **total, ordinary frame** | | **≈ 21 µs** |

That's 0.13% / 0.38% of a 16.7 ms frame natively, and real marches terminate
far earlier than the cap (the benchmark's realistic traces average 1–3 steps).
For scale, the existing 60 Hz physics tick costs roughly 9 µs of the same kind
of work (6 substeps × `de` + `nearest_point`). Wasm on a phone is perhaps 2–4×
slower per eval, so ≤ 0.25 ms worst case — still comfortably inside a
GPU-bound frame. Guard rails if a device disagrees: cap the solver to 60 Hz,
drop whiskers to 2/frame, and shrink the step caps; all three are constants,
and `PhaseTimings` will show which one to turn.

The alternative of reading back the coarse pass's depth texture
(`mrrm.rs`, already a `Rgba16Float` of hit distances) was considered and
rejected: browser readback is ≥ 1 frame latent and awkward, and at ~20 µs the
CPU march simply isn't worth avoiding.

---

## 7. Test plan

Pure-function tests (`solver.rs`, `visibility.rs`) against analytic `Sdf`
worlds — no Bevy, no fractal, in the fast `cargo test -p marble-csg` /
`cargo test -p marble-marcher-bevy` paths the repo already uses:

* **Framing** — in open space, the solved distance puts the marble within ±5%
  of the target screen fraction, at aspects 0.46 / 1.0 / 1.78 and radii
  0.02 / 0.15; and the rule reproduces the historical hand-tuned values in §4.2.
* **I1/I2 invariants** — after any sequence of random intents and target
  motions against a pillar/slab world, `de(eye) ≥ q` and the marble→eye
  segment is unobstructed, every step. This is requirement 1 as an assertion.
* **Pull-in beats push-out** — with a wall sweeping across the sightline, the
  distance reaches its constrained value within ~0.1 s and takes ≥ 0.4 s to
  recover, and never oscillates (bounded sign changes of `distance_vel` over
  600 steps).
* **No jitter on thin struts** — a picket fence of struts crossing the
  sightline at 10 Hz produces bounded total rotation (the min-occlusion-time
  and hysteresis gates are load-bearing; without them this test fails loudly).
* **Player authority** — a drag applied while the camera is fully occluded
  still changes intent by exactly the arcball rotation (bit-for-bit against
  today's `CameraOrbit::drag`), and with no obstruction the realized pose
  converges to intent within τ.
* **Frame-rate independence** — the same scripted input at dt = 1/30, 1/60,
  1/144 lands within a small tolerance of the same pose.
* **Regression** — every existing test in `camera.rs:286-570` keeps passing
  unchanged; `CameraIntent` is the same type with a new name.

Live/visual: a `?camprobe=1` mode that logs `(κ, t_free, d, s_min, steps)` per
frame to CSV along a scripted marble path per scene, plus the existing
`MM_SCREENSHOT` harness for before/after stills at known poses. The acceptance
bar per phase is stated below rather than "looks better".

---

## 8. Phasing

Each phase is independently shippable and independently revertable
(`?smartcam=0`).

* **Phase 0 — framing + instrumentation.** `d = f·r/(s·min(1,aspect))`, zoom as
  a multiplier, delete the per-scene distances, debug readout, `"camera"`
  phase timing. No occlusion work. *Accept:* marble reads at the target size
  on a phone and on a desktop window, unchanged feel otherwise.
* **Phase 1 — clearance.** The primary sweep, `t_free` pull-in with asymmetric
  damping, clearance push-out, swept move check. *Accept:* the eye is never
  inside geometry in any scene (probe log shows `de(eye) ≥ q` throughout a
  scripted run); the HollowDonut no longer needs its hand-tuned `0.6`.
* **Phase 2 — deocclusion.** `κ`, tangential slide, whisker search with
  hysteresis and the authority gate. *Accept:* rolling behind a pillar
  recovers a clear view within ~0.5 s without the camera ever crossing the
  pillar, and the picket-fence test shows no pumping.
* **Phase 3 — tight spaces.** FOV widening and the x-ray marble fallback.
  *Accept:* inside a Menger tunnel the marble stays visible and reasonably
  sized at all times.
* **Phase 4 — polish.** Velocity lead, idle re-follow, optional recenter
  gesture, per-scene target-fraction overrides if any scene wants one.

---

## 9. Risks and open questions

* **Feel is not provable.** Everything above is falsifiable by tests except
  the constants, which need a human on a real device. Phases exist so each
  behavior can be judged separately rather than as one wall of new motion.
* **Auto-rotation moves the control frame.** §4.9 rate-limits it, but if it
  reads badly the fallback is to feed *intent* to `PlayerInput` and accept the
  view/control mismatch, or to freeze the control basis while a correction is
  in flight. Worth trying the simple version first.
* **`GravityMode::Flying` is the default**, and a free-flying marble makes
  "behind the player" meaningless. All the elective behaviors are therefore
  optional and idle-gated; the safety behaviors are the ones that carry the
  design.
* **Fractal DE looseness** costs march steps near surfaces (the step caps
  bound it) and makes the pull-in mildly over-conservative. Safe direction.
* **Multiplayer**: only the local marble is framed. Whether a second player
  should influence framing (Odyssey's co-op does) is out of scope here.
* **Rejected**: physical drone camera (loses I1/I2 — can end up behind a wall);
  GPU depth readback (latency, no perf need); fixed/cinematic camera volumes
  (no level authoring pipeline in this port); world-up-based rigs (no world up).

---

## 10. Sources

Game-camera design:

* [Tech Breakdown: Third Person Cameras in Games — Little Polygon](https://blog.littlepolygon.com/posts/cameras/) — the picture-plane parameterization, framing offsets, blending in parameter space, the velocity "leash".
* [Cinemachine Deoccluder documentation](https://docs.unity3d.com/Packages/com.unity.cinemachine@3.1/manual/CinemachineDeoccluder.html) — camera radius, pull-camera-forward, minimum occlusion time, smoothing time, asymmetric damping, shot-quality scoring.
* [Avoid collisions and evaluate shots — Cinemachine](https://docs.unity3d.com/Packages/com.unity.cinemachine@3.1/manual/CinemachineColliderConfiner.html)
* [John Nesky, "50 Game Camera Mistakes", GDC 2014](https://gdcvault.com/play/1020460/50-Camera) ([video](https://www.youtube.com/watch?v=C7307qRmlMI)) — dynamic third-person cameras, line of sight as a hard requirement.
* [Third-Person Camera View in Games: common problems and solutions — Game Developer](https://www.gamedeveloper.com/design/third-person-camera-view-in-games---a-record-of-the-most-common-problems-in-modern-games-solutions-taken-from-new-and-retro-games) — occlusion dithering/silhouettes, camera bounce in tight spaces, whiskers, motion sickness, agency.
* [Mario Odyssey camera controller breakdown (Godot)](https://lewisstephens.itch.io/mario-odyssey-camera-controller-in-godot-44) — pitch→distance mapping, ground tracking, silhouette-instead-of-pull-back, damped velocity input, idle recenter timings.
* [A 3rd person camera in a complex voxel world](https://bonsairobo.medium.com/a-3rd-person-camera-in-complex-voxel-world-523944d5335c) — thin-vs-thick probe rays and why a zero-thickness ray isn't enough.

Math:

* [Iñigo Quilez, "Soft shadows in raymarched SDFs"](https://iquilezles.org/articles/rmshadows/) — `res = min(res, k·h/t)` and the closest-approach-corrected variant, adapted in §4.3(b).
* [Ryan Juckett, "Damped springs"](https://www.ryanjuckett.com/damped-springs/) — the exact critically damped solution used in §4.4.
* [Improved Lerp Smoothing — Game Developer](https://www.gamedeveloper.com/programming/improved-lerp-smoothing-) — frame-rate-independent exponential smoothing.
