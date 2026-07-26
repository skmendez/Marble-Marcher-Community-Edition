# Smart camera design

Research + design notes for turning the current free-orbit camera
(`app/src/camera.rs`) into a *directed* game camera — one that keeps a clear
view of the marble, frames it at a sensible size, moves like a drone operator
rather than a rigidly-attached boom, and still does exactly what the player
asks it to.

**Status: implemented, shipped behind `?smartcam=1`/`MM_SMARTCAM=1`**
(`app/src/smart_camera.rs`, `csg/src/visibility.rs`). Off by default pending
a play-test: the framing rule (§4.2) is on either way, the geometry-aware
behaviors need the flag.
Sections 1-5 below are the design as written *before* implementation, kept as
the record of the reasoning; §11 at the end lists what implementation
changed, what it measured, and what is still missing; §12-15 are the
play-test findings that followed, each with the measurement that reproduced
it. Where any of them disagree with §1-5, the later section is what the code
does. File/line references in §1-5 are to the pre-implementation code and
are left as they were.

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

---

## 11. What implementation changed

Built in this order: `csg/src/visibility.rs` (the sphere-trace queries),
`app/src/smart_camera.rs` (the solver + its Bevy system), then the wiring
(`render.rs` reads the rig, `physics_sys.rs` takes its control frame from
it, `camera.rs` keeps the intent and drives both). 210 tests pass
(`cargo test -p marble-csg -p marble-marcher-bevy`); all six scenes were
rendered headlessly and checked.

### Corrections to the design

**Visibility is measured only as far as the camera can actually go** (§4.3
assumed one march to the *intended* distance). Marching past
`free_distance` counts geometry the camera will never be in front of: a
camera with its back to a wall reads as permanently blocked, and the solver
slides around for no reason. `sweep` is now two passes — pass 1 finds the
reachable distance stepping by `h - camera_radius` (which is also what makes
the swept-ball test sound *between* samples, not just at them; a plain
`t += h` step let the eye clip walls the sweep called clear), pass 2
measures visibility over `[r, free_distance]` with the eye at
`free_distance`, so the perspective term uses the real eye position rather
than the hoped-for one. The distinction matters: "something is in the way"
and "there is a wall behind me" call for opposite responses.

**The push-out hold keys off the goal tightening**, not off the view being
clear (§4.4 said "clear for `PUSH_OUT_HOLD`"). Any measure of *current*
clearness deadlocks in a busy space: one 0.95-visible frame every so often
resets the timer forever and the camera, having pulled in once, never backs
out again. Keying it off "did the constraint just move in?" gives the
intended anti-pumping behavior without the deadlock.

**A third search trigger: `cramped`.** §4.5 fires the whisker search only on
sustained occlusion. But the common tight-space failure is a view that is
perfectly *clear* and much too close — an unobstructed close-up looks
entirely fine to occlusion logic, so nothing pushed the camera back out.
`cramped` (free distance under half of what framing wants, for 0.4 s) is
Cinemachine's shot-quality idea reduced to the part that matters here.

**Searches commit, and repositions are held.** Re-running the search every
frame and easing toward whatever won *that* frame thrashes between
near-equal candidates; and once a reposition lands, the decay back toward
intent immediately undoes it, since intent is what got the camera stuck in
the first place. Fixed with a 0.4 s commitment plus a 2.5 s post-reposition
lockout on the decay. Neither ever blocks the player: input writes the
realized camera directly.

**The deviation cap drags the intent along** rather than hauling the camera
back (§4.5 floated this as a "soft-clamp"; it turned out to be required). A
marble travelling down a curved tunnel rotates which directions are usable
while the intent quaternion sits where the player last left it, so the
deviation grows for reasons unrelated to the camera misbehaving — and at the
cap, the camera is then forbidden from going anywhere it can see from.

**Follow smoothing applies across the frame only, never along the view
axis.** A spring following a moving target trails it by roughly
`speed * tau`. Across the frame that is harmless and rather nice — the
marble leads slightly in the direction it is going. Along the view axis it
is neither: it quietly adds the trailing distance to the camera's distance
(a marble flying away at 3 units/s sat ~20% further back than the framing
rule asked for), and the instant the marble stops — which for a marble means
*hitting something* — all of it unwinds at the spring's rate. Reported from
play as the camera whipping in toward the marble on every collision, and
reproduced in empty space, which is what established it was never a
deocclusion problem. Zeroing the depth component of the focus error leaves
the eye exactly `distance` from the marble at all times, so how far away the
camera sits is purely the distance solver's business — damped, asymmetric,
geometry-aware — instead of something the follow spring gets an unowned say
in. It also measurably improved the tube: HollowDonut's mean visibility went
from 0.86 to 1.00 and its blocked frames from 66 to 0, because the sweep now
starts from where the camera is really looking.

**The search's first candidate is the local clearance gradient** at the
marble (`Sdf::outward`), not just the ring of rotations. In a tunnel that
points straight into the open middle, where hill-climbing 40° at a time
takes about a second and a half to arrive.

**`CameraOrbit` keeps its name** (the design proposed renaming it to
`CameraIntent`) — its doc now says what it is. Its `distance` field became
`zoom`, a multiplier on the framed distance, and `eye_and_basis` became the
associated `basis_from`, so the realized camera can share it.

### What it measures

`smart_camera::scene_probe` drives every scene through the real physics with
a scripted movement + camera-drag script, once with the solver on and once
with it off (`?smartcam=0`: same framing rule, no geometry awareness), and
reports what the camera did. `cargo test -p marble-marcher-bevy scene_probe
-- --nocapture`:

| scene | smart | mean vis | frames blocked | min eye clearance | screen size | `de` steps/frame |
|---|---|---|---|---|---|---|
| demo | off | 1.00 | 0/480 | +0.010 | 0.167 | 3.4 |
| demo | **on** | 1.00 | 0/480 | +0.004 | 0.164–0.175 | 4.2 |
| classic_only | off | 1.00 | 0/480 | +0.020 | 0.167 | 2.7 |
| classic_only | **on** | 1.00 | 0/480 | +0.004 | 0.167–0.175 | 3.0 |
| menger_sponge | off | 1.00 | 0/480 | +0.155 | 0.167 | 3.5 |
| menger_sponge | **on** | 1.00 | 0/480 | +0.142 | 0.167 | 3.9 |
| menger_sphere | off | 1.00 | 0/480 | +0.155 | 0.167 | 3.5 |
| menger_sphere | **on** | 1.00 | 0/480 | +0.142 | 0.167 | 3.9 |
| menger_oscillating_sphere | off | 1.00 | 0/480 | +0.053 | 0.167–0.193 | 3.8 |
| menger_oscillating_sphere | **on** | 1.00 | 0/480 | +0.045 | 0.165–0.172 | 3.5 |
| hollow_donut | off | **0.69** | **147/480** | **−0.061** | up to **1.34** | 8.0 |
| hollow_donut | **on** | 0.99 | 3/480 | +0.041 | up to 0.92 | 16.0 |
| cube_sphere_morph | off | 1.00 | 0/480 | +0.739 | 0.167 | 2.0 |
| cube_sphere_morph | **on** | 1.00 | 0/480 | +0.740 | 0.167 | 2.0 |

The rows that matter are the ones where geometry actually crowds the camera.
`hollow_donut` is the case: with the solver off, the marble is behind
geometry for 40% of the run, the eye dips inside the shell, and the camera
gets pinned close enough that the marble fills the frame outright. With it
on, the marble is visible on all but three frames of the run, the eye stays
outside the geometry throughout, and the worst-case framing improves by a
third. The four open scenes show *zero* distance travel over an
eight-second run: with nothing in the way, the camera simply holds its
frame. Where nothing is in
the way — the four open scenes — the two agree to within a few percent and
the solver costs a fraction of a `de` evaluation per frame.

Note the "off" column is *not* the pre-feature camera: the framing rule
applies either way, and one geometry-aware behavior stays on with the flag
off — the distance is still capped at the swept free distance, so the eye
does not end up inside a wall. That cap is what the deleted per-scene
distance constants were for (`HollowDonut`'s `0.6` was chosen because the
tube's interior free radius is `0.85`), so dropping them without it would
have left the default strictly worse than before this work. It costs
nothing in feel: the camera still points exactly where the player says,
instantly, with no damping and no automatic rotation.

Screen size is the fraction of the shorter screen dimension, against a
target of 0.167. Eye clearance is the distance field at the eye — invariant
I2, holding in practice on real fractal geometry and not just in the
analytic-world unit tests.

Cost is well under the §6 budget: 3–5 `de` evaluations per frame in the open
scenes (the budget assumed up to 24), ~14 in the tube where the search
actually runs. The `?debug=1` overlay reports the camera phase at 0.01 ms.

### Known limits

**HollowDonut still frames tight.** Inside a closed tube whose interior free
radius is `0.85`, with a `0.15` marble that spends the probe run pressed
against the wall, the framing rule's `1.36` barely fits across the tube at
all. The camera now keeps the marble visible for all but three frames of the run
and never enters geometry, but spends much of it closer than the framing rule
wants — up to 0.9 of the frame at worst. FOV widening recovers part of it. Two things make it
genuinely hard rather than merely untuned: the usable directions sweep
around the tube as the marble circles it, so the reposition is a chase
against a moving target; and the marble's thrust is camera-relative, so the
camera's own motion feeds back into where the marble goes. Ideas not yet
tried: preferring directions whose usefulness is *stable* under marble
motion (the tube's axial direction, rather than its radial one), and the
x-ray fallback below.

**Not implemented from the design:** the x-ray/silhouette marble fallback
(§4.8's second lever — the shader work is small and the hooks are all
there), the screen-space dead zone and velocity lead (§4.6 — the focus
spring plus a hard lag clamp covers the jitter case), and idle auto-follow
(§4.7 — deliberately: there is no world "up" or "behind" in `Flying` mode
for it to mean anything).

**A pre-existing bug this work surfaced:** `Object::nearest_point_scratch`
asserted its fold-history stack was *empty* on return from a `Fractal` node,
which is only true for a top-level one. `scenes::hollow_donut` nests a
`Fractal` (the skylight) inside another, so any physics contact there
panicked in debug builds. Now asserts the stack returned to its entry depth.

(A second one — the hard-coded `CompositeAlphaMode::PreMultiplied` panicking
at surface configuration on lavapipe, which broke
`scripts/headless_screenshot.sh` — was hit here too, but master fixed it
independently while this branch was in flight, so the merge kept master's
`cfg(target_arch)`-based version rather than this branch's env-var hatch.)

---

## 12. Orbiting into geometry

Reported from play, after §11 shipped: with a large structure beside the
marble, rotating the camera slightly took the distance from `1.411` to
`0.279` in one motion — the camera diving at the marble — while `vis` stayed
`1.00` the whole time. The marble was never occluded. The camera was simply
being orbited into a place it could not fit, and the only tool it had was
the dolly.

### What other games do

The most directly relevant writeup is [Vincent Michel's free-move-zone
camera design](https://www.gamedeveloper.com/design/third-person-camera-design-with-free-move-zone),
which spells out a *resolution order* when the character is hidden or the
camera is crowded:

1. rotate around the free-cam sphere (horizontal first),
2. move closer, respecting a minimum distance,
3. both together,
4. and only as a last resort, approach that minimum.

Two things there matter for this bug. **Rotation comes first and dollying
second** — the opposite of what this camera was doing. And the camera's
collider *slides* on the geometry it meets rather than being pushed inward:
"the camera should never come closer to the minimum distance from the free
move zone, particularly if it slides on ground/roof when being moved by the
player."

The same idea shows up as Cinemachine's *Preserve Camera Distance* strategy
(orbit around the obstruction rather than dolly through it), and the
[boxcast approach](https://straypixels.net/camera-boxcast/) sizes the probe
to the near plane for the same reason this camera's probe is a ball: a
zero-thickness ray reports clear while the near plane is already in the wall.

### What this camera does now

The player's rotation is no longer applied to the realized camera directly.
Input writes the *intent* only; the solver picks the rotation up in the same
frame — so it is still exactly 1:1 and undamped — and applies it through
`constrain_rotation`, which is collide-and-slide in the angular domain:

* Sweep the direction the player is asking for. If it leaves more than
  `WALL_COMFORT_FRACTION` (0.85) of the framing distance, nothing happens —
  the constraint is inert in open space, which is almost always.
* Otherwise, estimate which way clearance improves, in the camera's own
  screen plane, with two extra sweeps (a finite difference along `right` and
  `up`). That is the "surface" to slide along.
* Remove the component of the requested rotation that points into it,
  ramping from nothing at the comfort distance to *everything* at
  `WALL_FLOOR_FRACTION` (0.6). So rotation alone can never cost the camera
  more than 40% of its framing distance, and a rotation that would cost more
  simply doesn't happen. What remains slides the camera along the surface.
* Only the into-the-surface component is ever touched: turning away from a
  wall, or along it, tracks the request exactly.

One subtlety worth stating because it took a debugging round to find. When
the camera is pressed *square* against a face, the clearance gradient is a
**minimum**, not a slope: every direction improves, by almost nothing.
Normalising that near-zero vector yields a confident but meaningless "into
the wall" direction, which then blocks every rotation at once — precisely
where the player most needs to get out. So a gradient below
`WALL_GRADIENT_MIN_FRACTION` of the framing distance is treated as "no wall
direction here" and nothing is constrained.

The elective reposition (the cramped search, §11) now also waits for the
player to stop steering (`ELECTIVE_INPUT_IDLE`), and so does the decay back
toward intent. Safety repositioning — no camera position on this ray at all,
or a sustained total block — is never gated: it does not compete with the
player, it rescues them.

Pinned by two tests against a half-space wall, where "into" and "away" are
unambiguous (`free = 0.6 / u.x`): four seconds of pushing into the wall
never costs more than the floor allows and never loses sight of the marble;
and a single frame of turning away tracks the requested rotation to within
`1e-4` radians while buying clearance back.

### The commit hash in the overlay

`?debug=1`'s last line is now just the build's commit hash, baked in by
`app/build.rs`. It exists because "is the build I am looking at the build I
just pushed?" is otherwise unanswerable from a phone, and answering it wrong
costs a whole debugging round trip — a bug report against a stale deploy
looks exactly like a fix that didn't work, which is what happened between
§11's two fixes.

---

## 13. Phantom obstructions, and how fast a dolly should be

Reported after §12 shipped: still a "rapid dolly in on occlusion". The device
capture said what no amount of reasoning would have:

```
vis 0.55  d 2.063/4.816 (free 2.063)  clr 0.707/q0.053  steps 36
```

A camera pulled in to 2.06 — while its own clearance was **0.707**, against
a probe radius of **0.053**. Nothing was in the way. And `steps 36`, against
a step budget of 24.

### The bug: an exhausted march is not an obstruction

The sweep reports `exhausted` when it hits its step cap short of the goal,
and it could then only honestly claim clearance out to wherever it got. The
solver read that as "blocked, right here" and dollied to it.

That is fine when the march stops because it *found* something. It is
catastrophic when the march stops because it ran out of budget — which is
what happens on a loose distance field, where each step advances by the
underestimate rather than by the true distance. `Object::Morph`'s mid-blend
is loose by construction (its own doc says `|grad| < 1`), so in
`cube_sphere_morph` the march crawled a couple of units, gave up, and the
camera dived at a wall that was not there. `dev 0deg` in the capture is the
tell: no rotation ever ran, because on the sightline itself nothing was
wrong.

An exhausted march now contributes **no constraint at all**. That is safe
because the eye's own clearance is checked independently every frame — a
single `de`, which cannot crawl. If the camera really is near a surface,
that check finds it regardless of what the march did or didn't establish.
(The step budget also went from 24 to 40, which is a mitigation, not the
fix: a loose enough field will exhaust any budget.)

### How fast a pull-in should be

The second half of the complaint stands on its own: `PULL_IN_TAU` was
`0.05s`, which for a large correction is a cut rather than a dolly. It is
now chosen from the eye's *own* clearance, blending continuously between
`0.05s` when the eye is in contact with something and `0.30s` when it has
room — the size of the correction is irrelevant, only whether it is urgent.

Two cases motivate the split, and it is worth being precise about which is
which, because a plausible-sounding version of this change is wrong:

* For a **solid** obstruction, "the free distance collapsed" and "the eye is
  nearly touching something" are the *same event* — so the pull-in is fast,
  as it must be. Nothing changes there.
* For a **thin** one — a strut, a gear tooth, fractal filigree — the probe
  ball cannot pass but the eye is in open air well behind it with real
  clearance. That is where a 50ms snap is indefensible, and where the ease
  now applies.

Sitting *beyond* the sweep's bound is therefore not automatically unsafe,
which is why there is no clamp to it: a thin shell stops the probe while the
eye sails past into clear air on the far side. Only the eye's own clearance
can tell those apart, so that is what decides — and when it does go
negative, the camera takes the sweep's bound at once and the backstop
iterates until it is genuinely out.

Both halves are pinned:
`a_march_that_runs_out_of_budget_does_not_invent_an_obstruction` (a field
that underestimates by 20x, in empty space, must not move the camera) and
`a_thin_occluder_is_eased_past_not_snapped_to` (a shell thinner than the
probe ball: the camera crosses it, taking more than one frame to do it, and
never moves more than a fifth of the shot in any single frame).

## 14. The floor a marble is resting on

Reported after §13 shipped: *"the camera is still getting way too close to
the marble; an example is if the marble is up against a flat plane, if you
drag the camera into the plane it'll bring the distance to basically 0."*

Reproduced exactly, against an analytic half-space with a marble sitting on
it and a portrait phone's aspect (`dragging_at_a_floor`). Dragging steadily
downward settled here:

```
u.y=-0.214  d=0.450/1.460  free=0.455  vis=1.00  clr=+0.053  dev=109deg
```

The camera 12° under the horizon, at **31%** of its framing distance, with
the marble at half the screen — and 109° of stored disagreement with the
player, which is `MAX_CORRECTION`, i.e. the cap. Two independent faults,
both of which need this specific shape of scene to show up.

### Fault 1: every shallow ray over a plane exhausts its budget

§13 established that an exhausted march must contribute no constraint,
because a loose field crawls in open air and the camera would dive at
nothing. That was right, and it had a blind spot.

A swept sphere trace steps by `h - camera_radius`. Against a plane
approached at a shallow angle, `h` shrinks toward `camera_radius`
geometrically, so the steps shrink geometrically too and the march creeps
without ever formally touching down. Measured against the half-space above,
by angle below the horizon:

| angle | free  | exact | exhausted? |
|-------|-------|-------|------------|
| 3.4°  | 1.460 | 1.460 | no         |
| 4.6°  | 1.184 | 1.220 | **yes**    |
| 6.9°  | 0.811 | 0.814 | **yes**    |
| 9.2°  | 0.612 | 0.612 | **yes**    |
| 10.3° | 0.545 | 0.545 | no         |
| 17.2° | 0.330 | 0.330 | no         |

There is a band — and it is exactly the band where the floor first starts to
cost framing distance — in which the floor did not exist as far as the
wall-slide constraint was concerned. So the constraint never engaged there;
the player dragged straight through it and out the far side, where the march
does terminate and the free distance is already a third of what framing
wants. The distance solver, doing its job on a now-honest number, dollied to
it.

Note the third column: the stalled marches had *converged*. Their answers
were within 3% of exact. They were being discarded as worthless while being
almost right.

The fix is to distinguish the two ways a budget gets spent, which the final
sample's clearance already tells apart at no extra cost
(`visibility::GRAZING_STALL_RADII`):

* Stalled **onto a surface**: `h` is pinned just above `camera_radius` —
  that is *why* the steps went to zero. `t` is a real bound, and the ball
  demonstrably cannot get much further.
* Stalled **in open air**: `h` still has room at every sample; the field is
  merely slow. `t` means nothing. This is §13's case, unchanged.

The threshold is `1.5` camera radii, which sits between "effectively
touching" and any clearance a camera would care to keep.

### Fault 2: refused input was banked, not consumed

The `dev=109deg` above is the second bug, and it would have been a bug even
with a perfect sweep. Every frame the player pushed into the floor, the
part of the rotation the constraint removed stayed in the intent quaternion,
so the disagreement grew until it pinned against `MAX_CORRECTION`. The
player then had to drag 110° back before the camera moved *at all*.

A constraint you can feel resisting is fine. A control that has quietly
stopped being connected is not — and the second is what a hidden 110° of
banked input is, no matter how well-justified each frame's refusal was.

The refused rotation is now subtracted from the intent as well as from the
realized orientation. This is also what collide-and-slide means for input in
the games §12 borrowed from: the stick deflection that would drive the
camera into the wall is spent against the wall, and letting go and pushing
the other way moves the camera on the next frame. Pinned by
`pushing_into_the_floor_does_not_bank_a_dead_zone`, which asserts both
halves — no stored deviation, and a reversal that is honoured in full on the
first frame.

### Where it lands

Same scene, same drag, after both fixes:

```
u.y=-0.086  d=1.110/1.460  free=1.110  vis=1.00  clr=+0.055  dev=0deg
```

Stable for the whole run: 4.9° under the horizon at **76%** of the framing
distance (marble at 0.37 of the frame rather than 0.50), no banked
disagreement, and the full distance comes back as soon as the player lets go
(`the_shot_recovers_once_the_player_lets_go`).

76% rather than the 85% comfort threshold because the constraint's strength
ramps between `WALL_COMFORT_FRACTION` and `WALL_FLOOR_FRACTION`, and the
drag stalls partway up the ramp — resistance that builds is what keeps this
from reading as a hard stop.

The seven-scene probe is unchanged to three decimal places, which is the
point: this was a case the probe's flight paths never entered, not a
regression in one they did.

## 15. The same floor, zoomed out

§14 fixed the floor case and shipped, and the reply was "doesn't seem to have
had much of an effect", with device captures:

```
bunny: vis 1.00  d 0.816/3.016 (free 0.816)  size 0.335  zoom 3.10   (8.1 deg under)
bunny: vis 1.00  d 0.195/3.016 (free 0.195)  size 1.623  zoom 3.10  (36.7 deg under)
```

`zoom 3.10` is the part §14 missed. Zoom multiplies the framing distance, so
the camera wanted to be 3 units back from a marble sitting 0.15 above a
floor — a distance no angle more than ~2° below the horizon can deliver.
§14's regression test ran at zoom 1.0 and passed while the shipped build was
doing this.

Two mechanisms are supposed to resist. Both had a reason not to, and the
reasons are at opposite ends of the same range:

| angle under | free  | march steps needed | grad over the 0.05 rad probe |
|-------------|-------|--------------------|------------------------------|
| 2.6°        | 2.535 | 127                | 1.33                         |
| 3.6°        | 1.831 | 90                 | 0.81                         |
| 5.0°        | 1.319 | 64                 | 0.48                         |
| 8.1°        | 0.816 | 38                 | 0.21                         |
| 36.7°       | 0.192 | 5                  | 0.012                        |

The resistance band is `comfort` 2.56 down to `floor` 1.81, i.e. 2.6°–3.6°.
Resolving *any* of it needs 90–127 march steps against a budget of 40, so
every angle in the band came back `exhausted` — and §14's fix only rescued
stalls that had nearly converged, which at this zoom none had. Past the band
the march resolves easily, and there the *other* guard fired: the gradient
threshold was a fraction of the **framing** distance, `0.02 × 3.016 =
0.060`, which is larger than the actual gradient everywhere past 8°. So the
"equally tight in every direction" escape hatch fired on a gradient that was
perfectly well-defined, and the constraint switched itself off.

Unreachable from either side. Three fixes.

### Extrapolate through a stall instead of guessing about it

§14's rule — "a stall counts as a wall if the ball was nearly touching" — is
a proxy for what we actually want to know, and it is a proxy that degrades
exactly when the march is furthest from finishing.

Ask the question directly instead. At the stall we have a position, a
clearance, and (for four `de` calls) a gradient. Where the field is a real
distance field, that is a complete local model of the surface, and the
touchdown point is arithmetic — exact for a plane, which is what a grazing
stall usually is. Where it is *not* a real distance field the model is
worthless, and `|grad de|` says which case this is: a true SDF has `|grad| =
1`, and the fields that underestimate do it by going flat. §13's phantom
obstruction was a flat field; this floor is a `|grad| = 1` one. The same
number separates them, locally, with no reference to the step budget.

The prediction is then *checked* (one `de` at the predicted touchdown) and,
if it lands inside geometry, bisected back toward the last verified-clear
sample. That matters where the nearest surface changes identity along the
ray — down a corridor between two rows of pillars the linear model of the
pillar you are passing says nothing about the one you are approaching.

### Scale the gradient threshold to the query's error bar, not to the shot

`WALL_GRADIENT_MIN_RADII` is now a quarter of the *camera radius* — the
sweep's own resolution, since a touchdown located by backing off along the
ray can be off by about that much. Sizing it to the framing distance was
wrong by ~50× in the case that mattered.

That threshold was only ever that large to cover a real problem: pressed
square against a face, clearance improves in *both* screen directions, so
two one-sided probes both come back positive and their "gradient" points
diagonally away from a wall that is straight ahead. The fix for that is to
stop using one-sided differences. The gradient is now central (four probes
rather than two, paid only when already near a surface), so the symmetric
case cancels honestly and the threshold no longer has to paper over it.

### A correction that makes things worse is not a correction

Falling out of the pillar test: the last-resort backstop pulls the eye in
when its clearance is short, three passes of twice the shortfall. But
"closer to the marble" and "further from the geometry" are not the same
direction, and with the marble hugging a pillar and the eye in open air
behind it they are opposites. Unchecked, three passes walked the eye 0.019
*into* a wall it had been clear of — the last line of defence causing the
failure it exists to prevent. Each pass is now checked before it is taken,
and a final fallback takes the sweep's own verified-clear bound if the eye
is still inside anything.

### Where it lands

Same drag, at the reported zoom and marble size, held for 600 frames:

```
below 2.44deg  d 2.668/3.017  size 0.090  clr +0.036
```

88% of the framing distance, with the marble at 0.090 of the frame — which
is exactly what the framing rule asks for at this zoom, against the 0.335
and 1.623 in the captures. Pinned by `holds_its_distance_when_zoomed_out_too`,
kept separate from §14's zoom-1 test precisely because the zoom-1 test is
the one that said everything was fine.

The nine-scene probe is again unchanged to three decimals.
