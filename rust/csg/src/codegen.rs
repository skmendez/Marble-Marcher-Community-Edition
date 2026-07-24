//! M3: WGSL code generation from a `Fold`/`Object` tree.
//! See rust/DESIGN.md §5 and the C++ sources in
//! src/fractals/GLSLBase.hpp, GLSLCodeFactory.hpp, Fold*.hpp's `GLSL()`
//! methods, Object*.hpp's `GLSL()` methods, and the GLSL helpers in
//! `game_folder/shaders/compute/utility/distance_estimators.glsl`
//! (`de_sphere`, `de_box`, `mengerFold`, `planeFold`), ported to WGSL.

use crate::{fold::Fold, object::Object, Axis};

/// A string builder for generated WGSL: tracks indentation and a
/// monotonically increasing fresh-name counter.
///
/// ⚠ Fixes two C++ bugs (DESIGN.md §5) — do not replicate them:
///  - `FoldRepeat::GLSL` used a function-`static` depth counter for its loop
///    variable name, which leaks across separate code-generation calls
///    (regenerating a shader could produce different/colliding names).
///  - `ObjectClosest`/`ObjectIntersect`/`ObjectDifference::GLSL` used fixed
///    local names (`original_p_union`, `old_d_union`, ...), so a Union nested
///    inside a Union (or Intersect inside Intersect, etc.) emits two `let`s
///    with the same name, which is invalid WGSL (and would have been
///    silently-wrong GLSL shadowing bugs).
///
/// Every generated local and loop variable goes through [`CodeWriter::fresh`]
/// instead, so nested/repeated generation can never collide.
pub struct CodeWriter {
    out: String,
    indent: usize,
    /// Whether we're generating the color pass (`col_scene`) or the distance
    /// pass (`de_scene`). Orbit statements (`OrbitInit`/`OrbitMax`) and the
    /// orbit save/restore in combinators only emit when this is `true`.
    pub color_pass: bool,
    next_id: u32,
    /// Divides every `Fold::Repeat`'s loop bound at emission time (floored at
    /// [`MIN_REPEAT_ITERATIONS`]) -- used to generate a cheaper, lower-detail
    /// variant of the scene tree for the MRRM coarse and shadow/AO passes
    /// (`generate_coarse_shader`/`generate_shadow_shader`), neither of which
    /// needs the fine pass's full fractal surface fidelity (an approximate
    /// empty-space skip distance / occlusion test, respectively). `1` (the
    /// default, via [`CodeWriter::new`]) is a no-op -- [`generate_shader`]
    /// always uses it.
    ///
    /// This has to work at the *WGSL emission* level, not by transforming the
    /// `Object`/`Fold` tree beforehand: both of this codebase's real fractal
    /// scenes (`classic`'s `iters`, `menger_sponge`'s `depth`) drive their
    /// `Repeat` count from a runtime [`crate::IntValue::Param`], not a
    /// [`crate::IntValue::Const`] baked in at tree-construction time -- a
    /// tree-level transform could only ever reduce a `Const` count (a no-op
    /// for every scene that actually matters here), whereas dividing the
    /// *emitted expression* (`count.wgsl() / divisor`) works uniformly for
    /// both, since it's just wrapped around whatever runtime value the
    /// uniform buffer holds each frame.
    iteration_divisor: i32,
}

impl Default for CodeWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Floor for [`CodeWriter::iteration_divisor`]'s reduction -- these fractals'
/// characteristic recursive structure (Menger cross-voids, nested corner
/// folds) needs at least a couple of levels to read as "the same shape,
/// approximately" rather than degenerating to a single flat primitive, which
/// would make the coarse pass's hit-distance guess (or the shadow pass's
/// occlusion test) diverge enough from the fine pass's real geometry to
/// undermine MRRM's warm-start safety margin.
const MIN_REPEAT_ITERATIONS: i32 = 2;

impl CodeWriter {
    pub fn new() -> Self {
        Self {
            out: String::new(),
            indent: 0,
            color_pass: false,
            next_id: 0,
            iteration_divisor: 1,
        }
    }

    /// Returns a name unique across the lifetime of this `CodeWriter`:
    /// `{base}_{id}`.
    pub fn fresh(&mut self, base: &str) -> String {
        let id = self.next_id;
        self.next_id += 1;
        format!("{base}_{id}")
    }

    pub fn indent(&mut self) {
        self.indent += 1;
    }

    pub fn dedent(&mut self) {
        self.indent = self.indent.saturating_sub(1);
    }

    /// Appends one indented line (plus a trailing newline). An empty `line`
    /// writes a blank line with no leading whitespace.
    pub fn writeln(&mut self, line: &str) {
        if line.is_empty() {
            self.out.push('\n');
            return;
        }
        for _ in 0..self.indent {
            self.out.push_str("    ");
        }
        self.out.push_str(line);
        self.out.push('\n');
    }

    pub fn finish(self) -> String {
        self.out
    }

    /// Emit one [`Fold`] step. See the emission table in DESIGN.md §5.
    fn emit_fold(&mut self, fold: &Fold) {
        match fold {
            Fold::Abs => self.writeln("p = vec4<f32>(abs(p.xyz), p.w);"),
            Fold::Menger => self.writeln("menger_fold(&p);"),
            Fold::Rotate { axis, mat } => {
                // Cyclic component pairs (DESIGN.md §4): X->(y,z), Y->(z,x), Z->(x,y).
                let func = match axis {
                    Axis::X => "rot_yz",
                    Axis::Y => "rot_zx",
                    Axis::Z => "rot_xy",
                };
                self.writeln(&format!("p = {func}(p, {});", mat.wgsl()));
            }
            Fold::ScaleTranslate { scale, shift } => {
                // Scale multiplies all 4 components (including w, the scale
                // divisor); translate only affects xyz. Two statements
                // because WGSL has no swizzle assignment (`p.xyz += ...`).
                self.writeln(&format!("p = p * {};", scale.wgsl()));
                self.writeln(&format!("p = vec4<f32>(p.xyz + {}, p.w);", shift.wgsl()));
            }
            Fold::Plane { normal, offset } => {
                self.writeln(&format!(
                    "p = plane_fold(p, {}, {});",
                    normal.wgsl(),
                    offset.wgsl()
                ));
            }
            Fold::Modulo { axis, modulus } => {
                let c = match axis {
                    Axis::X => "x",
                    Axis::Y => "y",
                    Axis::Z => "z",
                };
                self.writeln(&format!("p.{c} = mod_fold(p.{c}, {});", modulus.wgsl()));
            }
            Fold::Series(folds) => {
                for f in folds {
                    self.emit_fold(f);
                }
            }
            Fold::Repeat { count, inner } => {
                let it = self.fresh("it");
                let bound = if self.iteration_divisor <= 1 {
                    count.wgsl()
                } else {
                    format!(
                        "max({}, {} / {})",
                        MIN_REPEAT_ITERATIONS,
                        count.wgsl(),
                        self.iteration_divisor
                    )
                };
                self.writeln(&format!("for (var {it}: i32 = 0; {it} < {bound}; {it}++) {{"));
                self.indent();
                self.emit_fold(inner);
                self.dedent();
                self.writeln("}");
            }
            Fold::OrbitInit(v) => {
                if self.color_pass {
                    self.writeln(&format!("orbit = {};", v.wgsl()));
                }
            }
            Fold::OrbitMax(v) => {
                if self.color_pass {
                    self.writeln(&format!("orbit = max(orbit, p.xyz * {});", v.wgsl()));
                }
            }
        }
    }

    /// Emit one [`Object`] node. See the emission table in DESIGN.md §5.
    fn emit_object(&mut self, obj: &Object) {
        match obj {
            Object::Sphere { radius } => {
                self.writeln(&format!("d = de_sphere(p, {});", radius.wgsl()));
            }
            Object::Cuboid { half_extent } => {
                self.writeln(&format!("d = de_box(p, {});", half_extent.wgsl()));
            }
            Object::Torus { major, minor } => {
                self.writeln(&format!(
                    "d = de_torus(p, {}, {});",
                    major.wgsl(),
                    minor.wgsl()
                ));
            }
            Object::Fractal { fold, base } => {
                self.emit_fold(fold);
                self.emit_object(base);
            }
            Object::Union(left, right) => self.emit_combine(left, right, Combine::Union),
            Object::Intersect(left, right) => self.emit_combine(left, right, Combine::Intersect),
            Object::Difference(left, right) => {
                self.emit_combine(left, right, Combine::Difference)
            }
            Object::Offset { base, offset } => {
                // Same local-units-over-p.w reasoning as `Object::de`'s
                // Offset arm: the base's `d` is already `/ p.w`-scaled, and
                // the divisor must be *this* node's entry-time `p.w` -- a
                // `Fractal` base's emitted folds mutate `p` (including `w`)
                // in place with no restore, so save it first (same
                // save-around-a-child shape as `emit_combine`'s `p_save`).
                let w_saved = self.fresh("w_save");
                self.writeln(&format!("let {w_saved} = p.w;"));
                self.emit_object(base);
                self.writeln(&format!("d = d - {} / {w_saved};", offset.wgsl()));
            }
            Object::Onion { base, thickness } => {
                // Same entry-time-`p.w` discipline as the `Offset` arm.
                let w_saved = self.fresh("w_save");
                self.writeln(&format!("let {w_saved} = p.w;"));
                self.emit_object(base);
                self.writeln(&format!("d = abs(d) - {} / {w_saved};", thickness.wgsl()));
            }
            // `emit_combine`'s save/restore shape, but the merge step is a
            // `mix` instead of a comparison -- including the color pass's
            // orbit, which must *blend* by the same factor rather than pick
            // a side (a picked orbit would pop at `t = 0.5` where neither
            // child "wins"). The `[0, 1]` clamp mirrors `Object::de`'s
            // Morph arm; it is what keeps the emitted field 1-Lipschitz
            // (sound) for out-of-range `t` params.
            Object::Morph { a, b, t } => {
                let p_saved = self.fresh("p_save");
                self.writeln(&format!("let {p_saved} = p;"));
                self.emit_object(a);
                let dl = self.fresh("dl");
                self.writeln(&format!("let {dl} = d;"));
                self.writeln(&format!("p = {p_saved};"));
                let ol = if self.color_pass {
                    let name = self.fresh("ol");
                    self.writeln(&format!("let {name} = orbit;"));
                    Some(name)
                } else {
                    None
                };
                self.emit_object(b);
                let tm = self.fresh("tm");
                self.writeln(&format!("let {tm} = clamp({}, 0.0, 1.0);", t.wgsl()));
                self.writeln(&format!("d = mix({dl}, d, {tm});"));
                if let Some(ol) = &ol {
                    self.writeln(&format!("orbit = mix({ol}, orbit, {tm});"));
                }
            }
        }
    }

    /// Shared shape for `Union`/`Intersect`/`Difference` (DESIGN.md §5): save
    /// `p` (and `orbit`, in the color pass) *before* the left child, save the
    /// left child's `d` *after* it, restore `p`, emit the right child, then
    /// compare. `Difference` negates the right child's `d` first.
    fn emit_combine(&mut self, left: &Object, right: &Object, kind: Combine) {
        let p_saved = self.fresh("p_save");
        self.writeln(&format!("let {p_saved} = p;"));
        self.emit_object(left);
        let dl = self.fresh("dl");
        self.writeln(&format!("let {dl} = d;"));
        self.writeln(&format!("p = {p_saved};"));
        let ol = if self.color_pass {
            let name = self.fresh("ol");
            self.writeln(&format!("let {name} = orbit;"));
            Some(name)
        } else {
            None
        };
        self.emit_object(right);
        if kind == Combine::Difference {
            self.writeln("d = -d;");
        }
        let op = match kind {
            Combine::Union => "<",
            Combine::Intersect | Combine::Difference => ">",
        };
        match &ol {
            Some(ol) => self.writeln(&format!(
                "if ({dl} {op} d) {{ d = {dl}; orbit = {ol}; }}"
            )),
            None => self.writeln(&format!("if ({dl} {op} d) {{ d = {dl}; }}")),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Combine {
    Union,
    Intersect,
    Difference,
}

/// Bindings + `SceneUniforms` (verified against bevy_sprite 0.16.1, material
/// bind group 2 — DESIGN.md §5). `misc2`/`bounding` are the second-round-perf
/// additions (ray-sphere clip + half-res shadow/AO): `misc2.x` is the shadow
/// pass's own render-target height (only meaningful on the *fine* pass's own
/// material, mirroring how `misc.z` is each pass's own resolution -- see
/// `sample_shadow`'s doc), `misc2.y` is the fine pass's `MM_SHADOW_LOD`
/// on/off flag (same uniform-flag-not-entity-toggle convention as `misc.w`'s
/// MRRM flag, for the same A/B-comparability reason); `misc2.w` is the
/// marble cubemap's current Y-axis rotation angle in radians (fine pass's
/// material only -- deterministic function of the shared simulation tick,
/// not wall-clock time, so every multiplayer peer renders the same phase
/// for the same tick; see `render.rs`'s `update_frame_data_impl`'s doc);
/// `bounding.xyz` is the
/// scene's world-space bounding-sphere center, `bounding.w` its radius, with
/// `radius <= 0.0` meaning "no bound" (either the scene is genuinely
/// unbounded -- `marble_csg::Object::bounding_sphere` returned `None` -- or
/// this uniform was never populated), which `ray_sphere_clip` (`MARCH_CORE`)
/// treats as "don't clip, march the full range" rather than "everything
/// misses". `misc3.x` is the fine pass's debug-view-mode selector (fine
/// pass only -- see `MARCHER`'s `fragment`; `w` unused, added as its own
/// field rather than squeezed into `misc`/`misc2` since both of those are
/// already fully occupied): `0` off, `1` the fine pass's own per-pixel
/// ray-march step-count heatmap (the original `?stepheat=1` view), `2` the
/// MRRM coarse pre-pass's own per-texel step count upscaled onto the fine
/// image, `3` the coarse pre-pass's own cached hit distance upscaled the
/// same way -- see `rust/app/src/live_debug.rs`'s `DebugViewMode` for the
/// Rust-side enum this integer encodes, and why it's a live, no-reload
/// toggle rather than a `Config`-seeded-at-startup one like every other
/// debug flag here. `misc3.y` is the fine pass's exposure multiplier for
/// the ACES tonemap (`MARCHER`'s `tonemap`; `?exposure=`/`MM_EXPOSURE`,
/// default 1.0 -- non-positive values fall back to 1.0 in the shader so an
/// unset uniform can't black the frame out). `misc3.z` is the fine pass's
/// material gamma for the terrain-albedo boost (`MARCHER`'s `fragment`;
/// `?material_gamma=`/`MM_MATERIAL_GAMMA`, default 0.5 = albedo squared,
/// same non-positive-means-unset guard).
///
/// The single authoritative field list for the GPU `SceneUniforms` ABI --
/// every `vec4<f32>`, same order the Rust-side `render.rs::SceneUniforms`
/// struct declares them in. This WGSL struct text below is *generated*
/// from this list rather than hand-typed a second time; `render.rs` can't
/// generate its own field declarations from this same list (no macro/derive
/// wiring for that here, and it's not worth building one for an 11-field,
/// rarely-changing struct), so it has its own `#[test]` that asserts its
/// field order against this exact constant -- a mismatch is a loud test
/// failure instead of a silent wrong-looking render.
pub const SCENE_UNIFORMS_FIELD_NAMES: [&str; 11] = [
    "cam_pos", "cam_right", "cam_up", "cam_forward", "sun", "sun_col", "bg_col", "misc", "misc2",
    "bounding", "misc3",
];

fn bindings() -> String {
    let mut s = String::from("struct SceneUniforms {\n");
    for name in SCENE_UNIFORMS_FIELD_NAMES {
        s.push_str("    ");
        s.push_str(name);
        s.push_str(": vec4<f32>,\n");
    }
    s.push_str(
        "}\n\n@group(2) @binding(0) var<uniform> scene: SceneUniforms;\n\
         @group(2) @binding(1) var<storage, read> params: array<vec4<f32>>;\n",
    );
    s
}

/// Static WGSL helper library, ported from
/// `game_folder/shaders/compute/utility/distance_estimators.glsl`
/// (`de_sphere`, `de_box`, `mengerFold`, `planeFold`) plus a euclidean
/// `fmodulo`/`mod_fold` pair (port of `FoldModulo`'s `fmodulo`) and
/// `rot_xy`/`rot_yz`/`rot_zx` (replacing GLSL's in-place `p.xy *= mat`,
/// which WGSL's no-swizzle-assignment rule forbids).
const HELPERS: &str = "\
fn de_sphere(p: vec4<f32>, r: f32) -> f32 {
    return (length(p.xyz) - r) / p.w;
}

fn de_box(p: vec4<f32>, s: vec3<f32>) -> f32 {
    let a = abs(p.xyz) - s;
    return (min(max(max(a.x, a.y), a.z), 0.0) + length(max(a, vec3<f32>(0.0)))) / p.w;
}

// Torus around the Y axis: ring radius `major` in the XZ plane, tube
// radius `minor` (mirrors `Object::Torus`'s CPU `de` exactly).
fn de_torus(p: vec4<f32>, major: f32, minor: f32) -> f32 {
    let q = vec2<f32>(length(p.xz) - major, p.y);
    return (length(q) - minor) / p.w;
}

fn menger_fold(p: ptr<function, vec4<f32>>) {
    var a = min((*p).x - (*p).y, 0.0);
    (*p).x -= a;
    (*p).y += a;
    a = min((*p).x - (*p).z, 0.0);
    (*p).x -= a;
    (*p).z += a;
    a = min((*p).y - (*p).z, 0.0);
    (*p).y -= a;
    (*p).z += a;
}

fn plane_fold(p: vec4<f32>, n: vec3<f32>, o: f32) -> vec4<f32> {
    let d = 2.0 * min(0.0, dot(p.xyz, n) - o);
    return vec4<f32>(p.xyz - d * n, p.w);
}

// Euclidean modulo (result always in [0, b)); port of FoldModulo::fmodulo.
fn fmodulo(a: f32, b: f32) -> f32 {
    let r = a % b;
    return select(r, r + b, r < 0.0);
}

fn mod_fold(x: f32, m: f32) -> f32 {
    return abs(fmodulo(x - m * 0.5, m) - m * 0.5);
}

fn rot_xy(p: vec4<f32>, m: mat2x2<f32>) -> vec4<f32> {
    let v = m * p.xy;
    return vec4<f32>(v.x, v.y, p.z, p.w);
}

fn rot_yz(p: vec4<f32>, m: mat2x2<f32>) -> vec4<f32> {
    let v = m * p.yz;
    return vec4<f32>(p.x, v.x, v.y, p.w);
}

fn rot_zx(p: vec4<f32>, m: mat2x2<f32>) -> vec4<f32> {
    let v = m * p.zx;
    return vec4<f32>(v.y, p.y, v.x, p.w);
}
";

/// Extra binding the *fine* marcher shader alone needs, on top of the
/// `scene`/`params` bindings every pass shares (`bindings()` above): the
/// coarse pass's cached hit-distance render target (MRRM — see `mrrm.rs` in
/// the app crate), read with `textureLoad` (exact texel, no sampler needed)
/// rather than `textureSample` -- a coarse texel is already a
/// many-fine-pixels-wide approximation, so exact sampling loses nothing an
/// interpolated sample would have given anyway.
///
/// ⚠ Known issue, environment-specific, *not* a bug in this shader: on this
/// project's llvmpipe (Mesa's software Vulkan renderer, used as this
/// project's native/CI fallback where no real GPU is available) test
/// environment, feeding *any* value derived from a texture fetch of this
/// binding -- via `textureLoad` here, or `textureSample` (tried as a
/// candidate fix, produced an identical result) -- into the fine march
/// loop's starting `t` (`march_scene` in `MARCH_CORE`, a 256-iteration
/// branchy loop) reproducibly segfaults llvmpipe's shader JIT. Root-caused
/// by careful bisection (see this change's session notes/commit message for
/// the full trail): naga validates the generated shader without error;
/// simpler variants (the texture fetch computed but *not* fed to
/// `march_scene`, or fed into a tiny/trivial loop) run without crashing;
/// the `MM_MRRM=0` fallback (which never executes this data flow at
/// runtime, since the `if` guarding it is skipped) never crashes either.
/// This points at a genuine llvmpipe JIT compiler limitation triggered by
/// this specific code shape, not a spec violation or logic error -- real
/// GPU drivers (native Vulkan/Metal/DX12) and the browser WebGPU backend
/// this project also targets (`--features web`) are a completely different
/// code path and not expected to share it. Shipped as originally designed
/// (`textureLoad`, no sampler) since that's the simpler, more obviously
/// correct approach and switching to `textureSample` bought nothing.
/// Verify this change's actual on-screen behavior on real GPU hardware or
/// in a browser build before relying on llvmpipe-only testing.
const COARSE_TEXTURE_BINDING: &str = "\
@group(2) @binding(2) var coarse_tex: texture_2d<f32>;
";

/// The fine pass's extra binding for the half-resolution shadow/AO pass's
/// cached visibility (`shadow_pass.rs`) -- R = shadow visibility, A =
/// traveled distance (`sample_shadow`'s doc). `textureLoad` (not
/// `textureSample`): `sample_shadow` does its own depth-aware 4-tap blend
/// by hand, so no hardware sampler is needed here either, same reasoning as
/// `COARSE_TEXTURE_BINDING`.
const SHADOW_TEXTURE_BINDING: &str = "\
@group(2) @binding(3) var shadow_tex: texture_2d<f32>;
";

/// The fine pass's extra binding for the live marble list (multiplayer
/// milestone 0): one `vec4<f32>` per marble (`xyz = center, w = radius`,
/// `w <= 0.0` meaning "inactive/hidden", same convention the old single
/// `scene.marble` uniform used), mirroring `bindings()`'s own
/// `params: array<vec4<f32>>` storage binding rather than a fixed-size
/// uniform array (sidesteps WGSL's uniform-array padding/alignment rules
/// entirely, and `arrayLength(&marbles)` gives the fragment shader the
/// active count for free with no separate count field to keep in sync).
/// Only the fine pass reads it -- `COARSE_MARCHER`/`SHADOW_MARCHER` skip all
/// shading (including the marble reflection), so neither needs this.
const MARBLE_BUFFER_BINDING: &str = "\
@group(2) @binding(4) var<storage, read> marbles: array<vec4<f32>>;
";

/// The marble's cubemap texture (`render.rs`'s `MarbleCubemap`, loaded from
/// `assets/marble_cubemap.png`): a real `texture_cube`, so unlike
/// `coarse_tex`/`shadow_tex` this genuinely needs a paired `sampler` for
/// `textureSample`'s hardware-filtered, direction-vector lookup (a cube
/// face's nearest-texel `textureLoad` equivalent isn't directly expressible
/// the way it is for a 2D texture -- filtering across face seams is exactly
/// what the sampler buys here). Fine pass only, same reasoning as
/// `MARBLE_BUFFER_BINDING`: the coarse/shadow passes skip all shading.
const MARBLE_TEXTURE_BINDING: &str = "\
@group(2) @binding(5) var marble_cubemap: texture_cube<f32>;
@group(2) @binding(6) var marble_cubemap_sampler: sampler;
";

/// The over-relaxed march loop + its supporting constants, shared verbatim
/// (DESIGN.md-style "no duplicated logic") between the coarse pass
/// (`COARSE_MARCHER`) and the fine pass (`MARCHER`) -- both call
/// `march_scene` with their own starting `t`/`pixel_angle`; only what happens
/// before (ray setup) and after (miss sentinel vs. full shading) differs.
const MARCH_CORE: &str = "\
const MAX_STEPS: i32 = 256;
// The fine pass's own *default* step budget (distinct from `MAX_STEPS`,
// which the coarse/shadow passes still use unchanged for their own
// from-`t=0` marches, and which also still serves as `march_scene`'s
// documented \"safe fallback\" ceiling) -- warm-started from MRRM's coarse
// hit distance (`t0` below), so a full-`MAX_STEPS` budget was measured
// overkill: verified visually across the Menger scenes (silhouette
// corners, recursive tunnel openings, close-up marble framing) and the
// Demo scene at several camera angles/distances with no missed geometry,
// no popping, no banding at 128 -- half the old budget for the pass that
// actually costs the most per pixel.
const FINE_MAX_STEPS: i32 = 128;
const MAX_DIST: f32 = 30.0;
// Starting over-relaxation factor for the primary march (Enhanced Sphere
// Tracing; ported from MMCE's ray_march, utility/ray_marching.glsl). MMCE
// reads this from a runtime `overrelax` uniform we don't have a located
// concrete value for; 1.2-1.5 is the well-established useful range in the
// sphere-tracing literature for this technique -- picked 1.4 as a
// reasonably aggressive middle value and verified visually (see codegen
// tests / this change's commit message for before/after screenshots).
const OVERRELAX: f32 = 1.4;
// Floor for the distance-scaled hit threshold, so it never collapses to
// (near-)zero extremely close to the camera/light. MMCE's equivalent is a
// `MIN_DIST` uniform; we don't have one, so this is a fixed small constant.
const MIN_HIT_DIST: f32 = 1e-5;
// Threshold for recognizing `COARSE_MARCHER`'s R-channel miss sentinel
// (`-1.0`, its doc) when reading the coarse render target back, e.g.
// `MARCHER`'s sky-miss-skip check -- deliberately *not* a bare `<= 0.0` test: a genuine
// hit's reported distance is `march_scene`'s `candidate_t`, which for
// domain-repeated (`Fold::Repeat`-heavy) scenes with a badly cheapened
// coarse-iteration DE (verified live: the `demo` scene) can itself land at
// or slightly below zero via the over-relaxed march's backtrack-on-overstep
// step (`t += (1.0 - omega) * prev_h`, which *decreases* `t`) -- a real,
// if degenerate, hit, not the `-1.0` sentinel. `<= 0.0` would misclassify
// that as a miss (confirmed live: it did, rendering the whole `demo` scene
// as flat sky before this threshold was tightened). `-0.5` sits comfortably
// below any plausible near-zero backtrack overshoot while still well above
// the exact `-1.0` sentinel, so it only ever matches a genuine miss.
const COARSE_MISS_SENTINEL_THRESHOLD: f32 = -0.5;

fn map(p: vec3<f32>) -> f32 {
    return de_scene(vec4<f32>(p, 1.0));
}

// Ray-vs-bounding-sphere pre-test (`scene.bounding`, from
// `marble_csg::Object::bounding_sphere`): every march pass calls this right
// after ray setup so a ray pointed at open sky can skip `march_scene`
// entirely instead of stepping all the way out to `max_d` through empty
// space. Returns `(t_near, t_far)`, both clamped to `[0, MAX_DIST]` --
// `radius <= 0.0` (no bound: an unbounded scene, or the uniform was never
// populated) returns `(0.0, MAX_DIST)`, i.e. \"don't clip, march the full
// range\" rather than \"everything misses\" -- and a clean miss returns
// `t_near > t_far` (`(1.0, -1.0)`), which every caller checks for before
// marching at all.
fn ray_sphere_clip(ro: vec3<f32>, rd: vec3<f32>, center: vec3<f32>, radius: f32) -> vec2<f32> {
    if (radius <= 0.0) {
        return vec2<f32>(0.0, MAX_DIST);
    }
    let oc = ro - center;
    let b = dot(oc, rd);
    let c = dot(oc, oc) - radius * radius;
    let h = b * b - c;
    if (h < 0.0) {
        return vec2<f32>(1.0, -1.0);
    }
    let sh = sqrt(h);
    return vec2<f32>(max(-b - sh, 0.0), min(-b + sh, MAX_DIST));
}

// Result of `march_scene`: `t` is the best (lowest relative-error) candidate
// distance found (see the loop's `candidate_t` bookkeeping below); `hit`
// is false for a MAX_DIST/MAX_STEPS miss (matches the pre-MRRM `hit_frac`
// semantics exactly -- callers must not treat `t` as meaningful when `hit`
// is false); `iters` is the loop iteration the break/exhaustion happened at,
// needed by the fine pass's ambient-occlusion term.
struct MarchResult {
    t: f32,
    hit: bool,
    iters: i32,
}

// Over-relaxed sphere tracing with backtrack-on-overstep (Enhanced Sphere
// Tracing; ported from MMCE's ray_march, utility/ray_marching.glsl) instead
// of naive `t += d` stepping: takes bigger-than-safe steps (`omega > 1`) to
// converge faster through well-behaved regions, and when a step turns out to
// have overshot the surface, backs off and relaxes `omega` toward 1 (more
// conservative) rather than tunneling through. Also tracks the best
// (lowest relative-error) candidate hit point seen so far, matching MMCE's
// `candidate_td`/`candidate_error` -- not currently used for a fallback
// \"soft hit\" if the loop exhausts without a clean break (that's treated as
// a miss, same as before this change), but kept since it's needed for `t`
// to reflect the tightest point found once the break condition below fires.
//
// `t0` is the starting distance -- `0.0` for a from-the-camera march (the
// coarse pass always, and the fine pass when MRRM is disabled or the coarse
// pass reported a miss for this texel), or the MRRM coarse-pass's
// backed-off hit-distance guess otherwise (`render.rs`/`mrrm.rs`). Starting
// past zero is exactly as safe as starting at zero: `candidate_err`'s
// distance-scaled threshold and the backtrack-on-overstep logic don't assume
// `t0 == 0` anywhere, so a wrong (too-far or too-near) `t0` just costs extra
// steps or triggers a backtrack, never a missed/incorrect hit -- see
// `render.rs`'s `fragment` doc for why this makes MRRM safe even when the
// coarse guess is wrong.
//
// `max_d` is the far cutoff -- `MAX_DIST` when there's no bounding-sphere
// clip to tighten it, or `ray_sphere_clip`'s `t_far` otherwise (always
// `<= MAX_DIST`, see its doc); same never-unsafe-only-wasteful reasoning as
// `t0` applies to a loose `max_d`, so a caller can always pass `MAX_DIST`
// as a safe fallback.
//
// `max_steps` is the loop bound every caller passes explicitly (rather than
// this function reading the `MAX_STEPS` constant itself): the coarse/shadow
// passes always pass `MAX_STEPS` unchanged, but the fine pass's caller
// (`MARCHER`) can pass a smaller runtime value when the `?perfprobe=`
// diagnostic (`perfprobe.rs`) wants to measure how much of the fine pass's
// GPU cost comes from its step budget -- see `SceneUniforms::misc2`'s doc
// for the uniform this rides in on.
fn march_scene(ro: vec3<f32>, rd: vec3<f32>, t0: f32, pixel_angle: f32, max_d: f32, max_steps: i32) -> MarchResult {
    var t = t0;
    var prev_h = 0.0;
    var omega = OVERRELAX;
    var candidate_t = t0;
    var candidate_err = 1e20;
    var iters = 0;
    var hit_frac = false;
    for (var i = 0; i < max_steps; i++) {
        iters = i;
        if (t > max_d) {
            break;
        }
        let h = map(ro + rd * t);
        if (prev_h * omega > max(h, 0.0) + max(prev_h, 0.0)) {
            t += (1.0 - omega) * prev_h;
            prev_h = 0.0;
            omega = (omega - 1.0) * 0.55 + 1.0;
        } else {
            let err = h / max(t, MIN_HIT_DIST);
            if (err < candidate_err) {
                candidate_err = err;
                candidate_t = t;
                if (h < 0.0) {
                    hit_frac = true;
                    break;
                }
                if (h < max(t * pixel_angle, MIN_HIT_DIST)) {
                    hit_frac = true;
                    break;
                }
            }
            t += h * omega;
            prev_h = h;
        }
    }
    return MarchResult(candidate_t, hit_frac, iters);
}
";

/// Coarse MRRM pre-pass fragment shader (see `mrrm.rs` in the app crate):
/// marches `de_scene` from `t=0` exactly like the fine pass used to
/// pre-MRRM, but skips all shading (no `calc_normal`/`col_scene`/`shadow`)
/// and writes just the resulting hit distance -- or `-1.0` on a miss -- into
/// the R channel of its render target (G/B unused, A=1), for the fine pass
/// to read back as a starting-`t` guess.
///
/// Returns a full `vec4<f32>` (not a bare `f32`) even though only R is
/// meaningful: Bevy's 2D camera pipeline always draws into an intermediate
/// "main texture" at a fixed format it picks itself -- `bevy_render`
/// 0.16.1's `prepare_view_targets` uses `ViewTarget::TEXTURE_FORMAT_HDR`
/// (`Rgba16Float`) when the camera has `hdr: true` set (`mrrm.rs`'s
/// `CoarseCamera`), or `TextureFormat::bevy_default()` otherwise -- *not*
/// whatever format the camera's actual target `Image` is; a later
/// (`bevy_core_pipeline::upscaling`) full-screen blit pass converts that
/// intermediate into the real target's actual format. Both of those
/// intermediate formats are 4-channel, so the draw pipeline's fragment
/// output must be a `vec4<f32>` regardless of the destination texture's own
/// channel count -- returning a bare scalar here mismatches what the base
/// `Mesh2dPipeline` pipeline descriptor declares and fails pipeline
/// validation (confirmed by trying it: \"RenderPipeline ... uses
/// attachments with formats \\[Some(Rgba8UnormSrgb)\\]\" vs. our
/// pipeline's declared format).
///
/// Named `fragment` (not `fragment_coarse`) even though it lives in its own
/// module, separate from the fine shader's `fragment` below: bevy_sprite's
/// `Mesh2dPipeline` hardcodes the fragment stage's entry-point name to
/// `\"fragment\"` for every `Material2d` (verified in bevy_sprite 0.16.1's
/// `mesh2d/mesh.rs::specialize`, which only lets a material swap the shader
/// *module* via `Material2d::fragment_shader()`, not the entry-point name) --
/// so this has to be its own complete WGSL module (its own `de_scene`/
/// `col_scene`/`march_scene` copies, generated fresh by
/// `generate_coarse_shader`) rather than a second entry point coexisting
/// with `fragment` in the same module `generate_shader` produces.
const COARSE_MARCHER: &str = "\
@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let ndc = vec2<f32>(mesh.uv.x * 2.0 - 1.0, 1.0 - mesh.uv.y * 2.0);
    let aspect = scene.misc.x;
    let ro = scene.cam_pos.xyz;
    let rd = normalize(
        scene.cam_right.xyz * ndc.x * aspect
        + scene.cam_up.xyz * ndc.y
        + scene.cam_forward.xyz * scene.cam_forward.w
    );

    // Cone-angle threshold computed from *this* (coarse) pass's own
    // resolution (`scene.misc.z` here is the coarse render target's height,
    // written per-frame by `mrrm.rs` -- deliberately not the fine
    // resolution): a coarse texel covers many fine pixels' worth of solid
    // angle, so using the fine cone angle would let this march step past
    // thin geometry a real fine ray would actually hit.
    let half_fov = atan(1.0 / scene.cam_forward.w);
    let pixel_angle = 2.0 * half_fov / max(scene.misc.z, 1.0);

    let clip = ray_sphere_clip(ro, rd, scene.bounding.xyz, scene.bounding.w);
    if (clip.x > clip.y) {
        return vec4<f32>(-1.0, 0.0, 0.0, 1.0);
    }

    // G channel: this pass's own `iters` count, for the fine pass's
    // `CoarseStepHeat` debug view (`scene.misc3.x == 2`, `MARCHER`'s
    // `fragment`) -- R/B/A keep their original meaning (hit distance / 0.0 /
    // 1.0) unchanged; G was unused before this, so this is additive, not a
    // format change (`CoarseRenderTarget`'s `Rgba16Float` target already has
    // room, `mrrm.rs`). A miss reports `MAX_STEPS` here (the full budget was
    // spent and still found nothing), matching a genuinely maximal-cost
    // pixel rather than the misleading \"0 steps\" a bare default would imply.
    let result = march_scene(ro, rd, clip.x, pixel_angle, clip.y, MAX_STEPS);
    if (result.hit) {
        return vec4<f32>(result.t, f32(result.iters), 0.0, 1.0);
    }
    return vec4<f32>(-1.0, f32(MAX_STEPS), 0.0, 1.0);
}
";

/// The half-resolution shadow/AO pre-pass's fragment shader
/// (`shadow_pass.rs`/`generate_shadow_shader`): ray setup, a ray-sphere
/// clip, then a march starting at the bounding-sphere clip entry point
/// (`clip.x`) -- deliberately *not* warm-started from MRRM's coarse buffer,
/// unlike `MARCHER`'s `fragment` -- then `calc_normal`/`shadow` on a genuine
/// hit, packed as `vec4(shadow_visibility, 0.0, 0.0, traveled_distance)`.
/// The `A` channel (traveled distance) is what `MARCHER`'s `sample_shadow`
/// needs for its depth-aware resample -- a miss (or bounding-sphere clip
/// miss) writes `MAX_DIST` there deliberately: `sample_shadow`'s weighting is
/// `1/(sz^2+(td-td_i)^2)`, so a \"far away\" sentinel just gets naturally
/// down-weighted against a real nearby hit rather than needing a branch on
/// the fine-pass side. Doesn't need full fractal fidelity (only an
/// approximate occlusion test), so its shader is generated with a reduced
/// `Fold::Repeat` iteration count, same as `COARSE_MARCHER` (see
/// `CodeWriter::iteration_divisor`).
///
/// This pass used to warm-start its own march from a single, uninterpolated
/// nearest-texel read of the coarse pass's cached hit distance (same
/// technique `MARCHER` uses), gated behind the same `scene.misc.w` MRRM
/// flag. Removed this session after a live investigation (mobile bug
/// report: the *entire* visible terrain on the default
/// `menger_oscillating_sphere` scene flipping between flat full-sun and flat
/// full-shadow under a camera nudge of a few hundredths of a unit) traced it
/// to this exact warm-start: the coarse pass's DE runs at
/// `COARSE_ITERATION_DIVISOR`-reduced fold iterations, and near a
/// Menger-fold crease Enhanced Sphere Tracing can have multiple close
/// candidate roots -- a single nearest-texel, non-interpolated sample of
/// that cheapened DE can warm-start this march onto a different fold than
/// the true nearest surface, especially right at a crease. Unlike the fine
/// pass's own coarse warm-start (this session's `sky_confirmed_miss`, doc
/// above, added a `map()`-based corroboration probe specifically because
/// this failure mode is real), this pass's hit point feeds directly into
/// the sun-shadow test *and*, via the fine pass's own separately-gated
/// shadow-tier warm-start reuse (`MARCHER`'s `fragment` doc), can propagate
/// a wrong fold into the fine pass's own warm-start too -- so a wrong fold
/// here isn't merely extra steps (self-correcting, see `march_scene`'s
/// over-relaxation doc), it's a materially wrong shadow value that can flip
/// for the whole frame at once, since every pixel's coarse texel is subject
/// to the same near-crease instability under a small camera change.
/// Confirmed live this session: at a fixed camera angle on the default
/// scene, with only trivial (<0.5-unit) camera drift from the marble
/// settling under gravity, cycling `mrrm` on/off/on swung the *fine* pass's
/// own avg steps/px between roughly 9.9 -> 11.3 -> 6.2 -- i.e. the
/// coarse-driven warm-start chain (coarse -> shadow -> fine) is genuinely
/// unstable under tiny position changes on this scene, matching the
/// hypothesis. This pass is already half-resolution with only
/// `SHADOW_STEPS` (16) fixed iterations, so the warm-start's perf upside
/// here is small next to the fine pass's own (measured, real) MRRM win --
/// not worth the correctness risk. The coarse-texture binding
/// (`COARSE_TEXTURE_BINDING`) is still declared in this pass's generated
/// source and bind-group layout (`shadow_pass.rs`'s `ShadowMarcherMaterial`)
/// so neither needs restructuring, it's simply unread now.
const SHADOW_MARCHER: &str = "\
@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let ndc = vec2<f32>(mesh.uv.x * 2.0 - 1.0, 1.0 - mesh.uv.y * 2.0);
    let aspect = scene.misc.x;
    let ro = scene.cam_pos.xyz;
    let rd = normalize(
        scene.cam_right.xyz * ndc.x * aspect
        + scene.cam_up.xyz * ndc.y
        + scene.cam_forward.xyz * scene.cam_forward.w
    );

    // This pass's *own* resolution's cone angle (`scene.misc.z` is this
    // pass's own render-target height, written by `shadow_pass.rs` -- same
    // per-pass-own-resolution convention as `COARSE_MARCHER`).
    let half_fov = atan(1.0 / scene.cam_forward.w);
    let pixel_angle = 2.0 * half_fov / max(scene.misc.z, 1.0);

    let clip = ray_sphere_clip(ro, rd, scene.bounding.xyz, scene.bounding.w);
    if (clip.x > clip.y) {
        return vec4<f32>(1.0, 0.0, 0.0, MAX_DIST);
    }

    // Deliberately *not* warm-started from MRRM's coarse buffer -- see this
    // constant's own doc above for why (a real correctness bug: a single
    // uninterpolated nearest-texel read of the coarse pass's cheapened-fold
    // DE could land this march on the wrong fold near a Menger-fold crease,
    // producing a materially wrong shadow value rather than just extra
    // steps). Always starts at the bounding-sphere clip entry point, the
    // same behavior every pass has when `?mrrm=0`.
    let march = march_scene(ro, rd, clip.x, pixel_angle, clip.y, MAX_STEPS);
    if (!march.hit) {
        return vec4<f32>(1.0, 0.0, 0.0, MAX_DIST);
    }

    let p = ro + rd * march.t;
    let eps = 1e-4 * max(march.t, 0.05);
    let n = calc_normal(p, eps);
    let sh = shadow(p + n * 2.0 * eps, scene.sun.xyz, pixel_angle);
    return vec4<f32>(sh, 0.0, 0.0, march.t);
}
";

/// `calc_normal`/`shadow` -- shared verbatim (same "no duplicated logic"
/// reasoning as `MARCH_CORE`) between the fine pass (`MARCHER`) and the new
/// half-resolution shadow/AO pass (`SHADOW_MARCHER`): both need a surface
/// normal and an occlusion query against the sun, just at different
/// resolutions. Not needed by the coarse (`COARSE_MARCHER`) pass, which
/// skips all shading -- but harmless dead code there is worse than a third
/// copy, so this lives in its own block included only where used.
const SHADING_CORE: &str = "\
// Reduced from 24 to 16 this session -- the improved-sphere-tracing soft
// shadow technique below already converges faster per step than a naive
// `min(d/t)` march would, and this pass only ever feeds a half-resolution,
// depth-aware-resampled visibility term into the fine pass (`sample_shadow`),
// not a direct per-fine-pixel shadow ray -- re-verified visually across the
// Menger scenes' corner/tunnel shadows and the Demo scene at several sun
// angles with no visible new banding/acne versus 24.
const SHADOW_STEPS: i32 = 16;
// Angular size (tangent of the half-angle) of the sun disc, controlling
// shadow penumbra softness in the improved soft-shadow technique below --
// MMCE passes this in as a per-scene `light_angle` parameter we don't have
// a located concrete value for; 0.06 gives a fairly crisp but not
// razor-hard directional-sun-like shadow, chosen by visual inspection.
const LIGHT_ANGLE: f32 = 0.06;

// Tetrahedral central-difference normal (4 `map()` calls, down from the
// prior axis-aligned 6-tap version's 6) -- Inigo Quilez's well-established
// cheaper equivalent: sampling at the 4 vertices of a regular tetrahedron
// centered on `p` instead of +-eps along each of the 3 axes gives the same
// gradient-estimate quality (both are first-order central differences,
// just over a different, still-symmetric sample set) for 33% fewer map()
// evaluations, the single most expensive operation in this whole shader.
// Re-verified visually across the Menger scenes and Demo at several camera
// angles/distances with no visible change in shading/normal quality.
fn calc_normal(p: vec3<f32>, eps: f32) -> vec3<f32> {
    let k = 0.5773502691896258; // 1/sqrt(3): unit-length tetrahedron vertex offset
    let e1 = vec3<f32>(k, -k, -k) * eps;
    let e2 = vec3<f32>(-k, -k, k) * eps;
    let e3 = vec3<f32>(-k, k, -k) * eps;
    let e4 = vec3<f32>(k, k, k) * eps;
    return normalize(e1 * map(p + e1) + e2 * map(p + e2) + e3 * map(p + e3) + e4 * map(p + e4));
}

// Improved soft shadows via the closest-distance-to-the-cone technique
// (Inigo Quilez's \"improved sphere tracing soft shadows\"; ported from
// MMCE's shadow_march, utility/ray_marching.glsl), replacing the earlier
// naive `min(d/t)` ratio approximation -- tracks the closest approach the
// ray makes to any occluder along its path (via the y/d bookkeeping below,
// not just each sample's own ratio), giving smoother, less banded
// penumbras and converging in fewer steps for a given quality. `pixel_angle`
// (see callers' doc) both terminates the march once further stepping
// can't change anything visible and is used as the base occlusion test
// (matching MMCE's `pos.w < max(fovray*dir.w, MIN_DIST)`).
// A ray that exhausts SHADOW_STEPS without either hitting an occluder or
// escaping past MAX_DIST reports **occluded** (0.0), not whatever partial
// visibility it had accumulated so far. Without this, enclosed geometry
// produces scalloped false-light bands: inside a closed shell (the
// hollow-donut scene) every step's `h` is bounded by the cavity radius, so
// 16 steps can never carry the ray to MAX_DIST -- every interior ray
// exhausts mid-flight, and the sawtooth iso-contours of \"how far did my
// budget happen to get\" render as jagged shadow lines on smooth walls
// (root-caused empirically: the artifact survives ?shadowlod=0 and doesn't
// match the ?stepheat=1 contours, and vanishes at SHADOW_STEPS = 64).
// Treating exhaustion as occlusion is also simply the honest answer -- the
// march could not verify that light reaches this point. Open scenes are
// unaffected in practice (outside geometry `h` grows rapidly, so
// sun-visible rays escape well within budget -- why 16 steps looked fine
// on every fractal scene and only the donut's smooth enclosed interior
// exposed it); deep crevices get slightly darker, which is the more
// correct direction. Penumbra edges keep their smooth arcsine remap: rays
// that genuinely escape still return their accumulated soft visibility.
fn shadow(ro: vec3<f32>, rd: vec3<f32>, pixel_angle: f32) -> f32 {
    var pos = ro;
    var t = 0.0;
    var h = map(pos);
    var light_visibility = 1.0;
    var ph = 1e5;
    var d_de_dt = 0.0;
    var escaped = false;
    for (var i = 0; i < SHADOW_STEPS; i++) {
        t += h;
        pos += rd * h;
        h = map(pos);

        let y = h * h / (2.0 * ph);
        let d = (h + ph) * 0.5 * (1.0 - d_de_dt);
        let ang = d / (max(MIN_HIT_DIST, t - y) * LIGHT_ANGLE);
        light_visibility = min(light_visibility, ang);

        d_de_dt = d_de_dt * 0.75 + 0.25 * (h - ph) / ph;
        ph = h;

        if (t > MAX_DIST) {
            escaped = true;
            break;
        }
        if (h < max(t * pixel_angle, MIN_HIT_DIST)) {
            return 0.0;
        }
    }
    if (!escaped) {
        return 0.0;
    }
    light_visibility = clamp(light_visibility, 0.0, 1.0);
    // Same \"looks better and is more physically accurate for a circular
    // light source\" remap MMCE applies (an arcsine-based curve rather than
    // returning the raw linear visibility).
    let lv2 = light_visibility * 2.0 - 1.0;
    return 0.5 + (lv2 * sqrt(max(1.0 - lv2 * lv2, 0.0)) + asin(clamp(lv2, -1.0, 1.0))) / 3.14159265;
}
";

/// Ray marcher + shading, appended after the scene functions in
/// [`generate_shader`]. Kept deliberately simple (v1 — DESIGN.md §5); will
/// grow toward MMCE's quality (fog, glow, GI) later.
const MARCHER: &str = "\
fn sphere_hit(ro: vec3<f32>, rd: vec3<f32>, center: vec3<f32>, r: f32) -> f32 {
    let oc = ro - center;
    let b = dot(oc, rd);
    let c = dot(oc, oc) - r * r;
    let h = b * b - c;
    if (h < 0.0) {
        return -1.0;
    }
    let t = -b - sqrt(h);
    if (t < 0.0) {
        return -1.0;
    }
    return t;
}

// Outline width as a fraction of each marble's own radius (not an absolute
// world-space distance) -- scenes vary hugely in marble scale (the Demo
// level's beware_of_bumps::MARBLE_RAD is 0.02, the Menger scenes' marbles
// are 0.15), so a fixed absolute width would read as a hairline in one and
// a thick band in another. 0.12 chosen by visual inspection as a clean,
// clearly-visible ring at typical on-screen marble sizes without swallowing
// the marble itself.
const OUTLINE_WIDTH_FRACTION: f32 = 0.12;
const OUTLINE_COLOR: vec3<f32> = vec3<f32>(0.1176, 0.0314, 0.3451); // #1E0858

// Per-marble color variation (\"coloring index\" = marble_idx % 4u, cycling
// for player counts beyond 4): each entry is (hue_shift in turns, sat_mul,
// val_mul), applied in HSV so a shift reads as a consistent recoloring
// rather than an arbitrary RGB blend. Values supplied directly by design,
// not derived here.
const TINTS: array<vec3<f32>, 4> = array<vec3<f32>, 4>(
    vec3<f32>( 0.0,        1.00, 1.00),  // 0: original
    vec3<f32>(-8.0/360.0,  1.15, 0.93),  // 1: tangerine
    vec3<f32>(10.0/360.0,  0.40, 1.12),  // 2: cream
    vec3<f32>(-55.0/360.0, 0.85, 1.02),  // 3: sakura
);

fn rgb2hsv(c: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    let p = mix(vec4<f32>(c.bg, K.wz), vec4<f32>(c.gb, K.xy), step(c.b, c.g));
    let q = mix(vec4<f32>(p.xyw, c.r), vec4<f32>(c.r, p.yzx), step(p.x, c.r));
    let d = q.x - min(q.w, q.y);
    let e = 1e-10;
    return vec3<f32>(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3<f32>(0.0), vec3<f32>(1.0)), c.y);
}

fn apply_tint(color: vec3<f32>, index: u32) -> vec3<f32> {
    let t = TINTS[index];
    var hsv = rgb2hsv(color);
    hsv.x = fract(hsv.x + t.x);
    hsv.y = clamp(hsv.y * t.y, 0.0, 1.0);
    hsv.z = clamp(hsv.z * t.z, 0.0, 1.0);
    return hsv2rgb(hsv);
}

// Depth-aware 4-tap resample of the half-resolution shadow/AO pass's cached
// visibility (`shadow_pass.rs`/`SHADOW_MARCHER`), ported from MMCE's
// `bilinear_surface` (`utility/interpolation.glsl`) -- weights each tap by
// `1 / (sz^2 + (td - td_i)^2)` instead of plain bilinear (`(1-dx)(1-dy)`
// etc. alone), so a tap whose stored traveled-distance `td_i` is wildly
// different from this pixel's own `td` (i.e. it's actually a different,
// unrelated surface behind/in front of a silhouette edge) gets smoothly
// down-weighted rather than blended in and haloing the edge. `sz` is the
// expected depth variation across one shadow-pass texel at this depth
// (`3.0 * td * shadow_pixel_angle`, mirroring MRRM's own
// `coarse_pixel_angle` back-off) -- MMCE's gamma-correction step in the
// original is dropped here: that's for resampling a color texture, this is
// a linear scalar visibility. `uv` is this pixel's screen UV (`mesh.uv`).
fn sample_shadow(uv: vec2<f32>, td: f32) -> f32 {
    let dims = vec2<f32>(textureDimensions(shadow_tex));
    let shadow_pixel_angle = 2.0 * atan(1.0 / scene.cam_forward.w) / max(scene.misc2.x, 1.0);
    let sz = max(3.0 * td * shadow_pixel_angle, MIN_HIT_DIST);
    let sz2 = sz * sz;

    let coord = uv * dims;
    let ci = vec2<i32>(floor(coord));
    let d = coord - floor(coord);
    let dims_i = vec2<i32>(dims) - vec2<i32>(1);
    let c00 = clamp(ci, vec2<i32>(0), dims_i);
    let c10 = clamp(ci + vec2<i32>(1, 0), vec2<i32>(0), dims_i);
    let c01 = clamp(ci + vec2<i32>(0, 1), vec2<i32>(0), dims_i);
    let c11 = clamp(ci + vec2<i32>(1, 1), vec2<i32>(0), dims_i);
    let a1 = textureLoad(shadow_tex, c00, 0);
    let a2 = textureLoad(shadow_tex, c10, 0);
    let a3 = textureLoad(shadow_tex, c01, 0);
    let a4 = textureLoad(shadow_tex, c11, 0);

    let w1 = (1.0 - d.x) * (1.0 - d.y) / (sz2 + (td - a1.w) * (td - a1.w));
    let w2 = d.x * (1.0 - d.y) / (sz2 + (td - a2.w) * (td - a2.w));
    let w3 = (1.0 - d.x) * d.y / (sz2 + (td - a3.w) * (td - a3.w));
    let w4 = d.x * d.y / (sz2 + (td - a4.w) * (td - a4.w));
    return (a1.r * w1 + a2.r * w2 + a3.r * w3 + a4.r * w4) / (w1 + w2 + w3 + w4);
}

// Vertical gradient of bg_col plus a sun disc/glow.
fn sky(rd: vec3<f32>) -> vec3<f32> {
    let t = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
    var col = mix(scene.bg_col.rgb * 0.6, scene.bg_col.rgb, t);
    let sun_amt = max(dot(rd, scene.sun.xyz), 0.0);
    col += scene.sun_col.rgb * pow(sun_amt, 256.0) * 2.0;
    col += scene.sun_col.rgb * pow(sun_amt, 8.0) * 0.3;
    return col;
}

// ACES filmic tone mapping (the Narkowicz polynomial fit), ported from
// MMCE's HDRmapping/ACESFilm (utility/shading.glsl) -- replaces the
// previous Reinhard `x/(1+x)` curve, which was the single biggest cause of
// this port's washed-out look vs. the C++ original: Reinhard lifts blacks
// (nothing ever reads as dark) and, because it compresses each channel
// independently toward 1, actively desaturates every bright color toward
// gray. ACES is an S-curve: real shadow contrast, punchier mids, and
// saturation preserved into the highlights, at the same cost class (a few
// multiplies). Exposure rides in `scene.misc3.y` (the fine pass's
// material is the only one that sets `misc3` -- and the only one that
// shades); `select` falls back to 1.0 for a non-positive value so an
// unset uniform can never render the whole frame black.
fn tonemap(col: vec3<f32>) -> vec3<f32> {
    let exposure = select(1.0, scene.misc3.y, scene.misc3.y > 0.0);
    let x = col * exposure;
    let aces = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
    return pow(clamp(aces, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
}

// `?stepheat=1` debug view (`scene.misc3.x`): a 3-stop dark-blue -> yellow ->
// red gradient over `iters / fine_max_steps`, the classic \"shader
// complexity\"/overdraw heatmap convention -- perceptually ordered (cold to
// hot), not a hue wheel, so there's no ambiguous wraparound at either end.
// Returned directly, not run through `tonemap()`: this is an already-final
// display color encoding a raw ratio, not a lit HDR surface value (same
// reasoning as the outline color's doc, just above `apply_tint`'s callers).
fn step_heat_color(frac: f32) -> vec3<f32> {
    let f = clamp(frac, 0.0, 1.0);
    let cold = vec3<f32>(0.0, 0.0, 0.35);
    let mid = vec3<f32>(1.0, 0.9, 0.0);
    let hot = vec3<f32>(1.0, 0.05, 0.0);
    if (f < 0.5) {
        return mix(cold, mid, f * 2.0);
    }
    return mix(mid, hot, (f - 0.5) * 2.0);
}

// View mode 3 (`CoarseStepDistance`, `scene.misc3.x == 3`): color-codes the
// MRRM coarse pre-pass's own cached hit distance (`coarse_tex`'s R channel)
// -- lets a texel-to-texel-inconsistent (aliased) pattern in this value be
// spotted directly, rather than only inferred from its downstream effect on
// step count. Normalized against twice the scene's own bounding-sphere
// radius (a scene-relative reference distance that adapts across this app's
// very differently-scaled scenes without a magic absolute constant),
// falling back to half `MAX_DIST` for a genuinely unbounded scene
// (`scene.bounding.w <= 0.0`, `ray_sphere_clip`'s doc). Deliberately a
// distinct color family (teal -> white -> magenta) from `step_heat_color`'s
// blue -> yellow -> red, so the two views are never visually ambiguous with
// each other if a screenshot from one is looked at without its query
// string handy.
fn coarse_distance_color(dist: f32) -> vec3<f32> {
    let reference = select(MAX_DIST * 0.5, scene.bounding.w * 2.0, scene.bounding.w > 0.0);
    let f = clamp(dist / max(reference, MIN_HIT_DIST), 0.0, 1.0);
    let near = vec3<f32>(0.0, 0.35, 0.35);
    let mid = vec3<f32>(1.0, 1.0, 1.0);
    let far = vec3<f32>(0.35, 0.0, 0.35);
    if (f < 0.5) {
        return mix(near, mid, f * 2.0);
    }
    return mix(mid, far, (f - 0.5) * 2.0);
}

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let ndc = vec2<f32>(mesh.uv.x * 2.0 - 1.0, 1.0 - mesh.uv.y * 2.0);
    let aspect = scene.misc.x;
    let ro = scene.cam_pos.xyz;
    let rd = normalize(
        scene.cam_right.xyz * ndc.x * aspect
        + scene.cam_up.xyz * ndc.y
        + scene.cam_forward.xyz * scene.cam_forward.w
    );

    // View modes 2 (`CoarseStepHeat`) and 3 (`CoarseStepDistance`,
    // `live_debug.rs`'s `DebugViewMode`) show the coarse pre-pass's own
    // cached output for this pixel's texel, ignoring this fine march's
    // `hit_frac`/`iters`/`t`/`marble_hit` entirely -- checked and returned
    // here, before any of that (marble test, march, shading) even runs,
    // which is both the semantically cleanest spot (these views are
    // supposed to show the coarse pass's numbers *untouched* by anything
    // the fine pass does) and the only spot that doesn't risk a later,
    // unrelated `if` (`hit_frac`'s own `scene.misc3.x > 0.5` check, meant
    // for mode 1 only) silently intercepting modes 2/3 too, since both are
    // also `> 0.5` -- confirmed the hard way: an earlier version placed
    // this after `if (hit_frac) {...}` instead, and modes 2/3 rendered
    // (wrong) fine-pass-heatmap colors for every hit pixel, having fallen
    // through that earlier check before ever reaching this one.
    //
    // The texture read itself, though, is NOT allowed to be conditioned on
    // `scene.misc3.x` the way the rest of this block is: an earlier
    // version wrapped the whole `textureLoad(coarse_tex, ...)` in the
    // `scene.misc3.x > 1.5` check below, which is naga-valid (this
    // module's own tests happily accept it) but reproducibly hung the GPU
    // on a real WebGPU backend after a few frames. Reading it here
    // unconditionally -- every pixel, top-level uniform flow -- and only
    // *branching on the result* fixed it, the same fix shape as this
    // const's own `marble_tex_sample` a bit further below (its doc has
    // that incident's full story: an implicit-derivative texture read
    // gated behind a non-uniform condition broke production once already).
    // `textureLoad` isn't supposed to need this per the WGSL spec (unlike
    // `textureSample`'s derivatives) -- a real browser-side gap between
    // the spec and what's actually safe here, not a documented rule,
    // bisected empirically against a live build.
    let coarse_debug_dims = vec2<i32>(textureDimensions(coarse_tex));
    let coarse_debug_texel = clamp(
        vec2<i32>(mesh.uv * vec2<f32>(coarse_debug_dims)),
        vec2<i32>(0),
        coarse_debug_dims - vec2<i32>(1),
    );
    let coarse_debug_sample = textureLoad(coarse_tex, coarse_debug_texel, 0);
    if (scene.misc3.x > 1.5) {
        // Coarse pass's own miss sentinel (`-1.0` in R, `mrrm.rs`'s doc)
        // defaults `color` to flat black rather than a meaningless color
        // from a negative/`MAX_STEPS` value; overwritten below on a hit.
        var color = vec3<f32>(0.0, 0.0, 0.0);
        if (coarse_debug_sample.r > 0.0) {
            if (scene.misc3.x > 2.5) {
                color = coarse_distance_color(coarse_debug_sample.r);
            } else {
                color = step_heat_color(coarse_debug_sample.g / f32(MAX_STEPS));
            }
        }
        return vec4<f32>(color, 1.0);
    }

    // Angular size of one pixel (small-angle tangent, radians-ish), used as
    // a distance-scaled hit threshold (MMCE's \"fovray\"/cone-angle
    // technique, utility/camera.glsl's `fovray` and ray_marching.glsl's
    // `angle` parameter): once a step's DE value is smaller than the
    // physical size one pixel covers at the current distance, further
    // refinement isn't visually distinguishable, so the march can stop.
    // `cam_forward.w` is the focal length used to build `rd` above
    // (`f = 1/tan(halfFOV)`, vertical), so `atan(1/f)` recovers the
    // vertical half-FOV; dividing the full vertical FOV by the vertical
    // pixel count (`misc.z`) gives the per-pixel angular size. This
    // threshold naturally scales with actual render resolution (via
    // `misc.z`, read fresh each frame) rather than being a shader constant,
    // so a future adaptive-resolution feature changing render scale won't
    // need this touched.
    let half_fov = atan(1.0 / scene.cam_forward.w);
    let pixel_angle = 2.0 * half_fov / max(scene.misc.z, 1.0);

    // Nearest hit among every active marble (multiplayer milestone 0: N
    // marbles instead of one, `marbles` storage buffer -- `MARBLE_BUFFER_BINDING`'s
    // doc). `arrayLength` gives the live count with no separate uniform
    // field to keep in sync. Inactive slots (`w <= 0.0`) are skipped, same
    // \"hidden\" convention the old single-marble uniform used.
    //
    // Each marble contributes exactly one candidate surface to this
    // reduction: its real body if the ray hits it, otherwise its *inflated*
    // (radius + OUTLINE_WIDTH) shell if the ray hits that instead (the thin
    // annulus just outside its own silhouette, where its outline ring
    // shows) -- a marble's own outline is only ever considered where its
    // own body missed, so a marble's body always wins over its own
    // outline. Every marble's body-or-outline pick is then reduced to a
    // single nearest-wins minimum together (`best_t`/`best_is_outline`/
    // `best_idx`), not as two separate per-category minimums each compared
    // only against terrain: that two-minimum approach let a farther
    // marble's outline shell beat a nearer marble's real body (or vice
    // versa) with no depth check between the two categories at all, since
    // neither minimum ever looked at the other -- reported live as a
    // farther marble's outline rendering in front of a nearer marble's
    // body. This single unified reduction fixes that: every candidate
    // (every marble's pick, plus terrain below) is depth-compared against
    // every other one, not just against terrain.
    var best_t = -1.0;
    var best_is_outline = false;
    var best_idx = 0u;
    let marble_count = arrayLength(&marbles);
    for (var mi = 0u; mi < marble_count; mi++) {
        let m = marbles[mi];
        if (m.w > 0.0) {
            let mt = sphere_hit(ro, rd, m.xyz, m.w);
            if (mt > 0.0) {
                if (best_t < 0.0 || mt < best_t) {
                    best_t = mt;
                    best_is_outline = false;
                    best_idx = mi;
                }
            } else {
                let ot = sphere_hit(ro, rd, m.xyz, m.w + m.w * OUTLINE_WIDTH_FRACTION);
                if (ot > 0.0 && (best_t < 0.0 || ot < best_t)) {
                    best_t = ot;
                    best_is_outline = true;
                    best_idx = mi;
                }
            }
        }
    }

    // Ray-vs-bounding-sphere pre-test (`marble_csg::Object::bounding_sphere`,
    // `render.rs`'s `bounding` uniform): a ray pointed at open sky can skip
    // the fractal march entirely instead of stepping all the way out to
    // `MAX_DIST` through empty space -- a clean miss here just leaves
    // `hit_frac` false, exactly like an exhausted march, so the marble test
    // above (whose sphere isn't necessarily inside this bound -- e.g.
    // `GravityMode::Flying` lets it fly anywhere) still works unchanged.
    let clip = ray_sphere_clip(ro, rd, scene.bounding.xyz, scene.bounding.w);

    var t = 0.0;
    var hit_frac = false;
    var iters = 0;
    // Declared at this outer scope (not just `let`-bound inside the block
    // below, where it's computed) because the AO term further down also
    // needs it, well after this block closes -- default `FINE_MAX_STEPS`
    // covers the `clip.x > clip.y` (no valid march at all) case too, where
    // the block below never runs and AO's divisor still needs *some* value.
    var fine_max_steps = FINE_MAX_STEPS;
    if (clip.x <= clip.y) {
        // MRRM (multi-resolution ray marching): start this march from the
        // coarse pre-pass's cached hit distance for this pixel instead of
        // the camera (`t=0`), skipping almost all of the empty-space
        // traversal neighboring pixels in the same coarse texel already
        // redid independently. `scene.misc.w` is the MRRM on/off toggle
        // (`MM_MRRM`, see `render.rs::update_material`) -- kept as a uniform
        // flag rather than an entity/system toggle so the *shape* of every
        // frame is identical whether MRRM is on or off (same cameras, same
        // passes run), which is what makes the A/B comparison this feature
        // was verified with trustworthy: only this one value differs
        // between the two runs.
        var t0 = clip.x;
        // True only when every one of the coarse pre-pass's 3x3 texels
        // neighboring this pixel's own texel reports a genuine miss (the
        // `-1.0` sentinel `COARSE_MARCHER` writes on both a bounding-sphere
        // clip fail and an exhausted march -- see its doc). Skips the whole
        // terrain march below (`MAX_STEPS` worth of wasted stepping through
        // open sky) when true: a confirmed miss across a 3x3 neighborhood at
        // the coarse pass's 1/8-linear (64x fewer pixels) resolution is
        // extremely unlikely to be hiding a real fine-resolution surface --
        // every one of those 9 coarse rays would have to have missed
        // geometry a fine ray through the same texel finds, despite the
        // coarse pass's own (cheapened, but still real) fractal evaluation.
        // Deliberately doesn't touch the marble hit-test above (already run,
        // unconditionally, before this block even starts) or the
        // marble/outline shading below -- a marble can still be in frame
        // over a terrain-confirmed-empty background; this only ever
        // short-circuits the *terrain* `hit_frac` to false, exactly what a
        // from-scratch march that ran to `MAX_STEPS` and found nothing would
        // have produced anyway, just without spending the steps.
        var sky_confirmed_miss = false;
        if (scene.misc.w > 0.5) {
            let coarse_dims = vec2<i32>(textureDimensions(coarse_tex));
            let texel = clamp(
                vec2<i32>(mesh.uv * vec2<f32>(coarse_dims)),
                vec2<i32>(0),
                coarse_dims - vec2<i32>(1),
            );
            let coarse_t = textureLoad(coarse_tex, texel, 0).r;

            // 3x3 neighborhood read for the sky-miss-skip check above --
            // each tap independently clamped to the texture bounds (same
            // clamp-to-edge convention as the center texel just above), so
            // an edge/corner pixel just re-samples its own row/column
            // instead of wrapping or reading garbage.
            let coarse_dims_max = coarse_dims - vec2<i32>(1);
            let cn_nw = textureLoad(coarse_tex, clamp(texel + vec2<i32>(-1, -1), vec2<i32>(0), coarse_dims_max), 0).r;
            let cn_n  = textureLoad(coarse_tex, clamp(texel + vec2<i32>( 0, -1), vec2<i32>(0), coarse_dims_max), 0).r;
            let cn_ne = textureLoad(coarse_tex, clamp(texel + vec2<i32>( 1, -1), vec2<i32>(0), coarse_dims_max), 0).r;
            let cn_w  = textureLoad(coarse_tex, clamp(texel + vec2<i32>(-1,  0), vec2<i32>(0), coarse_dims_max), 0).r;
            let cn_e  = textureLoad(coarse_tex, clamp(texel + vec2<i32>( 1,  0), vec2<i32>(0), coarse_dims_max), 0).r;
            let cn_sw = textureLoad(coarse_tex, clamp(texel + vec2<i32>(-1,  1), vec2<i32>(0), coarse_dims_max), 0).r;
            let cn_s  = textureLoad(coarse_tex, clamp(texel + vec2<i32>( 0,  1), vec2<i32>(0), coarse_dims_max), 0).r;
            let cn_se = textureLoad(coarse_tex, clamp(texel + vec2<i32>( 1,  1), vec2<i32>(0), coarse_dims_max), 0).r;
            sky_confirmed_miss = coarse_t <= COARSE_MISS_SENTINEL_THRESHOLD
                && cn_nw <= COARSE_MISS_SENTINEL_THRESHOLD && cn_n <= COARSE_MISS_SENTINEL_THRESHOLD && cn_ne <= COARSE_MISS_SENTINEL_THRESHOLD
                && cn_w <= COARSE_MISS_SENTINEL_THRESHOLD                                             && cn_e <= COARSE_MISS_SENTINEL_THRESHOLD
                && cn_sw <= COARSE_MISS_SENTINEL_THRESHOLD && cn_s <= COARSE_MISS_SENTINEL_THRESHOLD && cn_se <= COARSE_MISS_SENTINEL_THRESHOLD;

            let coarse_pixel_angle = 2.0 * half_fov / max(f32(coarse_dims.y), 1.0);
            // Shadow-tier warm-start: the half-resolution shadow/AO pass
            // (`shadow_pass.rs`/`SHADOW_MARCHER`) already ran its own march
            // at 4x pixel density vs. the coarse pass's 64x -- a strictly
            // tighter, more spatially-accurate starting-distance guess than
            // the coarse pass's, whenever it actually hit something. Its `A`
            // channel is `MAX_DIST` on a miss (`SHADOW_MARCHER`'s doc -- not
            // a negative sentinel, since `sample_shadow`'s far-away
            // down-weighting wants a large, not negative, value there), so
            // \"did it hit\" is a plain `< MAX_DIST` check; preferred outright
            // over the coarser guess (not blended) when true, same
            // single-warm-start-winner simplicity as MRRM's own
            // coarse-vs-camera choice. Falls back to the coarse guess above
            // when the shadow pass's own march missed too. Same back-off-by-
            // one-texel's-angular-footprint reasoning as the coarse guess,
            // just at this pass's own (denser) resolution.
            let shadow_dims = vec2<i32>(textureDimensions(shadow_tex));
            let shadow_texel = clamp(
                vec2<i32>(mesh.uv * vec2<f32>(shadow_dims)),
                vec2<i32>(0),
                shadow_dims - vec2<i32>(1),
            );
            let shadow_td = textureLoad(shadow_tex, shadow_texel, 0).a;
            if (shadow_td < MAX_DIST) {
                let shadow_pixel_angle = 2.0 * half_fov / max(f32(shadow_dims.y), 1.0);
                t0 = max(t0, shadow_td - shadow_td * shadow_pixel_angle);
            } else if (coarse_t > 0.0) {
                // Back off by roughly one coarse-pixel's angular footprint at
                // that depth, computed from the coarse pass's *own* resolution
                // (`coarse_dims`, not this pass's `pixel_angle`) -- a real
                // surface just outside the coarse sample's exact ray direction
                // (which the coarse pass, marching along a slightly different
                // ray through the same texel, wouldn't have seen) must still
                // land inside the fine march's search space, not be skipped
                // past by starting exactly on top of (or beyond) the coarse
                // guess. The over-relaxed march's backtrack-on-overstep handles
                // the rest -- see `march_scene`'s doc.
                t0 = max(t0, coarse_t - coarse_t * coarse_pixel_angle);
            }
        }

        // Safety corroboration against `COARSE_MARCHER`'s cheapened-
        // iteration DE being systematically *wrong* (not just imprecise)
        // for domain-repeated (`Fold::Repeat`-heavy) scenes -- verified
        // live against a real build: on the `demo` scene (`classic()`
        // unioned with `creme_spheres()`'s domain-repeated sphere lattice),
        // the coarse pass's reduced iteration count (`COARSE_ITERATION_DIVISOR`)
        // reports a confirmed miss for essentially *every* pixel, even
        // though the real, full-fidelity surface is right there -- without
        // this check, `sky_confirmed_miss` would skip the terrain march for
        // the whole frame and render it as flat sky, a real (not just
        // slower-than-hoped) correctness regression, not merely the
        // already-known-and-out-of-scope \"no MRRM speedup on this scene\"
        // finding. A single extra `map()` call here, using *this* pass's own
        // full-fidelity `de_scene` (not the coarse pass's reduced-iteration
        // copy), directly measures the real distance to geometry at the
        // march's own starting point, independent of whatever went wrong in
        // the coarse pass -- a tiny fixed cost next to the up-to-
        // `fine_max_steps` march it's guarding against wrongly skipping.
        // The generous 8x margin over the normal hit threshold errs toward
        // *not* trusting the skip (falling through to a real march, always
        // safe -- see `march_scene`'s doc) whenever the probe finds anything
        // even plausibly close, rather than toward saving the last few
        // percent of steps on a hair-trigger threshold.
        if (sky_confirmed_miss) {
            let sky_probe_d = map(ro + rd * t0);
            sky_confirmed_miss = sky_probe_d > max(t0 * pixel_angle, MIN_HIT_DIST) * 8.0;
        }

        // `scene.misc2.z`: `?perfprobe=`'s runtime fine-pass step-budget
        // override (`perfprobe.rs`) -- `0.0` (the default, and the only
        // value every non-probe run ever sees) means \"no override, use the
        // default FINE_MAX_STEPS budget\"; a positive value clamps this
        // march's step count for one probe window, to measure how much of
        // the fine pass's GPU cost the step budget itself accounts for.
        fine_max_steps = select(FINE_MAX_STEPS, i32(scene.misc2.z), scene.misc2.z > 0.5);
        if (!sky_confirmed_miss) {
            let march = march_scene(ro, rd, t0, pixel_angle, clip.y, fine_max_steps);
            t = march.t;
            hit_frac = march.hit;
            iters = march.iters;
        }
    }

    // The globally-nearest candidate (across every marble's own body-or-
    // outline pick, see the reduction loop above) wins over terrain, same
    // nearer-than-terrain check the old two-branch version used for each
    // category separately -- now applied once, to the single winner.
    let candidate_hit = best_t > 0.0 && (!hit_frac || best_t < t);
    let marble_hit = candidate_hit && !best_is_outline;

    // `textureSample` (unlike every other texture read in this shader --
    // `coarse_tex`/`shadow_tex` are both `textureLoad`) implicitly computes
    // screen-space derivatives for mip selection, and WGSL requires calls
    // that do that to happen in *uniform* control flow -- neighboring
    // fragment-shader invocations must agree on whether they execute it at
    // all. `marble_hit` is a per-pixel, data-dependent condition, so calling
    // `textureSample` from inside `if (marble_hit)` violates that
    // requirement. naga's offline validator doesn't catch this (confirmed:
    // this exact shader passes every naga parse+validate test in this
    // codebase); Chrome's real WebGPU implementation (Tint) enforces it
    // strictly and rejects the shader module outright, which is what broke
    // production in commit `86ebb1d` (reverted in `d5785ed`) -- root-caused
    // by elimination (naga-valid, real-browser-invalid, and the *only*
    // `textureSample` call in the whole generated shader) after the
    // in-place-image-mutation fix alone wasn't sufficient to resolve it.
    // Fix: sample unconditionally, every pixel, in the shader's own uniform
    // top-level flow, and only *use* the result inside the conditional --
    // `best_idx` defaults to `0u` and `marbles` is always non-empty
    // whenever a scene has a marble at all, so this is safe (if physically
    // meaningless) to evaluate even for pixels that hit no marble; the
    // wasted work on those pixels is a minor, acceptable cost for a
    // correctness requirement, not a bug.
    let hit_marble = marbles[best_idx];
    let mp = ro + rd * best_t;
    let mn = normalize(mp - hit_marble.xyz);
    // Spin the cubemap sample direction around the marble's local Y axis
    // (`scene.misc2.w`, radians -- this const's own doc on why it's
    // tick-driven, not wall-clock) rather than the marble itself, so this
    // is a pure shading effect with no bearing on physics/collision.
    let rot_a = scene.misc2.w;
    let rot_cos = cos(rot_a);
    let rot_sin = sin(rot_a);
    let mn_rotated = vec3<f32>(
        mn.x * rot_cos - mn.z * rot_sin,
        mn.y,
        mn.x * rot_sin + mn.z * rot_cos,
    );
    let marble_tex_sample = textureSample(marble_cubemap, marble_cubemap_sampler, mn_rotated).rgb;

    if (marble_hit) {
        let refl = reflect(rd, mn);
        let fresnel = pow(1.0 - max(dot(-rd, mn), 0.0), 5.0);
        // `apply_tint(.., best_idx % 4u)` recolors the cubemap sample per
        // marble (HSV hue/sat/val shift, `TINTS`'s doc) so multiple marbles
        // still read as distinct bodies -- the outline branch below applies
        // the exact same tint index to `OUTLINE_COLOR`, so a marble's ring
        // and body always read as one consistent color family.
        let base_col = apply_tint(marble_tex_sample, best_idx % 4u);
        let ambient = 0.3;
        let diffuse = max(dot(mn, scene.sun.xyz), 0.0);
        let shaded = base_col * (ambient + (1.0 - ambient) * diffuse);
        let rim = pow(fresnel, 2.0) * 0.4 * sky(refl);
        let marble_col = shaded + rim;
        return vec4<f32>(tonemap(marble_col), 1.0);
    }

    // Solid outline ring (`OUTLINE_WIDTH_FRACTION`'s doc): reached only
    // when the globally-nearest candidate across every marble (and
    // terrain) is specifically an outline pick, not a body -- `candidate_hit`
    // already folded in the terrain-occlusion check above, and
    // `!best_is_outline` already ruled out the body case in `marble_hit`,
    // so this is exactly its complement: the winner exists, is nearer than
    // terrain, and is an outline. Because the reduction above compares
    // every marble's body-or-outline pick against every other one (not
    // just against terrain), a nearer marble's real body correctly beats a
    // farther marble's outline here, and a nearer marble's own outline
    // (in the annulus where its body missed) correctly beats a farther
    // marble's body too -- the cross-marble depth check that was missing
    // before this fix.
    let outline_visible = candidate_hit && best_is_outline;
    if (outline_visible) {
        // Deliberately *not* run through `tonemap()`: that ACES+gamma
        // curve is calibrated for lit HDR surface values (which is why the
        // marble's own shaded color above goes through it), and applying it
        // to an already-final flat display color pushes it lighter *and*
        // shifts its hue (e.g. `OUTLINE_COLOR`'s #1E0858 comes out looking
        // like a lighter, bluer purple) -- reported live: the outline read
        // visibly lighter than the hex code even though the marble's own
        // (lit, then tonemapped) dark colors looked right. Returning the
        // tinted color directly is what actually displays the requested hex
        // value (modulo the per-marble tint, which is the point).
        return vec4<f32>(apply_tint(OUTLINE_COLOR, best_idx % 4u), 1.0);
    }

    if (hit_frac) {
        // `?stepheat=1`: full override, not a blend, matching how a
        // shader-complexity/overdraw debug view conventionally works --
        // reached only for an actual terrain march hit (the thing with real
        // `iters` data), so marble/outline pixels above still render
        // normally (they're closed-form sphere tests, not marched, so
        // there's no step count to show for them) and the miss/sky case
        // below gets its own distinct override rather than falling into
        // this gradient at the \"0 steps\" end, which would be visually
        // ambiguous with a genuinely cheap real hit. Safe as a plain
        // `> 0.5` even though `misc3.x` is now a 4-value selector, not a
        // bool: modes 2 and 3 already returned at the very top of this
        // function (before `hit_frac` was even computed), so nothing
        // reaches this point with `misc3.x` above `1.0`.
        if (scene.misc3.x > 0.5) {
            let heat_frac = f32(iters) / f32(fine_max_steps);
            return vec4<f32>(step_heat_color(heat_frac), 1.0);
        }
        let p = ro + rd * t;
        let eps = 1e-4 * max(t, 0.05);
        let n = calc_normal(p, eps);
        let col = col_scene(vec4<f32>(p, 1.0));
        // Orbit-trap values are `max(orbit, p.xyz * color)` accumulated over
        // however many fold iterations ran (DESIGN.md §5's OrbitMax) -- `p`
        // is the *folded* coordinate, which routinely reaches magnitudes
        // well past +-1 (up to the fractal's box half-extent, ~6 for the
        // classic scene). A hard `clamp(col.rgb, 0, 1)` here saturates most
        // pixels to white/near-white *before* any lighting is applied, which
        // reads as a uniformly pale/washed-out surface no matter how
        // ambient/diffuse/tonemap constants downstream are tuned (clamping
        // is lossy -- detail above 1.0 is gone, not just displayed brightly).
        // Compress with a Reinhard-style curve before clamping instead
        // (this is albedo-range normalization, deliberately *not* switched
        // to ACES along with `tonemap()` -- ACES is a display transform;
        // this just needs a monotonic squash of orbit magnitudes into
        // [0, 1)): this maps the *typical* orbit range
        // smoothly into (-1, 1) so the coloring keeps its shape (still
        // varies meaningfully across the surface -- it's the same orbit
        // value, just range-compressed) rather than being hard-clipped to a
        // flat white ceiling. The clamp after is still needed to guard
        // against negative components (a channel can go negative when the
        // folded p component is negative and no later iteration's max
        // overrides it) -- a negative `base` would go on to make `color`
        // negative and NaN in `tonemap`'s `pow`.
        let base_compressed = clamp(col.rgb / (1.0 + abs(col.rgb)), vec3<f32>(0.0), vec3<f32>(1.0));
        // Albedo gamma boost, ported from MMCE's `lighting()`
        // (utility/shading.glsl: `albedo = pow(albedo, 1/gamma_material)`,
        // shipped default `gamma_material = 0.5` -- i.e. the albedo is
        // *squared*, with the original's own comment reading \"square it to
        // make the fractals more colorfull\"). Squaring a [0,1) albedo
        // deepens mids and amplifies channel ratios (saturation), the
        // second-biggest contributor to the C++ look after the ACES curve.
        // Rides in `misc3.z` (`?material_gamma=`/`MM_MATERIAL_GAMMA`), same
        // fallback-guard convention as `tonemap`'s `misc3.y` exposure:
        // non-positive means unset, use the 0.5 default. Terrain albedo
        // only -- the marble's cubemap color and the flat outline/debug
        // colors are authored display values, not fractal orbit output, and
        // MMCE's own gamma applies only in its fractal-surface `lighting()`
        // path too.
        let material_gamma = select(0.5, scene.misc3.z, scene.misc3.z > 0.0);
        let base = pow(base_compressed, vec3<f32>(1.0 / material_gamma));
        let diffuse = max(dot(n, scene.sun.xyz), 0.0);
        // `scene.misc2.y` is the `MM_SHADOW_LOD` on/off flag (same
        // uniform-flag convention as MRRM's `misc.w`, for the same
        // A/B-comparability reason): resample the half-resolution
        // shadow/AO pass's cached visibility instead of marching a fresh
        // shadow ray for every full-res pixel.
        var sh = 0.0;
        if (scene.misc2.y > 0.5) {
            sh = sample_shadow(mesh.uv, t);
        } else {
            sh = shadow(p + n * 2.0 * eps, scene.sun.xyz, pixel_angle);
        }
        let ambient = 0.3 + 0.4 * max(dot(n, vec3<f32>(0.0, 1.0, 0.0)), 0.0);
        // Divides by `fine_max_steps` (the budget this specific march
        // actually ran under), not the shared `MAX_STEPS` constant -- with
        // `FINE_MAX_STEPS` now smaller than `MAX_STEPS`, and `?perfprobe=`'s
        // override shrinking it further still, dividing by the wrong
        // (larger) constant would cap this term well short of fully dark
        // in deep recursive crevices, silently flattening AO contrast.
        let ao = 1.0 - f32(iters) / f32(fine_max_steps);
        var color = base * (ambient + diffuse * sh) * ao;
        // Fades toward sky(rd) starting only past MAX_DIST * 0.5 (was
        // smoothstep(0.0, MAX_DIST, t), i.e. any hit at all started
        // fading): the old onset-at-zero curve already blended ~22% of the
        // pale sky-blue background in at t ~= 9 -- the Menger scenes' own
        // default camera distance (render.rs's MengerOscillatingSphere
        // override), not some rare zoomed-out edge case -- and against this
        // material's already-pale creme base color that read as a
        // near-total wash-out (reported live as excessive fog, confirmed
        // via a screenshot of the untouched default view, no zoom
        // involved). Since t can never exceed MAX_DIST for an actual hit
        // (march_scene's max_d cutoff), the upper bound needs to sit past
        // MAX_DIST too, or every distant-but-real hit would already be
        // fully faded; 1.5x caps a maximally-distant hit at 50% fog rather
        // than 100%, so even marching all the way out to MAX_DIST still
        // shows something, not a flat sky-colored wall.
        color = mix(color, sky(rd), smoothstep(MAX_DIST * 0.5, MAX_DIST * 1.5, t));
        return vec4<f32>(tonemap(color), 1.0);
    }

    // `?stepheat=1`: a ray that never hit anything gets a fixed, distinct
    // black rather than the normal sky color -- the heat gradient's own
    // \"cold\" end is a dark blue, close enough to a pale-sky-tinted black
    // that leaving the real sky color in here would read ambiguously as
    // \"very cheap hit\" instead of \"no hit at all\".
    if (scene.misc3.x > 0.5) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    return vec4<f32>(tonemap(sky(rd)), 1.0);
}
";

/// A small auxiliary pass (`rust/app/src/step_data.rs`) that runs the exact
/// same terrain march the fine pass (`MARCHER`) does -- same MRRM warm-start,
/// same sky-miss-skip, same shadow-tier warm-start (reading the *same*
/// `coarse_tex`/`shadow_tex`, the *same* `scene.misc.w` flag, the *same*
/// shared `fine_scene` uniform buffer), same `fine_max_steps` budget -- but
/// into a tiny fixed-size render target, writing the raw step count
/// (`f32(iters)`) as data instead of a shaded/heatmap color. `?gpuprofile=1`'s
/// overlay reads this small target back to the CPU and averages it as a
/// statistical estimate of the real fine pass's per-pixel step-count
/// distribution (`rust/app/src/step_data.rs`'s module doc has the full
/// design/tradeoff rationale -- this constant is deliberately just the
/// march itself, stripped of every fine-pass concern that doesn't affect
/// `iters`: no marble/outline hit-testing, no shading, no `?stepheat=1`
/// coloring, none of which change how many steps the terrain march takes).
/// Needs `shadow_tex` now (added alongside the fine pass's shadow-tier
/// warm-start reuse) even though this pass does no shading of its own --
/// otherwise this proxy would silently under-count relative to the real fine
/// pass whenever the shadow warm-start wins over the coarse one.
/// A ray that misses the bounding sphere entirely (`clip.x > clip.y`, the
/// march block below never runs) naturally reports `0` -- correct, since no
/// march step was ever spent on it, same as `MARCHER`'s own `var iters = 0;`
/// default for that case.
const STEPDATA_MARCHER: &str = "\
@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let ndc = vec2<f32>(mesh.uv.x * 2.0 - 1.0, 1.0 - mesh.uv.y * 2.0);
    let aspect = scene.misc.x;
    let ro = scene.cam_pos.xyz;
    let rd = normalize(
        scene.cam_right.xyz * ndc.x * aspect
        + scene.cam_up.xyz * ndc.y
        + scene.cam_forward.xyz * scene.cam_forward.w
    );
    let half_fov = atan(1.0 / scene.cam_forward.w);
    let pixel_angle = 2.0 * half_fov / max(scene.misc.z, 1.0);

    let clip = ray_sphere_clip(ro, rd, scene.bounding.xyz, scene.bounding.w);

    var iters = 0;
    if (clip.x <= clip.y) {
        // Identical MRRM warm-start/sky-miss-skip/shadow-tier warm-start to
        // `MARCHER`'s own -- see that fragment's doc for the exact reasoning;
        // kept byte-for-byte the same here so this pass is a faithful proxy
        // for the real fine pass's step count, not an approximation of it.
        var t0 = clip.x;
        var sky_confirmed_miss = false;
        if (scene.misc.w > 0.5) {
            let coarse_dims = vec2<i32>(textureDimensions(coarse_tex));
            let texel = clamp(
                vec2<i32>(mesh.uv * vec2<f32>(coarse_dims)),
                vec2<i32>(0),
                coarse_dims - vec2<i32>(1),
            );
            let coarse_t = textureLoad(coarse_tex, texel, 0).r;

            let coarse_dims_max = coarse_dims - vec2<i32>(1);
            let cn_nw = textureLoad(coarse_tex, clamp(texel + vec2<i32>(-1, -1), vec2<i32>(0), coarse_dims_max), 0).r;
            let cn_n  = textureLoad(coarse_tex, clamp(texel + vec2<i32>( 0, -1), vec2<i32>(0), coarse_dims_max), 0).r;
            let cn_ne = textureLoad(coarse_tex, clamp(texel + vec2<i32>( 1, -1), vec2<i32>(0), coarse_dims_max), 0).r;
            let cn_w  = textureLoad(coarse_tex, clamp(texel + vec2<i32>(-1,  0), vec2<i32>(0), coarse_dims_max), 0).r;
            let cn_e  = textureLoad(coarse_tex, clamp(texel + vec2<i32>( 1,  0), vec2<i32>(0), coarse_dims_max), 0).r;
            let cn_sw = textureLoad(coarse_tex, clamp(texel + vec2<i32>(-1,  1), vec2<i32>(0), coarse_dims_max), 0).r;
            let cn_s  = textureLoad(coarse_tex, clamp(texel + vec2<i32>( 0,  1), vec2<i32>(0), coarse_dims_max), 0).r;
            let cn_se = textureLoad(coarse_tex, clamp(texel + vec2<i32>( 1,  1), vec2<i32>(0), coarse_dims_max), 0).r;
            sky_confirmed_miss = coarse_t <= COARSE_MISS_SENTINEL_THRESHOLD
                && cn_nw <= COARSE_MISS_SENTINEL_THRESHOLD && cn_n <= COARSE_MISS_SENTINEL_THRESHOLD && cn_ne <= COARSE_MISS_SENTINEL_THRESHOLD
                && cn_w <= COARSE_MISS_SENTINEL_THRESHOLD                                             && cn_e <= COARSE_MISS_SENTINEL_THRESHOLD
                && cn_sw <= COARSE_MISS_SENTINEL_THRESHOLD && cn_s <= COARSE_MISS_SENTINEL_THRESHOLD && cn_se <= COARSE_MISS_SENTINEL_THRESHOLD;

            let coarse_pixel_angle = 2.0 * half_fov / max(f32(coarse_dims.y), 1.0);
            let shadow_dims = vec2<i32>(textureDimensions(shadow_tex));
            let shadow_texel = clamp(
                vec2<i32>(mesh.uv * vec2<f32>(shadow_dims)),
                vec2<i32>(0),
                shadow_dims - vec2<i32>(1),
            );
            let shadow_td = textureLoad(shadow_tex, shadow_texel, 0).a;
            if (shadow_td < MAX_DIST) {
                let shadow_pixel_angle = 2.0 * half_fov / max(f32(shadow_dims.y), 1.0);
                t0 = max(t0, shadow_td - shadow_td * shadow_pixel_angle);
            } else if (coarse_t > 0.0) {
                t0 = max(t0, coarse_t - coarse_t * coarse_pixel_angle);
            }
        }
        // Same full-fidelity-`map()` safety corroboration as `MARCHER`'s own
        // -- see that fragment's doc for why this is load-bearing, not just
        // an extra-cautious nicety (the `demo` scene's coarse pass reports a
        // confirmed miss for essentially every pixel).
        if (sky_confirmed_miss) {
            let sky_probe_d = map(ro + rd * t0);
            sky_confirmed_miss = sky_probe_d > max(t0 * pixel_angle, MIN_HIT_DIST) * 8.0;
        }
        let fine_max_steps = select(FINE_MAX_STEPS, i32(scene.misc2.z), scene.misc2.z > 0.5);
        if (!sky_confirmed_miss) {
            let march = march_scene(ro, rd, t0, pixel_angle, clip.y, fine_max_steps);
            iters = march.iters;
        }
    }

    return vec4<f32>(f32(iters), 0.0, 0.0, 1.0);
}
";

/// Bindings decl + helpers, nothing else (DESIGN.md §5).
pub fn generate_library() -> String {
    let b = bindings();
    let mut s = String::with_capacity(b.len() + HELPERS.len() + 1);
    s.push_str(&b);
    s.push('\n');
    s.push_str(HELPERS);
    s
}

/// `de_scene` + `col_scene` for `obj` (DESIGN.md §5).
pub fn generate_scene_functions(obj: &Object) -> String {
    generate_scene_functions_with_divisor(obj, 1)
}

/// Same as [`generate_scene_functions`], but every `Fold::Repeat` in `obj`
/// gets its loop bound divided by `divisor` at emission time (see
/// [`CodeWriter::iteration_divisor`]'s doc) -- used by
/// `generate_coarse_shader`/`generate_shadow_shader` to build a cheaper,
/// lower-detail variant for passes that don't need full fractal fidelity.
fn generate_scene_functions_with_divisor(obj: &Object, divisor: i32) -> String {
    let mut w = CodeWriter::new();
    w.iteration_divisor = divisor;

    w.color_pass = false;
    w.writeln("fn de_scene(p_in: vec4<f32>) -> f32 {");
    w.indent();
    w.writeln("var p = p_in;");
    w.writeln("var d = 1e20;");
    w.emit_object(obj);
    w.writeln("return d;");
    w.dedent();
    w.writeln("}");
    w.writeln("");

    w.color_pass = true;
    w.writeln("fn col_scene(p_in: vec4<f32>) -> vec4<f32> {");
    w.indent();
    w.writeln("var p = p_in;");
    w.writeln("var d = 1e20;");
    w.writeln("var orbit = vec3<f32>(0.0);");
    w.emit_object(obj);
    w.writeln("return vec4<f32>(orbit, d);");
    w.dedent();
    w.writeln("}");

    w.finish()
}

/// Full fragment shader for the app's *fine* (full-resolution) marcher pass:
/// import line, bindings, the MRRM coarse-texture + shadow-texture bindings,
/// library, scene functions, shared march core, shared shading core,
/// fine-only shading, and the `fragment` entry (DESIGN.md §5). Reads back
/// `mrrm.rs`'s coarse pre-pass render target as a starting-distance guess
/// and `shadow_pass.rs`'s half-resolution shadow/AO pass as a resampled
/// visibility value (see `MARCHER`'s `fragment` doc).
pub fn generate_shader(obj: &Object) -> String {
    let mut s = String::new();
    s.push_str("#import bevy_sprite::mesh2d_vertex_output::VertexOutput\n\n");
    s.push_str(&generate_library());
    s.push('\n');
    s.push_str(COARSE_TEXTURE_BINDING);
    s.push('\n');
    s.push_str(SHADOW_TEXTURE_BINDING);
    s.push('\n');
    s.push_str(MARBLE_BUFFER_BINDING);
    s.push('\n');
    s.push_str(MARBLE_TEXTURE_BINDING);
    s.push('\n');
    s.push_str(&generate_scene_functions(obj));
    s.push_str("\n\n");
    s.push_str(MARCH_CORE);
    s.push('\n');
    s.push_str(SHADING_CORE);
    s.push('\n');
    s.push_str(MARCHER);
    s
}

/// Full fragment shader for the small auxiliary step-count-data pass
/// (`rust/app/src/step_data.rs`, `?gpuprofile=1`'s cumulative-step-count
/// estimate): import line, bindings, the MRRM coarse-texture binding (needs
/// `coarse_tex` -- must warm-start identically to the fine pass) and the
/// shadow-texture binding (needs `shadow_tex` too, now that the fine pass's
/// warm-start prefers the shadow pass's own march distance when it hit
/// something -- `STEPDATA_MARCHER`'s doc), the *full fidelity* scene
/// functions (`generate_scene_functions`, divisor 1, NOT
/// `COARSE_ITERATION_DIVISOR` -- this pass exists to proxy the *fine* pass's
/// step count, not the coarse pass's own reduced-detail march), shared march
/// core, and the `STEPDATA_MARCHER` entry. Still no marble-buffer/
/// marble-cubemap bindings -- those affect shading, not step count, so this
/// pass's bind-group layout stays smaller than the fine pass's in that one
/// respect.
pub fn generate_stepdata_shader(obj: &Object) -> String {
    let mut s = String::new();
    s.push_str("#import bevy_sprite::mesh2d_vertex_output::VertexOutput\n\n");
    s.push_str(&generate_library());
    s.push('\n');
    s.push_str(COARSE_TEXTURE_BINDING);
    s.push('\n');
    s.push_str(SHADOW_TEXTURE_BINDING);
    s.push('\n');
    s.push_str(&generate_scene_functions(obj));
    s.push_str("\n\n");
    s.push_str(MARCH_CORE);
    s.push('\n');
    s.push_str(STEPDATA_MARCHER);
    s
}

/// Every `Fold::Repeat` in the coarse and shadow passes' generated trees
/// runs at `1/COARSE_ITERATION_DIVISOR` the fine pass's iteration count
/// (floored at [`MIN_REPEAT_ITERATIONS`]) -- see
/// [`CodeWriter::iteration_divisor`]'s doc for why neither pass needs full
/// fractal fidelity. Raised from `2` to `3` this session, re-verified
/// visually the same way the original `2` was (not a derived constant):
/// still leaves enough recursive detail for MRRM's warm-start guess and the
/// shadow pass's occlusion test to track the fine pass's real surface
/// closely across the Menger scenes' corners/tunnels and the Demo scene, at
/// several camera distances, without paying for iterations whose effect is
/// already sub-pixel at these passes' own coarser resolution. Floored at
/// [`MIN_REPEAT_ITERATIONS`] regardless, so this can't degenerate further.
const COARSE_ITERATION_DIVISOR: i32 = 3;

/// Full fragment shader for the app's MRRM *coarse* pre-pass (`mrrm.rs`):
/// import line + bindings (no coarse-texture binding -- this pass doesn't
/// read one) + library + a reduced-iteration copy of the scene functions
/// (`COARSE_ITERATION_DIVISOR`) + shared march core + the coarse `fragment`
/// entry (see `COARSE_MARCHER`'s doc for why this is a wholly separate
/// module from [`generate_shader`]'s, not a second entry point in the same
/// one). Deliberately regenerates `de_scene`/`col_scene` from the same `obj`
/// tree rather than sharing a compiled module with the fine shader -- cheap
/// (shader generation/compilation is a one-time startup/scene-change cost,
/// not per-frame), and it's what lets this pass's bind-group layout have no
/// `coarse_tex` binding at all instead of the fine pass's.
pub fn generate_coarse_shader(obj: &Object) -> String {
    let mut s = String::new();
    s.push_str("#import bevy_sprite::mesh2d_vertex_output::VertexOutput\n\n");
    s.push_str(&generate_library());
    s.push('\n');
    s.push_str(&generate_scene_functions_with_divisor(
        obj,
        COARSE_ITERATION_DIVISOR,
    ));
    s.push_str("\n\n");
    s.push_str(MARCH_CORE);
    s.push('\n');
    s.push_str(COARSE_MARCHER);
    s
}

/// Full fragment shader for the app's half-resolution shadow/AO pre-pass
/// (`shadow_pass.rs`): import line + bindings + the MRRM coarse-texture
/// binding (kept for bind-group-layout parity with `ShadowMarcherMaterial`
/// only -- no longer read by the fragment body, see `SHADOW_MARCHER`'s doc
/// for why its warm-start was removed) + library + a reduced-iteration copy
/// of the scene functions (same `COARSE_ITERATION_DIVISOR` as the coarse
/// pass -- this pass only needs an approximate occlusion test, not full
/// fractal fidelity either) + shared march core + shared shading core + the
/// `fragment` entry (see `SHADOW_MARCHER`'s doc). Wholly separate module
/// from both `generate_shader` and `generate_coarse_shader`, same "fixed
/// `\"fragment\"` entry-point name per `Material2d`" reasoning as
/// `COARSE_MARCHER`'s doc.
pub fn generate_shadow_shader(obj: &Object) -> String {
    let mut s = String::new();
    s.push_str("#import bevy_sprite::mesh2d_vertex_output::VertexOutput\n\n");
    s.push_str(&generate_library());
    s.push('\n');
    s.push_str(COARSE_TEXTURE_BINDING);
    s.push('\n');
    s.push_str(&generate_scene_functions_with_divisor(
        obj,
        COARSE_ITERATION_DIVISOR,
    ));
    s.push_str("\n\n");
    s.push_str(MARCH_CORE);
    s.push('\n');
    s.push_str(SHADING_CORE);
    s.push('\n');
    s.push_str(SHADOW_MARCHER);
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{scenes, IntValue, Params, ScalarValue, Vec3Value};

    /// The concrete `VertexOutput` struct (verified against bevy_sprite
    /// 0.16.1, DESIGN.md §5), substituted for the `#import` line so naga can
    /// parse+validate the shader standalone (no naga_oil preprocessing).
    const VERTEX_OUTPUT_STRUCT: &str = "\
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}
";

    fn full_source(obj: &Object) -> String {
        let shader = generate_shader(obj);
        let import_line = "#import bevy_sprite::mesh2d_vertex_output::VertexOutput\n";
        assert!(
            shader.contains(import_line),
            "expected shader to start with the bevy_sprite import line"
        );
        shader.replacen(import_line, VERTEX_OUTPUT_STRUCT, 1)
    }

    /// Same substitution as [`full_source`], for the MRRM coarse pre-pass
    /// shader (`generate_coarse_shader`) instead of the fine one.
    fn full_coarse_source(obj: &Object) -> String {
        let shader = generate_coarse_shader(obj);
        let import_line = "#import bevy_sprite::mesh2d_vertex_output::VertexOutput\n";
        assert!(
            shader.contains(import_line),
            "expected coarse shader to start with the bevy_sprite import line"
        );
        shader.replacen(import_line, VERTEX_OUTPUT_STRUCT, 1)
    }

    /// Same substitution as [`full_source`], for the half-resolution
    /// shadow/AO pass shader (`generate_shadow_shader`).
    fn full_shadow_source(obj: &Object) -> String {
        let shader = generate_shadow_shader(obj);
        let import_line = "#import bevy_sprite::mesh2d_vertex_output::VertexOutput\n";
        assert!(
            shader.contains(import_line),
            "expected shadow shader to start with the bevy_sprite import line"
        );
        shader.replacen(import_line, VERTEX_OUTPUT_STRUCT, 1)
    }

    /// Same substitution as [`full_source`], for the auxiliary step-count-
    /// data pass shader (`generate_stepdata_shader`, `?gpuprofile=1`).
    fn full_stepdata_source(obj: &Object) -> String {
        let shader = generate_stepdata_shader(obj);
        let import_line = "#import bevy_sprite::mesh2d_vertex_output::VertexOutput\n";
        assert!(
            shader.contains(import_line),
            "expected stepdata shader to start with the bevy_sprite import line"
        );
        shader.replacen(import_line, VERTEX_OUTPUT_STRUCT, 1)
    }

    /// Parse + validate `source` with naga; panic with the naga error
    /// (naga errors carry line/column) and the full source on failure, so
    /// test failures are debuggable without weakening validation.
    fn validate_wgsl(source: &str) {
        let module = match naga::front::wgsl::parse_str(source) {
            Ok(m) => m,
            Err(e) => panic!(
                "WGSL parse error:\n{}\n\n--- source ---\n{source}",
                e.emit_to_string(source)
            ),
        };
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default(),
        );
        if let Err(e) = validator.validate(&module) {
            panic!(
                "WGSL validation error:\n{}\n\n--- source ---\n{source}",
                e.emit_to_string(source)
            );
        }
    }

    fn sphere(r: f32) -> Object {
        Object::Sphere {
            radius: ScalarValue::Const(r),
        }
    }

    fn cuboid(he: glam::Vec3) -> Object {
        Object::Cuboid {
            half_extent: Vec3Value::Const(he),
        }
    }

    #[test]
    fn demo_scene_shader_validates() {
        let mut params = Params::new();
        let (obj, _handles) = scenes::demo_scene(&mut params);
        validate_wgsl(&full_source(&obj));
    }

    #[test]
    fn nested_union_validates() {
        // Regression test for the C++ ObjectClosest::GLSL bug: fixed local
        // names (`original_p_union` etc.) collide when a Union is nested
        // inside a Union.
        let obj = Object::Union(
            Box::new(Object::Union(
                Box::new(sphere(1.0)),
                Box::new(cuboid(glam::Vec3::splat(1.0))),
            )),
            Box::new(sphere(2.0)),
        );
        validate_wgsl(&full_source(&obj));
    }

    #[test]
    fn menger_oscillating_sphere_shader_validates() {
        // Also the first naga coverage of `Object::Difference` specifically
        // (`demo_scene`/`nested_union`/`nested_repeat` above only exercise
        // `Union`/`Intersect`) -- this scene's tree is `Difference(Fractal,
        // Sphere{radius: Param})`, same shape as the existing `menger_sphere`.
        let mut params = Params::new();
        let (obj, _handles) = scenes::menger_oscillating_sphere(&mut params);
        validate_wgsl(&full_source(&obj));
    }

    #[test]
    fn stepdata_shader_validates() {
        let mut params = Params::new();
        let (obj, _handles) = scenes::demo_scene(&mut params);
        validate_wgsl(&full_stepdata_source(&obj));
    }

    #[test]
    fn nested_repeat_validates() {
        // Regression test for the C++ FoldRepeat::GLSL bug: a function-static
        // depth counter for the loop variable name, which is wrong for a
        // Repeat nested inside a Repeat (both would want depth 0 freshly per
        // call, or collide/drift across calls).
        let inner = Fold::Repeat {
            count: IntValue::Const(3),
            inner: Box::new(Fold::Abs),
        };
        let outer = Fold::Repeat {
            count: IntValue::Const(2),
            inner: Box::new(inner),
        };
        let obj = Object::Fractal {
            fold: outer,
            base: Box::new(sphere(1.0)),
        };
        validate_wgsl(&full_source(&obj));
    }

    // The same three shapes as above, run through the MRRM coarse-pass
    // generator (`generate_coarse_shader`) instead -- its `fragment` entry,
    // bind-group layout, and lack of a `coarse_tex` binding are different
    // enough from the fine shader's (see `COARSE_MARCHER`'s doc) to warrant
    // independent naga validation rather than assuming "the fine shader
    // parses" implies "the coarse one does too".
    #[test]
    fn demo_scene_coarse_shader_validates() {
        let mut params = Params::new();
        let (obj, _handles) = scenes::demo_scene(&mut params);
        validate_wgsl(&full_coarse_source(&obj));
    }

    #[test]
    fn nested_union_coarse_shader_validates() {
        let obj = Object::Union(
            Box::new(Object::Union(
                Box::new(sphere(1.0)),
                Box::new(cuboid(glam::Vec3::splat(1.0))),
            )),
            Box::new(sphere(2.0)),
        );
        validate_wgsl(&full_coarse_source(&obj));
    }

    #[test]
    fn nested_repeat_coarse_shader_validates() {
        let inner = Fold::Repeat {
            count: IntValue::Const(3),
            inner: Box::new(Fold::Abs),
        };
        let outer = Fold::Repeat {
            count: IntValue::Const(2),
            inner: Box::new(inner),
        };
        let obj = Object::Fractal {
            fold: outer,
            base: Box::new(sphere(1.0)),
        };
        validate_wgsl(&full_coarse_source(&obj));
    }

    // Same three shapes again, through the half-resolution shadow/AO pass's
    // generator (`generate_shadow_shader`) -- see `SHADOW_MARCHER`'s doc for
    // why this is a wholly separate module from both the fine and coarse
    // ones, warranting its own independent naga validation.
    #[test]
    fn demo_scene_shadow_shader_validates() {
        let mut params = Params::new();
        let (obj, _handles) = scenes::demo_scene(&mut params);
        validate_wgsl(&full_shadow_source(&obj));
    }

    #[test]
    fn nested_union_shadow_shader_validates() {
        let obj = Object::Union(
            Box::new(Object::Union(
                Box::new(sphere(1.0)),
                Box::new(cuboid(glam::Vec3::splat(1.0))),
            )),
            Box::new(sphere(2.0)),
        );
        validate_wgsl(&full_shadow_source(&obj));
    }

    #[test]
    fn nested_repeat_shadow_shader_validates() {
        let inner = Fold::Repeat {
            count: IntValue::Const(3),
            inner: Box::new(Fold::Abs),
        };
        let outer = Fold::Repeat {
            count: IntValue::Const(2),
            inner: Box::new(inner),
        };
        let obj = Object::Fractal {
            fold: outer,
            base: Box::new(sphere(1.0)),
        };
        validate_wgsl(&full_shadow_source(&obj));
    }

    #[test]
    fn generation_is_deterministic() {
        let mut params = Params::new();
        let (obj, _handles) = scenes::demo_scene(&mut params);
        let a = generate_shader(&obj);
        let b = generate_shader(&obj);
        assert_eq!(a, b, "regenerating the same tree must be byte-identical");

        let a_coarse = generate_coarse_shader(&obj);
        let b_coarse = generate_coarse_shader(&obj);
        assert_eq!(
            a_coarse, b_coarse,
            "regenerating the same tree's coarse shader must be byte-identical"
        );

        // Also check generate_scene_functions independently (it's the entry
        // point exercising CodeWriter's fresh-name counter directly).
        let c = generate_scene_functions(&obj);
        let d = generate_scene_functions(&obj);
        assert_eq!(c, d);
    }

    #[test]
    fn fine_shader_has_marble_buffer_binding_but_coarse_and_shadow_do_not() {
        // Multiplayer milestone 0: only the fine pass shades/reflects
        // marbles (`MARBLE_BUFFER_BINDING`'s doc) -- the coarse and shadow
        // passes skip all shading, so referencing this binding there would
        // be a real bind-group-layout mismatch (`CoarseMarcherMaterial`/
        // `ShadowMarcherMaterial` have no such binding), same risk class as
        // `coarse_shader_has_no_coarse_texture_binding` below.
        let mut params = Params::new();
        let (obj, _handles) = scenes::demo_scene(&mut params);
        assert!(generate_shader(&obj).contains("var<storage, read> marbles"));
        assert!(!generate_coarse_shader(&obj).contains("marbles"));
        assert!(!generate_shadow_shader(&obj).contains("marbles"));
        assert!(!generate_stepdata_shader(&obj).contains("marbles"));
    }

    #[test]
    fn stepdata_shader_has_coarse_and_shadow_texture_bindings_but_no_marble_bindings() {
        // The stepdata pass must warm-start identically to the fine pass
        // (needs `coarse_tex` for MRRM/sky-miss-skip and `shadow_tex` for the
        // shadow-tier warm-start, `STEPDATA_MARCHER`'s doc), but doesn't
        // shade anything, so it must not reference the marble buffer or
        // marble cubemap bindings -- referencing either would be a real
        // bind-group-layout mismatch (`step_data::StepDataMaterial` declares
        // neither).
        let mut params = Params::new();
        let (obj, _handles) = scenes::demo_scene(&mut params);
        let src = generate_stepdata_shader(&obj);
        assert!(src.contains("coarse_tex"), "stepdata shader must reference coarse_tex:\n{src}");
        assert!(src.contains("shadow_tex"), "stepdata shader must reference shadow_tex:\n{src}");
        assert!(!src.contains("marble_cubemap"), "stepdata shader must not reference marble_cubemap:\n{src}");
    }

    #[test]
    fn coarse_shader_has_no_coarse_texture_binding() {
        // The coarse pass must never reference its own output -- if it did,
        // that would be a real bug (either a bind-group layout mismatch at
        // runtime, since `CoarseMarcherMaterial` has no such binding, or --
        // worse -- a self-referential sample of not-yet-written data).
        let mut params = Params::new();
        let (obj, _handles) = scenes::demo_scene(&mut params);
        let src = generate_coarse_shader(&obj);
        assert!(
            !src.contains("coarse_tex"),
            "coarse shader must not reference a coarse_tex binding:\n{src}"
        );
    }

    #[test]
    fn small_tree_snippets() {
        let obj = Object::Union(
            Box::new(sphere(1.0)),
            Box::new(cuboid(glam::Vec3::new(2.0, 3.0, 4.0))),
        );
        let src = generate_scene_functions(&obj);
        assert!(src.contains("de_sphere(p, 1.0)"), "{src}");
        assert!(src.contains("de_box(p, vec3<f32>(2.0, 3.0, 4.0))"), "{src}");
        assert!(src.contains("fn de_scene(p_in: vec4<f32>) -> f32 {"), "{src}");
        assert!(
            src.contains("fn col_scene(p_in: vec4<f32>) -> vec4<f32> {"),
            "{src}"
        );
    }

    #[test]
    fn offset_emission_divides_by_saved_entry_w() {
        // An Offset around a Fractal base whose folds rescale `p.w`: the
        // emitted subtraction must divide by the saved entry-time `w`, not
        // the post-fold `p.w` (see `emit_object`'s Offset arm).
        let obj = Object::Offset {
            base: Box::new(Object::Fractal {
                fold: crate::Fold::ScaleTranslate {
                    scale: ScalarValue::Const(2.0),
                    shift: Vec3Value::Const(glam::Vec3::ZERO),
                },
                base: Box::new(sphere(1.0)),
            }),
            offset: ScalarValue::Const(0.5),
        };
        let src = generate_scene_functions(&obj);
        assert!(src.contains("let w_save_0 = p.w;"), "{src}");
        assert!(src.contains("d = d - 0.5 / w_save_0;"), "{src}");
        assert!(!src.contains("/ p.w;"), "must not divide by mutated p.w:\n{src}");
    }

    #[test]
    fn onion_emission_takes_abs_before_the_saved_w_subtraction() {
        let obj = Object::Onion {
            base: Box::new(sphere(2.0)),
            thickness: ScalarValue::Const(0.3),
        };
        let src = generate_scene_functions(&obj);
        assert!(src.contains("d = abs(d) - 0.3 / w_save_0;"), "{src}");
    }

    #[test]
    fn morph_emission_mixes_d_always_and_orbit_only_in_color_pass() {
        let obj = Object::Morph {
            a: Box::new(sphere(1.0)),
            b: Box::new(cuboid(glam::Vec3::ONE)),
            t: ScalarValue::Const(0.5),
        };
        let src = generate_scene_functions(&obj);
        let split = src.find("fn col_scene").expect("col_scene present");
        let (de_part, col_part) = src.split_at(split);
        assert!(de_part.contains("d = mix(dl_"), "{de_part}");
        assert!(de_part.contains(", 0.0, 1.0);"), "t must be clamped:\n{de_part}");
        assert!(!de_part.contains("orbit = mix("), "{de_part}");
        assert!(col_part.contains("d = mix(dl_"), "{col_part}");
        assert!(col_part.contains("orbit = mix("), "{col_part}");
    }

    #[test]
    fn torus_emission_and_hollow_donut_shader_validate() {
        let obj = Object::Torus {
            major: ScalarValue::Const(3.0),
            minor: ScalarValue::Const(1.0),
        };
        let src = generate_scene_functions(&obj);
        assert!(src.contains("d = de_torus(p, 3.0, 1.0);"), "{src}");

        // The real hollow-donut scene tree, through all four shader variants.
        let mut params = Params::new();
        let (donut, _handles) = scenes::hollow_donut(&mut params);
        validate_wgsl(&full_source(&donut));
        validate_wgsl(&full_coarse_source(&donut));
        validate_wgsl(&full_shadow_source(&donut));
        validate_wgsl(&full_stepdata_source(&donut));
    }

    #[test]
    fn onion_and_morph_shader_validates() {
        // A morph whose children include an onion-of-fractal and a nested
        // morph -- exercises the fresh-name discipline (nested p_save/dl/tm
        // locals) plus the abs/w_save emission, through all four generated
        // shader variants.
        let mut params = Params::new();
        let (classic_obj, _handles) = scenes::classic(&mut params);
        let obj = Object::Morph {
            a: Box::new(Object::Onion {
                base: Box::new(classic_obj),
                thickness: ScalarValue::Const(0.05),
            }),
            b: Box::new(Object::Morph {
                a: Box::new(sphere(1.0)),
                b: Box::new(cuboid(glam::Vec3::ONE)),
                t: ScalarValue::Const(0.25),
            }),
            t: ScalarValue::Const(0.5),
        };
        validate_wgsl(&full_source(&obj));
        validate_wgsl(&full_coarse_source(&obj));
        validate_wgsl(&full_shadow_source(&obj));
        validate_wgsl(&full_stepdata_source(&obj));
    }

    #[test]
    fn offset_shader_validates() {
        let obj = Object::Union(
            Box::new(Object::Offset {
                base: Box::new(cuboid(glam::Vec3::ONE)),
                offset: ScalarValue::Const(0.25),
            }),
            Box::new(Object::Offset {
                base: Box::new(Object::Offset {
                    base: Box::new(sphere(1.0)),
                    offset: ScalarValue::Const(0.5),
                }),
                offset: ScalarValue::Const(-0.1),
            }),
        );
        validate_wgsl(&full_source(&obj));
        validate_wgsl(&full_coarse_source(&obj));
        validate_wgsl(&full_shadow_source(&obj));
        validate_wgsl(&full_stepdata_source(&obj));
    }

    #[test]
    fn color_pass_only_has_orbit_statements() {
        let mut params = Params::new();
        let (obj, _handles) = scenes::demo_scene(&mut params);
        let src = generate_scene_functions(&obj);

        let split = src.find("fn col_scene").expect("col_scene present");
        let (de_part, col_part) = src.split_at(split);

        assert!(
            !de_part.contains("orbit ="),
            "de_scene must not contain orbit statements:\n{de_part}"
        );
        assert!(
            col_part.contains("orbit ="),
            "col_scene must contain orbit statements:\n{col_part}"
        );
    }
}
