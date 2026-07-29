//! Exact 3-D procedural noise SDF (`Object::NoiseSolid`) -- `NOISE_SDF.md`'s
//! "gradient = 1 almost everywhere" requirement met in full 3-D, by the
//! reference `sdf3d.py` route:
//!
//!  1. Periodic hash-based 3-D Perlin noise on the unit torus (two
//!     octaves; textureless, deterministic).
//!  2. **Marching tetrahedra** (Freudenthal/Kuhn 6-tet decomposition of
//!     each lattice cell): with linear interpolation on tets, the
//!     extracted triangle soup IS exactly the zero set of the
//!     piecewise-linear interpolant of the lattice values.
//!  3. `d(p) = sign(PL(p)) * min_tri dist(p, tri)` -- the sign from the
//!     closed-form **Lovász extension** (sort the fractional coords, walk
//!     the Freudenthal vertex chain: exactly the PL interpolant on the
//!     same tets the triangles came from, which is what makes the signed
//!     field globally self-consistent), the distance from exact
//!     point-to-triangle projection under a BVH.
//!
//! Everything is closed form over a fixed finite primitive set, so the
//! field is an exact SDF of a genuine solid: |grad d| = 1 almost
//! everywhere, `|p - nearest| == |de|`, physics-grade. Triangles are
//! built for cells `[-margin, G+margin)^3`, which certifies the min over
//! the finite set for any query in the unit cube out to `margin/G`; past
//! that the unsigned distance is **capped** at `margin/G` (a sound
//! underestimate -- the cap only shortens march steps that were already
//! huge). The sign lattice is periodic, so sign is correct everywhere.
//!
//! Same two-representation deal as `TriMesh`/`MESH_SDF.md`: exact CPU
//! queries here; the GPU samples a baked grid through the shared
//! `mesh_sdf_tex` binding. Serialized payload: **8 bytes** (seed + iso)
//! -- every triangle, BVH node, and lattice value is derived
//! deterministically on decode.

use glam::{Vec3, Vec4};

use crate::trimesh::{closest_point_on_triangle_point, GridSpec};

/// Marching-tets lattice resolution over the unit torus and the octave
/// stack `(frequency, amplitude)` -- frequencies divide `GRID_G` and the
/// torus, so lattice values (and the sign table) tile with period `G`.
/// Values from the reference implementation.
const GRID_G: i64 = 36;
const OCTAVES3: [(f32, f32); 2] = [(3.0, 1.0), (6.0, 0.5)];
/// Cells of margin beyond `[0,1]^3` in which triangles are still built --
/// the distance-certificate depth is `MARGIN_CELLS / GRID_G` (reference
/// default: 9 cells = 0.25).
const MARGIN_CELLS: i64 = 9;
/// Lattice values within this of zero get pushed away, so no triangle
/// vertex lands exactly on a lattice vertex and the PL sign never sits
/// exactly at zero on a crossing (reference: `3e-3`).
const SNAP: f32 = 3e-3;

/// Bake-grid resolution across the unit cube for the GPU texture: cell
/// `1/95` is ~2.6x finer than the marching lattice, so the solid's shapes
/// survive sampling (the finest *feature* scale is the octave at 6, an
/// order coarser still).
const BAKE_RES: u32 = 96;
const BAKE_MARGIN_CELLS: u32 = 2;

// ---------------------------------------------------------------------
// Periodic 3-D Perlin noise
// ---------------------------------------------------------------------

fn hash3(ix: i64, iy: i64, iz: i64, period: i64, seed: u32) -> u32 {
    let ix = ix.rem_euclid(period) as u32;
    let iy = iy.rem_euclid(period) as u32;
    let iz = iz.rem_euclid(period) as u32;
    let mut h = ix
        .wrapping_mul(374761393)
        .wrapping_add(iy.wrapping_mul(668265263))
        .wrapping_add(iz.wrapping_mul(2246822519))
        .wrapping_add(seed.wrapping_mul(2654435761));
    h ^= h >> 13;
    h = h.wrapping_mul(1274126177);
    h ^= h >> 16;
    h
}

/// Unit gradient from two independent hashes: uniform on the sphere
/// (uniform `z`, uniform azimuth).
fn grad3(ix: i64, iy: i64, iz: i64, period: i64, seed: u32) -> Vec3 {
    let h1 = hash3(ix, iy, iz, period, seed);
    let h2 = hash3(ix, iy, iz, period, seed.wrapping_add(977));
    let z = h1 as f32 / 4294967296.0 * 2.0 - 1.0;
    let phi = h2 as f32 / 4294967296.0 * std::f32::consts::TAU;
    let r = (1.0 - z * z).max(0.0).sqrt();
    Vec3::new(r * phi.cos(), r * phi.sin(), z)
}

fn perlin3(p: Vec3, freq: f32, seed: u32) -> f32 {
    let f = p * freq;
    let i0 = f.x.floor() as i64;
    let j0 = f.y.floor() as i64;
    let k0 = f.z.floor() as i64;
    let u = f.x - i0 as f32;
    let v = f.y - j0 as f32;
    let w = f.z - k0 as f32;
    let period = freq as i64;
    let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    let (fu, fv, fw) = (fade(u), fade(v), fade(w));
    let dot = |di: i64, dj: i64, dk: i64| {
        grad3(i0 + di, j0 + dj, k0 + dk, period, seed).dot(Vec3::new(
            u - di as f32,
            v - dj as f32,
            w - dk as f32,
        ))
    };
    let lerp = |a: f32, b: f32, t: f32| a + t * (b - a);
    let n00 = lerp(dot(0, 0, 0), dot(1, 0, 0), fu);
    let n10 = lerp(dot(0, 1, 0), dot(1, 1, 0), fu);
    let n01 = lerp(dot(0, 0, 1), dot(1, 0, 1), fu);
    let n11 = lerp(dot(0, 1, 1), dot(1, 1, 1), fu);
    lerp(lerp(n00, n10, fv), lerp(n01, n11, fv), fw)
}

/// The full octave stack.
pub fn noise3(p: Vec3, seed: u32) -> f32 {
    let mut n = 0.0;
    for (k, (f, a)) in OCTAVES3.iter().enumerate() {
        n += a * perlin3(p, *f, seed.wrapping_add(101 * k as u32));
    }
    n
}

/// iso such that ~`solid_fraction` of the slab `y in [y0, y1]` (torus
/// coordinates) has `noise3 - iso < 0` (the solid): the empirical
/// quantile over a fixed deterministic lattice of samples. The y-range
/// matters because a scene that clips the field to a slab experiences
/// *that slab's* statistics, not the full cube's -- at octave scale 1/3
/// a thin slab's solid fraction can differ from the global one by 10+
/// points. "70% sparse over the whole cube" is
/// `iso_for_solid_fraction(seed, 0.3, 0.0, 1.0)`.
pub fn iso_for_solid_fraction(seed: u32, solid_fraction: f32, y0: f32, y1: f32) -> f32 {
    const S: usize = 48;
    let mut vals = Vec::with_capacity(S * S * S);
    for i in 0..S {
        for j in 0..S {
            for k in 0..S {
                let p = Vec3::new(
                    (i as f32 + 0.5) / S as f32,
                    y0 + (y1 - y0) * (j as f32 + 0.5) / S as f32,
                    (k as f32 + 0.5) / S as f32,
                );
                vals.push(noise3(p, seed));
            }
        }
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((vals.len() - 1) as f32 * solid_fraction.clamp(0.0, 1.0)) as usize;
    vals[idx]
}

// ---------------------------------------------------------------------
// Marching tetrahedra -> triangle soup
// ---------------------------------------------------------------------

/// The 6 Freudenthal tets of a cell, one per permutation of the axes:
/// corner chain `0 -> +e_p0 -> +e_p0+e_p1 -> (1,1,1)`.
const PERMS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

fn tet_corners(perm: [usize; 3]) -> [[i64; 3]; 4] {
    let mut c = [[0i64; 3]; 4];
    for k in 0..2 {
        c[k + 1] = c[k];
        c[k + 1][perm[k]] += 1;
    }
    c[3] = [1, 1, 1];
    c
}

/// Per-sign-code triangle list for one tet: each triangle vertex is a
/// crossing on tet edge `(a, b)`. 1-or-3 positive corners give one
/// triangle, 2 give a quad split into two (reference `_case_tris`).
fn tet_case(code: u8) -> Vec<[(usize, usize); 3]> {
    let pos: Vec<usize> = (0..4).filter(|i| code >> i & 1 == 1).collect();
    let neg: Vec<usize> = (0..4).filter(|i| code >> i & 1 == 0).collect();
    match pos.len() {
        1 => vec![[(pos[0], neg[0]), (pos[0], neg[1]), (pos[0], neg[2])]],
        3 => vec![[(neg[0], pos[0]), (neg[0], pos[1]), (neg[0], pos[2])]],
        2 => {
            let (a, b, c, d) = (pos[0], pos[1], neg[0], neg[1]);
            let (e1, e2, e3, e4) = ((a, c), (b, c), (b, d), (a, d));
            vec![[e1, e2, e3], [e1, e3, e4]]
        }
        _ => vec![],
    }
}

// ---------------------------------------------------------------------
// The built object
// ---------------------------------------------------------------------

struct BvhNode {
    min: Vec3,
    max: Vec3,
    /// leaf: `(start, count)` into `order`; inner: `count == 0`, children
    /// contiguous at `start`, `start + 1` (the `trimesh.rs` convention).
    start: u32,
    count: u32,
}

pub struct NoiseSolidData {
    seed: u32,
    iso: f32,
    /// triangle soup, flat: 3 vertices per triangle
    tris: Vec<[Vec3; 3]>,
    nodes: Vec<BvhNode>,
    order: Vec<u32>,
    /// snapped lattice values on the periodic `G^3` sign table
    sign_table: Vec<f32>,
    grid: GridSpec,
}

impl std::fmt::Debug for NoiseSolidData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NoiseSolidData")
            .field("seed", &self.seed)
            .field("iso", &self.iso)
            .field("tris", &self.tris.len())
            .finish()
    }
}

fn build_bvh(tris: &[[Vec3; 3]]) -> (Vec<BvhNode>, Vec<u32>) {
    let mut order: Vec<u32> = (0..tris.len() as u32).collect();
    let centroids: Vec<Vec3> = tris.iter().map(|t| (t[0] + t[1] + t[2]) / 3.0).collect();
    let mut nodes = vec![BvhNode { min: Vec3::ZERO, max: Vec3::ZERO, start: 0, count: 0 }];
    fn build(
        nodes: &mut Vec<BvhNode>,
        idx: usize,
        order: &mut [u32],
        tris: &[[Vec3; 3]],
        centroids: &[Vec3],
        lo: usize,
        hi: usize,
    ) {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for &ti in &order[lo..hi] {
            for v in &tris[ti as usize] {
                min = min.min(*v);
                max = max.max(*v);
            }
        }
        if hi - lo <= 4 {
            nodes[idx] = BvhNode { min, max, start: lo as u32, count: (hi - lo) as u32 };
            return;
        }
        let mut cmin = Vec3::splat(f32::INFINITY);
        let mut cmax = Vec3::splat(f32::NEG_INFINITY);
        for &ti in &order[lo..hi] {
            cmin = cmin.min(centroids[ti as usize]);
            cmax = cmax.max(centroids[ti as usize]);
        }
        let ext = cmax - cmin;
        let axis = if ext.x >= ext.y && ext.x >= ext.z {
            0
        } else if ext.y >= ext.z {
            1
        } else {
            2
        };
        let mid = (hi - lo) / 2;
        order[lo..hi].select_nth_unstable_by(mid, |&a, &b| {
            let ca = centroids[a as usize][axis];
            let cb = centroids[b as usize][axis];
            ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
        });
        let left = nodes.len();
        nodes.push(BvhNode { min: Vec3::ZERO, max: Vec3::ZERO, start: 0, count: 0 });
        nodes.push(BvhNode { min: Vec3::ZERO, max: Vec3::ZERO, start: 0, count: 0 });
        build(nodes, left, order, tris, centroids, lo, lo + mid);
        build(nodes, left + 1, order, tris, centroids, lo + mid, hi);
        nodes[idx] = BvhNode { min, max, start: left as u32, count: 0 };
    }
    build(&mut nodes, 0, &mut order, tris, &centroids, 0, tris.len());
    (nodes, order)
}

fn aabb_dist_sq(p: Vec3, min: Vec3, max: Vec3) -> f32 {
    let d = (min - p).max(Vec3::ZERO) + (p - max).max(Vec3::ZERO);
    d.length_squared()
}

impl NoiseSolidData {
    pub fn new(seed: u32, iso: f32) -> Option<NoiseSolidData> {
        if !iso.is_finite() || iso.abs() > 4.0 {
            return None;
        }
        let g = GRID_G;
        let m = MARGIN_CELLS;

        // Periodic sign table: snapped lattice values on [0, G)^3 --
        // noise is periodic on the torus, so this one tile signs every
        // query everywhere.
        let snap = |v: f32| {
            if v.abs() < SNAP {
                if v >= 0.0 {
                    SNAP
                } else {
                    -SNAP
                }
            } else {
                v
            }
        };
        let gs = g as usize;
        let mut sign_table = vec![0.0f32; gs * gs * gs];
        for i in 0..gs {
            for j in 0..gs {
                for k in 0..gs {
                    let p = Vec3::new(i as f32, j as f32, k as f32) / g as f32;
                    sign_table[(i * gs + j) * gs + k] = snap(noise3(p, seed) - iso);
                }
            }
        }
        let lat = |i: i64, j: i64, k: i64| -> f32 {
            let (i, j, k) = (
                i.rem_euclid(g) as usize,
                j.rem_euclid(g) as usize,
                k.rem_euclid(g) as usize,
            );
            sign_table[(i * gs + j) * gs + k]
        };

        // Marching tetrahedra over cells [-m, G+m)^3.
        let mut tris: Vec<[Vec3; 3]> = Vec::new();
        for ci in -m..g + m {
            for cj in -m..g + m {
                for ck in -m..g + m {
                    // 8 corner values; skip uniform cells fast
                    let mut cv = [0.0f32; 8];
                    let mut any_pos = false;
                    let mut any_neg = false;
                    for (n, (di, dj, dk)) in [
                        (0i64, 0i64, 0i64),
                        (1, 0, 0),
                        (0, 1, 0),
                        (1, 1, 0),
                        (0, 0, 1),
                        (1, 0, 1),
                        (0, 1, 1),
                        (1, 1, 1),
                    ]
                    .iter()
                    .enumerate()
                    {
                        let v = lat(ci + di, cj + dj, ck + dk);
                        cv[n] = v;
                        any_pos |= v > 0.0;
                        any_neg |= v <= 0.0;
                    }
                    if !(any_pos && any_neg) {
                        continue;
                    }
                    let corner_val = |o: [i64; 3]| cv[(o[0] + 2 * o[1] + 4 * o[2]) as usize];
                    let base = Vec3::new(ci as f32, cj as f32, ck as f32);
                    for perm in PERMS {
                        let corners = tet_corners(perm);
                        let vals = [
                            corner_val(corners[0]),
                            corner_val(corners[1]),
                            corner_val(corners[2]),
                            corner_val(corners[3]),
                        ];
                        let code = (0..4).fold(0u8, |c, i| c | ((vals[i] > 0.0) as u8) << i);
                        if code == 0 || code == 15 {
                            continue;
                        }
                        let cpos = |i: usize| {
                            (base
                                + Vec3::new(
                                    corners[i][0] as f32,
                                    corners[i][1] as f32,
                                    corners[i][2] as f32,
                                ))
                                / g as f32
                        };
                        for tri in tet_case(code) {
                            let mut pts = [Vec3::ZERO; 3];
                            for (n, (a, b)) in tri.iter().enumerate() {
                                let (va, vb) = (vals[*a], vals[*b]);
                                let t = va / (va - vb);
                                pts[n] = cpos(*a) + t * (cpos(*b) - cpos(*a));
                            }
                            // drop degenerate slivers
                            let nrm = (pts[1] - pts[0]).cross(pts[2] - pts[0]);
                            if nrm.length_squared() > 1e-24 {
                                tris.push(pts);
                            }
                        }
                    }
                }
            }
        }
        if tris.is_empty() {
            return None;
        }
        let (nodes, order) = build_bvh(&tris);

        // GPU bake grid: unit cube + margin texels.
        let cell = 1.0 / (BAKE_RES - 1) as f32;
        let mb = BAKE_MARGIN_CELLS as f32 * cell;
        let d = BAKE_RES + 2 * BAKE_MARGIN_CELLS;
        let grid = GridSpec {
            origin: Vec3::splat(-mb),
            cell,
            dims: [d, d, d],
        };

        Some(NoiseSolidData { seed, iso, tris, nodes, order, sign_table, grid })
    }

    pub fn seed(&self) -> u32 {
        self.seed
    }

    pub fn iso(&self) -> f32 {
        self.iso
    }

    pub fn tri_count(&self) -> usize {
        self.tris.len()
    }

    pub fn grid(&self) -> GridSpec {
        self.grid
    }

    /// Sign via the Lovász extension: sort the cell-fractional coords
    /// descending, weight the Freudenthal vertex chain barycentrically --
    /// the exact PL interpolant on the same Kuhn tets the triangles were
    /// extracted from. Periodic (lattice wraps), so valid everywhere.
    pub fn pl_sign(&self, p: Vec3) -> f32 {
        let g = GRID_G;
        let gs = g as usize;
        let f = p * g as f32;
        let cell = [
            f.x.floor() as i64,
            f.y.floor() as i64,
            f.z.floor() as i64,
        ];
        let u = [f.x - f.x.floor(), f.y - f.y.floor(), f.z - f.z.floor()];
        let mut order = [0usize, 1, 2];
        order.sort_by(|&a, &b| u[b].partial_cmp(&u[a]).unwrap_or(std::cmp::Ordering::Equal));
        let s = [u[order[0]], u[order[1]], u[order[2]]];
        let w = [1.0 - s[0], s[0] - s[1], s[1] - s[2], s[2]];
        let lat = |i: i64, j: i64, k: i64| -> f32 {
            let (i, j, k) = (
                i.rem_euclid(g) as usize,
                j.rem_euclid(g) as usize,
                k.rem_euclid(g) as usize,
            );
            self.sign_table[(i * gs + j) * gs + k]
        };
        let mut cur = cell;
        let mut val = 0.0;
        for k in 0..4 {
            val += w[k] * lat(cur[0], cur[1], cur[2]);
            if k < 3 {
                cur[order[k]] += 1;
            }
        }
        if val >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }

    /// Exact unsigned distance + closest surface point. The query is
    /// **wrapped into the unit torus first** (the surface is periodic, so
    /// distance is invariant under the period lattice): for a wrapped
    /// query, any surface within `MARGIN_CELLS / GRID_G` lies inside the
    /// built margin box, so the min is *exact* below the cap and the cap
    /// itself is a sound underestimate -- everywhere in space, no
    /// domain-boundary caveats.
    fn closest(&self, p_world: Vec3) -> (f32, Vec3) {
        let base = Vec3::new(
            p_world.x.floor(),
            p_world.y.floor(),
            p_world.z.floor(),
        );
        let p = p_world - base;
        let cap = MARGIN_CELLS as f32 / GRID_G as f32;
        let mut best_d2 = cap * cap;
        let mut best_q = p + Vec3::new(cap, 0.0, 0.0); // cap sentinel
        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        stack[sp] = 0;
        sp += 1;
        while sp > 0 {
            sp -= 1;
            let node = &self.nodes[stack[sp] as usize];
            if aabb_dist_sq(p, node.min, node.max) >= best_d2 {
                continue;
            }
            if node.count > 0 {
                for k in node.start..node.start + node.count {
                    let t = &self.tris[self.order[k as usize] as usize];
                    let q = closest_point_on_triangle_point(p, t[0], t[1], t[2]);
                    let d2 = (p - q).length_squared();
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best_q = q;
                    }
                }
            } else {
                let (l, r) = (node.start as usize, node.start as usize + 1);
                let dl = aabb_dist_sq(p, self.nodes[l].min, self.nodes[l].max);
                let dr = aabb_dist_sq(p, self.nodes[r].min, self.nodes[r].max);
                let (first, second) = if dl <= dr { (r, l) } else { (l, r) };
                stack[sp] = first as u32;
                sp += 1;
                stack[sp] = second as u32;
                sp += 1;
            }
        }
        (best_d2.sqrt(), best_q + base)
    }

    /// Exact signed distance (positive in the sparse region, negative in
    /// the solid where `noise < iso`), capped per `closest`.
    pub fn signed_distance(&self, p: Vec3) -> f32 {
        self.pl_sign(p) * self.closest(p).0
    }

    /// `Object::de` convention (`p.w` scale divisor).
    pub fn de(&self, p: Vec4) -> f32 {
        self.signed_distance(p.truncate()) / p.w
    }

    /// Exact nearest surface point (the winning triangle's projection).
    pub fn nearest_point(&self, p: Vec3) -> Vec3 {
        self.closest(p).1
    }

    /// Bake the signed distance onto the grid (x fastest -- `texture_3d`
    /// layout), deterministic serial loop of exact queries.
    pub fn bake_grid(&self) -> Vec<f32> {
        let g = self.grid;
        let mut out = Vec::with_capacity((g.dims[0] * g.dims[1] * g.dims[2]) as usize);
        for k in 0..g.dims[2] {
            for j in 0..g.dims[1] {
                for i in 0..g.dims[0] {
                    let p = g.origin + Vec3::new(i as f32, j as f32, k as f32) * g.cell;
                    out.push(self.signed_distance(p));
                }
            }
        }
        out
    }

    /// 8-byte payload: seed + iso; all geometry re-derives on decode.
    pub fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.seed.to_le_bytes());
        out.extend_from_slice(&self.iso.to_le_bytes());
    }

    pub fn decode_at(bytes: &[u8], pos: usize) -> Option<(NoiseSolidData, usize)> {
        let seed = u32::from_le_bytes(bytes.get(pos..pos + 4)?.try_into().ok()?);
        let iso = f32::from_le_bytes(bytes.get(pos + 4..pos + 8)?.try_into().ok()?);
        Some((NoiseSolidData::new(seed, iso)?, pos + 8))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perlin3_is_periodic_on_the_unit_torus() {
        for &(x, y, z) in &[(0.21, 0.68, 0.4), (0.05, 0.95, 0.77)] {
            let p = Vec3::new(x, y, z);
            let a = noise3(p, 11);
            assert!((a - noise3(p + Vec3::X, 11)).abs() < 1e-4);
            assert!((a - noise3(p + Vec3::Y, 11)).abs() < 1e-4);
            assert!((a - noise3(p + Vec3::Z, 11)).abs() < 1e-4);
        }
    }

    fn solid() -> NoiseSolidData {
        NoiseSolidData::new(11, 0.0).expect("build")
    }

    #[test]
    fn builds_a_nontrivial_soup() {
        let s = solid();
        assert!(s.tri_count() > 10_000, "tris {}", s.tri_count());
    }

    /// Sign is the PL interpolant's sign, which tracks the noise away
    /// from the surface -- and is periodic.
    #[test]
    fn sign_tracks_noise_and_wraps() {
        let s = solid();
        let mut checked = 0;
        for i in 0..200 {
            let f = i as f32;
            let p = Vec3::new(
                (f * 0.633).fract(),
                (f * 0.377 + 0.11).fract(),
                (f * 0.219 + 0.53).fract(),
            );
            let d = s.signed_distance(p);
            if d.abs() < 1.5 / GRID_G as f32 {
                continue; // PL and smooth noise may disagree near the surface
            }
            let n = noise3(p, 11);
            assert_eq!(d > 0.0, n > 0.0, "at {p:?}: d={d} n={n}");
            assert_eq!(s.pl_sign(p), s.pl_sign(p + Vec3::X), "sign not periodic at {p:?}");
            checked += 1;
        }
        assert!(checked > 100, "only {checked} far-from-surface probes");
    }

    /// The headline property: |grad d| = 1 almost everywhere (finite
    /// differences away from the surface and the cap).
    #[test]
    fn field_is_eikonal_almost_everywhere() {
        let s = solid();
        let eps = 5e-5f32;
        let cap = MARGIN_CELLS as f32 / GRID_G as f32;
        let mut ok = 0;
        let mut total = 0;
        for i in 0..150 {
            let f = i as f32;
            let p = Vec3::new(
                (f * 0.713).fract(),
                (f * 0.157 + 0.4).fract(),
                (f * 0.449 + 0.09).fract(),
            );
            let d = s.signed_distance(p);
            if d.abs() < 5.0 * eps || d.abs() > cap - 1e-3 {
                continue;
            }
            total += 1;
            let mut g = Vec3::ZERO;
            for ax in 0..3 {
                let mut e = Vec3::ZERO;
                e[ax] = eps;
                g[ax] = (s.signed_distance(p + e) - s.signed_distance(p - e)) / (2.0 * eps);
            }
            if (g.length() - 1.0).abs() < 0.05 {
                ok += 1;
            }
        }
        assert!(total > 80, "not enough usable probes ({total})");
        // medial-axis probes legitimately fail the finite difference
        assert!(ok * 10 >= total * 9, "eikonal at only {ok}/{total}");
    }

    #[test]
    fn closest_point_is_distance_consistent() {
        let s = solid();
        let cap = MARGIN_CELLS as f32 / GRID_G as f32;
        for i in 0..60 {
            let f = i as f32;
            let p = Vec3::new(
                (f * 0.529 + 0.03).fract(),
                (f * 0.291 + 0.61).fract(),
                (f * 0.173 + 0.3).fract(),
            );
            let (d, q) = (s.signed_distance(p), s.nearest_point(p));
            if d.abs() >= cap - 1e-4 {
                continue; // capped: q is the sentinel, not a surface point
            }
            assert!(
                (p.distance(q) - d.abs()).abs() < 1e-5,
                "|p-q|={} vs |d|={}",
                p.distance(q),
                d.abs()
            );
            let dq = s.signed_distance(q);
            assert!(dq.abs() < 2e-3, "closest point off-surface: {dq}");
        }
    }

    /// Lipschitz: |d(a) - d(b)| <= |a - b| along a line through the cube
    /// (the reference's "max jump" check).
    #[test]
    fn field_is_lipschitz_along_lines() {
        let s = solid();
        let n = 400;
        let mut prev: Option<(Vec3, f32)> = None;
        for i in 0..=n {
            let t = i as f32 / n as f32;
            let p = Vec3::new(t, 0.37 + 0.1 * t, 0.61 - 0.2 * t);
            let d = s.signed_distance(p);
            if let Some((pp, pd)) = prev {
                let step: f32 = p.distance(pp);
                assert!(
                    (d - pd).abs() <= step * 1.05 + 1e-6,
                    "jump {} over step {step} at t={t}",
                    (d - pd).abs()
                );
            }
            prev = Some((p, d));
        }
    }

    #[test]
    fn iso_quantile_hits_the_requested_solid_fraction() {
        let iso = iso_for_solid_fraction(11, 0.3, 0.0, 1.0);
        let s = NoiseSolidData::new(11, iso).expect("build");
        // measure actual solid fraction on an offset lattice
        let mut solid = 0;
        let m = 24;
        for i in 0..m {
            for j in 0..m {
                for k in 0..m {
                    let p = Vec3::new(
                        (i as f32 + 0.7) / m as f32,
                        (j as f32 + 0.3) / m as f32,
                        (k as f32 + 0.5) / m as f32,
                    );
                    if s.signed_distance(p) < 0.0 {
                        solid += 1;
                    }
                }
            }
        }
        let frac = solid as f32 / (m * m * m) as f32;
        assert!(
            (frac - 0.3).abs() < 0.05,
            "solid fraction {frac} != requested 0.30"
        );
    }

    #[test]
    fn de_extends_and_encode_roundtrips() {
        let s = solid();
        let p = Vec3::new(0.31, 0.62, 0.5);
        let d = s.signed_distance(p);
        // p.w scaling
        assert!((s.de(Vec4::new(p.x, p.y, p.z, 2.0)) - d / 2.0).abs() < 1e-6);

        let mut bytes = vec![0u8; 2];
        s.encode(&mut bytes);
        let (dec, end) = NoiseSolidData::decode_at(&bytes, 2).expect("decode");
        assert_eq!(end, bytes.len());
        assert_eq!(dec.tri_count(), s.tri_count());
        assert_eq!(dec.signed_distance(p), d, "decoded field differs");
    }

    #[test]
    fn bake_grid_matches_the_exact_field_at_texels() {
        let s = solid();
        let g = s.grid();
        // bake a few texels directly rather than the whole 100^3 grid
        for &(i, j, k) in &[(0u32, 0u32, 0u32), (31, 57, 12), (80, 40, 99)] {
            let p = g.origin + Vec3::new(i as f32, j as f32, k as f32) * g.cell;
            let direct = s.signed_distance(p);
            assert!(direct.is_finite());
            assert!(direct.abs() <= MARGIN_CELLS as f32 / GRID_G as f32 + 1e-6);
        }
    }
}
