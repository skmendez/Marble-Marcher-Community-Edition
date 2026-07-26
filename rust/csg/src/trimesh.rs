//! Triangle meshes as exact signed distance fields (`Object::TriMesh`).
//!
//! The two-representation design from `rust/MESH_SDF.md`: the **CPU** answers
//! every query *exactly* — a BVH finds the closest point on the closest
//! triangle, and the sign comes from the angle-weighted pseudonormal of the
//! feature actually hit (Bærentzen & Aanæs 2005, provably correct for any
//! closed 2-manifold — the edge/vertex cases are exactly what a naive
//! face-normal dot product gets wrong). Physics therefore collides against a
//! field of the same quality as the analytic primitives: `de` is the true
//! signed distance and `nearest_point` the true closest surface point.
//!
//! The **GPU** samples a baked distance grid instead (`bake_grid`, uploaded
//! as a `texture_3d` by the app): one manually-trilinear lookup per march
//! step, O(1) in triangle count. The grid is *derived* state — peers re-bake
//! it deterministically from the synced mesh bytes (a serial loop of exact
//! queries; f32 arithmetic is IEEE-identical across native and wasm), so it
//! never crosses the wire and never enters the `Object` encoding.
//!
//! Construction validates closedness (every undirected edge shared by
//! exactly two consistently-wound triangles): the pseudonormal sign argument
//! *requires* a closed manifold, so a mesh that isn't one is rejected at the
//! door (`TriMeshData::new` returns `None`, and a hostile/corrupt decode
//! fails instead of producing a tree whose `de` lies about containment).

use glam::{Vec3, Vec4};

/// Grid resolution target for [`TriMeshData::grid`]: cell count along the
/// mesh's longest axis. 64 puts the cell size of a unit-scale prop near
/// pixel scale at the game's typical view distances; the volume is
/// ~64³ f32 ≈ 1 MB, well within budget (MESH_SDF.md §2).
const GRID_RES: u32 = 64;

/// Empty cells of padding around the mesh's AABB on every side. Two cells
/// keep the trilinear stencil off the boundary for any in-box query and
/// guarantee boundary samples are positive (the outside-box `max` rule in
/// the shader relies on clamped samples never lying).
const GRID_MARGIN_CELLS: u32 = 2;

/// Hard caps for decode (`scene_sync` fix-6 discipline): a hostile length
/// prefix must be rejected before any allocation is sized from it. Generous
/// for props (the bunny is 662/1316) while keeping worst-case scene-sync
/// payloads bounded (~1.5 MB).
const MAX_VERTS: usize = 65_536;
const MAX_TRIS: usize = 131_072;

/// Baked-grid placement: `point(i,j,k) = origin + cell * (i,j,k)`, with
/// `dims` grid *points* (not cells) per axis. Derived deterministically from
/// the mesh bounds at construction, so codegen (which inlines these
/// constants into WGSL) and the app's bake agree by construction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridSpec {
    pub origin: Vec3,
    pub cell: f32,
    pub dims: [u32; 3],
}

/// Where on a triangle a closest-point query landed — selects which
/// pseudonormal decides the sign. Interior hits use the face normal; edge
/// and vertex hits use the angle-weighted averages, which is what makes the
/// sign test exact at creases (Bærentzen & Aanæs 2005).
#[derive(Clone, Copy)]
enum Feature {
    Face,
    Edge(usize),
    Vert(usize),
}

struct BvhNode {
    min: Vec3,
    max: Vec3,
    /// Leaf: `(start, count)` into `tri_order`. Inner: `(!0, left_child)`
    /// with `right_child = left_child + 1`... encoded as: `count == 0` means
    /// inner node with children at `start` and `start + 1`.
    start: u32,
    count: u32,
}

/// An immutable, validated triangle mesh with everything `Object::TriMesh`
/// needs precomputed: BVH, per-feature pseudonormals, bounding sphere, and
/// the bake-grid placement. Built once (construction or decode) and shared
/// via `Arc` — `Object` stays cheaply `Clone`.
pub struct TriMeshData {
    verts: Vec<Vec3>,
    tris: Vec<[u32; 3]>,
    face_normal: Vec<Vec3>,
    /// Per triangle, per edge `(v0v1, v1v2, v2v0)`: normalized sum of the
    /// two adjacent face normals (both incident triangles store the same
    /// vector for their shared edge).
    edge_pn: Vec<[Vec3; 3]>,
    /// Per vertex: angle-weighted sum of incident face normals, normalized.
    vert_pn: Vec<Vec3>,
    nodes: Vec<BvhNode>,
    tri_order: Vec<u32>,
    center: Vec3,
    radius: f32,
    grid: GridSpec,
    /// FNV-1a over the encoded bytes — identity for `Debug`/round-trip
    /// comparison without dumping thousands of floats into test output.
    content_hash: u64,
}

impl std::fmt::Debug for TriMeshData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TriMeshData")
            .field("verts", &self.verts.len())
            .field("tris", &self.tris.len())
            .field("content_hash", &self.content_hash)
            .finish()
    }
}

fn closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (Vec3, Feature) {
    // Ericson, Real-Time Collision Detection §5.1.5, with the Voronoi
    // region kept so the caller knows which pseudonormal applies.
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, Feature::Vert(0));
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, Feature::Vert(1));
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let t = d1 / (d1 - d3);
        return (a + ab * t, Feature::Edge(0));
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, Feature::Vert(2));
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let t = d2 / (d2 - d6);
        return (a + ac * t, Feature::Edge(2));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + (c - b) * t, Feature::Edge(1));
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (a + ab * v + ac * w, Feature::Face)
}

fn aabb_dist_sq(p: Vec3, min: Vec3, max: Vec3) -> f32 {
    let d = (min - p).max(Vec3::ZERO) + (p - max).max(Vec3::ZERO);
    d.length_squared()
}

const LEAF_SIZE: u32 = 4;

impl TriMeshData {
    /// Build from raw vertices and triangles. Returns `None` unless the mesh
    /// is a closed, consistently oriented 2-manifold with in-range indices
    /// and finite, non-degenerate geometry — the preconditions the
    /// pseudonormal sign proof needs (module doc).
    pub fn new(verts: Vec<Vec3>, tris: Vec<[u32; 3]>) -> Option<TriMeshData> {
        if verts.is_empty() || tris.is_empty() {
            return None;
        }
        if verts.len() > MAX_VERTS || tris.len() > MAX_TRIS {
            return None;
        }
        for v in &verts {
            if !v.is_finite() {
                return None;
            }
        }
        let nv = verts.len() as u32;
        for t in &tris {
            if t[0] >= nv || t[1] >= nv || t[2] >= nv {
                return None;
            }
            if t[0] == t[1] || t[1] == t[2] || t[2] == t[0] {
                return None;
            }
        }

        // Closedness + consistent winding: every undirected edge must be
        // used exactly twice, once in each direction.
        let mut edge_uses: std::collections::HashMap<(u32, u32), (u32, i32)> =
            std::collections::HashMap::new();
        for t in &tris {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                let key = (a.min(b), a.max(b));
                let e = edge_uses.entry(key).or_insert((0, 0));
                e.0 += 1;
                e.1 += if a < b { 1 } else { -1 };
            }
        }
        if edge_uses.values().any(|&(n, dir)| n != 2 || dir != 0) {
            return None;
        }

        // Face normals (reject degenerate triangles: a zero-area face has
        // no normal to weight, and its "closest feature" logic collapses).
        let mut face_normal = Vec::with_capacity(tris.len());
        for t in &tris {
            let n = (verts[t[1] as usize] - verts[t[0] as usize])
                .cross(verts[t[2] as usize] - verts[t[0] as usize]);
            let len = n.length();
            if !(len > 1e-20) {
                return None;
            }
            face_normal.push(n / len);
        }

        // Edge pseudonormals: sum of the two incident face normals.
        let mut edge_sum: std::collections::HashMap<(u32, u32), Vec3> =
            std::collections::HashMap::new();
        for (ti, t) in tris.iter().enumerate() {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                let key = (a.min(b), a.max(b));
                *edge_sum.entry(key).or_insert(Vec3::ZERO) += face_normal[ti];
            }
        }
        let edge_pn: Vec<[Vec3; 3]> = tris
            .iter()
            .map(|t| {
                [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])].map(|(a, b)| {
                    edge_sum[&(a.min(b), a.max(b))].normalize_or_zero()
                })
            })
            .collect();

        // Vertex pseudonormals: incident-angle-weighted face normals.
        let mut vert_pn = vec![Vec3::ZERO; verts.len()];
        for (ti, t) in tris.iter().enumerate() {
            for k in 0..3 {
                let v = verts[t[k] as usize];
                let e1 = (verts[t[(k + 1) % 3] as usize] - v).normalize_or_zero();
                let e2 = (verts[t[(k + 2) % 3] as usize] - v).normalize_or_zero();
                let angle = e1.dot(e2).clamp(-1.0, 1.0).acos();
                vert_pn[t[k] as usize] += face_normal[ti] * angle;
            }
        }
        for pn in &mut vert_pn {
            *pn = pn.normalize_or_zero();
        }

        // Median-split BVH over triangle centroids.
        let mut tri_order: Vec<u32> = (0..tris.len() as u32).collect();
        let centroids: Vec<Vec3> = tris
            .iter()
            .map(|t| {
                (verts[t[0] as usize] + verts[t[1] as usize] + verts[t[2] as usize]) / 3.0
            })
            .collect();
        let mut nodes = vec![BvhNode {
            min: Vec3::ZERO,
            max: Vec3::ZERO,
            start: 0,
            count: 0,
        }];
        build_bvh_into(
            &mut nodes,
            0,
            &mut tri_order,
            &centroids,
            &verts,
            &tris,
            0,
            tris.len(),
        );

        // Bounding sphere: AABB center, exact max vertex distance.
        let (mut lo, mut hi) = (verts[0], verts[0]);
        for v in &verts {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
        let center = (lo + hi) * 0.5;
        let radius = verts
            .iter()
            .map(|v| (*v - center).length())
            .fold(0.0f32, f32::max);

        // Grid spec: GRID_RES cells along the longest axis, margin cells of
        // guaranteed-outside padding all around.
        let extent = hi - lo;
        let cell = extent.max_element() / (GRID_RES - 1) as f32;
        let m = GRID_MARGIN_CELLS as f32 * cell;
        let origin = lo - Vec3::splat(m);
        let dims = [
            (extent.x / cell).ceil() as u32 + 1 + 2 * GRID_MARGIN_CELLS,
            (extent.y / cell).ceil() as u32 + 1 + 2 * GRID_MARGIN_CELLS,
            (extent.z / cell).ceil() as u32 + 1 + 2 * GRID_MARGIN_CELLS,
        ];
        let grid = GridSpec { origin, cell, dims };

        let mut mesh = TriMeshData {
            verts,
            tris,
            face_normal,
            edge_pn,
            vert_pn,
            nodes,
            tri_order,
            center,
            radius,
            grid,
            content_hash: 0,
        };
        let mut bytes = Vec::new();
        mesh.encode(&mut bytes);
        mesh.content_hash = fnv1a(&bytes);
        Some(mesh)
    }

    pub fn vert_count(&self) -> usize {
        self.verts.len()
    }

    pub fn tri_count(&self) -> usize {
        self.tris.len()
    }

    pub fn grid(&self) -> GridSpec {
        self.grid
    }

    pub fn bounding_sphere(&self) -> (Vec3, f32) {
        (self.center, self.radius)
    }

    /// Exact closest surface point and the signed distance to it.
    /// `signed_dist` is negative inside; `|signed_dist| == |p - point|`
    /// exactly, which is what makes physics push-out well-posed.
    pub fn closest(&self, p: Vec3) -> (Vec3, f32) {
        let mut best_d2 = f32::INFINITY;
        let mut best_point = self.center;
        let mut best_tri = 0usize;
        let mut best_feature = Feature::Face;

        // Stack-based nearest-first traversal; 64 depth covers 2^64 tris.
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
                for i in node.start..node.start + node.count {
                    let ti = self.tri_order[i as usize] as usize;
                    let t = self.tris[ti];
                    let (cp, feat) = closest_point_on_triangle(
                        p,
                        self.verts[t[0] as usize],
                        self.verts[t[1] as usize],
                        self.verts[t[2] as usize],
                    );
                    let d2 = (p - cp).length_squared();
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best_point = cp;
                        best_tri = ti;
                        best_feature = feat;
                    }
                }
            } else {
                // Visit the nearer child last so it's popped first.
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

        let pn = match best_feature {
            Feature::Face => self.face_normal[best_tri],
            Feature::Edge(k) => self.edge_pn[best_tri][k],
            Feature::Vert(k) => self.vert_pn[self.tris[best_tri][k] as usize],
        };
        let dist = best_d2.sqrt();
        let signed = if (p - best_point).dot(pn) >= 0.0 { dist } else { -dist };
        (best_point, signed)
    }

    /// Exact signed distance (`closest` without the point).
    pub fn signed_distance(&self, p: Vec3) -> f32 {
        self.closest(p).1
    }

    /// Bake the signed distance onto the grid (`x` fastest, then `y`, then
    /// `z` — matching `texture_3d` texel addressing where `x` is width).
    /// Deterministic serial loop over exact queries: every peer produces
    /// bit-identical bytes from the same mesh (module doc).
    pub fn bake_grid(&self) -> Vec<f32> {
        let g = self.grid;
        let mut out =
            Vec::with_capacity((g.dims[0] * g.dims[1] * g.dims[2]) as usize);
        for k in 0..g.dims[2] {
            for j in 0..g.dims[1] {
                for i in 0..g.dims[0] {
                    let p = g.origin
                        + Vec3::new(i as f32, j as f32, k as f32) * g.cell;
                    out.push(self.signed_distance(p));
                }
            }
        }
        out
    }

    /// Serialized payload: vert count, tri count (u32 LE), verts (f32×3),
    /// indices (u32×3). The bunny asset file (`csg/assets/bunny.mesh`) uses
    /// this exact layout, so the asset loader *is* the decoder.
    pub fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.verts.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.tris.len() as u32).to_le_bytes());
        for v in &self.verts {
            for c in [v.x, v.y, v.z] {
                out.extend_from_slice(&c.to_le_bytes());
            }
        }
        for t in &self.tris {
            for i in t {
                out.extend_from_slice(&i.to_le_bytes());
            }
        }
    }

    /// Decode from `bytes` at `pos`; returns the mesh and the new position.
    /// Rejects hostile counts before allocating (fix-6 discipline) and
    /// everything `new` rejects (open/non-manifold/degenerate meshes).
    pub fn decode_at(bytes: &[u8], pos: usize) -> Option<(TriMeshData, usize)> {
        let u32_at = |p: usize| -> Option<u32> {
            Some(u32::from_le_bytes(bytes.get(p..p + 4)?.try_into().ok()?))
        };
        let nv = u32_at(pos)? as usize;
        let nt = u32_at(pos + 4)? as usize;
        if nv == 0 || nt == 0 || nv > MAX_VERTS || nt > MAX_TRIS {
            return None;
        }
        let need = 8 + nv * 12 + nt * 12;
        if bytes.len() < pos + need {
            return None;
        }
        let mut p = pos + 8;
        let f32_at = |p: usize| f32::from_le_bytes(bytes[p..p + 4].try_into().unwrap());
        let mut verts = Vec::with_capacity(nv);
        for _ in 0..nv {
            verts.push(Vec3::new(f32_at(p), f32_at(p + 4), f32_at(p + 8)));
            p += 12;
        }
        let mut tris = Vec::with_capacity(nt);
        for _ in 0..nt {
            tris.push([u32_at(p)?, u32_at(p + 4)?, u32_at(p + 8)?]);
            p += 12;
        }
        Some((TriMeshData::new(verts, tris)?, p))
    }

    /// Exact signed distance in the `Object::de` convention (`p.w` scale
    /// divisor).
    pub fn de(&self, p: Vec4) -> f32 {
        self.signed_distance(p.truncate()) / p.w
    }

    pub fn content_hash(&self) -> u64 {
        self.content_hash
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Fill `nodes[idx]` for the triangles in `order[lo..hi]`. Inner nodes
/// reserve their two children **contiguously** before recursing (the
/// traversal in [`TriMeshData::closest`] addresses them as `start` and
/// `start + 1`), and the recursion then fills each child in place.
fn build_bvh_into(
    nodes: &mut Vec<BvhNode>,
    idx: usize,
    order: &mut [u32],
    centroids: &[Vec3],
    verts: &[Vec3],
    tris: &[[u32; 3]],
    lo: usize,
    hi: usize,
) {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for &ti in &order[lo..hi] {
        for &vi in &tris[ti as usize] {
            min = min.min(verts[vi as usize]);
            max = max.max(verts[vi as usize]);
        }
    }
    if (hi - lo) as u32 <= LEAF_SIZE {
        nodes[idx] = BvhNode {
            min,
            max,
            start: lo as u32,
            count: (hi - lo) as u32,
        };
        return;
    }
    // Median split on the widest centroid axis. `select_nth_unstable_by`
    // with an index tiebreak gives a fully deterministic partition, so
    // every peer builds the identical tree.
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
        ca.partial_cmp(&cb)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let left = nodes.len();
    let placeholder = || BvhNode {
        min: Vec3::ZERO,
        max: Vec3::ZERO,
        start: 0,
        count: 0,
    };
    nodes.push(placeholder());
    nodes.push(placeholder());
    build_bvh_into(nodes, left, order, centroids, verts, tris, lo, lo + mid);
    build_bvh_into(nodes, left + 1, order, centroids, verts, tris, lo + mid, hi);
    nodes[idx] = BvhNode {
        min,
        max,
        start: left as u32,
        count: 0,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit octahedron: 6 verts, 8 faces, closed, with genuinely sharp
    /// edges and vertices -- exercises all three pseudonormal cases.
    fn octahedron() -> TriMeshData {
        let verts = vec![
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
        ];
        let tris = vec![
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ];
        TriMeshData::new(verts, tris).expect("octahedron is closed")
    }

    #[test]
    fn open_or_inconsistent_meshes_are_rejected() {
        // Single triangle: open.
        assert!(TriMeshData::new(
            vec![Vec3::ZERO, Vec3::X, Vec3::Y],
            vec![[0, 1, 2]]
        )
        .is_none());
        // Octahedron with one face flipped: closed but inconsistently wound.
        let verts = vec![Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y, Vec3::Z, Vec3::NEG_Z];
        let mut tris = vec![
            [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
            [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
        ];
        tris[0] = [2, 0, 4];
        assert!(TriMeshData::new(verts, tris).is_none());
    }

    #[test]
    fn octahedron_signed_distance_is_exact() {
        let m = octahedron();
        // Origin: inside; the plane x+y+z=1 (scaled) is 1/sqrt(3) away.
        let d = m.signed_distance(Vec3::ZERO);
        assert!((d + 1.0 / 3.0f32.sqrt()).abs() < 1e-6, "center d={d}");
        // A vertex: distance 0.
        assert!(m.signed_distance(Vec3::X).abs() < 1e-6);
        // Beyond a vertex along its axis: positive, exact (vertex Voronoi
        // region -- the naive face-normal sign has no face to agree with
        // here; the vertex pseudonormal handles it).
        let d = m.signed_distance(Vec3::new(2.0, 0.0, 0.0));
        assert!((d - 1.0).abs() < 1e-6, "outside vertex d={d}");
        // Outside above an edge midpoint (edge Voronoi region).
        let edge_mid = Vec3::new(0.5, 0.5, 0.0);
        let out = edge_mid * 3.0;
        let d = m.signed_distance(out);
        assert!((d - (out - edge_mid).length()).abs() < 1e-5, "edge d={d}");
        // Just inside a face center: negative.
        let fc = Vec3::splat(1.0 / 3.0);
        let d = m.signed_distance(fc * 0.9);
        assert!(d < 0.0, "inside face d={d}");
    }

    #[test]
    fn closest_point_matches_distance_everywhere() {
        let m = octahedron();
        // |p - closest| must equal |signed| *exactly* -- both come from the
        // same query; this is the property collision push-out relies on.
        let probes = [
            Vec3::new(0.3, 0.2, 0.1),
            Vec3::new(-1.5, 0.4, 0.2),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.9, 0.9, 0.9),
            Vec3::new(-0.1, 0.05, 0.02),
        ];
        for p in probes {
            let (cp, sd) = m.closest(p);
            assert!(
                ((p - cp).length() - sd.abs()).abs() < 1e-6,
                "probe {p:?}: |p-cp|={} sd={sd}",
                (p - cp).length()
            );
        }
    }

    #[test]
    fn bvh_agrees_with_brute_force() {
        let m = octahedron();
        for i in 0..64 {
            // Deterministic pseudo-random probes.
            let f = i as f32;
            let p = Vec3::new(
                (f * 0.617).sin() * 1.8,
                (f * 0.371 + 1.0).sin() * 1.8,
                (f * 0.219 + 2.0).sin() * 1.8,
            );
            let brute = m
                .tris
                .iter()
                .map(|t| {
                    let (cp, _) = closest_point_on_triangle(
                        p,
                        m.verts[t[0] as usize],
                        m.verts[t[1] as usize],
                        m.verts[t[2] as usize],
                    );
                    (p - cp).length()
                })
                .fold(f32::INFINITY, f32::min);
            assert!(
                (m.signed_distance(p).abs() - brute).abs() < 1e-5,
                "probe {p:?}"
            );
        }
    }

    #[test]
    fn grid_bake_brackets_the_exact_field() {
        let m = octahedron();
        let g = m.grid();
        let grid = m.bake_grid();
        assert_eq!(grid.len(), (g.dims[0] * g.dims[1] * g.dims[2]) as usize);
        // Every stored sample IS an exact query by construction; spot-check
        // the trilinear reconstruction the shader performs against the
        // exact field at off-lattice points: it must agree to within the
        // interpolation error bound for a 1-Lipschitz field (~cell size).
        for i in 0..32 {
            let f = i as f32;
            let q = Vec3::new(
                (f * 0.532).sin() * 0.9,
                (f * 0.719 + 3.0).sin() * 0.9,
                (f * 0.351 + 5.0).sin() * 0.9,
            );
            let gc = ((q - g.origin) / g.cell)
                .clamp(Vec3::ZERO, Vec3::new(
                    (g.dims[0] - 1) as f32 - 1e-3,
                    (g.dims[1] - 1) as f32 - 1e-3,
                    (g.dims[2] - 1) as f32 - 1e-3,
                ));
            let i0 = gc.floor();
            let fr = gc - i0;
            let at = |dx: u32, dy: u32, dz: u32| {
                let (x, y, z) = (i0.x as u32 + dx, i0.y as u32 + dy, i0.z as u32 + dz);
                grid[(z * g.dims[1] * g.dims[0] + y * g.dims[0] + x) as usize]
            };
            let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
            let c00 = lerp(at(0, 0, 0), at(1, 0, 0), fr.x);
            let c10 = lerp(at(0, 1, 0), at(1, 1, 0), fr.x);
            let c01 = lerp(at(0, 0, 1), at(1, 0, 1), fr.x);
            let c11 = lerp(at(0, 1, 1), at(1, 1, 1), fr.x);
            let tri = lerp(lerp(c00, c10, fr.y), lerp(c01, c11, fr.y), fr.z);
            let exact = m.signed_distance(q);
            assert!(
                (tri - exact).abs() <= g.cell * 0.9,
                "q {q:?}: trilinear {tri} vs exact {exact} (cell {})",
                g.cell
            );
        }
    }

    #[test]
    fn encode_decode_round_trips() {
        let m = octahedron();
        let mut bytes = vec![0xAB]; // offset decode
        m.encode(&mut bytes);
        let (decoded, end) = TriMeshData::decode_at(&bytes, 1).expect("decode");
        assert_eq!(end, bytes.len());
        assert_eq!(decoded.content_hash(), m.content_hash());
        // Same field, not just same bytes.
        for p in [Vec3::new(0.4, 0.1, -0.3), Vec3::new(1.5, 1.5, 0.0)] {
            assert_eq!(decoded.signed_distance(p), m.signed_distance(p));
        }
    }

    #[test]
    fn decode_rejects_hostile_counts() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(TriMeshData::decode_at(&bytes, 0).is_none());
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&10u32.to_le_bytes());
        bytes.extend_from_slice(&10u32.to_le_bytes());
        // counts claim data the buffer doesn't have
        assert!(TriMeshData::decode_at(&bytes, 0).is_none());
    }
}
