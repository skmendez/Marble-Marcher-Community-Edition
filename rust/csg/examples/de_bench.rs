//! Micro-benchmark for the CPU-side CSG queries: how expensive is one
//! `Object::de` / `Object::nearest_point` evaluation, and therefore how many
//! sphere-trace steps a per-frame camera occlusion/clearance query can
//! afford? Sizes the performance budget in `rust/CAMERA.md` §6 (the smart
//! camera design's whole approach rests on CPU sphere tracing being cheap
//! relative to a GPU-bound frame, which is a claim worth being able to
//! re-measure rather than assert).
//!
//! `cargo run --release -p marble-csg --example de_bench` -- release matters:
//! `marble-csg` is `opt-level = 3` in the release profile (workspace
//! `Cargo.toml`), which is the configuration a shipped build actually runs
//! these queries under. Reports both realistic traces (which terminate
//! early) and a fixed-step worst case (the full step budget spent).

use std::time::Instant;

use glam::Vec3;
use marble_csg::scenes::{
    beware_of_bumps, demo_scene, menger_sphere, set_fractal_params, set_menger_params,
};
use marble_csg::{Object, Params};

fn bench_de(name: &str, object: &Object, params: &Params, probe: Vec3) {
    // warmup
    let mut acc = 0.0f32;
    for i in 0..10_000 {
        let p = probe + Vec3::splat(i as f32 * 1e-5);
        acc += object.de(p.extend(1.0), params);
    }
    let n = 200_000;
    let start = Instant::now();
    for i in 0..n {
        let p = probe + Vec3::splat(i as f32 * 1e-5);
        acc += object.de(p.extend(1.0), params);
    }
    let el = start.elapsed();
    println!(
        "{name}: {:.1} ns/de  ({} evals in {:?})  [acc={acc}]",
        el.as_nanos() as f64 / n as f64,
        n,
        el
    );
}

/// One sphere trace from `ro` toward `target`, capped at `max_steps`.
/// Returns (hit?, steps, min_clearance_ratio).
fn sphere_trace(
    object: &Object,
    params: &Params,
    ro: Vec3,
    target: Vec3,
    max_steps: u32,
    eps: f32,
) -> (bool, u32, f32) {
    let seg = target - ro;
    let max_t = seg.length();
    let rd = seg / max_t;
    let mut t = eps;
    let mut res = 1.0f32;
    for i in 0..max_steps {
        let h = object.de((ro + rd * t).extend(1.0), params);
        if h < eps {
            return (true, i + 1, 0.0);
        }
        res = res.min(8.0 * h / t);
        t += h.max(eps);
        if t >= max_t {
            return (false, i + 1, res);
        }
    }
    (false, max_steps, res)
}

fn bench_trace(name: &str, object: &Object, params: &Params, eye: Vec3, target: Vec3, eps: f32) {
    let (hit, steps, res) = sphere_trace(object, params, eye, target, 64, eps);
    let n = 20_000;
    let start = Instant::now();
    let mut total_steps = 0u64;
    for i in 0..n {
        let jitter = Vec3::splat(i as f32 * 1e-6);
        let (_, s, _) = sphere_trace(object, params, eye + jitter, target, 64, eps);
        total_steps += s as u64;
    }
    let el = start.elapsed();
    println!(
        "{name} trace: hit={hit} steps={steps} clearance={res:.3} | {:.2} us/trace (avg {:.1} steps)",
        el.as_micros() as f64 / n as f64,
        total_steps as f64 / n as f64
    );
}

fn bench_np(name: &str, object: &Object, params: &Params, probe: Vec3) {
    let mut hist = Vec::new();
    let mut acc = Vec3::ZERO;
    for i in 0..10_000 {
        let p = probe + Vec3::splat(i as f32 * 1e-5);
        acc += object.nearest_point_scratch(p.extend(1.0), params, &mut hist);
    }
    let n = 100_000;
    let start = Instant::now();
    for i in 0..n {
        let p = probe + Vec3::splat(i as f32 * 1e-5);
        acc += object.nearest_point_scratch(p.extend(1.0), params, &mut hist);
    }
    let el = start.elapsed();
    println!(
        "{name}: {:.1} ns/nearest_point ({} evals) [acc={acc}]",
        el.as_nanos() as f64 / n as f64,
        n
    );
}

/// Worst case: a march that always spends its whole step budget.
fn bench_fixed_steps(name: &str, object: &Object, params: &Params, ro: Vec3, steps: u32) {
    let n = 5_000;
    let mut acc = 0.0f32;
    let start = Instant::now();
    for i in 0..n {
        let mut t = 0.01f32;
        let rd = Vec3::new(0.3, 0.9, 0.31).normalize();
        for _ in 0..steps {
            let h = object.de((ro + rd * t + Vec3::splat(i as f32 * 1e-6)).extend(1.0), params);
            acc += h;
            t += 0.001; // fixed tiny step: guarantees the full budget is spent
        }
    }
    let el = start.elapsed();
    println!(
        "{name}: {:.1} us per {steps}-step march (worst case) [acc={acc}]",
        el.as_micros() as f64 / n as f64
    );
}

fn main() {
    // Demo scene ("Beware Of Bumps"), iters = 16.
    let mut params = Params::new();
    let (demo, handles) = demo_scene(&mut params);
    set_fractal_params(
        &mut params,
        &handles,
        beware_of_bumps::SCALE,
        beware_of_bumps::ANG1,
        beware_of_bumps::ANG2,
        beware_of_bumps::SHIFT,
        beware_of_bumps::COLOR,
        beware_of_bumps::ITERS,
    );
    let start = beware_of_bumps::START;
    bench_de("demo(iters=16)", &demo, &params, start);
    // Camera sits 0.2 back along the default view direction.
    let fwd = Vec3::new(-1.448f32.sin() * 0.899f32.cos(), 0.899f32.sin(), -1.448f32.cos() * 0.899f32.cos());
    let eye = start - fwd * 0.2;
    bench_trace("demo", &demo, &params, eye, start, beware_of_bumps::MARBLE_RAD * 0.5);
    bench_np("demo(iters=16)", &demo, &params, start);
    bench_fixed_steps("demo(iters=16)", &demo, &params, start, 32);
    bench_fixed_steps("demo(iters=16)", &demo, &params, start, 64);

    // Menger sphere at the app's MENGER_DEPTH = 5.
    let mut mparams = Params::new();
    let (menger, mhandles) = menger_sphere(&mut mparams);
    set_menger_params(&mut mparams, &mhandles, 5, Vec3::new(0.6, 0.6, 0.6));
    let mstart = Vec3::new(3.32, 1.69, 3.22);
    bench_de("menger_sphere(depth=5)", &menger, &mparams, mstart);
    let meye = mstart + Vec3::new(1.0, 0.5, 1.0).normalize() * 1.2;
    bench_trace("menger_sphere", &menger, &mparams, meye, mstart, 0.05);
    bench_np("menger_sphere(depth=5)", &menger, &mparams, mstart);
    bench_fixed_steps("menger_sphere(depth=5)", &menger, &mparams, mstart, 32);
}
