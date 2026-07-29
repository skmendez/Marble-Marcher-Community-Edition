#!/usr/bin/env bash
# Render the game headlessly (no GPU, no display) and capture a screenshot.
#
# Designed for CI / cloud containers: Xvfb as the display, Mesa's lavapipe
# (llvmpipe) software Vulkan driver as the GPU. Verified in a stock Ubuntu
# 24.04 container:
#
#   apt-get update
#   apt-get install -y xvfb mesa-vulkan-drivers=24.0.5-1ubuntu1
#   rust/scripts/headless_screenshot.sh /tmp/shot.png
#
# The Mesa version pin matters: noble-updates' Mesa 25.2.8 (LLVM 20)
# llvmpipe corrupts memory executing these generated shaders, while noble
# GA's 24.0.5 (LLVM 17) runs them fine -- matching the Mesa era this
# project's original llvmpipe verification used (MILESTONES.md M4/M6).
#
# The trigger is specifically `de_scene`'s *dynamically bounded* fold loop
# (`Fold::Repeat`'s trip count is a runtime uniform -- the scene's iteration
# count param). Established by bisection on 25.2.8:
#   - Only fractal scenes crash. cube_sphere_morph and hollow_donut render
#     fine; menger_*, demo and classic_only segfault. Both groups still run
#     `march_scene`'s 256-iteration loop, so it is not loops in general.
#   - Emitting a compile-time-constant trip count instead of the uniform-
#     derived one makes every crashing scene render fine.
#   - Symptom is SIGSEGV on an llvmpipe-0 worker thread with JIT'd frames;
#     at LP_NATIVE_VECTOR_WIDTH=128 it becomes `malloc(): invalid next size`
#     instead, i.e. heap corruption rather than a clean fault.
#   - Not worked around by LP_DEBUG=nopt, GALLIVM_DEBUG=nopt, LP_NUM_THREADS=0
#     or LP_MAX_SHADER_VARIANTS=1. vkcube runs fine on the same driver, so
#     lavapipe is not broken generally.
# The shader itself is valid: naga validates it, Mesa 24/LLVM 17 runs it, and
# real GPU / browser WebGPU backends run it in production. This looks like an
# upstream llvmpipe+LLVM 20 codegen bug and is worth reporting as one.
#
# SwiftShader's Vulkan ICD is not an alternative: naga-generated SPIR-V for
# these shaders crashes its Subzero JIT too, and it also lacks
# VK_KHR_xlib_surface for presenting under Xvfb.
#
# NOTE: this script used to force MM_MRRM=0, on the belief that llvmpipe
# segfaulted when a coarse-texture fetch fed the march loop's starting `t`.
# That is not reproducible: on the pinned 24.0.5 all scenes render correctly
# with MRRM *on*, and on 25.2.8 they crash identically with MRRM on or off.
# MRRM is not implicated either way, so the flag is gone and the headless
# path now exercises the real (MRRM-enabled) shipping configuration.
#
# Flags:
#  - BEVY_ASSET_ROOT: running the binary directly (not `cargo run`) makes
#    Bevy fall back to resolving `assets/` next to the *executable*
#    (target/debug/assets, which doesn't exist), silently breaking every
#    asset load -- including the marble cubemap. Point it at the app crate.
#  - MM_WINDOW_SIZE: small window -- this per-pixel ray marcher is far
#    slower on a CPU rasterizer than on GPU hardware (480x360 still reached
#    ~28 FPS on llvmpipe in an 8-core container, so this is comfortable).
#  - MM_SCREENSHOT_DELAY_SECS: llvmpipe compiles the generated shaders at
#    first use; capturing too early would show only the clear color (see
#    `debug_screenshot.rs`'s module doc).
#
# Usage: headless_screenshot.sh [output.png] [WxH] [delay_secs]
set -euo pipefail

OUT="${1:-/tmp/mm_screenshot.png}"
SIZE="${2:-480x360}"
DELAY="${3:-60}"

# Mesa's ICD manifest is lvp_icd.x86_64.json in the 24.x packages and
# lvp_icd.json in 25.x -- pick whichever exists.
ICD=""
for f in /usr/share/vulkan/icd.d/lvp_icd.x86_64.json /usr/share/vulkan/icd.d/lvp_icd.json; do
  [ -f "$f" ] && ICD="$f" && break
done
if [ -z "$ICD" ]; then
  echo "error: no lavapipe Vulkan ICD found -- install mesa-vulkan-drivers" >&2
  echo "  (pin 24.0.5-1ubuntu1 on Ubuntu 24.04; 25.x llvmpipe crashes, see header)" >&2
  exit 1
fi

cd "$(dirname "$0")/.."
cargo build -p marble-marcher-bevy

VK_ICD_FILENAMES="$ICD" \
BEVY_ASSET_ROOT="app" \
MM_SCREENSHOT="$OUT" \
MM_SCREENSHOT_DELAY_SECS="$DELAY" \
MM_WINDOW_SIZE="$SIZE" \
WGPU_BACKEND=vulkan \
xvfb-run -a target/debug/marble-marcher-bevy

echo "screenshot written to $OUT"
