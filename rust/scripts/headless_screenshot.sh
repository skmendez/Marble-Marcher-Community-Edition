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
# llvmpipe segfaults executing these generated march shaders even with the
# MM_MRRM=0 mitigation below (reproduced: JIT-code crash on an llvmpipe-0
# worker thread), while noble GA's 24.0.5 (LLVM 17) runs them fine --
# matching the Mesa era this project's original llvmpipe verification used
# (MILESTONES.md M4/M6). SwiftShader's Vulkan ICD is not an alternative:
# naga-generated SPIR-V for these shaders crashes its Subzero JIT too, and
# it also lacks VK_KHR_xlib_surface for presenting under Xvfb.
#
# Flags:
#  - MM_MRRM=0: llvmpipe's shader JIT reproducibly segfaults when a
#    coarse-texture fetch feeds a march loop's starting `t` (see
#    `marble_csg::codegen::COARSE_TEXTURE_BINDING`'s doc). The flag skips
#    that data flow at runtime in both consumers (fine pass, and the shadow
#    pass via the same `misc.w` lane). Real GPUs don't need this.
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
MM_MRRM=0 \
WGPU_BACKEND=vulkan \
xvfb-run -a target/debug/marble-marcher-bevy

echo "screenshot written to $OUT"
