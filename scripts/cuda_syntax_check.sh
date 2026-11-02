#!/usr/bin/env bash
# Parse the rasterizer's CUDA sources with clang (host side only, stub CUDA headers from
# scripts/cuda_stub/) to catch syntax and type errors in new code before a GPU build.
# Pre-existing errors remain: glm functions not marked __device__ for clang, and the `<< <` launch
# spacing that nvcc accepts. Compare counts against the previous commit, or read new lines only.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
R="$ROOT/submodules/diff-t-rasterization"
for f in "${@:-forward backward rasterizer_impl}"; do
  for g in $f; do
    out=$(clang++ -x cuda --cuda-host-only -nocudainc -nocudalib --cuda-gpu-arch=sm_75 -std=c++17 -fsyntax-only \
      -ferror-limit=0 -include "$ROOT/scripts/cuda_stub/cuda_runtime.h" -I"$ROOT/scripts/cuda_stub" \
      -I"$R/third_party/glm" -I"$R/cuda_rasterizer" "$R/cuda_rasterizer/$g.cu" 2>&1)
    echo "$g.cu: $(echo "$out" | grep -c 'error:') errors"
    echo "$out" | grep "error:" | grep -v -E "expected expression|glm|transpose|length'|operator\[\]|binary expression|overloaded '\*='|auxiliary.h" | sed 's#.*/cuda_rasterizer/##'
  done
done
