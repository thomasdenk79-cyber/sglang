#!/usr/bin/env bash
# Build the opt-in "next" runtime venv (LAB_STACK=next) side by side with the pinned venv.
# Never modifies the base venv; reads its package set and swaps only the torch/kernel stack.
#   tools/build_venv_next.sh [DEST]   (default /tmp/opencode/venv-next)
# Status 2026-11-24: builds (torch 2.14.1+cu130, triton 3.8.0, sglang-kernel 0.4.11), but
# sglang-kernel 0.4.11 drops symbols the donor imports (verify_tree_greedy,
# tree_speculative_sampling_target_only, segment_packbits, sgl_kernel.flash_attn/flash_mla, ...),
# and 0.4.6.post1 fails to load on torch 2.14 (c10 ABI). Not servable until those are ported.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-${LAB_VENV_NEXT:-/tmp/opencode/venv-next}}"
BASE_VENV="${BASE_VENV:-/srv/community-repos/sglang-rtxpro6000/.venv}"

# Stack pins. sglang-kernel and sgl-deep-ep wheels are built against an exact torch
# (requires_dist torch==X); keep these three in lockstep or the C++ op ABI breaks.
TORCH_VER="${TORCH_VER:-2.14.1}"
TORCH_CU="${TORCH_CU:-cu130}"
SGL_KERNEL_VER="${SGL_KERNEL_VER:-0.4.11}"
SGL_DEEP_EP_VER="${SGL_DEEP_EP_VER:-0.1.3}"
SGL_DEEP_GEMM_VER="${SGL_DEEP_GEMM_VER:-0.1.5.post3}"
FLASHINFER_VER="${FLASHINFER_VER:-0.6.17}"

command -v uv >/dev/null || { echo "uv required" >&2; exit 1; }
[ -x "$BASE_VENV/bin/python" ] || { echo "base venv missing: $BASE_VENV" >&2; exit 1; }
[ -e "$DEST" ] && { echo "$DEST exists; remove it first" >&2; exit 1; }

work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
# Everything except the stack packages and the editable sglang checkout.
"$BASE_VENV/bin/python" -m pip freeze --exclude-editable \
  | grep -viE '^(torch|torchaudio|torchvision|torchcodec|triton|sglang|sglang-kernel|sgl-deep-ep|sgl-deep-gemm|flashinfer-[a-z-]+|nvidia-[a-z0-9-]+|cuda-toolkit|tokenspeed-triton|torch_c_dlpack_ext)==' \
  > "$work/rest.txt"

uv venv --python 3.12 "$DEST"
PY="$DEST/bin/python"
IDX="https://download.pytorch.org/whl/$TORCH_CU"

uv pip install --python "$PY" --index-url "$IDX" --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match \
  "torch==$TORCH_VER+$TORCH_CU" torchvision torchaudio
uv pip install --python "$PY" --extra-index-url "$IDX" --index-strategy unsafe-best-match \
  "torch==$TORCH_VER+$TORCH_CU" \
  "sglang-kernel==$SGL_KERNEL_VER" "sgl-deep-ep==$SGL_DEEP_EP_VER" \
  "sgl-deep-gemm==$SGL_DEEP_GEMM_VER" "flashinfer-python[cu13]==$FLASHINFER_VER"
# Remaining deps at the base venv's versions; never let them move torch.
echo "torch==$TORCH_VER+$TORCH_CU" > "$work/constraints.txt"
uv pip install --python "$PY" --extra-index-url "$IDX" --index-strategy unsafe-best-match \
  -c "$work/constraints.txt" -r "$work/rest.txt" || \
  echo "[build_venv_next] some base pins failed to resolve; see above" >&2
# Donor sglang as editable without letting its torch==2.13 pin win.
uv pip install --python "$PY" --no-deps -e "$REPO/python"

"$PY" - <<'EOF'
import importlib.metadata as m
for p in ("torch", "triton", "sglang-kernel", "sgl-deep-ep", "sgl-deep-gemm", "flashinfer-python"):
    try: print(f"{p:18} {m.version(p)}")
    except m.PackageNotFoundError: print(f"{p:18} MISSING")
EOF
