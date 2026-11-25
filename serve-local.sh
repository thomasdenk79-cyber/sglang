#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors
set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
PRINT_ONLY="${FLASH_NEXT_PRINT_ONLY:-0}"
unset FLASH_NEXT_PRINT_ONLY
ARGS=()
for arg in "$@"; do
  if [ "$arg" = --print ]; then PRINT_ONLY=1; else ARGS+=("$arg"); fi
done
set -- "${ARGS[@]}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST=12.0
CACHE_BASE="${XDG_CACHE_HOME:-$HOME/.cache}/flash-next-fast/${FLASH_NEXT_EDITION:-normal}"
export HF_HOME="${HF_HOME:-$CACHE_BASE/hf}" XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_BASE/xdg}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_BASE/triton}" SGLANG_JIT_CACHE_DIR="${SGLANG_JIT_CACHE_DIR:-$CACHE_BASE/jit}"
: "${TARGET_MODEL:?Set TARGET_MODEL to the model directory}"
MEM_FRACTION="${MEM_FRACTION:-0.96}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
PORT="${PORT:-8001}"
MAMBA_SLOTS="${MAMBA_SLOTS:-24}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"

EXTRA_ARGS=()
if [ -n "$MAX_TOTAL_TOKENS" ]; then
  EXTRA_ARGS+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
fi
if [ "$CONTEXT_LENGTH" -gt 262144 ]; then
  export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
  EXTRA_ARGS+=(--json-model-override-args '{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":2.0,"original_max_position_embeddings":262144}}}')
fi

# Opt-in startup speedups (all off by default; each is runtime-neutral).
[ "${LAB_FAST_AUTOTUNE_REPLAY:-0}" = 1 ] && export SGLANG_STARTUP_AUTOTUNE_REPLAY=1
[ "${LAB_FAST_DRAFT_SHARDS:-0}" = 1 ] && export SGLANG_STARTUP_WEIGHT_SHARD_FILTER=1
[ "${LAB_FAST_EMBED_TABLE_CACHE:-0}" = 1 ] && export SGLANG_STARTUP_MTP_EMBED_TABLE_CACHE=1
[ "${LAB_FAST_DROP_CACHE:-0}" = 1 ] && EXTRA_ARGS+=(--weight-loader-drop-cache-after-load)
[ "${LAB_FAST_SKIP_WARMUP:-0}" = 1 ] && EXTRA_ARGS+=(--skip-server-warmup)
case "${LAB_FAST_LOAD_FORMAT:-}" in
  ""|safetensors) ;;
  fastsafetensors) EXTRA_ARGS+=(--load-format fastsafetensors) ;;
  *) echo "[serve-local] LAB_FAST_LOAD_FORMAT must be empty, safetensors or fastsafetensors" >&2; exit 2 ;;
esac
if [ -n "${LAB_FAST_LOAD_THREADS:-}" ]; then
  [[ "$LAB_FAST_LOAD_THREADS" =~ ^[1-9][0-9]?$ ]] || { echo "[serve-local] LAB_FAST_LOAD_THREADS must be 1..99" >&2; exit 2; }
  EXTRA_ARGS+=(--model-loader-extra-config "{\"enable_multithread_load\":true,\"num_threads\":$LAB_FAST_LOAD_THREADS}")
fi
READAHEAD_GB="${LAB_FAST_READAHEAD_GB:-0}"
[[ "$READAHEAD_GB" =~ ^[0-9]+$ ]] || { echo "[serve-local] LAB_FAST_READAHEAD_GB must be a nonnegative integer" >&2; exit 2; }

if [ "${FLASH_NEXT_ADMIN_API_KEY+x}" = x ]; then
  [ -n "$FLASH_NEXT_ADMIN_API_KEY" ] || { echo "[serve-local] FLASH_NEXT_ADMIN_API_KEY must not be empty" >&2; exit 2; }
  ADMIN_API_KEY="$FLASH_NEXT_ADMIN_API_KEY"
else
  ADMIN_API_KEY="$("${PYTHON:-python3}" -I -B -c 'import secrets; print(secrets.token_hex(32))')"
fi
unset FLASH_NEXT_ADMIN_API_KEY
export -n ADMIN_API_KEY

CMD=("${PYTHON:-python3}" -P -B "$REPO/server.py" serve \
  --model-path "$TARGET_MODEL" \
  --load-format safetensors \
  --served-model-name flash-next \
  --host 127.0.0.1 --port "$PORT" --tp 1 \
  --dtype bfloat16 --quantization modelopt_fp4 --kv-cache-dtype "${LAB_KV_DTYPE:-fp8_e4m3}" ${LAB_DRAFT_KV_DTYPE:+--speculative-draft-kv-cache-dtype "$LAB_DRAFT_KV_DTYPE"} \
  --mem-fraction-static "$MEM_FRACTION" \
  --context-length "$CONTEXT_LENGTH" \
  --page-size 64 --max-running-requests 4 --sleep-on-idle \
  --chunked-prefill-size 8192 \
  --mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype bfloat16 \
  --max-mamba-cache-size "$MAMBA_SLOTS" --gdn-mtp-cache-mode none \
  --linear-attn-decode-backend flashinfer --linear-attn-prefill-backend flashinfer \
  --ple-offload-embedding \
  --chat-template "$TARGET_MODEL/chat_template.jinja" \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"medium"}' \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --speculative-draft-model-quantization unquant --watchdog-timeout 1800 \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  "$@" --admin-api-key="$ADMIN_API_KEY")

if [ "$PRINT_ONLY" = 1 ]; then
  exec "${PYTHON:-python3}" -I -B -c '
import json, os, re, sys
allowed = {
    "ADAPTIVE_CONFIG", "CC", "CONC", "CONTEXT_LENGTH", "CPATH", "CUDACXX",
    "CUDAHOSTCXX", "CXX", "DISPLAY_RESERVE_GB", "GPU_MEMORY_GB", "HF_HOME",
    "HF_HUB_CACHE", "HF_HUB_OFFLINE", "HUGGINGFACE_HUB_CACHE", "LANG", "LC_ALL",
    "LIBRARY_PATH", "MAMBA_SLOTS", "MAX_TOTAL_TOKENS", "MEM_FRACTION",
    "NUMBA_CACHE_DIR", "PATH", "PORT", "PYTHON", "PYTHONDONTWRITEBYTECODE",
    "PYTHONNOUSERSITE", "PYTHONPATH", "PYTORCH_CUDA_ALLOC_CONF", "TARGET_MODEL",
    "TEMP", "TMP", "TMPDIR", "TRANSFORMERS_CACHE", "TRANSFORMERS_OFFLINE",
    "WA_MEM_FRACTION",
}
prefixes = ("SGLANG_", "FLASHINFER_", "FLASH_NEXT_", "CUDA_", "TORCH", "TRITON_", "XDG_")
sensitive = re.compile("TOKEN|KEY|SECRET|PASS|AUTH|CRED", re.IGNORECASE)
env = {key: "<redacted>" if sensitive.search(key) else value
       for key, value in os.environ.items()
       if key in allowed or key.startswith(prefixes)}
argv = sys.argv[1:]
_NONSECRET_TOKEN_OPTIONS = frozenset(
    {"bucket-time-to-first-token", "kt-max-deferred-experts-per-token"}
)
sensitive_suffixes = (
    "key", "token", "secret", "password", "passwd", "auth", "cred", "creds",
    "credential", "credentials",
)
# Keep these parser-derived lists in sync via test/registered/unit/test_argv_redaction.py.
sensitive_options = {
    "admin-api-key", "api-key", "ssl-keyfile-password",
}
nonsensitive_prefixes = {"ssl-keyfile"}
redact_next = False
for index, arg in enumerate(argv):
    name, separator, value = arg.partition("=")
    normalized_name = name.lstrip("-").lower().replace("_", "-")
    checked_name = bool(normalized_name) and normalized_name not in nonsensitive_prefixes
    # A real secret option, by exact name or abbreviation.
    names_secret_option = checked_name and arg.startswith("--") and (
        any(option.startswith(normalized_name) for option in sensitive_options)
        # Protect HF tokens even though this parser lacks --hf-token.
        or "hf-token".startswith(normalized_name)
    )
    is_sensitive = arg.startswith("-") and checked_name and (
        names_secret_option
        or (
            normalized_name not in _NONSECRET_TOKEN_OPTIONS
            and normalized_name.endswith(sensitive_suffixes)
        )
    )
    if redact_next and not names_secret_option:
        # A pending value is masked whatever it starts with. After a bare "--" or
        # a credential-looking token, the next token is masked too.
        argv[index] = "<redacted>"
        redact_next = arg == "--" or (is_sensitive and not separator)
        continue
    redact_next = False
    if is_sensitive:
        if separator:
            argv[index] = name + "=<redacted>"
        else:
            redact_next = True
print(json.dumps({"argv": argv, "env": env}, sort_keys=True))
' "${CMD[@]}"
fi
if [ "$READAHEAD_GB" != 0 ]; then
  # Warm the page cache with the first shards (loader order) while Python
  # imports and builds the model (~35 s). Capped below MemAvailable - 8 GiB so
  # it never pushes pinned PLE/HiCache allocations into reclaim or swap.
  avail_gb=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo)
  cap_gb=$(( avail_gb > 8 ? avail_gb - 8 : 0 ))
  [ "$READAHEAD_GB" -gt "$cap_gb" ] && READAHEAD_GB="$cap_gb"
  echo "[serve-local] readahead: up to ${READAHEAD_GB} GiB of $TARGET_MODEL shards" >&2
  if [ "$READAHEAD_GB" -gt 0 ]; then
    setsid nohup nice -n 10 bash -c '
      budget=$(( $1 * 1073741824 ))
      for f in $(ls "$2"/*.safetensors | LC_ALL=C sort); do
        [ "$budget" -gt 0 ] || break
        cat "$f" > /dev/null; budget=$(( budget - $(stat -c %s "$f") ))
      done' _ "$READAHEAD_GB" "$TARGET_MODEL" > /dev/null 2>&1 < /dev/null &
  fi
fi
cd "$REPO"
exec "${CMD[@]}"
