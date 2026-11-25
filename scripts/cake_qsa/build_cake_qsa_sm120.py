#!/usr/bin/env python3
"""Build FlashInfer's experimental Cake QSA NVFP4 decode as a standalone package.

The flashinfer 0.7.4 wheel ships `flashinfer/experimental/cake_qsa_nvfp4` without
`csrc/` and `aot/`. This script copies the package from a v0.7.4 source checkout,
checks every source against `build.json` sha256, compiles the requested arch
libraries with nvcc into `aot/` and writes `aot/build_info.json`. The result is a
top-level `cake_qsa_nvfp4` package that imports only torch and tvm_ffi, so it
loads side by side with the donor's patched FlashInfer 0.6.x trees.

Usage:
  python scripts/cake_qsa/build_cake_qsa_sm120.py \
      --src /tmp/opencode/flashinfer-v074 --out /tmp/opencode/cake-qsa-sm120
  export SGLANG_QSA_CAKE_PATH=/tmp/opencode/cake-qsa-sm120
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cuda_version(nvcc: str) -> tuple[int, int]:
    text = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
    m = re.search(r"release (\d+)\.(\d+)", text)
    if not m:
        raise RuntimeError(f"cannot parse nvcc version: {text}")
    return int(m.group(1)), int(m.group(2))


def flag_applies(cond: str, cuda: tuple[int, int]) -> bool:
    op, major, minor = re.fullmatch(r"(==|>=)(\d+)\.(\d+)", cond).groups()
    want = (int(major), int(minor))
    return cuda == want if op == "==" else cuda >= want


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="flashinfer v0.7.4 source checkout")
    ap.add_argument("--out", required=True, help="directory that will hold cake_qsa_nvfp4/")
    ap.add_argument("--arch", default="sm_120a")
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("MAX_JOBS", "2")))
    args = ap.parse_args()

    import tvm_ffi.libinfo

    cuda_home = Path(os.environ.get("CUDA_HOME", "/opt/cuda"))
    nvcc = os.environ.get("CUDACXX", str(cuda_home / "bin" / "nvcc"))
    cxx = os.environ.get("CXX", "g++")
    src_pkg = Path(args.src) / "flashinfer" / "experimental" / "cake_qsa_nvfp4"
    pkg = Path(args.out) / "cake_qsa_nvfp4"
    if pkg.exists():
        shutil.rmtree(pkg)
    shutil.copytree(src_pkg, pkg, ignore=shutil.ignore_patterns("__pycache__", "aot"))
    (pkg / "aot").mkdir()

    manifest = json.loads((pkg / "build.json").read_text())
    common = manifest["common"]
    cuda = cuda_version(nvcc)
    libs = [lib for lib in manifest["libraries"] if lib["arch"] == args.arch]
    if not libs:
        raise SystemExit(f"no {args.arch} libraries in build.json")

    checked = 0
    for lib in libs:
        for rel, digest in lib["sha256"].items():
            got = sha256(pkg / rel)
            if got != digest:
                raise SystemExit(f"sha256 mismatch {rel}: {got} != {digest}")
            checked += 1
    print(f"sha256 ok: {checked}/{checked} sources of {len(libs)} {args.arch} libraries")

    includes = [str(pkg / d) for d in common["include_dirs"]] + [
        tvm_ffi.libinfo.find_include_path(),
        tvm_ffi.libinfo.find_dlpack_include_path(),
        str(cuda_home / "include"),
    ]

    def flags(lib: dict) -> list[str]:
        out = list(common["nvcc_flags"]) + list(lib["nvcc_flags"])
        for item in list(common.get("nvcc_flags_if", [])) + list(lib["nvcc_flags_if"]):
            if flag_applies(item["cuda"], cuda):
                out += item["flags"]
        return out + manifest["arch_flags"][args.arch] + [f"-I{i}" for i in includes]

    def build(lib: dict) -> str:
        objs = []
        for rel in lib["sha256"]:
            if not rel.endswith(".cu"):
                continue
            obj = pkg / "aot" / (Path(rel).stem + ".o")
            cmd = [nvcc, "-ccbin", cxx, "-c", str(pkg / rel), "-o", str(obj)] + flags(lib)
            subprocess.run(cmd, check=True)
            objs.append(str(obj))
        so = pkg / lib["path"]
        link = [cxx] + objs + ["-o", str(so), f"-L{cuda_home / 'lib64'}"] + common["link_flags"]
        subprocess.run(link, check=True)
        for o in objs:
            os.unlink(o)
        return lib["name"]

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for name in pool.map(build, libs):
            print(f"built {name}")

    cudart = cuda[0] * 1000 + cuda[1] * 10  # CUDART_VERSION of the toolkit headers
    (pkg / "aot" / "build_info.json").write_text(
        json.dumps(
            {
                "schema": "cake-build-info/1",
                "arches": [args.arch],
                "cuda": f"{cuda[0]}.{cuda[1]}",
                "cudart_version": cudart,
            },
            indent=2,
        )
    )
    print(f"package ready: {pkg}")


if __name__ == "__main__":
    main()
