#!/usr/bin/env python3
"""Write reproducible toolchain, repository, ELF, model, and golden identity."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(*command: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()


def git_identity(root: Path) -> dict[str, object]:
    top = Path(run("git", "rev-parse", "--show-toplevel", cwd=root))
    changed = run(
        "git", "ls-files", "-m", "-o", "--exclude-standard", cwd=top
    ).splitlines()
    files = {
        path: sha256_file(top / path)
        for path in sorted(changed)
        if (top / path).is_file()
    }
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=top, check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return {
        "root": str(top),
        "head": run("git", "rev-parse", "HEAD", cwd=top),
        "status": run("git", "status", "--short", cwd=top),
        "tracked_diff_sha256": sha256_bytes(diff),
        "changed_file_sha256": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--elf", required=True, type=Path)
    parser.add_argument("--compiler-bin", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--api-include", required=True, type=Path)
    parser.add_argument("--bench-root", required=True, type=Path)
    parser.add_argument("--golden-dir", required=True, type=Path)
    parser.add_argument("--target-triple", required=True)
    parser.add_argument("--sysroot", required=True, type=Path)
    parser.add_argument("--resource-dir", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--dump-range", required=True)
    parser.add_argument("--run-exit-status", required=True, type=int)
    parser.add_argument("--gfrun", required=True, type=Path)
    parser.add_argument("--gfsim", required=True, type=Path)
    args = parser.parse_args()

    compiler_repo = Path(run("git", "rev-parse", "--show-toplevel", cwd=args.compiler_bin))
    api_repo = Path(run("git", "rev-parse", "--show-toplevel", cwd=args.api_include))
    runtime_files = [
        args.runtime_root / "sysroot/usr/lib/libc.a",
        args.runtime_root / "sysroot/usr/lib/libc++.a",
        args.runtime_root / "sysroot/usr/lib/libc++abi.a",
        args.runtime_root / "sysroot/usr/lib/libunwind.a",
    ]
    golden_files = sorted(path for path in args.golden_dir.iterdir() if path.is_file())
    installed_api_root = args.resource_dir / "include/tileop-api"
    installed_api_files = sorted(path for path in installed_api_root.rglob("*")
                                 if path.is_file())
    if not installed_api_root.is_dir() or not installed_api_files:
        raise RuntimeError("missing installed TileOp API headers: " +
                           str(installed_api_root))
    identity: dict[str, object] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "target": {
            "triple": args.target_triple,
            "sysroot": str(args.sysroot),
            "resource_dir": str(args.resource_dir),
            "api_include": str(args.api_include),
        },
        "commands": {
            "gfrun": [
                str(args.gfrun),
                "--pto059-execution-mask-dev", "-s", "softcore.multiThreadNum=1",
                "--dump-force", "--dump-memory",
                f"{args.dump_range}:{args.artifact_dir / 'gfrun.mem'}",
                "-f", str(args.elf),
            ],
            "gfsim": [
                str(args.gfsim),
                "--pto059-execution-mask-dev", "-s", "core.threadCount=1",
                "core.stdPeCount=1", "--dump-force", "--dump-memory",
                f"{args.dump_range}:{args.artifact_dir / 'gfsim.mem'}",
                "-f", str(args.elf),
            ],
        },
        "run_exit_status": args.run_exit_status,
        "compiler_version": run(str(args.compiler_bin / "clang++"), "--version"),
        "artifacts": {
            "elf": {"path": str(args.elf), "sha256": sha256_file(args.elf)},
            "clang": sha256_file(args.compiler_bin / "clang-15"),
            "lld": sha256_file(args.compiler_bin / "lld"),
            "llvm_objdump": sha256_file(args.compiler_bin / "llvm-objdump"),
            "gfrun": sha256_file(args.gfrun),
            "gfsim": sha256_file(args.gfsim),
            "runtime": {str(path): sha256_file(path) for path in runtime_files},
            "api_header": sha256_file(args.api_include / "common/pto_tileop.hpp"),
            # -mlxbc preincludes the installed TileOp API. Record that actual
            # header tree as well as the explicitly supplied include checkout.
            "installed_tileop_headers": {
                str(path.relative_to(args.resource_dir)): sha256_file(path)
                for path in installed_api_files
            },
            "goldens": {path.name: sha256_file(path) for path in golden_files},
        },
        "repositories": {
            "benchmark": git_identity(args.bench_root),
            "compiler": git_identity(compiler_repo),
            "model": git_identity(args.model_root),
            "tileop_api": git_identity(api_repo),
        },
        "evidence": {
            str(path.relative_to(args.artifact_dir)): sha256_file(path)
            for path in sorted(args.artifact_dir.rglob("*"))
            if path.is_file() and path != args.out
        },
    }
    content_identity = dict(identity)
    content_identity.pop("created_utc")
    identity["content_id"] = sha256_bytes(
        json.dumps(content_identity, sort_keys=True, separators=(",", ":")).encode()
    )
    args.out.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
