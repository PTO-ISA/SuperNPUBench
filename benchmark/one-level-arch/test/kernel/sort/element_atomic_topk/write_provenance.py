#!/usr/bin/env python3
"""Write reproducible toolchain, repository, ELF, model, and golden identity."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "verification"))
from gfrun_dump import GFRUN_TIMEOUT_SECONDS, dump_specs, plan_dump


IMMUTABLE_INPUT_FIELDS = (
    "target",
    "compiler_version",
    "artifacts",
    "repositories",
    "commands",
)


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
        timeout=30,
    ).stdout.strip()


def git_identity(root: Path) -> dict[str, object]:
    try:
        top = Path(run("git", "rev-parse", "--show-toplevel", cwd=root))
    except (OSError, subprocess.SubprocessError):
        return {"root": str(root), "head": None, "status": None,
                "identity_available": False, "reason": "not a Git worktree"}
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
        "identity_available": True,
        "head": run("git", "rev-parse", "HEAD", cwd=top),
        "status": run("git", "status", "--short", cwd=top),
        "tracked_diff_sha256": sha256_bytes(diff),
        "changed_file_sha256": files,
    }


def immutable_inputs(identity: dict[str, object]) -> dict[str, object]:
    missing = [field for field in IMMUTABLE_INPUT_FIELDS if field not in identity]
    if missing:
        raise RuntimeError(
            "input provenance is missing immutable fields: " + ", ".join(missing)
        )
    return {field: identity[field] for field in IMMUTABLE_INPUT_FIELDS}


def _different_paths(before: Any, after: Any, prefix: str = "") -> list[str]:
    if isinstance(before, dict) and isinstance(after, dict):
        paths: list[str] = []
        for key in sorted(set(before) | set(after)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                paths.append(path)
            else:
                paths.extend(_different_paths(before[key], after[key], path))
        return paths
    return [] if before == after else [prefix]


def verify_immutable_inputs(
    before_identity: dict[str, object], current_identity: dict[str, object]
) -> None:
    before = immutable_inputs(before_identity)
    current = immutable_inputs(current_identity)
    changed = _different_paths(before, current)
    if changed:
        raise RuntimeError(
            "validation inputs changed after execution: " + ", ".join(changed)
        )


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
    parser.add_argument("--dump-range", default="")
    parser.add_argument("--compiler-source", type=Path)
    parser.add_argument("--model-source", type=Path)
    parser.add_argument("--api-source", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="record partial inputs for an already failed run")
    parser.add_argument("--run-exit-status", required=True, type=int)
    parser.add_argument("--gfrun", required=True, type=Path)
    parser.add_argument("--gfsim", type=Path)
    parser.add_argument("--models", choices=("gfrun", "gfrun,gfsim"),
                        default="gfrun,gfsim")
    parser.add_argument("--pe-count", type=int, choices=(1, 4), default=1)
    parser.add_argument("--verify-inputs-from", type=Path)
    args = parser.parse_args()
    selected_models = args.models.split(",")
    if args.pe_count == 4 and args.models != "gfrun":
        parser.error("four-PE functional acceptance requires --models gfrun")
    if "gfsim" in selected_models and args.gfsim is None:
        parser.error("--gfsim is required when gfsim is selected")

    problems: list[str] = []

    def capture(label, operation):
        try:
            return operation()
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            problems.append(f"{label}: {error}")
            return None

    def file_hash(path: Path):
        return capture(str(path), lambda: sha256_file(path))

    def repository(fallback: Path, explicit: Path | None):
        result = capture(str(explicit or fallback),
                         lambda: git_identity(explicit or fallback))
        if explicit is not None and result and result.get("identity_available") is False:
            problems.append(f"explicit source checkout is not a Git worktree: {explicit}")
        return result

    runtime_files = [args.sysroot / "usr/lib" / name for name in
                     ("libc.a", "libc++.a", "libc++abi.a", "libunwind.a")]
    golden_files = capture("golden directory", lambda: sorted(
        path for path in args.golden_dir.iterdir() if path.is_file())) or []
    if not golden_files:
        problems.append("no golden files were produced")
    installed_api_root = args.resource_dir / "include/tileop-api"
    installed_api_files = sorted(path for path in installed_api_root.rglob("*")
                                 if path.is_file())
    if not installed_api_files:
        problems.append("missing installed TileOp API headers: " + str(installed_api_root))
    chunks = capture("memory dump range", lambda: plan_dump(
        args.dump_range, args.artifact_dir / "gfrun.mem")) if args.dump_range else None
    if not args.dump_range:
        problems.append("memory dump range was not produced")
    identity: dict[str, object] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "target": {
            "triple": args.target_triple,
            "sysroot": str(args.sysroot),
            "resource_dir": str(args.resource_dir),
            "api_include": str(args.api_include),
            "selected_models": selected_models,
            "pe_count": args.pe_count,
            "gfrun_timeout_seconds": GFRUN_TIMEOUT_SECONDS,
            "gfrun_memory_dump": {"range": args.dump_range,
                                  "output": str(args.artifact_dir / "gfrun.mem"),
                                  "chunks": chunks},
        },
        "commands": {
            "gfrun": [
                str(args.gfrun),
                "--pto059-execution-mask-dev", "-s", f"softcore.multiThreadNum={args.pe_count}",
                "--dump-force", "--dump-memory",
                *dump_specs(chunks or []),
                "-f", str(args.elf),
            ],
            "gfsim": [
                str(args.gfsim),
                "--pto059-execution-mask-dev", "-s", "core.threadCount=1",
                "core.stdPeCount=1", "bctrl.assemble_enable=true",
                "--dump-force", "--dump-memory",
                f"{args.dump_range}:{args.artifact_dir / 'gfsim.mem'}",
                "-f", str(args.elf),
            ],
        },
        "run_exit_status": args.run_exit_status,
        "compiler_version": capture("compiler version", lambda: run(
            str(args.compiler_bin / "clang++"), "--version")),
        "artifacts": {
            "elf": {"path": str(args.elf), "sha256": file_hash(args.elf)},
            # Hash the invoked driver names; frozen toolchains need not retain
            # the build tree's versioned clang-15 alias.
            "clang": file_hash(args.compiler_bin / "clang++"),
            "lld": file_hash(args.compiler_bin / "ld.lld"),
            "llvm_objdump": file_hash(args.compiler_bin / "llvm-objdump"),
            "gfrun": file_hash(args.gfrun),
            **({"gfsim": file_hash(args.gfsim)}
               if "gfsim" in selected_models else {}),
            "runtime": {str(path): file_hash(path) for path in runtime_files},
            "api_header": file_hash(args.api_include / "common/pto_tileop.hpp"),
            # -mlxbc preincludes the installed TileOp API. Record that actual
            # header tree as well as the explicitly supplied include checkout.
            "installed_tileop_headers": {
                str(path.relative_to(args.resource_dir)): file_hash(path)
                for path in installed_api_files
            },
            "goldens": {path.name: file_hash(path) for path in golden_files},
        },
        "repositories": {
            "benchmark": repository(args.bench_root, None),
            "compiler": repository(args.compiler_bin, args.compiler_source),
            "model": repository(args.model_root, args.model_source),
            "tileop_api": repository(args.api_include, args.api_source),
        },
        "evidence": {
            str(path.relative_to(args.artifact_dir)): file_hash(path)
            for path in sorted(args.artifact_dir.rglob("*"))
            if path.is_file() and path != args.out
        },
    }
    if "gfsim" not in selected_models:
        del identity["commands"]["gfsim"]
    if not chunks:
        identity["commands"] = {}
    if args.verify_inputs_from is not None:
        before_identity = capture("input provenance", lambda: json.loads(
            args.verify_inputs_from.read_text(encoding="utf-8")))
        if not isinstance(before_identity, dict):
            problems.append("input provenance must be a JSON object")
        else:
            capture("input immutability", lambda: verify_immutable_inputs(before_identity, identity))
    incomplete_allowed = args.allow_incomplete and args.run_exit_status != 0
    provenance_failed = bool(problems) and not incomplete_allowed
    identity["provenance_status"] = "incomplete" if problems else "complete"
    identity["provenance_errors"] = problems
    if provenance_failed and args.run_exit_status == 0:
        identity["run_exit_status"] = 1
        identity["execution_exit_status"] = 0
    content_identity = dict(identity)
    content_identity.pop("created_utc")
    identity["content_id"] = sha256_bytes(
        json.dumps(content_identity, sort_keys=True, separators=(",", ":")).encode()
    )
    args.out.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    return int(provenance_failed)


if __name__ == "__main__":
    raise SystemExit(main())
