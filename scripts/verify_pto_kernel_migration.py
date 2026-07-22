#!/usr/bin/env python3
"""Verify that imported PTO-Kernel sources stay on the PTO 0.57 surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


PTO_CALL = re.compile(r"\b((?:T|M)[A-Z][A-Z0-9_]*)\s*\(")
SCALAR_POINTER_ACCESS = re.compile(r"\b[A-Za-z_]\w*_ptr\s*\[")
REMOVED_API = re.compile(r"\b(?:TSYNC|RecordEvent|WaitEvents|event_t)\b")


def read_allowlist(root: Path) -> set[str]:
    path = root / "scripts" / "data" / "linxisa-0.57-intrinsics.txt"
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def check_migration(root: Path, errors: list[str]) -> None:
    kernel_root = root / "benchmark" / "one-level-arch" / "kernels" / "pto_kernels"
    support_root = root / "benchmark" / "one-level-arch" / "include" / "pto_kernel"
    test_root = root / "benchmark" / "one-level-arch" / "test" / "kernel" / "pto_kernels"
    manifest_path = kernel_root / "migration.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowlist = read_allowlist(root)

    if manifest.get("schema") != 1:
        errors.append(f"{manifest_path}: expected schema 1")
    revision = manifest.get("revision", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        errors.append(f"{manifest_path}: revision is not a full Git commit")

    kernels = manifest.get("kernels", [])
    support_headers = manifest.get("support_headers", [])
    benchmarks = [item.get("benchmark", "") for item in kernels]
    if len(kernels) != 9 or len(set(benchmarks)) != 9:
        errors.append(f"{manifest_path}: expected nine unique migrated kernels")

    declared_sources = {item.get("source", "") for item in kernels}
    actual_sources = {
        path.relative_to(kernel_root).as_posix()
        for path in kernel_root.rglob("*.cpp")
    }
    if actual_sources != declared_sources:
        errors.append(
            "PTO migration source set differs from migration.json: "
            f"missing={sorted(declared_sources - actual_sources)}, "
            f"extra={sorted(actual_sources - declared_sources)}"
        )

    declared_headers = {item.get("source", "") for item in support_headers}
    actual_headers = {
        path.relative_to(support_root).as_posix()
        for path in support_root.rglob("*.hpp")
    }
    if len(support_headers) != 7 or actual_headers != declared_headers:
        errors.append(
            "PTO support-header set differs from migration.json: "
            f"missing={sorted(declared_headers - actual_headers)}, "
            f"extra={sorted(actual_headers - declared_headers)}"
        )
    for item in support_headers:
        path = support_root / item["source"]
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item.get("sha256"):
            errors.append(f"{path}: content differs from the pinned migration digest")
        text = path.read_text(encoding="utf-8")
        for include in re.findall(r'^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]', text, re.MULTILINE):
            if include.startswith(("common/", "pto/")):
                errors.append(f"{path}: support include is not isolated under pto_kernel/: {include}")

    build_surface = "\n".join(
        (test_root / name).read_text(encoding="utf-8")
        for name in ("Makefile", "compile.all")
    )
    if "PTO_HOST_SIM" in build_surface:
        errors.append(f"{test_root}: active Linx build enables the host-simulation backend")

    compile_text = (test_root / "compile.all").read_text(encoding="utf-8")
    compile_cases = set(re.findall(r"\bpto_[a-z0-9_]+\b", compile_text))
    if compile_cases != set(benchmarks):
        errors.append(
            "PTO migration manifest differs from compile.all: "
            f"missing={sorted(set(benchmarks) - compile_cases)}, "
            f"extra={sorted(compile_cases - set(benchmarks))}"
        )

    for item in kernels:
        benchmark = item["benchmark"]
        source = kernel_root / item["source"]
        wrapper = test_root / "src" / f"{benchmark}.cpp"
        if not source.is_file():
            errors.append(f"missing migrated kernel: {source}")
            continue
        if not wrapper.is_file():
            errors.append(f"missing migrated benchmark wrapper: {wrapper}")
        elif f'"pto_kernels/{item["source"]}"' not in wrapper.read_text(encoding="utf-8"):
            errors.append(f"{wrapper}: does not include the declared migrated source")

        text = source.read_text(encoding="utf-8")
        calls = set(PTO_CALL.findall(text))
        unsupported = calls - allowlist
        if not calls:
            errors.append(f"{source}: no PTO intrinsic calls found")
        if unsupported:
            errors.append(f"{source}: unsupported PTO calls {sorted(unsupported)}")
        if "PTO_QEMU_SMOKE" in text or SCALAR_POINTER_ACCESS.search(text):
            errors.append(f"{source}: contains a scalar data-path fallback")
        if REMOVED_API.search(text):
            errors.append(f"{source}: contains a removed event/synchronization API")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    errors: list[str] = []
    check_migration(args.repo_root.resolve(), errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(f"PTO kernel migration verification failed with {len(errors)} errors")
    print("PTO kernel migration verified: 9 tile-only kernels, PTO 0.57 allowlist clean.")


if __name__ == "__main__":
    main()
