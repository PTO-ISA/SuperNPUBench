#!/usr/bin/env python3
"""Verify the pinned DeepSeek TileKernels migration and generated reference."""

from __future__ import annotations

import json
import re
from pathlib import Path


REVISION = "36d9e45d38e204ebb87e6f6e833821eee0482fe5"
CALL = re.compile(r"\b((?:T|M)[A-Z][A-Z0-9_]*)\s*(?:<[^;{}()]*>)?\s*\(")
BANNED = re.compile(r"\b(?:RecordEvent|WaitEvents|TSYNC)\b", re.IGNORECASE)
SCALAR_FALLBACK = re.compile(
    r"scalar (?:fallback|completion)|标量(?:补全|cursor)|待 (?:scan|toolchain)",
    re.IGNORECASE,
)


def catalog_names(root: Path) -> set[str]:
    catalog = json.loads(
        (root / "scripts/data/intrinsic-catalog.json").read_text(encoding="utf-8")
    )
    return {str(entry["name"]) for entry in catalog["operations"]}


def check_migration(root: Path, errors: list[str]) -> None:
    path = root / "benchmark/one-level-arch/kernels/deepseek/migration.json"
    if not path.is_file():
        errors.append(f"missing DeepSeek migration manifest: {path}")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("kernel_modules", [])
    if data.get("schema") != 1:
        errors.append("DeepSeek migration manifest schema must be 1")
    if data.get("upstream", {}).get("revision") != REVISION:
        errors.append("DeepSeek migration manifest does not use the pinned revision")
    if data.get("minimum_thread_fragment_bytes") != 128:
        errors.append("DeepSeek migration manifest must require 128-byte fragments")
    if len(entries) != 37:
        errors.append(f"DeepSeek manifest has {len(entries)} modules; expected 37")

    from generate_deepseek_manifest import SOURCES

    expected = {
        f"tile_kernels/{family}/{filename}"
        for family, filenames in SOURCES.items()
        for filename in filenames
    }
    actual = {str(entry.get("source")) for entry in entries}
    if actual != expected:
        errors.append(
            "DeepSeek source inventory mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )

    allowed = catalog_names(root)
    referenced_implementations: set[Path] = set()
    for entry in entries:
        source = str(entry.get("source", ""))
        for field in ("title", "summary", "algorithm", "partition", "tile_flow", "verification"):
            if not entry.get(field):
                errors.append(f"{source}: missing manifest field {field}")
        intrinsics = set(entry.get("intrinsics", []))
        if not intrinsics <= allowed:
            errors.append(f"{source}: non-public intrinsics {sorted(intrinsics - allowed)}")
        implementations = entry.get("implementations", [])
        if not implementations:
            errors.append(f"{source}: no translated implementation")
        for relative in implementations:
            implementation = root / str(relative)
            referenced_implementations.add(implementation)
            if not implementation.is_file():
                errors.append(f"{source}: missing implementation {relative}")

    for implementation in referenced_implementations:
        if not implementation.is_file():
            continue
        text = implementation.read_text(encoding="utf-8", errors="replace")
        if BANNED.search(text):
            errors.append(f"{implementation.relative_to(root)}: uses a removed completion API")
        if SCALAR_FALLBACK.search(text):
            errors.append(f"{implementation.relative_to(root)}: retains a scalar tensor fallback")
        calls = set(CALL.findall(re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.DOTALL)))
        if not calls <= allowed:
            errors.append(
                f"{implementation.relative_to(root)}: calls non-public tile operations "
                f"{sorted(calls - allowed)}"
            )

    docs = root / "docs/benchmarks/deepseek"
    pages = list(docs.glob("*/*.md")) if docs.is_dir() else []
    if len(pages) != 37:
        errors.append(f"generated DeepSeek reference has {len(pages)} pages; expected 37")
    for page in pages:
        text = page.read_text(encoding="utf-8")
        for heading in (
            "Core C++ kernel",
            "Algorithm walkthrough",
            "Source-to-intrinsic mapping",
            "Block and thread partition",
            "Memory and tile flow",
            "Build and verification",
        ):
            if f"## {heading}" not in text:
                errors.append(f"{page.relative_to(root)}: missing {heading} section")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    errors: list[str] = []
    check_migration(root, errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(f"DeepSeek migration verification failed with {len(errors)} errors")
    print("DeepSeek migration verified: 37 pinned source modules and generated pages.")


if __name__ == "__main__":
    main()
