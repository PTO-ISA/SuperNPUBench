#!/usr/bin/env python3
"""Check microbenchmark operation coverage against pto-spec and TileOP API."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


DEFAULT_SPEC = Path("/Users/blacktraker/gitproj/DV4/pto-spec")
DEFAULT_COMPILER = Path(
    "/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/"
    "output/linx_blockisa_llvm_musl/bin"
)


def find_tileop_header(compiler_dir: Path) -> Path:
    candidates = sorted(
        (compiler_dir.parent / "lib" / "clang").glob(
            "*/include/tileop-api/jcore/template_asm.hpp"
        )
    )
    if not candidates:
        raise FileNotFoundError(f"cannot locate installed TileOP header below {compiler_dir}")
    return candidates[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-root", type=Path, default=DEFAULT_SPEC)
    parser.add_argument(
        "--compiler-dir",
        type=Path,
        default=Path(os.environ.get("COMPILER_DIR", DEFAULT_COMPILER)),
    )
    parser.add_argument(
        "--coverage", type=Path, default=Path(__file__).with_name("coverage.json")
    )
    args = parser.parse_args()

    catalog_path = args.spec_root / "spec/catalog/tile-operations.json"
    catalog = json.loads(catalog_path.read_text())
    coverage = json.loads(args.coverage.read_text())
    header_path = find_tileop_header(args.compiler_dir)
    header = header_path.read_text(errors="replace")

    spec_ops = {item["name"] for item in catalog["operations"]}
    deleted = set(catalog.get("deleted_names", []))
    api_ops = set(re.findall(r"\bvoid\s+([A-Z][A-Z0-9_]+)\s*\(", header))
    active = {
        item["operation"]
        for item in coverage["active"]
        if item["family"] != "scalar"
    }
    unsupported = {item["operation"] for item in coverage["unsupported"]}
    inventoried = active | unsupported

    print("family coverage (spec / active / unsupported-only):")
    for family in ("TEPL", "TLSU", "CUBE"):
        family_ops = {
            item["name"] for item in catalog["operations"] if item["family"] == family
        }
        print(
            f"  {family}: {len(family_ops)} / "
            f"{len(family_ops & active)} / {len(family_ops - active)}"
        )

    sections = [
        ("spec operations", spec_ops),
        ("active microbenchmark operations", active),
        ("operations with unsupported records", unsupported),
        ("unsupported-only current operations", spec_ops - active),
        ("spec operations absent from installed API", spec_ops - api_ops),
        ("retired names still exposed as compatibility API", deleted & api_ops),
        ("spec operations absent from coverage inventory", spec_ops - inventoried),
        ("non-spec tile operations marked active", active - spec_ops),
    ]
    for title, values in sections:
        print(f"{title}: {len(values)}")
        if title.startswith("spec operations absent from installed API") or title.startswith(
            "retired"
        ) or title.startswith("spec operations absent from coverage") or title.startswith(
            "non-spec"
        ):
            if values:
                print("  " + ", ".join(sorted(values)))

    errors = (spec_ops - inventoried) | (active - spec_ops)
    if errors:
        print("FAIL: coverage inventory is not aligned with pto-spec")
        return 1
    print("PASS: every current PTO operation is active or explicitly unsupported; no retired operation is active")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
