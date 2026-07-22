#!/usr/bin/env python3
"""Validate the workbook-derived intrinsic catalog and render its index.

The catalog is checked in so publishing the manual does not depend on Apple
Numbers, an external ISA checkout, or a target-specific documentation tree.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "scripts" / "data" / "intrinsic-catalog.json"
ALLOWLIST_PATH = ROOT / "scripts" / "data" / "linxisa-0.57-intrinsics.txt"
INDEX_PATH = ROOT / "docs" / "intrinsics" / "index.md"
MKDOCS_PATH = ROOT / "mkdocs.yml"
NAV_START = "  # INTRINSIC-NAV:START"
NAV_END = "  # INTRINSIC-NAV:END"
EXPECTED_WORKBOOK_SHA256 = (
    "0387b39f108599f2469c2e2374a46ea4b832a96fee7bf7aa5d7cc575fdc7864a"
)


def load_catalog() -> dict:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    operations = catalog.get("operations", [])
    names = [operation.get("name") for operation in operations]
    if catalog.get("schema") != 1:
        raise ValueError("intrinsic catalog must use schema 1")
    if len(names) != 111 or len(set(names)) != 111:
        raise ValueError("intrinsic catalog must contain 111 unique operations")
    workbook = catalog.get("workbook", {})
    if workbook.get("sha256") != EXPECTED_WORKBOOK_SHA256:
        raise ValueError("intrinsic catalog workbook digest does not match the authority")
    if workbook.get("operation_count") != 111:
        raise ValueError("intrinsic catalog workbook count must be 111")

    allowlist = [
        line.strip()
        for line in ALLOWLIST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if names != allowlist:
        raise ValueError("catalog operation order differs from the workbook allowlist")

    required = {
        "name",
        "slug",
        "category",
        "subcategory",
        "display_title",
        "summary",
        "outputs",
        "inputs",
        "data_types",
        "cxx_signatures",
        "provenance",
    }
    for operation in operations:
        missing = required - set(operation)
        if missing:
            raise ValueError(f"{operation.get('name')}: missing catalog fields {sorted(missing)}")
        if operation["slug"] != operation["name"].lower():
            raise ValueError(f"{operation['name']}: slug must be lowercase operation name")
    return catalog


def render_index(catalog: dict) -> str:
    grouped: OrderedDict[str, OrderedDict[str, list[dict]]] = OrderedDict()
    for operation in catalog["operations"]:
        grouped.setdefault(operation["category"], OrderedDict()).setdefault(
            operation["subcategory"], []
        ).append(operation)

    lines = [
        "# C++ tile intrinsic reference",
        "",
        "This reference contains the 111 operations in the authoritative API",
        "workbook and the two execution-identity functions used by kernels. The",
        "workbook defines the programming interface; compiler implementation status",
        "does not determine whether an operation belongs in this reference.",
        "",
        "Read [operand and tile conventions](conventions.md) before relying on an",
        "individual operation's constraints.",
        "",
        "## Execution identity",
        "",
        "| Function | Purpose |",
        "| --- | --- |",
        "| [`get_thread_idx()`](get_thread_idx.md) | Select the current thread's fragment or control path. |",
        "| [`get_block_idx()`](get_block_idx.md) | Select the block's global-tensor partition. |",
        "",
    ]
    for category, subcategories in grouped.items():
        lines.extend([f"## {category}", ""])
        for subcategory, operations in subcategories.items():
            lines.extend(
                [
                    f"### {subcategory}",
                    "",
                    "| Intrinsic | Operation |",
                    "| --- | --- |",
                ]
            )
            for operation in operations:
                lines.append(
                    f"| [`{operation['name']}`]({operation['slug']}.md) | "
                    f"{operation['summary']} |"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def grouped_operations(catalog: dict) -> OrderedDict[str, OrderedDict[str, list[dict]]]:
    grouped: OrderedDict[str, OrderedDict[str, list[dict]]] = OrderedDict()
    for operation in catalog["operations"]:
        grouped.setdefault(operation["category"], OrderedDict()).setdefault(
            operation["subcategory"], []
        ).append(operation)
    return grouped


def render_nav(catalog: dict) -> str:
    lines = [
        NAV_START,
        "  - Intrinsic Reference:",
        "      - Start here: intrinsics/index.md",
        "      - Operand and tile conventions: intrinsics/conventions.md",
        "      - Execution identity:",
        "          - get_thread_idx: intrinsics/get_thread_idx.md",
        "          - get_block_idx: intrinsics/get_block_idx.md",
    ]
    for category, subcategories in grouped_operations(catalog).items():
        lines.append(f"      - {category}:")
        for subcategory, operations in subcategories.items():
            lines.append(f"          - {subcategory}:")
            for operation in operations:
                lines.append(
                    f"              - {operation['name']}: "
                    f"intrinsics/{operation['slug']}.md"
                )
    lines.append(NAV_END)
    return "\n".join(lines)


def replace_nav(mkdocs: str, generated: str) -> str:
    start = mkdocs.find(NAV_START)
    end = mkdocs.find(NAV_END)
    if start < 0 or end < 0 or end < start:
        raise ValueError("mkdocs.yml is missing intrinsic navigation markers")
    end += len(NAV_END)
    return mkdocs[:start] + generated + mkdocs[end:]


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="rewrite the intrinsic index")
    mode.add_argument("--check", action="store_true", help="check catalog/index consistency")
    args = parser.parse_args()

    catalog = load_catalog()
    expected = render_index(catalog)
    if args.write:
        INDEX_PATH.write_text(expected, encoding="utf-8")
        mkdocs = MKDOCS_PATH.read_text(encoding="utf-8")
        MKDOCS_PATH.write_text(
            replace_nav(mkdocs, render_nav(catalog)), encoding="utf-8"
        )
        print("Wrote the 111-operation C++ intrinsic index and navigation.")
        return
    if not INDEX_PATH.is_file() or INDEX_PATH.read_text(encoding="utf-8") != expected:
        raise SystemExit("intrinsic index is stale; run scripts/sync_golden_manual.py --write")
    mkdocs = MKDOCS_PATH.read_text(encoding="utf-8")
    if replace_nav(mkdocs, render_nav(catalog)) != mkdocs:
        raise SystemExit("intrinsic navigation is stale; run scripts/sync_golden_manual.py --write")
    print("Intrinsic catalog verified: 111 workbook operations and two identity APIs.")


if __name__ == "__main__":
    main()
