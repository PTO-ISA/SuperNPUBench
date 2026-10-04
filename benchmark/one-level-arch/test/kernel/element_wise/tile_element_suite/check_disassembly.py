#!/usr/bin/env python3
"""Check the native TILE/element-wise/TILE lowering and negative canaries."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


class CheckError(RuntimeError):
    pass


def instruction_lines(text: str) -> list[str]:
    return [
        line
        for line in text.splitlines()
        if re.match(r"^\s*[0-9a-f]+:\s+[0-9a-f]{4,8}\s+", line, re.I)
    ]


def validate(text: str, case: str) -> None:
    lines = instruction_lines(text)
    is_topk = case == "topk_boundaries" or case.startswith("topk_boundaries_")
    expected_sites = 2 if is_topk else 1
    tlea = [line for line in lines if "BSTART.TEPL" in line and "TLEA, U32" in line]
    atomic = [line for line in lines if "BSTART.TLSU" in line and "MGATHER.ADD, U32" in line]
    subview = [line for line in lines if re.search(r"\bB\.SUBVIEW\b", line)]
    masked_ior = [
        line for line in lines
        if re.search(r"\bB\.IOR\b", line) and "ExecMaskPresent" in line
    ]
    if len(tlea) != expected_sites:
        raise CheckError(
            f"expected exactly {expected_sites} static TLEA sites, found {len(tlea)}"
        )
    if len(atomic) != expected_sites:
        raise CheckError(
            "expected exactly "
            f"{expected_sites} static MGATHER.ADD sites, found {len(atomic)}"
        )
    if not subview:
        raise CheckError("missing TPARTVIEW/B.SUBVIEW lowering")
    if len(masked_ior) != expected_sites:
        raise CheckError("atomic is missing its native execution mask")
    if not any(re.search(r"\bTLOAD\b", line) for line in lines):
        raise CheckError("missing pre-element TLOAD")
    if not any(re.search(r"\bTANDS\b", line) for line in lines):
        raise CheckError("missing pre-element TANDS")
    if (case == "selected_radix_tile_element" or is_topk) and not any(
        re.search(r"\bTSHRS\b", line) for line in lines
    ):
        raise CheckError("selected radix case is missing TSHRS")
    atomic_index = lines.index(atomic[0])
    if not is_topk and not any(
        re.search(r"\bTADDS\b", line) for line in lines[atomic_index + 1 :]
    ):
        raise CheckError("missing post-element TADDS")
    if not any(re.search(r"\bTSTORE\b", line) for line in lines[atomic_index + 1 :]):
        raise CheckError("missing post-element TSTORE")
    if "CUBE_M32" not in text or not re.search(r"C\.B\.DIMI\s+32", text):
        raise CheckError("missing native local Tile geometry")


def rejected(name: str, text: str, case: str) -> None:
    try:
        validate(text, case)
    except CheckError:
        return
    raise CheckError(f"negative canary unexpectedly accepted: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--dis", required=True, type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    text = args.dis.read_text(encoding="utf-8")
    try:
        validate(text, args.case)
        if args.self_test:
            rejected("missing TLEA", text.replace("TLEA, U32", "TLEA_REMOVED, U32", 1), args.case)
            rejected("extra atomic", text + "\n" + next(line for line in text.splitlines() if "MGATHER.ADD, U32" in line), args.case)
            rejected("missing execution mask", text.replace("ExecMaskPresent", "NoExecMask", 1), args.case)
            post_op = (
                "TSTORE" if (args.case == "topk_boundaries" or
                              args.case.startswith("topk_boundaries_"))
                else "TADDS"
            )
            rejected(
                "missing post TileOp",
                text.replace(post_op, f"{post_op}_REMOVED"),
                args.case,
            )
    except (CheckError, StopIteration) as error:
        print(f"disassembly check failed: {error}")
        return 1
    print(f"{args.case}: disassembly check PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
