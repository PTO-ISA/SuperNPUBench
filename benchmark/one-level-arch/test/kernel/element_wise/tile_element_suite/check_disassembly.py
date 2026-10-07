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


def function_body(text: str, symbol_fragment: str) -> str:
    lines = text.splitlines()
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if re.match(r"^[0-9a-f]+ <.*" + re.escape(symbol_fragment), line)
        ),
        None,
    )
    if start is None:
        raise CheckError(f"missing function containing {symbol_fragment}")
    end = len(lines)
    for index in range(start + 1, len(lines)):
        match = re.match(r"^[0-9a-f]+ <([^>]+)>:", lines[index])
        if match and not match.group(1).startswith(".L"):
            end = index
            break
    return "\n".join(lines[start:end])


def validate_expression(text: str, ir_text: str | None) -> None:
    body = function_body(text, "element_expression_chain")
    lines = instruction_lines(body)
    expected = {
        "TADD": 1,
        "TSUB": 2,
        "TMUL": 1,
        "TDIV": 1,
        "TREM": 1,
        "TAND": 1,
        "TOR": 1,
        "TXOR": 3,
        "TSHL": 1,
        "TSHR": 1,
    }
    binary_indices: list[int] = []
    for mnemonic, count in expected.items():
        matches = [
            index
            for index, line in enumerate(lines)
            if re.search(rf"\bBSTART\.TEPL\s+{mnemonic}, U32\b", line)
        ]
        if len(matches) != count:
            raise CheckError(
                f"expected exactly {count} {mnemonic} element operations, "
                f"found {len(matches)}"
            )
        binary_indices.extend(matches)

    for index in sorted(binary_indices):
        bundle = lines[index : index + 5]
        if len(bundle) < 5 or "B.DATR" not in bundle[1] or "CUBE_M32" not in bundle[1]:
            raise CheckError("element expression operation is missing CUBE_M32 B.DATR")
        if not any(re.search(r"C\.B\.DIMI\s+1,", line) for line in bundle):
            raise CheckError("element expression operation is missing one-column geometry")
        if not any(re.search(r"C\.B\.DIMI\s+32,", line) for line in bundle):
            raise CheckError("element expression operation is missing 32-element geometry")
        bindings = [line for line in bundle if "B.IOT" in line]
        if len(bindings) != 1 or not re.search(
            r"B\.IOT\s+[mntu]#[0-9]+, [mntu]#[0-9]+, "
            r"mask=1111, last,\s+->[mntu]<128B>",
            bindings[0],
        ):
            raise CheckError("element expression escaped native Tile-to-Tile binding")

    if len([line for line in lines if re.search(r"\bTLOAD\b", line)]) != 1:
        raise CheckError("expression kernel must contain exactly one static TLOAD")
    if len([line for line in lines if re.search(r"\bTSTORE\b", line)]) != 3:
        raise CheckError("expression kernel must contain exactly three static TSTORE sites")
    if not any(re.search(r"\bB\.SUBVIEW\b", line) for line in lines):
        raise CheckError("expression kernel is missing TPARTVIEW/B.SUBVIEW lowering")

    if ir_text is None:
        raise CheckError("expression kernel requires LLVM IR evidence")
    ir_body_match = re.search(
        r"define[^\n]*element_expression_chain.*?^}", ir_text, re.M | re.S
    )
    if ir_body_match is None:
        raise CheckError("missing expression kernel in LLVM IR")
    ir_body = ir_body_match.group(0)
    selectors = [
        int(selector)
        for selector in re.findall(
            r"call <32 x i32> "
            r"@llvm\.linx\.experimental\.ew\.tbinary[^\n]*\(.*?"
            r"i64 29,\s*i64 ([0-9]+),",
            ir_body,
            re.S,
        )
    ]
    for selector in range(10):
        if selector not in selectors:
            raise CheckError(f"missing compiler element-binary selector {selector}")
    if len(selectors) != 13:
        raise CheckError(
            f"expected 13 native element expression operations, found {len(selectors)}"
        )
    calls = re.findall(
        r"^\s*(%[-.a-zA-Z0-9]+) = call <32 x i32> "
        r"@llvm\.linx\.experimental\.ew\.tbinary[^\n]*"
        r"i64 29,\s*i64 ([0-9]+),\s*"
        r"<32 x i32> (%[-.a-zA-Z0-9]+),\s*"
        r"<32 x i32> (%[-.a-zA-Z0-9]+)\)",
        ir_body,
        re.M,
    )
    producer = {result: index for index, (result, _op, _lhs, _rhs) in enumerate(calls)}
    has_long_lived_value = any(
        int(op) == 7
        and any(
            operand in producer and index - producer[operand] >= 5
            for operand in (lhs, rhs)
        )
        for index, (_result, op, lhs, rhs) in enumerate(calls)
    )
    if not has_long_lived_value:
        raise CheckError(
            "expression IR is missing the non-linear long-lived intermediate"
        )
    if "extractelement" in ir_body or "insertelement" in ir_body:
        raise CheckError("element loop fell back to scalar vector element access")


def validate(text: str, case: str, ir_text: str | None = None) -> None:
    if case == "element_expression_chain":
        validate_expression(text, ir_text)
        return
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


def rejected(name: str, text: str, case: str, ir_text: str | None = None) -> None:
    try:
        validate(text, case, ir_text)
    except CheckError:
        return
    raise CheckError(f"negative canary unexpectedly accepted: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--dis", required=True, type=Path)
    parser.add_argument("--ir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    text = args.dis.read_text(encoding="utf-8")
    ir_text = args.ir.read_text(encoding="utf-8") if args.ir else None
    try:
        validate(text, args.case, ir_text)
        if args.self_test:
            if args.case == "element_expression_chain":
                rejected(
                    "missing TREM",
                    text.replace("TREM, U32", "TREM_REMOVED, U32", 1),
                    args.case,
                    ir_text,
                )
                rejected(
                    "non-M32 element carrier",
                    text.replace("CUBE_M32", "CUBE_M16"),
                    args.case,
                    ir_text,
                )
                rejected(
                    "scalar extract fallback",
                    text,
                    args.case,
                    re.sub(
                        r"(define[^\n]*element_expression_chain[^\n]*\n)",
                        r"\1  %bad = extractelement <32 x i32> undef, i32 0\n",
                        ir_text or "",
                        count=1,
                    ),
                )
                rejected(
                    "missing compiler selector",
                    text,
                    args.case,
                    re.sub(
                        r"i64 29,\s*i64 4,",
                        "i64 29, i64 3,",
                        ir_text or "",
                        count=1,
                    ),
                )
                print(f"{args.case}: disassembly check PASS")
                return 0
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
