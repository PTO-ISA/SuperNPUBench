#!/usr/bin/env python3
"""Check the native TILE/element-wise/TILE lowering and negative canaries."""

from __future__ import annotations

import argparse
import re
from collections import Counter
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


def ir_function_body(text: str, symbol_fragment: str) -> str:
    match = re.search(
        r"define[^\n]*" + re.escape(symbol_fragment) + r".*?^}",
        text,
        re.M | re.S,
    )
    if match is None:
        raise CheckError(f"missing IR function containing {symbol_fragment}")
    return match.group(0)


def validate_generic_predicated_cfg(text: str, ir_text: str | None) -> None:
    if ir_text is None:
        raise CheckError("generic predicated CFG case requires LLVM IR evidence")

    for name in ("nested_three_way", "reversed_continue", "direct_signed_store"):
        direct_store = name == "direct_signed_store"
        body = function_body(text, name)
        lines = instruction_lines(body)
        gathers = [
            line for line in lines
            if re.search(r"\bBSTART\.TLSU\s+MGATHER, U32\b", line)
        ]
        scatters = [
            line for line in lines
            if re.search(
                r"\bBSTART\.TLSU\s+MSCATTER, "
                + (r"S32\b" if direct_store else r"U32\b"), line
            )
        ]
        if len(gathers) < 3:
            raise CheckError(f"{name} needs affine U32 MGATHER sites for all inputs")
        if not scatters or "MSCATTER.MASK" in body:
            raise CheckError(f"{name} needs an ordinary U32 MSCATTER site")
        if sum(bool(re.search(r"\bBSTART\.TEPL\s+TDIV, S32\b", line))
               for line in lines) != 1:
            raise CheckError(f"{name} lost its single masked signed division")
        if not any(re.search(r"\bBSTART\.TEPL\s+TCMPS, (S32|U32)\b", line)
                   for line in lines):
            raise CheckError(f"{name} is missing scalar GPR predicate comparisons")
        select_sites = [
            line for line in lines
            if re.search(r"\bBSTART\.TEPL\s+TSEL, U32\b", line)
        ]
        if (direct_store and select_sites) or (not direct_store and not select_sites):
            raise CheckError(f"{name} is missing GPR predicate value merges")
        masked_binary_sites = sum(
            bool(re.search(r"\bBSTART\.TEPL\s+"
                           r"(TADD|TSUB|TMUL|TDIV|TXOR),", line))
            for line in lines
        )
        if sum("ExecMaskPresent" in line for line in lines) < (
            len(gathers) + len(scatters) + masked_binary_sites
        ):
            raise CheckError(f"{name} lost required operation execution masks")

        ir = ir_function_body(ir_text, name)
        if re.search(
            r"extractelement|insertelement|"
            r"@llvm\.linx\.experimental\.element\.(region|view)",
            ir,
        ):
            raise CheckError(f"{name} contains residual scalar/region lowering")

        def calls(stem: str) -> list[str]:
            return re.findall(
                r"call [^\n]*@llvm\.linx\.experimental\.ew\."
                + re.escape(stem) + r"[^\n]*",
                ir,
            )

        gather_calls = calls("mgather.gpr.masked")
        scatter_calls = calls("mscatter.gpr.masked")
        binary_calls = calls("tbinary.gpr.masked")
        compare_calls = calls("tcmps.gpr")
        select_calls = calls("tsel.gpr")
        if len(gather_calls) < 3 or not scatter_calls:
            raise CheckError(f"{name} lost its three-load/one-store GM shape")
        if len(binary_calls) < (2 if direct_store else 5):
            raise CheckError(f"{name} has too few semantic Tile binary operations")
        operations = [
            (int(dtype), int(opcode))
            for call in binary_calls
            for dtype, opcode in re.findall(
                r"i64 32,\s*i64 1,\s*i64 (17|25),\s*i64 29,\s*"
                r"i64 ([0-9]+),",
                call,
            )
        ]
        # `value * 3 - 17` may canonically fold to add(-17), so opcode 1 is
        # optional. The remaining operations cannot disappear without changing
        # one of the three source-level arms.
        required_opcodes = (0, 3) if direct_store else (0, 2, 3, 7)
        for opcode in required_opcodes:
            if not any(candidate == opcode for _dtype, candidate in operations):
                raise CheckError(f"{name} is missing masked tbinary opcode {opcode}")
        if any(opcode == 4 for _dtype, opcode in operations):
            raise CheckError(f"{name} unexpectedly emitted remainder opcode 4")
        if len(operations) != len(binary_calls):
            raise CheckError(f"{name} has a non-i32/M32 masked tbinary ABI")
        if any((dtype == 17) != (opcode == 3)
               for dtype, opcode in operations):
            raise CheckError(f"{name} lost signed-division/raw-U32 opcode typing")
        for call in gather_calls:
            if not re.search(
                r"i64 32,\s*i64 1,\s*i64 25,\s*i64 0,\s*"
                r"i64 29,\s*i64 24,.*i64 0,\s*i64 0,\s*i64 1\)",
                call,
            ):
                raise CheckError(f"{name} MGATHER lost M32/PadZero/ZERO1 ABI")
        for call in scatter_calls:
            scatter_type = 17 if direct_store else 25
            if not re.search(
                rf"i64 32,\s*i64 1,\s*i64 {scatter_type},\s*i64 29,\s*"
                r"i64 24,.*i64 0,\s*i64 0,\s*i64 0\)",
                call,
            ):
                raise CheckError(f"{name} MSCATTER lost M32/ZERO0 ABI")
        minimum_compares = 2 if direct_store else 3
        if len(compare_calls) < minimum_compares or any(
            not re.search(r"i64 32,\s*i64 1,\s*i64 (17|24|25),\s*i64 29,", call)
            for call in compare_calls
        ):
            raise CheckError(f"{name} lost i32/M32 scalar GPR comparisons")
        compare_types = {
            int(dtype)
            for call in compare_calls
            for dtype in re.findall(
                r"i64 32,\s*i64 1,\s*i64 (17|24|25),\s*i64 29,", call
            )
        }
        required_compare_types = {17, 24} if direct_store else {17, 25}
        if not required_compare_types.issubset(compare_types):
            raise CheckError(f"{name} lost signed and raw-U32 branch tests")
        if direct_store:
            if select_calls:
                raise CheckError(f"{name} unexpectedly introduced a value merge")
        elif len(select_calls) < 2 or any(
                not re.search(
                    r"i64 32,\s*i64 1,\s*i64 25,\s*i64 29,", call
                ) for call in select_calls):
            raise CheckError(f"{name} lost raw-U32/M32 GPR value merges")


def validate_generic_typed_tile_cfg(text: str, ir_text: str | None) -> None:
    body = function_body(text, "generic_typed_tile_cfg_i32")
    lines = instruction_lines(body)

    def sites(pattern: str) -> list[int]:
        return [index for index, line in enumerate(lines) if re.search(pattern, line)]

    tload = sites(r"\bBSTART\.TLSU\s+TLOAD, S32\b")
    subview = sites(r"\bB\.SUBVIEW\b")
    tadds = sites(r"\bTADDS\b")
    tmov = sites(r"\bBSTART\.TLSU\s+TMOV, S32\b")
    tstore = sites(r"\bBSTART\.TLSU\s+TSTORE, S32\b")
    if len(tload) != 1 or not subview:
        raise CheckError("typed CFG lost TLOAD128/TPARTVIEW32 lowering")
    if len(tadds) != 2 or len(tmov) != 1 or len(tstore) != 1:
        raise CheckError("typed CFG lost its exact TileOp bridge shape")
    if not (tload[0] < tadds[0] < tadds[1] < tmov[0] < tstore[0]):
        raise CheckError("typed CFG no longer alternates TileOp/element/TileOp in order")
    if not (tload[0] < subview[0] < tstore[0]):
        raise CheckError("typed CFG moved TPARTVIEW outside the Tile lifetime")
    # The generic path implements truncating signed division through absolute
    # values, one masked U32 TDIV, and a sign restore. This avoids PTO's floor
    # remainder/division ambiguity while preserving C++ semantics.
    if sum(bool(re.search(r"\bBSTART\.TEPL\s+TDIV, U32\b", line))
           for line in lines) != 1:
        raise CheckError("typed CFG lost its masked signed division")
    if sum(bool(re.search(r"\bBSTART\.TEPL\s+TXOR, U32\b", line))
           for line in lines) < 2:
        raise CheckError("typed CFG lost one of its ordinary C++ XOR expressions")
    if sum("ExecMaskPresent" in line for line in lines) < 4:
        raise CheckError("typed CFG operations lost predicate execution masks")
    if "CUBE_M32" not in body or "M322ND" not in body or "ND2M32" not in body:
        raise CheckError("typed CFG lost S32 M32 typed transport")

    if ir_text is None:
        raise CheckError("typed CFG case requires LLVM IR evidence")
    ir = ir_function_body(ir_text, "generic_typed_tile_cfg_i32")
    if re.search(
        r"extractelement|insertelement|"
        r"@llvm\.linx\.experimental\.element\.(region|view)",
        ir,
    ):
        raise CheckError("typed CFG contains residual scalar/region/view lowering")
    binary_calls = re.findall(
        r"call <32 x i32> "
        r"@llvm\.linx\.experimental\.ew\.tbinary\.gpr\.masked[^\n]*",
        ir,
    )
    select_calls = re.findall(
        r"call <32 x i32> @llvm\.linx\.experimental\.ew\.tsel\.gpr[^\n]*",
        ir,
    )
    compare_calls = re.findall(
        r"call i64 @llvm\.linx\.experimental\.ew\.tcmps\.gpr[^\n]*",
        ir,
    )
    if len(binary_calls) < 5 or len(select_calls) < 2 or len(compare_calls) < 3:
        raise CheckError("typed CFG lost generic masked CFG operations or value merges")
    if "llvm.linx.experimental.ew.tdiv" in ir:
        raise CheckError("typed CFG used a non-generic unmasked division path")


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
    # LLVM can legally fold ~(-x) into x - 1 under U32 modulo arithmetic.
    # Accept only those two exact native forms; the optimized form also needs
    # the matching source/UINT32_MAX dataflow proof below.
    optimized_expected = dict(expected, TADD=2, TSUB=1, TXOR=2)
    binary_indices: list[int] = []
    counts: dict[str, int] = {}
    for mnemonic, count in expected.items():
        matches = [
            index
            for index, line in enumerate(lines)
            if re.search(rf"\bBSTART\.TEPL\s+{mnemonic}, U32\b", line)
        ]
        counts[mnemonic] = len(matches)
        binary_indices.extend(matches)
    if counts not in (expected, optimized_expected):
        raise CheckError(f"unexpected native expression operation counts: {counts}")

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
    if re.search(
        r"\bcall\b[^\n]*@llvm\.linx\.experimental\.element\.(region|view)\b",
        ir_body,
    ):
        raise CheckError("required element region/view contract survived lowering")
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
    expected_calls = sum(counts.values())
    if len(selectors) != expected_calls:
        raise CheckError(
            f"expected {expected_calls} native expression operations, found {len(selectors)}"
        )
    if sum("ExecMaskPresent" in line for line in lines) != expected_calls:
        raise CheckError("generic expression operations lost their execution masks")
    calls = re.findall(
        r"^\s*(%[-.a-zA-Z0-9]+) = call <32 x i32> "
        r"@llvm\.linx\.experimental\.ew\.tbinary\.gpr\.masked[^\n]*"
        r"i64 29,\s*i64 ([0-9]+),\s*"
        r"<32 x i32> (%[-.a-zA-Z0-9]+),\s*"
        r"<32 x i32> (%[-.a-zA-Z0-9]+)[^)\n]*\)",
        ir_body,
        re.M,
    )
    producer = {result: index for index, (result, _op, _lhs, _rhs) in enumerate(calls)}
    if expected_calls == 12:
        minus_one_values = {
            result
            for result, scalar in re.findall(
                r"^\s*(%[-.a-zA-Z0-9]+) = freeze i32 (-?[0-9]+)$",
                ir_body, re.M,
            )
            if int(scalar) in (-1, 0xFFFFFFFF)
        }
        minus_one_values.update(
            result
            for result, source in re.findall(
                r"^\s*(%[-.a-zA-Z0-9]+) = zext i32 "
                r"(%[-.a-zA-Z0-9]+) to i64$",
                ir_body, re.M,
            )
            if source in minus_one_values
        )
        maximum_splats = {
            result
            for result, scalar in re.findall(
                r"^\s*(%[-.a-zA-Z0-9]+) = call <32 x i32> "
                r"@llvm\.linx\.experimental\.ew\.tci[^\n]*"
                r"\(i64 32, i64 1, i64 25, i64 29, i64 "
                r"(%[-.a-zA-Z0-9]+|-?[0-9]+), i64 0\)",
                ir_body, re.M,
            )
            if scalar in minus_one_values or
            (not scalar.startswith("%") and int(scalar) in (-1, 0xFFFFFFFF))
        }
        if len(calls) != 12 or int(calls[-1][1]) != 0:
            raise CheckError("optimized unary region must finish with native TADD")
        _result, _op, lhs, rhs = calls[-1]
        original_input = calls[0][2]
        if not ((lhs == original_input and rhs in maximum_splats) or
                (rhs == original_input and lhs in maximum_splats)):
            raise CheckError("optimized unary region is missing exact input + UINT32_MAX")
    required_span = 5 if expected_calls == 13 else 4
    has_long_lived_value = any(
        int(op) == 7
        and any(
            operand in producer and index - producer[operand] >= required_span
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


def validate_signed_expression(text: str, ir_text: str | None) -> None:
    body = function_body(text, "signed_element_expression")
    lines = instruction_lines(body)
    for mnemonic, dtype, count in (
        ("TADD", "S32", 2), ("TADD", "U32", 2),
        ("TMUL", "U32", 2), ("TDIV", "S32", 2),
        ("TSUB", "U32", 1), ("TAND", "U32", 1),
        ("TXOR", "U32", 1), ("TSHR", "U32", 1),
    ):
        if sum(bool(re.search(rf"\bBSTART\.TEPL\s+{mnemonic}, {dtype}\b", line))
               for line in lines) != count:
            raise CheckError(f"signed expression lost {mnemonic}/{dtype} count {count}")
    if "TREM" in body:
        raise CheckError("C++ signed remainder cannot lower directly to PTO TREM")
    for mnemonic, count in (("TLOAD", 1), ("TSTORE", 3)):
        sites = [line for line in lines
                 if re.search(rf"\bBSTART\.TLSU\s+{mnemonic},", line)]
        if len(sites) != count or any(
                not re.search(rf"\b{mnemonic}, S32\b", line) for line in sites):
            raise CheckError(f"signed expression needs exactly {count} total {mnemonic}/S32 sites")
    if "CUBE_M32" not in body or "M322ND" not in body or "ND2M32" not in body:
        raise CheckError("signed expression lost typed M32 transport")
    if sum("ExecMaskPresent" in line for line in lines) != 12:
        raise CheckError("signed generic operations lost their execution masks")
    if ir_text is None:
        raise CheckError("signed expression requires LLVM IR evidence")
    match = re.search(r"define[^\n]*signed_element_expression.*?^}", ir_text, re.M | re.S)
    if match is None:
        raise CheckError("missing signed expression kernel in IR")
    ir = match.group(0)
    if re.search(r"extractelement|insertelement|@llvm\.linx\.experimental\.element\.(region|view)", ir):
        raise CheckError("signed expression contains residual scalar/view lowering")
    calls = re.findall(
        r"^\s*(%[-.a-zA-Z0-9]+) = call <32 x i32> "
        r"@llvm\.linx\.experimental\.ew\.tbinary\.gpr\.masked[^\n]*\("
        r"i64 32, i64 1, i64 (17|25), i64 29, i64 ([0-9]+), "
        r"<32 x i32> (%[-.a-zA-Z0-9]+), <32 x i32> (%[-.a-zA-Z0-9]+)"
        r"[^)\n]*\)", ir, re.M)
    expected_operations = Counter({
        ("25", "0"): 2,
        ("17", "0"): 2,
        ("25", "1"): 1,
        ("25", "2"): 2,
        ("17", "3"): 2,
        ("25", "5"): 1,
        ("25", "7"): 1,
        ("25", "9"): 1,
    })
    if Counter((dtype, op) for _, dtype, op, _, _ in calls) != expected_operations:
        raise CheckError("signed expression IR lost exact operation dtype/shape")
    operations = {result: (dtype, op, lhs, rhs) for result, dtype, op, lhs, rhs in calls}
    proves_remainder = False
    for _, dtype, op, lhs, rhs in calls:
        if dtype != "25" or op != "1" or rhs not in operations:
            continue
        mdtype, multiply, quotient, divisor = operations[rhs]
        if mdtype != "25" or multiply != "2" or quotient not in operations:
            continue
        qdtype, divide, dividend, qdivisor = operations[quotient]
        proves_remainder |= (qdtype == "17" and divide == "3" and
                             dividend == lhs and qdivisor == divisor)
    if not proves_remainder:
        raise CheckError("signed % lost exact dividend - trunc_quotient * divisor dataflow")


def validate_indexed_gather(text: str, ir_text: str | None) -> None:
    body = function_body(text, "indexed_gather_tile_element")
    lines = instruction_lines(body)

    def sites(pattern: str) -> list[int]:
        return [
            index for index, line in enumerate(lines)
            if re.search(pattern, line)
        ]

    tload = sites(r"\bTLOAD\b")
    tadds = sites(r"\bTADDS\b")
    tlea = sites(r"\bBSTART\.TEPL\s+TLEA, U32\b")
    gather = sites(r"\bBSTART\.TLSU\s+MGATHER, U32\b")
    tstore = sites(r"\bTSTORE\b")
    if len(tload) != 1:
        raise CheckError(f"expected one static TLOAD, found {len(tload)}")
    if len(tadds) != 2:
        raise CheckError(f"expected two static TADDS sites, found {len(tadds)}")
    if len(tlea) != 1:
        raise CheckError(f"expected one U32 TLEA site, found {len(tlea)}")
    if len(gather) != 1:
        raise CheckError(
            f"expected one ordinary U32 MGATHER site, found {len(gather)}"
        )
    if len(tstore) != 1:
        raise CheckError(f"expected one static TSTORE, found {len(tstore)}")
    if not (tload[0] < tadds[0] < tlea[0] < gather[0] < tadds[1] < tstore[0]):
        raise CheckError("kernel does not alternate TileOp/gather/TileOp in order")
    if "MGATHER.ADD" in body:
        raise CheckError("ordinary indexed load was replaced by an atomic opcode")
    if not sites(r"\bB\.SUBVIEW\b"):
        raise CheckError("missing TPARTVIEW/B.SUBVIEW lowering")
    masked_ior = [
        line for line in lines
        if re.search(r"\bB\.IOR\b", line) and "ExecMaskPresent" in line
    ]
    if len(masked_ior) != 1:
        raise CheckError("ordinary gather is missing its GPR execution mask")
    if "CUBE_M32" not in body or not sites(r"C\.B\.DIMI\s+32,"):
        raise CheckError("ordinary gather lost the ElementTile M32 geometry")

    if ir_text is None:
        raise CheckError("ordinary indexed gather requires LLVM IR evidence")
    ir_body_match = re.search(
        r"define[^\n]*indexed_gather_tile_element.*?^}", ir_text, re.M | re.S
    )
    if ir_body_match is None:
        raise CheckError("missing ordinary indexed gather kernel in LLVM IR")
    ir_body = ir_body_match.group(0)
    if re.search(
        r"\bcall\b[^\n]*@llvm\.linx\.experimental\.element\.(region|view)\b",
        ir_body,
    ):
        raise CheckError("required element region/view contract survived lowering")
    tlea_call = re.search(
        r"call <32 x i64> "
        r"@llvm\.linx\.experimental\.ew\.tlea[^\n]*\("
        r"i64 32, i64 1, i64 25, i64 29, <32 x i32>[^\n]*i64 32\)",
        ir_body,
    )
    if tlea_call is None:
        raise CheckError("TLEA IR lost U32-to-byte scaling or M32 geometry")
    if "@llvm.linx.experimental.ew.tcmps.gpr" not in ir_body:
        raise CheckError("tail condition did not produce a GPR execution mask")
    gather_call = re.search(
        r"call <32 x i32> "
        r"@llvm\.linx\.experimental\.ew\.mgather\.gpr\.masked[^\n]*\("
        r"i64 32, i64 1, i64 25, i64 0, i64 29, i64 24, ptr [^,]+, "
        r"<32 x i64> [^,]+, i64 %[-.a-zA-Z0-9]+, i64 0, i64 0, i64 1\)",
        ir_body,
    )
    if gather_call is None:
        raise CheckError("missing ordinary GPR-masked MGATHER intrinsic contract")
    if "mgather.add" in ir_body:
        raise CheckError("ordinary indexed load emitted an atomic intrinsic")
    if "extractelement" in ir_body or "insertelement" in ir_body:
        raise CheckError("indexed gather fell back to scalar vector element access")


def validate_generic_atomic(text: str, ir_text: str | None) -> None:
    body = function_body(text, "generic_atomic_cfg_i32")
    lines = instruction_lines(body)
    atomics = [
        line for line in lines
        if re.search(r"\bBSTART\.TLSU\s+MGATHER\.ADD, U32\b", line)
    ]
    tleas = [
        line for line in lines
        if re.search(r"\bBSTART\.TEPL\s+TLEA, S32\b", line)
    ]
    if len(atomics) != 3:
        raise CheckError(
            f"paired/null-base case needs three atomic sites, found {len(atomics)}"
        )
    if len(tleas) != 1:
        raise CheckError(
            "signed indices need one cached B32-to-B64 TLEA stream across bases"
        )
    masked_ior = [
        line for line in lines
        if re.search(r"\bB\.IOR\b", line) and "ExecMaskPresent" in line
    ]
    if len(masked_ior) < 3:
        raise CheckError("atomic sites lost their CFG execution masks")
    if sum(bool(re.search(r"\bBSTART\.TLSU\s+TLOAD, S32\b", line))
           for line in lines) != 2:
        raise CheckError("signed index/delta inputs lost their two S32 TLOAD sites")
    if sum(bool(re.search(r"\bBSTART\.TLSU\s+TLOAD, U32\b", line))
           for line in lines) != 1:
        raise CheckError("inactive atomic gate lost its U32 TLOAD site")
    if sum(bool(re.search(r"\bBSTART\.TLSU\s+TSTORE, U32\b", line))
           for line in lines) != 4:
        raise CheckError("paired atomic observations lost their four U32 TSTORE sites")
    if "CUBE_M32" not in body:
        raise CheckError("generic atomic case lost public ElementTile M32 transport")

    if ir_text is None:
        raise CheckError("generic atomic CFG case requires LLVM IR evidence")
    ir = ir_function_body(ir_text, "generic_atomic_cfg_i32")
    if re.search(
        r"atomicrmw|extractelement|insertelement|"
        r"@llvm\.linx\.experimental\.element\.(region|view)",
        ir,
    ):
        raise CheckError("generic atomic CFG contains residual scalar/view lowering")
    atomic_calls = re.findall(
        r"call <32 x i32> "
        r"@llvm\.linx\.experimental\.ew\.mgather\.add\.masked[^\n]*",
        ir,
    )
    if len(atomic_calls) != 3:
        raise CheckError("generic atomic CFG lost one of its three masked atomics")
    atomic_roots = [
        match.groups()
        for call in atomic_calls
        for match in [re.search(
            r"ptr (%[-.a-zA-Z0-9]+), <32 x i64> (%[-.a-zA-Z0-9]+),",
            call,
        )]
        if match is not None
    ]
    if (len(atomic_roots) != 3 or atomic_roots[0] != atomic_roots[1] or
            atomic_roots[2][0] == atomic_roots[0][0] or
            atomic_roots[2][1] != atomic_roots[0][1]):
        raise CheckError(
            "all atomics must share one index offset while retaining the "
            "distinct inactive null-base root"
        )
    tlea_calls = re.findall(
        r"call <32 x i64> @llvm\.linx\.experimental\.ew\.tlea[^\n]*"
        r"i64 32\)",
        ir,
    )
    if len(tlea_calls) != 1 or "i64 17" not in tlea_calls[0]:
        raise CheckError("generic atomic CFG lost signed 32-bit byte-index conversion")
    if "<32 x i8>" in ir or "<32 x i16>" in ir:
        raise CheckError("generic atomic index conversion narrowed below B32")
    if len(re.findall(r"@llvm\.linx\.experimental\.ew\.tcmps\.gpr", ir)) < 4:
        raise CheckError("nested atomic CFG lost element/tail/poison predicates")


def validate(text: str, case: str, ir_text: str | None = None) -> None:
    if case == "generic_typed_tile_cfg_i32":
        validate_generic_typed_tile_cfg(text, ir_text)
        return
    if case == "generic_predicated_cfg_i32":
        validate_generic_predicated_cfg(text, ir_text)
        return
    if case == "signed_element_expression":
        validate_signed_expression(text, ir_text)
        return
    if case == "indexed_gather_tile_element":
        validate_indexed_gather(text, ir_text)
        return
    if case == "generic_atomic_cfg_i32":
        validate_generic_atomic(text, ir_text)
        return
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
    if case in ("histogram_tile_element", "selected_radix_tile_element"):
        if ir_text is None:
            raise CheckError("generic atomic migration requires LLVM IR evidence")
        ir = ir_function_body(ir_text, case)
        if re.search(
            r"atomicrmw|extractelement|insertelement|"
            r"@llvm\.linx\.experimental\.element\.(region|view)",
            ir,
        ):
            raise CheckError("generic atomic migration contains residual lowering")
        if len(re.findall(
                r"@llvm\.linx\.experimental\.ew\.mgather\.add\.masked", ir)) != 1:
            raise CheckError("generic atomic migration lost its masked atomic")
    if is_topk:
        if ir_text is None:
            raise CheckError("generic Top-K migration requires LLVM IR evidence")
        if re.search(
            r"atomicrmw|extractelement|insertelement|"
            r"@llvm\.linx\.experimental\.element\.(region|view)",
            ir_text,
        ):
            raise CheckError("generic Top-K migration contains residual lowering")
        if len(re.findall(
                r"call <32 x i32> "
                r"@llvm\.linx\.experimental\.ew\.mgather\.add\.masked",
                ir_text,
        )) != 2:
            raise CheckError("generic Top-K migration lost its two masked atomics")
        if len(re.findall(
                r"call <32 x i64> @llvm\.linx\.experimental\.ew\.tlea[^\n]*"
                r"i64 32\)",
                ir_text,
        )) != 2:
            raise CheckError("generic Top-K migration lost its two B32 TLEA sites")


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
            is_topk = (args.case == "topk_boundaries" or
                       args.case.startswith("topk_boundaries_"))
            if args.case in ("element_expression_chain", "signed_element_expression", "indexed_gather_tile_element", "generic_predicated_cfg_i32", "generic_typed_tile_cfg_i32", "generic_atomic_cfg_i32", "histogram_tile_element", "selected_radix_tile_element"):
                for marker in ("region", "view"):
                    rejected(
                        f"residual element {marker} contract",
                        text,
                        args.case,
                        re.sub(
                            rf"(define[^\n]*{args.case}[^\n]*\n)",
                            rf"\1  call void @llvm.linx.experimental.element.{marker}()\n",
                            ir_text or "",
                            count=1,
                        ),
                    )
            if args.case == "generic_atomic_cfg_i32":
                rejected(
                    "missing one atomic execution mask",
                    text.replace("ExecMaskPresent", "NoExecMask"),
                    args.case,
                    ir_text,
                )
                rejected(
                    "unsigned byte-index conversion",
                    text.replace("TLEA, S32", "TLEA, U32", 1),
                    args.case,
                    ir_text,
                )
                rejected(
                    "residual scalar atomic",
                    text,
                    args.case,
                    re.sub(
                        r"(define[^\n]*generic_atomic_cfg_i32[^\n]*\n)",
                        r"\1  %bad = atomicrmw add ptr null, i32 1 monotonic\n",
                        ir_text or "",
                        count=1,
                    ),
                )
                print(f"{args.case}: disassembly check PASS")
                return 0
            if is_topk:
                for marker in ("region", "view"):
                    rejected(
                        f"residual element {marker} contract",
                        text,
                        args.case,
                        re.sub(
                            r"(define[^\n]*\n)",
                            rf"\1  call void @llvm.linx.experimental.element.{marker}()\n",
                            ir_text or "",
                            count=1,
                        ),
                    )
                rejected(
                    "residual scalar atomic",
                    text,
                    args.case,
                    re.sub(
                        r"(define[^\n]*\n)",
                        r"\1  %bad = atomicrmw add ptr null, i32 1 monotonic\n",
                        ir_text or "",
                        count=1,
                    ),
                )
            if args.case == "generic_typed_tile_cfg_i32":
                rejected(
                    "missing TMOV seed",
                    text.replace("TMOV, S32", "TMOV_REMOVED, S32", 1),
                    args.case,
                    ir_text,
                )
                rejected(
                    "scalar insert fallback",
                    text,
                    args.case,
                    re.sub(
                        r"(define[^\n]*generic_typed_tile_cfg_i32[^\n]*\n)",
                        r"\1  %bad = insertelement <32 x i32> undef, i32 0, i32 0\n",
                        ir_text or "",
                        count=1,
                    ),
                )
                print(f"{args.case}: disassembly check PASS")
                return 0
            if args.case == "generic_predicated_cfg_i32":
                rejected(
                    "unmasked binary substitution",
                    text,
                    args.case,
                    (ir_text or "").replace("tbinary.gpr.masked", "tbinary", 1),
                )
                rejected(
                    "missing masked scatter",
                    text.replace("MSCATTER, U32", "MSCATTER_REMOVED, U32", 1),
                    args.case,
                    ir_text,
                )
                rejected(
                    "scalar extract fallback",
                    text,
                    args.case,
                    re.sub(
                        r"(define[^\n]*nested_three_way[^\n]*\n)",
                        r"\1  %bad = extractelement <32 x i32> undef, i32 0\n",
                        ir_text or "",
                        count=1,
                    ),
                )
                print(f"{args.case}: disassembly check PASS")
                return 0
            if args.case == "signed_element_expression":
                rejected("unsigned load transport", text.replace("TLOAD, S32", "TLOAD, U32", 1), args.case, ir_text)
                rejected("unsigned store transport", text.replace("TSTORE, S32", "TSTORE, U32", 1), args.case, ir_text)
                for mnemonic in ("TLOAD", "TSTORE"):
                    site = next(line for line in text.splitlines()
                                if re.search(rf"BSTART\.TLSU\s+{mnemonic}, S32", line))
                    rejected(f"extra unsigned {mnemonic} site",
                             text.replace(site, site + "\n" + site.replace("S32", "U32"), 1),
                             args.case, ir_text)
                rejected("unsigned division", text.replace("TDIV, S32", "TDIV, U32", 1), args.case, ir_text)
                rejected("direct floor TREM", text.replace("TSUB, U32", "TREM, U32", 1), args.case, ir_text)
                rejected("wrong shift dtype", text.replace("TSHR, U32", "TSHR, S32", 1), args.case, ir_text)
                rejected("wrong remainder dataflow", text, args.case,
                         re.sub(r"(i64 32, i64 1, i64 25, i64 29, i64 )1,",
                                r"\g<1>0,", ir_text or "", count=1))
                print(f"{args.case}: disassembly check PASS")
                return 0
            if args.case == "indexed_gather_tile_element":
                rejected(
                    "missing execution mask",
                    text.replace("ExecMaskPresent", "NoExecMask", 1),
                    args.case,
                    ir_text,
                )
                rejected(
                    "non-M32 gather carrier",
                    text.replace("CUBE_M32", "CUBE_M16"),
                    args.case,
                    ir_text,
                )
                rejected(
                    "wrong TLEA element width",
                    text,
                    args.case,
                    re.sub(
                        r"(@llvm\.linx\.experimental\.ew\.tlea[^\n]*"
                        r"i64 32\))",
                        lambda match: match.group(1).replace("i64 32)",
                                                              "i64 16)"),
                        ir_text or "",
                        count=1,
                    ),
                )
                rejected(
                    "scalar extract fallback",
                    text,
                    args.case,
                    re.sub(
                        r"(define[^\n]*indexed_gather_tile_element[^\n]*\n)",
                        r"\1  %bad = extractelement <32 x i32> undef, i32 0\n",
                        ir_text or "",
                        count=1,
                    ),
                )
                rejected(
                    "atomic opcode substitution",
                    text.replace("MGATHER, U32", "MGATHER.ADD, U32", 1),
                    args.case,
                    ir_text,
                )
                print(f"{args.case}: disassembly check PASS")
                return 0
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
                if len(re.findall(r"call <32 x i32> @llvm\.linx\.experimental\.ew\.tbinary",
                                  ir_text or "")) == 12:
                    rejected(
                        "incorrect optimized unary constant", text, args.case,
                        ((ir_text or "").replace("freeze i32 -1",
                                                 "freeze i32 -2", 1)
                         if "freeze i32 -1" in (ir_text or "")
                         else (ir_text or "").replace(
                             "i64 4294967295, i64 0",
                             "i64 4294967294, i64 0", 1)),
                    )
                    unary_calls = list(re.finditer(
                        r"call <32 x i32> @llvm\.linx\.experimental\.ew\.tbinary[^\n]*",
                        ir_text or "",
                    ))
                    last_call = unary_calls[-1]
                    altered = last_call.group(0).replace("i64 29, i64 0,",
                                                       "i64 29, i64 1,", 1)
                    rejected(
                        "incorrect optimized unary operation", text, args.case,
                        (ir_text or "")[:last_call.start()] + altered +
                        (ir_text or "")[last_call.end():],
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
