#!/usr/bin/env python3
"""Check the native TILE/element-wise/TILE lowering and negative canaries."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path


class CheckError(RuntimeError):
    pass


NATIVE_INSTRUCTION = re.compile(
    r"^\s*[0-9a-f]+:\s+(?:[0-9a-f]{4,8}\s+)+"
    r"([^\s]+)(?:\s+(.*))?$",
    re.I,
)
GPR = re.compile(r"^(?:zero|ra|sp|gp|tp|t|s[0-9]+|a[0-9]+|x[0-9]+)$")


def instruction_lines(text: str) -> list[str]:
    return [
        line
        for line in text.splitlines()
        if re.match(r"^\s*[0-9a-f]+:\s+[0-9a-f]{4,8}\s+", line, re.I)
    ]


def native_instruction(line: str) -> tuple[str, str]:
    match = NATIVE_INSTRUCTION.match(line)
    if match is None:
        raise CheckError(f"malformed native instruction: {line}")
    return match.group(1).upper(), (match.group(2) or "").strip()


def validate_selected_mask_conjunction(body: str) -> str:
    """Prove both selected-radix predicates reach the atomic B.IOR mask."""
    lines = instruction_lines(body)
    atomic = [
        index for index, line in enumerate(lines)
        if native_instruction(line)[0] == "BSTART.TLSU" and
        re.search(r"\bMGATHER\.ADD,\s*U32\b", native_instruction(line)[1])
    ]
    if len(atomic) != 1:
        raise CheckError("selected radix needs exactly one native atomic")
    atomic_index = atomic[0]
    bundle_end = next(
        (index for index in range(atomic_index + 1, len(lines))
         if native_instruction(lines[index])[0].startswith("BSTART.")),
        len(lines),
    )
    binders = [
        (index, native_instruction(lines[index])[1])
        for index in range(atomic_index + 1, bundle_end)
        if native_instruction(lines[index])[0] == "B.IOR" and
        "EXECMASKPRESENT" in native_instruction(lines[index])[1].upper()
    ]
    if len(binders) != 1:
        raise CheckError("selected radix atomic needs one execution-mask binder")
    binder_index, binder_operands = binders[0]
    binder = re.search(r"\[([^,]+),([^,]+),([^\]]+)\]", binder_operands)
    if binder is None:
        raise CheckError("selected radix atomic has a malformed B.IOR")
    _base, mask, unused = (value.strip() for value in binder.groups())
    if not GPR.fullmatch(mask) or unused != "zero":
        raise CheckError("selected radix atomic has an invalid mask binding")

    compare_starts = [
        index for index in range(atomic_index)
        if native_instruction(lines[index])[0] == "BSTART.TEPL" and
        re.search(r"\bTCMPS,\s*U32\b", native_instruction(lines[index])[1])
    ]
    if len(compare_starts) != 2:
        raise CheckError("selected radix needs its tail and digit comparisons")
    compare_writes: dict[int, int] = {}
    for ordinal, start in enumerate(compare_starts):
        end = next(
            (index for index in range(start + 1, atomic_index)
             if native_instruction(lines[index])[0].startswith("BSTART.")),
            atomic_index,
        )
        writes = [
            index for index in range(start + 1, end)
            if native_instruction(lines[index])[0] == "B.IOR" and
            re.search(r"->\s*([^\s,]+)", native_instruction(lines[index])[1])
        ]
        if len(writes) != 1:
            raise CheckError("selected radix comparison lacks one GPR result")
        compare_writes[writes[0]] = ordinal

    provenance: dict[str, frozenset[int]] = {}
    final_and_line: str | None = None
    first_compare = min(compare_writes)
    for index in range(first_compare, binder_index):
        mnemonic, operands = native_instruction(lines[index])
        destination = re.search(r"->\s*([^\s,]+)", operands)
        if destination is None:
            continue
        dst = destination.group(1)
        if not GPR.fullmatch(dst) or dst == "zero":
            continue
        if index in compare_writes:
            provenance[dst] = frozenset({compare_writes[index]})
            continue
        if mnemonic in ("AND", "C.AND"):
            operation = re.fullmatch(
                r"([^,]+),\s*([^,]+),\s*->\s*([^\s,]+)", operands
            )
            if operation is None:
                raise CheckError("selected radix has a malformed scalar AND")
            lhs, rhs, out = (value.strip() for value in operation.groups())
            if not all(GPR.fullmatch(value) for value in (lhs, rhs, out)):
                raise CheckError("selected radix AND does not use plain GPRs")
            provenance[out] = (provenance.get(lhs, frozenset()) |
                               provenance.get(rhs, frozenset()))
            if out == mask:
                final_and_line = lines[index]
            continue
        provenance[dst] = frozenset()

    if provenance.get(mask, frozenset()) != frozenset({0, 1}):
        raise CheckError(
            "selected radix atomic mask does not conjunct both predicates"
        )
    if final_and_line is None:
        raise CheckError("selected radix atomic mask is not produced by AND")
    return final_and_line


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


def function_body_exact(text: str, symbol: str) -> str:
    lines = text.splitlines()
    header = re.compile(r"^[0-9a-f]+ <" + re.escape(symbol) + r">:$")
    start = next(
        (index for index, line in enumerate(lines) if header.match(line)), None)
    if start is None:
        raise CheckError(f"missing function {symbol}")
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


def ir_function_body_exact(text: str, symbol: str) -> str:
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        match = re.match(
            r'^define\b.*@(?:"([^"]+)"|([^\s(]+))\(', line)
        if match and (match.group(1) or match.group(2)) == symbol:
            start = index
            break
    if start is None:
        raise CheckError(f"missing IR function {symbol}")
    end = next(
        (index + 1 for index in range(start + 1, len(lines))
         if lines[index] == "}"),
        None,
    )
    if end is None:
        raise CheckError(f"unterminated IR function {symbol}")
    return "\n".join(lines[start:end])


def matching_function_bodies(text: str, symbol_fragment: str) -> list[tuple[str, str]]:
    names = re.findall(r"^[0-9a-f]+ <([^>]+)>:", text, re.M)
    matches = []
    for name in names:
        if name.startswith(".L") or symbol_fragment not in name:
            continue
        matches.append((name, function_body_exact(text, name)))
    return matches


def native_function_pairs(text: str, ir_text: str | None,
                          symbol_fragment: str,
                          required_patterns: tuple[str, ...]) -> list[tuple[str, str, str]]:
    if ir_text is None:
        raise CheckError(f"{symbol_fragment} requires actual-source IR")
    matches = matching_function_bodies(text, symbol_fragment)
    if not matches:
        raise CheckError(f"missing native {symbol_fragment} function")
    pairs = []
    for name, body in matches:
        missing = [pattern for pattern in required_patterns
                   if not re.search(pattern, body)]
        if missing:
            raise CheckError(
                f"native {name} missing required operation {missing[0]}")
        pairs.append((name, body, ir_function_body_exact(ir_text, name)))
    return pairs


def ir_tci_constants(ir: str) -> dict[str, int]:
    scalar_values = {
        result: int(value)
        for result, value in re.findall(
            r"^\s*(%[-.a-zA-Z0-9]+) = freeze i32 (-?[0-9]+)$", ir, re.M)
    }
    extensions = {
        result: source
        for result, source in re.findall(
            r"^\s*(%[-.a-zA-Z0-9]+) = (?:zext|sext) i32 "
            r"(%[-.a-zA-Z0-9]+) to i64$", ir, re.M)
    }

    def resolve(value: str) -> int | None:
        if not value.startswith("%"):
            return int(value)
        seen = set()
        while value in extensions and value not in seen:
            seen.add(value)
            value = extensions[value]
        return scalar_values.get(value)

    constants = {}
    for result, value in re.findall(
            r"^\s*(%[-.a-zA-Z0-9]+) = call <32 x i32> "
            r"@llvm\.linx\.experimental\.ew\.tci[^\n]*"
            r"\(i64 32, i64 1, i64 (?:17|25), i64 29, i64 "
            r"(%[-.a-zA-Z0-9]+|-?[0-9]+), i64 0\)", ir, re.M):
        constant = resolve(value)
        if constant is not None:
            constants[result] = constant
    return constants


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
    if len(tadds) != 2 or len(tstore) != 1:
        raise CheckError("typed CFG lost its exact TileOp bridge shape")
    # The seed belongs between the second TADDS and TSTORE. Typed queue
    # maintenance may additionally copy the 128-element parent after TSTORE.
    seed_moves = [site for site in tmov if tadds[1] < site < tstore[0]]
    if len(seed_moves) != 1:
        raise CheckError("typed CFG lost its exact raw TMOV seed")
    if not (tload[0] < tadds[0] < tadds[1] < seed_moves[0] < tstore[0]):
        raise CheckError("typed CFG no longer alternates TileOp/element/TileOp in order")
    if not (tload[0] < subview[0] < tstore[0]):
        raise CheckError("typed CFG moved TPARTVIEW outside the Tile lifetime")
    # The generic path implements truncating signed division through absolute
    # values, one masked U32 TDIV, and a sign restore. This avoids PTO's floor
    # remainder/division ambiguity while preserving C++ semantics.
    if sum(bool(re.search(r"\bBSTART\.TEPL\s+TDIV, U32\b", line))
           for line in lines) != 1:
        raise CheckError("typed CFG lost its masked signed division")
    if sum(bool(re.search(r"\bBSTART\.TEPL\s+TXOR, [US]32\b", line))
           for line in lines) != 2:
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
    calls = re.findall(
        r"^\s*(%[-.a-zA-Z0-9]+) = call <32 x i32> "
        r"@llvm\.linx\.experimental\.ew\.tbinary\.gpr\.masked[^\n]*\("
        r"i64 32, i64 1, i64 (17|25), i64 29, i64 ([0-9]+), "
        r"<32 x i32> (%[-.a-zA-Z0-9]+), <32 x i32> (%[-.a-zA-Z0-9]+)"
        r"[^)\n]*\)", ir, re.M)
    tci_constants = ir_tci_constants(ir)
    xor_constants = []
    xor_sources = []
    for _result, _dtype, op, lhs, rhs in calls:
        if op != "7":
            continue
        constant_operands = [operand for operand in (lhs, rhs)
                             if operand in tci_constants]
        source_operands = [operand for operand in (lhs, rhs)
                           if operand not in tci_constants]
        if len(constant_operands) != 1 or len(source_operands) != 1:
            raise CheckError("typed CFG XOR lost its source/constant SSA pairing")
        xor_constants.append(tci_constants[constant_operands[0]])
        xor_sources.append(source_operands[0])
    if sorted(xor_constants) != [5, 85] or len(set(xor_sources)) != 2:
        raise CheckError("typed CFG XOR lost source constants 5/85 or SSA inputs")


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
        ("TADD", "S32", 1), ("TADD", "U32", 1),
        ("TMUL", "U32", 2), ("TDIV", "S32", 2),
        ("TSUB", "S32", 1), ("TAND", "U32", 1),
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
    if sum("ExecMaskPresent" in line for line in lines) != 10:
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
        ("25", "0"): 1,
        ("17", "0"): 1,
        ("17", "1"): 1,
        ("25", "2"): 2,
        ("17", "3"): 2,
        ("25", "5"): 1,
        ("25", "7"): 1,
        ("25", "9"): 1,
    })
    if Counter((dtype, op) for _, dtype, op, _, _ in calls) != expected_operations:
        raise CheckError("signed expression IR lost exact operation dtype/shape")
    operations = {result: (dtype, op, lhs, rhs) for result, dtype, op, lhs, rhs in calls}
    tci_constants = ir_tci_constants(ir)

    def split_constant(lhs: str, rhs: str, value: int) -> str | None:
        if tci_constants.get(lhs) == value and rhs not in tci_constants:
            return rhs
        if tci_constants.get(rhs) == value and lhs not in tci_constants:
            return lhs
        return None

    canonical_arithmetic = False
    for multiply, (dtype, op, lhs, rhs) in operations.items():
        if dtype != "25" or op != "2":
            continue
        if split_constant(lhs, rhs, 3) is None:
            continue
        for added, (add_dtype, add_op, add_lhs, add_rhs) in operations.items():
            if (add_dtype != "25" or add_op != "0" or
                    multiply not in (add_lhs, add_rhs)):
                continue
            other = add_rhs if add_lhs == multiply else add_lhs
            if tci_constants.get(other) != 50:
                continue
            canonical_arithmetic |= any(
                div_dtype == "17" and div_op == "3" and div_lhs == added
                for div_dtype, div_op, div_lhs, _div_rhs in operations.values())
    if not canonical_arithmetic:
        raise CheckError("signed arithmetic lost canonical input * 3 + 50 division path")

    proves_remainder = False
    for _, dtype, op, lhs, rhs in calls:
        if dtype != "17" or op != "1" or rhs not in operations:
            continue
        mdtype, multiply, quotient, divisor = operations[rhs]
        if mdtype != "25" or multiply != "2" or quotient not in operations:
            continue
        qdtype, divide, dividend, qdivisor = operations[quotient]
        proves_remainder |= (qdtype == "17" and divide == "3" and
                             dividend == lhs and qdivisor == divisor)
    if not proves_remainder:
        raise CheckError("signed % lost exact dividend - trunc_quotient * divisor dataflow")

    proves_shifted_negation = False
    for shifted, (dtype, op, lhs, rhs) in operations.items():
        if (dtype != "25" or op != "9" or
                lhs in tci_constants or tci_constants.get(rhs) != 2):
            continue
        for masked, (and_dtype, and_op, and_lhs, and_rhs) in operations.items():
            if (and_dtype != "25" or and_op != "5" or
                    shifted not in (and_lhs, and_rhs)):
                continue
            other = and_rhs if and_lhs == shifted else and_lhs
            if tci_constants.get(other) != 255:
                continue
            for inverted, (xor_dtype, xor_op, xor_lhs, xor_rhs) in operations.items():
                if (xor_dtype != "25" or xor_op != "7" or
                        masked not in (xor_lhs, xor_rhs)):
                    continue
                other = xor_rhs if xor_lhs == masked else xor_lhs
                if tci_constants.get(other) != -86:
                    continue
                proves_shifted_negation |= any(
                    add_dtype == "17" and add_op == "0" and
                    inverted in (add_lhs, add_rhs) and
                    tci_constants.get(
                        add_rhs if add_lhs == inverted else add_lhs) == 1
                    for add_dtype, add_op, add_lhs, add_rhs
                    in operations.values())
    if not proves_shifted_negation:
        raise CheckError("signed bit expression lost ((x >> 2) & 255) ^ 85 negation")


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


def validate_group_count(text: str, ir_text: str | None) -> None:
    body = function_body(text, "group_countILi16")
    lines = instruction_lines(body)

    def sites(pattern: str) -> list[int]:
        return [index for index, line in enumerate(lines) if re.search(pattern, line)]

    loads = sites(r"\bBSTART\.TLSU\s+TLOAD, U32\b")
    adds = sites(r"\bBSTART\.TEPL\s+TADDS, U32\b")
    tleas = sites(r"\bBSTART\.TEPL\s+TLEA, U32\b")
    atomics = sites(r"\bBSTART\.TLSU\s+MGATHER\.ADD, (S32|U32)\b")
    if len(loads) != 1 or len(adds) != 1:
        raise CheckError("group_count lost its snapshot or TileOp bridges")
    if len(tleas) != 1 or len(atomics) != 1:
        raise CheckError("group_count needs one B32-to-B64 stream and one atomic site")
    if not (loads[0] < adds[0] < atomics[0]):
        raise CheckError("group_count no longer materializes indices before atomics")
    if not sites(r"\bB\.SUBVIEW\b") or "CUBE_M32" not in body:
        raise CheckError("group_count lost public TPARTVIEW/M32 transport")
    atomic_bundle = re.search(
        r"BSTART\.TLSU\s+MGATHER\.ADD, (?:S32|U32).*?(?:BSTART|$)", body, re.S
    )
    if atomic_bundle is None or "ExecMaskPresent" not in atomic_bundle.group(0):
        raise CheckError("group_count atomic lost its range predicate mask")
    if "TCVT" in body:
        raise CheckError("integer-only group_count unexpectedly emitted TCVT")

    if ir_text is None:
        raise CheckError("group_count requires LLVM IR evidence")
    ir = ir_function_body(ir_text, "group_countILi16")
    if re.search(
        r"atomicrmw|extractelement|insertelement|"
        r"@llvm\.linx\.experimental\.element\.(region|view)|"
        r"@llvm\.linx\.experimental\.ew\.tcvt",
        ir,
    ):
        raise CheckError("group_count contains residual scalar/view/TCVT lowering")
    atomic_calls = re.findall(
        r"call <32 x i32> "
        r"@llvm\.linx\.experimental\.ew\.mgather\.add\.masked[^\n]*",
        ir,
    )
    if len(atomic_calls) != 1:
        raise CheckError("group_count lost its single masked atomic intrinsic")
    load_position = ir.find("BSTART.TLSU TLOAD")
    reset = re.search(
        r"store i32 0, ptr [^\n]+|"
        r"call void @llvm\.memset[^\n]*i8 0, i64 128,",
        ir,
    )
    atomic_position = ir.find("@llvm.linx.experimental.ew.mgather.add.masked")
    if (load_position < 0 or reset is None or atomic_position < 0 or
            not (load_position < reset.start() < atomic_position)):
        raise CheckError("group_count must snapshot before its in-kernel reset")
    tlea_calls = re.findall(
        r"call <32 x i64> @llvm\.linx\.experimental\.ew\.tlea[^\n]*"
        r"i64 32\)",
        ir,
    )
    if len(tlea_calls) != 1 or not re.search(
        r"i64 32, i64 1, i64 25, i64 29, <32 x i32>", tlea_calls[0]
    ):
        raise CheckError("group_count lost unsigned B32-to-B64 byte indexing")
    if "<32 x i8>" in ir or "<32 x i16>" in ir:
        raise CheckError("group_count narrowed its index below B32")
    bound = re.search(
        r"(%[\w.]+) = freeze i32 32\s+(%[\w.]+) = zext i32 \1 to i64", ir
    )
    compares = re.findall(
        r"call i64 @llvm\.linx\.experimental\.ew\.tcmps\.gpr[^\n]*", ir
    )
    if (bound is None or len(compares) != 1 or not re.search(
        r"i64 32, i64 1, i64 25, i64 29, <32 x i32> [^,]+, i64 "
        + re.escape(bound.group(2)) + r", i64 2\)", compares[0]
    )):
        raise CheckError("group_count lost its unsigned index < 32 range predicate")
    if "@llvm.linx.experimental.ew.tbinary" in ir:
        raise CheckError("group_count introduced an unnecessary index dtype conversion")


def validate_original_atomic(text: str, ir_text: str | None) -> None:
    if ir_text is None:
        raise CheckError("original atomic histogram requires LLVM IR evidence")
    body = function_body(text, "elementwise_atomic_histogram")
    ir = ir_function_body(ir_text, "elementwise_atomic_histogram")
    if re.search(r"atomicrmw|extractelement|insertelement|llvm\.vp\.|"
                 r"llvm\.linx\.experimental\.element\.(region|view)", ir):
        raise CheckError("original atomic histogram has residual lowering")
    atomics = re.findall(r"call <32 x i32> @llvm\.linx\.experimental\.ew\.mgather\.add\.masked[^\n]*", ir)
    reads = re.findall(r"call <32 x i32> @llvm\.linx\.experimental\.ew\.mgather\.gpr\.masked[^\n]*", ir)
    offsets = re.findall(r"call <32 x i64> @llvm\.linx\.experimental\.ew\.tlea[^\n]*", ir)
    if len(atomics) != 1 or len(reads) != 1 or len(offsets) != 2:
        raise CheckError("original histogram lost its two ordered memory phases")
    if ir.index(atomics[0]) >= ir.index(reads[0]):
        raise CheckError("histogram gather precedes atomic phase")
    if not all("i64 17, i64 29, <32 x i32>" in c and c.endswith("i64 32)") for c in offsets):
        raise CheckError("original signed B32 indices must produce B64 byte offsets")
    if not atomics[0].endswith("i64 0, i64 0, i64 1)") or not reads[0].endswith("i64 0, i64 0, i64 1)"):
        raise CheckError("original histogram lost defined inactive memory results")
    if len(re.findall(r"BSTART\.TLSU\s+MGATHER\.ADD, U32", body)) != 1:
        raise CheckError("original histogram needs native atomic addition")
    if len(re.findall(r"BSTART\.TLSU\s+MGATHER, U32", body)) != 1:
        raise CheckError("original histogram needs native indexed gather")
    native = "\n".join(instruction_lines(body))
    memory_bundles = re.findall(r"BSTART\.TLSU\s+MGATHER(?:\.ADD)?, U32.*?(?=BSTART|$)", native, re.S)
    if len(memory_bundles) != 2 or not all("ExecMaskPresent" in b for b in memory_bundles):
        raise CheckError("both original memory phases require execution masks")
    if len(re.findall(r"BSTART\.TEPL\s+TLEA, S32", native)) != 2:
        raise CheckError("original histogram lost signed B32 native addressing")
    if not all(op in body for op in ("TLOAD", "B.SUBVIEW", "TSTORE")):
        raise CheckError("original histogram lost official Tile transport")


def validate_concat_original(text: str, ir_text: str | None) -> None:
    if ir_text is None:
        raise CheckError("original concat requires LLVM IR evidence")
    body = function_body(text, "concat_gather")
    ir = ir_function_body(ir_text, "concat_gather")
    if re.search(r"extractelement|insertelement|freeze <32 x|llvm\.vp\.|"
                 r"llvm\.linx\.experimental\.element\.(region|view)", ir):
        raise CheckError("concat contains residual generic lowering")
    reads = re.findall(r"call <32 x i32> @llvm\.linx\.experimental\.ew\.mgather\.gpr\.masked[^\n]*", ir)
    offsets = re.findall(r"call <32 x i64> @llvm\.linx\.experimental\.ew\.tlea[^\n]*", ir)
    if len(reads) != 1 or len(offsets) != 1 or "i64 25, i64 29, <32 x i32>" not in offsets[0]:
        raise CheckError("concat needs a native B32-indexed readonly gather")
    if not offsets[0].endswith("i64 32)") or not reads[0].endswith("i64 0, i64 0, i64 1)"):
        raise CheckError("concat lost B64 byte offsets or defined memory results")
    if not re.search(r"<512 x i32>[^\n]*i64 64, i32 17, i32 16, i32 32", ir):
        raise CheckError("concat lost its whole 512-element S32 store")
    native = "\n".join(instruction_lines(body))
    if len(re.findall(r"BSTART\.TLSU\s+MGATHER, U32", native)) != 1:
        raise CheckError("concat lost its indexed read")
    if len(re.findall(r"BSTART\.TLSU\s+TSTORE, S32", native)) != 1:
        raise CheckError("concat must publish one complete batch")
    if "M322ND" not in native or "B.ASSEMBLE" not in native:
        raise CheckError("concat lost zero-copy assembly or parent transport")
    if not re.search(r"B\.ASSEMBLE\s+1, 0,", native) or not re.search(r"B\.ASSEMBLE\s+0, 1,", native):
        raise CheckError("concat lost INIT/LAST publication")
    if "<2KB>" not in native:
        raise CheckError("concat parent allocation no longer preserves 512 elements")


def validate_concat_f16_original(text: str, ir_text: str | None) -> None:
    if ir_text is None:
        raise CheckError("original F16 concat requires LLVM IR evidence")
    body = function_body(text, "concat_gather")
    ir = ir_function_body(ir_text, "concat_gather")
    if re.search(r"extractelement|insertelement|freeze <64 x half>|llvm\.vp\.|"
                 r"llvm\.linx\.experimental\.element\.(region|view)|"
                 r"\b(fpext|fptrunc|sitofp|uitofp|fptosi|fptoui)\b|"
                 r"llvm\.linx\.experimental\.ew\.tcvt", ir):
        raise CheckError("F16 concat contains residual or numeric conversion lowering")
    reads = re.findall(
        r"call <64 x half> "
        r"@llvm\.linx\.experimental\.ew\.mgather\.gpr\.masked\.v64f16\.v64i64[^\n]*",
        ir,
    )
    offsets = re.findall(
        r"call <64 x i64> @llvm\.linx\.experimental\.ew\.tlea[^\n]*",
        ir,
    )
    if len(reads) != 1 or len(offsets) != 1:
        raise CheckError("F16 concat needs one native 64-element readonly gather")
    if not re.search(r"i64 32, i64 2, i64 4, i64 0, i64 29, i64 24,",
                     reads[0]):
        raise CheckError("F16 concat lost its FP16/M32/U64 gather ABI")
    if not re.search(r"i64 32, i64 2, i64 25, i64 29, <64 x i32>", offsets[0]) or not offsets[0].endswith("i64 16)"):
        raise CheckError("F16 concat lost its U32-to-B64 offset shape or 16-bit element scaling")
    if not re.search(r"BSTART\.TLSU TSTORE[^\n]*<512 x half>", ir):
        raise CheckError("F16 concat lost its whole 512-element raw store")

    native = "\n".join(instruction_lines(body))
    if len(re.findall(r"BSTART\.TLSU\s+MGATHER, FP16", native)) != 1:
        raise CheckError("F16 concat lost its indexed read")
    if len(re.findall(r"BSTART\.TLSU\s+TMOV, FP16", native)) != 3:
        raise CheckError("F16 concat needs raw TMOV for INIT, MID and LAST")
    if len(re.findall(r"BSTART\.TLSU\s+TSTORE, FP16", native)) != 1:
        raise CheckError("F16 concat must publish one complete batch")
    if re.search(r"BSTART\.TEPL\s+TCVT, FP16", native):
        raise CheckError("F16 concat introduced a numeric identity conversion")
    if "B.ASSEMBLE" not in native:
        raise CheckError("F16 concat lost zero-copy parent assembly")
    if not re.search(r"B\.ASSEMBLE\s+1, 0,", native) or not re.search(
            r"B\.ASSEMBLE\s+0, 1,", native):
        raise CheckError("F16 concat lost INIT/LAST publication")
    if "<1KB>" not in native:
        raise CheckError("F16 concat parent allocation no longer preserves 512 elements")


def without_parent_queue_copies(body: str, terminal: str,
                                      parent_types: dict[str, str]) -> str:
    """Validate allocator copies separately from the source-level Tile bridge."""
    terminal_sites = list(re.finditer(
        rf"BSTART\.TLSU\s+{terminal}, FP32\b", body))
    seen: set[str] = set()

    def keep_semantic_copy(match: re.Match[str]) -> str:
        bundle = match.group(0)
        if "<512B>" not in bundle:
            return bundle
        binding = re.search(
            r"B\.IOT\s+([tu])#\d+, mask=1111, last,\s+->([tu])<512B>",
            bundle)
        dtype = match.group(1)
        if (not terminal_sites or match.start() < terminal_sites[-1].start() or
                not binding or binding[1] != binding[2] or
                parent_types.get(binding[1]) != dtype or binding[1] in seen):
            raise CheckError("FP32 parent queue copy lost its dtype, identity or placement")
        if (not re.search(r"B\.DATR\s+CUBE_M32\.normal, Zero\b", bundle) or
                any(token in bundle for token in
                    ("B.SUBVIEW", "B.ASSEMBLE", "B.IOR", "ExecMaskPresent")) or
                len(re.findall(r"\bB\.DATR\b", bundle)) != 1 or
                len(re.findall(r"\bB\.IOT\b", bundle)) != 1 or
                len(re.findall(r"\bB\.DIMI?\b", bundle)) != 3 or
                sorted(re.findall(r"C\.B\.DIMI\s+(\d+),\s+->lb([012])\b", bundle)) !=
                sorted([("4", "0"), ("32", "1"), ("4", "2")])):
            raise CheckError("FP32 parent queue copy changed its complete 512B descriptor")
        seen.add(binding[1])
        return ""

    return re.sub(r"BSTART\.TLSU\s+TMOV, (\w+)\b.*?(?=BSTART|\Z)",
                  keep_semantic_copy, body, flags=re.S)


def validate_original_f32(text: str, ir_text: str | None) -> None:
    body = function_body(text, "elementwise_if_for")
    body = without_parent_queue_copies(
        body, "TSTORE", {"t": "FP32", "u": "FP32"})
    lines = instruction_lines(body)
    transport = [re.search(r"BSTART\.TLSU\s+(TLOAD|TMOV|TSTORE), FP32\b", line)
                 for line in lines]
    transport = [match.group(1) for match in transport if match]
    if transport != ["TLOAD", "TLOAD", "TMOV", "TMOV", "TSTORE"]:
        raise CheckError("FP32 original lost its exact TLOAD128/part-TMOV/TSTORE bridge")
    if len(re.findall(r"\bB\.SUBVIEW\b", body)) != 2:
        raise CheckError("FP32 original needs two raw logical part views")
    if any(token not in body for token in ("ND2M32", "M322ND", "CUBE_M32")):
        raise CheckError("FP32 original lost native typed transport")
    if re.search(r"\bTCVT\b|BSTART\.TEPL\s+(TADDS|TSUBS), FP32\b", body):
        raise CheckError("FP32 original introduced a numeric identity conversion")
    # Inspect the mask of each actual operation's bundle, including FNEG when
    # ordinary LLVM folds a +/- diamond into a sign select followed by FADD.
    bundles = re.findall(r"BSTART\.TEPL\s+(TCMPS?|TADD|TSUB|TNEG), FP32\b"
                         r"(.*?)(?=BSTART|\Z)", body, re.S)
    counts = Counter(op for op, _bundle in bundles)
    folded = counts == Counter({"TCMPS": 1, "TNEG": 1, "TADD": 1})
    diamond = counts == Counter({"TCMPS": 1, "TADD": 1, "TSUB": 1})
    if not (folded or diamond):
        raise CheckError("FP32 original lost comparison or native add/subtract semantics")
    for op, bundle in bundles:
        if "ExecMaskPresent" not in bundle:
            raise CheckError(f"FP32 original {op} lost its own execution mask")
        if not re.search(r"C\.B\.DIMI\s+32,\s+->lb1", bundle):
            raise CheckError(f"FP32 original {op} changed its 32-element domain")
        if op == "TCMPS" and not re.search(
                r"B\.DATR\s+NORM, DTYPE_NONE, Zero, byte0, GT, RNONE, nosat, 0, 1", bundle):
            raise CheckError("FP32 original changed its native ordered comparison")
    if len(re.findall(r"BSTART\.TEPL\s+TSEL, FP32\b", body)) != 1:
        raise CheckError("FP32 original lost its raw-bit value merge")
    if ir_text is None:
        raise CheckError("FP32 original requires actual-source LLVM IR")
    ir = ir_function_body(ir_text, "elementwise_if_for")
    if re.search(r"extractelement|insertelement|@llvm\.vp\.|"
                 r"@llvm\.linx\.experimental\.element\.(region|view)|"
                 r"\b(fadd|fsub|fcmp|fneg|fpext|fptrunc|sitofp|uitofp|fptosi|fptoui)\b|"
                 r"@llvm\.linx\.experimental\.ew\.tcvt", ir):
        raise CheckError("FP32 original has residual scalar/region/conversion lowering")
    definitions = dict(re.findall(r"^\s*(%[\w.]+) = ([^\n]+)", ir, re.M))

    def calls(stem: str) -> list[tuple[str, str]]:
        return [(value, expr) for value, expr in definitions.items()
                if "@llvm.linx.experimental.ew." + stem + ".v32f32(" in expr]

    compares = calls("tcmps.gpr.masked")
    binary = calls("tbinary.gpr.masked")
    negates = calls("tneg.gpr.masked")
    selects = calls("tsel.gpr")
    if len(compares) != 1 or len(selects) != 1:
        raise CheckError("FP32 original lost its one comparison/one merge graph")
    compare_value, compare_expr = compares[0]
    compare = re.search(r"i64 32, i64 1, i64 1, i64 29, <32 x float> (%[\w.]+), "
                        r"i64 (%[\w.]+|[0-9]+), i64 3, i64 (%[\w.]+|[0-9]+)\)", compare_expr)
    if compare is None:
        raise CheckError("FP32 original lost its masked ordered >0 comparison")
    lhs, zero_bits, compare_mask = compare.groups()
    # The scalar comparison constant must carry +0's bits, not a conversion.
    if zero_bits != "0":
        raw = re.fullmatch(r"zext i32 (%[\w.]+) to i64", definitions.get(zero_bits, ""))
        bits = re.fullmatch(r"bitcast float (%[\w.]+) to i32", definitions.get(raw[1], "")) if raw else None
        if not bits or definitions.get(bits[1]) != "freeze float 0.000000e+00":
            raise CheckError("FP32 original changed the source comparison constant")

    def mask(value: str, positive: bool, visiting: frozenset[str] = frozenset()) -> int:
        if value == compare_value:
            return 0xFFFFFFFF if positive else 0
        if not value.startswith("%"):
            return int(value) & 0xFFFFFFFFFFFFFFFF
        if value in visiting:
            raise CheckError("FP32 original has a cyclic predicate expression")
        expr = definitions.get(value, "")
        child_visiting = visiting | {value}
        frozen = re.fullmatch(r"freeze i1 (true|false)", expr)
        if frozen:
            return int(frozen[1] == "true")
        selected = re.fullmatch(r"select i1 (%[\w.]+), i64 ([0-9]+), i64 ([0-9]+)", expr)
        if selected:
            return int(selected[2] if mask(selected[1], positive, child_visiting) else selected[3])
        binary_mask = re.fullmatch(r"(and|or|xor) i64 (%[\w.]+|[0-9]+), (%[\w.]+|[0-9]+)", expr)
        if binary_mask:
            op, left, right = binary_mask.groups()
            a = mask(left, positive, child_visiting)
            b = mask(right, positive, child_visiting)
            return {"and": a & b, "or": a | b, "xor": a ^ b}[op]
        raise CheckError("FP32 original has an unproved operation mask")

    def require_mask(value: str, expected: tuple[int, int]) -> None:
        if (mask(value, False), mask(value, True)) != expected:
            raise CheckError("FP32 original changed an operation's active domain")

    full = (0xFFFFFFFF, 0xFFFFFFFF)
    require_mask(compare_mask, full)
    select_value, select_expr = selects[0]
    selected = re.search(r"i64 32, i64 1, i64 1, i64 29, i64 (%[\w.]+|[0-9]+), "
                         r"i64 0, <32 x float> (%[\w.]+), <32 x float> (%[\w.]+)\)", select_expr)
    if not selected:
        raise CheckError("FP32 original lost F32 raw-bit merge typing")
    predicate, true_value, false_value = selected.groups()
    require_mask(predicate, (0, 0xFFFFFFFF))
    binary_parts = []
    for value, expr in binary:
        operation = re.search(r"i64 32, i64 1, i64 1, i64 29, i64 ([01]), "
                              r"<32 x float> (%[\w.]+), <32 x float> (%[\w.]+), "
                              r"i64 (%[\w.]+|[0-9]+), i64 0, i64 0, i64 1\)", expr)
        if not operation:
            raise CheckError("FP32 original lost native FP32 binary typing")
        binary_parts.append((value, *operation.groups()))
    if folded:
        if len(binary_parts) != 1 or len(negates) != 1:
            raise CheckError("FP32 folded expression lost its exact TNEG/TSEL/TADD graph")
        negate_value, negate_expr = negates[0]
        negate = re.search(r"i64 32, i64 1, i64 1, i64 29, <32 x float> (%[\w.]+), "
                           r"i64 (%[\w.]+|[0-9]+)\)", negate_expr)
        result_value, opcode, add_lhs, add_rhs, add_mask = binary_parts[0]
        if (not negate or true_value != negate[1] or false_value != negate_value or
                opcode != "0" or (add_lhs, add_rhs) != (lhs, select_value)):
            raise CheckError("FP32 folded expression changed negate/select/add dataflow")
        rhs = negate[1]
        require_mask(negate[2], full)
        require_mask(add_mask, full)
    else:
        if len(binary_parts) != 2 or negates:
            raise CheckError("FP32 diamond lost its exact FADD/FSUB graph")
        by_opcode = {operation[1]: operation for operation in binary_parts}
        if set(by_opcode) != {"0", "1"}:
            raise CheckError("FP32 diamond lost one arithmetic arm")
        summed, subtracted = by_opcode["0"], by_opcode["1"]
        if (true_value, false_value) != (summed[0], subtracted[0]) or summed[2:4] != subtracted[2:4] or summed[2] != lhs:
            raise CheckError("FP32 diamond changed add/subtract/merge dataflow")
        rhs = summed[3]
        require_mask(summed[4], (0, 0xFFFFFFFF))
        require_mask(subtracted[4], (0xFFFFFFFF, 0))
        result_value = select_value
    # Both operands must be the actual corresponding raw Tile partition copies.
    for value, source in ((lhs, "%lhs"), (rhs, "%rhs")):
        transport_expr = definitions.get(value, "")
        parent = re.search(r'asm sideeffect "BSTART\.TLSU TMOV,.*\(<128 x float> (%[\w.]+),', transport_expr)
        if not parent or not re.search(r'asm sideeffect "BSTART\.TLSU TLOAD,.*\(ptr ' + re.escape(source) + ',', definitions.get(parent[1], "")):
            raise CheckError("FP32 original substituted a source Tile partition")
    # TSTORE consumes the computed result through its standard LCSSA publication.
    stores = re.findall(r'call void asm sideeffect "BSTART\.TLSU TSTORE,[^\n]*', ir)
    if len(stores) != 1:
        raise CheckError("FP32 original lost its one Tile store")
    stored = re.search(r'<32 x float> (%[\w.]+), i64 16, i32 1, i32 1, i32 32\)', stores[0])
    if not stored:
        raise CheckError("FP32 original changed its Tile store shape")
    publication = definitions.get(stored[1], "")
    if stored[1] != result_value and not re.search(r'phi <32 x float> \[ ' + re.escape(result_value) + r',', publication):
        raise CheckError("FP32 original store no longer consumes its computed result")


def validate_masked_gather_scatter_f32(text: str,
                                       ir_text: str | None) -> None:
    body = function_body(text, "elementwise_masked_gather_scatter")
    body = without_parent_queue_copies(
        body, "MSCATTER", {"t": "S32", "u": "U32"})
    lines = instruction_lines(body)

    def sites(pattern: str) -> list[int]:
        return [index for index, line in enumerate(lines)
                if re.search(pattern, line)]

    signed_loads = sites(r"\bBSTART\.TLSU\s+TLOAD, S32\b")
    mask_loads = sites(r"\bBSTART\.TLSU\s+TLOAD, U32\b")
    signed_moves = sites(r"\bBSTART\.TLSU\s+TMOV, S32\b")
    mask_moves = sites(r"\bBSTART\.TLSU\s+TMOV, U32\b")
    raw_moves = sites(r"\bBSTART\.TLSU\s+TMOV, FP32\b")
    tleas = sites(r"\bBSTART\.TEPL\s+TLEA, S32\b")
    gathers = sites(r"\bBSTART\.TLSU\s+MGATHER, FP32\b")
    scatters = sites(r"\bBSTART\.TLSU\s+MSCATTER, FP32\b")
    if list(map(len, (signed_loads, mask_loads, signed_moves, mask_moves,
                      raw_moves, tleas, gathers, scatters))) != [1, 1, 1, 1,
                                                                  1, 2, 1, 1]:
        raise CheckError("FP32 masked memory lost its exact Tile/memory bridge")
    if len(re.findall(r"\bB\.SUBVIEW\b", body)) != 2:
        raise CheckError("FP32 masked memory needs two borrowed logical part views")
    if not (signed_loads[0] < signed_moves[0] < tleas[0] < gathers[0] <
            raw_moves[0] < tleas[1] < scatters[0]):
        raise CheckError("FP32 gather/TMOV/scatter source order changed")
    if re.search(r"\bTCVT\b|BSTART\.TEPL\s+(TADDS|TSUBS), FP32\b", body):
        raise CheckError("FP32 masked memory introduced numeric value transport")
    if any(token not in body for token in ("ND2M32", "CUBE_M32")):
        raise CheckError("FP32 masked memory lost typed M32 transport")
    native = "\n".join(lines)
    memory_bundles = re.findall(
        r"BSTART\.TLSU\s+(MGATHER|MSCATTER), FP32\b.*?(?=BSTART|$)",
        native, re.S,
    )
    # Inspect full bundles separately because findall above returns the opcode.
    full_bundles = re.findall(
        r"BSTART\.TLSU\s+(?:MGATHER|MSCATTER), FP32\b.*?(?=BSTART|$)",
        native, re.S,
    )
    if len(memory_bundles) != 2 or any(
            "ExecMaskPresent" not in bundle for bundle in full_bundles):
        raise CheckError("FP32 gather and scatter each require an execution mask")

    if ir_text is None:
        raise CheckError("FP32 masked memory requires actual-source LLVM IR")
    ir = ir_function_body(ir_text, "elementwise_masked_gather_scatter")
    if re.search(r"extractelement|insertelement|@llvm\.vp\.|"
                 r"@llvm\.linx\.experimental\.element\.(region|view|scatter)|"
                 r"\b(fadd|fsub|fmul|fdiv|fneg|fpext|fptrunc|sitofp|uitofp|fptosi|fptoui)\b|"
                 r"@llvm\.linx\.experimental\.ew\.tcvt", ir):
        raise CheckError("FP32 masked memory has residual scalar/region/conversion lowering")
    tlea_calls = re.findall(
        r"call <32 x i64> @llvm\.linx\.experimental\.ew\.tlea[^\n]*",
        ir,
    )
    if len(tlea_calls) != 2 or any(
            not re.search(r"i64 32, i64 1, i64 17, i64 29, "
                          r"<32 x i32> .*i64 32\)", call)
            for call in tlea_calls):
        raise CheckError("FP32 masked memory lost signed B32 to B64 addressing")
    gather_calls = re.findall(
        r"call <32 x float> "
        r"@llvm\.linx\.experimental\.ew\.mgather\.gpr\.masked[^\n]*", ir,
    )
    scatter_calls = re.findall(
        r"call void "
        r"@llvm\.linx\.experimental\.ew\.mscatter\.gpr\.masked[^\n]*", ir,
    )
    if len(gather_calls) != 1 or len(scatter_calls) != 1:
        raise CheckError("FP32 masked memory needs one native gather and scatter")
    if not re.search(r"i64 32, i64 1, i64 1, i64 0, i64 29, i64 16,",
                     gather_calls[0]):
        raise CheckError("FP32 gather lost F32/M32/S64 typing")
    if not re.search(r"i64 32, i64 1, i64 1, i64 29, i64 16,",
                     scatter_calls[0]):
        raise CheckError("FP32 scatter lost F32/M32/S64 typing")
    raw_transport = re.findall(
        r'call <32 x float> asm sideeffect "BSTART\.TLSU TMOV,[^\n]*', ir,
    )
    if len(raw_transport) != 1:
        raise CheckError("FP32 gathered payload needs one raw official TMOV")


def validate_concat_scatter(text: str, case: str, ir_text: str | None) -> None:
    candidates = re.findall(r"^[0-9a-f]+ <([^>]*concat_scatter[^>]*)>:", text, re.M)
    bodies = [(name, function_body(text, name)) for name in candidates]
    bodies = [(name, body) for name, body in bodies
              if re.search(r"BSTART\.TLSU\s+TLOAD,", body) and
                 re.search(r"BSTART\.TLSU\s+MSCATTER,", body)]
    if len(bodies) != 1:
        raise CheckError("concat scatter needs one native snapshot/scatter function")
    symbol, body = bodies[0]
    fp16 = "_f16_" in case
    fourpe = case == "concat_scatter_fourpe"
    dtype, lanes = ("FP16", 64) if fp16 else (("FP32", 32) if fourpe else ("S32", 32))
    loads = list(re.finditer(rf"BSTART\.TLSU\s+TLOAD, {dtype}\b", body))
    stores = list(re.finditer(rf"BSTART\.TLSU\s+MSCATTER, {dtype}\b", body))
    parts = [match for match in re.finditer(
        rf"BSTART\.TLSU\s+TMOV, {dtype}\b.*?(?=BSTART|\Z)", body, re.S)
        if re.search(r"\bB\.SUBVIEW\b", match.group(0))]
    if (len(loads) != 1 or len(stores) != 1 or len(parts) != 1 or
            not loads[0].start() < parts[0].start() < stores[0].start()):
        raise CheckError("concat scatter lost full-input snapshot before part stores")
    load_bundle = re.search(rf"BSTART\.TLSU\s+TLOAD, {dtype}\b.*?(?=BSTART|\Z)",
                            body, re.S).group(0)
    capacity = "<512B>" if fourpe else ("<1KB>" if fp16 else "<2KB>")
    if capacity not in load_bundle:
        raise CheckError("concat scatter changed its original512 snapshot capacity")
    if not re.search(r"BSTART\.TEPL\s+TLEA, U32\b", body):
        raise CheckError("concat scatter lost B32 indices to B64 byte addressing")
    store_bundle = re.search(rf"BSTART\.TLSU\s+MSCATTER, {dtype}\b.*?(?=BSTART|\Z)",
                             body, re.S).group(0)
    if "ExecMaskPresent" not in store_bundle or "TCVT" in body:
        raise CheckError("concat scatter lost execution mask or raw value transport")
    if ir_text is None:
        raise CheckError("concat scatter requires actual-source IR")
    ir = ir_function_body(ir_text, symbol)
    if re.search(r"extractelement|insertelement|@llvm\.vp\.|"
                 r"@llvm\.linx\.experimental\.element\.(region|view|scatter)", ir):
        raise CheckError("concat scatter retains scalar or intermediate element effects")
    payload = "half" if fp16 else ("float" if fourpe else "i32")
    scatters = re.findall(r"call void @llvm\.linx\.experimental\.ew\.mscatter\.gpr\.masked[^\n]*", ir)
    if len(scatters) != 1 or f"<{lanes} x {payload}>" not in scatters[0]:
        raise CheckError("concat scatter changed its typed element payload")
    if f"<{lanes} x i64>" not in scatters[0]:
        raise CheckError("concat scatter lost native B64 byte offsets")


def validate_broadcast_inline(text: str, ir_text: str | None) -> None:
    pairs = native_function_pairs(
        text, ir_text, "broadcast_s32_element_tile",
        (r"BSTART\.TLSU\s+MGATHER, U32\b",
         r"BSTART\.TLSU\s+TSTORE, S32\b"))
    if len(pairs) != 1:
        raise CheckError("inline broadcast needs one native broadcast function")
    _, body, ir = pairs[0]
    if not re.search(r"\bB\.ASSEMBLE\b", body) or "<2KB>" not in body:
        raise CheckError("inline broadcast lost its original512-element snapshot")
    if not re.search(r"BSTART\.TEPL\s+TLEA, U32\b", body):
        raise CheckError("inline broadcast lost B32 address indices")
    if re.search(r"extractelement|insertelement|@llvm\.vp\.|@llvm\.linx\.experimental\.element\.(region|view|gather)", ir):
        raise CheckError("inline broadcast retains scalar/intermediate element effects")


def validate_broadcast_fourpe(text: str, ir_text: str | None) -> None:
    pairs = native_function_pairs(
        text, ir_text, "multi_thread9broadcast",
        (r"BSTART\.TLSU\s+TLOAD, FP32\b",
         r"BSTART\.TLSU\s+TSTORE, FP32\b"))
    if len(pairs) != 1:
        raise CheckError("four-PE broadcast needs one native broadcast function")
    _, body, ir = pairs[0]
    if "<512B>" not in body or "<128 x i32>" not in ir:
        raise CheckError("four-PE broadcast changed its original128-element tile")
    load = re.search(r"BSTART\.TLSU\s+TLOAD, FP32\b.*?(?=BSTART|\Z)", body, re.S)
    if not load or not re.search(r"C\.B\.DIMI\s+128,\s+->lb0", load[0]) or not re.search(r"C\.B\.DIMI\s+128,\s+->lb2", load[0]):
        raise CheckError("four-PE broadcast changed its original transport geometry")


def validate_gather_hierarchy(text: str, ir_text: str | None) -> None:
    if ir_text is None:
        raise CheckError("gather requires actual-source IR")
    body = function_body(text, "_Z6gather")
    ir = ir_function_body(ir_text, "_Z6gather")
    for opcode in ("MGATHER", "MSCATTER"):
        if not re.search(rf"BSTART\.TLSU\s+{opcode}, FP32\b", body):
            raise CheckError("gather missing FP32 native " + opcode)
    if not re.search(r"BSTART\.TEPL\s+TLEA, U32\b", body):
        raise CheckError("gather requires B32 logical address indices")
    if not re.search(r"\bB\.ASSEMBLE\b", body) or not re.search(r"\bB\.SUBVIEW\b", body):
        raise CheckError("gather lost the complete public hierarchical snapshot")
    if not re.search(r"call <32 x float> @llvm\.linx\.experimental\.ew\.mgather", ir):
        raise CheckError("gather missing logical32 native value reads")
    if not re.search(r"call void @llvm\.linx\.experimental\.ew\.mscatter[^\n]*<32 x float>", ir):
        raise CheckError("gather missing native raw value writes")
    if re.search(r"extractelement|insertelement|@llvm\.vp\.|"
                 r"@llvm\.linx\.experimental\.element\.(region|view|gather|scatter)", ir):
        raise CheckError("gather retains scalar or intermediate element effects")
    if re.search(r"alloca \[(32|128) x i32\]", ir):
        raise CheckError("gather has scalar index snapshot scratch")


def validate_concat_gather_fourpe(text: str, ir_text: str | None) -> None:
    pairs = native_function_pairs(
        text, ir_text, "_Z13concat_gather",
        (r"BSTART\.TLSU\s+MGATHER, FP32\b",
         r"BSTART\.TLSU\s+TSTORE, FP32\b"))
    if len(pairs) != 1:
        raise CheckError("four-PE concat gather needs one native gather function")
    _, body, ir = pairs[0]
    if not re.search(r"call <32 x float> @llvm\.linx\.experimental\.ew\.mgather\.gpr\.masked", ir):
        raise CheckError("four-PE concat gather missing FP32 native reads")
    for pattern in (r"BSTART\.TLSU\s+MGATHER, FP32\b", r"BSTART\.TLSU\s+TSTORE, FP32\b",
                    r"BSTART\.TEPL\s+TLEA, U32\b", r"B\.ASSEMBLE\s+1, 0,", r"B\.ASSEMBLE\s+0, 1,"):
        if not re.search(pattern, body):
            raise CheckError("four-PE concat gather missing native snapshot path")
    if "<512B>" not in body or "<128 x float>" not in ir:
        raise CheckError("four-PE concat gather changed original128 snapshot")
    if re.search(r"extractelement|insertelement|@llvm\.vp\.|@llvm\.linx\.experimental\.element\.(region|view|gather)", ir):
        raise CheckError("four-PE concat gather retains scalar/intermediate element effects")


def validate_indexed_snapshot(text: str, ir_text: str | None) -> None:
    fragment = ("transpose_f16_element_tile"
                if "transpose_f16_element_tile" in text
                else "broadcast_f16_element_tile")
    pairs = native_function_pairs(
        text, ir_text, fragment,
        (r"BSTART\.TLSU\s+MGATHER, FP16\b",
         r"BSTART\.TLSU\s+TMOV, FP16\b",
         r"BSTART\.TLSU\s+TSTORE, FP16\b"))
    ir = "\n".join(pair[2] for pair in pairs)
    if not all(re.search(r"\bB\.ASSEMBLE\b", pair[1]) for pair in pairs):
        raise CheckError("indexed snapshot missing closed hierarchical snapshot")
    if not all(re.search(r"\bB\.SUBVIEW\b", pair[1]) for pair in pairs):
        raise CheckError("indexed snapshot missing borrowed output parts")
    if not all(re.search(r"BSTART\.TEPL\s+TLEA, U32\b", pair[1]) for pair in pairs):
        raise CheckError("indexed snapshot requires B32 indices to B64 byte offsets")
    for _, body, _ in pairs:
        stores = re.findall(
            r"BSTART\.TLSU\s+TSTORE, FP16\b.*?(?=BSTART|\Z)",
            body, re.S)
        if not stores or any("ExecMaskPresent" not in bundle
                             for bundle in stores):
            raise CheckError(
                "indexed snapshot tail stores must retain execution masks")
    if not all(re.search(
            r"call <64 x half> @llvm\.linx\.experimental\.ew\.mgather", pair[2])
            for pair in pairs):
        raise CheckError("indexed snapshot missing native logical64 FP16 gather")
    if re.search(r"extractelement|insertelement|@llvm\.vp\.|"
                 r"@llvm\.linx\.experimental\.element\.(region|view|gather)", ir):
        raise CheckError("indexed snapshot retains scalar or intermediate element effects")


def validate_leaf_gather_scatter(text: str, case: str, ir_text: str | None) -> None:
    fragment = ("process_view_copy_tile" if case.startswith("view_copy_")
                else "process_gather_tile")
    half = "_f16_" in case
    floating = "_f32_" in case
    read_type, write_type = (("FP16", "FP16") if half else
                             (("FP32", "FP32") if floating else ("U32", "S32")))
    carrier = "<64 x half>" if half else ("<32 x float>" if floating else "<32 x i32>")
    pairs = native_function_pairs(
        text, ir_text, fragment,
        (rf"BSTART\.TLSU\s+MGATHER, {read_type}\b",
         rf"BSTART\.TLSU\s+MSCATTER, {write_type}\b"))
    for _, body, ir in pairs:
        for opcode, dtype in (("MGATHER", read_type), ("MSCATTER", write_type)):
            bundles = re.findall(
                rf"BSTART\.TLSU\s+{opcode}, {dtype}\b.*?(?=BSTART|\Z)",
                body, re.S)
            if not bundles or any("ExecMaskPresent" not in bundle
                                  for bundle in bundles):
                raise CheckError(case + " missing masked native " + opcode)
        if not re.search(r"BSTART\.TEPL\s+TLEA, U32\b", body):
            raise CheckError(case + " requires B32 logical address indices")
        if not re.search(r"call " + re.escape(carrier) +
                         r" @llvm\.linx\.experimental\.ew\.mgather", ir):
            raise CheckError(case + " lost the complete physical input Tile")
        if not re.search(r"call void @llvm\.linx\.experimental\.ew\.mscatter[^\n]*" +
                         re.escape(carrier), ir):
            raise CheckError(case + " lost native scatter from the input snapshot")
        if re.search(r"extractelement|insertelement|@llvm\.vp\.|"
                     r"@llvm\.linx\.experimental\.element\."
                     r"(region|view|gather|scatter)", ir):
            raise CheckError(case + " retains scalar or intermediate element effects")


def validate(text: str, case: str, ir_text: str | None = None) -> None:
    if case.startswith(("view_copy_", "gather_v2_")):
        validate_leaf_gather_scatter(text, case, ir_text)
        return
    if case.startswith("broadcast_inline_s32_"):
        validate_broadcast_inline(text, ir_text)
        return
    if case == "broadcast_fourpe":
        validate_broadcast_fourpe(text, ir_text)
        return
    if case == "gather_fourpe" or case.startswith("gather_f32_"):
        validate_gather_hierarchy(text, ir_text)
        return
    if case == "concat_gather_fourpe":
        validate_concat_gather_fourpe(text, ir_text)
        return
    if case.startswith(("broadcast_f16_", "transpose_f16_")):
        validate_indexed_snapshot(text, ir_text)
        return
    if case.startswith("concat_scatter_"):
        validate_concat_scatter(text, case, ir_text)
        return
    if case in ("elementwise_masked_gather_scatter",
                "elementwise_masked_gather_scatter_strong"):
        validate_masked_gather_scatter_f32(text, ir_text)
        return
    if case in ("elementwise_if_for", "elementwise_if_for_strong"):
        validate_original_f32(text, ir_text)
        return
    if case in ("group_count_s32_original", "group_count_s32_strong"):
        validate_group_count(text, ir_text)
        return
    if case in ("concat_gather_f16_original", "concat_gather_f16_alias_probe"):
        validate_concat_f16_original(text, ir_text)
        return
    if case in ("concat_gather_s32_original", "concat_gather_s32_alias_probe"):
        validate_concat_original(text, ir_text)
        return
    if case == "elementwise_atomic_histogram":
        validate_original_atomic(text, ir_text)
        return
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
    native_text = (
        function_body(text, case)
        if case in ("histogram_tile_element", "selected_radix_tile_element")
        else text
    )
    lines = instruction_lines(native_text)
    is_topk = case == "topk_boundaries" or case.startswith("topk_boundaries_")
    expected_sites = 2 if is_topk else 1
    tlea = [line for line in lines if "BSTART.TEPL" in line and "TLEA, U32" in line]
    atomic = [line for line in lines if "BSTART.TLSU" in line and "MGATHER.ADD, U32" in line]
    subview = [line for line in lines if re.search(r"\bB\.SUBVIEW\b", line)]
    masked_ior = [
        line for line in lines
        if re.search(r"\bB\.IOR\b", line) and "ExecMaskPresent" in line
    ]
    # TopK also clears four workspace streams through element loops.
    expected_tleas = expected_sites + (4 if is_topk else 0)
    if len(tlea) != expected_tleas:
        raise CheckError(
            f"expected exactly {expected_tleas} static TLEA sites, found {len(tlea)}"
        )
    if len(atomic) != expected_sites:
        raise CheckError(
            "expected exactly "
            f"{expected_sites} static MGATHER.ADD sites, found {len(atomic)}"
        )
    if not subview:
        raise CheckError("missing TPARTVIEW/B.SUBVIEW lowering")
    if is_topk:
        # Scalar >> and & expressions introduce their own masked arithmetic.
        # Require the mask in each atomic bundle instead of counting all IORs.
        for atomic_line in atomic:
            start = lines.index(atomic_line)
            end = next((i for i in range(start + 1, len(lines))
                        if "BSTART." in lines[i]), len(lines))
            binders = [line for line in lines[start + 1:end]
                       if re.search(r"\bB\.IOR\b", line)]
            if len(binders) != 1 or "ExecMaskPresent" not in binders[0]:
                raise CheckError("atomic is missing its native execution mask")
    elif len(masked_ior) != expected_sites:
        raise CheckError("atomic is missing its native execution mask")
    if not any(re.search(r"\bTLOAD\b", line) for line in lines):
        raise CheckError("missing pre-element TLOAD")
    if is_topk:
        if not any(re.search(r"\bTANDS?\b", line) for line in lines) or not any(
                re.search(r"\bTSHRS?\b", line) for line in lines):
            raise CheckError("Top-K scalar bit expressions lost native lowering")
    else:
        if not any(re.search(r"\bTANDS\b", line) for line in lines):
            raise CheckError("missing pre-element TANDS")
        if case == "selected_radix_tile_element" and not any(
                re.search(r"\bTSHRS\b", line) for line in lines):
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
        if case == "selected_radix_tile_element":
            validate_selected_mask_conjunction(native_text)
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
        )) != 6:
            raise CheckError("generic Top-K migration lost its six B32 TLEA sites")
        stores = 0
        for function in re.findall(r"^define\b.*?^\}", ir_text, re.MULTILINE | re.DOTALL):
            offsets = set(re.findall(
                r"^\s*(%[\w.]+) = call <32 x i64> "
                r"@llvm\.linx\.experimental\.ew\.tlea[^\n(]*"
                r"\(i64 32, i64 1, i64 25, i64 29, <32 x i32> %[\w.]+, i64 32\)",
                function, re.MULTILINE,
            ))
            for call in function.splitlines():
                is_atomic = "call <32 x i32> @llvm.linx.experimental.ew.mgather.add.masked" in call
                is_store = "call void @llvm.linx.experimental.ew.mscatter.gpr.masked" in call
                if not (is_atomic or is_store):
                    continue
                stores += is_store
                offset = re.search(r"<32 x i64> (%[\w.]+),", call)
                if offset is None or offset.group(1) not in offsets:
                    raise CheckError("Top-K memory offset lacks its B32 U32-to-B64 TLEA producer")
        if stores != 4:
            raise CheckError("Top-K lost its four masked initialization stores")


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
            if args.case.startswith(("view_copy_", "gather_v2_")):
                read_type, write_type = (("FP16", "FP16") if "_f16_" in args.case else (("FP32", "FP32") if "_f32_" in args.case else ("U32", "S32")))
                for opcode, dtype in (("MGATHER", read_type), ("MSCATTER", write_type)):
                    rejected("missing " + opcode, text.replace(opcode + ", " + dtype, opcode + "_REMOVED, " + dtype), args.case, ir_text)
                rejected("wrong index width", text.replace("TLEA, U32", "TLEA, U64"), args.case, ir_text)
                broken = re.sub(r"(BSTART\.TLSU\s+MSCATTER, (?:FP16|FP32|S32)\b.*?)(?=BSTART|\Z)",
                                lambda match: match[0].replace("ExecMaskPresent", "NoExecMask"), text, flags=re.S)
                rejected("unmasked scatter", broken, args.case, ir_text)
                print(f"{args.case}: disassembly check PASS")
                return 0
            if args.case.startswith("broadcast_inline_s32_"):
                for opcode, dtype in (("MGATHER", "U32"), ("TSTORE", "S32")):
                    rejected("missing " + opcode, text.replace(opcode + ", " + dtype, opcode + "_REMOVED, " + dtype), args.case, ir_text)
                rejected("missing snapshot", text.replace("B.ASSEMBLE", "B.ASSEMBLE_REMOVED"), args.case, ir_text)
                rejected("wrong index width", text.replace("TLEA, U32", "TLEA, U64"), args.case, ir_text)
                print(f"{args.case}: disassembly check PASS")
                return 0
            if args.case == "broadcast_fourpe":
                for opcode in ("TLOAD", "TSTORE"):
                    rejected("missing " + opcode, text.replace(opcode + ", FP32", opcode + "_REMOVED, FP32"), args.case, ir_text)
                print(f"{args.case}: disassembly check PASS")
                return 0
            if args.case == "gather_fourpe" or args.case.startswith("gather_f32_"):
                for opcode in ("MGATHER", "MSCATTER"):
                    rejected("missing " + opcode, text.replace(opcode + ", FP32", opcode + "_REMOVED, FP32"), args.case, ir_text)
                for directive in ("B.ASSEMBLE", "B.SUBVIEW"):
                    rejected("missing " + directive, text.replace(directive, directive + "_REMOVED"), args.case, ir_text)
                rejected("wrong index width", text.replace("TLEA, U32", "TLEA, U64"), args.case, ir_text)
                print(f"{args.case}: disassembly check PASS")
                return 0
            if args.case == "concat_gather_fourpe":
                for opcode in ("MGATHER", "TSTORE"):
                    rejected("missing " + opcode, text.replace(opcode + ", FP32", opcode + "_REMOVED, FP32"), args.case, ir_text)
                rejected("missing LAST", text.replace("B.ASSEMBLE", "B.ASSEMBLE_REMOVED"), args.case, ir_text)
                rejected("wrong index width", text.replace("TLEA, U32", "TLEA, U64"), args.case, ir_text)
                print(f"{args.case}: disassembly check PASS")
                return 0
            if args.case.startswith(("broadcast_f16_", "transpose_f16_")):
                for opcode in ("MGATHER", "TSTORE"):
                    rejected("missing " + opcode, text.replace(opcode + ", FP16", opcode + "_REMOVED, FP16"), args.case, ir_text)
                for directive in ("B.ASSEMBLE", "B.SUBVIEW"):
                    rejected("missing " + directive, text.replace(directive, directive + "_REMOVED"), args.case, ir_text)
                rejected("wrong index width", text.replace("TLEA, U32", "TLEA, U64"), args.case, ir_text)
                rejected("unmasked tail", text.replace("ExecMaskPresent", "NoExecMask"), args.case, ir_text)
                print(f"{args.case}: disassembly check PASS")
                return 0
            if args.case.startswith("concat_scatter_"):
                dtype = "FP16" if "_f16_" in args.case else (
                    "FP32" if args.case == "concat_scatter_fourpe" else "S32")
                rejected("missing full input snapshot", text.replace(
                    "TLOAD, " + dtype, "TLOAD_REMOVED, " + dtype, 1), args.case, ir_text)
                rejected("missing borrowed snapshot", text.replace(
                    "B.SUBVIEW", "B.SUBVIEW_REMOVED", 1), args.case, ir_text)
                rejected("wrong index width", text.replace(
                    "TLEA, U32", "TLEA, U64", 1), args.case, ir_text)
                rejected("unmasked scatter", re.sub(
                    rf"(BSTART\.TLSU\s+MSCATTER, {dtype}.*?)(?=BSTART|\Z)",
                    lambda m: m[0].replace("ExecMaskPresent", "NoExecMask"),
                    text, count=1, flags=re.S), args.case, ir_text)
                rejected("numeric copy", text.replace(
                    "TMOV, " + dtype, "TCVT, " + dtype, 1), args.case, ir_text)
                print(f"{args.case}: disassembly check PASS")
                return 0
            if args.case in ("elementwise_if_for", "elementwise_if_for_strong",
                             "elementwise_masked_gather_scatter",
                             "elementwise_masked_gather_scatter_strong"):
                # Queue copies must not hide a missing semantic part copy, and
                # malformed full-parent copies must never be filtered away.
                rejected("missing borrowed part view", text.replace(
                    "B.SUBVIEW", "B.SUBVIEW_REMOVED", 1), args.case, ir_text)
                semantic_copy = next((match for match in re.finditer(
                    r"BSTART\.TLSU\s+TMOV, \w+\b.*?(?=BSTART|\Z)", text, re.S)
                    if "B.SUBVIEW" in match.group(0)), None)
                if semantic_copy:
                    rejected("missing complete part-copy bundle", text[:semantic_copy.start()] +
                             text[semantic_copy.end():], args.case, ir_text)
                parent_copy = next((match for match in re.finditer(
                    r"BSTART\.TLSU\s+TMOV, \w+\b.*?(?=BSTART|\Z)", text, re.S)
                    if "<512B>" in match.group(0)), None)
                if parent_copy:
                    original = parent_copy.group(0)
                    for label, broken in (
                        ("wrong parent-copy shape", re.sub(
                            r"(C\.B\.DIMI\s+)32(,\s+->lb1)",
                            r"\g<1>16\2", original)),
                        ("overridden parent-copy dimension", re.sub(
                            r"(C\.B\.DIMI[^\n]*->lb1[^\n]*)",
                            r"\1\n f00d: 00000000 C.B.DIMI 16, ->lb1", original)),
                        ("missing parent-copy dtype", re.sub(
                            r"TMOV, \w+", "TMOV, DTYPE_NONE", original)),
                        ("wrong parent-copy destination", re.sub(
                            r"->[tu]<512B>", "->n<512B>", original)),
                    ):
                        rejected(label, text[:parent_copy.start()] + broken +
                                 text[parent_copy.end():], args.case, ir_text)
            if args.case in ("elementwise_masked_gather_scatter",
                             "elementwise_masked_gather_scatter_strong"):
                for opcode in ("MGATHER", "MSCATTER"):
                    rejected("missing " + opcode, text.replace(
                        opcode + ", FP32", opcode + "_REMOVED, FP32", 1),
                        args.case, ir_text)
                    rejected("unmasked " + opcode, re.sub(
                        rf"(BSTART\.TLSU\s+{opcode}, FP32.*?)(?=BSTART|\Z)",
                        lambda match: match.group(0).replace(
                            "ExecMaskPresent", "NoExecMask"),
                        text, count=1, flags=re.S), args.case, ir_text)
                rejected("missing raw gathered TMOV", text.replace(
                    "TMOV, FP32", "TMOV_REMOVED, FP32", 1),
                    args.case, ir_text)
                rejected("numeric gathered transport", text.replace(
                    "TMOV, FP32", "TADDS, FP32", 1),
                    args.case, ir_text)
                rejected("unsigned indexed addressing", text.replace(
                    "TLEA, S32", "TLEA, U32", 1), args.case, ir_text)
                rejected("scalar extract fallback", text, args.case, re.sub(
                    r"(define[^\n]*elementwise_masked_gather_scatter[^\n]*\n)",
                    r"\1  %bad = extractelement <32 x float> undef, i32 0\n",
                    ir_text or "", count=1))
                rejected("residual scatter effect", text, args.case, re.sub(
                    r"(define[^\n]*elementwise_masked_gather_scatter[^\n]*\n)",
                    r"\1  call void @llvm.linx.experimental.element.scatter()\n",
                    ir_text or "", count=1))
                print(f"{args.case}: disassembly check PASS")
                return 0
            if args.case in ("elementwise_if_for", "elementwise_if_for_strong"):
                body = function_body(text, "elementwise_if_for")
                arithmetic = "TNEG" if "TNEG, FP32" in body else "TSUB"
                for op in ("TCMPS", "TADD", arithmetic):
                    rejected("missing " + op + " bundle mask", re.sub(
                        rf"(BSTART\.TEPL\s+{op}, FP32.*?)(?=BSTART|\Z)",
                        lambda match: match.group(0).replace("ExecMaskPresent", "NoExecMask"),
                        text, count=1, flags=re.S), args.case, ir_text)
                rejected("missing rawpart TMOV", text.replace("TMOV, FP32", "TMOV_REMOVED, FP32", 1), args.case, ir_text)
                rejected("FP identity conversion", text.replace("TMOV, FP32", "TCVT, FP32", 1), args.case, ir_text)
                rejected("add-zero identity transport", text.replace("TMOV, FP32", "TADDS, FP32", 1), args.case, ir_text)
                rejected("wrong arithmetic dtype", text.replace(arithmetic + ", FP32", arithmetic + ", U32", 1), args.case, ir_text)
                for marker in ("region", "view"):
                    rejected("residual element " + marker, text, args.case, re.sub(
                        r"(define[^\n]*elementwise_if_for[^\n]*\n)",
                        rf"\1  call void @llvm.linx.experimental.element.{marker}()\n",
                        ir_text or "", count=1))
                rejected("scalar extract fallback", text, args.case, re.sub(
                    r"(define[^\n]*elementwise_if_for[^\n]*\n)",
                    r"\1  %bad = extractelement <32 x float> undef, i32 0\n", ir_text or "", count=1))
                rejected("unmasked compare substitution", text, args.case,
                         (ir_text or "").replace("tcmps.gpr.masked.v32f32", "tcmps.gpr.v32f32", 1))
                rejected("unmasked arithmetic substitution", text, args.case,
                         (ir_text or "").replace("tbinary.gpr.masked.v32f32", "tbinary.v32f32", 1))
                rejected("wrong native comparison selector", text.replace("byte0, GT, RNONE", "byte0, LT, RNONE", 1), args.case, ir_text)
                rejected("wrong comparison selector", text, args.case, re.sub(
                    r"(@llvm\.linx\.experimental\.ew\.tcmps\.gpr\.masked\.v32f32[^\n]*), i64 3,",
                    r"\1, i64 2,", ir_text or "", count=1))
                rejected("empty arithmetic domain", text, args.case, re.sub(
                    r"(@llvm\.linx\.experimental\.ew\.tbinary\.gpr\.masked\.v32f32[^\n]*), i64 %[\w.]+, i64 0, i64 0, i64 1\)",
                    r"\1, i64 0, i64 0, i64 0, i64 1)", ir_text or "", count=1))
                if arithmetic == "TNEG":
                    rejected("unmasked negate substitution", text, args.case,
                             (ir_text or "").replace("tneg.gpr.masked.v32f32", "tneg.v32f32", 1))
                    rejected("wrong negate operand", text, args.case, re.sub(
                        r"(@llvm\.linx\.experimental\.ew\.tneg\.gpr\.masked\.v32f32[^\n]*<32 x float> )%[\w.]+",
                        r"\1%wrong.input", ir_text or "", count=1))
                rejected("wrong merge predicate", text, args.case, re.sub(
                    r"(@llvm\.linx\.experimental\.ew\.tsel\.gpr\.v32f32\(i64 32, i64 1, i64 1, i64 29, i64 )%[\w.]+",
                    r"\g<1>0", ir_text or "", count=1))
                rejected("wrong add operand", text, args.case, re.sub(
                    r"(@llvm\.linx\.experimental\.ew\.tbinary\.gpr\.masked\.v32f32[^\n]*<32 x float> )%[\w.]+",
                    r"\1%wrong.input", ir_text or "", count=1))
                print(f"{args.case}: disassembly check PASS")
                return 0
            if args.case in ("concat_gather_s32_original", "concat_gather_s32_alias_probe"):
                rejected("missing assembly completion", text.replace("B.ASSEMBLE", "B.ASSEMBLE_REMOVED"), args.case, ir_text)
                rejected("shrunk logical batch", text.replace("<2KB>", "<128B>"), args.case, ir_text)
                rejected("wrong byte-offset dtype", text, args.case,
                         (ir_text or "").replace("call <32 x i64> @llvm.linx.experimental.ew.tlea", "call <32 x i32> @llvm.linx.experimental.ew.tlea", 1))
                for marker in ("region", "view"):
                    rejected("residual element " + marker, text, args.case,
                             re.sub(r"(define[^\n]*concat_gather[^\n]*\n)",
                                    rf"\1  call void @llvm.linx.experimental.element.{marker}()\n", ir_text or "", count=1))
                print(f"{args.case}: disassembly check PASS")
                return 0
            if args.case in ("concat_gather_f16_original", "concat_gather_f16_alias_probe"):
                rejected("missing raw part TMOV", text.replace(
                    "TMOV, FP16", "TMOV_REMOVED, FP16", 1),
                    args.case, ir_text)
                rejected("numeric part conversion", text.replace(
                    "TMOV, FP16", "TCVT, FP16", 1),
                    args.case, ir_text)
                rejected("missing assembly completion", text.replace(
                    "B.ASSEMBLE", "B.ASSEMBLE_REMOVED"), args.case, ir_text)
                rejected("shrunk logical parent", text.replace(
                    "<1KB>", "<128B>"), args.case, ir_text)
                rejected("wrong element address scaling", text, args.case, re.sub(
                    r"(@llvm\.linx\.experimental\.ew\.tlea[^\n]*, i64 )16\)",
                    r"\g<1>32)", ir_text or "", count=1))
                rejected("residual scalar extraction", text, args.case, re.sub(
                    r"(define[^\n]*concat_gather[^\n]*\n)",
                    r"\1  %bad = extractelement <64 x half> undef, i32 0\n",
                    ir_text or "", count=1))
                print(f"{args.case}: disassembly check PASS")
                return 0
            is_topk = (args.case == "topk_boundaries" or
                       args.case.startswith("topk_boundaries_"))
            if args.case in ("element_expression_chain", "signed_element_expression", "indexed_gather_tile_element", "generic_predicated_cfg_i32", "generic_typed_tile_cfg_i32", "generic_atomic_cfg_i32", "histogram_tile_element", "selected_radix_tile_element", "elementwise_atomic_histogram"):
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
            if args.case == "selected_radix_tile_element":
                selected_body = function_body(text, args.case)
                conjunction = validate_selected_mask_conjunction(selected_body)
                mnemonic, operands = native_instruction(conjunction)
                assert mnemonic in ("AND", "C.AND")
                broken = conjunction.replace(
                    "\t" + mnemonic.lower() + "\t",
                    "\tor\t",
                    1,
                )
                if broken == conjunction:
                    broken = re.sub(
                        r"\b(?:C\.)?AND\b", "OR", conjunction, count=1,
                        flags=re.I,
                    )
                rejected(
                    "selected mask is not a conjunction",
                    text.replace(conjunction, broken, 1),
                    args.case,
                    ir_text,
                )
            if args.case == "elementwise_atomic_histogram":
                rejected("missing atomic effect", text.replace("MGATHER.ADD, U32", "MGATHER, U32", 1), args.case, ir_text)
                rejected("missing memory masks", text.replace("ExecMaskPresent", "NoExecMask"), args.case, ir_text)
                rejected("wrong index signedness", text.replace("TLEA, S32", "TLEA, U32", 1), args.case,
                         (ir_text or "").replace("i64 17, i64 29, <32 x i32>", "i64 25, i64 29, <32 x i32>", 1))
                print(f"{args.case}: disassembly check PASS")
                return 0
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
            if args.case in ("group_count_s32_original", "group_count_s32_strong"):
                for marker in ("region", "view"):
                    rejected(
                        f"residual element {marker} contract",
                        text,
                        args.case,
                        re.sub(
                            r"(define[^\n]*group_countILi16[^\n]*\n)",
                            rf"\1  call void @llvm.linx.experimental.element.{marker}()\n",
                            ir_text or "",
                            count=1,
                        ),
                    )
                rejected(
                    "atomic bundle missing execution mask",
                    re.sub(
                        r"(BSTART\.TLSU\s+MGATHER\.ADD, (?:S32|U32).*?)(?=BSTART|\Z)",
                        lambda match: match.group(0).replace(
                            "ExecMaskPresent", "NoExecMask"
                        ),
                        text,
                        count=1,
                        flags=re.S,
                    ),
                    args.case,
                    ir_text,
                )
                rejected(
                    "reset before snapshot",
                    text.replace("TLOAD, U32", "TLOAD_REMOVED, U32", 1),
                    args.case,
                    ir_text,
                )
                rejected(
                    "signed byte-index conversion",
                    text.replace("TLEA, U32", "TLEA, S32", 1),
                    args.case,
                    ir_text,
                )
                rejected(
                    "wrong group bound",
                    text,
                    args.case,
                    (ir_text or "").replace("freeze i32 32", "freeze i32 33", 1),
                )
                rejected(
                    "narrow B32-to-B64 result",
                    text,
                    args.case,
                    (ir_text or "").replace(
                        "call <32 x i64> @llvm.linx.experimental.ew.tlea",
                        "call <32 x i32> @llvm.linx.experimental.ew.tlea", 1),
                )
                rejected(
                    "integer path emitted TCVT",
                    text.replace("TADDS, U32", "TCVT, U32", 1),
                    args.case,
                    ir_text,
                )
                rejected(
                    "residual scalar atomic",
                    text,
                    args.case,
                    re.sub(
                        r"(define[^\n]*group_countILi16[^\n]*\n)",
                        r"\1  %bad = atomicrmw add ptr null, i32 1 monotonic\n",
                        ir_text or "",
                        count=1,
                    ),
                )
                print(f"{args.case}: disassembly check PASS")
                return 0
            if is_topk:
                rejected(
                    "unrelated atomic byte offsets", text, args.case,
                    re.sub(
                        r"(ew\.mgather\.add\.masked[^\n]*<32 x i64> )%[\w.]+",
                        r"\1%unrelated.offsets", ir_text or "", count=1,
                    ),
                )
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
                    "missing native XOR",
                    re.sub(r"TXOR, ([US]32)", r"TXOR_REMOVED, \1", text,
                           count=1),
                    args.case,
                    ir_text,
                )
                rejected(
                    "wrong XOR source constant",
                    text,
                    args.case,
                    (ir_text or "").replace("freeze i32 85", "freeze i32 86", 1),
                )
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
                rejected("missing multiply", text.replace("TMUL, U32", "TMUL_REMOVED, U32", 1), args.case, ir_text)
                rejected("direct floor TREM", text.replace("TSUB, S32", "TREM, S32", 1), args.case, ir_text)
                rejected("wrong shift dtype", text.replace("TSHR, U32", "TSHR, S32", 1), args.case, ir_text)
                rejected("reversed shift operands", text, args.case,
                         re.sub(
                             r"(i64 25, i64 29, i64 9, <32 x i32> )"
                             r"(%[-.a-zA-Z0-9]+)(, <32 x i32> )"
                             r"(%[-.a-zA-Z0-9]+)",
                             r"\1\4\3\2", ir_text or "", count=1))
                rejected("wrong arithmetic constant", text, args.case,
                         (ir_text or "").replace("freeze i32 50", "freeze i32 51", 1))
                rejected("wrong mask constant", text, args.case,
                         (ir_text or "").replace("freeze i32 255", "freeze i32 254", 1))
                rejected("wrong remainder dataflow", text, args.case,
                         re.sub(r"(i64 32, i64 1, i64 17, i64 29, i64 )1,",
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
            if args.case == "topk_boundaries" or args.case.startswith("topk_boundaries_"):
                broken_mask = re.sub(
                    r"(BSTART\.TLSU\s+MGATHER\.ADD, U32.*?)(?=BSTART|\Z)",
                    lambda match: match[0].replace("ExecMaskPresent", "NoExecMask"),
                    text, count=1, flags=re.S)
                rejected("missing atomic execution mask", broken_mask, args.case)
                rejected("missing scalar shift lowering", re.sub(r"\bTSHRS?\b", "TSHR_REMOVED", text), args.case)
                rejected("missing scalar mask lowering", re.sub(r"\bTANDS?\b", "TAND_REMOVED", text), args.case)
            else:
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
