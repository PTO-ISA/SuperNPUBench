#!/usr/bin/env python3
"""Prove exact MoE histogram helpers use masked native Tile operations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from native_bstart import normalize_disassembly


class CheckError(RuntimeError):
    pass


@dataclass(frozen=True)
class HelperSpec:
    label: str
    clear_source: str
    count_source: str
    clear_symbol: str
    count_symbol: str


OLD = HelperSpec("old", "clearElementHistogram", "addElementHistogramPart",
                 "clearElementHistogram", "addElementHistogramPart")
DYNAMIC = HelperSpec(
    "dynamic", "gt_dyn_clear_element_histogram", "addElementHistogramPart",
    "gt_dyn_clear_element_histogram", "addElementHistogramPart",
)


def extract_cpp_function(source: str, name: str) -> str:
    definitions: list[str] = []
    for match in re.finditer(rf"\b{re.escape(name)}\s*\(", source):
        opening = source.find("{", match.end())
        semicolon = source.find(";", match.end())
        if opening < 0 or (semicolon >= 0 and semicolon < opening):
            continue
        depth = 0
        for index in range(opening, len(source)):
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
                if depth == 0:
                    definitions.append(source[match.start():index + 1])
                    break
    if len(definitions) != 1:
        raise CheckError(f"expected one definition of {name}, found {len(definitions)}")
    return definitions[0]


def extract_ir_function(ir: str, symbol_fragment: str) -> str:
    definitions = []
    lines = ir.splitlines()
    for start, line in enumerate(lines):
        if line.startswith("define ") and symbol_fragment in line:
            end = start + 1
            while end < len(lines) and lines[end] != "}":
                end += 1
            if end == len(lines):
                raise CheckError(f"unterminated IR function: {symbol_fragment}")
            definitions.append("\n".join(lines[start:end + 1]))
    if len(definitions) != 1:
        raise CheckError(
            f"expected one IR function containing {symbol_fragment}, found {len(definitions)}"
        )
    return definitions[0]


def extract_native_function(disassembly: str, symbol_fragment: str) -> str:
    pattern = re.compile(
        rf"^[0-9a-f]+ <[^>]*{re.escape(symbol_fragment)}[^>]*>:\n"
        rf"(.*?)(?=^[0-9a-f]+ <(?!\.LBB)|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    matches = pattern.findall(disassembly)
    if len(matches) != 1:
        raise CheckError(
            f"expected one native function containing {symbol_fragment}, found {len(matches)}"
        )
    return matches[0]


def require_all(text: str, tokens: tuple[str, ...], context: str) -> None:
    missing = [token for token in tokens if token not in text]
    if missing:
        raise CheckError(f"{context}: missing {missing}")


def reject_scalar_ir(body: str, context: str) -> None:
    forbidden = {
        "scalar atomicrmw": r"\batomicrmw\b",
        "atomic helper call": r"(?:__atomic|__sync)_",
        "scalarized extractelement": r"\bextractelement\b",
        "scalarized insertelement": r"\binsertelement\b",
        "scalar i32 load": r"\bload i32\b",
        "scalar i32 store": r"\bstore i32\b",
    }
    for description, pattern in forbidden.items():
        if re.search(pattern, body):
            raise CheckError(f"{context}: {description} remains in exact helper IR")


ALLOWED_IR_CALLS = frozenset({
    "llvm.umin.i32",
    "llvm.linx.experimental.ew.tci.v32i32",
    "llvm.linx.experimental.ew.tci.v32i64",
    "llvm.linx.experimental.ew.tcmps.gpr.v32i32",
    "llvm.linx.experimental.ew.tcmps.gpr.v32i64",
    "llvm.linx.experimental.ew.tlea.v32i64.v32i32",
    "llvm.linx.experimental.ew.mgather.gpr.masked.v32i32.v32i64",
    "llvm.linx.experimental.ew.tsel.gpr.v32i32",
    "llvm.linx.experimental.ew.mgather.add.masked.v32i32.v32i64.v32i32",
    "llvm.linx.experimental.ew.mscatter.gpr.masked.v32i64.v32i32",
})


def validate_ir_calls(body: str, context: str) -> None:
    for line in body.splitlines():
        if not re.search(r"\b(?:tail )?call\b", line):
            continue
        if re.search(r"\bcall\b.*\basm\b", line):
            raise CheckError(f"{context}: unadmitted inline assembly call")
        match = re.search(r"@([^\s(]+)\(", line)
        if match is None:
            raise CheckError(f"{context}: unrecognized call site")
        callee = match.group(1)
        if callee not in ALLOWED_IR_CALLS:
            raise CheckError(f"{context}: unadmitted call to {callee}")


def validate_source(source: str, spec: HelperSpec) -> None:
    clear = extract_cpp_function(source, spec.clear_source)
    count = extract_cpp_function(source, spec.count_source)
    for body, kind, marked_count in ((clear, "clear", 1), (count, "count", 1)):
        if body.count("#pragma pto element for") != marked_count:
            raise CheckError(
                f"{spec.label} {kind}: expected {marked_count} marked loop(s)"
            )
        if "element <" not in body or "if (element <" not in body:
            raise CheckError(f"{spec.label} {kind}: fixed loop or valid guard missing")
    require_all(count, ("= input[element]",
                        "__atomic_fetch_add", "__ATOMIC_RELAXED"),
                f"{spec.label} count source")
    if "TLOAD(" in count:
        raise CheckError(f"{spec.label} count: unbounded parent Tile load present")
    if re.search(r"histogram\s*\[[^]]+\]\s*(?:\+\+|\+=)", count):
        raise CheckError(f"{spec.label} count: scalar histogram update present")
    part_constant = "kHistogramPartElements" if spec is OLD else "kPartElements"
    if not re.search(rf"{part_constant}\s*=\s*32\s*;", source):
        raise CheckError(f"{spec.label}: wrong element-loop width")
    if "ElementTile" in count or "TPARTELEMENT" in count:
        raise CheckError(f"{spec.label}: fused loop must not stage input through a Tile view")
    if f"element < {part_constant}" not in count:
        raise CheckError(f"{spec.label}: count loop differs from logical width")


def validate_clear_ir(ir: str, spec: HelperSpec) -> None:
    body = extract_ir_function(ir, spec.clear_symbol)
    require_all(body, ("llvm.linx.experimental.ew.tci.v32i32",
                       "llvm.linx.experimental.ew.tlea.v32i64.v32i32",
                       "llvm.linx.experimental.ew.mscatter.gpr.masked.v32i64.v32i32",
                       "%pto.mask.profile"),
                f"{spec.label} clear IR")
    reject_scalar_ir(body, f"{spec.label} clear")
    validate_ir_calls(body, f"{spec.label} clear")


def validate_count_ir(ir: str, spec: HelperSpec) -> None:
    body = extract_ir_function(ir, spec.count_symbol)
    if "BSTART.TLOAD" in body:
        raise CheckError(f"{spec.label} count IR: unbounded parent Tile load present")
    require_all(body, ("<32 x i32>",
                       "llvm.linx.experimental.ew.tci.v32i32",
                       "llvm.linx.experimental.ew.tcmps.gpr.v32i32",
                       "llvm.linx.experimental.ew.tlea.v32i64.v32i32",
                       "llvm.linx.experimental.ew.mgather.gpr.masked.v32i32.v32i64",
                       "llvm.linx.experimental.ew.mgather.add.masked.v32i32.v32i64.v32i32",
                       "%pto.mask.profile"), f"{spec.label} count IR")
    if body.count("llvm.linx.experimental.ew.tlea.v32i64.v32i32") < 2:
        raise CheckError(f"{spec.label} count IR: input and atomic address paths missing")
    # LLVM may widen the loop induction variable to i64 while expert IDs
    # remain U32. Both comparisons must still lower to native Tile masks.
    comparisons = re.findall(
        r"@llvm\.linx\.experimental\.ew\.tcmps\.gpr\.v32i(?:32|64)\(", body)
    if len(comparisons) < 2:
        raise CheckError(f"{spec.label} count IR: valid/expert masks missing")
    reject_scalar_ir(body, f"{spec.label} count")
    validate_ir_calls(body, f"{spec.label} count")


def reject_scalar_native(body: str, context: str) -> None:
    instruction = re.compile(
        r"^\s*[0-9a-f]+:\s+(?:(?:[0-9a-f]{4}|[0-9a-f]{8})\s+)+"
        r"([A-Za-z][A-Za-z0-9.]*)",
        re.IGNORECASE,
    )
    scalar_memory = re.compile(
        r"^(?:C\.)?(?:(?:LB|LBU|LBI|LH|LHU|LHI|LW|LWU|LWI|LWUI|LD|LDI)"
        r"(?:\.U)?|(?:SB|SBI|SH|SHI|SW|SWI|SD|SDI)|AMO[A-Z0-9.]*)$"
        r"|^C\.(?:LW|LD|SW|SD)SP$",
        re.IGNORECASE,
    )
    native_call = re.compile(
        r"^(?:C\.)?(?:CALL|JAL|JALR)$", re.IGNORECASE
    )
    for line in body.splitlines():
        match = instruction.match(line)
        if match is None:
            continue
        mnemonic = match.group(1)
        if scalar_memory.fullmatch(mnemonic):
            raise CheckError(
                f"{context}: scalar data-memory instruction {mnemonic} remains"
            )
        if native_call.fullmatch(mnemonic):
            raise CheckError(f"{context}: native call {mnemonic} remains")
        if mnemonic.upper() in ("BSTART.STD", "C.BSTART.STD") and \
                re.search(r"\bDIRECT\b", line):
            raise CheckError(f"{context}: direct native call/branch remains")
    if re.search(r"(?:__atomic|__sync)_", body):
        raise CheckError(f"{context}: scalar atomic helper remains")


def require_masked_tlsu_bundle(body: str, operation: str, context: str) -> None:
    lines = body.splitlines()
    start_pattern = re.compile(
        rf"\bBSTART\.TLSU\s+{re.escape(operation)}(?=[,\s]|$)"
    )
    starts = [index for index, line in enumerate(lines)
              if start_pattern.search(line)]
    if len(starts) != 1:
        raise CheckError(
            f"{context}: expected one {operation} bundle, found {len(starts)}"
        )
    start = starts[0]
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if "BSTART." in lines[index]:
            end = index
            break
    bundle = "\n".join(lines[start:end])
    if not re.search(r"\bB\.IOR\b.*\bExecMaskPresent\b", bundle):
        raise CheckError(f"{context}: {operation} bundle lacks its execution mask")


def validate_clear_native(disassembly: str, spec: HelperSpec) -> None:
    body = extract_native_function(disassembly, spec.clear_symbol)
    require_all(body, ("BSTART.TEPL\tTCI", "BSTART.TEPL\tTLEA"),
                f"{spec.label} native clear")
    require_masked_tlsu_bundle(body, "MSCATTER", f"{spec.label} native clear")
    reject_scalar_native(body, f"{spec.label} native clear")


def validate_count_native(disassembly: str, spec: HelperSpec) -> None:
    body = extract_native_function(disassembly, spec.count_symbol)
    require_all(body, ("BSTART.TEPL\tTCI", "TCMPS",
                       "BSTART.TEPL\tTLEA"),
                f"{spec.label} native count")
    require_masked_tlsu_bundle(body, "MGATHER", f"{spec.label} native count")
    require_masked_tlsu_bundle(body, "MGATHER.ADD",
                               f"{spec.label} native count")
    if "TLOAD" in body:
        raise CheckError(f"{spec.label} native count: unbounded parent Tile load present")
    reject_scalar_native(body, f"{spec.label} native count")


def validate_artifact(source: str, ir: str, disassembly: str,
                      spec: HelperSpec) -> None:
    disassembly = normalize_disassembly(disassembly)
    validate_source(source, spec)
    validate_clear_ir(ir, spec)
    validate_count_ir(ir, spec)
    validate_clear_native(disassembly, spec)
    validate_count_native(disassembly, spec)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-source", required=True, type=Path)
    parser.add_argument("--dynamic-source", required=True, type=Path)
    parser.add_argument("--old-ir", required=True, type=Path)
    parser.add_argument("--dynamic-ir", required=True, type=Path)
    parser.add_argument("--old-dis", required=True, type=Path)
    parser.add_argument("--dynamic-dis", required=True, type=Path)
    args = parser.parse_args()
    try:
        validate_artifact(args.old_source.read_text(), args.old_ir.read_text(),
                          args.old_dis.read_text(), OLD)
        validate_artifact(args.dynamic_source.read_text(),
                          args.dynamic_ir.read_text(),
                          args.dynamic_dis.read_text(), DYNAMIC)
    except CheckError as error:
        raise SystemExit(str(error)) from None
    print("MoE element histogram lowering: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
