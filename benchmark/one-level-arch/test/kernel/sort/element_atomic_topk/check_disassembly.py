#!/usr/bin/env python3
"""Validate the native coherence, full histogram, and selected histogram sites."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


class CheckError(RuntimeError):
    pass


@dataclass
class Instruction:
    raw: int
    mnemonic: str
    operands: str
    line: str


INSTRUCTION = re.compile(
    r"^\s*[0-9a-f]+:\s+([0-9a-f]{4,8})(?:\s+[0-9a-f]{4,8})*\s+"
    r"([^\s]+)(?:\s+(.*))?$",
    re.IGNORECASE,
)
GPR = re.compile(r"^(?:zero|ra|sp|gp|tp|t|s[0-9]+|a[0-9]+|x[0-9]+)(?:#[0-9]+)?$")
TRANSIENT = re.compile(r"^([tu])#([1-4])$")


def parse(text: str) -> list[Instruction]:
    result: list[Instruction] = []
    for line in text.splitlines():
        match = INSTRUCTION.match(line)
        if match:
            result.append(
                Instruction(
                    raw=int(match.group(1), 16),
                    mnemonic=match.group(2).upper(),
                    operands=(match.group(3) or "").strip(),
                    line=line,
                )
            )
    return result


def destination(operands: str) -> str:
    match = re.search(r"->\s*([^\s,]+)", operands)
    if not match:
        raise CheckError(f"instruction has no destination: {operands}")
    return match.group(1)


def validate_mask_flow(segment: list[Instruction], compares: list[int],
                       and_index: int | None, tlea_index: int,
                       bior_index: int, mask: str, ordinal: int) -> None:
    """Prove register versions from comparisons to the atomic mask binder."""
    registers: dict[str, tuple[str, int]] = {}
    transient: dict[str, list[tuple[str, int]]] = {"t": [], "u": []}
    compare_values: list[tuple[str, int]] = []
    mask_value: tuple[str, int] | None = None
    selects = [index for index in range(compares[-1] + 1, tlea_index)
               if segment[index].mnemonic == "CSEL"]
    if len(selects) > 1:
        raise CheckError(f"site {ordinal}: multiple outer mask selects")
    select_index = selects[0] if selects else None

    def read(name: str) -> tuple[str, int]:
        if name == "zero":
            return ("zero", -1)
        relative = TRANSIENT.fullmatch(name)
        if relative:
            bank, distance = relative.group(1), int(relative.group(2))
            values = transient[bank]
            if len(values) >= distance:
                return values[-distance]
            raise CheckError(f"site {ordinal}: unproved transient mask {name}")
        return registers.get(name, ("entry", -1))

    def write(name: str, value: tuple[str, int]) -> None:
        if name == "zero":
            return
        if name in transient:
            transient[name].append(value)
        else:
            registers[name] = value

    for index in range(compares[0], bior_index):
        inst = segment[index]
        # Compact fall-through STD boundaries are harmless. Branch/call/return
        # headers cannot connect values merely by linear disassembly order.
        if (re.search(r"BSTART\.(STD|AUX|FP)$", inst.mnemonic) and inst.operands) or \
                inst.mnemonic.startswith("FRET.") or \
                inst.mnemonic in ("FENTRY", "FEXIT", "RET", "EBREAK", "ECALL", "ACRC"):
            raise CheckError(f"site {ordinal}: control transfer in mask slice")
        dest_match = re.search(r"->\s*([^\s,]+)", inst.operands)
        dst = dest_match.group(1) if dest_match else None
        if index in compares:
            if dst is None or dst == "zero" or not GPR.fullmatch(dst):
                raise CheckError(f"site {ordinal}: compare lacks a GPR result")
            value = ("compare", index)
            compare_values.append(value)
            write(dst, value)
            if len(compares) == 1:
                mask_value = value
            continue
        if index == and_index:
            match = re.fullmatch(
                r"([^,]+),\s*([^,]+),\s*->\s*([^\s,]+)", inst.operands)
            if match is None:
                raise CheckError(f"site {ordinal}: malformed scalar AND")
            lhs, rhs, dst = (part.strip() for part in match.groups())
            if dst == "zero" or not all(GPR.fullmatch(part) for part in (lhs, rhs, dst)):
                raise CheckError(f"site {ordinal}: AND operands are not GPRs")
            values = [read(lhs), read(rhs)]
            if sorted(values) != sorted(compare_values):
                raise CheckError(f"site {ordinal}: AND lost a compare version")
            mask_value = ("and", index)
            write(dst, mask_value)
            continue
        if index == select_index:
            # CSEL's assembler order is predicate, true source, false source.
            match = re.fullmatch(
                r"([^,]+),\s*([^,]+),\s*([^,]+),\s*->\s*([^\s,]+)",
                inst.operands)
            if match is None:
                raise CheckError(f"site {ordinal}: malformed mask CSEL")
            predicate, true_value, false_value, dst = (
                part.strip() for part in match.groups())
            if predicate == "zero" or dst == "zero" or "#" in predicate or \
                    not all(GPR.fullmatch(part) for part in
                            (predicate, true_value, false_value, dst)):
                raise CheckError(f"site {ordinal}: unproved outer predicate")
            if false_value != "zero" or read(true_value) != mask_value:
                raise CheckError(f"site {ordinal}: CSEL is not mask-or-zero")
            if read(predicate) in compare_values or read(predicate) == mask_value:
                raise CheckError(f"site {ordinal}: outer predicate uses mask result")
            if and_index is not None and segment[and_index].mnemonic == "C.AND":
                if true_value != "t#1" or index != and_index + 1:
                    raise CheckError(f"site {ordinal}: compressed AND is not consumed immediately")
            mask_value = ("select", index)
            write(dst, mask_value)
            continue
        if dst is not None and GPR.fullmatch(dst):
            write(dst, ("other", index))

    if mask_value is None or read(mask) != mask_value:
        raise CheckError(f"site {ordinal}: B.IOR lost the final mask version")


def validate_generic_mask_flow(segment: list[Instruction], compares: list[int],
                               bior_index: int, mask: str,
                               ordinal: int) -> None:
    """Track every compare through generic CFG AND/CSEL mask construction."""
    registers: dict[str, frozenset[tuple[str, int]]] = {}
    transient: dict[str, list[frozenset[tuple[str, int]]]] = {"t": [], "u": []}

    def read(name: str) -> frozenset[tuple[str, int]]:
        if name == "zero":
            return frozenset()
        relative = TRANSIENT.fullmatch(name)
        if relative:
            bank, distance = relative.group(1), int(relative.group(2))
            values = transient[bank]
            return values[-distance] if len(values) >= distance else frozenset()
        return registers.get(name, frozenset())

    def write(name: str, value: frozenset[tuple[str, int]]) -> None:
        if name == "zero":
            return
        if name in transient:
            transient[name].append(value)
        else:
            registers[name] = value

    required: set[tuple[str, int]] = set()
    for index, inst in enumerate(segment[:bior_index]):
        dest_match = re.search(r"->\s*([^\s,]+)", inst.operands)
        dst = dest_match.group(1) if dest_match else None
        if dst is None or not GPR.fullmatch(dst):
            continue
        if index in compares:
            marker = ("compare", index)
            required.add(marker)
            write(dst, frozenset({marker}))
            continue
        if inst.mnemonic in ("AND", "C.AND"):
            match = re.fullmatch(
                r"([^,]+),\s*([^,]+),\s*->\s*([^\s,]+)", inst.operands
            )
            if match:
                lhs, rhs, out = (part.strip() for part in match.groups())
                write(out, read(lhs) | read(rhs))
                continue
        if inst.mnemonic == "CSEL":
            match = re.fullmatch(
                r"([^,]+),\s*([^,]+),\s*([^,]+),\s*->\s*([^\s,]+)",
                inst.operands,
            )
            if match:
                _predicate, yes, no, out = (part.strip() for part in match.groups())
                write(out, read(yes) | read(no))
                continue
        write(dst, frozenset({("scalar", index)}))

    observed = read(mask)
    if not observed or not required.issubset(observed):
        raise CheckError(f"site {ordinal}: atomic mask lost CFG predicate provenance")


def validate_segment(segment: list[Instruction], ordinal: int) -> None:
    expected_compares = (0, 1, 2)[ordinal]
    def positions(mnemonic: str) -> list[int]:
        return [index for index, inst in enumerate(segment) if inst.mnemonic == mnemonic]

    tci = positions("BSTART.TEPL")
    tci = [index for index in tci if re.search(r"\bTCI,\s*U32\b", segment[index].operands)]
    compares = positions("TCMPS")
    scalar_and = positions("AND") + positions("C.AND")
    tlea = [
        index
        for index in positions("BSTART.TEPL")
        if re.search(r"\bTLEA,\s*U32\b", segment[index].operands)
    ]
    atomic = [
        index
        for index in positions("BSTART.TLSU")
        if re.search(r"\bMGATHER\.ADD,\s*U32\b", segment[index].operands)
    ]
    if len(tci) < 2:
        raise CheckError(f"helper {ordinal}: expected lane and one-value TCI")
    if len(compares) != expected_compares:
        raise CheckError(f"helper {ordinal}: wrong number of TCMPS GPR producers")
    if len(scalar_and) < max(0, expected_compares - 1):
        raise CheckError(f"helper {ordinal}: too few scalar predicate AND operations")
    if len(tlea) != 1:
        raise CheckError(f"helper {ordinal}: expected exactly one TLEA")
    if len(atomic) != 1:
        raise CheckError(f"helper {ordinal}: expected exactly one MGATHER.ADD")

    compare_dsts = [destination(segment[index].operands) for index in compares]
    if any(not GPR.fullmatch(value) for value in compare_dsts):
        raise CheckError(f"site {ordinal}: TCMPS destination is not a scalar GPR")
    if not (any(index < atomic[0] for index in tci) and tlea[0] < atomic[0] and
            all(index < atomic[0] for index in compares) and
            sum(index < atomic[0] for index in scalar_and) >=
            max(0, expected_compares - 1)):
        raise CheckError(f"site {ordinal}: predicate/TLEA/value setup follows atomic")

    atomic_index = atomic[0]
    bundle_end = next(
        (index for index in range(atomic_index + 1, len(segment))
         if "BSTART." in segment[index].mnemonic or
         segment[index].mnemonic in ("TSTORE", "BSTOP", "C.BSTOP")), len(segment))
    atomic_bundle = segment[atomic_index + 1:bundle_end]
    if sum(inst.mnemonic == "B.DATR" for inst in atomic_bundle) != 1 or \
            sum(inst.mnemonic == "B.IOR" for inst in atomic_bundle) != 1:
        raise CheckError(f"helper {ordinal}: atomic bundle needs exactly one DATR and IOR")
    datr = next(
        (inst for inst in atomic_bundle if inst.mnemonic == "B.DATR"),
        None,
    )
    if datr is None or (datr.raw & (1 << 13)) == 0:
        raise CheckError(f"helper {ordinal}: MGATHER.ADD does not encode Zero=1")
    bior = next(
        (inst for inst in atomic_bundle if inst.mnemonic == "B.IOR"),
        None,
    )
    if bior is None or "EXECMASKPRESENT" not in bior.operands.upper():
        raise CheckError(f"helper {ordinal}: missing native execution-mask B.IOR")
    if (bior.raw & (1 << 26)) == 0:
        raise CheckError(f"helper {ordinal}: B.IOR ExecMaskPresent bit26 is clear")
    binder = re.search(r"\[([^,]+),([^,]+),([^\]]+)\]", bior.operands)
    if not binder:
        raise CheckError(f"helper {ordinal}: malformed masked atomic B.IOR")
    base, mask, unused = (value.strip() for value in binder.groups())
    if not GPR.fullmatch(base) or not GPR.fullmatch(mask) or unused != "zero":
        raise CheckError(f"helper {ordinal}: B.IOR is not [base, active-mask, zero]")
    bior_index = segment.index(bior)
    validate_generic_mask_flow(segment, compares, bior_index, mask, ordinal)

    body = "\n".join(inst.line for inst in segment)
    if "CUBE_M32" not in body or not re.search(r"C\.B\.DIMI\s+32", body):
        raise CheckError(f"helper {ordinal}: missing local 32-row CUBE_M32 geometry")


def atomic_segments(text: str) -> list[list[Instruction]]:
    instructions = parse(text)
    starts = [index for index, inst in enumerate(instructions) if inst.mnemonic == "B.SUBVIEW"]
    segments: list[list[Instruction]] = []
    for start in starts:
        end = next(
            (index for index in range(start + 1, len(instructions)) if instructions[index].mnemonic == "TSTORE"),
            None,
        )
        if end is not None:
            segments.append(instructions[start : end + 1])
    return segments


def validate(text: str) -> None:
    segments = atomic_segments(text)
    if len(segments) != 3:
        raise CheckError(f"expected exactly three histogram atomic sites, found {len(segments)}")
    for ordinal, segment in enumerate(segments):
        validate_segment(segment, ordinal)


def validate_ir(text: str) -> None:
    if re.search(
        r"atomicrmw|extractelement|insertelement|"
        r"@llvm\.linx\.experimental\.element\.(region|view)",
        text,
    ):
        raise CheckError("optimized Top-K IR contains residual element lowering")
    if "<32 x i8>" in text or "<32 x i16>" in text:
        raise CheckError("Top-K histogram index narrowed below B32")
    atomics = re.findall(
        r"call <32 x i32> "
        r"@llvm\.linx\.experimental\.ew\.mgather\.add\.masked[^\n]*",
        text,
    )
    if len(atomics) != 3:
        raise CheckError(f"expected three masked atomic intrinsics, found {len(atomics)}")
    tleas = re.findall(
        r"call <32 x i64> @llvm\.linx\.experimental\.ew\.tlea[^\n]*"
        r"i64 32\)",
        text,
    )
    if len(tleas) != 3:
        raise CheckError(f"expected three B32 TLEA intrinsics, found {len(tleas)}")
    if any("i64 25" not in call for call in tleas):
        raise CheckError("Top-K histogram index lost its U32 descriptor")


def expect_rejected(name: str, text: str) -> None:
    try:
        validate(text)
    except CheckError:
        return
    raise CheckError(f"negative canary unexpectedly accepted: {name}")


def expect_ir_rejected(name: str, text: str) -> None:
    try:
        validate_ir(text)
    except CheckError:
        return
    raise CheckError(f"IR negative canary unexpectedly accepted: {name}")


def self_test(text: str) -> None:
    validate(text)
    first_segment = atomic_segments(text)[0]
    atomic_index = next(
        index for index, inst in enumerate(first_segment)
        if inst.mnemonic == "BSTART.TLSU" and "MGATHER.ADD" in inst.operands
    )
    datr = next(
        inst for inst in first_segment[atomic_index + 1:]
        if inst.mnemonic == "B.DATR" and inst.raw & (1 << 13)
    )
    cleared = datr.raw & ~(1 << 13)
    expect_rejected(
        "missing Zero",
        text.replace(f"{datr.raw:08x}", f"{cleared:08x}", 1),
    )
    tlea_line = next(line for line in text.splitlines() if "BSTART.TEPL" in line and "TLEA" in line)
    expect_rejected("extra TLEA", text.replace(tlea_line, tlea_line + "\n" + tlea_line, 1))
    expect_rejected(
        "unrelated mask",
        re.sub(r"(B\.IOR\s+\[[^,]+,)[^,]+(,zero\].*ExecMaskPresent)", r"\1t#31\2", text, count=1),
    )
    expect_rejected(
        "non-GPR compare",
        re.sub(r"(TCMPS[^\n]*->)\s*[^\s,]+", r"\1M#1", text, count=1),
    )
    for ordinal, segment in enumerate(atomic_segments(text)):
        compares = [inst for inst in segment if inst.mnemonic == "TCMPS"]
        bior = next(inst for inst in segment
                    if inst.mnemonic == "B.IOR" and "ExecMaskPresent" in inst.operands)
        mask_binder = re.search(r"\[([^,]+),([^,]+),([^\]]+)\]", bior.operands)
        assert mask_binder is not None
        _base, mask, _unused = (part.strip() for part in mask_binder.groups())

        def overwrite_after(candidate: str, inst: Instruction, reg: str) -> str:
            writer = f"    fffe: 00000000\taddi zero, 7, ->{reg}"
            return candidate.replace(inst.line, inst.line + "\n" + writer, 1)

        for inst in compares:
            expect_rejected(f"site{ordinal}: overwritten compare",
                            overwrite_after(text, inst, destination(inst.operands)))
        bior_index = segment.index(bior)
        final_writer = next(
            (inst for inst in reversed(segment[:bior_index])
             if re.search(r"->\s*" + re.escape(mask) + r"(?:\s|,|$)", inst.operands)),
            None,
        )
        if final_writer is None:
            raise CheckError(f"site {ordinal}: mask has no local scalar producer")
        if compares:
            expect_rejected(
                f"site{ordinal}: overwritten final mask",
                overwrite_after(text, final_writer, mask),
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dis", required=True, type=Path)
    parser.add_argument("--ir", required=True, type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    text = args.dis.read_text(encoding="utf-8")
    ir_text = args.ir.read_text(encoding="utf-8")
    try:
        self_test(text) if args.self_test else validate(text)
        validate_ir(ir_text)
        if args.self_test:
            expect_ir_rejected(
                "residual scalar atomic",
                re.sub(
                    r"(define[^\n]*\n)",
                    r"\1  %bad = atomicrmw add ptr null, i32 1 monotonic\n",
                    ir_text,
                    count=1,
                ),
            )
            expect_ir_rejected(
                "narrow index fallback",
                ir_text.replace("<32 x i32>", "<32 x i8>", 1),
            )
    except CheckError as error:
        print(f"disassembly check failed: {error}")
        return 1
    print("disassembly check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
