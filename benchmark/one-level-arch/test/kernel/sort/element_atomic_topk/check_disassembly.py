#!/usr/bin/env python3
"""Validate the ordered native lowering of both histogram helpers."""

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


def validate_segment(segment: list[Instruction], ordinal: int) -> None:
    def positions(mnemonic: str) -> list[int]:
        return [index for index, inst in enumerate(segment) if inst.mnemonic == mnemonic]

    tci = positions("BSTART.TEPL")
    tci = [index for index in tci if re.search(r"\bTCI,\s*U32\b", segment[index].operands)]
    compares = positions("TCMPS")
    scalar_and = positions("AND")
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
    if len(compares) != 2:
        raise CheckError(f"helper {ordinal}: expected exactly two TCMPS GPR producers")
    if len(scalar_and) != 1:
        raise CheckError(f"helper {ordinal}: expected exactly one scalar predicate AND")
    if len(tlea) != 1:
        raise CheckError(f"helper {ordinal}: expected exactly one TLEA")
    if len(atomic) != 1:
        raise CheckError(f"helper {ordinal}: expected exactly one MGATHER.ADD")

    and_index = scalar_and[0]
    if not (tci[0] < compares[0] < compares[1] < and_index < tlea[0] < atomic[0]):
        raise CheckError(f"helper {ordinal}: lowering order is not TCI/TCMPS/AND/TLEA/atomic")

    compare_dsts = [destination(segment[index].operands) for index in compares]
    if any(not GPR.fullmatch(value) for value in compare_dsts):
        raise CheckError(f"helper {ordinal}: TCMPS destination is not a scalar GPR")
    and_match = re.search(
        r"^([^,]+),\s*([^,]+),\s*->\s*([^\s,]+)", segment[and_index].operands
    )
    if not and_match:
        raise CheckError(f"helper {ordinal}: malformed scalar AND")
    and_sources = {and_match.group(1).strip(), and_match.group(2).strip()}
    and_dst = and_match.group(3).strip()
    if and_sources != set(compare_dsts) or not GPR.fullmatch(and_dst):
        raise CheckError(f"helper {ordinal}: AND does not combine the two TCMPS GPRs")

    atomic_index = atomic[0]
    datr = next(
        (inst for inst in segment[atomic_index + 1 :] if inst.mnemonic == "B.DATR"),
        None,
    )
    if datr is None or (datr.raw & (1 << 13)) == 0:
        raise CheckError(f"helper {ordinal}: MGATHER.ADD does not encode Zero=1")
    bior = next(
        (inst for inst in segment[atomic_index + 1 :] if inst.mnemonic == "B.IOR"),
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
    if not GPR.fullmatch(base) or mask != and_dst or unused != "zero":
        raise CheckError(f"helper {ordinal}: B.IOR is not [base, AND-mask, zero]")

    body = "\n".join(inst.line for inst in segment)
    if "CUBE_M32" not in body or not re.search(r"C\.B\.DIMI\s+32", body):
        raise CheckError(f"helper {ordinal}: missing local 32-row CUBE_M32 geometry")


def validate(text: str) -> None:
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
    if len(segments) != 2:
        raise CheckError(f"expected exactly two histogram helper bodies, found {len(segments)}")
    for ordinal, segment in enumerate(segments):
        validate_segment(segment, ordinal)


def expect_rejected(name: str, text: str) -> None:
    try:
        validate(text)
    except CheckError:
        return
    raise CheckError(f"negative canary unexpectedly accepted: {name}")


def self_test(text: str) -> None:
    validate(text)
    expect_rejected("missing Zero", text.replace("19f03ea3", "19f01ea3", 1))
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dis", required=True, type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    text = args.dis.read_text(encoding="utf-8")
    try:
        self_test(text) if args.self_test else validate(text)
    except CheckError as error:
        print(f"disassembly check failed: {error}")
        return 1
    print("disassembly check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
