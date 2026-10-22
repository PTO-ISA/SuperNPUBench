"""Strict parsing and regexes for canonical Tile BSTART spellings."""

from __future__ import annotations

from dataclasses import dataclass
import re


TLSU_OPERATIONS = frozenset({
    "TIMG2COL", "TLOAD", "TSTORE", "TMOV", "TPREFETCH", "GMOV",
    "MGATHER", "MSCATTER", "MGATHER.MASK", "MSCATTER.MASK",
    "MGATHER.CAS", "MGATHER.EXCH", "MGATHER.MAX", "MGATHER.MIN",
    "MGATHER.ADD", "MGATHER.INC", "MGATHER.DEC", "MGATHER.AND",
    "MGATHER.OR", "MGATHER.XOR", "MSCATTER.MAX", "MSCATTER.MIN",
    "MSCATTER.ADD", "MSCATTER.INC", "MSCATTER.DEC", "MSCATTER.AND",
    "MSCATTER.OR", "MSCATTER.XOR", "MSCATTER.POPC",
})

CUBE_OPERATIONS = frozenset({
    "TMATMUL", "TMATMUL.BIAS", "TMATMUL.ACC",
    "TMATMULMX", "TMATMULMX.BIAS", "TMATMULMX.ACC",
    "TGEMV", "TGEMV.BIAS", "TGEMV.ACC",
    "TGEMVMX", "TGEMVMX.BIAS", "TGEMVMX.ACC",
})

OPERATIONS_BY_ENGINE = {"TLSU": TLSU_OPERATIONS, "CUBE": CUBE_OPERATIONS}
GENERIC_ALIASES = {("TLSU", "13"): "GMOV", ("TLSU", "28"): "TIMG2COL"}

OBJDUMP_LINE = re.compile(
    r"^(?P<prefix>\s*[0-9a-f]+:\s+(?:[0-9a-f]{4,8}\s+)+)"
    r"(?P<mnemonic>[^\s]+)(?:\s+(?P<operands>.*))?$",
    re.IGNORECASE,
)
BARE_LINE = re.compile(
    r"^(?P<prefix>\s*)(?P<mnemonic>BSTART\.[^\s]+)"
    r"(?:\s+(?P<operands>.*))?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BStart:
    engine: str
    operation: str
    dtype: str


def _validate(engine: str, operation: str) -> tuple[str, str]:
    engine = engine.upper()
    operation = operation.upper()
    if engine not in OPERATIONS_BY_ENGINE:
        raise ValueError(f"unsupported native BSTART engine: {engine}")
    if operation not in OPERATIONS_BY_ENGINE[engine]:
        raise ValueError(f"{operation} is not a {engine} operation")
    return engine, operation


def parse(mnemonic: str, operands: str) -> BStart | None:
    """Parse one llvm-objdump instruction without accepting near matches."""
    mnemonic = mnemonic.upper()
    operands = operands.upper().strip()
    if mnemonic in ("BSTART.TLSU", "BSTART.CUBE"):
        engine = mnemonic.removeprefix("BSTART.")
        match = re.fullmatch(r"([A-Z0-9.]+),\s*([A-Z0-9_]+)", operands)
        if match is None:
            return None
        operation = GENERIC_ALIASES.get((engine, match.group(1)), match.group(1))
        if operation not in OPERATIONS_BY_ENGINE[engine]:
            return None
        return BStart(engine, operation, match.group(2))
    if not mnemonic.startswith("BSTART."):
        return None
    operation = mnemonic.removeprefix("BSTART.")
    matches = [engine for engine, operations in OPERATIONS_BY_ENGINE.items()
               if operation in operations]
    if len(matches) != 1 or re.fullmatch(r"[A-Z0-9_]+", operands) is None:
        return None
    return BStart(matches[0], operation, operands)


def normalize_disassembly(text: str) -> str:
    """Normalize only recognized TLSU/CUBE headers to the generic spelling."""
    normalized: list[str] = []
    for line in text.splitlines(keepends=True):
        ending = "\n" if line.endswith("\n") else ""
        content = line[:-1] if ending else line
        match = OBJDUMP_LINE.match(content) or BARE_LINE.match(content)
        if match is None:
            normalized.append(line)
            continue
        header = parse(match.group("mnemonic"), match.group("operands") or "")
        if header is None:
            normalized.append(line)
            continue
        normalized.append(
            f'{match.group("prefix")}BSTART.{header.engine} '
            f'{header.operation}, {header.dtype}{ending}'
        )
    return "".join(normalized)


def bstarts(text: str) -> list[BStart]:
    """Return recognized headers while ignoring unrelated instructions."""
    result: list[BStart] = []
    for line in text.splitlines():
        match = OBJDUMP_LINE.match(line) or BARE_LINE.match(line)
        if match is None:
            continue
        header = parse(match.group("mnemonic"), match.group("operands") or "")
        if header is not None:
            result.append(header)
    return result


def contains(text: str, engine: str, operation: str,
             dtype: str | None = None) -> bool:
    engine, operation = _validate(engine, operation)
    dtype = dtype.upper() if dtype else None
    return any(header.engine == engine and header.operation == operation and
               (dtype is None or header.dtype == dtype)
               for header in bstarts(text))
