#!/usr/bin/env python3
"""Keep TopK arithmetic expressed in scalar C++ under element pragmas."""

import re
from pathlib import Path


def validate(source: str, pragma_count: int = 4) -> None:
    # Ignore comments so documentation can name the APIs being replaced.
    code = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S)
    tile_calls = set(re.findall(r"\b(T[A-Z][A-Z0-9_]*)\s*(?:<[^;\n]*?>)?\s*\(", code))
    if tile_calls - {"TLOAD", "TMOV", "TSTORE", "TPARTELEMENT", "TPARTVIEW"}:
        raise ValueError("TopK must express element arithmetic as scalar C++")
    if len(re.findall(r"^\s*#pragma pto element for\s*$", code, re.M)) != pragma_count:
        raise ValueError("TopK must retain its element histogram loops")


def main() -> None:
    kernel = (Path(__file__).resolve().parents[4]
              / "kernels/basic_op/topk/element_atomic_topk.hpp")
    driver = Path(__file__).resolve().parents[1] / "src/element_atomic_topk.cpp"
    for path, pragma_count in ((kernel, 4), (driver, 1)):
        source = path.read_text(encoding="utf-8")
        validate(source, pragma_count)
        mutations = [source + f"\n{opcode}(values, view, 0);"
                     for opcode in ("TADDS", "TSHRS", "TANDS", "TADD",
                                    "TSHR", "TMULS", "TXOR", "TCMPS")]
        mutations.append(source.replace("#pragma pto element for", "", 1))
        for mutation in mutations:
            try:
                validate(mutation, pragma_count)
            except ValueError:
                continue
            raise AssertionError("source regression checker accepted a mutation")
    print("TopK scalar source check and mutation canaries PASS")


if __name__ == "__main__":
    main()
