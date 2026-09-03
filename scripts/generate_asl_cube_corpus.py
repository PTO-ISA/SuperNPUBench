#!/usr/bin/env python3
"""Generate ASL sidecars and independent deterministic CUBE goldens."""

from __future__ import annotations

import argparse
import json
import pathlib
import struct

from generate_asl_scalar_corpus import (
    RESULT_BYTES,
    add_identity_arguments,
    discover_elfs,
    next_power_of_two,
    resolve_identity,
    sha256,
    validate_elf_contract,
)


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def input_value(dtype: str, index: int) -> float | int:
    value = 1 if dtype == "i8" else 0.25
    if dtype == "fp32":
        return value
    if dtype == "fp16":
        return struct.unpack("<e", struct.pack("<e", value))[0]
    if dtype == "i8":
        raw = int(value) & 0xFF
        return raw - 256 if raw >= 128 else raw
    if dtype == "bf16":
        return 0.0
    raise ValueError(f"unsupported cube dtype: {dtype}")


def cube_golden(operation: str, dtype: str, shape: list[int]) -> bytes:
    m_size, n_size, k_size = shape
    if dtype == "bf16":
        payload = bytes(m_size * n_size * 4)
        return payload + bytes(RESULT_BYTES - len(payload))
    left = [input_value(dtype, index) for index in range(m_size * k_size)]
    right = [input_value(dtype, index) for index in range(k_size * n_size)]
    integral = dtype == "i8"
    result: list[float | int] = []
    for m_index in range(m_size):
        for n_index in range(n_size):
            value: float | int = 1 if operation == "TMATMUL_ACC" else 0
            if operation == "TMATMUL_BIAS":
                value = 1
            for k_index in range(k_size):
                product = left[m_index * k_size + k_index] * right[k_index * n_size + n_index]
                if integral:
                    value = int(value) + int(product)
                else:
                    value = f32(f32(float(value)) + f32(float(product)))
            result.append(value)
    if integral:
        payload = b"".join(struct.pack("<i", int(value)) for value in result)
    else:
        payload = b"".join(struct.pack("<f", f32(float(value))) for value in result)
    if len(payload) > RESULT_BYTES:
        raise ValueError("cube result exceeds uniform result carrier")
    return payload + bytes(RESULT_BYTES - len(payload))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path(__file__).parents[1])
    parser.add_argument("--elf-root", type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    add_identity_arguments(parser)
    arguments = parser.parse_args()

    repo = arguments.repo.resolve()
    elf_root = (arguments.elf_root or repo / "output" / "asl_corpus" / "microbenchmark").resolve()
    identities = resolve_identity(arguments, repo)
    coverage = json.loads((repo / "microbenchmark" / "coverage.json").read_text())
    active = [row for row in coverage["active"] if row["family"] == "cube"]
    elf_by_name = discover_elfs(elf_root)
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for case in active:
        case_id = case["name"]
        elf = elf_by_name.get(case_id)
        if elf is None:
            raise ValueError(f"missing active cube ELF: {case_id}")
        dtype, shape, operation = case["dtype"], case["shape"], case["operation"]
        source_path = repo / "microbenchmark" / "cube" / "src" / f"{case_id}.cpp"
        if not source_path.is_file():
            raise ValueError(f"missing active CUBE source: {case_id}")
        golden = output / f"{case_id}.golden.bin"
        sidecar = output / f"{case_id}.sidecar.json"
        golden.write_bytes(cube_golden(operation, dtype, shape))
        header, symbols, segments = validate_elf_contract(arguments.llvm_readelf, elf)
        high_address = max(segment["address"] + segment["memsz"] for segment in segments)
        memory_bytes = next_power_of_two(max(0x40000, high_address + 0x4000))
        logical_size = shape[0] * shape[1] * 4
        result_symbol = symbols["cross_model_result"]
        document = {
            "schema": "pto-asl-elf-sidecar-v1",
            "case_id": f"cube.{case_id}",
            "identity": identities,
            "source": {"path": source_path.relative_to(repo).as_posix(),
                       "sha256": sha256(source_path)},
            "elf": {"path": elf.name, "sha256": sha256(elf), "machine": 0xE9,
                    "entry": header["Entry"], "segments": segments},
            "model": {"profile": "bounded-reference-v1", "pe_count": 1,
                      "memory_bytes": memory_bytes, "tile_elements": 8192,
                      "runtime_typecheck": "minimal"},
            "start": {"symbol": "main", "pc": symbols["main"]["Value"],
                      "return_symbol": "cross_model_stop",
                      "return_pc": symbols["cross_model_stop"]["Value"]},
            "execution": {"classification": "cube",
                          "stop_symbol": "cross_model_stop",
                          "stop_pc": symbols["cross_model_stop"]["Value"],
                          "stop_after_hits": 1, "max_steps": 1000000,
                          "stack_top": memory_bytes - 0x1000},
            "result": {
                "symbol": "cross_model_result", "size_symbol": "cross_model_result_size",
                "address": result_symbol["Value"], "size": RESULT_BYTES,
                "segments": [
                    {"offset": 0, "size": logical_size,
                     "dtype": "i32" if dtype == "i8" else "fp32",
                     "shape": shape[:2], "comparison": "exact"},
                    {"offset": logical_size, "size": RESULT_BYTES - logical_size,
                     "dtype": "u8", "shape": [RESULT_BYTES - logical_size],
                     "comparison": "exact-zero-padding"},
                ],
                "golden": {"path": golden.name, "sha256": sha256(golden),
                           "derivation": "host-language-cube-reference"},
            },
        }
        sidecar.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        rows.append("|".join([document["case_id"], str(elf), str(sidecar), str(golden)]))
    (output / "cases.index").write_text("\n".join(rows) + "\n")
    print(f"generated cube ASL corpus: {len(rows)} cases in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
