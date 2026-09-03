#!/usr/bin/env python3
"""Generate ASL sidecars and independent deterministic goldens for TLSU cases."""

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


def dtype_value(dtype: str, index: int) -> bytes:
    value = 2
    if dtype == "fp16":
        return struct.pack("<e", value)
    if dtype == "fp32":
        return struct.pack("<f", value)
    if dtype == "i32":
        return struct.pack("<i", int(value))
    raise ValueError(f"unsupported memory dtype: {dtype}")


def golden_bytes(case_id: str, dtype: str, element_count: int) -> bytes:
    source = [dtype_value(dtype, index) for index in range(element_count)]
    operation = case_id.split("_", 1)[0]
    if operation in {"tload", "tstore"}:
        result = source
    elif operation == "mgather":
        result = [source[(index * 7) % element_count] for index in range(element_count)]
    elif operation == "mscatter":
        result = [bytes(len(source[0])) for _ in range(element_count)]
        for index, value in enumerate(source):
            result[(index * 7) % element_count] = value
    else:
        raise ValueError(f"unsupported active memory operation: {operation}")
    payload = b"".join(result)
    if len(payload) > RESULT_BYTES:
        raise ValueError(f"memory result exceeds carrier: {case_id}")
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
    active = [row for row in coverage["active"] if row["family"] == "memory"]
    elf_by_name = discover_elfs(elf_root)
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for case in active:
        case_id = case["name"]
        elf = elf_by_name.get(case_id)
        if elf is None:
            raise ValueError(f"missing active memory ELF: {case_id}")
        dtype = case["dtype"]
        source_path = repo / "microbenchmark" / "memory" / "src" / f"{case_id}.cpp"
        if not source_path.is_file():
            raise ValueError(f"missing active memory source: {case_id}")
        shape = case["shape"]
        element_count = shape[0] * shape[1]
        golden = output / f"{case_id}.golden.bin"
        sidecar = output / f"{case_id}.sidecar.json"
        golden.write_bytes(golden_bytes(case_id, dtype, element_count))
        header, symbols, segments = validate_elf_contract(arguments.llvm_readelf, elf)
        high_address = max(segment["address"] + segment["memsz"] for segment in segments)
        memory_bytes = next_power_of_two(max(0x20000, high_address + 0x2000))
        logical_size = element_count * {"fp16": 2, "fp32": 4, "i32": 4}[dtype]
        result_symbol = symbols["cross_model_result"]
        document = {
            "schema": "pto-asl-elf-sidecar-v1",
            "case_id": f"memory.{case_id}",
            "identity": identities,
            "source": {"path": source_path.relative_to(repo).as_posix(),
                       "sha256": sha256(source_path)},
            "elf": {
                "path": elf.name,
                "sha256": sha256(elf),
                "machine": 0xE9,
                "entry": header["Entry"],
                "segments": segments,
            },
            "model": {
                "profile": "bounded-reference-v1",
                "pe_count": 1,
                "memory_bytes": memory_bytes,
                "tile_elements": element_count,
                "runtime_typecheck": "minimal",
            },
            "start": {
                "symbol": "main",
                "pc": symbols["main"]["Value"],
                "return_symbol": "cross_model_stop",
                "return_pc": symbols["cross_model_stop"]["Value"],
            },
            "execution": {
                "classification": "memory",
                "stop_symbol": "cross_model_stop",
                "stop_pc": symbols["cross_model_stop"]["Value"],
                "stop_after_hits": 1,
                "max_steps": 50000,
                "stack_top": memory_bytes - 0x1000,
            },
            "result": {
                "symbol": "cross_model_result",
                "size_symbol": "cross_model_result_size",
                "address": result_symbol["Value"],
                "size": RESULT_BYTES,
                "segments": [
                    {"offset": 0, "size": logical_size, "dtype": dtype,
                     "shape": shape, "comparison": "exact"},
                    {"offset": logical_size, "size": RESULT_BYTES - logical_size,
                     "dtype": "u8", "shape": [RESULT_BYTES - logical_size],
                     "comparison": "exact-zero-padding"},
                ],
                "golden": {"path": golden.name, "sha256": sha256(golden),
                           "derivation": "host-language-tlsu-reference"},
            },
        }
        sidecar.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        rows.append("|".join([document["case_id"], str(elf), str(sidecar), str(golden)]))
    (output / "cases.index").write_text("\n".join(rows) + "\n")
    print(f"generated memory ASL corpus: {len(rows)} cases in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
