#!/usr/bin/env python3
"""Generate ASL sidecars and independent deterministic vector goldens."""

from __future__ import annotations

import argparse
import json
import math
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


DTYPE_BYTES = {"fp16": 2, "fp32": 4, "i16": 2, "i32": 4}


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def float_result(dtype: str, value: float) -> float:
    try:
        packed = struct.pack("<e" if dtype == "fp16" else "<f", value)
    except OverflowError:
        return math.copysign(math.inf, value)
    return struct.unpack("<e" if dtype == "fp16" else "<f", packed)[0]


def signed_result(dtype: str, value: int) -> int:
    bits = DTYPE_BYTES[dtype] * 8
    raw = value & ((1 << bits) - 1)
    return raw - (1 << bits) if raw >= (1 << (bits - 1)) else raw


def cast(dtype: str, value: float | int) -> float | int:
    if dtype.startswith("fp"):
        return float_result(dtype, float(value))
    return signed_result(dtype, int(value))


def input_values(dtype: str, count: int) -> list[float | int]:
    return [cast(dtype, f32(f32(float(index)) * f32(0.1)))
            for index in range(count)]


def float_div(left: float, right: float) -> float:
    if right == 0.0:
        if left == 0.0:
            return math.nan
        return math.copysign(math.inf, left * right)
    return left / right


def float_rem(left: float, right: float) -> float:
    if right == 0.0 or math.isinf(left) or math.isnan(left) or math.isnan(right):
        return math.nan
    return math.fmod(left, right)


def integer_div(left: int, right: int) -> int:
    return math.trunc(left / right)


def integer_rem(left: int, right: int) -> int:
    return left - integer_div(left, right) * right


def binary(dtype: str, operation: str,
           left: float | int, right: float | int) -> float | int:
    base = operation.removeprefix("TPART") if operation.startswith("TPART") else operation[1:]
    if base == "ADD":
        value = left + right
    elif base == "SUB":
        value = left - right
    elif base == "MUL":
        value = left * right
    elif base == "DIV":
        value = (float_div(float(left), float(right)) if dtype.startswith("fp")
                 else integer_div(int(left), int(right)))
    elif base == "REM":
        value = (float_rem(float(left), float(right)) if dtype.startswith("fp")
                 else integer_rem(int(left), int(right)))
    elif base == "AND":
        value = int(left) & int(right)
    elif base == "OR":
        value = int(left) | int(right)
    elif base == "XOR":
        value = int(left) ^ int(right)
    elif base == "SHL":
        value = int(left) << (int(right) & (DTYPE_BYTES[dtype] * 8 - 1))
    elif base == "SHR":
        value = int(left) >> (int(right) & (DTYPE_BYTES[dtype] * 8 - 1))
    elif base == "MAX":
        value = max(left, right)
    elif base == "MIN":
        value = min(left, right)
    else:
        raise ValueError(f"unsupported vector binary operation: {operation}")
    return cast(dtype, value)


def unary(dtype: str, operation: str, value: float | int) -> float | int:
    if operation == "TABS":
        result = abs(value)
    elif operation == "TNOT":
        result = ~int(value)
    elif operation == "TNEG":
        result = -value
    elif operation == "TEXP":
        result = math.exp(float(value))
    elif operation == "TLOG":
        result = -math.inf if value == 0 else math.log(float(value))
    elif operation == "TRECIP":
        result = float_div(1.0, float(value))
    elif operation == "TSQRT":
        result = math.sqrt(float(value))
    elif operation == "TRSQRT":
        result = float_div(1.0, math.sqrt(float(value)))
    elif operation == "TRELU":
        result = max(value, 0)
    elif operation == "TCVT":
        result = value
    else:
        raise ValueError(f"unsupported vector unary operation: {operation}")
    return cast(dtype, result)


def pack_values(dtype: str, values: list[float | int]) -> bytes:
    code = {"fp16": "e", "fp32": "f", "i16": "h", "i32": "i"}[dtype]
    return b"".join(struct.pack("<" + code, value) for value in values)


def vector_values(operation: str, dtype: str,
                  shape: list[int]) -> tuple[list[float | int], list[int]]:
    rows, columns = shape
    count = rows * columns
    source = [cast(dtype, 2)] * max(count, rows * 32)
    other = [cast(dtype, 3 if operation == "TCONCAT" else 1)] * len(source)

    binary_operations = {
        "TADD", "TSUB", "TMUL", "TDIV", "TREM", "TAND", "TOR", "TXOR",
        "TSHL", "TSHR", "TMAX", "TMIN", "TPARTADD", "TPARTMUL",
        "TPARTMAX", "TPARTMIN",
    }
    unary_operations = {
        "TABS", "TNOT", "TNEG", "TEXP", "TLOG", "TRECIP", "TSQRT",
        "TRSQRT", "TRELU", "TCVT",
    }
    if operation in binary_operations:
        return ([binary(dtype, operation, source[index], other[index])
                 for index in range(count)], shape)
    if operation in unary_operations:
        return ([unary(dtype, operation, source[index])
                 for index in range(count)], shape)
    if operation.endswith("S") and operation != "TEXPANDS":
        scalar = cast(dtype, 0.5)
        base = operation[:-1]
        return ([binary(dtype, base, source[index], scalar)
                 for index in range(count)], shape)
    if operation == "TEXPANDS":
        return ([cast(dtype, 0.5)] * count, shape)
    if operation == "TCI":
        base = cast(dtype, 7)
        return ([cast(dtype, int(base) + index) for index in range(count)], shape)
    if operation == "TROWEXPAND":
        return ([source[row] for row in range(rows) for _ in range(columns)], shape)
    if operation == "TCOLEXPAND":
        return ([source[column] for _ in range(rows) for column in range(columns)], shape)
    if operation.startswith("TROWEXPAND") or operation.startswith("TCOLEXPAND"):
        row_axis = operation.startswith("TROW")
        suffix = operation[len("TROWEXPAND") if row_axis else len("TCOLEXPAND"):]
        result: list[float | int] = []
        for row in range(rows):
            for column in range(columns):
                broadcast = other[row] if row_axis else other[column]
                left = source[row * columns + column]
                if suffix == "EXPDIF":
                    difference = cast(dtype, left - broadcast)
                    result.append(cast(dtype, math.exp(float(difference))))
                else:
                    result.append(binary(dtype, "T" + suffix, left, broadcast))
        return result, shape
    if operation == "TCONCAT":
        source_columns = 32 // DTYPE_BYTES[dtype]
        result = []
        for row in range(rows):
            start = row * source_columns
            result.extend(source[start:start + source_columns])
            result.extend(other[start:start + source_columns])
        return result, [rows, source_columns * 2]
    raise ValueError(f"unsupported active vector operation: {operation}")


def vector_golden(operation: str, dtype: str,
                  shape: list[int]) -> tuple[bytes, list[int]]:
    values, result_shape = vector_values(operation, dtype, shape)
    payload = pack_values(dtype, values)
    if len(payload) > RESULT_BYTES:
        raise ValueError("vector result exceeds uniform result carrier")
    return payload + bytes(RESULT_BYTES - len(payload)), result_shape


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=pathlib.Path,
                        default=pathlib.Path(__file__).parents[1])
    parser.add_argument("--elf-root", type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    add_identity_arguments(parser)
    arguments = parser.parse_args()

    repo = arguments.repo.resolve()
    elf_root = (arguments.elf_root or repo / "output" / "asl_corpus" / "microbenchmark").resolve()
    identities = resolve_identity(arguments, repo)
    coverage = json.loads((repo / "microbenchmark" / "coverage.json").read_text())
    active = [row for row in coverage["active"] if row["family"] == "vector"]
    elf_by_name = discover_elfs(elf_root)
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    for case in active:
        case_id = case["name"]
        elf = elf_by_name.get(case_id)
        if elf is None:
            raise ValueError(f"missing active vector ELF: {case_id}")
        dtype, shape, operation = case["dtype"], case["shape"], case["operation"]
        source_path = repo / "microbenchmark" / "vector" / "src" / f"{case_id}.cpp"
        if not source_path.is_file():
            raise ValueError(f"missing active vector source: {case_id}")
        golden = output / f"{case_id}.golden.bin"
        sidecar = output / f"{case_id}.sidecar.json"
        golden_bytes, result_shape = vector_golden(operation, dtype, shape)
        golden.write_bytes(golden_bytes)

        header, symbols, segments = validate_elf_contract(arguments.llvm_readelf, elf)
        result_symbol = symbols["cross_model_result"]
        size_symbol = symbols["cross_model_result_size"]
        if result_symbol["Size"] != RESULT_BYTES or size_symbol["Value"] != RESULT_BYTES:
            raise ValueError(f"invalid result symbol contract: {elf}")
        high_address = max(segment["address"] + segment["memsz"] for segment in segments)
        memory_bytes = next_power_of_two(max(0x40000, high_address + 0x4000))
        logical_size = math.prod(result_shape) * DTYPE_BYTES[dtype]
        document = {
            "schema": "pto-asl-elf-sidecar-v1",
            "case_id": f"vector.{case_id}",
            "identity": identities,
            "source": {"path": source_path.relative_to(repo).as_posix(),
                       "sha256": sha256(source_path)},
            "elf": {
                "path": elf.name, "sha256": sha256(elf), "machine": 0xE9,
                "entry": header["Entry"], "segments": segments,
            },
            "model": {
                "profile": "bounded-reference-v1", "pe_count": 1,
                "memory_bytes": memory_bytes, "tile_elements": 8192,
                "runtime_typecheck": "minimal",
            },
            "start": {
                "symbol": "main", "pc": symbols["main"]["Value"],
                "return_symbol": "cross_model_stop",
                "return_pc": symbols["cross_model_stop"]["Value"],
            },
            "execution": {
                "classification": "vector",
                "stop_symbol": "cross_model_stop",
                "stop_pc": symbols["cross_model_stop"]["Value"],
                "stop_after_hits": 1, "max_steps": 100000,
                "stack_top": memory_bytes - 0x1000,
            },
            "result": {
                "symbol": "cross_model_result",
                "size_symbol": "cross_model_result_size",
                "address": result_symbol["Value"], "size": RESULT_BYTES,
                "segments": [
                    {"offset": 0, "size": logical_size, "dtype": dtype,
                     "shape": result_shape, "comparison": "exact"},
                    {"offset": logical_size, "size": RESULT_BYTES - logical_size,
                     "dtype": "u8", "shape": [RESULT_BYTES - logical_size],
                     "comparison": "exact-zero-padding"},
                ],
                "golden": {
                    "path": golden.name, "sha256": sha256(golden),
                    "derivation": "host-language-vector-reference",
                },
            },
        }
        sidecar.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        rows.append("|".join(
            [document["case_id"], str(elf), str(sidecar), str(golden)]))

    (output / "cases.index").write_text("\n".join(rows) + "\n")
    print(f"generated vector ASL corpus: {len(rows)} cases in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
