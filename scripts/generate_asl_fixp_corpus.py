#!/usr/bin/env python3
"""Generate ASL sidecars and independent deterministic FIXP goldens."""

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


FP32_MODES = {
    "keep_acc", "keep_acc_relu", "s_qf_f32", "v_qf_f32",
    "rowmax", "rowmax_init", "groupmax_8", "groupmax_16",
    "groupmax_128", "rowgroup_maxabs", "bias", "acc", "mx",
    "mxbias", "mxacc", "gemv", "gemv_bias", "gemv_acc", "gemv_mx",
    "gemv_mx_bias", "gemv_mx_acc", "shared", "legacy3",
}
FP16_MODES = {
    "f16", "f16_relu", "s_deqf16", "s_qf_f16", "v_deqf16",
    "v_qf_f16", "f16_prelu", "f16_groupmax",
}
BF16_MODES = {
    "bf16", "bf16_relu", "s_qf_bf16", "s_qs_bf16", "v_qf_bf16",
    "v_qs_bf16",
}
I16_MODES = {"s_shifts16", "s_qf_s16", "v_shifts16", "v_qf_s16"}
I8_MODES = {
    "s_reqs8", "s_qf_s8", "v_reqs8", "v_qf_s8", "s8_relu",
    "s8_lrelu", "v_s8_relu", "s8_prelu", "s8_rowmax", "bias_s8",
    "acc_s8", "mx_s8", "gemv_s8", "gemv_mx_s8", "s8_shared",
    "vqf_s8_prelu",
}
S4X2_MODES = {"s_qf_s4", "v_qf_s4"}
HIF8_MODES = {"s_qf_hif8", "v_qf_hif8"}
FP8_MODES = {"s_qf_fp8", "v_qf_fp8"}

# With zero A/B operands, only keep-acc Bias/ACC carriers publish non-zero
# result bytes.  Their deterministic auxiliary storage repeats mk_desc(1,0,9),
# whose little-endian bytes form alternating FP32 values 0x00002000 and zero.
MATRIX_AUX_MODES = {"bias", "acc", "mxbias", "mxacc"}
GEMV_AUX_MODES = {"gemv_bias", "gemv_acc", "gemv_mx_bias", "gemv_mx_acc"}


def result_profile(mode: str, shape: list[int]) -> tuple[str, int, list[int]]:
    result_shape = [1, shape[1]] if mode.startswith("gemv") else shape[:2]
    element_count = result_shape[0] * result_shape[1]
    if mode in FP32_MODES:
        return "fp32", element_count * 4, result_shape
    if mode in FP16_MODES:
        return "fp16", element_count * 2, result_shape
    if mode in BF16_MODES:
        return "bf16", element_count * 2, result_shape
    if mode in I16_MODES:
        return "i16", element_count * 2, result_shape
    if mode in I8_MODES:
        return "i8", element_count, result_shape
    if mode in S4X2_MODES:
        return "s4x2", element_count, result_shape
    if mode in HIF8_MODES:
        return "hif8", element_count, result_shape
    if mode in FP8_MODES:
        return "fp8-e4m3", element_count, result_shape
    raise ValueError(f"unsupported active FIXP mode: {mode}")


def golden_bytes(mode: str, logical_size: int, shape: list[int]) -> bytes:
    """Evaluate the carrier's deterministic zero-input result on the host."""
    payload = bytearray(logical_size)
    if mode in MATRIX_AUX_MODES | GEMV_AUX_MODES:
        element_count = shape[0] * shape[1]
        for index in range(element_count):
            bits = 0x00002000 if index % 2 == 0 else 0
            struct.pack_into("<I", payload, index * 4, bits)
    return bytes(payload) + bytes(RESULT_BYTES - logical_size)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo", type=pathlib.Path, default=pathlib.Path(__file__).parents[1]
    )
    parser.add_argument("--elf-root", type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    add_identity_arguments(parser)
    arguments = parser.parse_args()

    repo = arguments.repo.resolve()
    elf_root = (arguments.elf_root or repo / "output" / "asl_corpus" / "microbenchmark").resolve()
    identities = resolve_identity(arguments, repo)
    coverage = json.loads((repo / "microbenchmark" / "coverage.json").read_text())
    active = [row for row in coverage["active"] if row["family"] == "fixp"]
    elf_by_name = discover_elfs(elf_root)
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    for case in active:
        case_id = case["name"]
        mode = case["mode"]
        source_path = repo / "microbenchmark" / "fixp" / "src" / "fixp_tmatmul.cpp"
        if not source_path.is_file():
            raise ValueError("missing FIXP producer source")
        elf = elf_by_name.get(case_id)
        if elf is None:
            raise ValueError(f"missing active FIXP ELF: {case_id}")
        dtype, logical_size, result_shape = result_profile(mode, case["shape"])
        if logical_size > RESULT_BYTES:
            raise ValueError(f"FIXP result exceeds uniform result carrier: {case_id}")
        golden = output / f"{case_id}.golden.bin"
        sidecar = output / f"{case_id}.sidecar.json"
        golden.write_bytes(golden_bytes(mode, logical_size, result_shape))

        header, symbols, segments = validate_elf_contract(arguments.llvm_readelf, elf)
        high_address = max(
            segment["address"] + segment["memsz"] for segment in segments
        )
        memory_bytes = next_power_of_two(max(0x40000, high_address + 0x4000))
        result_symbol = symbols["cross_model_result"]
        document = {
            "schema": "pto-asl-elf-sidecar-v1",
            "case_id": f"fixp.{case_id}",
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
                "tile_elements": 8192,
                "runtime_typecheck": "minimal",
            },
            "start": {
                "symbol": "main",
                "pc": symbols["main"]["Value"],
                "return_symbol": "cross_model_stop",
                "return_pc": symbols["cross_model_stop"]["Value"],
            },
            "execution": {
                "classification": "fixp",
                "stop_symbol": "cross_model_stop",
                "stop_pc": symbols["cross_model_stop"]["Value"],
                "stop_after_hits": 1,
                "max_steps": 1000000,
                "stack_top": memory_bytes - 0x1000,
            },
            "result": {
                "symbol": "cross_model_result",
                "size_symbol": "cross_model_result_size",
                "address": result_symbol["Value"],
                "size": RESULT_BYTES,
                "segments": [
                    {
                        "offset": 0,
                        "size": logical_size,
                        "dtype": dtype,
                        "shape": result_shape,
                        "comparison": "exact",
                    },
                    {
                        "offset": logical_size,
                        "size": RESULT_BYTES - logical_size,
                        "dtype": "u8",
                        "shape": [RESULT_BYTES - logical_size],
                        "comparison": "exact-zero-padding",
                    },
                ],
                "golden": {
                    "path": golden.name,
                    "sha256": sha256(golden),
                    "derivation": "host-language-zero-input-fixp-reference",
                },
            },
        }
        sidecar.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        rows.append("|".join([
            document["case_id"], str(elf), str(sidecar), str(golden)
        ]))

    (output / "cases.index").write_text("\n".join(rows) + "\n")
    print(f"generated FIXP ASL corpus: {len(rows)} cases in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
