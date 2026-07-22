#!/usr/bin/env python3
"""Build the checked-in DeepSeek migration manifest from translated headers."""

from __future__ import annotations

import json
import re
from pathlib import Path


REVISION = "36d9e45d38e204ebb87e6f6e833821eee0482fe5"
SOURCES = {
    "engram": (
        "engram_fused_weight_kernel.py", "engram_gate_kernel.py",
        "engram_grad_w_reduce_kernel.py", "engram_hash_kernel.py",
    ),
    "mhc": (
        "expand_kernel.py", "head_compute_mix_kernel.py",
        "multilayer_recompute_kernel.py", "norm_fn_kernel.py", "post_kernel.py",
        "pre_apply_mix_kernel.py", "pre_big_fuse_kernel.py",
        "pre_split_mixes_kernel.py", "sinkhorn_kernel.py",
    ),
    "moe": (
        "aux_fi_kernel.py", "expand_to_fused_kernel.py",
        "get_fused_mapping_kernel.py", "group_count_kernel.py",
        "inplace_unique_group_indices_kernel.py", "mask_indices_by_tp_kernel.py",
        "normalize_weight_kernel.py", "reduce_fused_kernel.py",
        "top2_sum_gate_kernel.py", "topk_gate_kernel.py",
        "topk_sum_and_topk_group_idx_kernel.py",
    ),
    "quant": (
        "cast_back_e5m6_kernel.py", "cast_back_kernel.py",
        "per_block_cast_kernel.py", "per_block_cast_lossless_kernel.py",
        "per_channel_cast_and_transpose_kernel.py", "per_channel_cast_fused_kernel.py",
        "per_channel_cast_kernel.py", "per_token_cast_kernel.py",
        "per_token_cast_to_e5m6_kernel.py",
        "swiglu_backward_and_per_token_cast_kernel.py",
        "swiglu_forward_and_per_channel_cast_and_transpose_kernel.py",
        "swiglu_forward_and_per_token_cast_kernel.py",
    ),
    "transpose": ("batched_transpose_kernel.py",),
}

TITLE_OVERRIDES = {
    "engram_fused_weight": "Fuse Engram embedding weights tile by tile",
    "engram_gate": "Apply and differentiate the Engram gate",
    "engram_grad_w_reduce": "Reduce Engram weight gradients across tokens",
    "engram_hash": "Compute tiled n-gram hash indices",
    "expand": "Expand hidden states for multi-head composition",
    "head_compute_mix": "Mix per-head activations and gradients",
    "multilayer_recompute": "Recompute multi-layer hidden-state products",
    "norm_fn": "Normalize and merge multi-head features",
    "post": "Apply the post-composition transform",
    "pre_apply_mix": "Apply the pre-composition mixing matrix",
    "pre_big_fuse": "Fuse normalization, mixing, and residual preparation",
    "pre_split_mixes": "Split pre-composition mixing coefficients",
    "sinkhorn": "Normalize routing weights with Sinkhorn iterations",
    "aux_fi": "Compute auxiliary-loss-free expert statistics",
    "expand_to_fused": "Expand token activations into expert-major storage",
    "get_fused_mapping": "Build token-to-expert routing maps",
    "group_count": "Count routed tokens per expert group",
    "inplace_unique_group_indices": "Make expert-group indices unique in place",
    "mask_indices_by_tp": "Mask and localize tensor-parallel expert indices",
    "normalize_weight": "Normalize top-k routing weights",
    "reduce_fused": "Reduce expert outputs back to token order",
    "top2_sum_gate": "Compute top-two group sums and gates",
    "topk_gate": "Select and gate top-k expert groups",
    "topk_sum_and_topk_group_idx": "Reduce group scores and select top-k groups",
    "cast_back_e5m6": "Restore E5M6-encoded activations with tile scaling",
    "cast_back": "Restore quantized activations from scaling factors",
    "per_block_cast": "Quantize activations with per-block scales",
    "per_block_cast_lossless": "Pack lossless per-block quantization data",
    "per_channel_cast_and_transpose": "Quantize per channel while transposing tiles",
    "per_channel_cast_fused": "Fuse per-channel quantization stages",
    "per_channel_cast": "Quantize activations with per-channel scales",
    "per_token_cast": "Quantize activations with per-token scales",
    "per_token_cast_to_e5m6": "Encode per-token scales in E5M6 form",
    "swiglu_backward_and_per_token_cast": "Differentiate SwiGLU and quantize token gradients",
    "swiglu_forward_and_per_channel_cast_and_transpose": "Fuse SwiGLU, channel quantization, and transpose",
    "swiglu_forward_and_per_token_cast": "Fuse SwiGLU with per-token quantization",
    "batched_transpose": "Transpose a batch of tiled matrices",
}

IMPLEMENTATION_OVERRIDES = {
    ("engram", "engram_fused_weight"): ("fused_weight_pto.hpp",),
    ("mhc", "expand"): ("expand_to_mhc_pto.hpp", "expand_to_mhc_bwd_pto.hpp"),
    ("moe", "aux_fi"): ("group_count_aux_fi_pto.hpp",),
    ("moe", "group_count"): ("group_count_aux_fi_pto.hpp",),
    ("quant", "per_channel_cast"): ("per_channel_cast_kernel_pto.hpp",),
    ("quant", "swiglu_forward_and_per_token_cast"): ("swiglu_fused_cast_pto.hpp",),
}

FAMILY_TEXT = {
    "engram": {
        "summary": "This Engram translation keeps lookup, gating, or gradient work in typed tiles.",
        "algorithm": "Load the thread-owned token and weight fragments, apply the documented gate, hash, or reduction stages, then store the valid result region.",
        "partition": "The block partitions tokens across threads. Each active thread owns contiguous, non-overlapping token rows with at least 128 bytes of valid tile data.",
        "tile_flow": "Global tensor views feed thread-local tiles. Arithmetic and reductions create new tile versions before the final result is stored.",
    },
    "mhc": {
        "summary": "This multi-head composition translation expresses mixing, normalization, and recompute stages as tile dataflow.",
        "algorithm": "Load hidden-state and coefficient panels, compose matrix, reduction, and pointwise operations, and write the resulting head or gradient panel.",
        "partition": "Blocks partition the outer token range; threads partition head or hidden dimensions into disjoint fragments of at least 128 bytes.",
        "tile_flow": "Global panels enter local tiles, matrix and reduction stages produce typed intermediates, and block-shared tiles are used when a value must be visible across the group.",
    },
    "moe": {
        "summary": "This mixture-of-experts translation keeps routing scores, indices, counts, and expert values in tile operations.",
        "algorithm": "Load a routing fragment, apply selection, counting, normalization, gather, or reduction stages, and store the fragment in token-major or expert-major order.",
        "partition": "Blocks partition tokens or experts; threads own disjoint routing fragments of at least 128 bytes and derive their coordinates from execution identity.",
        "tile_flow": "Scores and indices are loaded into typed tiles, transformed through the routing pipeline, and committed through regular or indexed tile memory operations.",
    },
    "quant": {
        "summary": "This quantization translation derives scaling factors and converted values without scalar tensor loops.",
        "algorithm": "Load an activation fragment, compute an absolute maximum with tile reductions, derive the scale, convert the valid values, and store data and scale tiles.",
        "partition": "Blocks partition outer rows; threads own non-overlapping token, channel, or block fragments whose valid payload is at least 128 bytes.",
        "tile_flow": "Activation tiles are converted to the accumulation type, reduced for scale calculation, transformed, converted to the output type, and stored with their scale tiles.",
    },
    "transpose": {
        "summary": "This layout translation transposes batched matrices entirely through typed tiles.",
        "algorithm": "Load one thread-owned matrix panel, transpose its tile axes, and store the corresponding output panel.",
        "partition": "Blocks partition batch elements and tile rows; threads own disjoint source and destination panels of at least 128 bytes.",
        "tile_flow": "A global source view produces a local tile, the layout operation creates the transposed tile version, and a global destination view receives the valid region.",
    },
}

CALL = re.compile(r"\b((?:T|M)[A-Z][A-Z0-9_]*)\s*(?:<[^;{}()]*>)?\s*\(")
FUNCTION = re.compile(r"\b(?:void|auto)\s+([a-zA-Z_]\w*)\s*\(")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    base = root / "benchmark/one-level-arch/kernels/deepseek"
    catalog = json.loads((root / "scripts/data/intrinsic-catalog.json").read_text())
    allowed = {str(entry["name"]) for entry in catalog["operations"]}
    entries = []
    inventory = []
    for family, files in SOURCES.items():
        for filename in files:
            stem = filename.removesuffix("_kernel.py")
            source = f"tile_kernels/{family}/{filename}"
            inventory.append(source)
            implementation_names = IMPLEMENTATION_OVERRIDES.get(
                (family, stem), (f"{stem}_pto.hpp",)
            )
            implementations = [
                f"benchmark/one-level-arch/kernels/deepseek/{family}/{name}"
                for name in implementation_names
            ]
            texts = [(root / path).read_text(encoding="utf-8") for path in implementations]
            code = "\n".join(re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.DOTALL) for text in texts)
            functions = list(dict.fromkeys(FUNCTION.findall(code)))
            intrinsics = sorted(set(CALL.findall(code)) & allowed)
            entry = {
                "source": source,
                "family": family,
                "title": TITLE_OVERRIDES[stem],
                **FAMILY_TEXT[family],
                "implementations": implementations,
                "functions": functions,
                "intrinsics": intrinsics,
                "verification": "source-validated",
            }
            entries.append(entry)

    payload = {
        "schema": 1,
        "upstream": {
            "repository": "https://github.com/deepseek-ai/TileKernels",
            "revision": REVISION,
        },
        "minimum_thread_fragment_bytes": 128,
        "source_inventory": inventory,
        "kernel_modules": entries,
    }
    target = base / "migration.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {target.relative_to(root)} with {len(entries)} modules.")


if __name__ == "__main__":
    main()
