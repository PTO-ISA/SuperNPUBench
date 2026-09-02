#!/usr/bin/env python3

import argparse
import json
import random
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import FrozenSet, Optional, Tuple

from qsmla_reference import qsmla_reference


@dataclass(frozen=True)
class QsmlaCase:
    name: str
    b: int
    s1: int
    s2: int
    n1: int
    n2: int
    d: int
    k: int
    k1: Optional[int] = None
    tm: int = 32
    tk: int = 32
    td: int = 64
    win_left: int = 1
    win_right: int = 1
    softmax_scale: float = 0.125
    source: str = "supernpubench:stage0-smoke"
    mode: str = "SWA"
    q_layout: str = "BSND"
    kv_layout: str = "BSND"
    logical_dtype: str = "fp16"
    source_storage_dtype: str = "fp16"
    stage0_compute_dtype: str = "fp16"
    enable_stage: str = "stage0_reference"
    coverage: FrozenSet[str] = frozenset()
    reference_feasible: bool = True
    note: str = ""
    cmp_s2: int = 0
    ori_topk: int = 0
    cmp_topk: int = 0
    cmp_ratio: int = 1
    seed: int = 20260831
    mode_generation_feasible: bool = False

    @property
    def ori_s2(self) -> int:
        """Alias used by the unified five-mode generator."""
        return self.s2


_QSMLA_STAGE0_CASES: Tuple[QsmlaCase, ...] = (
    QsmlaCase(
        name="baseline_swa",
        b=1, s1=64, s2=128, n1=1, n2=1, d=512, k=128,
        source="supernpubench:existing",
        coverage=frozenset({"baseline", "k128", "s1_ne_s2"}),
    ),
    QsmlaCase(
        name="transformer_decode_first",
        b=1, s1=1, s2=8192, n1=64, n2=1, d=512, k=512,
        win_left=127, win_right=0, softmax_scale=0.04419417,
        source="transformer:decode_first",
        mode="CSA", q_layout="TND", kv_layout="PA_BBND",
        logical_dtype="hifp8", source_storage_dtype="uint8",
        enable_stage="future_semantics",
        coverage=frozenset({"q_tail", "batch_gqa", "k512", "s1_ne_s2"}),
        reference_feasible=False,
        note="Exact production shape; retained as a Stage 1 compile target.",
    ),
    QsmlaCase(
        name="transformer_prefill_first",
        b=1, s1=8192, s2=8192, n1=64, n2=1, d=512, k=512,
        win_left=127, win_right=0, softmax_scale=0.04419417,
        source="transformer:prefill_first",
        mode="CSA", q_layout="TND", kv_layout="PA_BBND",
        logical_dtype="hifp8", source_storage_dtype="uint8",
        enable_stage="future_semantics",
        coverage=frozenset({"batch_gqa", "k512"}),
        reference_feasible=False,
        note="Exact production prefill shape; Stage 1 compile target only.",
    ),
    QsmlaCase(
        name="transformer_csa_small_prefill",
        b=1, s1=16, s2=1024, n1=64, n2=1, d=512, k=512,
        win_left=127, win_right=0, softmax_scale=0.04419417,
        source="transformer:csa_small_prefill",
        mode="CSA", q_layout="TND", kv_layout="PA_BBND",
        logical_dtype="hifp8", source_storage_dtype="uint8",
        enable_stage="future_semantics",
        coverage=frozenset({"q_tail", "batch_gqa", "k512", "s1_ne_s2"}),
        reference_feasible=False,
        note="Shape-only coverage in Stage 0; semantics remain continuous SWA.",
    ),
    QsmlaCase(
        name="transformer_ori_sparse_tnd_pa",
        b=4, s1=4, s2=8192, n1=128, n2=1, d=512, k=0, k1=64,
        source="transformer:ori_sparse_tnd_pa",
        mode="ORI_SPARSE", q_layout="TND", kv_layout="PA_BBND",
        logical_dtype="hifp8", source_storage_dtype="uint8",
        enable_stage="future_semantics",
        coverage=frozenset({"batch_gqa", "q_tail", "s1_ne_s2"}),
        reference_feasible=False,
        note="K1 is the ORI sparse candidate width; requires sparse, TND, PA and HIFLOAT8 semantics.",
    ),
    QsmlaCase(
        name="transformer_ori_sparse_bsnd_pa",
        b=4, s1=4, s2=8192, n1=128, n2=1, d=512, k=0, k1=128,
        source="transformer:ori_sparse_bsnd_pa",
        mode="ORI_SPARSE", q_layout="BSND", kv_layout="PA_BBND",
        logical_dtype="hifp8", source_storage_dtype="uint8",
        enable_stage="future_semantics",
        coverage=frozenset({"batch_gqa", "q_tail", "k128", "s1_ne_s2"}),
        reference_feasible=False,
        note="BSND query plus ORI sparse PA path; K1=128.",
    ),
    QsmlaCase(
        name="transformer_ori_cmp_sparse_tnd_pa",
        b=4, s1=4, s2=8192, n1=128, n2=1, d=512, k=512, k1=128,
        source="transformer:ori_cmp_sparse_tnd_pa",
        mode="ORI_CMP_SPARSE", q_layout="TND", kv_layout="PA_BBND",
        logical_dtype="hifp8", source_storage_dtype="uint8",
        enable_stage="future_semantics",
        coverage=frozenset({"batch_gqa", "q_tail", "k128", "k512", "s1_ne_s2"}),
        reference_feasible=False,
        note="Dual sparse ORI/CMP path with TND query and PA KV.",
    ),
    QsmlaCase(
        name="transformer_ori_cmp_sparse_bsnd_pa",
        b=4, s1=4, s2=8192, n1=128, n2=1, d=512, k=512, k1=128,
        source="transformer:ori_cmp_sparse_bsnd_pa",
        mode="ORI_CMP_SPARSE", q_layout="BSND", kv_layout="PA_BBND",
        logical_dtype="hifp8", source_storage_dtype="uint8",
        enable_stage="future_semantics",
        coverage=frozenset({"batch_gqa", "q_tail", "k128", "k512", "s1_ne_s2"}),
        reference_feasible=False,
        note="Same numeric shape as TND variant but a distinct BSND semantic case.",
    ),
    QsmlaCase(
        name="typical_bsnd_swa_1",
        b=8, s1=4, s2=131072, n1=128, n2=1, d=512, k=0,
        tm=64, win_left=128, win_right=576,
        source="user:typical_case_1",
        enable_stage="stage1_compile",
        coverage=frozenset({"bsnd_layout", "batch_gqa", "g128_split", "s1_ne_s2", "typical_large_s2"}),
        reference_feasible=False,
        note="User-provided production BSND SWA shape; compile-only because S2 is 131072.",
    ),
    QsmlaCase(
        name="typical_bsnd_swa_2",
        b=8, s1=4, s2=131072, n1=128, n2=1, d=512, k=0,
        tm=64, win_left=128, win_right=576,
        source="user:typical_case_2",
        enable_stage="stage1_compile",
        coverage=frozenset({"bsnd_layout", "batch_gqa", "g128_split", "s1_ne_s2", "typical_large_s2"}),
        reference_feasible=False,
        note="User-provided production BSND SWA shape; compile-only because S2 is 131072.",
    ),
    QsmlaCase(
        name="typical_bsnd_swa_3",
        b=8, s1=4, s2=131072, n1=128, n2=1, d=512, k=0,
        tm=64, win_left=128, win_right=576,
        source="user:typical_case_3",
        enable_stage="stage1_compile",
        coverage=frozenset({"bsnd_layout", "batch_gqa", "g128_split", "s1_ne_s2", "typical_large_s2"}),
        reference_feasible=False,
        note="User-provided production BSND SWA shape; compile-only because S2 is 131072.",
    ),
    QsmlaCase(
        name="typical_bsnd_swa_s2_128",
        b=8, s1=4, s2=128, n1=128, n2=1, d=512, k=0,
        tm=64, win_left=128, win_right=576,
        source="user:typical_reduced_s2",
        enable_stage="stage1_compile",
        coverage=frozenset({"bsnd_layout", "batch_gqa", "g128_split", "s1_ne_s2", "typical_reduced_s2"}),
        reference_feasible=False,
        note="Production-like BSND SWA shape with S2 reduced to 128.",
    ),
    QsmlaCase(
        name="typical_bsnd_swa_s2_1024",
        b=8, s1=4, s2=1024, n1=128, n2=1, d=512, k=0,
        tm=64, win_left=128, win_right=576,
        source="user:typical_reduced_s2",
        enable_stage="stage1_compile",
        coverage=frozenset({"bsnd_layout", "batch_gqa", "g128_split", "s1_ne_s2", "typical_reduced_s2"}),
        reference_feasible=False,
        note="Production-like BSND SWA shape with S2 reduced to 1024.",
    ),
    QsmlaCase(
        name="typical_bsnd_swa_s2_4096",
        b=8, s1=4, s2=4096, n1=128, n2=1, d=512, k=0,
        tm=64, win_left=128, win_right=576,
        source="user:typical_reduced_s2",
        enable_stage="stage1_compile",
        coverage=frozenset({"bsnd_layout", "batch_gqa", "g128_split", "s1_ne_s2", "typical_reduced_s2"}),
        reference_feasible=False,
        note="Production-like BSND SWA shape with S2 reduced to 4096.",
    ),
    QsmlaCase(
        name="smoke_batch_gqa",
        b=2, s1=3, s2=5, n1=4, n2=2, d=8, k=128,
        tm=2, tk=3, td=4,
        coverage=frozenset({"batch_gqa", "q_tail", "kv_tail", "k128", "s1_ne_s2"}),
    ),
    QsmlaCase(
        name="smoke_all_tails_k512",
        b=1, s1=5, s2=7, n1=2, n2=1, d=10, k=512,
        tm=4, tk=4, td=6,
        coverage=frozenset({"batch_gqa", "q_tail", "kv_tail", "d_tail", "k512", "s1_ne_s2"}),
    ),
    QsmlaCase(
        name="smoke_bsnd_g64",
        b=1, s1=2, s2=3, n1=64, n2=1, d=8, k=128,
        tm=64, tk=2, td=4,
        q_layout="BSND", kv_layout="BSND",
        coverage=frozenset({"bsnd_layout", "g64", "kv_tail", "k128", "s1_ne_s2"}),
        note="Stage 1 direct [G,D] MM1 smoke with G=64.",
    ),
    QsmlaCase(
        name="smoke_bsnd_g128",
        b=1, s1=1, s2=3, n1=128, n2=1, d=8, k=512,
        tm=64, tk=2, td=4,
        q_layout="BSND", kv_layout="BSND",
        coverage=frozenset({"bsnd_layout", "g128_split", "q_tail", "kv_tail", "k512", "s1_ne_s2"}),
        note="Stage 1 split-G smoke: G=128 must be processed as two M=64 slices.",
    ),
)


_QSMLA_MODE_CASES: Tuple[QsmlaCase, ...] = (
    QsmlaCase(name="swa_small", mode="SWA", b=1, s1=1, s2=128,
              n1=64, n2=1, d=512, k=0, cmp_s2=64,
              tm=64, tk=32, td=64,
              win_left=127, win_right=0, source="supernpubench:five-mode",
              enable_stage="gfrun", reference_feasible=False,
              mode_generation_feasible=True),
    QsmlaCase(name="hca_small", mode="HCA", b=1, s1=1, s2=128,
              n1=64, n2=1, d=512, k=0, cmp_s2=64, cmp_ratio=4,
              tm=64, tk=32, td=64, win_left=127, win_right=0,
              source="supernpubench:five-mode", enable_stage="gfrun",
              reference_feasible=False, mode_generation_feasible=True),
    QsmlaCase(name="csa_small", mode="CSA", b=1, s1=1, s2=128,
              n1=64, n2=1, d=512, k=40, cmp_s2=64, cmp_topk=40,
              cmp_ratio=4, tm=64, tk=32, td=64, win_left=127,
              win_right=0, source="supernpubench:five-mode",
              enable_stage="gfrun", reference_feasible=False,
              mode_generation_feasible=True),
    QsmlaCase(name="ori_sparse_small", mode="ORI_SPARSE", b=1, s1=1,
              s2=128, n1=64, n2=1, d=512, k=0, k1=40, cmp_s2=64,
              ori_topk=40, cmp_ratio=4, tm=64, tk=32, td=64, win_left=127,
              win_right=0, source="supernpubench:five-mode",
              enable_stage="gfrun", reference_feasible=False,
              mode_generation_feasible=True),
    QsmlaCase(name="ori_cmp_sparse_small", mode="ORI_CMP_SPARSE",
              b=1, s1=1, s2=128, n1=64, n2=1, d=512, k=40, k1=40,
              cmp_s2=64, ori_topk=40, cmp_topk=40, cmp_ratio=4,
              tm=64, tk=32, td=64, win_left=127, win_right=0,
              source="supernpubench:five-mode", enable_stage="gfrun",
              reference_feasible=False, mode_generation_feasible=True),
    QsmlaCase(name="typical_bsnd_csa_s2_128_topk32", mode="CSA",
              b=8, s1=4, s2=128, n1=128, n2=1, d=512, k=32,
              cmp_s2=32, cmp_topk=32, cmp_ratio=4, tm=64, tk=32, td=64,
              win_left=128, win_right=576, source="user:typical_csa",
              enable_stage="gfrun", reference_feasible=False,
              mode_generation_feasible=True,
              coverage=frozenset({"typical_csa", "bsnd_layout", "batch_gqa",
                                  "g128_split", "s1_ne_s2", "typical_reduced_s2"})),
    QsmlaCase(name="typical_bsnd_csa_s2_1024_topk128", mode="CSA",
              b=8, s1=4, s2=1024, n1=128, n2=1, d=512, k=128,
              cmp_s2=256, cmp_topk=128, cmp_ratio=4, tm=64, tk=32, td=64,
              win_left=128, win_right=576, source="user:typical_csa",
              enable_stage="gfrun", reference_feasible=False,
              mode_generation_feasible=True,
              coverage=frozenset({"typical_csa", "bsnd_layout", "batch_gqa",
                                  "g128_split", "s1_ne_s2", "typical_reduced_s2"})),
    QsmlaCase(name="typical_bsnd_csa_s2_4096_topk512", mode="CSA",
              b=8, s1=4, s2=4096, n1=128, n2=1, d=512, k=512,
              cmp_s2=1024, cmp_topk=512, cmp_ratio=4, tm=64, tk=32, td=64,
              win_left=128, win_right=576, source="user:typical_csa",
              enable_stage="compile_only", reference_feasible=False,
              coverage=frozenset({"typical_csa", "bsnd_layout", "batch_gqa",
                                  "g128_split", "s1_ne_s2", "typical_reduced_s2"}),
              note="Compile-only: deterministic input generation is intentionally disabled."),
    QsmlaCase(name="typical_bsnd_csa_s2_131072_topk512", mode="CSA",
              b=8, s1=4, s2=131072, n1=128, n2=1, d=512, k=512,
              cmp_s2=32768, cmp_topk=512, cmp_ratio=4, tm=64, tk=32, td=64,
              win_left=128, win_right=576, source="user:typical_csa",
              enable_stage="compile_only", reference_feasible=False,
              coverage=frozenset({"typical_csa", "bsnd_layout", "batch_gqa",
                                  "g128_split", "s1_ne_s2", "typical_large_s2"}),
              note="Compile-only: raw ORI KV input alone is 1 GiB in fp16."),
)

QSMLA_STAGE0_CASES = _QSMLA_STAGE0_CASES
QSMLA_MODE_CASES = _QSMLA_MODE_CASES
QSMLA_CASES = QSMLA_STAGE0_CASES + QSMLA_MODE_CASES
QSMLA_EXECUTABLE_CASES = tuple(
    case for case in QSMLA_MODE_CASES if case.mode_generation_feasible
)


def validate_case(case: QsmlaCase) -> Optional[str]:
    positive = (case.b, case.n1, case.n2, case.d, case.tm, case.tk, case.td)
    if any(value <= 0 for value in positive):
        return "B/N1/N2/D/Tm/Tk/Td must be positive"
    if case.s1 < 0 or case.s2 < 0 or case.k < 0:
        return "S1/S2/K must be non-negative"
    if case.k1 is not None and case.k1 < 0:
        return "K1 must be non-negative when present"
    if case.n1 % case.n2 != 0:
        return "N1 must be divisible by N2"
    if case.win_left < -1 or case.win_right < -1:
        return "window bounds must be -1 or non-negative"
    return None


def case_output_dir(case: QsmlaCase, root: Path) -> Path:
    return Path(root) / case.name
def _fp16(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]


def _tensor4(b, s, n, d, rng):
    return [[[[ _fp16(rng.uniform(-0.125, 0.125)) for _ in range(d)]
              for _ in range(n)] for _ in range(s)] for _ in range(b)]


def _sparse_tensor(b, s1, n2, topk, source_s2):
    base = list(range(topk))
    if topk >= 8:
        base[1] = base[0]       # duplicate is retained
        base[3] = -2            # invalid negative is skipped
        base[5] = source_s2     # overflow is skipped
        base[-1] = -1           # explicit terminator
    return [[[list(base) for _ in range(n2)] for _ in range(s1)] for _ in range(b)]


def _length_tensor(b, s1, n2, length):
    return [[[length for _ in range(n2)] for _ in range(s1)] for _ in range(b)]


def _flatten(values):
    if isinstance(values, (list, tuple)):
        for value in values:
            yield from _flatten(value)
    else:
        yield values


def _write(path, format_code, values):
    flattened = list(_flatten(values))
    path.write_bytes(struct.pack(f"<{len(flattened)}{format_code}", *flattened))


def generate_case(case, output_root):
    output = Path(output_root) / case.name
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(case.seed)

    q = _tensor4(case.b, case.s1, case.n1, case.d, rng)
    ori_kv = _tensor4(case.b, case.ori_s2, case.n2, case.d, rng)
    cmp_kv = None if case.mode in {"SWA", "ORI_SPARSE"} else _tensor4(
        case.b, case.cmp_s2, case.n2, case.d, rng)
    ori_indices = None if case.mode in {"SWA", "HCA", "CSA"} else _sparse_tensor(
        case.b, case.s1, case.n2, case.ori_topk, case.ori_s2)
    cmp_indices = None if case.mode not in {"CSA", "ORI_CMP_SPARSE"} else _sparse_tensor(
        case.b, case.s1, case.n2, case.cmp_topk, case.cmp_s2)
    ori_lengths = None if case.mode in {"SWA", "HCA", "CSA"} else _length_tensor(
        case.b, case.s1, case.n2, case.ori_topk - 5)
    cmp_lengths = None if case.mode != "ORI_CMP_SPARSE" else _length_tensor(
        case.b, case.s1, case.n2, case.cmp_topk - 5)

    golden = qsmla_reference(
        q=q, ori_kv=ori_kv, cmp_kv=cmp_kv,
        ori_sparse_indices=ori_indices, cmp_sparse_indices=cmp_indices,
        ori_topk_length=ori_lengths, cmp_topk_length=cmp_lengths,
        mode=case.mode, softmax_scale=case.softmax_scale,
        cmp_ratio=case.cmp_ratio,
        win_left=case.win_left, win_right=case.win_right,
    )

    _write(output / "q.fp16.bin", "e", q)
    _write(output / "ori_kv.fp16.bin", "e", ori_kv)
    if cmp_kv is not None:
        _write(output / "cmp_kv.fp16.bin", "e", cmp_kv)
    if ori_indices is not None:
        _write(output / "ori_sparse_indices.int32.bin", "i", ori_indices)
        _write(output / "ori_topk_length.int32.bin", "i", ori_lengths)
    if cmp_indices is not None:
        _write(output / "cmp_sparse_indices.int32.bin", "i", cmp_indices)
    if cmp_lengths is not None:
        _write(output / "cmp_topk_length.int32.bin", "i", cmp_lengths)
    _write(output / "golden.fp32.bin", "f", golden)

    manifest = asdict(case)
    manifest["ori_s2"] = case.ori_s2
    manifest["coverage"] = sorted(case.coverage)
    manifest.update({
        "q_layout": "BSND", "kv_layout": "BSND",
        "input_dtype": "fp16", "index_dtype": "int32",
        "golden_dtype": "fp32",
    })
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description="Generate deterministic QSMLA inputs for executable cases")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--all", action="store_true")
    action.add_argument("--case", choices=[case.name for case in QSMLA_CASES])
    parser.add_argument("--output-root", type=Path, default=Path("qsmla_output"))
    args = parser.parse_args()

    if args.all:
        selected = QSMLA_EXECUTABLE_CASES
    else:
        selected = tuple(case for case in QSMLA_CASES if case.name == args.case)
        if selected and not selected[0].mode_generation_feasible:
            parser.error(
                f"case {selected[0].name} is compile-only and has no generated input"
            )
    for case in selected:
        output = generate_case(case, args.output_root)
        print(output)


if __name__ == "__main__":
    main()
