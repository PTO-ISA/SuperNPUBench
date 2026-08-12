#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, Optional, Tuple


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
    q_layout: str = "CONTIGUOUS_2D"
    kv_layout: str = "CONTIGUOUS_2D"
    logical_dtype: str = "fp16"
    source_storage_dtype: str = "fp16"
    stage0_compute_dtype: str = "fp16"
    enable_stage: str = "stage0_reference"
    coverage: FrozenSet[str] = frozenset()
    reference_feasible: bool = True
    note: str = ""


QSMLA_STAGE0_CASES: Tuple[QsmlaCase, ...] = (
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
