#!/usr/bin/env python3

import argparse
import json
import random
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import FrozenSet, Optional, Tuple

from qsmla_reference import decode_hif8, encode_hif8, qsmla_reference


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
    logical_dtype: str = "hifp8"
    source_storage_dtype: str = "uint8"
    stage0_compute_dtype: str = "fp32"
    enable_stage: str = "stage0_reference"
    coverage: FrozenSet[str] = frozenset()
    reference_feasible: bool = True
    # 是否纳入 --all-reference 批量生成（超大用例可由 --case 单独生成）。
    bulk_reference: bool = True
    note: str = ""
    cmp_s2: int = 0
    ori_topk: int = 0
    cmp_topk: int = 0
    cmp_ratio: int = 1
    seed: int = 20260831
    mode_generation_feasible: bool = False
    q_descale: float = 0.5
    ori_kv_descale: float = 0.25
    cmp_kv_descale: float = 0.125

    @property
    def ori_s2(self) -> int:
        """Alias used by the unified five-mode generator."""
        return self.s2


_QSMLA_STAGE0_CASES: Tuple[QsmlaCase, ...] = (
)


_QSMLA_MODE_CASES: Tuple[QsmlaCase, ...] = (
    QsmlaCase(name="swa_small", mode="SWA", b=1, s1=1, s2=128,
              n1=64, n2=1, d=512, k=0, cmp_s2=64,
              tm=64, tk=32, td=64, win_left=127, win_right=0,
              softmax_scale=0.04419417,
              source="supernpubench:five-mode", enable_stage="gfrun",
              reference_feasible=True, mode_generation_feasible=True,
              coverage=frozenset({"hif8", "bf16_output", "descale"})),
    QsmlaCase(name="hca_small", mode="HCA", b=1, s1=1, s2=128,
              n1=64, n2=1, d=512, k=0, cmp_s2=64, cmp_ratio=4,
              tm=64, tk=32, td=64, win_left=127, win_right=0,
              softmax_scale=0.04419417,
              source="supernpubench:five-mode", enable_stage="gfrun",
              reference_feasible=True, mode_generation_feasible=True,
              coverage=frozenset({"hif8", "bf16_output", "descale"})),
    QsmlaCase(name="csa_small", mode="CSA", b=1, s1=1, s2=128,
              n1=64, n2=1, d=512, k=40, cmp_s2=64, cmp_topk=40,
              cmp_ratio=4, tm=64, tk=32, td=64, win_left=127,
              win_right=0, softmax_scale=0.04419417,
              source="supernpubench:five-mode", enable_stage="gfrun",
              reference_feasible=True, mode_generation_feasible=True,
              coverage=frozenset({"hif8", "bf16_output", "descale"})),
    # 小号多 work 用例：B=2,S1=2 -> kWorkCount=4；D=128 让 tile 更小，
    # 用于在 v3.x 上验证"多 work"路径（生产形状的缩小版）。
    QsmlaCase(name="csa_mw_small", mode="CSA", b=2, s1=2, s2=256,
              n1=64, n2=1, d=128, k=64, cmp_s2=128, cmp_topk=64,
              cmp_ratio=4, tm=64, tk=32, td=64, win_left=255,
              win_right=0, softmax_scale=0.04419417,
              source="supernpubench:five-mode", enable_stage="gfrun",
              reference_feasible=True, mode_generation_feasible=True,
              coverage=frozenset({"hif8", "bf16_output", "descale",
                                  "multi_work", "small_d"})),
    # 大用例：10 个 KV 块（8 个 ORI 满块 + 2 个 CMP 块），用于验证
    # KV 预取深度（preload=1 vs 3）在多块场景下的收益。
    QsmlaCase(name="csa_medium", mode="CSA", b=1, s1=1, s2=1024,
              n1=64, n2=1, d=512, k=40, cmp_s2=512, cmp_topk=256,
              cmp_ratio=4, tm=64, tk=32, td=64, win_left=1023,
              win_right=0, softmax_scale=0.04419417,
              source="supernpubench:five-mode", enable_stage="gfrun",
              reference_feasible=True, mode_generation_feasible=True,
              coverage=frozenset({"hif8", "bf16_output", "descale",
                                  "csa_medium"})),
    QsmlaCase(name="ori_sparse_small", mode="ORI_SPARSE", b=1, s1=1,
              s2=128, n1=64, n2=1, d=512, k=0, k1=40, cmp_s2=64,
              ori_topk=40, cmp_ratio=4, tm=64, tk=32, td=64,
              win_left=127, win_right=0, softmax_scale=0.04419417,
              source="supernpubench:five-mode", enable_stage="gfrun",
              reference_feasible=True, mode_generation_feasible=True,
              coverage=frozenset({"hif8", "bf16_output", "descale"})),
    QsmlaCase(name="ori_cmp_sparse_small", mode="ORI_CMP_SPARSE",
              b=1, s1=1, s2=128, n1=64, n2=1, d=512, k=40, k1=40,
              cmp_s2=64, ori_topk=40, cmp_topk=40, cmp_ratio=4,
              tm=64, tk=32, td=64, win_left=127, win_right=0,
              softmax_scale=0.04419417,
              source="supernpubench:five-mode", enable_stage="gfrun",
              reference_feasible=True, mode_generation_feasible=True,
              coverage=frozenset({"hif8", "bf16_output", "descale"})),
    QsmlaCase(name="typical_bsnd_csa_s2_128_topk32", mode="CSA",
              b=8, s1=4, s2=128, n1=128, n2=1, d=512, k=32,
              cmp_s2=32, cmp_topk=32, cmp_ratio=4, tm=64, tk=32, td=64,
              win_left=128, win_right=576, source="user:typical_csa",
              enable_stage="gfrun", reference_feasible=False,
              mode_generation_feasible=True,
              coverage=frozenset({"typical_csa", "bsnd_layout", "batch_gqa",
                                  "g128_split", "s1_ne_s2", "typical_reduced_s2"})),
    QsmlaCase(name="typical_bsnd_csa_s2_2048_topk1024_c4", mode="CSA",
              b=8, s1=4, s2=2048, n1=128, n2=1, d=512, k=1024,
              cmp_s2=4096, cmp_topk=1024, cmp_ratio=4, tm=64, tk=32,
              td=64, win_left=128, win_right=576,
              softmax_scale=0.04419417,
              source="user:typical_csa_c4",
              enable_stage="gfrun", reference_feasible=True,
              bulk_reference=False,
              mode_generation_feasible=True,
              coverage=frozenset({"typical_csa", "bsnd_layout", "batch_gqa",
                                  "g128_split", "s1_ne_s2", "typical_reduced_s2"}),
              note="Network CSA case, C4 dead-data trim (S2 131072->2048, "
                   "cmp_s2 32768->4096); block-wise equivalent to the "
                   "production shape, 64 works per ELF."),
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


def _hif8_tensor(b, s, n, d, rng):
    encoded = [[[[encode_hif8(rng.uniform(-2.0, 2.0)) for _ in range(d)]
                 for _ in range(n)] for _ in range(s)] for _ in range(b)]
    decoded = [[[[decode_hif8(value) for value in row] for row in heads]
                 for heads in seq] for seq in encoded]
    return encoded, decoded


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


def _transpose_kv(kv):
    """[B, S, N2, D] -> [B, D, S*N2] row-major transposed view.

    The output flattens as element (d, t) at d*(S*N2) + t within each batch,
    matching the single-PE CSA kernel's [D, S2] GM view for the BMM1 K^T
    B-tile. Works for both decoded floats and raw HIF8 payloads.
    """
    return [
        [[kv[b][t][n][d]
          for t in range(len(kv[b]))
          for n in range(len(kv[b][t]))]
         for d in range(len(kv[b][0][0]))]
        for b in range(len(kv))
    ]


def _flatten(values):
    if isinstance(values, (list, tuple)):
        for value in values:
            yield from _flatten(value)
    else:
        yield values


def _write(path, format_code, values):
    flattened = list(_flatten(values))
    path.write_bytes(struct.pack(f"<{len(flattened)}{format_code}", *flattened))


def generate_case(case, output_root, dtype="HIF8"):
    output = Path(output_root) / case.name
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(case.seed)

    dtype = dtype.upper()
    if dtype not in {"HIF8", "FP16"}:
        raise ValueError(f"unsupported QSMLA_DTYPE: {dtype}")
    is_hif8 = dtype == "HIF8"
    q_descale = case.q_descale if is_hif8 else 1.0
    ori_kv_descale = case.ori_kv_descale if is_hif8 else 1.0
    cmp_kv_descale = case.cmp_kv_descale if is_hif8 else 1.0
    if is_hif8:
        q_raw, q = _hif8_tensor(case.b, case.s1, case.n1, case.d, rng)
        ori_raw, ori_kv = _hif8_tensor(case.b, case.ori_s2, case.n2, case.d, rng)
        if case.mode in {"SWA", "ORI_SPARSE"}:
            cmp_raw = cmp_kv = None
        else:
            cmp_raw, cmp_kv = _hif8_tensor(
                case.b, case.cmp_s2, case.n2, case.d, rng)
    else:
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
        q_descale=q_descale,
        ori_kv_descale=ori_kv_descale,
        cmp_kv_descale=cmp_kv_descale,
        quantize_probability=is_hif8,
    )

    if is_hif8:
        _write(output / "q.hif8.bin", "B", q_raw)
        _write(output / "ori_kv.hif8.bin", "B", ori_raw)
        # qli-style pre-transposed ORI KV view [B, D, OriS2] (row-major per
        # batch: element (d, t) at d*OriS2 + t) for the single-PE CSA BMM1
        # B-tile contract (CUBE_N8 ND source). Raw payload is byte-rearranged
        # without decoding.
        _write(output / "ori_kv_t.hif8.bin", "B", _transpose_kv(ori_raw))
        if cmp_raw is not None:
            _write(output / "cmp_kv.hif8.bin", "B", cmp_raw)
    else:
        _write(output / "q.fp16.bin", "e", q)
        _write(output / "ori_kv.fp16.bin", "e", ori_kv)
        _write(output / "ori_kv_t.fp16.bin", "e", _transpose_kv(ori_kv))
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
        "qsmla_dtype": dtype,
        "q_descale": q_descale,
        "ori_kv_descale": ori_kv_descale,
        "cmp_kv_descale": cmp_kv_descale,
        "q_layout": "BSND", "kv_layout": "BSND",
        "input_dtype": "hif8" if is_hif8 else "fp16",
        "output_dtype": "bf16" if is_hif8 else "fp16",
        "index_dtype": "int32", "golden_dtype": "fp32",
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
    parser.add_argument(
        "--dtype", choices=("HIF8", "FP16"), default="HIF8",
        help="generate primary HIF8 data by default; FP16 is opt-in",
    )
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
        output = generate_case(case, args.output_root, dtype=args.dtype)
        print(output)


if __name__ == "__main__":
    main()
