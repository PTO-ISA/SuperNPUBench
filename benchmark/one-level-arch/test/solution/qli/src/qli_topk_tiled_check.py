#!/usr/bin/env python3
"""qli_topk_tiled — QLI 分数阶段 + topk_tiled 引擎端到端精度检查。

流程：生成 fp8 输入（Q / 预转置 K^T / W / scales）与 golden 到
compare/<elf-stem>/，gfrun 4-PE 跑 hosted res_check ELF，然后比对：
  1. errors.bin 全零（topk_tiled 内部守卫：候选溢出 / 空区间）
  2. scores_readback.bin vs golden 分数（cosine / max_abs —— matmul 链路）
  3. TopK 输出（output.bin = 每行 topk 个无序索引）：
     fp32 模式: 选中值有序多重集 == golden topk 值多重集（精确 FP32 topk，
                并列容忍），索引互异且在界内；同时报告索引集合命中率
     fp16 模式: 输出索引的 FP16 sortable-key 多重集 == golden top-K keys
                （并列为多重集语义），索引互异且在界内

用法：
  python3 qli_topk_tiled_check.py --gfrun <gfrun> \
      [--sq 1 --skv 131072 --topk 1024 --mode fp32|fp16 --seed N --timeout S]

ELF 名由 (Sq, Skv, mode) 推导，须与 Makefile 产物一致：
  solution_qli/qli_topk_tiled_<mode>_B1_Sq<Sq>_Skv<Skv>.elf
"""
from __future__ import annotations

import argparse
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    from ml_dtypes import float8_e4m3fn
except ImportError:
    sys.exit("ERROR: ml_dtypes not installed. Run: pip install ml_dtypes")

SCRIPT_DIR = Path(__file__).resolve().parent
ONE_LEVEL = SCRIPT_DIR.parents[3]          # .../one-level-arch
ELF_DIR = ONE_LEVEL / "output" / "solution" / "qli" / "elf" / "solution_qli"
COMPARE_ROOT = ONE_LEVEL / "compare"

D = 128
G = 64


def fp16_key(x):
    """RNE-round 到 FP16 并返回 16-bit sortable key（与 load_bin 一致）。"""
    bits = struct.unpack("<H", struct.pack("<e", x))[0]
    return (bits ^ 0xFFFF) if (bits >> 15) else (bits ^ 0x8000)


def fp16_key_vec(arr):
    bits = np.asarray(arr, dtype=np.float32).astype(np.float16).view(np.uint16)
    return np.where(bits >> 15, bits ^ 0xFFFF, bits ^ 0x8000).astype(np.uint32)


def gen_case(case_dir, sq, skv, topk, seed):
    rng = np.random.default_rng(seed)
    q_f32 = rng.standard_normal((sq * G, D), dtype=np.float32)
    k_f32 = rng.standard_normal((skv, D), dtype=np.float32)
    q8 = q_f32.astype(float8_e4m3fn)
    k8 = k_f32.astype(float8_e4m3fn)
    w = rng.standard_normal(sq * G, dtype=np.float32)
    scale_q = (rng.standard_normal(sq * G, dtype=np.float32) * 0.01).astype(np.float32)
    scale_k = (rng.standard_normal(skv, dtype=np.float32) * 0.01).astype(np.float32)

    # kernel 读的是量化后的 fp8 字节；golden 同样用量化后的值计算
    q8f = q8.astype(np.float32).reshape(sq, G, D)
    k8f = k8.astype(np.float32)                        # [Skv, D]
    qk = np.einsum("sgd,nd->sgn", q8f, k8f)            # [Sq, g, Skv]
    qk = np.maximum(qk, 0.0)                           # ReLU
    wq = (w * scale_q).reshape(sq, G)
    scores = np.einsum("sgn,sg->sn", qk, wq) * scale_k  # [Sq, Skv] FP32
    scores = scores.astype(np.float32)

    case_dir.mkdir(parents=True, exist_ok=True)
    q8.tofile(case_dir / "srcq.bin")
    # 预转置 K^T [D, Skv]：与 [Skv, D] 同值不同布局，免设备侧转置
    np.ascontiguousarray(k8.T).tofile(case_dir / "srckt.bin")
    w.tofile(case_dir / "srcw.bin")
    scale_q.tofile(case_dir / "srcsq.bin")
    scale_k.tofile(case_dir / "srcsk.bin")
    np.array([sq, skv, topk], dtype=np.int64).tofile(case_dir / "shape.bin")

    # golden topk：按 (-value, index) 排序取前 topk（索引 tie-break，仅作参考集合）
    ref_idx = np.zeros((sq, topk), dtype=np.int32)
    ref_val = np.zeros((sq, topk), dtype=np.float32)
    for r in range(sq):
        order = np.lexsort((np.arange(skv), -scores[r]))
        ref_idx[r] = order[:topk].astype(np.int32)
        ref_val[r] = scores[r, order[:topk]]
    return scores, ref_idx, ref_val


def run_check(args):
    tag = f"qli_topk_tiled_{args.mode}_B1_Sq{args.sq}_Skv{args.skv}"
    elf = ELF_DIR / f"{tag}.elf"
    if not elf.exists():
        return "SKIP", f"missing {elf}"
    case_dir = COMPARE_ROOT / tag
    for stale in ("output.bin", "errors.bin", "scores_readback.bin"):
        (case_dir / stale).unlink(missing_ok=True)

    scores, ref_idx, ref_val = gen_case(case_dir, args.sq, args.skv, args.topk, args.seed)
    if not np.any(scores):
        return "FAIL", "golden scores all zero (vacuous)"

    cmd = [str(args.gfrun), "-s", "softcore.multiThreadNum=4", "-f", str(elf)]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, timeout=args.timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        (case_dir / "gfrun.log").write_text(exc.stdout or "", encoding="utf-8")
        return "TIMEOUT", f"after {args.timeout}s: {case_dir / 'gfrun.log'}"
    log = proc.stdout or ""
    (case_dir / "gfrun.log").write_text(log, encoding="utf-8")
    if proc.returncode != 0 or "Reach the End" not in log:
        return "FAIL", f"gfrun rc={proc.returncode}, log: {case_dir / 'gfrun.log'}"

    errors = np.fromfile(case_dir / "errors.bin", dtype=np.int32)
    if errors.size != args.sq or np.any(errors != 0):
        return "FAIL", f"errors={errors.tolist()} (topk_tiled guard tripped)"

    # 分数链路
    sim_scores = np.fromfile(case_dir / "scores_readback.bin",
                             dtype=np.float32).reshape(args.sq, args.skv)
    cos = float(np.dot(sim_scores.ravel(), scores.ravel()) /
                (np.linalg.norm(sim_scores.ravel()) * np.linalg.norm(scores.ravel())))
    max_abs = float(np.abs(sim_scores - scores).max())

    # TopK 输出
    out = np.fromfile(case_dir / "output.bin", dtype=np.int32).reshape(args.sq, args.topk)
    in_range = bool(np.all((out >= 0) & (out < args.skv)))
    distinct = all(len(set(row.tolist())) == args.topk for row in out)
    setm = sum(1 for r in range(args.sq) if set(out[r].tolist()) == set(ref_idx[r].tolist()))

    if not in_range or not distinct:
        return "FAIL", (f"index sanity: in_range={in_range} distinct={distinct}; "
                        f"log: {case_dir / 'gfrun.log'}")

    if args.mode == "fp32":
        # 选中值多重集 vs golden topk 值多重集（tie 容忍的精确性判据）
        val_ok = True
        worst = 0.0
        for r in range(args.sq):
            sv = np.sort(sim_scores[r, out[r]])[::-1]
            rv = np.sort(ref_val[r])[::-1]
            d = float(np.abs(sv - rv).max()) if sv.shape == rv.shape else float("inf")
            worst = max(worst, d)
            if sv.shape != rv.shape or not np.allclose(sv, rv, atol=1e-5, rtol=0):
                val_ok = False
        score_ok = cos > 0.999999 and max_abs < 1e-4
        ok = val_ok and score_ok
        return ("PASS" if ok else "FAIL"), (
            f"cosine={cos:.6f} max_abs={max_abs:.3e} val-multiset "
            f"{'OK' if val_ok else 'MISMATCH'} (worst {worst:.3e}) set={setm}/{args.sq}")

    # fp16 模式：FP16 sortable-key 多重集
    key_ok = True
    for r in range(args.sq):
        sim_keys = np.sort(fp16_key_vec(sim_scores[r, out[r]]))[::-1]
        ref_keys = np.sort(fp16_key_vec(ref_val[r]))[::-1]
        if sim_keys.shape != ref_keys.shape or not np.array_equal(sim_keys, ref_keys):
            key_ok = False
            break
    score_ok = cos > 0.999999 and max_abs < 1e-4
    ok = key_ok and score_ok
    return ("PASS" if ok else "FAIL"), (
        f"cosine={cos:.6f} max_abs={max_abs:.3e} fp16-key-multiset "
        f"{'OK' if key_ok else 'MISMATCH'} set={setm}/{args.sq}")


def main():
    ap = argparse.ArgumentParser(description="qli_topk_tiled end-to-end check")
    ap.add_argument("--gfrun", required=True, type=Path)
    ap.add_argument("--sq", type=int, default=1)
    ap.add_argument("--skv", type=int, default=131072)
    ap.add_argument("--topk", type=int, default=1024)
    ap.add_argument("--mode", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()
    status, detail = run_check(args)
    print(f"[qli_topk_tiled {args.mode} Sq={args.sq} Skv={args.skv} "
          f"topk={args.topk} seed={args.seed}] {status}: {detail}")
    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
