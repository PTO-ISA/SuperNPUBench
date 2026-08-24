#!/usr/bin/env python3
"""
QLI Golden 数据生成与精度比对脚本 — FP8 Q/K + FP32 W 场景

输入:
  Q  [Sq*g, D]   FP8_E4M3FN  (量化路径, 绕过 INT8 ACCCVT NaN bug)
  K  [Skv, D]    FP8_E4M3FN
  W  [Sq*g]      FP32        (连续格式, 与 kernel TLOAD 一致)
  scale_q [Sq*g] FP32        (per-token-head Q 量化 scale)
  scale_k [Skv]  FP32        (per-token K 量化 scale)

输出:
  reference_scores.bin   [Sq, Skv]    FP32  (numpy golden 参考)
  reference_indices.bin  [Sq, topK]   INT32 (numpy argmax TopK)

用法:
  python3 gen_qli_golden.py --mode gen    --outdir <compare_dir>
  python3 gen_qli_golden.py --mode verify --sim <res.bin> --ref <reference_scores.bin> \
                                       --sim_idx <indices.bin> --ref_idx <reference_indices.bin>

前置条件:
  pip install ml_dtypes numpy
"""

import argparse
import os
import sys
import numpy as np

try:
    from ml_dtypes import float8_e4m3fn
except ImportError:
    sys.exit("ERROR: ml_dtypes not installed. Run: pip install ml_dtypes")


# ============================================================
# 1. 参数定义
# ============================================================
SQ    = 64
G     = 64
D     = 128
SKV   = 128
TOPK  = 128
SEED  = 42


# ============================================================
# 2. 输入数据生成
# ============================================================
def gen_inputs(outdir):
    os.makedirs(outdir, exist_ok=True)

    np.random.seed(SEED)

    # Q: [Sq*g, D] FP8_E4M3FN
    q_f32 = np.random.randn(SQ * G, D).astype(np.float32)
    q_fp8 = q_f32.astype(float8_e4m3fn)
    q_fp8.tofile(os.path.join(outdir, "srcq.bin"))
    print(f"  srcq.bin: {q_fp8.nbytes} bytes  Q [{SQ*G}, {D}] FP8_E4M3FN")

    # K: [Skv, D] FP8_E4M3FN
    # K continues from Q's random state (sequential, no reseed)
    k_f32 = np.random.randn(SKV, D).astype(np.float32)
    k_fp8 = k_f32.astype(float8_e4m3fn)
    k_fp8.tofile(os.path.join(outdir, "srck.bin"))
    print(f"  srck.bin: {k_fp8.nbytes} bytes  K [{SKV}, {D}] FP8_E4M3FN")

    # W: [Sq*g] FP32 continuous
    # kernel TLOAD 用 RowMajor<kTm, 1> 从 w_ptr + i*g + gi*kTm 读取连续 float
    np.random.seed(SEED)
    w_vals = np.random.randn(SQ * G).astype(np.float32)
    w_vals.tofile(os.path.join(outdir, "srcw.bin"))
    print(f"  srcw.bin: {w_vals.nbytes} bytes  W [{SQ*G}] FP32 (continuous)")

    # scale_q: [Sq*g] FP32 — per-token-head Q 量化 scale
    np.random.seed(SEED)
    scaleQ = (np.random.randn(SQ * G) * 0.01).astype(np.float32)
    scaleQ.tofile(os.path.join(outdir, "srcsq.bin"))
    print(f"  srcsq.bin: {scaleQ.nbytes} bytes  scale_q [{SQ*G}] FP32")

    # scale_k: [Skv] FP32 — per-token K 量化 scale
    np.random.seed(SEED)
    scaleK = (np.random.randn(SKV) * 0.01).astype(np.float32)
    scaleK.tofile(os.path.join(outdir, "srcsk.bin"))
    print(f"  srcsk.bin: {scaleK.nbytes} bytes  scale_k [{SKV}] FP32")

    # Return FP8-quantized Q/K as float32 for golden computation.
    # kernel TMATMUL reads FP8 Q/K and computes in FP32 accumulator,
    # so golden must use FP8-quantized values (converted to float32)
    # to match the hardware computation chain.
    q_fp8_f32 = q_fp8.astype(np.float32)
    k_fp8_f32 = k_fp8.astype(np.float32)

    return q_fp8_f32, k_fp8_f32, w_vals, scaleQ, scaleK


# ============================================================
# 3. Golden 参考计算
# ============================================================
def compute_golden(q_fp8_f32, k_fp8_f32, w_f32, scale_q, scale_k):
    """
    numpy golden — 与 kernel 计算匹配:
      score[s1, s2] = scale_k[s2] * Σ_g (W[s1,g] * scale_q[s1,g] * ReLU(QK[g,s1,s2]))

    Q/K 使用 FP8 量化后的值 (q_fp8_f32, k_fp8_f32), 与 kernel TMATMUL 输入一致。
    FP8→FP32 转换是无损的 (FP8 e4m3 的每个值都能被 FP32 精确表示),
    乘法和累加在 FP32 中进行 (与硬件 FP32 累加器一致)。

    注: golden 用 numpy 向量化一次性计算全部 Skv, 不需要像 kernel 那样
        按 kTk 分块 (分块是 tile register 大小限制, numpy 不受此约束)。
    """
    Q = q_fp8_f32.reshape(SQ, G, D)      # [Sq, g, D]
    K = k_fp8_f32                         # [Skv, D]
    wq = w_f32 * scale_q                  # [Sq*g] → W * scale_q

    ref_scores = np.zeros((SQ, SKV), dtype=np.float32)
    for i in range(SQ):
        QK = Q[i] @ K.T                   # [g, Skv] — 一次性计算, 不分块
        QK = np.maximum(QK, 0)            # ReLU
        wq_i = wq[i*G:(i+1)*G]            # [g]
        weighted = QK * wq_i[:, np.newaxis]  # [g, Skv]
        ref_scores[i] = weighted.sum(axis=0) * scale_k  # [Skv]

    return ref_scores


def compute_topk(ref_scores):
    """迭代 argmax TopK, 并列取最小索引 (与 numpy argmax 一致)"""
    ref_indices = np.zeros((SQ, TOPK), dtype=np.int32)
    for i in range(SQ):
        row = ref_scores[i].copy()
        for k in range(TOPK):
            idx = np.argmax(row)
            ref_indices[i, k] = idx
            row[idx] = -np.inf
    return ref_indices


# ============================================================
# 4. 精度比对
# ============================================================
def topk_set_match(sim_idx, ref_idx):
    """逐行比较 TopK 索引集合（无序 set 契约）。返回 (set_match_count, total_rows)"""
    sq = sim_idx.shape[0]
    ok = sum(1 for i in range(sq) if set(sim_idx[i].tolist()) == set(ref_idx[i].tolist()))
    return ok, sq


def verify(sim_res_path, ref_res_path, sim_idx_path, ref_idx_path):
    sim = np.fromfile(sim_res_path, dtype=np.float32).reshape(SQ, SKV)
    ref = np.fromfile(ref_res_path, dtype=np.float32).reshape(SQ, SKV)
    sim_idx = np.fromfile(sim_idx_path, dtype=np.int32).reshape(SQ, TOPK)
    ref_idx = np.fromfile(ref_idx_path, dtype=np.int32).reshape(SQ, TOPK)

    nan_cnt = np.isnan(sim).sum()
    total = sim.size

    print(f"\n{'='*60}")
    print(f"Score 矩阵比对 [{SQ}, {SKV}]")
    print(f"{'='*60}")

    if nan_cnt > 0:
        print(f"  NaN count:    {nan_cnt}/{total} ({nan_cnt/total*100:.1f}%)")
        print(f"  Result:       FAIL (NaN)")
        return False

    norm_sim = np.linalg.norm(sim)
    norm_ref = np.linalg.norm(ref)
    cosine = np.dot(sim.flatten(), ref.flatten()) / (norm_sim * norm_ref) if norm_sim and norm_ref else 0.0

    max_err = np.max(np.abs(sim - ref))
    mean_err = np.mean(np.abs(sim - ref))
    match_rate = np.mean(sim_idx == ref_idx)           # 参考：顺序精确匹配
    setm, sq = topk_set_match(sim_idx, ref_idx)        # 主判据：集合一致

    print(f"  NaN count:    0/{total}")
    print(f"  Cosine:       {cosine:.6f}")
    print(f"  Max abs err:  {max_err:.6f}")
    print(f"  Mean abs err: {mean_err:.6f}")
    print(f"  TopK exact:   {match_rate:.1%}   (参考; radix-set 契约下无序输出不要求)")
    print(f"  TopK set:     {setm}/{sq}   (主判据)")
    print(f"  Result:       {'PASS' if cosine > 0.99 and setm == sq else 'FAIL'}")

    return cosine > 0.99 and setm == sq


# ============================================================
# 5. 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="QLI Golden generator for FP8+FP32 scenario")
    parser.add_argument("--mode", required=True, choices=["gen", "verify"],
                        help="gen: 生成输入+golden; verify: 比对 sim vs golden")
    parser.add_argument("--outdir", default=None, help="输出目录 (gen mode)")
    parser.add_argument("--sim", default=None, help="sim res.bin (verify mode)")
    parser.add_argument("--ref", default=None, help="reference_scores.bin (verify mode)")
    parser.add_argument("--sim_idx", default=None, help="sim indices.bin (verify mode)")
    parser.add_argument("--ref_idx", default=None, help="reference_indices.bin (verify mode)")
    args = parser.parse_args()

    if args.mode == "gen":
        outdir = args.outdir or os.path.join(os.path.dirname(__file__),
                    "../../../../../compare/qli_fp8_B1_Sq64_Skv128_g64_Tm16_Tk32")
        outdir = os.path.abspath(outdir)
        print(f"Generating golden data to: {outdir}")

        q_fp8_f32, k_fp8_f32, w_f32, scale_q, scale_k = gen_inputs(outdir)

        ref_scores = compute_golden(q_fp8_f32, k_fp8_f32, w_f32, scale_q, scale_k)
        ref_scores.tofile(os.path.join(outdir, "reference_scores.bin"))
        print(f"  reference_scores.bin: {ref_scores.nbytes} bytes  [{SQ}, {SKV}] FP32")

        ref_indices = compute_topk(ref_scores)
        ref_indices.tofile(os.path.join(outdir, "reference_indices.bin"))
        print(f"  reference_indices.bin: {ref_indices.nbytes} bytes  [{SQ}, {TOPK}] INT32")

        print("\nDone.")

    elif args.mode == "verify":
        if not all([args.sim, args.ref, args.sim_idx, args.ref_idx]):
            sys.exit("verify mode requires --sim --ref --sim_idx --ref_idx")
        ok = verify(args.sim, args.ref, args.sim_idx, args.ref_idx)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
