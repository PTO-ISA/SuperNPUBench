#ifndef QUANT_SPARSE_FLASH_MLA_V3_HPP
#define QUANT_SPARSE_FLASH_MLA_V3_HPP

// ============================================================================
// QuantSparseFlashMla v3 — 4-PE CSA kernel, ported from the ops-transformer
// arch35 reference (ops-transformer/attention/quant_sparse_flash_mla).
//
// Design goal: mirror the ops kernel's per-block computation flow and data
// movement (what is moved, when, and how much) within the SuperScalar 4-PE
// SPMD model.  Written against the PTO Tile-OP primitives only; it does not
// reuse the earlier SuperNPUBench QSMLA implementations.
//
// Ops-reference mapping (arch35 stage -> v3):
//   ProcessVec0         -> stage block rows into the GM v0-res buffer
//                          (indexed TopK gather: one 512 B row per
//                           TLOAD/TSTORE pair, mirroring CopyInSingleKv;
//                           contiguous tail blocks use the same row path)
//   IterateLoadQK       -> TLOAD<SharedMatrixRight, PEMask=1> (one shared
//                          load per 128-token block; ORI full blocks connect
//                          directly to GM, per user decision)
//   IterateBmm1CSA      -> TMATMUL(tScore, tQ, tKV, keep_acc())
//                          (single K=512 product, per user decision; output
//                           stays on a Vec-operable CUBE_M16 tile, matching
//                           the ops Fixpipe L0C->UB behaviour)
//   ProcessVec1         -> online softmax: x scale', + mask, rowmax,
//                          expdif, rowsum/TFMA, x16, TCVT -> HIF8
//   IterateBmm2CSA      -> TMATMUL(tPV, tPShard, tKV, keep_acc().transpose_b())
//   ProcessVec2         -> FlashUpdate: first block O = PV*descale,
//                          later blocks O = O*alpha + PV*descale
//   Vec2 epilogue       -> O / sum, O / 16, cast BF16, TSTORE
//
// Block size kS2Base = 128 KV tokens (ops s2BaseSize).  Tail blocks keep the
// ops behaviour: only valid rows are staged, invalid columns are masked with
// -1e30 ("dirty data removed by mask").
//
// Concurrency notes (gfrun/gfsim round-robin, PE0 always ahead):
//   - the partial-block mask is written in full by every PE (identical
//     values, so concurrent writes are benign) before its own TLOAD;
//   - the KV staging into v0-res is done by PE0 only, and the shared KV
//     TLOAD is initiated by PE0 (PEMask=1), so the store->load order is
//     guaranteed on the initiating thread;
//   - TMATMUL is cooperative (PE_MASK=1111) and waits for all PEs.
// ============================================================================

#include <common/pto_tileop.hpp>
#include "qsmla_config.hpp"
#include "qsmla_mode.hpp"

using namespace pto;

// ---------------------------------------------------------------------------
// Compile-time type/constant bundle for the v3 kernel.
// ---------------------------------------------------------------------------
template <typename qdtype, typename kvdtype, typename ModeConfig>
struct qsmla_v3_types {
    using Base = typename ModeConfig::Base;
    static_assert(ModeConfig::Mode == QsmlaMode::CSA,
                  "v3 implements the CSA mode only");

    static constexpr int S1 = Base::S1;
    static constexpr int S2 = Base::S2;
    static constexpr int N1 = Base::N1;
    static constexpr int N2 = Base::N2;
    static constexpr int D = Base::D;
    static constexpr int G = Base::G;
    static constexpr int CmpS2 = ModeConfig::CmpS2;
    static constexpr int CmpTopK = ModeConfig::CmpTopK;

    static_assert(G <= 64 && G % 16 == 0 && N1 % 16 == 0,
                  "v3 requires G in [16,64] divisible by the 4-PE split");

    static constexpr int kGroupM = G;        // cooperative M (ops mRealSize)
    static constexpr int kPeNum = 4;
    static constexpr int kPeTm = kGroupM / kPeNum;  // rows per PE (16)
    static constexpr int kS2Base = 128;      // ops s2BaseSize

    // Q: shared Left operand, loaded once per work item (ops CopyQGmToL1).
    using tileQMatrix = SharedMatrixLeft<qdtype, 64, D, kGroupM, D>;
    using tileQ = SharedTile<tileQMatrix>;
    // KV: shared Right operand, one 128-token block (ops l1RightBuffers).
    // Consumed by QK^T (no TransB: stored [N, K]) and by PV (TransB: [K, N])
    // — the ops K=V shared-latent behaviour.
    using tileKVMatrix = SharedMatrixRight<kvdtype, kS2Base, D>;
    using tileKV = SharedTile<tileKVMatrix>;
    // S = Q@K^T per-PE [16, 128] FP32 (ops bmm1Buffers after Fixpipe).
    using tileScore = CubeTileM16<float, kPeTm, kS2Base>;
    using tileScoreRed = VecTileM16<float, kPeTm, kS2Base, kPeTm, 1>;
    using tileRow = VecTileM16<float, kPeTm, 1, kPeTm, 1>;
    // P quantized to HIF8, Local Left shard (ops l1PBuffers).
    using tilePShard = CubeTileM16<qdtype, kPeTm, kS2Base>;
    // O / PV, per-PE [16, 512] FP32 (ops stage2OutBuf/bmm2Buffers).  Carried
    // on Left CUBE_M16 tiles (fa tW style) instead of the Acc tile: the
    // descale/flash-update chain runs TMULS/TADD on them, which TimingSim's
    // PTO #291 check rejects on Acc operands.
    using tileO = CubeTileM16<float, kPeTm, D>;
    using tilePV = CubeTileM16<float, kPeTm, D>;
    using tileOCast = CubeAccumulatorM16<__bf16, kPeTm, D>;
    // Partial-block mask tile — the exact tScore type so the TADD mask
    // application keeps one tile_shape (Location included).
    using tileMask = CubeTileM16<float, kPeTm, kS2Base>;
    // One KV row for the scalar-indexed staging path (ops CopyInSingleKv).
    using tileKvRow = Tile<Location::Vec, kvdtype, 1, D, BLayout::RowMajor>;

    using gmQView = global_tensor<qdtype, RowMajor<kGroupM, D>>;
    using gmKVView = global_tensor<kvdtype, RowMajor<kS2Base, D>>;
    using gmRowView = global_tensor<kvdtype, RowMajor<1, D>>;
    using gmOutView = global_tensor<__bf16, RowMajor<kPeTm, D>>;
    using gmOView = global_tensor<float, RowMajor<kPeTm, D>>;
    using gmMaskView = global_tensor<float, RowMajor<kPeTm, kS2Base>>;
};

// ---------------------------------------------------------------------------
// v3 entry point — 4-PE CSA (HIF8 in / BF16 out).
//
//   v0_res_scratch : [kS2Base, D] kvdtype staging buffer (ops v0Res).  CMP
//                    indexed blocks and ORI tail blocks are staged here by
//                    PE0 before the shared KV load.
//   mask_scratch   : 2 x [kPeTm, kS2Base] float mask buffers.  Region 0 is
//                    the all-zero mask written once per work item (ops
//                    zero_mask_buf); region 1 is rebuilt per partial block.
//   o_scratch      : [B*S1*N1, D] float O buffer.  Model-gap fallback: the
//                    ops kernel keeps O resident in UB (stage2OutBuf), but a
//                    resident O tile crossing the in-block scalar staging
//                    loops trips the compiler's Shared-pool keep-alive TMOV
//                    (Local TMOV assertion, gfrun) — the same gap that
//                    blocked the qsmla-oq-resident experiments.  v3 therefore
//                    half-residents O in GM (one TSTORE + one TLOAD per
//                    block, numerically identical).
//
// Everything lives in one function with a unified block loop (ORI blocks
// first, CMP blocks after — ops s2LoopCount < / >= oriKvLoopEndIdx) so the
// Shared Q handle never crosses a function boundary.
// ---------------------------------------------------------------------------
template <typename qdtype, typename kvdtype, typename odttype, typename ModeConfig>
void quant_sparse_flash_mla_csa_v3_4pe_pto(
    odttype *out_ptr, qdtype *q_ptr, kvdtype *ori_kv_ptr, kvdtype *cmp_kv_ptr,
    const int * /*ori_indices*/, const int *cmp_indices,
    const int * /*ori_lengths*/, const int * /*cmp_lengths*/,
    float softmax_scale,
    float q_descale, float ori_kv_descale, float cmp_kv_descale,
    int cmp_ratio, int win_left, int win_right,
    kvdtype *v0_res_scratch, float *mask_scratch, float *o_scratch)
{
    using Types = qsmla_v3_types<qdtype, kvdtype, ModeConfig>;
    using tileQMatrix = typename Types::tileQMatrix;
    using tileQ = typename Types::tileQ;
    using tileKVMatrix = typename Types::tileKVMatrix;
    using tileKV = typename Types::tileKV;
    using tileScore = typename Types::tileScore;
    using tileScoreRed = typename Types::tileScoreRed;
    using tileRow = typename Types::tileRow;
    using tilePShard = typename Types::tilePShard;
    using tileO = typename Types::tileO;
    using tilePV = typename Types::tilePV;
    using tileOCast = typename Types::tileOCast;
    using tileMask = typename Types::tileMask;
    using tileKvRow = typename Types::tileKvRow;
    using gmQView = typename Types::gmQView;
    using gmKVView = typename Types::gmKVView;
    using gmRowView = typename Types::gmRowView;
    using gmOutView = typename Types::gmOutView;
    using gmOView = typename Types::gmOView;
    using gmMaskView = typename Types::gmMaskView;

    constexpr int S1 = Types::S1;
    constexpr int S2 = Types::S2;
    constexpr int N1 = Types::N1;
    constexpr int N2 = Types::N2;
    constexpr int D = Types::D;
    constexpr int CmpS2 = Types::CmpS2;
    constexpr int CmpTopK = Types::CmpTopK;
    constexpr int kGroupM = Types::kGroupM;
    constexpr int kPeTm = Types::kPeTm;
    constexpr int kPeNum = Types::kPeNum;
    constexpr int kS2Base = Types::kS2Base;
    constexpr int kWorkCount = Types::Base::B * S1 * N2;

    constexpr float kProbScale = 16.0f;         // ops hifp8ScaleValue
    constexpr float kProbScaleRecip = 1.0f / 16.0f;
    constexpr float kMaskInactive = -1.0e30f;   // ops mask sentinel

    // Loop-invariant fixpipe options (hoisted, fa_2d_unroll_gmma style).
    constexpr auto qkOptions = fixp::keep_acc();
    constexpr auto pvOptions = fixp::keep_acc().transpose_b();

    const uint32_t tid = get_thread_idx();
    const float score_base = softmax_scale * q_descale;

    // Per-thread CSA TopK collection result (identical on every PE; the
    // index array itself is read-only so every PE can collect in parallel).
    int selected[CmpTopK > 0 ? CmpTopK : 1];

    for (int work = 0; work < kWorkCount; ++work) {
        const int batch = work / (S1 * N2);
        const int rem = work % (S1 * N2);
        const int q_token = rem / N2;
        const int kv_head = rem % N2;
        const int q_row_base = (batch * S1 + q_token) * N1 + kv_head * Types::G;

        // --- Q load, once per work item (ops CopyQGmToL1) -------------------
        tileQ tQ;
        gmQView gQ(q_ptr + q_row_base * D);
        TLOAD<tileQMatrix, 1>(tQ, gQ);

        // --- online-softmax state init (ops Vec1 first-block init) ----------
        tileRow tMax, tSum;
        TEXPANDS(tMax, -1.0e30f);
        TEXPANDS(tSum, 0.0f);

        // All-zero mask, written once per work item (ops Step E zero_mask_buf
        // fast path: full blocks consume the shared zero buffer instead of
        // materialising a fresh mask).  Every PE writes the full buffer —
        // identical values, so the concurrent writes are benign.
        {
            const int seg = kS2Base / kPeNum;
            const int col_begin = static_cast<int>(tid) * seg;
            const int col_end = col_begin + seg;
            for (int r = 0; r < kPeTm; ++r) {
                float *row = mask_scratch + r * kS2Base;
                for (int c = col_begin; c < col_end; ++c) {
                    row[c] = 0.0f;
                }
            }
        }

        // --- block source descriptor (ops ComputeS2LoopInfo) ----------------
        // ORI window (golden/ops Band-mode window semantics), then CMP TopK
        // collection with the causal-end filter (ops CalcCurValidS2 +
        // ProcessSparseKv: stop at the first -1, drop out-of-range and
        // over-causal indices).
        const QsmlaSwaRange range =
            qsmla_swa_range(S2, S1, q_token, win_left, win_right);
        const int ori_rows = range.end - range.begin;
        const int ori_blocks = (ori_rows + kS2Base - 1) / kS2Base;
        const int cmp_end =
            qsmla_csa_cmp_valid_end(CmpS2, S1, q_token, cmp_ratio);
        int cnt = 0;
        {
            const int *cand =
                cmp_indices + ((batch * S1 + q_token) * N2 + kv_head) * CmpTopK;
            for (int k = 0; k < CmpTopK; ++k) {
                const int idx = cand[k];
                if (idx == -1) {
                    break;
                }
                if (idx < 0 || idx >= CmpS2 || idx >= cmp_end) {
                    continue;
                }
                selected[cnt++] = idx;
            }
        }
        const int cmp_blocks = (cnt + kS2Base - 1) / kS2Base;
        const int total_blocks = ori_blocks + cmp_blocks;

        for (int blk = 0; blk < total_blocks; ++blk) {
            const bool is_cmp = blk >= ori_blocks;
            const bool is_first = blk == 0;

            int lo = 0;
            int hi = kS2Base;
            float kv_descale;
            kvdtype *kv_src = nullptr;

            if (!is_cmp) {
                // Blocks start at the window begin (ops s2LineStartIdx), so
                // the valid range is [0, hi) inside the block.
                const int remain = range.end - (range.begin + blk * kS2Base);
                hi = kS2Base < remain ? kS2Base : remain;
                kv_descale = ori_kv_descale;
                if (range.begin + (blk + 1) * kS2Base <= S2) {
                    // Full block inside the batch: direct shared load from GM
                    // (user decision; ops stages every block through v0Res).
                    kv_src = ori_kv_ptr +
                             (batch * S2 + range.begin + blk * kS2Base) * D;
                } else {
                    // Tail block: stage the valid rows through v0-res (ops
                    // Vec0 contiguous path); leftover rows stay dirty and are
                    // masked below.
                    if (tid == 0) {
                        kvdtype *src = ori_kv_ptr +
                            (batch * S2 + range.begin + blk * kS2Base) * D;
                        for (int r = 0; r < hi; ++r) {
                            tileKvRow tRow;
                            gmRowView gSrc(src + r * D);
                            TLOAD(tRow, gSrc);
                            gmRowView gDst(v0_res_scratch + r * D);
                            TSTORE(gDst, tRow);
                        }
                    }
                    kv_src = v0_res_scratch;
                }
            } else {
                const int row0 = (blk - ori_blocks) * kS2Base;
                const int remain = cnt - row0;
                hi = kS2Base < remain ? kS2Base : remain;
                kv_descale = cmp_kv_descale;
                if (tid == 0) {
                    // Indexed gather: one 512 B row per TLOAD/TSTORE pair
                    // (ops CopyInSingleKv).  PE0 owns the staging so the
                    // shared load it initiates observes the completed stores.
                    for (int r = 0; r < hi; ++r) {
                        kvdtype *src = cmp_kv_ptr +
                            (batch * CmpS2 + selected[row0 + r]) * D;
                        tileKvRow tRow;
                        gmRowView gSrc(src);
                        TLOAD(tRow, gSrc);
                        gmRowView gDst(v0_res_scratch + r * D);
                        TSTORE(gDst, tRow);
                    }
                }
                kv_src = v0_res_scratch;
            }

            // Mask source selection (ops: full blocks share the zero mask
            // buffer; partial blocks rebuild a [lo, hi) mask).  The TLOAD
            // itself stays unconditional — a Cube-tile TLOAD inside a
            // runtime conditional trips a backend Shared-copy bug — so the
            // choice is expressed purely through the GM pointer.
            const bool partial = (lo != 0 || hi != kS2Base);
            float *mask_src = mask_scratch;  // region 0: all-zero
            if (partial) {
                float *tail = mask_scratch + kPeTm * kS2Base;  // region 1
                const int seg = kS2Base / kPeNum;
                const int col_begin = static_cast<int>(tid) * seg;
                const int col_end = col_begin + seg;
                for (int r = 0; r < kPeTm; ++r) {
                    float *row = tail + r * kS2Base;
                    for (int c = col_begin; c < col_end; ++c) {
                        row[c] = (c >= lo && c < hi) ? 0.0f : kMaskInactive;
                    }
                }
                mask_src = tail;
            }

            // --- IterateLoadQK: KV block -> shared Right operand ------------
            tileKV tKV;
            gmKVView gKV(kv_src);
            TLOAD<tileKVMatrix, 1>(tKV, gKV);

            // --- IterateBmm1CSA: S = Q @ K^T (single K=512 product) ---------
            tileScore tScore;
            TMATMUL(tScore, tQ, tKV, qkOptions);

            // --- ProcessVec1: online softmax ---------------------------------
            const float score_scale = score_base * kv_descale;
            TMULS(tScore, tScore, score_scale);

            tileMask tMask;
            gmMaskView gMask(mask_src);
            TLOAD(tMask, gMask);
            TADD(tScore, tScore, tMask);
            // Full blocks add the shared all-zero mask (numerically a no-op,
            // mirroring the ops vector-side mask add on the zero buffer).

            tileScoreRed tLocalMaxR;
            TROWMAX(tLocalMaxR, tScore);
            auto tLocalMax = TREDUCEPREFIXVIEW<tileRow>(tLocalMaxR);

            tileRow tNewMax;
            TMAX(tNewMax, tMax, tLocalMax);

            // P0 = exp(S' - m_new); alpha = exp(m_old - m_new) for later
            // blocks only (the first block has no accumulated O to rescale).
            TROWEXPANDEXPDIF(tScore, tScore, tNewMax);

            tileRow tScale;
            if (!is_first) {
                TROWEXPANDEXPDIF(tScale, tMax, tNewMax);
            }

            tileScoreRed tLocalSumR;
            TROWSUM(tLocalSumR, tScore);
            auto tLocalSum = TREDUCEPREFIXVIEW<tileRow>(tLocalSumR);

            tileRow tNewSum;
            if (is_first) {
                TADD(tNewSum, tSum, tLocalSum);
            } else {
                TFMA(tNewSum, tSum, tScale, tLocalSum);
            }

            // --- Quantize P to HIF8 (ops: exp * 16 -> cast, l1PBuffers) -----
            tilePShard tPShard;
            TMULS(tScore, tScore, kProbScale);
            TCVT(tPShard, tScore);

            // --- IterateBmm2CSA + ProcessVec2: PV and flash update ----------
            tileO tO;
            if (is_first) {
                // First block: O = PV * descale (ops Vec2 direct-copy path).
                TMATMUL(tO, tPShard, tKV, pvOptions, kGroupM);
                TMULS(tO, tO, kv_descale);
            } else {
                gmOView gOLoad(o_scratch +
                    (q_row_base + static_cast<int>(tid) * kPeTm) * D);
                TLOAD(tO, gOLoad);
                tilePV tPV;
                TMATMUL(tPV, tPShard, tKV, pvOptions, kGroupM);
                TMULS(tPV, tPV, kv_descale);
                TROWEXPANDMUL(tO, tO, tScale);  // O *= alpha
                TADD(tO, tO, tPV);              // O += PV * descale
            }
            {
                gmOView gOStore(o_scratch +
                    (q_row_base + static_cast<int>(tid) * kPeTm) * D);
                TSTORE(gOStore, tO);
            }

            tMax = tNewMax;
            tSum = tNewSum;
        }

        // --- epilogue (ops Vec2 last block + Bmm2DataCopyOut) ----------------
        tileO tO;
        {
            gmOView gOLoad(o_scratch +
                (q_row_base + static_cast<int>(tid) * kPeTm) * D);
            TLOAD(tO, gOLoad);
        }
        TROWEXPANDDIV(tO, tO, tSum);            // O /= rowSum  (LastDiv)
        TMULS(tO, tO, kProbScaleRecip);         // O /= 16      (undo P scale)
        tileOCast tOCast;
        TCVT(tOCast, tO);                       // cast BF16
        gmOutView gOut(
            out_ptr + (q_row_base + static_cast<int>(tid) * kPeTm) * D);
        TSTORE(gOut, tOCast);
    }
}

#endif // QUANT_SPARSE_FLASH_MLA_V3_HPP
