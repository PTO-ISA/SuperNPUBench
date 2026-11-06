#ifndef QUANT_SPARSE_FLASH_MLA_V3_3_3_HPP
#define QUANT_SPARSE_FLASH_MLA_V3_3_3_HPP

// ============================================================================
// QuantSparseFlashMla v3.3.3 —— v3.3.1（O 全 UB 驻留 + staged-KV 行级 ping-pong）
// + **可配置预取深度**：staging slice 份数 = 预取距离 = `QSMLA_PRELOAD`
//（编译期常量，默认 2）。对齐 ops 的 v0ResGmBuffers / l1RightBuffers 环形缓冲。
//
// (historical) v3.3.1 —— v3.3（O 全 UB 驻留 + staged-KV 预取）
// + staging 内部的**行级 ping-pong**：两个行 tile 交替，使第 r+1 行的
// TLOAD 与第 r 行的 TSTORE 重叠（对齐 ops 的 stage0InBuf/stage0OutBuf 双缓冲）。
//
// (historical) v3.3 —— v3.2（staged-KV 预取）+ O 全 UB 驻留
//（= 把 v3.1.1 的 O 累加器跨 block 循环携带）。
// 每个块同时只存活 1 个 Shared KV tile（两个并发 Shared 句柄会让模型死锁）。
// 这里预取的是“下一块的 GM 暂存”（索引 gather / 尾块行）到备用 slice：
// 其 TLOAD/TSTORE 是异步 TLSU 操作，可与当前块的 CUBE/VEC 计算重叠。
//
//（历史）—— 4-PE CSA kernel，从 ops-transformer arch35 参考实现
//（ops-transformer/attention/quant_sparse_flash_mla）移植而来。
//
// v3.1 = v3 + PA_BBND（page-attention 块表）KV 布局支持。传入非空
// QsmlaPaConfig 即通过块表寻址 KV cache；默认（空表）保持原 BSND 连续布局。
// PA 模式下关闭“ORI 满块直连 Shared TLOAD”快路径（改为按行经 v0-res、
// 用块表映射地址逐行暂存）。
//
// 设计目标：在 SuperScalar 4-PE SPMD 模型内，镜像 ops kernel 的逐块计算流与
// 数据搬运（搬什么、何时搬、搬多少）。只使用 PTO Tile-OP 原语编写，不复用
// 早先的 SuperNPUBench QSMLA 实现。
//
// ops 参考映射（arch35 阶段 -> v3）：
//   ProcessVec0         -> 把块内行暂存到 GM 的 v0-res 缓冲
//                          （TopK 索引 gather：每次 TLOAD/TSTORE 搬一行 512B，
//                           对齐 CopyInSingleKv；连续尾块走同一条行路径）
//   IterateLoadQK       -> TLOAD<SharedMatrixRight, PEMask=1>（每个 128-token 块
//                          一次共享加载；ORI 满块按用户决定直连 GM）
//   IterateBmm1CSA      -> TMATMUL(tScore, tQ, tKV, keep_acc())
//                          （单次 K=512 乘，按用户决定；结果留在可做向量运算的
//                           CUBE_M16 tile 上，对齐 ops 的 Fixpipe L0C->UB 行为）
//   ProcessVec1         -> online softmax：缩放、加 mask、rowmax、expdif、
//                          rowsum/TFMA、乘 16、TCVT -> HIF8
//   IterateBmm2CSA      -> TMATMUL(tPV, tPShard, tKV, keep_acc().transpose_b())
//   ProcessVec2         -> FlashUpdate：首块 O = PV*descale，
//                          后续块 O = O*alpha + PV*descale
//   Vec2 epilogue       -> O / sum、O / 16、转 BF16、TSTORE
//
// 块大小 kS2Base = 128 个 KV token（ops s2BaseSize）。尾块保持 ops 行为：
// 只暂存有效行，无效列用 -1e30 掩掉（“脏数据由 mask 清除”）。
//
// 并发说明（gfrun/gfsim 为轮转调度，PE0 始终领先）：
//   - 部分块的 mask 由每个 PE 在自身 TLOAD 之前整块写入（值完全相同，因此
//     并发写是良性的）；
//   - KV 暂存到 v0-res 只由 PE0 执行，共享 KV TLOAD 也由 PE0（PEMask=1）
//     发起，从而在发起线程上保证 store->load 的先后顺序；
//   - TMATMUL 是协作式（PE_MASK=1111），会等待全部 PE。
// ============================================================================

#include <common/pto_tileop.hpp>
#include "qsmla_config.hpp"
#include "qsmla_mode.hpp"

using namespace pto;

// ---------------------------------------------------------------------------
// v3.1 新增：PA_BBND（page-attention 块表）KV 寻址。
//
// 物理 KV cache 布局（BBND，N2=1）：[BlockNum, BlockSize, 1, D]。逻辑上第 pos
// 行位于池中的行号：
//     block_table[batch * max_blocks + pos / block_size] * block_size
//         + pos % block_size
// （对齐 ops arch35 quant_sparse_flash_mla_csa_block_vector.h 的 blockTableGm；
// N2=1 时池内行步长仍为 D）。空表退化为 BSND 连续映射（按 batch 基行），
// 因此同一份 kernel 可服务两种布局。
// ---------------------------------------------------------------------------
struct QsmlaPaConfig {
    const int32_t *ori_block_table = nullptr;   // [B, ori_max_blocks]
    const int32_t *cmp_block_table = nullptr;   // [B, cmp_max_blocks]
    int32_t ori_block_size = 0;
    int32_t cmp_block_size = 0;
    int32_t ori_max_blocks = 0;
    int32_t cmp_max_blocks = 0;
};

inline int64_t qsmla_pa_physical_row(
    const int32_t *table, int32_t max_blocks, int32_t block_size,
    int32_t batch, int32_t pos)
{
    if (table == nullptr || block_size <= 0) {
        return pos;
    }
    const int32_t logical_block = pos / block_size;
    const int32_t in_block = pos - logical_block * block_size;
    return static_cast<int64_t>(table[batch * max_blocks + logical_block]) *
               block_size +
           in_block;
}


// ---------------------------------------------------------------------------
// v3.1 新增：Query 的 GM 布局。BSND 为 (b, s, n, d)；TND 为 (t, n, d)，其中
// 展平后的 token 行 t = cu_seqlens_q[batch] + s（ops arch35
// offset_calculator GM_Q_OUT_TND；cu_seqlens 数组带一个前导 0）。空
// cu_seqlens_q 退化为按 batch 的均匀基址 batch * S1，因此同一份 kernel 可
// 服务两种布局。
// ---------------------------------------------------------------------------
struct QsmlaQueryConfig {
    const int32_t *cu_seqlens_q = nullptr;  // TND prefix sums [B + 1], first 0
    bool q_tnd = false;                     // false: BSND, true: TND
};

inline int64_t qsmla_q_token_base(
    const QsmlaQueryConfig &qc, int32_t batch, int32_t s1)
{
    if (qc.q_tnd && qc.cu_seqlens_q != nullptr) {
        return static_cast<int64_t>(qc.cu_seqlens_q[batch]);
    }
    return static_cast<int64_t>(batch) * s1;
}


// ---------------------------------------------------------------------------
// v3 kernel 的编译期类型/常量集合。
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

    static_assert(G % 16 == 0 && N1 % 16 == 0,
                  "v3 requires G divisible by the 4-PE M16 split");

    static constexpr int kGroupM = Base::GSliceMax;  // 64-head G slice (ops mRealSize)
    static constexpr int kPeNum = 4;
    static constexpr int kPeTm = kGroupM / kPeNum;  // rows per PE (16)
    static constexpr int kS2Base = 128;      // ops s2BaseSize

    // Q：共享 Left 操作数，每个 work item 加载一次（ops CopyQGmToL1）。
    using tileQMatrix = SharedMatrixLeft<qdtype, 64, D, kGroupM, D>;
    using tileQ = SharedTile<tileQMatrix>;
    // KV：共享 Right 操作数，一次一个 128-token 块（ops l1RightBuffers）。
    // 同时被 QK^T（不做 TransB：按 [N, K] 存储）与 PV（做 TransB：[K, N]）消费
    // —— 即 ops 中 K=V 共享隐变量的行为。
    using tileKVMatrix = SharedMatrixRight<kvdtype, kS2Base, D>;
    using tileKV = SharedTile<tileKVMatrix>;
    // S = Q@K^T，每 PE [16, 128] FP32（ops bmm1Buffers，Fixpipe 之后）。
    using tileScore = CubeTileM16<float, kPeTm, kS2Base>;
    using tileScoreRed = VecTileM16<float, kPeTm, kS2Base, kPeTm, 1>;
    using tileRow = VecTileM16<float, kPeTm, 1, kPeTm, 1>;
    // P 量化为 HIF8，Local Left 分片（ops l1PBuffers）。
    using tilePShard = CubeTileM16<qdtype, kPeTm, kS2Base>;
    // O / PV，每 PE [16, 512] FP32（ops stage2OutBuf/bmm2Buffers）。承载在 Left
    // CUBE_M16 tile 上（fa tW 风格）而非 Acc tile：descale/FlashUpdate 链要在其
    // 上做 TMULS/TADD，而 TimingSim 的 PTO #291 校验拒绝在 Acc 操作数上做这些
    // 运算。
    using tileO = CubeTileM16<float, kPeTm, D>;
    using tilePV = CubeTileM16<float, kPeTm, D>;
    using tileOCast = CubeAccumulatorM16<__bf16, kPeTm, D>;
    // 部分块 mask tile —— 与 tScore 完全同型，使 TADD 加 mask 时保持同一
    // tile_shape（含 Location）。
    using tileMask = CubeTileM16<float, kPeTm, kS2Base>;
    // 标量索引暂存路径用的单行 KV tile（ops CopyInSingleKv）。
    using tileKvRow = Tile<Location::Vec, kvdtype, 1, D, BLayout::RowMajor>;

    using gmQView = global_tensor<qdtype, RowMajor<kGroupM, D>>;
    using gmKVView = global_tensor<kvdtype, RowMajor<kS2Base, D>>;
    using gmRowView = global_tensor<kvdtype, RowMajor<1, D>>;
    using gmOutView = global_tensor<__bf16, RowMajor<kPeTm, D>>;
    using gmOView = global_tensor<float, RowMajor<kPeTm, D>>;
    using gmMaskView = global_tensor<float, RowMajor<kPeTm, kS2Base>>;
};

// ---------------------------------------------------------------------------
// v3 入口 —— 4-PE CSA（HIF8 入 / BF16 出）。
//
//   v0_res_scratch : [kS2Base, D] kvdtype 暂存缓冲（ops v0Res）。CMP 索引块与
//                    ORI 尾块在共享 KV 加载前由 PE0 暂存到这里。
//   mask_scratch   : 2 × [kPeTm, kS2Base] float mask 缓冲。区域 0 是每个 work
//                    item 写一次的全区零 mask（ops zero_mask_buf）；区域 1 在
//                    每个部分块重建。
//   o_scratch      : [B*S1*N1, D] float O 缓冲。模型缺口的兜底方案：ops kernel
//                    把 O 常驻在 UB（stage2OutBuf），但驻留的 O tile 跨过块内
//                    标量暂存循环会触发编译器的 Shared-pool keep-alive TMOV
//                    （gfrun 的 Local TMOV 断言）—— 与阻塞 qsmla-oq-resident
//                    实验的是同一个缺口。因此 v3 让 O 半驻留在 GM（每块一次
//                    TSTORE + 一次 TLOAD，数值等价）。
//
// 全部逻辑集中在一个函数、一个统一的块循环里（先 ORI 块，后 CMP 块 —— 对齐
// ops 的 s2LoopCount < / >= oriKvLoopEndIdx），使 Shared Q 句柄不跨越函数
// 边界。
// ---------------------------------------------------------------------------
template <typename qdtype, typename kvdtype, typename odttype, typename ModeConfig>
void quant_sparse_flash_mla_csa_v3_3_3_4pe_pto(
    odttype *out_ptr, qdtype *q_ptr, kvdtype *ori_kv_ptr, kvdtype *cmp_kv_ptr,
    const int * /*ori_indices*/, const int *cmp_indices,
    const int * /*ori_lengths*/, const int * /*cmp_lengths*/,
    float softmax_scale,
    float q_descale, float ori_kv_descale, float cmp_kv_descale,
    int cmp_ratio, int win_left, int win_right,
    kvdtype *v0_res_scratch, float *mask_scratch, float *o_scratch,
    const QsmlaPaConfig &pa = QsmlaPaConfig(),
    const QsmlaQueryConfig &qc = QsmlaQueryConfig())
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
    constexpr int kWorkCount = Types::Base::WorkCount;

    constexpr float kProbScale = 16.0f;         // ops hifp8ScaleValue
    constexpr float kProbScaleRecip = 1.0f / 16.0f;
    constexpr float kMaskInactive = -1.0e30f;   // ops mask sentinel

    // 循环不变的 fixpipe 选项（已外提，fa_2d_unroll_gmma 风格）。
    constexpr auto qkOptions = fixp::keep_acc();
    constexpr auto pvOptions = fixp::keep_acc().transpose_b();

    const uint32_t tid = get_thread_idx();
    const float score_base = softmax_scale * q_descale;

    // 每线程的 CSA TopK 收集结果（各 PE 完全相同；索引数组本身只读，所以每个 PE
    // 都可以并行收集）。
    int selected[CmpTopK > 0 ? CmpTopK : 1];

    for (int work = 0; work < kWorkCount; ++work) {
        // G 切分对齐 v2 的统一路径：Config::decode_work 给出 64-head 切片（g_slice）
        // 与 head 基址 q_head_begin = kv_head*G + g_slice*GSliceMax，使 work 顺序、
        // Q 行基址与输出偏移都与 v2 布局一致。
        const QsmlaWorkItem item = Types::Base::decode_work(work);
        const int batch = item.batch;
        const int q_token = item.q_token;
        const int kv_head = item.kv_head;
        // Q 行布局（qc）：BSND 基址为 batch * S1，TND 基址为 cu_seqlens_q[batch]
        // （对等长 benchmark 用例两者给出同一个展平 token 行）。
        const int64_t q_tok = qsmla_q_token_base(qc, batch, S1) + q_token;
        const int64_t q_row_base = q_tok * N1 + item.q_head_begin;
        // v3.1：BSND 行带每 batch 基址；PA_BBND 行是来自块表的绝对池行（基址 0）。
        const int64_t kOriBaseRow = (pa.ori_block_table == nullptr)
            ? static_cast<int64_t>(batch) * S2 : 0;
        const int64_t kCmpBaseRow = (pa.cmp_block_table == nullptr)
            ? static_cast<int64_t>(batch) * CmpS2 : 0;

        // --- Q 加载：每个 work item 一次（ops CopyQGmToL1）-------------------
        tileQ tQ;
        gmQView gQ(q_ptr + q_row_base * D);
        TLOAD<tileQMatrix, 1>(tQ, gQ);

        // --- online-softmax 状态初始化（ops Vec1 首块初始化）---------------
        tileRow tMax, tSum;
        TEXPANDS(tMax, -1.0e30f);
        TEXPANDS(tSum, 0.0f);

        // v3.3：O 累加器在整个块循环期间全 UB 驻留。
        tileO tO;

        // 全区零 mask：每个 work item 写一次（ops Step E zero_mask_buf 快路径：满块
        // 直接复用共享零缓冲，不必现造 mask）。每个 PE 都写整块 —— 值完全相同，因此
        // 并发写是良性的。
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

        // --- 块源描述（ops ComputeS2LoopInfo）----------------
        // 先定 ORI 窗口（golden/ops 的 Band 模式窗口语义），再做带因果端过滤的 CMP
        // TopK 收集（ops CalcCurValidS2 + ProcessSparseKv：遇到第一个 -1 停止，丢弃
        // 越界与超出因果范围的索引）。
        const QsmlaSwaRange range =
            qsmla_swa_range(S2, S1, q_token, win_left, win_right);
        const int ori_rows = range.end - range.begin;
        const int ori_blocks = (ori_rows + kS2Base - 1) / kS2Base;
        const int cmp_end =
            qsmla_csa_cmp_valid_end(CmpS2, S1, q_token, cmp_ratio);
        int cnt = 0;
        {
            const int *cand =
                cmp_indices + (q_tok * N2 + kv_head) * CmpTopK;
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

#ifndef QSMLA_V3_1_ORI_FORCE_STAGE
#define QSMLA_V3_1_ORI_FORCE_STAGE 0
#endif
        // ------------------------------------------------------------------
        // v3.2：staged-KV 预取。resolve_block() 计算一个块的 KV 源并把它的行暂存到
        // 指定的 GM slice；compute_one() 对已解码的块执行 mask + QK^T + softmax + PV。
        // 同一时刻只存活 1 个 Shared KV 句柄（v3.1 约定）。
        // ------------------------------------------------------------------

        auto resolve_block = [&](int blk, kvdtype *stage_slot,
                                 int &out_lo, int &out_hi,
                                 float &out_descale, kvdtype *&out_src,
                                 tileKvRow &tRowA, tileKvRow &tRowB) __attribute__((always_inline)) {
                const bool is_cmp = blk >= ori_blocks;
                const bool is_first = blk == 0;

                int lo = 0;
                int hi = kS2Base;
                float kv_descale;
                kvdtype *kv_src = nullptr;

                if (!is_cmp) {
                    // 块从窗口起点开始（ops s2LineStartIdx），
                    // 因此块内有效范围是 [0, hi)。
                    const int remain = range.end - (range.begin + blk * kS2Base);
                    hi = kS2Base < remain ? kS2Base : remain;
                    kv_descale = ori_kv_descale;
    #ifndef QSMLA_V3_1_ORI_FORCE_STAGE
    #define QSMLA_V3_1_ORI_FORCE_STAGE 0
    #endif
                    const int ori_pos0 = range.begin + blk * kS2Base;
                    const bool ori_full_block =
                        (range.begin + (blk + 1) * kS2Base <= S2);
                    // PA_BBND：当逻辑块按 block_size 对齐、且 PA 块大小等于 kernel 块大小时，
                    // 整个块就是一个连续物理块，映射后的基址仍支持直连共享加载。
                    const bool pa_ori_direct = ori_full_block &&
                        (pa.ori_block_table != nullptr) &&
                        (pa.ori_block_size == kS2Base) &&
                        (ori_pos0 % pa.ori_block_size == 0);
                    if (ori_full_block && QSMLA_V3_1_ORI_FORCE_STAGE == 0 &&
                        (pa.ori_block_table == nullptr || pa_ori_direct)) {
                        // batch 内的满块：直接从 GM 做共享加载。
                        const int64_t base_row = (pa.ori_block_table == nullptr)
                            ? static_cast<int64_t>(batch) * S2 + ori_pos0
                            : static_cast<int64_t>(
                                  pa.ori_block_table[batch * pa.ori_max_blocks +
                                                     ori_pos0 / pa.ori_block_size]) *
                                  pa.ori_block_size;
                        kv_src = ori_kv_ptr + base_row * D;
                    } else {
                        // 尾块：把有效行经 v0-res 暂存（ops Vec0 连续路径）；
                        // 剩余行保持脏数据，由下面的 mask 清除。
                        if (tid == 0) {
                            // v3.3.1 行级 ping-pong：两个行 tile 交替，第 r+1 行的
                            // TLOAD 与第 r 行的 TSTORE 重叠（tile 复用，避免逐行新建句柄）。
                            int r = 0;
                            for (; r + 1 < hi; r += 2) {
                                const int64_t row0 = kOriBaseRow +
                                    qsmla_pa_physical_row(pa.ori_block_table,
                                        pa.ori_max_blocks, pa.ori_block_size,
                                        batch, range.begin + blk * kS2Base + r);
                                gmRowView gSrc0(ori_kv_ptr + row0 * D);
                                TLOAD(tRowA, gSrc0);
                                const int64_t row1 = kOriBaseRow +
                                    qsmla_pa_physical_row(pa.ori_block_table,
                                        pa.ori_max_blocks, pa.ori_block_size,
                                        batch, range.begin + blk * kS2Base + r + 1);
                                gmRowView gSrc1(ori_kv_ptr + row1 * D);
                                TLOAD(tRowB, gSrc1);
                                gmRowView gDst0(stage_slot + r * D);
                                TSTORE(gDst0, tRowA);
                                gmRowView gDst1(stage_slot + (r + 1) * D);
                                TSTORE(gDst1, tRowB);
                            }
                            if (r < hi) {
                                const int64_t row = kOriBaseRow +
                                    qsmla_pa_physical_row(pa.ori_block_table,
                                        pa.ori_max_blocks, pa.ori_block_size,
                                        batch, range.begin + blk * kS2Base + r);
                                gmRowView gSrc(ori_kv_ptr + row * D);
                                TLOAD(tRowA, gSrc);
                                gmRowView gDst(stage_slot + r * D);
                                TSTORE(gDst, tRowA);
                            }
                        }
                        kv_src = stage_slot;
                    }
                } else {
                    const int row0 = (blk - ori_blocks) * kS2Base;
                    const int remain = cnt - row0;
                    hi = kS2Base < remain ? kS2Base : remain;
                    kv_descale = cmp_kv_descale;
                    if (tid == 0) {
                        // 索引 gather：每次 TLOAD/TSTORE 搬一行 512B（ops CopyInSingleKv）。
                        // v3.3.1 行级 ping-pong：两个行 tile 交替，两行一组发出搬运。
                        int r = 0;
                        for (; r + 1 < hi; r += 2) {
                            const int64_t kvRow0 = kCmpBaseRow +
                                qsmla_pa_physical_row(pa.cmp_block_table,
                                    pa.cmp_max_blocks, pa.cmp_block_size,
                                    batch, selected[row0 + r]);
                            gmRowView gSrc0(cmp_kv_ptr + kvRow0 * D);
                            TLOAD(tRowA, gSrc0);
                            const int64_t kvRow1 = kCmpBaseRow +
                                qsmla_pa_physical_row(pa.cmp_block_table,
                                    pa.cmp_max_blocks, pa.cmp_block_size,
                                    batch, selected[row0 + r + 1]);
                            gmRowView gSrc1(cmp_kv_ptr + kvRow1 * D);
                            TLOAD(tRowB, gSrc1);
                            gmRowView gDst0(stage_slot + r * D);
                            TSTORE(gDst0, tRowA);
                            gmRowView gDst1(stage_slot + (r + 1) * D);
                            TSTORE(gDst1, tRowB);
                        }
                        if (r < hi) {
                            const int64_t row = kCmpBaseRow +
                                qsmla_pa_physical_row(pa.cmp_block_table,
                                    pa.cmp_max_blocks, pa.cmp_block_size,
                                    batch, selected[row0 + r]);
                            gmRowView gSrc(cmp_kv_ptr + row * D);
                            TLOAD(tRowA, gSrc);
                            gmRowView gDst(stage_slot + r * D);
                            TSTORE(gDst, tRowA);
                        }
                    }
                    kv_src = stage_slot;
                }
            out_lo = lo;
            out_hi = hi;
            out_descale = kv_descale;
            out_src = kv_src;
        };

#ifndef QSMLA_PRELOAD
#define QSMLA_PRELOAD 2
#endif
        // v3.3.3：可配置预取深度 —— staging slice 份数 = 预取距离 = kPreload。
        // 对应 ops 的 v0ResGmBuffers / l1RightBuffers（三缓冲）环形结构。
        constexpr int kPreload = QSMLA_PRELOAD;
        static_assert(kPreload >= 1, "QSMLA_PRELOAD 必须 >= 1");

        // 每个 slot 一份 staging 区，外加它的描述符环（lo/hi/descale/src）。
        kvdtype *v0_slots[kPreload];
        int d_lo[kPreload];
        int d_hi[kPreload];
        float d_des[kPreload];
        kvdtype *d_src[kPreload];
        for (int i = 0; i < kPreload; ++i) {
            v0_slots[i] = v0_res_scratch + i * kS2Base * D;
            d_lo[i] = 0;
            d_hi[i] = kS2Base;
            d_des[i] = 0.0f;
            d_src[i] = v0_slots[i];
        }

        // 前奏：先把前 min(kPreload, total_blocks) 块暂存好，供开头几拍消费。
        const int preload_fill =
            (total_blocks < kPreload) ? total_blocks : kPreload;
        for (int i = 0; i < preload_fill; ++i) {
            tileKvRow fillRowA, fillRowB;
            resolve_block(i, v0_slots[i], d_lo[i], d_hi[i], d_des[i], d_src[i],
                          fillRowA, fillRowB);
        }

        for (int blk = 0; blk < total_blocks; ++blk) {
            const bool is_first = blk == 0;
            const int slot = blk % kPreload;

            // 预取：在本块计算的同时，暂存「blk + (kPreload-1)」块到对应 slot。
            // 距离取 kPreload-1 而不是 kPreload：否则 blk 与 blk+kPreload 会映射到
            // 同一个 slot，把当前块尚未加载的数据覆盖掉。
            const int pf = blk + kPreload - 1;
            if (pf < total_blocks) {
                const int pslot = pf % kPreload;
                tileKvRow stageRowA, stageRowB;
                resolve_block(pf, v0_slots[pslot], d_lo[pslot], d_hi[pslot],
                              d_des[pslot], d_src[pslot],
                              stageRowA, stageRowB);
            }

            // 当前块的共享 KV 加载（数据由前奏或前 kPreload 拍暂存好，或满 ORI
            // 块直接用 GM 指针）。
            tileKV kv;
            gmKVView gKV(d_src[slot]);
            TLOAD<tileKVMatrix, 1>(kv, gKV);

            {
                const int lo = d_lo[slot];
                const int hi = d_hi[slot];
                const float kv_descale = d_des[slot];

                // mask 源选择（ops：满块共用零 mask 缓冲；部分块重建 [lo, hi) 的 mask）。
                // TLOAD 本身保持无条件执行 —— 运行时条件分支里的 Cube-tile TLOAD 会触发后端
                // Shared-copy 缺陷 —— 因此只用 GM 指针来表达选择。
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

                // --- IterateBmm1CSA：S = Q @ K^T（单次 K=512 乘）-------------
                tileScore tScore;
                TMATMUL(tScore, tQ, kv, qkOptions);

                // --- ProcessVec1：online softmax ---------------------------------
                const float score_scale = score_base * kv_descale;
                TMULS(tScore, tScore, score_scale);

                tileMask tMask;
                gmMaskView gMask(mask_src);
                TLOAD(tMask, gMask);
                TADD(tScore, tScore, tMask);
                // 满块加上共享的全零 mask（数值上是恒等操作，对齐 ops 在零缓冲上加 mask 的
                // 向量侧行为）。

                tileScoreRed tLocalMaxR;
                TROWMAX(tLocalMaxR, tScore);
                auto tLocalMax = TREDUCEPREFIXVIEW<tileRow>(tLocalMaxR);

                tileRow tNewMax;
                TMAX(tNewMax, tMax, tLocalMax);

                // P0 = exp(S' - m_new)；alpha = exp(m_old - m_new) 仅后续块需要
                // （首块没有已累积的 O 需要重缩放）。
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

                // --- 把 P 量化为 HIF8（ops：exp * 16 -> cast，l1PBuffers）------
                tilePShard tPShard;
                TMULS(tScore, tScore, kProbScale);
                TCVT(tPShard, tScore);

                // --- IterateBmm2CSA + ProcessVec2：PV 与 flash 更新 ----------
                if (is_first) {
                    // 首块：O = PV * descale（ops Vec2 直接拷贝路径）。
                    TMATMUL(tO, tPShard, kv, pvOptions, kGroupM);
                    TMULS(tO, tO, kv_descale);
                } else {
                    tilePV tPV;
                    TMATMUL(tPV, tPShard, kv, pvOptions, kGroupM);
                    TMULS(tPV, tPV, kv_descale);
                    TROWEXPANDMUL(tO, tO, tScale);  // O *= alpha
                    TADD(tO, tO, tPV);              // O += PV * descale
                }

                tMax = tNewMax;
                tSum = tNewSum;

            }

        }

        // --- epilogue（ops Vec2 最后一块 + Bmm2DataCopyOut）----------------
        TROWEXPANDDIV(tO, tO, tSum);            // O /= rowSum  (LastDiv)
        TMULS(tO, tO, kProbScaleRecip);         // O /= 16      (undo P scale)
        tileOCast tOCast;
        TCVT(tOCast, tO);                       // cast BF16
        gmOutView gOut(
            out_ptr + (q_row_base + static_cast<int>(tid) * kPeTm) * D);
        TSTORE(gOut, tOCast);
    }
}

#endif // QUANT_SPARSE_FLASH_MLA_V3_3_3_HPP
