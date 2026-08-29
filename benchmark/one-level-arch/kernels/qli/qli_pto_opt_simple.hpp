#ifndef QLI_PTO_OPT_SIMPLE_HPP
#define QLI_PTO_OPT_SIMPLE_HPP

// =============================================================================
// qli_pto_opt_simple.hpp — Quant Lightning Indexer (精简 radix-select TopK)
// =============================================================================
//
// 【算子功能】
//   QuantLightningIndexer (QLI) 是 SparseFlashAttention (SFA) 的前处理算子。
//   从全量 K/V 序列中选出最关键的 token 索引，供后续稀疏注意力使用。
//
// 【计算公式】（与参考实现 quant_lightning_indexer_v2 一致）
//   score[s1, s2] = scale_k[s2] * Σ_g ( W[s1, g] * scale_q[s1, g] * ReLU(QK[g, s1, s2]) )
//
//   展开为 7 个步骤:
//     Step 1: S = Q @ K^T                       → [g, Skv]   量化矩阵乘
//     Step 2: S = ReLU(S)                        → [g, Skv]   激活过滤负信号
//     Step 3: load scale_q, fuse W *= scale_q    → [g, 1]    反量化 scale 折入权重
//     Step 4: S *= W*scale_q (broadcast)         → [g, Skv]   权重加权
//     Step 5: out = [1]_g @ S (ReduceG)          → [1, Skv]   沿 head 维度求和
//     Step 6: out *= scale_k                     → [1, Skv]   反量化 scale_k 后乘
//     Step 7: indices = TopK(out)                → [topK]     选取关键索引
//
// 【与参考实现的一致性】
//   参考实现 (quant_lightning_indexer_v2) 的 golden 公式:
//     score[s1,s2] = k_scale[s2] * Σ_g ( w[s1,g] * q_scale[s1,g] * ReLU(QK[g,s1,s2]) )
//   - scale_q[s1, g] 折入 weights，在 g-reduction 之内、ReLU 之后
//   - scale_k[s2] 在 g-reduction 之外作为后乘因子
//   本实现严格遵循此两阶段应用顺序。
//
// 【数据布局 — BSND】
//   Q:        [B, S, N, D] → 展平为 [S*N, D]，同一 token 的 g 个 head 连续存放
//   K:        [Skv, D]      → Key 不分组，所有 head 共享
//   W:        [S, N]        → 与 query 前 2 维同形，float
//   scale_q:  [S, N]        → 与 W 同形，per-token per-head，float
//   scale_k:  [Skv]         → per-token（N2=1），float
//   scores:   [S, Skv]      → ReduceG 后的输出
//   indices:  [S, topK]     → TopK 索引输出
//
// 【数据类型】
//   dtype: Q/K 的数据类型，支持 __half / int8_t
//   W / scale_q / scale_k: 统一使用 float（与参考实现 arch35 一致）
//
// 【与 CANN 原版的对应关系】
//   CANN 代码位置: op_kernel/arch22/quant_lightning_indexer_v2_*.h
//
//   Step 1 (Q@K^T):  CANN ComputeQk (Mmad)              → QLI TMATMUL(tSacc, tQ, tK)
//   Step 2 (ReLU):   CANN FixpSToL1 reluPre=1 (Fixpipe) → QLI TMAX(tS, tS, tZero)
//   Step 3 (W*Sq):   CANN ProcessVec0 Mul(W, QScale)    → QLI TMUL(tWf, tWf, tSq)
//   Step 4 (W广播):  CANN Brcb(W*QScale)                → QLI TCOLEXPANDMUL(tS, tS, tWf)
//   Step 5 (ReduceG):CANN ComputeWs Mmad([1,g]×[g,Skv]) → QLI TCOLSUM(tPartial, tRed)
//   Step 6 (ScK):    CANN ProcessVec1 Mul(mmIn, kScale) → QLI TMUL(tPartial, tPartial, tSk)
//   Step 7 (TopK):   CANN SortAll+MergeSort+LD           → QLI qli_topk_simple (NPU tile op)
//
//   CANN ProcessVec0 中先计算 W *= Scale_Q（Mul(inWeights, inQScale)），再 Brcb 广播；
//   QLI 在 Step 3 用 TMUL 融合 W *= scale_q，在 Step 4 用 TCOLEXPANDMUL 广播。
//   CANN ProcessVec1 中 Mul(mmIn, kScale) 在 g-reduction 之后应用 scale_k；
//   QLI 在 Step 6 用 TMUL 在 TCOLSUM 之后应用 scale_k。
//
// 【与 FA 的区别】
//   FA:  O = softmax(Q@K^T / √d) @ V    — online softmax, P@V 矩阵乘, 输出 [S, vD]
//   QLI: out = TopK(scale_k * ReduceG(W*scale_q * ReLU(Q@K^T)))  — 无 softmax, 无 V
//
// 【工具链约束】
//   - TCOLEXPANDMUL 使用 TileOP-API 标准算子（src0/src1/dst 均为 float，
//     满足 dtype 断言）。早期用于绕过 dtype 断言 bug 的 _TEPL 内联汇编变体
//     （test/common/template_asm.h）已随 v5 迁移弃用。
//   - TCOLSUM 参考 reduction/reducesum_colvec_pto.hpp 的归约模式。
//   - TCVT 用于布局转换（ColMajor → RowMajor）以满足 TCOLSUM 的输入要求。
//
// 【tile 尺寸约束】
//   每个 tile 活跃尺寸须在 128B..8KB（DavinciOO active PE-local profile）。
//   当前配置: dtype=__half/int8_t, D=128, kTm=16, kTk=32
//     tileQ:   [16, 128] × 2B = 4KB (fp16) / 2KB (int8)  ✅
//     tileK:   [32, 128] × 2B = 8KB (fp16) / 4KB (int8)  ✅
//     tileS:   [16, 32]  × 4B = 2KB (float)               ✅
//     tileWf:  [16, 8]   × 4B = 512B (float) valid[16,1]  ✅
//     tileSq:  [16, 8]   × 4B = 512B (float) valid[16,1]  ✅
//     tileSk:  [1,  32]  × 4B = 128B (float)              ✅ (同 tileSum)
//     tileSum: [1,  32]  × 4B = 128B (float)              ✅ (刚好达到下限)
//
// 【已知简化（相对 CANN 原版）】
//   - 不支持 Causal Mask (CANN sparse_mode=3, rightDownCausal)
//   - 不支持 actual_seq_lengths (变长序列)
//   - 不支持 PageAttention (block_table 间接寻址)
//   - TopK 在 NPU tile op 实现（精简 radix-select）
//   - g=kTm（未分块循环），g>16 时需扩展为 g/kTm 内层循环
// =============================================================================
//
// 【本文件与 qli_pto_opt.hpp 的关系】
//   qli_pto_opt.hpp 是完整功能的 radix-select TopK 实现（含自定义汇编 THISTOGRAMX、
//   NaN/-0 防御、rev 最小索引等增强特性）。
//   本文件是精简版，key 构建仅符号翻转（无 NaN/-0 防御）、提取无 rev 最小索引、
//   tail 分支用 FOR_EACH_CHUNK 宏统一处理（代码量约 60%）。
//   精度验证通过（cosine=1.0, set=100%），性能相当。
// =============================================================================

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <limits>

using namespace pto;

// -----------------------------------------------------------------------------
// qli_pto — QLI 核心计算（Step 1-6，NPU tile op 实现）
// -----------------------------------------------------------------------------
//
// 模板参数:
//   dtype   : Q/K 的数据类型（__half / int8_t）
//   Sq      : Query 序列长度 S
//   Skv     : Key 序列长度（通常 = Sq）
//   D       : Head 维度（固定 128）
//   g       : Head 数 N，kTm 必须等于 g
//   kTm     : tile M 维度（= g）
//   kTk     : tile K/N 维度（K block 大小，如 32）
//
// 输入:
//   q_ptr       : Q [Sq*g, D]       BSND 布局，dtype
//   k_ptr       : K [Skv, D]        所有 head 共享，dtype
//   wb_ptr      : 预广播 W*scale_q [Sq*g, kTk] float（行 r 全列同值）
//   scale_k_ptr : scale_k [Skv]     per-token 反量化 scale，float
//
// 输出:
//   scores_ptr  : scores [Sq, Skv]  ReduceG + scale_k 后的 score 矩阵，float
//
// 注：v0.58.4 row-expansion 校验要求广播源物理单列（<128B 不可加载），
// 故 W*scale_q 由调用方预广播为 [kTm, kTk] tile，kernel 内用普通 TMUL。
template <typename dtype, int Sq, int Skv, int D, int g, int kTm, int kTk>
void qli_pto(float* scores_ptr,
             dtype* q_ptr, dtype* k_ptr,
             float* wb_ptr,
             float* scale_k_ptr,
             float* temp_gm = nullptr)   // [kTm, kTk] CUBE->Vec 桥接临时区
{
    constexpr int Qb = Sq;
    constexpr int Kb = Skv / kTk;
    constexpr int Gb = g / kTm;
    static_assert(g % kTm == 0, "g must be multiple of kTm for G-blocking");
    static_assert(kTm <= 32, "CUBE_M16/M32 matmul supports kTm <= 32");

    // 临时区缺省：紧随 scores 之后（调用方保证可写 kTm*kTk*4 字节）
    static float* s_default_temp = nullptr;
    if (temp_gm == nullptr) {
        if (s_default_temp == nullptr) {
            s_default_temp = scores_ptr + (uint64_t)Sq * Skv + 2048;
        }
        temp_gm = s_default_temp;
    }

    using gmQ   = global_tensor<dtype,  RowMajor<Sq * g, D>>;
    using gmK   = global_tensor<dtype,  RowMajor<D, Skv>>;   // K^T [D, Skv] 行主序（CUBE_N8 契约）
    using gmOut = global_tensor<float,  RowMajor<Sq, Skv>>;
    using gmTmp = global_tensor<float,  RowMajor<kTm, kTk>>;

    // PTO v0.58.4 CUBE cell-layout：TMATMUL 的 A 用 CUBE_M16/M32、B 用 CUBE_N8、
    // D 用 CUBE 累加器；TSTORE_CUBE 写回 GM 后 TLOAD 回 Vec tile 继续 Vec 链
    using tileQ    = std::conditional_t<(kTm <= 16),
                       CubeTileM16<dtype, kTm, D>, CubeTileM32<dtype, kTm, D>>;
    using tileK    = CubeTileN8<dtype, D, kTk>;
    using tileSCube = std::conditional_t<(kTm <= 16),
                       CubeAccumulatorM16<float, kTm, kTk>,
                       CubeAccumulatorM32<float, kTm, kTk>>;
    using tileS    = Tile<Location::Vec, float, kTm, kTk, BLayout::RowMajor>;
    using tileWb   = Tile<Location::Vec, float, kTm, kTk, BLayout::RowMajor>;
    using tileSum  = Tile<Location::Vec, float, 1, kTk, BLayout::RowMajor>;
    using tileSk   = Tile<Location::Vec, float, 1, kTk, BLayout::RowMajor>;

    using itQ   = global_iterator<gmQ,   tileQ>;
    using itK   = global_iterator<gmK,   tileK>;
    using itOut = global_iterator<gmOut, tileSum>;

    itQ   gIterQ(q_ptr);
    itK   gIterK(k_ptr);
    itOut gIterOut(scores_ptr);

    for (int i = 0; i < Qb; i++) {
        for (int j = 0; j < Kb; j++) {
            tileK tK;
            auto gK = gIterK(0, j);
            TLOAD_CUBE(tK, gK);

            tileSk tSk;
            {
                using gmSkLocal = global_tensor<float, RowMajor<1, Skv>>;
                using itSkLocal = global_iterator<gmSkLocal, tileSk>;
                itSkLocal gIterSk(scale_k_ptr);
                auto gSk = gIterSk(0, j);
                TLOAD(tSk, gSk);
            }

            tileSum tSum;
            tileSum tZeroSum;
            TEXPANDS(tZeroSum, 0.0f);

            #pragma clang loop unroll(full)
            for (int gi = 0; gi < Gb; gi++) {
                tileQ tQ;
                auto gQ = gIterQ(i * Gb + gi, 0);
                TLOAD_CUBE(tQ, gQ);

                // 预广播 W*scale_q tile [kTm, kTk]（调用方 wb_ptr 提供，
                // 行 r 全列同值 = w[r]*scale_q[r]；规避 v0.58.4 单列广播源校验）
                tileWb tWb;
                {
                    using gmWbLocal = global_tensor<float, RowMajor<kTm, kTk>>;
                    using itWbLocal = global_iterator<gmWbLocal, tileWb>;
                    itWbLocal gIterWb(wb_ptr + (uint64_t)(i * Gb + gi) * kTm * kTk);
                    auto gWb = gIterWb(0, 0);
                    TLOAD(tWb, gWb);
                }

                // TMATMUL（CUBE cell-layout）→ TSTORE_CUBE 桥接回 Vec
                tileSCube tSCube;
                TMATMUL(tSCube, tQ, tK);
                gmTmp gTmp(temp_gm);
                TSTORE_CUBE(gTmp, tSCube);

                tileS tS;
                TLOAD(tS, gTmp);

                tileS tZero;
                TEXPANDS(tZero, 0.0f);
                TMAX(tS, tS, tZero);

                TMUL(tS, tS, tWb);

                tileSum tPartial;
                TCOLSUM(tPartial, tS);

                if (gi == 0) {
                    TADD(tSum, tZeroSum, tPartial);
                } else {
                    TADD(tSum, tSum, tPartial);
                }
            }

            TMUL(tSum, tSum, tSk);

            auto gOut = gIterOut(i, j);
            TSTORE(gOut, tSum);
        }
    }
}

// -----------------------------------------------------------------------------
// THISTOGRAMX — 自研展开版 THISTOGRAM（允许 Idx 与 src 不同 shape）
// -----------------------------------------------------------------------------
template <typename tile_o, typename tile_s, typename tile_idx>
void THISTOGRAMX(tile_o& dst, tile_s& src, tile_idx& idx, int ByteId) {
#define THISTOGRAMX_ASM(BYTE_NAME)                                    \
  asm volatile(                                                        \
    "BSTART.TEPL 104, %D1\n"                                           \
    "B.DATR %D1, " BYTE_NAME ", Zero\n"                                \
    "B.DIM %3, 0, ->LB0\n"                                             \
    "B.DIM %4, 0, ->LB1\n"                                             \
    "B.DIM zero, %c5, ->LB2\n"                                         \
    "B.IOT %6, %7, mask=1111, last, ->%0<%Z8>\n"                       \
    ""                                                                 \
    : "=Tr"(dst.data())                                                \
    : "i"(type_traits<typename tile_s::DType>::TypeCode),              \
      "i"(type_traits<typename tile_o::DType>::TypeCode),              \
      "r"(dst.GetValidCol()),                                          \
      "r"(src.GetValidRow()),                                          \
      "i"(tile_o::Cols),                                               \
      "Tr"(src.data()),                                                \
      "Tr"(idx.data()),                                                \
      "i"(tile_type_traits<typename tile_o::TileDType>::TilesizeCode))
  switch (ByteId) {
    case 0: THISTOGRAMX_ASM("Byte0"); break;
    case 1: THISTOGRAMX_ASM("Byte1"); break;
    case 2: THISTOGRAMX_ASM("Byte2"); break;
    default: THISTOGRAMX_ASM("Byte3"); break;
  }
#undef THISTOGRAMX_ASM
}

// -----------------------------------------------------------------------------
// qli_topk_radix — 4 轮 MSD radix-select TopK（Step 7，模板函数风格）
// -----------------------------------------------------------------------------
// 【算法】
//   1. float→uint32 sortable key（仅符号翻转，7 tile op）
//      key = (sign) ? ~bits : (bits | 0x80000000)
//   2. 逐字节 MSD radix（Byte3→Byte0）→ kth_value
//   3. 提取：key > kth 全部入选，key == kth 边界并列补足
//      每轮提取 = TROWARGMAX + 索引消零（TMUL，无标量 store）
// -----------------------------------------------------------------------------
namespace qli_radix {

using RU = uint32_t;

template <int CK, int CKV = CK>
using TKey = Tile<Location::Vec, RU, 1, CK, BLayout::RowMajor, 1, CKV>;

// Step 1: float bits → sortable key（仅符号翻转，7 tile op）
template <int CK, int CKV>
inline void RadixMakeKey(RU* dst, const RU* src) {
    using gk = global_tensor<RU, RowMajor<1, CKV>>;
    gk gs(const_cast<RU*>(src));
    TKey<CK, CKV> bits; TLOAD(bits, gs);
    TKey<CK, CKV> sign; TANDS(sign, bits, 0x80000000u);
    TKey<CK, CKV> neg;  TNOT(neg, bits);
    TKey<CK, CKV> pos;  TORS(pos, bits, 0x80000000u);
    TKey<CK, CKV> s01;  TSHRS(s01, sign, 31u);
    TKey<CK, CKV> diff; TSUB(diff, neg, pos);
    TMUL(diff, diff, s01);
    TKey<CK, CKV> key;  TADD(key, pos, diff);
    gk gd(dst); TSTORE(gd, key);
}

// Step 2: 单 chunk 直方图（THISTOGRAMX + Idx 前缀）
template <int CK, int CKV>
inline void RadixChunkHist(RU* key_ptr, int byteId, RU* prefix, RU* hist) {
    using gk = global_tensor<RU, RowMajor<1, CKV>>;
    using tidx = Tile<Location::Vec, RU, 4, 8, BLayout::RowMajor>;
    using gidx = global_tensor<RU, RowMajor<4, 8>>;
    using th = Tile<Location::Vec, RU, 1, 256, BLayout::RowMajor>;
    using gh = global_tensor<RU, RowMajor<1, 256>>;
    gk g(key_ptr);
    TKey<CK, CKV> key; TLOAD(key, g);
    tidx idxTile; { gidx gi(prefix); TLOAD(idxTile, gi); }
    th hist_tile; THISTOGRAMX(hist_tile, key, idxTile, byteId);
    gh gout(hist); TSTORE(gout, hist_tile);
}

// Step 3: 从 key tile pop n 个最大元素（TROWARGMAX + 索引消零）
template <int CK, int CKV>
inline void RadixPopN(TKey<CK, CKV>& mv, RU chunkBase, int32_t* out, int n) {
    using t1  = Tile<Location::Vec, RU, 1, 32, BLayout::RowMajor, 1, 1>;
    using t1i = Tile<Location::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 1>;
    using gi1 = global_tensor<RU, RowMajor<1, 1>>;
    using gi1i = global_tensor<int32_t, RowMajor<1, 1>>;
    TKey<CK, CKV> idxTile; TCI(idxTile, chunkBase);
    for (int k = 0; k < n; k++) {
        t1 best; TROWARGMAX(best, mv);
        t1 bestg; TADDS(bestg, best, chunkBase);
        t1i besti; TCVT(besti, bestg);
        gi1i gout(out + k); TSTORE(gout, besti);
        RU bv = 0; gi1 gbv(&bv); TSTORE(gbv, bestg);
        TKey<CK, CKV> bbc; TEXPANDS(bbc, bv);
        // TCMP tile-tile 在 f94bc12 工具链不可用（cmode 助记符不同步）：
        // 用 TSUB 求差 + TCMPS==0 判等价
        TKey<CK, CKV> diff; TSUB(diff, idxTile, bbc);
        TKey<CK, CKV> isp; TCMPS<CmpMode::EQ>(isp, diff, 0u);
        TKey<CK, CKV> one; TEXPANDS(one, 1u);
        TKey<CK, CKV> np; TSUB(np, one, isp);
        TMUL(mv, mv, np);
    }
}

// Step 3（单 chunk）: 先 pop GT 再 pop EQ 补足
template <int CK, int CKV>
inline void RadixExtract(RU* key_ptr, RU kthVal, RU chunkBase, int32_t* outBase, int& outPos, int& needEq) {
    using gk = global_tensor<RU, RowMajor<1, CKV>>;
    using t1 = Tile<Location::Vec, RU, 1, 32, BLayout::RowMajor, 1, 1>;
    using gi1 = global_tensor<RU, RowMajor<1, 1>>;
    gk g(key_ptr);

    // GT: key > kth（TCMPS 标量比较，f94bc12 工具链 TCMP tile-tile 不可用）
    {
        TKey<CK, CKV> key; TLOAD(key, g);
        TKey<CK, CKV> isgt; TCMPS<CmpMode::GT>(isgt, key, kthVal);
        t1 s; TROWSUM(s, isgt); RU cnt = 0; gi1 gcnt(&cnt); TSTORE(gcnt, s);
        TKey<CK, CKV> cand; TMUL(cand, key, isgt);
        RadixPopN<CK, CKV>(cand, chunkBase, outBase + outPos, (int)cnt);
        outPos += (int)cnt;
    }
    // EQ: key == kth（补足剩余名额）
    if (needEq > 0) {
        TKey<CK, CKV> key; TLOAD(key, g);
        TKey<CK, CKV> iseq; TCMPS<CmpMode::EQ>(iseq, key, kthVal);
        t1 s; TROWSUM(s, iseq); RU cnt = 0; gi1 gcnt(&cnt); TSTORE(gcnt, s);
        int take = (int)cnt; if (take > needEq) take = needEq;
        TKey<CK, CKV> cand; TMUL(cand, key, iseq);
        RadixPopN<CK, CKV>(cand, chunkBase, outBase + outPos, take);
        outPos += take; needEq -= take;
    }
}

// 单个 chunk 的直方图 + 差分合并到 per_bin
template <int CK, int CKV>
inline void HistMergeChunk(RU* krow, int c, int stride, int r, RU* prefix, RU* hist, RU* per_bin) {
    RadixChunkHist<CK, CKV>(krow + c * stride, r, prefix, hist);
    volatile RU* rh = reinterpret_cast<volatile RU*>(hist);
    RU prev = 0;
    for (int b = 0; b < 256; b++) { RU cum = rh[b]; per_bin[b] += (cum - prev); prev = cum; }
}

// 单个 chunk 的提取（调用 RadixExtract，stride = MaxTileCol）
template <int CK, int CKV>
inline void ExtractChunk(RU* krow, int c, int stride, RU kthVal, int32_t* outBase, int& outPos, int& needEq) {
    RadixExtract<CK, CKV>(krow + c * stride, kthVal, (RU)c * stride, outBase, outPos, needEq);
}

}  // namespace qli_radix

// ==================== qli_topk_radix 主函数 ====================
template <int Sq, int Skv, int topK>
void qli_topk_radix(float* scores_gm, int32_t* indices_gm) {
    using namespace qli_radix;
    static_assert(Skv % 8 == 0, "Skv must be multiple of 8");
    static_assert(topK <= Skv, "topK must be <= Skv");

    constexpr int MaxTileCol = 2048;
    constexpr int NumChunks = (Skv + MaxTileCol - 1) / MaxTileCol;
    constexpr int TailCols = Skv - (NumChunks - 1) * MaxTileCol;
    constexpr int SinglePhy = (Skv < 128) ? 128 : Skv;
    constexpr int TailPhy = (TailCols < 128) ? 128 : TailCols;

    uint32_t* key_scratch = reinterpret_cast<uint32_t*>(
            reinterpret_cast<uint8_t*>(indices_gm) + (uint64_t)Sq * topK * 4 + 8192);
    uint32_t* hist_scratch = key_scratch + (uint64_t)Sq * Skv;
    uint32_t* prefix_buf = hist_scratch + 256;

    for (int i = 0; i < Sq; i++) {
        const RU* row = reinterpret_cast<const RU*>(scores_gm) + (uint64_t)i * Skv;
        RU* krow = key_scratch + (uint64_t)i * Skv;

        // ---- Step 1: float → sortable key（每 chunk 7 tile op）----
        if constexpr (NumChunks == 1) {
            RadixMakeKey<SinglePhy, Skv>(krow, row);
        } else if constexpr (TailCols != MaxTileCol) {
            for (int c = 0; c < NumChunks - 1; c++)
                RadixMakeKey<MaxTileCol, MaxTileCol>(krow + c * MaxTileCol, row + c * MaxTileCol);
            RadixMakeKey<TailPhy, TailCols>(krow + (NumChunks - 1) * MaxTileCol, row + (NumChunks - 1) * MaxTileCol);
        } else {
            for (int c = 0; c < NumChunks; c++)
                RadixMakeKey<MaxTileCol, MaxTileCol>(krow + c * MaxTileCol, row + c * MaxTileCol);
        }

        // ---- Step 2: 4 轮 MSD radix → kth_value ----
        RU kth_value = 0;
        int remaining = topK;
        for (int w = 0; w < 32; w++) prefix_buf[w] = 0;

        for (int r = 3; r >= 0; r--) {
            RU per_bin[256] = {0};
            // 前缀: 已定高位字节（ByteId<3 时生效）
            if (r < 3) {
                volatile RU* pb = reinterpret_cast<volatile RU*>(prefix_buf);
                RU b3 = (kth_value >> 24) & 0xFFu;
                RU b2 = (kth_value >> 16) & 0xFFu;
                RU b1 = (kth_value >> 8) & 0xFFu;
                for (int cc = 0; cc < 8; cc++) {
                    pb[0*8+cc] = b3; pb[1*8+cc] = b2;
                    pb[2*8+cc] = b1; pb[3*8+cc] = 0;
                }
            }
            // 直方图（每 chunk 1×THISTOGRAM）+ 差分合并
            if constexpr (NumChunks == 1) {
                HistMergeChunk<SinglePhy, Skv>(krow, 0, MaxTileCol, r, prefix_buf, hist_scratch, per_bin);
            } else if constexpr (TailCols != MaxTileCol) {
                for (int c = 0; c < NumChunks - 1; c++)
                    HistMergeChunk<MaxTileCol, MaxTileCol>(krow, c, MaxTileCol, r, prefix_buf, hist_scratch, per_bin);
                HistMergeChunk<TailPhy, TailCols>(krow, NumChunks - 1, MaxTileCol, r, prefix_buf, hist_scratch, per_bin);
            } else {
                for (int c = 0; c < NumChunks; c++)
                    HistMergeChunk<MaxTileCol, MaxTileCol>(krow, c, MaxTileCol, r, prefix_buf, hist_scratch, per_bin);
            }

            // 高桶向下累计 → kth_byte
            RU cum = 0; int kth_byte = 0;
            for (int b = 255; b >= 0; b--) {
                cum += per_bin[b];
                if (cum >= (RU)remaining) { kth_byte = b; break; }
            }
            remaining -= (int)(cum - per_bin[kth_byte]);
            kth_value |= ((RU)kth_byte) << (r * 8);
        }

        // ---- Step 3: 提取 topK 索引（每 chunk: 先 GT 后 EQ）----
        int outPos = 0;
        int needEq = remaining;
        if constexpr (NumChunks == 1) {
            ExtractChunk<SinglePhy, Skv>(krow, 0, MaxTileCol, kth_value, indices_gm + i * topK, outPos, needEq);
        } else if constexpr (TailCols != MaxTileCol) {
            for (int c = 0; c < NumChunks - 1; c++)
                ExtractChunk<MaxTileCol, MaxTileCol>(krow, c, MaxTileCol, kth_value, indices_gm + i * topK, outPos, needEq);
            ExtractChunk<TailPhy, TailCols>(krow, NumChunks - 1, MaxTileCol, kth_value, indices_gm + i * topK, outPos, needEq);
        } else {
            for (int c = 0; c < NumChunks; c++)
                ExtractChunk<MaxTileCol, MaxTileCol>(krow, c, MaxTileCol, kth_value, indices_gm + i * topK, outPos, needEq);
        }
    }
}

#endif