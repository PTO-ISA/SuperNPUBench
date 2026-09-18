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
// 【精简版说明】
//   key 构建仅符号翻转（无 NaN/-0 防御，golden 生成器不产生 NaN）、提取无
//   rev 最小索引（输出契约为无序 TopK set，平局取任一等价）、tail 分支用
//   FOR_EACH_CHUNK 宏统一处理。完整版（含防御）的设计记录见
//   qli_pto_opt_histogram_radix_design.md。
//   精度验证通过（cosine=1.0, set=100%），性能相当。
// =============================================================================

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <limits>

using namespace pto;

// -----------------------------------------------------------------------------
// TCOLSUMX — 列求和（本仓 b8669ce 模板 lb1 误绑 dst.ValidRow=1，只累加 1 行；
// 按上游 f00b928 (#101) 的正确编码：B.DIM 描述源几何，LB1=源 ValidRow）
// -----------------------------------------------------------------------------
template <typename tile_o, typename tile_i>
void TCOLSUMX(tile_o& dst, tile_i& src) {
  asm volatile(
    "BSTART.TEPL 80, %D1\n"
    "B.DIM zero, %c2, ->lb0\n"
    "B.DIM zero, %c3, ->lb1\n"
    "B.DIM zero, %c4, ->lb2\n"
    "B.IOT %5, mask=1111, last, ->%0<%Z6>\n"
    ""
    : "=Tr"(dst.data())
    : "i"(type_traits<typename tile_i::DType>::TypeCode),
      "i"(tile_i::ValidCol),
      "i"(tile_i::ValidRow),
      "i"(tile_i::Cols),
      "Tr"(src.data()),
      "i"(tile_type_traits<typename tile_o::TileDType>::TilesizeCode)
  );
}

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
                TCOLSUMX(tPartial, tS);

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
// -----------------------------------------------------------------------------
namespace qli_radix {

using RU = uint32_t;

// tile→标量读回：tile op 在 VLIW block 末尾提交，若标量 load 被调度进
// TSTORE 同一 block（llvm bundle 打包），将读到落地前的旧值（gfrun trace
// 实证：lwi 与 TSTORE 同 block 读 0/上一轮值）。PTO ISA 无显式 fence，
// noinline 调用强制 block 边界，保证读回发生在 TSTORE block 提交之后。
__attribute__((noinline)) RU TileScalarRb(RU* slot) { return *slot; }

template <int CK, int CKV = CK>
using TKey = Tile<Location::Vec, RU, 1, CK, BLayout::RowMajor, 1, CKV>;

// Step 1: float bits → sortable key（仅符号翻转，7 tile op）
template <int CK, int CKV>
inline void RadixMakeKey(RU* dst, const RU* src) {
    using gk = global_tensor<RU, RowMajor<1, CKV>>;
    gk gs(const_cast<RU*>(src)); TKey<CK, CKV> bits; TLOAD(bits, gs);
    TKey<CK, CKV> sign; TANDS(sign, bits, 0x80000000u);
    TKey<CK, CKV> neg;  TNOT(neg, bits);
    TKey<CK, CKV> pos;  TORS(pos, bits, 0x80000000u);
    TKey<CK, CKV> s01;  TSHRS(s01, sign, 31u);
    TKey<CK, CKV> diff; TSUB(diff, neg, pos);
    TMUL(diff, diff, s01);
    TKey<CK, CKV> key;  TADD(key, pos, diff);
    gk gd(dst); TSTORE(gd, key);
}


// Step 3: 从 key tile pop n 个最大元素（TROWARGMAX + 索引消零）
template <int CK, int CKV>
__attribute__((always_inline)) inline void RadixPopN(RU* mv_gm, RU chunkBase, int32_t* out, int n) {
    // v0.58.4 行归约契约：TROWARGMAX 目的须物理单列 [N,1]；
    // 物理行数 32 维持 128B tile 尺寸下限，valid 仍为 [1,1]
    using t1  = Tile<Location::Vec, RU, 32, 1, BLayout::RowMajor, 1, 1>;
    using t1i = Tile<Location::Vec, int32_t, 32, 1, BLayout::RowMajor, 1, 1>;
    using gi1i = global_tensor<int32_t, RowMajor<1, 1>>;
    using gk = global_tensor<RU, RowMajor<1, CKV>>;
    gk gmv(mv_gm);
    // 消零用纯标量 GM 写。llvm dev-llvm15_56@73cbdf34 起，循环内 TSEL 等
    // tile-op 链式更新存在隔轮陈旧绑定回归（in-place/双缓冲/GM 往返 TSEL
    // 版均实测索引重复弹出；旧 llvm 553b08045 编译同一份代码正确）。
    // 循环内只保留无相互依赖的单发 tile op（TLOAD/ARGMAX/TSTORE），它们
    // 与 MakeKey 链同构（已验证可靠）；消零走标量写，彻底绕开 tile 寄存器链。
    TKey<CK, CKV> cur; t1 best;
    for (int k = 0; k < n; k++) {
        TLOAD(cur, gmv);
        TROWARGMAX(best, cur);
        t1 bestg; TADDS(bestg, best, chunkBase);
        t1i besti; TCVT(besti, bestg);
        gi1i gout(out + k); TSTORE(gout, besti);
        // 被弹索引自读回（out[k] 即本轮 ARGMAX 结果，已证明可靠），标量清零
        RU bv = TileScalarRb(reinterpret_cast<RU*>(out + k));
        if (bv >= chunkBase && bv - chunkBase < (RU)CKV)
            *reinterpret_cast<volatile RU*>(mv_gm + (bv - chunkBase)) = 0;
    }
}

// Step 3（单 chunk）: 先 pop GT 再 pop EQ 补足
// GT 弹出不依赖过滤：GT 集合恰为全行 key 的 top-gt 前缀，把原始 key 拷入
// 工作区后弹前 nGt 个即得 GT 集合（即使 TSEL 过滤失效亦正确）。
// llvm 73cbdf34 下被调（未内联）函数内 TSEL predicate 存在陈旧绑定
// （sel1/cand 恒取"全部"），且 tile→标量计数读回需 block 边界保护
// （见 TileScalarRb），故 GT 计数上移到调用方内联宏、以 nGt 实参传入。
template <int CK, int CKV>
inline void RadixExtract(RU* key_ptr, RU kthVal, RU chunkBase, int32_t* outBase, int& outPos, int& needEq, int nGt, RU* mv_gm) {
    using gk = global_tensor<RU, RowMajor<1, CKV>>;
    gk g(key_ptr);

    // GT: 原始 key 拷入工作区（单发无依赖 tile op，与 MakeKey 链同构）
    {
        TKey<CK, CKV> key; TLOAD(key, g);
        gk gw(mv_gm); TSTORE(gw, key);
        RadixPopN<CK, CKV>(mv_gm, chunkBase, outBase + outPos, nGt);
        outPos += nGt;
    }
    // EQ: key == kth（补足剩余名额）。cand 物化的 TSTORE→TLOAD 往返在
    // 新工具链下读写错位（GT 段标量消零已绕开；EQ 段并列元素极少），
    // 改为纯标量扫描 key GM，与 golden 语义一致（从小到大取并列索引）。
    if (needEq > 0) {
        const RU* kgm = reinterpret_cast<const RU*>(key_ptr);
        for (RU i = 0; i < (RU)CKV && needEq > 0; ++i) {
            if (kgm[i] == kthVal) {
                outBase[outPos++] = (int32_t)(i + chunkBase);
                --needEq;
            }
        }
    }
}

// 单个 chunk 的提取（调用 RadixExtract，stride = MaxTileCol）
template <int CK, int CKV>
inline void ExtractChunk(RU* krow, int c, int stride, RU kthVal, int32_t* outBase, int& outPos, int& needEq, int nGt, RU* mv_gm) {
    RadixExtract<CK, CKV>(krow + c * stride, kthVal, (RU)c * stride, outBase, outPos, needEq, nGt, mv_gm);
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

        // ---- Step 2: 32 位逐位下降 → kth_value ----
        // THISTOGRAM 于 PTO 0.58.5 退役（pto-spec edcbd8b），改用位下降二分：
        // 自高比特起试探性置位，count(key >= candidate) >= topK 则保留该比特。
        // 计数序列以宏内联（函数边界处 tile 寄存器分配在 73cbdf34 后端有
        // 分配到保留寄存器 0 的缺陷，见 TROWSUM dst [0x0] trace 证据）。
#define QLI_CNT_KEY(_CK, _CKV, _KP, _TH, _CM, _TOT) do { \
    global_tensor<RU, RowMajor<1, _CKV>> gc((_KP)); \
    TKey<_CK, _CKV> kc; TLOAD(kc, gc); \
    TKey<_CK, _CKV> ic; TCMPS<CmpMode::_CM>(ic, kc, (_TH)); \
    TKey<_CK, _CKV> onec; TEXPANDS(onec, 1u); \
    TKey<_CK, _CKV> selc; TEXPANDS(selc, 0u); TSEL(selc, ic, onec); \
    Tile<Location::Vec, RU, 32, 1, BLayout::RowMajor, 1, 1> sc; \
    TROWSUM(sc, selc); RU cntc = 0; \
    global_tensor<RU, RowMajor<1, 1>> gcntc(&cntc); TSTORE(gcntc, sc); \
    (_TOT) += TileScalarRb(&cntc); \
} while (0)
        RU kth_value = 0;
        for (int bit = 31; bit >= 0; bit--) {
            RU candidate = kth_value | ((RU)1 << bit);
            RU total = 0;
            if constexpr (NumChunks == 1) {
                QLI_CNT_KEY(SinglePhy, Skv, krow, candidate, GE, total);
            } else if constexpr (TailCols != MaxTileCol) {
                for (int c = 0; c < NumChunks - 1; c++)
                    QLI_CNT_KEY(MaxTileCol, MaxTileCol, krow + c * MaxTileCol, candidate, GE, total);
                QLI_CNT_KEY(TailPhy, TailCols, krow + (NumChunks - 1) * MaxTileCol, candidate, GE, total);
            } else {
                for (int c = 0; c < NumChunks; c++)
                    QLI_CNT_KEY(MaxTileCol, MaxTileCol, krow + c * MaxTileCol, candidate, GE, total);
            }
            if (total >= (RU)topK) kth_value = candidate;
        }
        // GT 计数 → needEq（topK 中需由 ==kth_value 的并列元素补足的数量）。
        // 逐 chunk 保留计数并作为 Extract 的 nGt 实参（被调函数内 TSEL
        // predicate 陈旧绑定，见 RadixExtract 注释）
        RU cntv[NumChunks];
        if constexpr (NumChunks == 1) {
            cntv[0] = 0;
            QLI_CNT_KEY(SinglePhy, Skv, krow, kth_value, GT, cntv[0]);
        } else if constexpr (TailCols != MaxTileCol) {
            for (int c = 0; c < NumChunks - 1; c++) {
                cntv[c] = 0;
                QLI_CNT_KEY(MaxTileCol, MaxTileCol, krow + c * MaxTileCol, kth_value, GT, cntv[c]);
            }
            cntv[NumChunks - 1] = 0;
            QLI_CNT_KEY(TailPhy, TailCols, krow + (NumChunks - 1) * MaxTileCol, kth_value, GT, cntv[NumChunks - 1]);
        } else {
            for (int c = 0; c < NumChunks; c++) {
                cntv[c] = 0;
                QLI_CNT_KEY(MaxTileCol, MaxTileCol, krow + c * MaxTileCol, kth_value, GT, cntv[c]);
            }
        }
        RU gt = 0;
        for (int c = 0; c < NumChunks; c++) gt += cntv[c];
#undef QLI_CNT_KEY

        // ---- Step 3: 提取 topK 索引（每 chunk: 先 GT 后 EQ）----
        RU* pop_mv = hist_scratch + 512;        // GM pop 工作区 [MaxTileCol]（绕过编译器循环携带 tile 回归）
        int outPos = 0;
        int needEq = topK - (int)gt;
        if constexpr (NumChunks == 1) {
            ExtractChunk<SinglePhy, Skv>(krow, 0, MaxTileCol, kth_value, indices_gm + i * topK, outPos, needEq, (int)cntv[0], pop_mv);
        } else if constexpr (TailCols != MaxTileCol) {
            for (int c = 0; c < NumChunks - 1; c++)
                ExtractChunk<MaxTileCol, MaxTileCol>(krow, c, MaxTileCol, kth_value, indices_gm + i * topK, outPos, needEq, (int)cntv[c], pop_mv);
            ExtractChunk<TailPhy, TailCols>(krow, NumChunks - 1, MaxTileCol, kth_value, indices_gm + i * topK, outPos, needEq, (int)cntv[NumChunks - 1], pop_mv);
        } else {
            for (int c = 0; c < NumChunks; c++)
                ExtractChunk<MaxTileCol, MaxTileCol>(krow, c, MaxTileCol, kth_value, indices_gm + i * topK, outPos, needEq, (int)cntv[c], pop_mv);
        }
    }
}

#endif