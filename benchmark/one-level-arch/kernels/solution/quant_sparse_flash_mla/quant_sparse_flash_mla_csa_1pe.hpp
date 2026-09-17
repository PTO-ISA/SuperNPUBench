#ifndef QUANT_SPARSE_FLASH_MLA_CSA_1PE_HPP
#define QUANT_SPARSE_FLASH_MLA_CSA_1PE_HPP

// =============================================================================
// quant_sparse_flash_mla_csa_1pe.hpp
//   Quant Sparse Flash MLA（CSA 模式）—— 单 PE Local CUBE 实现
//
// 【定位】参考 4-PE 统一实现（quant_sparse_flash_mla.hpp）的 CSA 双源语义，
//   跑在单 PE Local CUBE 路径（机制来自已验证的 quant_sparse_flash_mla_tadd.hpp）。
//   用于单核 CSA 的精度/性能对照（vs 4-PE csa_tadd_4pe）。
//   dtype 双态：FP16（descale=1，P 不量化）或 HIF8（与官方算子/4-PE 对齐：
//   per-tensor descale、P×16 量化 HIF8、PV×kvDescale、输出×1/16、BF16 输出）。
//
// 【CSA 语义】（与 4-PE 逐项对齐）
//   O = softmax(Q·K^T × softmax_scale + mask) · V，双逻辑源拼接：
//   - ORI 源：窗口 [diag-wl, diag+wr+1) ∩ [0, S2) 的连续 token；
//     块对齐裁剪（qsmla_swa_block_range）+ 窗口 mask（对齐边缘屏蔽）
//   - CMP 源：cmp_valid_end = clamp(⌊(CmpS2·ratio−S1+q+1)/ratio⌋, 0, CmpS2)；
//     candidate = cmp_topk_length（缺省 CmpTopK）clamp 到 [0, CmpTopK]，
//     collect 剔除越界/超 valid_end 索引（-1 终止），按逻辑序 gather
//   - 双源顺序：先 ORI 后 CMP；两遍式各自按相同顺序重访
//   - score 缩放：softmax_scale × q_descale × kv_descale（FP16 下三者中
//     descale 均为 1；HIF8 下 per-tensor 反量化，ORI/CMP 各自 kv_descale）
//   - HIF8 量化契约（对齐官方算子）：P×16 → HIF8 → PV × kv_descale →
//     输出 ×1/16（kHif8ProbabilityScale=16 与 hifp8ScaleValue 一致）
//
// 【单 PE 机制】
//   - 转置/gather 一律 MGATHER 索引搬运（方案 transpose，参考
//     basic_op/transpose 的 tile_transpose_nd）：TTRANS 已退役（TEPL
//     0x6E）且 TLOAD_CUBE 仅支持 ND 源（issue #17：ISA 能力缺口）
//   - ORI 侧 qli 式输入预转置：K^T 布局由 ori_kv_t_ptr 直接提供
//     （[D, S2] 每 batch），kernel 内零转置；V 保持原样 [S2, D]
//   - CMP 侧 MGATHER 双链直出 batch_cmp（行序 + 转置序，dtype 无关）
//   - mask tile 化（U32 位模式域 TCI/TCMPS/TSEL 链）
//   - 方阵 kTk == kTd：未打补丁 TileOP 的 Local-B 双约定仅在方阵上一致
//   - 行归约源 tW [kTm, kTk] FP32 ≤ 2048B（pto-spec 0.58.6 契约）
//   - 主段 Db 循环 unroll_count(4)：unroll(full) 的 32 组在飞 tile
//     峰值会打满 256KB tile 池（HIF8 轨迹，gfsim CheckAddrOverflow）
//
// 【函数化边界约束】（v2 kernel 同款先例）
//   tile 对象必须保持函数局部：跨函数的 tile 引用/捕获会触发编译器
//   raw TSTORE 栈 spill（模型 RecordRawTileTransport 断言）。因此
//   辅助函数只封装"tile 全局部"的块，跨块状态一律经 GM scratch 传递
//   （score/prob/pv 缓冲），online-softmax 状态更新（tMax/tSum）与
//   Pass2 的 P 构造（tMax/tInvSum 依赖）有意保留在主函数内联。
// =============================================================================

#include <common/pto_tileop.hpp>
// NOTE: 不 include bench 的 "template_asm.h"——它的旧式 MGATHER 封装
// （BSTART.TMA 4，全局命名空间）与 TileOP-API 的活跃 MGATHER
// （BSTART.TLSU MGATHER，namespace pto）同名且参数为非 const 引用，
// 会在重载决议中遮蔽 API 版并触发 "Match Instruction Error"。
// 本 kernel 的全部指令（含 MGATHER/TCI/TDIVS）都来自 pto_tileop.hpp。
#include "qsmla_config.hpp"
#include "qsmla_mode.hpp"
#include <bit>
#include <type_traits>

using namespace pto;

// =============================================================================
// Section 1 — Tile 类型目录与编译期契约
//
// 单 PE Local CUBE 的全部 tile / GM 视图 / 迭代器类型与形状常量。
// CUBE 操作数与累加器为 STATIC（PTO v0.58 拒绝动态 valid 形状）。
// =============================================================================
template <typename qdtype, typename kvdtype, typename odttype,
          typename Config, typename ModeConfig>
struct CsaTiles {
    // ---- dtype 别名（供辅助函数以 Tiles::xxx 引用）----
    using qdtype_alias = qdtype;
    using kvdtype_alias = kvdtype;
    using odttype_alias = odttype;

    // ---- 形状常量 ----
    static constexpr int kB = Config::B;
    static constexpr int s1 = Config::S1;
    static constexpr int s2 = Config::S2;          // ORI S2
    static constexpr int N1 = Config::N1;
    static constexpr int D = Config::D;
    static constexpr int kTm = Config::TileM;
    static constexpr int kTk = Config::TileK;
    static constexpr int kTd = Config::TileD;
    static constexpr int kDb = D / kTd;
    static constexpr int kCmpS2 = ModeConfig::CmpS2;
    static constexpr int kCmpTopK = ModeConfig::CmpTopK;
    // 转置/行缓冲的物理容量：向上对齐到 kTk 的倍数，保证 iterator
    // 分块（列宽 kTk）永不越过分配的行距边界。
    static constexpr int kCmpCap =
        (kCmpTopK + kTk - 1) / kTk * kTk;
    static constexpr int kOriBlkMax = (s2 + kTk - 1) / kTk;
    static constexpr int kCmpBlkMax = (kCmpTopK + kTk - 1) / kTk;
    static constexpr int kMBlockCount = N1 / kTm;  // M 维 = head（每 q_token）

    // ---- 编译期契约 ----
    static_assert((std::is_same_v<qdtype, __half> &&
                   std::is_same_v<kvdtype, __half> &&
                   std::is_same_v<odttype, __half>) ||
                      (std::is_same_v<qdtype, __hif8> &&
                       std::is_same_v<kvdtype, __hif8> &&
                       std::is_same_v<odttype, __bf16>),
                  "single-PE CSA supports FP16 or HIF8 dtype sets");
    static_assert(Config::N2 == 1,
                  "single-PE CSA requires contiguous N2=1 KV");
    static_assert(kTk == kTd,
                  "unpatched-headers single-PE requires square B tiles "
                  "(Tk == Td_block)");
    static_assert(kTm <= 32, "single-PE Local CUBE supports TileM <= 32");
    static_assert(kTm * kTk * 4 <= 2048,
                  "row-reduction source tW must be <= 2048 bytes "
                  "(pto-spec 0.58.6)");
    static_assert(D % kTd == 0, "D-tail unsupported");
    static_assert(N1 % kTm == 0, "N1 must be a multiple of TileM");
    static_assert(kCmpTopK > 0, "CSA requires positive CmpTopK");

    // HIF8 量化契约（与 4-PE kUseHif8Probability / 官方 hifp8ScaleValue 对齐）
    static constexpr bool kUseHif8Probability =
        std::is_same_v<qdtype, __hif8>;
    static constexpr float kHif8ProbabilityScale = 16.0f;

    // ---- CUBE 操作数 / 累加器 ----
    using tileQ = CubeTileM32<qdtype, kTm, kTd>;            // A: Q [Tm, Td]
    using tileKRight = CubeTileN8<kvdtype, kTd, kTk>;       // B: K^T [Td, Tk]
    using tileV = CubeTileN8<kvdtype, kTk, kTd>;            // B: V   [Tk, Td]
    using tileWLeft = CubeTileM32<qdtype, kTm, kTk>;        // A: P   [Tm, Tk]
    using tileScoreCube = CubeAccumulatorM32<float, kTm, kTk>;
    using tilePVCube = CubeAccumulatorM32<float, kTm, kTd>;

    // ---- Vec 引擎 tile ----
    using tileW = Tile<Location::Vec, float, kTm, kTk, BLayout::RowMajor>;
    using tileMask = tileW;
    using tileP = Tile<Location::Vec, qdtype, kTm, kTk, BLayout::RowMajor>;
    using tileO = Tile<Location::Vec, float, kTm, kTd, BLayout::RowMajor>;
    using tileOCast =
        Tile<Location::Vec, odttype, kTm, kTd, BLayout::RowMajor>;
    using tileRowState =
        Tile<Location::Vec, float, kTm, 1, BLayout::RowMajor, kTm, 1>;

    // ---- GM 视图与迭代器 ----
    using gmQO = global_tensor<qdtype, RowMajor<N1, D>>;
    using gmO = global_tensor<odttype, RowMajor<N1, D>>;
    using itQ = global_iterator<gmQO, tileQ>;
    using itO = global_iterator<gmO, tileOCast>;
    using gmOriT = global_tensor<kvdtype, RowMajor<D, s2>>;
    using itKtOri = global_iterator<gmOriT, tileKRight>;
    using gmKV = global_tensor<kvdtype, RowMajor<s2, D>>;
    using itVOri = global_iterator<gmKV, tileV>;
    using gmCmpT = global_tensor<kvdtype, RowMajor<D, kCmpCap>>;
    using itKtCmp = global_iterator<gmCmpT, tileKRight>;
    using gmCmpRows = global_tensor<kvdtype, RowMajor<kCmpCap, D>>;
    using itVCmp = global_iterator<gmCmpRows, tileV>;
    using gmMask = global_tensor<float, RowMajor<kTm, kTk>>;
    using gmScore = global_tensor<float, RowMajor<kTm, kTk>>;
    using gmProb = global_tensor<qdtype, RowMajor<kTm, kTk>>;
    using gmPV = global_tensor<float, RowMajor<kTm, kTd>>;
};

// =============================================================================
// Section 2 — CMP staging（MGATHER 双链）
//
// 逻辑行 i 的 byte 基址先查表成 row_bases（标量，kCmpCap 个），两条
// MGATHER 链直接从 batch_cmp 产出：
//   a) cmp_rows_buf [TopK, D] 行序 —— BMM2 的 V 源
//   b) cmp_t [D, TopK] 转置序 —— BMM1 的 K^T 源
// 未选中逻辑行取 batch_cmp 第 0 行：cmp mask 已把这些列的 P 压为精确 0
//（exp(-1e30) 下溢），0×正常值 = 0，且 batch_cmp 无 NaN payload，不污染
// exp / P 量化路径。dtype 无关：MGATHER 逐元素 byte offset，FP16/HIF8
// 共用一份代码。
// tile 全局部 → 可安全函数化（见文件头"函数化边界约束"）。
// =============================================================================
template <typename Tiles>
static __attribute__((always_inline)) inline void csa_stage_cmp(
    typename Tiles::kvdtype_alias* batch_cmp,
    const int* cmp_selected,
    int cmp_count,
    typename Tiles::kvdtype_alias* cmp_rows_buf,
    typename Tiles::kvdtype_alias* cmp_t)
{
    using kvdtype = typename Tiles::kvdtype_alias;
    constexpr int D = Tiles::D;
    constexpr int kCmpCap = Tiles::kCmpCap;
    constexpr int kCmpS2 = Tiles::kCmpS2;

    alignas(64) std::uint32_t row_bases[kCmpCap];
    for (int i = 0; i < kCmpCap; ++i) {
        const int row = i < cmp_count ? cmp_selected[i] : 0;
        row_bases[i] =
            static_cast<std::uint32_t>(row) *
            static_cast<std::uint32_t>(D) *
            static_cast<std::uint32_t>(sizeof(kvdtype));
    }
    using CmpGlobal = global_tensor<kvdtype, RowMajor<1, kCmpS2 * D>>;
    using RowBaseGlobal =
        global_tensor<std::uint32_t, RowMajor<1, kCmpCap>>;
    using TransDataTile =
        Tile<Location::Vec, kvdtype, 1, 512, BLayout::RowMajor>;
    using TransOffsetTile =
        Tile<Location::Vec, std::uint32_t, 1, 512, BLayout::RowMajor>;
    static_assert(D * kCmpCap % 512 == 0 &&
                      D * kCmpCap * sizeof(kvdtype) % 32 == 0,
                  "MGATHER cmp staging requires 512-element tiles "
                  "with 32-byte rows");
    CmpGlobal cmp_src(batch_cmp);
    RowBaseGlobal row_base_src(row_bases);
    global_iterator<CmpGlobal, TransDataTile> rows_dst(cmp_rows_buf);
    global_iterator<CmpGlobal, TransDataTile> trans_dst2(cmp_t);
    constexpr int kCmpTiles = D * kCmpCap / 512;
    for (int tile_index = 0; tile_index < kCmpTiles; ++tile_index) {
        TransOffsetTile linear;
        TCI(linear, static_cast<std::uint32_t>(tile_index) * 512u);

        // ---- a) 行序：k = i * D + d ----
        // tile 恰为一整逻辑行（tile 宽 512 == D）：行基址是标量
        // row_bases[tile_index]，用 U32 TEXPANDS 广播 + TCI 列序 × sizeof
        // 合成 offset（省去查表 MGATHER 与 4 个中间 tile）
        TransOffsetTile row_base_a;
        TEXPANDS(row_base_a, row_bases[tile_index]);
        TransOffsetTile dseq;
        TCI(dseq, 0u);
        TMULS(dseq, dseq, static_cast<std::uint32_t>(sizeof(kvdtype)));
        TransOffsetTile byte_off_a;
        TADD(byte_off_a, row_base_a, dseq);
        TransDataTile rows_data;
        MGATHER(rows_data, cmp_src, byte_off_a);
        auto rows_g = rows_dst(0, tile_index);
        TSTORE(rows_g, rows_data);

        // ---- b) 转置序：k' = d * kCmpCap + i ----
        TransOffsetTile dpart_b;
        TDIVS(dpart_b, linear, static_cast<std::uint32_t>(kCmpCap));
        TransOffsetTile dscaled_b;
        TMULS(dscaled_b, dpart_b, static_cast<std::uint32_t>(kCmpCap));
        TransOffsetTile ipart_b;
        TSUB(ipart_b, linear, dscaled_b);
        TransOffsetTile ibytes_b;
        TMULS(ibytes_b, ipart_b, static_cast<std::uint32_t>(4));
        TransOffsetTile row_base_b;
        MGATHER(row_base_b, row_base_src, ibytes_b);
        TransOffsetTile dbytes_b;
        TMULS(dbytes_b, dpart_b,
              static_cast<std::uint32_t>(sizeof(kvdtype)));
        TransOffsetTile byte_off_b;
        TADD(byte_off_b, row_base_b, dbytes_b);
        TransDataTile trans_data;
        MGATHER(trans_data, cmp_src, byte_off_b);
        auto trans_g = trans_dst2(0, tile_index);
        TSTORE(trans_g, trans_data);
    }
}

// =============================================================================
// Section 3 — mask 构造（tile 化）
//
// 行无关（同一 q_token 的全部 head 共享窗口/索引）→ [1,kTk] 值行逐行
// TSTORE kTm 份。链路（U32 位模式域，TSEL 只接受整数 dst——模型
// IsLogicalIntegerTeplDataType；TCMPS 的 packed predicate 不能作数值
// tile 的算术源）：
//   TCI token 序列（U32）→ TCMPS 标量比较出 U8 predicate →
//   TSEL 选 0x0 / bit_cast(-1e30f)（U32 域，位模式原样落 GM）→
//   主段按 float 视图 TLOAD，位解释还原 0/-1e30f（与标量循环逐位一致）。
// tile 全局部 → 可安全函数化。
// =============================================================================
template <typename Tiles>
static __attribute__((always_inline)) inline void csa_build_masks(
    const QsmlaSwaRange& ori_range,
    int ori_blk_begin,
    int ori_blk_count,
    int cmp_blk_count,
    int cmp_count,
    float (*ori_masks)[Tiles::kTm * Tiles::kTk],
    float (*cmp_masks)[Tiles::kTm * Tiles::kTk])
{
    constexpr int kTk = Tiles::kTk;
    constexpr int kTm = Tiles::kTm;
    using MaskIdxTile =
        Tile<Location::Vec, std::uint32_t, 1, kTk, BLayout::RowMajor>;
    using MaskPredTile =
        Tile<Location::Vec, std::uint8_t, 1, kTk, BLayout::RowMajor>;
    using gmMaskRowU32 = global_tensor<std::uint32_t, RowMajor<1, kTk>>;

    const std::uint32_t kMaskNeg =
        std::bit_cast<std::uint32_t>(-1.0e30f);
    MaskIdxTile nbits;
    TEXPANDS(nbits, kMaskNeg);

    for (int j = 0; j < ori_blk_count; ++j) {
        const std::uint32_t token_base =
            static_cast<std::uint32_t>((ori_blk_begin + j) * kTk);
        MaskIdxTile tseq;
        TCI(tseq, token_base);
        MaskPredTile mlo;
        TCMPS<CmpMode::LT>(mlo, tseq,
                           static_cast<std::uint32_t>(ori_range.begin));
        MaskPredTile mhi;
        TCMPS<CmpMode::GE>(mhi, tseq,
                           static_cast<std::uint32_t>(ori_range.end));
        MaskIdxTile mval;
        TEXPANDS(mval, 0u);
        TSEL(mval, mlo, nbits);
        TSEL(mval, mhi, nbits);
        for (int r = 0; r < kTm; ++r) {
            gmMaskRowU32 gMaskRow(
                reinterpret_cast<std::uint32_t*>(ori_masks[j] + r * kTk));
            TSTORE(gMaskRow, mval);
        }
    }
    for (int j = 0; j < cmp_blk_count; ++j) {
        const std::uint32_t valid_cols =
            static_cast<std::uint32_t>(cmp_count - j * kTk);
        MaskIdxTile tcol;
        TCI(tcol, 0u);
        MaskPredTile mhi;
        TCMPS<CmpMode::GE>(mhi, tcol, valid_cols);
        MaskIdxTile mval;
        TEXPANDS(mval, 0u);
        TSEL(mval, mhi, nbits);
        for (int r = 0; r < kTm; ++r) {
            gmMaskRowU32 gMaskRow(
                reinterpret_cast<std::uint32_t*>(cmp_masks[j] + r * kTk));
            TSTORE(gMaskRow, mval);
        }
    }
}

// =============================================================================
// Section 4 — MM1 score 块（Q @ K^T 的 D 维分段累加 + scale + mask）
//
// Db 个 D 切片累加进函数局部的 tW，乘 score_scale、加 mask 后落 GM
// score scratch（调用者 TLOAD 续做 softmax / P 构造）。Pass1/Pass2 的
// ORI/CMP 四处共用；tile 全局部（tW 不跨函数），跨块状态走 GM scratch。
// =============================================================================
template <typename Tiles, typename QIter, typename KIter>
static __attribute__((always_inline)) inline void csa_mm1_score(
    QIter& gIterQ,
    int m_block,
    KIter& k_iter,
    int k_blk,
    float score_scale,
    float* mask_row,
    typename Tiles::gmScore& gScore)
{
    using tileW = typename Tiles::tileW;
    tileW tW;
    TEXPANDS(tW, 0.0f);
#pragma clang loop unroll_count(4)
    for (int dd = 0; dd < Tiles::kDb; ++dd) {
        typename Tiles::tileQ tQ;
        auto gQ = gIterQ(m_block, dd);
        TLOAD_CUBE(tQ, gQ);
        typename Tiles::tileKRight tK;
        auto gK = k_iter(dd, k_blk);
        TLOAD_CUBE(tK, gK);
        typename Tiles::tileScoreCube tScoreCube;
        TMATMUL(tScoreCube, tQ, tK);
        TSTORE_CUBE(gScore, tScoreCube);
        tileW tWPartial;
        TLOAD(tWPartial, gScore);
        TADD(tW, tW, tWPartial);
    }
    TMULS(tW, tW, score_scale);
    typename Tiles::tileMask tMask;
    typename Tiles::gmMask gMaskBuf(mask_row);
    auto gMask = gMaskBuf;
    TLOAD(tMask, gMask);
    TADD(tW, tW, tMask);
    TSTORE(gScore, tW);
}

// =============================================================================
// Section 5 — Pass2 的 P·V 块
//
// 消费已发布的 P（gProb，qdtype）与一个 V tile 视图，BMM2 + HIF8 的
// PV×kv_descale 后落 GM pv scratch（调用者 TLOAD + TADD tO）。
// Pass2 的 ORI/CMP 两处共用；tile 全局部。
// =============================================================================
template <typename Tiles>
static __attribute__((always_inline)) inline void csa_pass2_pv(
    typename Tiles::gmProb& gProb,
    auto gV,
    float kv_descale,
    typename Tiles::gmPV& gPV)
{
    typename Tiles::tileWLeft tWLeft;
    TLOAD_CUBE(tWLeft, gProb);
    typename Tiles::tileV tV;
    TLOAD_CUBE(tV, gV);
    typename Tiles::tilePVCube tPVCube;
    TMATMUL(tPVCube, tWLeft, tV);
    TSTORE_CUBE(gPV, tPVCube);
    typename Tiles::tileO tPV;
    TLOAD(tPV, gPV);
    if constexpr (Tiles::kUseHif8Probability) {
        TMULS(tPV, tPV, kv_descale);
    }
    TSTORE(gPV, tPV);
}

// =============================================================================
// Section 6 — 主 kernel
//
// 结构：batch → q_token（CMP staging + mask 构造）→ m_block（Pass1 归约
// (m,l) + Pass2 逐 D 块累加 O）。online-softmax 状态更新（tMax/tSum）
// 与 Pass2 的 P 构造（tMax/tInvSum 依赖）保持内联——见文件头
// "函数化边界约束"。
// =============================================================================
template <typename qdtype, typename kvdtype, typename odttype,
          typename Config, typename ModeConfig>
void quant_sparse_flash_mla_csa_1pe_pto(
    odttype* out_ptr,
    qdtype* q_ptr,
    kvdtype* ori_kv_ptr,
    // qli-style pre-transposed ORI KV view: [D, S2] row-major per batch
    // (BMM1 K^T B-tile source). The transpose is provided by the input
    // layout instead of being performed in-kernel (TTRANS retired,
    // TLOAD_CUBE is ND-only — TileOP-API issue #17); BMM2's V keeps the
    // natural [S2, D] ori_kv_ptr layout.
    kvdtype* ori_kv_t_ptr,
    kvdtype* cmp_kv_ptr,
    const int* cmp_sparse_indices,
    const int* cmp_topk_length,
    float softmax_scale,
    float q_descale,
    float ori_kv_descale,
    float cmp_kv_descale,
    int cmp_ratio,
    int ori_win_left,
    int ori_win_right)
{
    using Tiles = CsaTiles<qdtype, kvdtype, odttype, Config, ModeConfig>;
    using tileW = typename Tiles::tileW;
    using tileRowState = typename Tiles::tileRowState;
    using tileP = typename Tiles::tileP;
    using tileO = typename Tiles::tileO;
    using tileOCast = typename Tiles::tileOCast;

    constexpr int kB = Tiles::kB;
    constexpr int s1 = Tiles::s1;
    constexpr int s2 = Tiles::s2;
    constexpr int N1 = Tiles::N1;
    constexpr int D = Tiles::D;
    constexpr int kTm = Tiles::kTm;
    constexpr int kTk = Tiles::kTk;
    constexpr int kTd = Tiles::kTd;
    constexpr int kDb = Tiles::kDb;
    constexpr int kCmpS2 = Tiles::kCmpS2;
    constexpr int kCmpTopK = Tiles::kCmpTopK;
    constexpr int kOriBlkMax = Tiles::kOriBlkMax;
    constexpr int kMBlockCount = Tiles::kMBlockCount;
    constexpr bool kUseHif8Probability = Tiles::kUseHif8Probability;
    constexpr float kHif8ProbabilityScale = Tiles::kHif8ProbabilityScale;

    // score 反量化缩放（FP16 下 descale 均为 1，数值不变）
    const float ori_score_scale = softmax_scale * q_descale * ori_kv_descale;
    const float cmp_score_scale = softmax_scale * q_descale * cmp_kv_descale;

    // ---- 栈 scratch ----
    // 注：ORI 预转置缓冲不存在——K^T 布局由输入 ori_kv_t_ptr 直接提供
    //（qli 式预转置，见签名注释）
    alignas(64) kvdtype cmp_rows_buf[Tiles::kCmpCap * D];
    alignas(64) kvdtype cmp_t[D * Tiles::kCmpCap];
    alignas(64) float ori_masks[kOriBlkMax][kTm * kTk];      // 16KB
    alignas(64) float cmp_masks[Tiles::kCmpBlkMax][kTm * kTk];  // 6KB
    alignas(64) float score_scratch[kTm * kTk];
    alignas(64) qdtype prob_scratch[kTm * kTk];
    alignas(64) float pv_scratch[kTm * kTd];
    int cmp_selected[kCmpTopK];

    typename Tiles::gmScore gScore(score_scratch);
    typename Tiles::gmProb gProb(prob_scratch);
    typename Tiles::gmPV gPV(pv_scratch);

    // ================= batch 循环 =================
    for (int b = 0; b < kB; ++b) {
        kvdtype* batch_ori =
            ori_kv_ptr + static_cast<std::size_t>(b) * s2 * D;
        // 每 batch 的预转置 K^T 视图（[D, S2] 行主序，D*S2 元素）
        kvdtype* batch_ori_t =
            ori_kv_t_ptr + static_cast<std::size_t>(b) * s2 * D;
        kvdtype* batch_cmp =
            cmp_kv_ptr + static_cast<std::size_t>(b) * kCmpS2 * D;
        qdtype* batch_q =
            q_ptr + static_cast<std::size_t>(b) * s1 * N1 * D;
        odttype* batch_out =
            out_ptr + static_cast<std::size_t>(b) * s1 * N1 * D;

        // ================= q_token 循环 =================
        for (int q_token = 0; q_token < s1; ++q_token) {
            // ---- CMP 索引收集（标量）----
            const int cmp_valid_end = qsmla_csa_cmp_valid_end(
                kCmpS2, s1, q_token, cmp_ratio);
            int cmp_count = 0;
            {
                int candidate_count = kCmpTopK;
                if (cmp_topk_length != nullptr) {
                    candidate_count =
                        cmp_topk_length[static_cast<std::size_t>(b) * s1 +
                                        q_token];
                }
                candidate_count = qsmla_sparse_clamp(
                    candidate_count, 0, kCmpTopK);
                cmp_count = qsmla_sparse_collect_indices(
                    cmp_selected, kCmpTopK,
                    cmp_sparse_indices +
                        static_cast<std::size_t>(
                            (b * s1 + q_token)) * kCmpTopK,
                    candidate_count, kCmpS2, cmp_valid_end);
            }
            csa_stage_cmp<Tiles>(batch_cmp, cmp_selected, cmp_count,
                                 cmp_rows_buf, cmp_t);

            // ---- ORI 窗口与块范围（块对齐裁剪 + 边缘 mask，tadd 语义）----
            const QsmlaSwaRange ori_range = qsmla_swa_range(
                s2, s1, q_token, ori_win_left, ori_win_right);
            const QsmlaSwaRange ori_blocks =
                qsmla_swa_block_range(ori_range, kTk);
            const int ori_blk_begin = ori_blocks.begin;
            const int ori_blk_count = ori_blocks.end - ori_blocks.begin;
            const int cmp_blk_count = (cmp_count + kTk - 1) / kTk;
            csa_build_masks<Tiles>(ori_range, ori_blk_begin, ori_blk_count,
                                   cmp_blk_count, cmp_count, ori_masks,
                                   cmp_masks);

            // Q / O / KV 迭代器（本 q_token 的 [N1, D] 视图）
            typename Tiles::itQ gIterQ(
                batch_q + static_cast<std::size_t>(q_token) * N1 * D);
            typename Tiles::itO gIterO(
                batch_out + static_cast<std::size_t>(q_token) * N1 * D);
            typename Tiles::itKtOri itOriT(batch_ori_t);
            typename Tiles::itVOri itVOriGm(batch_ori);
            typename Tiles::itKtCmp itCmpT(cmp_t);
            typename Tiles::itVCmp itVCmpRows(cmp_rows_buf);

            // ================= m_block（head 块）循环 =================
            for (int m_block = 0; m_block < kMBlockCount; ++m_block) {

                // ============ Pass 1：online softmax 归约 (m, l) ============
                tileRowState tMax;
                tileRowState tSum;
                TEXPANDS(tMax, -1e30f);
                TEXPANDS(tSum, 0.0f);

                // ---- Pass 1 / ORI 源 ----
                for (int j = 0; j < ori_blk_count; ++j) {
                    csa_mm1_score<Tiles>(gIterQ, m_block, itOriT,
                                         ori_blk_begin + j, ori_score_scale,
                                         ori_masks[j], gScore);
                    tileW tW;
                    TLOAD(tW, gScore);
                    tileRowState tLocalMax;
                    tileRowState tNewMax;
                    TROWMAX(tLocalMax, tW);
                    TMAX(tNewMax, tMax, tLocalMax);
                    tileRowState tScale;
                    TSUB(tScale, tMax, tNewMax);
                    TEXP(tScale, tScale);
                    tileRowState tScaledOldSum;
                    TMUL(tScaledOldSum, tSum, tScale);
                    TROWEXPANDSUB(tW, tW, tNewMax);
                    TEXP(tW, tW);
                    tileRowState tLocalSum;
                    TROWSUM(tLocalSum, tW);
                    TADD(tSum, tScaledOldSum, tLocalSum);
                    tMax = tNewMax;
                }

                // ---- Pass 1 / CMP 源 ----
                for (int j = 0; j < cmp_blk_count; ++j) {
                    csa_mm1_score<Tiles>(gIterQ, m_block, itCmpT, j,
                                         cmp_score_scale, cmp_masks[j],
                                         gScore);
                    tileW tW;
                    TLOAD(tW, gScore);
                    tileRowState tLocalMax;
                    tileRowState tNewMax;
                    TROWMAX(tLocalMax, tW);
                    TMAX(tNewMax, tMax, tLocalMax);
                    tileRowState tScale;
                    TSUB(tScale, tMax, tNewMax);
                    TEXP(tScale, tScale);
                    tileRowState tScaledOldSum;
                    TMUL(tScaledOldSum, tSum, tScale);
                    TROWEXPANDSUB(tW, tW, tNewMax);
                    TEXP(tW, tW);
                    tileRowState tLocalSum;
                    TROWSUM(tLocalSum, tW);
                    TADD(tSum, tScaledOldSum, tLocalSum);
                    tMax = tNewMax;
                }

                // ============ Pass 2：P 计算与 O = Σ P·V 累加 ============
                tileRowState tInvSum;
                TRECIP(tInvSum, tSum);

                // dd 外层（tO 寿命单 D 块，tadd 结构）
                for (int dd = 0; dd < kDb; ++dd) {
                    tileO tO;
                    TEXPANDS(tO, 0.0f);

                    // ---- Pass 2 / ORI 源 ----
                    for (int j = 0; j < ori_blk_count; ++j) {
                        csa_mm1_score<Tiles>(gIterQ, m_block, itOriT,
                                             ori_blk_begin + j,
                                             ori_score_scale, ori_masks[j],
                                             gScore);
                        tileW tW;
                        TLOAD(tW, gScore);
                        TROWEXPANDSUB(tW, tW, tMax);
                        TEXP(tW, tW);
                        TROWEXPANDMUL(tW, tW, tInvSum);
                        if constexpr (kUseHif8Probability) {
                            // 官方 QSMLA 契约：P×16 量化 HIF8 参与 BMM2，
                            // 输出统一 ×1/16 还原；online l 保持未量化
                            TMULS(tW, tW, kHif8ProbabilityScale);
                        }
                        tileP tP;
                        TCVT(tP, tW);
                        TSTORE(gProb, tP);
                        csa_pass2_pv<Tiles>(gProb,
                                            itVOriGm(ori_blk_begin + j, dd),
                                            ori_kv_descale, gPV);
                        tileO tPV;
                        TLOAD(tPV, gPV);
                        TADD(tO, tO, tPV);
                    }

                    // ---- Pass 2 / CMP 源 ----
                    for (int j = 0; j < cmp_blk_count; ++j) {
                        csa_mm1_score<Tiles>(gIterQ, m_block, itCmpT, j,
                                             cmp_score_scale, cmp_masks[j],
                                             gScore);
                        tileW tW;
                        TLOAD(tW, gScore);
                        TROWEXPANDSUB(tW, tW, tMax);
                        TEXP(tW, tW);
                        TROWEXPANDMUL(tW, tW, tInvSum);
                        if constexpr (kUseHif8Probability) {
                            TMULS(tW, tW, kHif8ProbabilityScale);
                        }
                        tileP tP;
                        TCVT(tP, tW);
                        TSTORE(gProb, tP);
                        csa_pass2_pv<Tiles>(gProb, itVCmpRows(j, dd),
                                            cmp_kv_descale, gPV);
                        tileO tPV;
                        TLOAD(tPV, gPV);
                        TADD(tO, tO, tPV);
                    }

                    // 写回本 D 块输出（HIF8：还原 P 的 ×16 量化系数）
                    if constexpr (kUseHif8Probability) {
                        TMULS(tO, tO, 1.0f / kHif8ProbabilityScale);
                    }
                    tileOCast tOCast;
                    TCVT(tOCast, tO);
                    auto gO = gIterO(m_block, dd);
                    TSTORE(gO, tOCast);
                }
            }
        }
    }
}

#endif // QUANT_SPARSE_FLASH_MLA_CSA_1PE_HPP
