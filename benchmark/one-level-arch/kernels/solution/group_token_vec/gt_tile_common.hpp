#ifndef GT_TILE_COMMON_HPP
#define GT_TILE_COMMON_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>

// ============================================================================
// gt_tile_common — group_token_vec 系 (单PE / mt / mt_dyn) 共享 tile 基础设施
//
// 全部契约经 tile_probe 探针工程在 gfrun 上实证 (副本 tilework, 阶段0):
//   [C1] TCMPS<TSEL>TROWSUM/TROWMIN/TROWMAX/TREMS/TSHRS/TDIVS 在多行 u32
//        tile (16×16 / 4×16 / 1×32) 可用 (旧注释 "TCMPS u32 被汇编器拒绝"
//        已过时, qli #177 与本轮探针双重实证)。
//   [C2] MSCATTER_ADD: 汇编器助记符表缺 "MSCATTER.ADD" (LinxV5AsmParser
//        parseTileOPTMA), 须数字编码 BSTART.TLSU 21 (模型 TMA_MSCATTER_ADD=21,
//        TMAEngine::ExecuteGMReductionValue 已实现)。index tile 为相对 base
//        的**字节位移** (GMByteDisplacement; address%elementBytes 须 0)。
//        tile 内重复下标按 row-major 元素序串行 RMW → 计数稳定序。
//   [C3] MGATHER_ADD (助记符可用): old 值发布到 dst tile, 重复下标同上确定
//        序 → 可替代标量 "idx = cnt[section]++" 原子写指针。
//   [C4] [N×1] 列 tile 禁用于标量 TEPL (TSUBS/TADDS/TSHLS...) 与原子族:
//        Cols=1 → lb0/lb2=1 被 dim-opt 省略 → 模型描述符契约断言/非法。
//        归约 ([32×1,vN×1]) 输出只可 TSTORE; 需要行向量时经 GM 往返
//        (TSTORE [N×1] → TLOAD [1×N]) 转形状。
//   [C5] TCI 仅 ValidRow==1 ([1×N] 行 ramp); TTRANS 已退休 (0.58.5)。
//   [C6] 动态 ValidRow (运行时 vr) 链会按物理行泄漏 lane (TLOAD/TEXPANDS/
//        MSCATTER 全链, 探针 P12 负面实证) → 计数/散射链一律静态 valid;
//        dyn 尾块走标量兜底。
//   [C7] TCMPS 结果为 packed predicate (U8 存储), 不可参与算术; 用 TSEL
//        物化 0/1 (dst 预填假分支值, TSEL 假分支 = dst 旧值)。
//   [C8] 行归约目的 tile 物理单列且物理 32 行保 128B 下限:
//        Tile<Vec,u32,32,1,RowMajor,N,1> (与 qli t1 / dyn Phase3a 同款)。
//   [C9] 相邻非零标量 store 会被后端合并为 16B tile store, gfrun 对 bss
//        该类写不可靠 (探针 P13 首败根因; mega_moe driver 同款规避) →
//        小数组 (<128B tile 下限) 初始化一律 volatile 逐元素写。
//
// 第二批契约 (去原子化 gfsim 兼容链, 探针 P17-P24 双门禁实证 ——
// gfrun R2=0 + gfsim Total Cycles; 背景: TimingSim TLSU 无 GM 原子族完成
// 路径 (Core.cpp:1433-1446), MSCATTER_ADD/MGATHER_ADD 仅 gfrun 可用):
//   [C10] 计数链 (直方图/等值计数): TCMPS<EQ>+TSEL(物化0/1)+TCOLSUM
//         ([N×M]→[1×M], dst 列几何须同源)+TROWSUM(→[32×1,v1×1])+TSTORE
//         → 标量读回累加。bin 循环仅覆盖合法值域 → 无需越界守卫。
//   [C11] 成对比较 rank (原子写指针等价物, 稳定序 = 标量前缀计数):
//         TROWEXPAND(单列源→右广播 Mc[i][j]=v[i]) + TCOLEXPAND(单行源→
//         下广播 Mr[i][j]=v[j]; 源物理 Cols 必须 == dst 物理 Cols) +
//         TCMP<EQ> + TSEL(mat, eq, TTRI) (TTRI=下三角含对角) + TROWSUM
//         → rankIncl = 1+#{j<i: v_j==v_i}; -1 修正折入 base 或 TSUBS。
//         base 查表 = 平 MGATHER (支持族); 散射 = 平 MSCATTER + TCI ramp。
//         跨块进位 = 每块后 per-bin 计数链 [C10] + 标量 volatile RMW。
//   [C12] gfsim 兼容指令面板 (双门禁全过): TLOAD/TSTORE/TCVT/TEXPANDS/TCI/
//         TADD(S)/TSUB(S)/TMULS/TDIVS/TREMS/TSHLS/TSHRS/TAND/TCMP(S)/TSEL/
//         TROWSUM/TCOLSUM/TROWMIN/TROWMAX/TTRI/TROWEXPAND/TCOLEXPAND/
//         平 MGATHER/平 MSCATTER/TGEMV_MX 系。禁用 (gfsim 卡死):
//         MGATHER_ADD/MSCATTER_ADD 等 TLSU 原子族 (Function 8-27)。
//   [C13] 成本实测 (tile_cost 微基准, 8192 元素/128 bin): 全 tile 计数链
//         382K cycles vs TLOAD+标量直方图 244K (1.57×); 但 Phase2/3 元素
//         循环 tile 化节省 >300K → 算子总 cycles 仍低于标量基线 1.015M。
//         gfrun wall: 4096 链 ≈ 6s。
// ============================================================================

namespace gt_tile {

using namespace pto;   // CmpMode 可见性 (与 qli 同款)

// ---- 共享 tile 形状 (静态 valid; [C4]/[C6]/[C8]) ----
using T16x16  = Tile<Location::Vec, uint32_t, 16, 16, BLayout::RowMajor>;
using T4x16   = Tile<Location::Vec, uint32_t, 4, 16, BLayout::RowMajor>;
using T4x4    = Tile<Location::Vec, uint32_t, 4, 4, BLayout::RowMajor>;
using T32x32  = Tile<Location::Vec, uint32_t, 32, 32, BLayout::RowMajor>;
using TRed16  = Tile<Location::Vec, uint32_t, 32, 1, BLayout::RowMajor, 16, 1>;
using TRed4   = Tile<Location::Vec, uint32_t, 32, 1, BLayout::RowMajor, 4, 1>;
using TRed32  = Tile<Location::Vec, uint32_t, 32, 1, BLayout::RowMajor>;
using TRow16  = Tile<Location::Vec, uint32_t, 1, 32, BLayout::RowMajor, 1, 16>;
using TRow4   = Tile<Location::Vec, uint32_t, 1, 32, BLayout::RowMajor, 1, 4>;
using T1x32   = Tile<Location::Vec, uint32_t, 1, 32, BLayout::RowMajor>;
using TCol16  = Tile<Location::Vec, uint32_t, 1, 16, BLayout::RowMajor>;
using TCol4   = Tile<Location::Vec, uint32_t, 1, 4, BLayout::RowMajor>;
using TSum1   = Tile<Location::Vec, uint32_t, 32, 1, BLayout::RowMajor, 1, 1>;
using T16x8   = Tile<Location::Vec, uint32_t, 16, 8, BLayout::RowMajor>;

// ---- GM 视图 ----
using G16x1  = global_tensor<uint32_t, RowMajor<16, 1>>;
using G32x1  = global_tensor<uint32_t, RowMajor<32, 1>>;
using G4x1   = global_tensor<uint32_t, RowMajor<4, 1>>;
using G1x16  = global_tensor<uint32_t, RowMajor<1, 16>>;
using G1x4   = global_tensor<uint32_t, RowMajor<1, 4>>;
using G1x32  = global_tensor<uint32_t, RowMajor<1, 32>>;
using G1x1   = global_tensor<uint32_t, RowMajor<1, 1>>;
using G16x8  = global_tensor<uint32_t, RowMajor<16, 8>>;
using GFlat  = global_tensor<uint32_t, RowMajor<-1, -1>>;

// ---- MSCATTER_ADD 数字编码入口 ([C2]) —— 仅 gfrun 可用 (TimingSim TLSU 无
//      原子族完成路径 [C12], gfsim 卡死)。kernel 主线已改用 [C10] 计数链 /
//      [C11] 成对 rank; 本 wrapper 保留供探针与未来模型补齐后回切。----
template <typename IndexTile, typename ValueTile>
static inline void mscatter_add(uint64_t base, IndexTile &idxBytes,
                                 ValueTile &ones)
{
    static_assert(IndexTile::Rows == ValueTile::Rows &&
                      IndexTile::Cols == ValueTile::Cols,
                  "index/value tiles must match");
    asm volatile(
        "BSTART.TLSU 21, %D[DataType]\n"
        "B.DIM zero, %c[VCOL], ->lb0\n"
        "B.DIM zero, %c[VROW], ->lb1\n"
        "B.DIM zero, %c[Col], ->lb2\n"
        "B.IOT %[Idx], %[Val], mask=1111, last\n"
        "B.IOR [%[Base]], []\n"
        :
        : [Idx] "Tr"(idxBytes.data()),
          [Val] "Tr"(ones.data()),
          [Base] "r"(base),
          [DataType] "i"(type_traits<typename ValueTile::DType>::TypeCode),
          [VCOL] "i"(IndexTile::ValidCol),
          [VROW] "i"(IndexTile::ValidRow),
          [Col] "i"(IndexTile::Cols)
        : "memory");
}

}  // namespace gt_tile

#endif  // GT_TILE_COMMON_HPP
