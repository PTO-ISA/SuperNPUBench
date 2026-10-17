#ifndef DISPATCH_TILE_COMMON_HPP
#define DISPATCH_TILE_COMMON_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>

// ============================================================================
// dispatch_tile_common — moe_dispatch 系 (v2 / mt / mt_dyn) 共享 tile 基础设施
//
// 契约与 group_token_vec/gt_tile_common.hpp 同源 (tile_probe 探针 gfrun 实证,
// 摘要):
//   [C2] MSCATTER_ADD: 汇编器助记符表缺 "MSCATTER.ADD", 须数字编码
//        BSTART.TLSU 21 (模型 TMA_MSCATTER_ADD=21 已实现); index tile 为
//        相对 base 的元素下标 (PTO v0.58.6); tile 内重复下标按 row-major 顺序串行 RMW。
//   [C3] MGATHER_ADD: old 值发布 (原子写指针 tile 化), 重复下标确定序。
//   [C4] 原子族/标量 TEPL 一律 [1×N] 行 tile (N>=2, lb0=ValidCol
//        必须存在 —— dim-opt 省略值 1 的 B.DIM); [N×1] 归约输出只可 TSTORE。
//   [C6] 动态 ValidRow 链泄漏 lane → 计数/散射链一律静态 valid, 尾部标量。
//   [C7] TCMPS 结果 predicate 不可算术, TSEL 物化 (dst 预填假分支值)。
//   [C9] 相邻非零标量 store 合并 16B → gfrun 写丢失 → 小数组初始化 volatile。
//
// 第二批契约 (去原子化 gfsim 兼容链, 与 gt_tile_common.hpp 同源, 探针
// P17-P24 双门禁实证 —— gfrun R2=0 + gfsim Total Cycles; 背景: TimingSim
// TLSU 无 GM 原子族完成路径 (Core.cpp TlsuBlockEntry default 分支准入后
// 无执行/退休路径), MSCATTER_ADD/MGATHER_ADD 仅 gfrun 可用):
//   [C10] 计数链 (直方图/等值计数): TCMPS<EQ>+TSEL(物化0/1)+TROWSUM
//         (→[32×1,v1×1])+TSTORE→标量读回累加。bin 循环仅覆盖合法值域
//         → 无需越界守卫; dyn 末块 lane 越段用 TCI ramp+TCMPS<GE> 守卫
//         (或一次性 TSEL 到 -1 哨兵, 谓词 tile 不得 loop-carried)。
//   [C11] 成对比较 rank (原子写指针等价物, 稳定序 = 标量前缀计数):
//         同一 GM 数据双视图 TLOAD ([1×32] 行 + [32×1] 列) →
//         TROWEXPAND(单列源→右广播 Mc[i][j]=v[i]) + TCOLEXPAND(单行源→
//         下广播 Mr[i][j]=v[j]; 源物理 Cols 必须 == dst 物理 Cols) +
//         TCMP<EQ> + TSEL(mat, eq, TTRI) (TTRI=下三角含对角) + TROWSUM
//         → rankIncl = 1+#{j<i: v_j==v_i}; -1 修正折入 TSUBS。
//         base/carry 查表 = 平 MGATHER (支持族); 跨块进位 = 每块后
//         per-bin 计数链 [C10] + 标量 volatile RMW。
//         越段/非法 lane 的 rank 为垃圾值但被 TTRI 掩蔽出不有效行的
//         前缀计数, 且输出侧按 expertIds 守卫跳过, 不消费。
//   [C12] gfsim 兼容指令面板 (双门禁全过): TLOAD/TSTORE/TCVT/TEXPANDS/TCI/
//         TADD(S)/TSUB(S)/TMULS/TDIVS/TREMS/TSHLS/TSHRS/TAND/TCMP(S)/TSEL/
//         TROWSUM/TCOLSUM/TROWMIN/TROWMAX/TTRI/TROWEXPAND/TCOLEXPAND/
//         平 MGATHER/平 MSCATTER/TGEMV_MX 系。禁用 (gfsim 卡死):
//         MGATHER_ADD/MSCATTER_ADD 等 TLSU 原子族 (Function 8-27)。
// ============================================================================

namespace dispatch_tile {

using namespace pto;   // CmpMode 可见性

// ---- 共享 tile 形状 (静态 valid, [C4]/[C6]) ----
using T1x32   = Tile<Location::Vec, uint32_t, 1, 32, BLayout::RowMajor>;
using T1x32v8 = Tile<Location::Vec, uint32_t, 1, 32, BLayout::RowMajor, 1, 8>;
using TI1x32  = Tile<Location::Vec, int32_t, 1, 32, BLayout::RowMajor>;
using TI1x32v8 = Tile<Location::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 8>;
// 成对比较 rank ([C11]) 与计数链 ([C10]) 形状 (int32 payload)
using TI32x1  = Tile<Location::Vec, int32_t, 32, 1, BLayout::RowMajor>;
using TI32x32 = Tile<Location::Vec, int32_t, 32, 32, BLayout::RowMajor>;
using TSum1I  = Tile<Location::Vec, int32_t, 32, 1, BLayout::RowMajor, 1, 1>;
using G1x32   = global_tensor<uint32_t, RowMajor<1, 32>>;
using GI1x32  = global_tensor<int32_t, RowMajor<1, 32>>;
using GI32x1  = global_tensor<int32_t, RowMajor<32, 1>>;
using GI1x1   = global_tensor<int32_t, RowMajor<1, 1>>;
using GFlat   = global_tensor<uint32_t, RowMajor<-1, -1>>;

// ---- MSCATTER_ADD 数字编码入口 ([C2]) —— 仅 gfrun 可用 (TimingSim TLSU
//      无原子族完成路径 [C12], gfsim 卡死)。kernel 主线已改用 [C10] 计数
//      链 / [C11] 成对 rank; 本 wrapper 保留供探针与未来模型补齐后回切。----
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

}  // namespace dispatch_tile

#endif  // DISPATCH_TILE_COMMON_HPP
