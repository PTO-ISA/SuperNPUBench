#ifndef REDUCEMAXCOLVEC_KERNEL_HPP
#define REDUCEMAXCOLVEC_KERNEL_HPP

#include <common/pto_tileop.hpp>
#include <pto/pto-inst.hpp>
#include <cstdint>
#include <cstdio>

using namespace pto;

/**
 * @brief 非对齐列最大值（120×8 特化，Tile ISA 实现）
 *
 * 算法（以 gIM=120, gIN=8 为例）：
 *   1. Reshape：将 120×8 内存重新解释为 15×64（gIM/8 × gIN*8）
 *   2. 第一次 TCOLMAX：(15,64) → (1,64)，64 个 lane 满载
 *   3. TMOV：将 (1,64) 重解释为 (8,8) RowMajor（相同数据，不同形状）
 *   4. 第二次 TCOLMAX：(8,8) → (1,8)，等价于对 stride=8 的 8 组元素分别取最大值
 *   5. TSTORE：写回 1×8 结果
 *
 * @tparam dtype   数据类型
 * @tparam gIM     输入行数（如 120）
 * @tparam gIN     输入列数（如 8）
 * @tparam tM      tile 物理行数
 * @tparam tN      tile 物理列数
 * @tparam tM_VLD  tile 有效行数
 */
template<typename dtype, int gIM, int gIN, int tM, int tN, int tM_VLD>
void reducemax_col_rand(
    dtype *in_ptr,
    dtype *out_ptr
)
{
    // GM 视角：将 gIM×gIN 重新解释为 (gIM/8)×(gIN*8)
    using gm_shapeIn = global_tensor<dtype, RowMajor<gIM / 8, gIN * 8>>;
    using gm_shapeOut = global_tensor<dtype, RowMajor<1, gIN>>;

    // 数据 tile：(tM/8) × (tN*8)，如 15×64
    using tile_shapeData = Tile<Location::Vec, dtype, tM / 8, tN * 8,
                              BLayout::RowMajor, tM_VLD / 8, tN * 8>;
    // 中间结果 tile：1 × (tN*8)，如 1×64
    using tile_shapeTmp = Tile<Location::Vec, dtype, 1, tN * 8, BLayout::RowMajor>;
    // 重解释 tile：将 1×64 视为 8×8 RowMajor
    using tile_shapeReshaped = Tile<Location::Vec, dtype, tN, tN * 8, BLayout::RowMajor>;
    // 最终结果 tile：1 × (tN*8)，有效区域 1×tN，如 1×8
    using tile_shapeMax = Tile<Location::Vec, dtype, 1, tN * 8,
                             BLayout::RowMajor, 1, tN>;

    gm_shapeIn inGm(in_ptr);
    gm_shapeOut outGm(out_ptr);

    tile_shapeData dataTile;
    tile_shapeTmp TmpTile;
    tile_shapeReshaped reshapedTile;
    tile_shapeMax MaxTile;
    tile_shapeMax oldMaxTile;

    using itIn = global_iterator<gm_shapeIn, tile_shapeData>;
    using itOut = global_iterator<gm_shapeOut, tile_shapeMax>;

    itIn gIIter(in_ptr);
    itOut gOIter(out_ptr);

    auto gO = gOIter(0, 0);
    TEXPANDS(oldMaxTile, static_cast<dtype>(0));

    auto gI = gIIter(0, 0);
    TLOAD(dataTile, gI);

    // Step 1: 第一次列归约 (tM/8 × tN*8) → (1 × tN*8)
    TCOLMAX(TmpTile, dataTile);

    // Step 2: 重解释形状 (1 × tN*8) → (tN × tN*8)，即 1×64 → 8×8
    TRESHAPE(reshapedTile, TmpTile);

    // Step 3: 第二次列归约 (tN × tN*8) → (1 × tN*8)，即 8×8 → 1×8
    TCOLMAX(MaxTile, reshapedTile);

    // Step 4: 与旧值取最大值
    TMAX(MaxTile, MaxTile, oldMaxTile);

    // Step 5: 写回
    TSTORE(gO, MaxTile);
}

#endif
