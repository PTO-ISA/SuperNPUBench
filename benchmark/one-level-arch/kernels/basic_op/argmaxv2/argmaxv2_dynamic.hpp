#pragma once

// ============================================================================
// ArgMaxV2 四 PE SPMD 动态版包装（运行期 tiling）。
//
// 静态版入口 argmaxv2<DType, IdType, Axis, Dims...>（argmaxv2.hpp）的
// 动态对应实现，风格对齐 solution/normalization 的 static/dynamic 拆分
// （rms_norm / group_norm_grad）：
//
//   - 静态版：shape 为编译期模板参数包，编译期折叠（ArgMaxNdShape），
//     整除约束走 static_assert（不满足时行路径 PE0 单跑兜底）；
//   - 动态版：host 侧完成同样的三段折叠后经 tiling 传入运行期
//     {g_outer, g_axis, g_inner}，kernel 侧 fail-closed 校验后执行，
//     任意尾数 / 不整除形状统一走 ceil 分区（normalization 同风格）。
//
// tiling（ArgMaxV2TilingData，host 侧构造）：
//   g_outer  非归约前缀积 prod(d0..d(axis-1))
//   g_axis   归约维长度 d(axis)
//   g_inner  非归约后缀积 prod(d(axis+1)..d(n-1))
//   tile_m   行方向有效块宽（strip）；<= 0 → 物理默认满块
//   tile_n   列方向有效块宽（strip）；<= 0 → 物理默认满块
//   axis_last  1 = 归约轴为最后一维（行归约，含 1D/任意 rank 尾轴）；
//              0 = 归约轴为中间/首维（列归约）
//
// PE 分区（相对静态版的泛化，均衡分区：地板除 + 余数给前几个 PE）：
//   - 行归约：base = g_outer / PeCount，前 g_outer % PeCount 个 PE 各多 1
//     行；g_outer >= PeCount 时全部 PE 参与计算（如 g_outer=5、4PE →
//     2/1/1/1，替代 ceil 的 2/2/1/0 空转形态）；g_outer < PeCount 时退化
//     为前 g_outer 个 PE 各 1 行。每 PE 独占连续行区间，无写冲突。替代
//     静态版的「整除才 SPMD、否则 PE0 单跑」兜底。
//   - 列归约：按列块 ceil 分区（blocks = ceil(g_inner / tile_n)，
//     blocks_per_pe = ceil(blocks / PeCount)），窗口自动块对齐；
//     每列归约线独立，g_inner 无整除约束（静态版要求
//     g_inner % (PeCount*tN) == 0）。outer 为循环维度不受 PE 约束。
//
// 语义与静态版完全一致（对齐 CANN/PyTorch）：平局返回首个最大值索引；
// fp16/fp32/bf16 的 NaN 视为最大值（首个 NaN 索引）；±0.0 相等。
// 输出 layout：去掉 axis 维后的 RowMajor 连续张量（keepdim 由上层实现）。
// 输入 dtype 支持 fp16/fp32/bf16/int32（int64 输入同静态版拒绝）；
// 输出 dtype 为 int32/int64。
//
// 物理 tile 形状按 dtype traits 固定（行/列归约各一套默认形状，入口无
// tile 模板参数，对齐 normalization 动态版约定）；tiling 的 tile_m/tile_n
// 只收缩运行期 valid（<= 物理容量，fail-closed 校验）。
// ============================================================================

#include "basic_op/utils/spmd_partition.hpp"
#include "basic_op/argmaxv2/detail/argmaxv2_dynamic.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace supernpu::multi_thread {

namespace argmaxv2_dyn {

// 运行期 tiling（host 侧构造；见文件头注释的字段说明）。
struct ArgMaxV2TilingData {
    int64_t g_outer;
    int64_t g_axis;
    int64_t g_inner;
    int64_t tile_m;
    int64_t tile_n;
    int64_t axis_last;
};

}  // namespace argmaxv2_dyn

// ----------------------------------------------------------------------------
// argmaxv2_dynamic：任意维度 ArgMaxV2 的四 PE 动态入口。
//
//   argmaxv2_dynamic<dtype, idtype>(input, output, &tiling);
//   PE 数可用 PeCount 模板参数覆盖（默认 kDefaultPeCount）。
//
// fail-closed 契约（校验失败直接 return，不写任何输出）：
//   g_outer/g_axis/g_inner > 0；axis_last ∈ {0, 1}；
//   0 < tile_m <= 物理tM、0 < tile_n <= 物理tN（<= 0 取默认满块）；
//   tid >= PeCount 的 PE 空转（normalization 同风格防护）。
// ----------------------------------------------------------------------------
template <typename DType, typename IdType,
          std::uint32_t PeCount = kDefaultPeCount>
void argmaxv2_dynamic(DType *input, IdType *output,
                      const argmaxv2_dyn::ArgMaxV2TilingData *tiling) {
    static_assert(!std::is_same_v<DType, int64_t>,
                  "int64 input is not supported: the tile-op layer has no "
                  "64-bit reduce/compare instructions (TROWARGMAX/TCOLARGMAX "
                  "contracts and the model reduce/compare dtype whitelist "
                  "cap at 32-bit)");
    static_assert(PeCount > 0, "PeCount must be positive");

    const int64_t gOuter = tiling->g_outer;
    const int64_t gAxis = tiling->g_axis;
    const int64_t gInner = tiling->g_inner;
    const int64_t axisLast = tiling->axis_last;
    const std::uint32_t tid = get_thread_idx();

    if (gOuter <= 0 || gAxis <= 0 || gInner <= 0 ||
        (axisLast != 0 && axisLast != 1) || tid >= PeCount) {
        return;
    }

    if (axisLast != 0) {
        // 尾轴：非归约维折叠为行 → [g_outer, g_axis] 行归约
        constexpr int kTM = argmaxv2_detail::ArgMaxTraits<DType>::kRowTileM;
        constexpr int kTN = argmaxv2_detail::ArgMaxTraits<DType>::kRowTileN;
        const int64_t tileM = tiling->tile_m > 0 ? tiling->tile_m : kTM;
        const int64_t tileN = tiling->tile_n > 0 ? tiling->tile_n : kTN;
        if (tileM <= 0 || tileM > kTM || tileN <= 0 || tileN > kTN) {
            return;
        }

        // 均衡分区（地板除 + 余数给前几个 PE）：base = g_outer / PeCount，
        // 前 g_outer % PeCount 个 PE 各多 1 行；g_outer >= PeCount 时全部
        // PE 参与计算（如 g_outer=5、4PE → 2/1/1/1），对 g_outer 无整除
        // 约束；g_outer < PeCount 时前 g_outer 个 PE 各 1 行、其余空转。
        const int64_t tid64 = static_cast<int64_t>(tid);
        const int64_t baseRows = gOuter / PeCount;
        const int64_t extraPes = gOuter % PeCount;
        const int64_t peRows = baseRows + (tid64 < extraPes ? 1 : 0);
        if (peRows <= 0) {
            return;
        }
        // 行起点 = 前 tid 个 PE 的行数之和 = tid*base + min(tid, extra)。
        const int64_t peStart =
            tid64 * baseRows + (tid64 < extraPes ? tid64 : extraPes);

        ::argmaxv2_row_dynamic<DType, IdType, kTM, kTN>(
            input + peStart * gAxis, output + peStart, peRows, gAxis, tileM,
            tileN);
    } else {
        // 中间/首轴：逐 outer 切片 [g_axis, g_inner] 列归约
        constexpr int kTM = argmaxv2_detail::ArgMaxTraits<DType>::kColTileM;
        constexpr int kTN = argmaxv2_detail::ArgMaxTraits<DType>::kColTileN;
        const int64_t tileM = tiling->tile_m > 0 ? tiling->tile_m : kTM;
        const int64_t tileN = tiling->tile_n > 0 ? tiling->tile_n : kTN;
        if (tileM <= 0 || tileM > kTM || tileN <= 0 || tileN > kTN) {
            return;
        }

        // 列块 ceil 分区：窗口 = [tid*blocksPerPe*tileN, ...) 自动块对齐；
        // 每列归约线独立，g_inner 无整除约束，末块可为部分块。
        const int64_t blocks = (gInner + tileN - 1) / tileN;
        const int64_t blocksPerPe = (blocks + PeCount - 1) / PeCount;
        const int64_t colBegin =
            static_cast<int64_t>(tid) * blocksPerPe * tileN;
        if (colBegin >= gInner) {
            return;
        }
        const int64_t colEnd = colBegin + blocksPerPe * tileN < gInner
                                   ? colBegin + blocksPerPe * tileN
                                   : gInner;

        for (int64_t o = 0; o < gOuter; ++o) {
            ::argmaxv2_col_dynamic<DType, IdType, kTM, kTN>(
                input + o * gAxis * gInner, output + o * gInner, gAxis, gInner,
                colBegin, colEnd, tileM, tileN);
        }
    }
}

}  // namespace supernpu::multi_thread
