#ifndef MEGA_MOE_GMM_HPP
#define MEGA_MOE_GMM_HPP

#include <cstdint>

namespace mega_moe_gmm {

// 这些函数在独立翻译单元 (mega_moe_gmm.cpp) 中实现，
// 避免 mega_moe_sim.hpp 中的 int64 结构体触发 v2i64 BUILD_VECTOR 编译器崩溃。
// Tile 寄存器是函数局部的，不能跨函数传递，所以每个函数完整包含 tile 操作链。

void gmm_pipeline_for_token(
    float* yOut,
    const float* xIn,
    const float* w1fp32,
    const float* w2fp32,
    const int32_t* topkIds,
    const float* topkWeights,
    float* workspace,   // y1/y2 intermediate buffer
    uint32_t token,
    uint32_t bs,
    uint32_t h,
    uint32_t hiddenDim,
    uint32_t expertPerRank);

} // namespace mega_moe_gmm

#endif
