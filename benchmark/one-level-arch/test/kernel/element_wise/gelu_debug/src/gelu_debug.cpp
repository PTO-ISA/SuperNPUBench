#include <common/pto_tileop.hpp>

#include <cstdint>
#include <cstdio>

#include "fileop.h"
#include "single_thread/element_wise/gelu_debug.hpp"


#ifndef DTYPE
#define DTYPE int32_t
#endif

#ifndef tMs
#define tMs 512
#endif

#ifndef gMs
#define gMs (24 * 512 * 1024)
#endif

#ifndef Approximate
#define Approximate false
#endif

// 【静态随机偏移】手动写死，模拟内存地址不是数组起始位置
#define OFFSET_IN  11   // 输入静态偏移
#define OFFSET_OUT 17   // 输出静态偏移
// ============================================================================
// main
// ============================================================================
int main() {

    using dtype = DTYPE;
    // ==========================
    // 申明空间，留出静态偏移空间
    // ==========================
    dtype  input_buf[gMs  + OFFSET_IN];   // 前面留空，模拟随机地址
    dtype  output_buf[gMs + OFFSET_OUT];

    // ==========================
    // 【静态随机地址】
    // ==========================
    dtype* input  = input_buf  + OFFSET_IN;
    dtype* output = output_buf + OFFSET_OUT;

    // debug 变体: tile 计算体内联, 不跨函数传递 tile (见 gelu_debug.hpp)
    gelu_debug<dtype, gMs, tMs>(input, output, Approximate);
}
