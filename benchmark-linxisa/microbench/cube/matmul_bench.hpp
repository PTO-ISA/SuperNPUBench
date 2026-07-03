#ifndef MATMUL_BENCH_HPP
#define MATMUL_BENCH_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <cstdio>
#include "benchmark.h"

using namespace pto;

// Helper function to copy Acc to Vec and then to global memory
template <is_global_data_v GmOut, is_tile_data_v TileAcc>
void TCOPYOUT_ACC(GmOut &Gout, TileAcc &tAcc){
    using TileAccOut = Tile<Location::Vec, typename TileAcc::DType, TileAcc::Rows, TileAcc::Cols, BLayout::RowMajor, TileAcc::ValidRow, TileAcc::ValidCol>;
    TileAccOut tAccOut;
    TCVT(tAccOut, tAcc);
    TCOPYOUT(Gout, tAccOut);
}

// ============================================================================
// Matrix Multiplication Benchmark: C = A × B
// ============================================================================

/**
 * @brief 基础矩阵乘法 benchmark
 * 
 * 测试 MATMUL 指令的性能
 * 
 * @tparam dtype 数据类型 (FP32, FP16, BF16, INT8)
 * @tparam tM Tile 行数
 * @tparam tN Tile 列数
 * @tparam tK Tile 内积维度
 */
template <typename dtype, int tM, int tN, int tK>
void matmul_bench(float* c_ptr, dtype* a_ptr, dtype* b_ptr) {
    using gm_shapeA = global_tensor<dtype, RowMajor<tM, tK>>;
    using gm_shapeB = global_tensor<dtype, RowMajor<tK, tN>>;
    using gm_shapeC = global_tensor<float, RowMajor<tM, tN>>;
    
    using tile_shapeA = TileLeft<dtype, tM, tK>;
    using tile_shapeB = TileRight<dtype, tK, tN>;
    using tile_shapeACC = TileAcc<float, tM, tN>;
    
    using itA = global_iterator<gm_shapeA, tile_shapeA>;
    using itB = global_iterator<gm_shapeB, tile_shapeB>;
    using itC = global_iterator<gm_shapeC, tile_shapeACC>;
    
    itA gAIter(a_ptr);
    itB gBIter(b_ptr);
    itC gCIter(c_ptr);
    
    auto gA = gAIter(0, 0);
    auto gB = gBIter(0, 0);
    auto gC = gCIter(0, 0);
    
    tile_shapeA tA;
    tile_shapeB tB;
    tile_shapeACC tACC;
    
    BENCHSTART;
    
    // 加载 A 和 B
    TCOPYIN(tA, gA);
    TCOPYIN(tB, gB);
    
    // 执行矩阵乘法: C = A × B
    MATMUL(tACC, tA, tB);
    
    // 存储结果 C (需要先从 Acc 转换到 Vec)
    TCOPYOUT_ACC(gC, tACC);
    
    BENCHEND;
}

// ============================================================================
// Benchmark Configurations
// ============================================================================

// FP32 矩阵乘法
void matmul_fp32_16x16(float* c, float* a, float* b) {
    matmul_bench<float, 16, 16, 16>(c, a, b);
}

void matmul_fp32_32x32(float* c, float* a, float* b) {
    matmul_bench<float, 32, 32, 32>(c, a, b);
}

void matmul_fp32_64x64(float* c, float* a, float* b) {
    matmul_bench<float, 64, 64, 64>(c, a, b);
}

// FP16 矩阵乘法
void matmul_fp16_16x16(float* c, __half* a, __half* b) {
    matmul_bench<__half, 16, 16, 16>(c, a, b);
}

void matmul_fp16_32x32(float* c, __half* a, __half* b) {
    matmul_bench<__half, 32, 32, 32>(c, a, b);
}

// BF16 矩阵乘法
void matmul_bf16_16x16(float* c, __bf16* a, __bf16* b) {
    matmul_bench<__bf16, 16, 16, 16>(c, a, b);
}

void matmul_bf16_32x32(float* c, __bf16* a, __bf16* b) {
    matmul_bench<__bf16, 32, 32, 32>(c, a, b);
}

// Note: INT8 matmul requires special tile sizes and is not included in basic benchmarks
// Use kernels/matmul for INT8 matmul tests

#endif // MATMUL_BENCH_HPP
