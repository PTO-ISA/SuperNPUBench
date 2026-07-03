#ifndef MATMUL_BENCH_HPP
#define MATMUL_BENCH_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <cstdio>

using namespace pto;

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
    
    // 加载 A 和 B
    TCOPYIN(tA, gA);
    TCOPYIN(tB, gB);
    
    // 执行矩阵乘法: C = A × B
    MATMUL(tACC, tA, tB);
    
    // 存储结果 C
    TCOPYOUT(gC, tACC);
}

/**
 * @brief 分块矩阵乘法 benchmark
 * 
 * 测试大矩阵的分块计算性能
 * 
 * @tparam dtype 数据类型
 * @tparam gM 全局矩阵行数
 * @tparam gN 全局矩阵列数
 * @tparam gK 全局矩阵内积维度
 * @tparam tM Tile 行数
 * @tparam tN Tile 列数
 * @tparam tK Tile 内积维度
 */
template <typename dtype, int gM, int gN, int gK, int tM, int tN, int tK>
void matmul_tiled_bench(float* c_ptr, dtype* a_ptr, dtype* b_ptr) {
    using gm_shapeA = global_tensor<dtype, RowMajor<gM, gK>>;
    using gm_shapeB = global_tensor<dtype, RowMajor<gK, gN>>;
    using gm_shapeC = global_tensor<float, RowMajor<gM, gN>>;
    
    using tile_shapeA = TileLeft<dtype, tM, tK>;
    using tile_shapeB = TileRight<dtype, tK, tN>;
    using tile_shapeACC = TileAcc<float, tM, tN>;
    
    using itA = global_iterator<gm_shapeA, tile_shapeA>;
    using itB = global_iterator<gm_shapeB, tile_shapeB>;
    using itC = global_iterator<gm_shapeC, tile_shapeACC>;
    
    itA gAIter(a_ptr);
    itB gBIter(b_ptr);
    itC gCIter(c_ptr);
    
    const int Mb = gM / tM;
    const int Nb = gN / tN;
    const int Kb = gK / tK;
    
    for (int i = 0; i < Mb; ++i) {
        for (int j = 0; j < Nb; ++j) {
            auto gC = gCIter(i, j);
            tile_shapeACC tACC;
            
            for (int k = 0; k < Kb; ++k) {
                auto gA = gAIter(i, k);
                auto gB = gBIter(k, j);
                
                tile_shapeA tA;
                tile_shapeB tB;
                
                TCOPYIN(tA, gA);
                TCOPYIN(tB, gB);
                
                if (k == 0) {
                    MATMUL(tACC, tA, tB);
                } else {
                    MATMACC(tACC, tA, tB);
                }
            }
            
            TCOPYOUT(gC, tACC);
        }
    }
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

// INT8 矩阵乘法
void matmul_int8_16x16(float* c, int8_t* a, int8_t* b) {
    matmul_bench<int8_t, 16, 16, 16>(c, a, b);
}

void matmul_int8_32x32(float* c, int8_t* a, int8_t* b) {
    matmul_bench<int8_t, 32, 32, 32>(c, a, b);
}

#endif // MATMUL_BENCH_HPP
