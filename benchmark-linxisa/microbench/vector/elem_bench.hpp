#ifndef VECTOR_ELEM_BENCH_HPP
#define VECTOR_ELEM_BENCH_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>

using namespace pto;

// ============================================================================
// Element-wise Vector Operations Benchmark
// ============================================================================

/**
 * @brief 逐元素加法 benchmark: C = A + B
 */
template <typename dtype, int tM, int tN>
void tadd_bench(dtype* c_ptr, dtype* a_ptr, dtype* b_ptr) {
    using gm_shape = global_tensor<dtype, RowMajor<tM, tN>>;
    using tile_shape = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    
    using it = global_iterator<gm_shape, tile_shape>;
    
    it gAIter(a_ptr);
    it gBIter(b_ptr);
    it gCIter(c_ptr);
    
    auto gA = gAIter(0, 0);
    auto gB = gBIter(0, 0);
    auto gC = gCIter(0, 0);
    
    tile_shape tA, tB, tC;
    
    TCOPYIN(tA, gA);
    TCOPYIN(tB, gB);
    
    TADD(tC, tA, tB);
    
    TCOPYOUT(gC, tC);
}

/**
 * @brief 逐元素减法 benchmark: C = A - B
 */
template <typename dtype, int tM, int tN>
void tsub_bench(dtype* c_ptr, dtype* a_ptr, dtype* b_ptr) {
    using gm_shape = global_tensor<dtype, RowMajor<tM, tN>>;
    using tile_shape = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    
    using it = global_iterator<gm_shape, tile_shape>;
    
    it gAIter(a_ptr);
    it gBIter(b_ptr);
    it gCIter(c_ptr);
    
    auto gA = gAIter(0, 0);
    auto gB = gBIter(0, 0);
    auto gC = gCIter(0, 0);
    
    tile_shape tA, tB, tC;
    
    TCOPYIN(tA, gA);
    TCOPYIN(tB, gB);
    
    TSUB(tC, tA, tB);
    
    TCOPYOUT(gC, tC);
}

/**
 * @brief 逐元素乘法 benchmark: C = A * B
 */
template <typename dtype, int tM, int tN>
void tmul_bench(dtype* c_ptr, dtype* a_ptr, dtype* b_ptr) {
    using gm_shape = global_tensor<dtype, RowMajor<tM, tN>>;
    using tile_shape = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    
    using it = global_iterator<gm_shape, tile_shape>;
    
    it gAIter(a_ptr);
    it gBIter(b_ptr);
    it gCIter(c_ptr);
    
    auto gA = gAIter(0, 0);
    auto gB = gBIter(0, 0);
    auto gC = gCIter(0, 0);
    
    tile_shape tA, tB, tC;
    
    TCOPYIN(tA, gA);
    TCOPYIN(tB, gB);
    
    TMUL(tC, tA, tB);
    
    TCOPYOUT(gC, tC);
}

/**
 * @brief 逐元素除法 benchmark: C = A / B
 */
template <typename dtype, int tM, int tN>
void tdiv_bench(dtype* c_ptr, dtype* a_ptr, dtype* b_ptr) {
    using gm_shape = global_tensor<dtype, RowMajor<tM, tN>>;
    using tile_shape = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    
    using it = global_iterator<gm_shape, tile_shape>;
    
    it gAIter(a_ptr);
    it gBIter(b_ptr);
    it gCIter(c_ptr);
    
    auto gA = gAIter(0, 0);
    auto gB = gBIter(0, 0);
    auto gC = gCIter(0, 0);
    
    tile_shape tA, tB, tC;
    
    TCOPYIN(tA, gA);
    TCOPYIN(tB, gB);
    
    TDIV(tC, tA, tB);
    
    TCOPYOUT(gC, tC);
}

/**
 * @brief 逐元素最大值 benchmark: C = max(A, B)
 */
template <typename dtype, int tM, int tN>
void tmax_bench(dtype* c_ptr, dtype* a_ptr, dtype* b_ptr) {
    using gm_shape = global_tensor<dtype, RowMajor<tM, tN>>;
    using tile_shape = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    
    using it = global_iterator<gm_shape, tile_shape>;
    
    it gAIter(a_ptr);
    it gBIter(b_ptr);
    it gCIter(c_ptr);
    
    auto gA = gAIter(0, 0);
    auto gB = gBIter(0, 0);
    auto gC = gCIter(0, 0);
    
    tile_shape tA, tB, tC;
    
    TCOPYIN(tA, gA);
    TCOPYIN(tB, gB);
    
    TMAX(tC, tA, tB);
    
    TCOPYOUT(gC, tC);
}

/**
 * @brief 逐元素最小值 benchmark: C = min(A, B)
 */
template <typename dtype, int tM, int tN>
void tmin_bench(dtype* c_ptr, dtype* a_ptr, dtype* b_ptr) {
    using gm_shape = global_tensor<dtype, RowMajor<tM, tN>>;
    using tile_shape = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    
    using it = global_iterator<gm_shape, tile_shape>;
    
    it gAIter(a_ptr);
    it gBIter(b_ptr);
    it gCIter(c_ptr);
    
    auto gA = gAIter(0, 0);
    auto gB = gBIter(0, 0);
    auto gC = gCIter(0, 0);
    
    tile_shape tA, tB, tC;
    
    TCOPYIN(tA, gA);
    TCOPYIN(tB, gB);
    
    TMIN(tC, tA, tB);
    
    TCOPYOUT(gC, tC);
}

// ============================================================================
// Benchmark Configurations
// ============================================================================

// FP32
void tadd_fp32_16x16(float* c, float* a, float* b) {
    tadd_bench<float, 16, 16>(c, a, b);
}

void tsub_fp32_16x16(float* c, float* a, float* b) {
    tsub_bench<float, 16, 16>(c, a, b);
}

void tmul_fp32_16x16(float* c, float* a, float* b) {
    tmul_bench<float, 16, 16>(c, a, b);
}

void tdiv_fp32_16x16(float* c, float* a, float* b) {
    tdiv_bench<float, 16, 16>(c, a, b);
}

void tmax_fp32_16x16(float* c, float* a, float* b) {
    tmax_bench<float, 16, 16>(c, a, b);
}

void tmin_fp32_16x16(float* c, float* a, float* b) {
    tmin_bench<float, 16, 16>(c, a, b);
}

// FP16
void tadd_fp16_16x16(__half* c, __half* a, __half* b) {
    tadd_bench<__half, 16, 16>(c, a, b);
}

void tmul_fp16_16x16(__half* c, __half* a, __half* b) {
    tmul_bench<__half, 16, 16>(c, a, b);
}

// BF16 - Commented out due to compiler issues with __blkc_bf16 type
// void tadd_bf16_16x16(__bf16* c, __bf16* a, __bf16* b) {
//     tadd_bench<__bf16, 16, 16>(c, a, b);
// }

// void tmul_bf16_16x16(__bf16* c, __bf16* a, __bf16* b) {
//     tmul_bench<__bf16, 16, 16>(c, a, b);
// }

// INT32
void tadd_int32_16x16(int32_t* c, int32_t* a, int32_t* b) {
    tadd_bench<int32_t, 16, 16>(c, a, b);
}

void tmul_int32_16x16(int32_t* c, int32_t* a, int32_t* b) {
    tmul_bench<int32_t, 16, 16>(c, a, b);
}

#endif // VECTOR_ELEM_BENCH_HPP
