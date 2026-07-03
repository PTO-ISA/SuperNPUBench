#ifndef VECTOR_REDUCE_BENCH_HPP
#define VECTOR_REDUCE_BENCH_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>

using namespace pto;

// ============================================================================
// Reduction Operations Benchmark
// ============================================================================

/**
 * @brief 行最大值归约 benchmark: dst[i] = max(src[i, :])
 * 
 * 将每行的最大值归约到一列
 */
template <typename dtype, int tM, int tN>
void trowmax_bench(dtype* dst_ptr, dtype* src_ptr) {
    using gm_shape_src = global_tensor<dtype, RowMajor<tM, tN>>;
    using gm_shape_dst = global_tensor<dtype, RowMajor<tM, 1>>;
    
    using tile_shape_src = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    using tile_shape_dst = Tile<Location::Vec, dtype, tM, 1, BLayout::RowMajor>;
    
    using it_src = global_iterator<gm_shape_src, tile_shape_src>;
    using it_dst = global_iterator<gm_shape_dst, tile_shape_dst>;
    
    it_src gSrcIter(src_ptr);
    it_dst gDstIter(dst_ptr);
    
    auto gSrc = gSrcIter(0, 0);
    auto gDst = gDstIter(0, 0);
    
    tile_shape_src tSrc;
    tile_shape_dst tDst;
    
    TCOPYIN(tSrc, gSrc);
    
    // 行最大值归约
    TROWMAX(tDst, tSrc);
    
    TCOPYOUT(gDst, tDst);
}

/**
 * @brief 列最大值归约 benchmark: dst[j] = max(src[:, j])
 * 
 * 将每列的最大值归约到一行
 */
template <typename dtype, int tM, int tN>
void tcolmax_bench(dtype* dst_ptr, dtype* src_ptr) {
    using gm_shape_src = global_tensor<dtype, RowMajor<tM, tN>>;
    using gm_shape_dst = global_tensor<dtype, RowMajor<1, tN>>;
    
    using tile_shape_src = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    using tile_shape_dst = Tile<Location::Vec, dtype, 1, tN, BLayout::RowMajor>;
    
    using it_src = global_iterator<gm_shape_src, tile_shape_src>;
    using it_dst = global_iterator<gm_shape_dst, tile_shape_dst>;
    
    it_src gSrcIter(src_ptr);
    it_dst gDstIter(dst_ptr);
    
    auto gSrc = gSrcIter(0, 0);
    auto gDst = gDstIter(0, 0);
    
    tile_shape_src tSrc;
    tile_shape_dst tDst;
    
    TCOPYIN(tSrc, gSrc);
    
    // 列最大值归约
    TCOLMAX(tDst, tSrc);
    
    TCOPYOUT(gDst, tDst);
}

/**
 * @brief 行最小值归约 benchmark: dst[i] = min(src[i, :])
 */
template <typename dtype, int tM, int tN>
void trowmin_bench(dtype* dst_ptr, dtype* src_ptr) {
    using gm_shape_src = global_tensor<dtype, RowMajor<tM, tN>>;
    using gm_shape_dst = global_tensor<dtype, RowMajor<tM, 1>>;
    
    using tile_shape_src = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    using tile_shape_dst = Tile<Location::Vec, dtype, tM, 1, BLayout::RowMajor>;
    
    using it_src = global_iterator<gm_shape_src, tile_shape_src>;
    using it_dst = global_iterator<gm_shape_dst, tile_shape_dst>;
    
    it_src gSrcIter(src_ptr);
    it_dst gDstIter(dst_ptr);
    
    auto gSrc = gSrcIter(0, 0);
    auto gDst = gDstIter(0, 0);
    
    tile_shape_src tSrc;
    tile_shape_dst tDst;
    
    TCOPYIN(tSrc, gSrc);
    
    TROWMIN(tDst, tSrc);
    
    TCOPYOUT(gDst, tDst);
}

/**
 * @brief 列最小值归约 benchmark: dst[j] = min(src[:, j])
 */
template <typename dtype, int tM, int tN>
void tcolmin_bench(dtype* dst_ptr, dtype* src_ptr) {
    using gm_shape_src = global_tensor<dtype, RowMajor<tM, tN>>;
    using gm_shape_dst = global_tensor<dtype, RowMajor<1, tN>>;
    
    using tile_shape_src = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    using tile_shape_dst = Tile<Location::Vec, dtype, 1, tN, BLayout::RowMajor>;
    
    using it_src = global_iterator<gm_shape_src, tile_shape_src>;
    using it_dst = global_iterator<gm_shape_dst, tile_shape_dst>;
    
    it_src gSrcIter(src_ptr);
    it_dst gDstIter(dst_ptr);
    
    auto gSrc = gSrcIter(0, 0);
    auto gDst = gDstIter(0, 0);
    
    tile_shape_src tSrc;
    tile_shape_dst tDst;
    
    TCOPYIN(tSrc, gSrc);
    
    TCOLMIN(tDst, tSrc);
    
    TCOPYOUT(gDst, tDst);
}

/**
 * @brief 行求和归约 benchmark: dst[i] = sum(src[i, :])
 */
template <typename dtype, int tM, int tN>
void trowsum_bench(dtype* dst_ptr, dtype* src_ptr) {
    using gm_shape_src = global_tensor<dtype, RowMajor<tM, tN>>;
    using gm_shape_dst = global_tensor<dtype, RowMajor<tM, 1>>;
    
    using tile_shape_src = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    using tile_shape_dst = Tile<Location::Vec, dtype, tM, 1, BLayout::RowMajor>;
    
    using it_src = global_iterator<gm_shape_src, tile_shape_src>;
    using it_dst = global_iterator<gm_shape_dst, tile_shape_dst>;
    
    it_src gSrcIter(src_ptr);
    it_dst gDstIter(dst_ptr);
    
    auto gSrc = gSrcIter(0, 0);
    auto gDst = gDstIter(0, 0);
    
    tile_shape_src tSrc;
    tile_shape_dst tDst;
    
    TCOPYIN(tSrc, gSrc);
    
    TROWEXPAND(tDst, tSrc);
    
    TCOPYOUT(gDst, tDst);
}

/**
 * @brief 列求和归约 benchmark: dst[j] = sum(src[:, j])
 */
template <typename dtype, int tM, int tN>
void tcolsum_bench(dtype* dst_ptr, dtype* src_ptr) {
    using gm_shape_src = global_tensor<dtype, RowMajor<tM, tN>>;
    using gm_shape_dst = global_tensor<dtype, RowMajor<1, tN>>;
    
    using tile_shape_src = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    using tile_shape_dst = Tile<Location::Vec, dtype, 1, tN, BLayout::RowMajor>;
    
    using it_src = global_iterator<gm_shape_src, tile_shape_src>;
    using it_dst = global_iterator<gm_shape_dst, tile_shape_dst>;
    
    it_src gSrcIter(src_ptr);
    it_dst gDstIter(dst_ptr);
    
    auto gSrc = gSrcIter(0, 0);
    auto gDst = gDstIter(0, 0);
    
    tile_shape_src tSrc;
    tile_shape_dst tDst;
    
    TCOPYIN(tSrc, gSrc);
    
    TCOLEXPAND(tDst, tSrc);
    
    TCOPYOUT(gDst, tDst);
}

// ============================================================================
// Benchmark Configurations
// ============================================================================

// FP32
void trowmax_fp32_16x16(float* dst, float* src) {
    trowmax_bench<float, 16, 16>(dst, src);
}

void tcolmax_fp32_16x16(float* dst, float* src) {
    tcolmax_bench<float, 16, 16>(dst, src);
}

void trowmin_fp32_16x16(float* dst, float* src) {
    trowmin_bench<float, 16, 16>(dst, src);
}

void tcolmin_fp32_16x16(float* dst, float* src) {
    tcolmin_bench<float, 16, 16>(dst, src);
}

void trowsum_fp32_16x16(float* dst, float* src) {
    trowsum_bench<float, 16, 16>(dst, src);
}

void tcolsum_fp32_16x16(float* dst, float* src) {
    tcolsum_bench<float, 16, 16>(dst, src);
}

// FP16
void trowmax_fp16_16x16(__half* dst, __half* src) {
    trowmax_bench<__half, 16, 16>(dst, src);
}

void tcolmax_fp16_16x16(__half* dst, __half* src) {
    tcolmax_bench<__half, 16, 16>(dst, src);
}

// BF16
void trowmax_bf16_16x16(__bf16* dst, __bf16* src) {
    trowmax_bench<__bf16, 16, 16>(dst, src);
}

void tcolmax_bf16_16x16(__bf16* dst, __bf16* src) {
    tcolmax_bench<__bf16, 16, 16>(dst, src);
}

// INT32
void trowmax_int32_16x16(int32_t* dst, int32_t* src) {
    trowmax_bench<int32_t, 16, 16>(dst, src);
}

void tcolmax_int32_16x16(int32_t* dst, int32_t* src) {
    tcolmax_bench<int32_t, 16, 16>(dst, src);
}

#endif // VECTOR_REDUCE_BENCH_HPP
