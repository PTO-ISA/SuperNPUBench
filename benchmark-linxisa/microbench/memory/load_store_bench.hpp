#ifndef MEMORY_LOAD_STORE_BENCH_HPP
#define MEMORY_LOAD_STORE_BENCH_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>

using namespace pto;

// ============================================================================
// Memory Load/Store Operations Benchmark
// ============================================================================

/**
 * @brief TLOAD benchmark: 从全局内存加载到 Tile
 * 
 * 测试连续内存加载性能
 */
template <typename dtype, int tM, int tN>
void tload_bench(dtype* tile_ptr, dtype* global_ptr) {
    using gm_shape = global_tensor<dtype, RowMajor<tM, tN>>;
    using tile_shape = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    
    using it = global_iterator<gm_shape, tile_shape>;
    
    it gIter(global_ptr);
    auto gTile = gIter(0, 0);
    
    tile_shape tTile;
    
    // 从全局内存加载到 Tile
    TLOAD(tTile, gTile);
    
    // 存储回全局内存（用于验证）
    TSTORE(gTile, tTile);
}

/**
 * @brief TSTORE benchmark: 从 Tile 存储到全局内存
 * 
 * 测试连续内存存储性能
 */
template <typename dtype, int tM, int tN>
void tstore_bench(dtype* global_ptr, dtype* tile_ptr) {
    using gm_shape = global_tensor<dtype, RowMajor<tM, tN>>;
    using tile_shape = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    
    using it = global_iterator<gm_shape, tile_shape>;
    
    it gIter(global_ptr);
    auto gTile = gIter(0, 0);
    
    tile_shape tTile;
    
    // 加载到 Tile
    TLOAD(tTile, gTile);
    
    // 从 Tile 存储到全局内存
    TSTORE(gTile, tTile);
}

/**
 * @brief TCOPYIN benchmark: 带布局转换的拷贝
 * 
 * 测试从全局内存拷贝到 Tile（可能涉及布局转换）
 */
template <typename dtype, int tM, int tN>
void tcopyin_bench(dtype* tile_ptr, dtype* global_ptr) {
    using gm_shape = global_tensor<dtype, RowMajor<tM, tN>>;
    using tile_shape = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    
    using it = global_iterator<gm_shape, tile_shape>;
    
    it gIter(global_ptr);
    auto gTile = gIter(0, 0);
    
    tile_shape tTile;
    
    // 带布局转换的拷贝
    TCOPYIN(tTile, gTile);
    
    TCOPYOUT(gTile, tTile);
}

/**
 * @brief TCOPYOUT benchmark: 带布局转换的拷贝
 * 
 * 测试从 Tile 拷贝到全局内存（可能涉及布局转换）
 */
template <typename dtype, int tM, int tN>
void tcopyout_bench(dtype* global_ptr, dtype* tile_ptr) {
    using gm_shape = global_tensor<dtype, RowMajor<tM, tN>>;
    using tile_shape = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    
    using it = global_iterator<gm_shape, tile_shape>;
    
    it gIter(global_ptr);
    auto gTile = gIter(0, 0);
    
    tile_shape tTile;
    
    TCOPYIN(tTile, gTile);
    
    // 带布局转换的拷贝
    TCOPYOUT(gTile, tTile);
}

/**
 * @brief 跨步访问 benchmark
 * 
 * 测试非连续内存访问性能
 */
template <typename dtype, int tM, int tN, int stride>
void strided_load_bench(dtype* tile_ptr, dtype* global_ptr) {
    using gm_shape = global_tensor<dtype, RowMajor<tM * stride, tN>>;
    using tile_shape = Tile<Location::Vec, dtype, tM, tN, BLayout::RowMajor>;
    
    using it = global_iterator<gm_shape, tile_shape>;
    
    it gIter(global_ptr);
    
    // 每隔 stride 行访问一次
    for (int i = 0; i < tM; ++i) {
        auto gTile = gIter(i * stride, 0);
        tile_shape tTile;
        TLOAD(tTile, gTile);
        TSTORE(gTile, tTile);
    }
}

// ============================================================================
// Benchmark Configurations
// ============================================================================

// FP32
void tload_fp32_16x16(float* tile, float* global) {
    tload_bench<float, 16, 16>(tile, global);
}

void tstore_fp32_16x16(float* global, float* tile) {
    tstore_bench<float, 16, 16>(global, tile);
}

void tcopyin_fp32_16x16(float* tile, float* global) {
    tcopyin_bench<float, 16, 16>(tile, global);
}

void tcopyout_fp32_16x16(float* global, float* tile) {
    tcopyout_bench<float, 16, 16>(global, tile);
}

void tload_fp32_32x32(float* tile, float* global) {
    tload_bench<float, 32, 32>(tile, global);
}

void tstore_fp32_32x32(float* global, float* tile) {
    tstore_bench<float, 32, 32>(global, tile);
}

// FP16
void tload_fp16_16x16(__half* tile, __half* global) {
    tload_bench<__half, 16, 16>(tile, global);
}

void tstore_fp16_16x16(__half* global, __half* tile) {
    tstore_bench<__half, 16, 16>(global, tile);
}

void tload_fp16_32x32(__half* tile, __half* global) {
    tload_bench<__half, 32, 32>(tile, global);
}

// BF16
void tload_bf16_16x16(__bf16* tile, __bf16* global) {
    tload_bench<__bf16, 16, 16>(tile, global);
}

void tstore_bf16_16x16(__bf16* global, __bf16* tile) {
    tstore_bench<__bf16, 16, 16>(global, tile);
}

// INT32
void tload_int32_16x16(int32_t* tile, int32_t* global) {
    tload_bench<int32_t, 16, 16>(tile, global);
}

void tstore_int32_16x16(int32_t* global, int32_t* tile) {
    tstore_bench<int32_t, 16, 16>(global, tile);
}

// INT8
void tload_int8_16x16(int8_t* tile, int8_t* global) {
    tload_bench<int8_t, 16, 16>(tile, global);
}

void tstore_int8_16x16(int8_t* global, int8_t* tile) {
    tstore_bench<int8_t, 16, 16>(global, tile);
}

#endif // MEMORY_LOAD_STORE_BENCH_HPP
