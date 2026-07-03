#include "load_store_bench.hpp"
#include "benchmark.h"

// Test configuration
constexpr int M = 16;
constexpr int N = 16;

int main() {
    // Use stack-allocated arrays instead of malloc
    float global_mem[M * N];
    float tile_mem[M * N];
    
    // Initialize with simple pattern
    for (int i = 0; i < M * N; i++) {
        global_mem[i] = (float)(i % 10) * 0.1f;
        tile_mem[i] = 0.0f;
    }
    
    // Run benchmark
    BENCHSTART;
    tload_fp32_16x16(tile_mem, global_mem);
    BENCHEND;
    
    return 0;
}
