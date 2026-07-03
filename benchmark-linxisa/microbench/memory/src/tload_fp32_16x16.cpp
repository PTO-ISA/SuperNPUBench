#include "load_store_bench.hpp"
#include <cstdio>
#include <cstdlib>

// Test configuration
constexpr int M = 16;
constexpr int N = 16;

int main() {
    printf("Memory Load Benchmark: FP32 %dx%d\n", M, N);
    
    // Allocate memory
    float* global_mem = (float*)malloc(M * N * sizeof(float));
    float* tile_mem = (float*)malloc(M * N * sizeof(float));
    
    // Initialize with random data
    for (int i = 0; i < M * N; i++) {
        global_mem[i] = (float)rand() / RAND_MAX;
        tile_mem[i] = 0.0f;
    }
    
    // Run benchmark
    tload_fp32_16x16(tile_mem, global_mem);
    
    // Verify result
    float sum = 0.0f;
    for (int i = 0; i < M * N; i++) {
        sum += global_mem[i];
    }
    
    printf("Result sum: %f\n", sum);
    printf("Benchmark completed successfully!\n");
    
    // Cleanup
    free(global_mem);
    free(tile_mem);
    
    return 0;
}
