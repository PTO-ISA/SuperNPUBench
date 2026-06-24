#include <common/pto_tileop.hpp>
#include <benchmark_support/npu/npu_vec_simt.h>
#include "benchmark.h"

#define INSERT_NUM 4096
#define CAP 8192

void compute_magic_and_shift(uint32_t d, uint32_t& magic, uint32_t& shift) {
    // const uint64_t two_to_k = 1ULL << 32;  // 2^32
    // shift = 0;
    // while ((1ULL << shift) < d) shift++;
 
    // magic = (two_to_k * ((uint64_t)1 << shift)) / d;
    magic = 0;
    return;
}

int main(){
    int8_t slot[16*CAP]; // entry size is 16byte: 1 key(8byte) + 2 value(4byte*2)
    int32_t keys[INSERT_NUM];
    int32_t values[INSERT_NUM];
    uint32_t capacity = CAP;
    uint32_t capacity_magic = 1;
    uint32_t capacity_shift = 1;
    for (int i = 0; i < 16*CAP; i++) {
        slot[i] = -1;
    }
    compute_magic_and_shift(CAP, capacity_magic, capacity_shift);
    hashtable_insert<1024, 1><<<1024,1,1>>>((int8_t*)slot, keys, values, capacity, 0, INSERT_NUM, 0, capacity_magic, 0, capacity_shift, 0);
}