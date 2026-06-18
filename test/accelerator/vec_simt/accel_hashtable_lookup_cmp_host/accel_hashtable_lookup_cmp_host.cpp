#include <common/pto_tileop.hpp>
#include "../../include/accelerator_vec_simt.h"
#include "benchmark.h"

#define LOOKUP_NUM 4096
#define CAP 8192

void compute_magic_and_shift(uint32_t d, uint32_t& magic, uint32_t& shift) {

    magic = 0;

    return;
}

int main(){
    int8_t slot[16*CAP]; // entry size is 16byte: 1 key(8byte) + 2 value(4byte*2)
    int32_t keys[LOOKUP_NUM];
    int32_t values_output[LOOKUP_NUM];
    uint32_t capacity = CAP;
    uint32_t capacity_magic = 1;
    uint32_t capacity_shift = 1;
    compute_magic_and_shift(CAP, capacity_magic, capacity_shift);
    hashtable_lookup((uint8_t*)slot, keys, values_output, capacity, 0, LOOKUP_NUM, 0, capacity_magic, 0, capacity_shift, 0);
}