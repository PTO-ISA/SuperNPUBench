#include <common/pto_tileop.hpp>
#include "benchmark.h"

#define MAP_MEM_BASE 0x4000802000ULL
#define OUT_COUNT 32768

int main(){
    __half* out = (__half*)MAP_MEM_BASE;

    for (int i = 0; i < OUT_COUNT; ++i) {
        out[i] = (__half)((float)i * 0.001f);
    }

    BENCHSTART;
    BENCHEND;

    return 0;
}
