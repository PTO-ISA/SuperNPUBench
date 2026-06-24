#include <common/pto_tileop.hpp>
#include <cstring>
#include "benchmark.h"

typedef float dtype;

#ifndef LOOP
#define LOOP 256
#endif

#ifndef CHAIN_NUM
#define CHAIN_NUM 1
#endif

using tile_shape = Tile<Location::Vec, dtype, 16, 16, BLayout::RowMajor>;

#include "lat_bw_func.h"

#ifdef MT
#define MAIN_FUNC(name) \
    int main(){ \
            tile_shape placeholder; \
            BENCHSTART; \
            test_##name<<<LOOP,1,1>>>(1, placeholder.data()); \
            BENCHEND; \
            return 0; \
    }
#else
#define MAIN_FUNC(name) \
    int main(){ \
            tile_shape placeholder; \
            BENCHSTART; \
            test_##name<<<1,1,1>>>(LOOP, placeholder.data()); \
            BENCHEND; \
            return 0; \
    }
#endif


#include "lat_bw_vec.h"