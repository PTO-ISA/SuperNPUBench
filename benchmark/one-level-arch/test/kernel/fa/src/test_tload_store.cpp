#include <common/pto_tileop.hpp>
#include "benchmark.h"

#define S1 64
#define D 512
#define KTM 32
#define KTD 64
#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN 4*1024
#define MAP_MEM_BASE 0x4000802000ULL

using namespace pto;

static void init_deterministic(__half* data, int count, int seed) {
    for (int i = 0; i < count; ++i) {
        float val = ((float)((i * 31 + seed * 17) % 100)) / 100.0f - 0.5f;
        data[i] = (__half)val;
    }
}

int main(){
    using dtype = __half;

    dtype qp[S1*D + 2*ALIGN];
    dtype* q = (dtype*)(((uint64_t)qp & ALIGN_MASK) + ALIGN);

    init_deterministic(q, S1*D, 1);

    dtype* out = (dtype*)MAP_MEM_BASE;

    using gmQ = global_tensor<dtype, RowMajor<S1, D>>;
    using gmO = global_tensor<dtype, RowMajor<S1, D>>;
    using tileQ = Tile<Location::Vec, dtype, KTM, KTD, BLayout::RowMajor>;
    using tileO = Tile<Location::Vec, dtype, KTM, KTD, BLayout::RowMajor>;

    using itQ = global_iterator<gmQ, tileQ>;
    using itO = global_iterator<gmO, tileO>;

    itQ gIterQ(q);
    itO gIterO(out);

    const int Qb = S1 / KTM;
    const int Db = D / KTD;

    BENCHSTART;
    for (int i = 0; i < Qb; ++i) {
        for (int dd = 0; dd < Db; ++dd) {
            tileQ tQ;
            auto gQ = gIterQ(i, dd);
            TLOAD(tQ, gQ);

            tileO tO;
            auto gO = gIterO(i, dd);
            TSTORE(gO, tQ);
        }
    }
    BENCHEND;

    return 0;
}
