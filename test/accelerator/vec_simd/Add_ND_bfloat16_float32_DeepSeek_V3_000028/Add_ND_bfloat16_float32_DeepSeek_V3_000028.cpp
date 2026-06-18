#include <common/pto_tileop.hpp>
#include "../../include/accelerator_vec_simd.h"

#define kM 1024
#define kN 1024

#define kTM 64
#define kTN 64

int main(){
    float srca[kM*kN];
    float srcb[kM*kN];
    float dst[kM*kN];

    matadd<float, kM, kN, kTM, kTN>(dst, srca, srcb);
}