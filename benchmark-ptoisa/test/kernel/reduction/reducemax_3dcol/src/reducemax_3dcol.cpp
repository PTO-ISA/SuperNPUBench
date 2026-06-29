#include <common/pto_tileop.hpp>

#include <cstdint>

#include "fileop.h"
#include "reduction/reducemax_colvec_unalign_120_8.hpp"


#ifndef DType
#define DType int32_t
#endif

#ifndef Nums
#define Nums 381
#endif

#ifndef gIMs
#define gIMs 120
#endif

#ifndef gINs
#define gINs 8
#endif


#ifndef tMs
#define tMs 32
#endif

#ifndef tNs
#define tNs 128
#endif

#ifndef tM_VLDs
#define tM_VLDs 120
#endif


// ============================================================================
// main
// ============================================================================
int main() {
    using dtype = DType;

    dtype input_buf[Nums*gIMs*gINs];
//    dtype zero_buf[1*gIN];    
    dtype output_buf[Nums*1*gINs];

    dtype* input=input_buf;
//    dtype* zero=zero_buf;    
    dtype* output=output_buf;    

    for(int i = 0; i < Nums; i++){
        reducemax_col_rand<dtype, gIMs, gINs, tMs, tNs, tM_VLDs>(&input[i*gIMs*gINs], &output[i*gINs]);
    }
//    writeBinaryFile("/remote/lms01/q50057645/jcore_project/JanusCoreBench/test/ascpp/reducesum_colsum/src/result.bin", (uint8_t*)output, gIN * sizeof(dtype));

}

