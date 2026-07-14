#include <common/pto_tileop.hpp>

#include <cstdint>

#include "fileop.h"
#include "reduction/reducemax_colvec_pto.hpp"


#ifndef DType
#define DType int32_t
#endif


#ifndef tM
#define tM 32
#endif

#ifndef tN
#define tN 64
#endif




#define gIM 256    
#define gIN 256    
// ============================================================================
// main
// ============================================================================
int main() {
    using dtype = DType;

    dtype input_buf[gIM*gIN];
//    dtype zero_buf[1*gIN];    
    dtype output_buf[1*gIN];

    dtype* input=input_buf;
//    dtype* zero=zero_buf;    
    dtype* output=output_buf;    

//    readBinaryFile("/remote/lms01/q50057645/jcore_project/JanusCoreBench/test/ascpp/reducemax_col/src/data_256x256.bin", (uint8_t*)input, gIM * gIN * sizeof(dtype));
//    readBinaryFile("/remote/lms01/q50057645/jcore_project/JanusCoreBench/test/ascpp/reducemax_col/src/data_8192x1024.bin", (uint8_t*)input, gIM * gIN * sizeof(dtype));
//    readBinaryFile("/remote/lms01/q50057645/jcore_project/JanusCoreBench/test/ascpp/reducemax_col/src/data1x256_zero.bin", (uint8_t*)zero, gIN * sizeof(dtype));    
//    reducesum_colsum_rand<dtype, gIM, gIN, tN>(input, output);
    reducemax_col_rand<dtype, gIM, gIN, tM, tN>(input, output);
//    writeBinaryFile("/remote/lms01/q50057645/jcore_project/JanusCoreBench/test/ascpp/reducemax_col/src/result_max.bin", (uint8_t*)output, gIN * sizeof(dtype));

}

