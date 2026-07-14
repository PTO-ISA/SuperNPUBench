#include <common/pto_tileop.hpp>
//#include "template_asm.h"

#include <cstdint>
#include <cstdio>

#include "fileop.h"
#include "concat/concat_scatter_pto.hpp"


#ifndef DType
#define DType int32_t
#endif

#ifndef tMs
#define tMs 1024
#endif

#ifndef MAX_DIMs
#define MAX_DIMs 8
#endif

#ifndef NUMs
#define NUMs 1000
#endif

#ifndef IN_SHAPEs
#define IN_SHAPEs 64,2
#endif

#ifndef OUT_SHAPEs
#define OUT_SHAPEs 64,2000
#endif
//#define IN_SHAPE {64, 8, 64, 4, 128}
//#define OUT_SHAPE {64, 8, 128, 4, 64}

//#define IN_DIM 3
//#define OUT_DIM 3
//#define IN_DIM 4
//#define OUT_DIM 4

#ifndef DATA_DIMs
#define DATA_DIMs 2
#endif

#ifndef CONCAT_DIMs
#define CONCAT_DIMs 1
#endif
//#define IN_DIM 5
//#define OUT_DIM 5


#ifndef gIMs
#define gIMs (1000*64*2) 
#endif

#ifndef gOMs
#define gOMs (64*2000)
#endif
//#define gIM (64 * 8 * 64 * 4 * 128)    
//#define gOM (64 * 8 * 128 * 4 * 64)  



// ============================================================================
// main
// ============================================================================




int main() {
    using dtype = DType;
    size_t in_shape[MAX_DIMs] = {IN_SHAPEs};
    size_t out_shape[MAX_DIMs] = {OUT_SHAPEs};
//    size_t in_dim = IN_DIMs;
//    size_t out_dim = OUT_DIMs;
//    size_t transpose_dim1 = TRANSPOSE_DIM1;
//    size_t transpose_dim0 = TRANSPOSE_DIM0;

    dtype input_buf[gIMs];
    dtype output_buf[gOMs];


    dtype* input  = input_buf;
    dtype* output = output_buf;



//    readBinaryFile("/remote/lms01/q50057645/jcore_project/JanusCoreBench/test/ascpp/concat_scatter/src/data_1000_64_2.bin", (uint8_t*)input, gIMs * sizeof(dtype));       
    concat_scatter<dtype, MAX_DIMs, gIMs, gOMs, tMs, DATA_DIMs, CONCAT_DIMs>(input, output, in_shape, out_shape);
//    writeBinaryFile("/remote/lms01/q50057645/jcore_project/JanusCoreBench/test/ascpp/concat_scatter/src/result.bin", (uint8_t*)output, gOMs * sizeof(dtype));
 


}

