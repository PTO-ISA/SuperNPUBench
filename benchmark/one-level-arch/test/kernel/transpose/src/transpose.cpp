#include <common/pto_tileop.hpp>
//#include "template_asm.h"

#include <cstdint>
#include <cstdio>

#include "fileop.h"
#include "transpose/transpose_pto.hpp"


#ifndef DType
#define DType int32_t
#endif

#ifndef tMs
#define tMs 1024
#endif

#ifndef MAX_DIMs
#define MAX_DIMs 8
#endif

//#define IN_SHAPE {2, 128, 64}
//#define OUT_SHAPE {2, 64, 128}
//#define IN_SHAPE {8, 1024, 2048}
//#define OUT_SHAPE {8, 2048, 1024}
//#define IN_SHAPE {64, 8, 4096, 3}
//#define OUT_SHAPE {64, 8, 3, 4096}
#ifndef IN_SHAPEs
#define IN_SHAPEs 64,8,64,4,16,7
#endif

#ifndef OUT_SHAPEs
#define OUT_SHAPEs 64,8,16,4,64,7
#endif
//#define IN_SHAPE {64, 8, 64, 4, 128}
//#define OUT_SHAPE {64, 8, 128, 4, 64}

//#define IN_DIM 3
//#define OUT_DIM 3
//#define IN_DIM 4
//#define OUT_DIM 4

#ifndef IN_DIMs
#define IN_DIMs 6
#endif

#ifndef OUT_DIMs
#define OUT_DIMs 6
#endif
//#define IN_DIM 5
//#define OUT_DIM 5


//#define TRANSPOSE_DIM1 1
//#define TRANSPOSE_DIM0 2
//#define TRANSPOSE_DIM1 2
//#define TRANSPOSE_DIM0 3
#ifndef TRANSPOSE_DIM1s
#define TRANSPOSE_DIM1s 2
#endif

#ifndef TRANSPOSE_DIM0s
#define TRANSPOSE_DIM0s 4
#endif
//#define TRANSPOSE_DIM1 2
//#define TRANSPOSE_DIM0 4

//#define gIM (2 * 128 * 64)    
//#define gOM (2 * 64 * 128)    
//#define gIM (8 * 1024 * 2048)    
//#define gOM (8 * 2048 * 1024)    
//#define gIM (64 * 8 * 4096 * 3)    
//#define gOM (64 * 8 * 3 * 4096)    

#ifndef gIMs
#define gIMs (64 * 8 * 64 * 4 * 16 * 7) 
#endif

#ifndef gOMs
#define gOMs (64 * 8 * 16 * 4 * 64 * 7)
#endif
//#define gIM (64 * 8 * 64 * 4 * 128)    
//#define gOM (64 * 8 * 128 * 4 * 64)  



// ============================================================================
// main
// ============================================================================




int main() {
    using dtype = DType;
    uint32_t in_shape[128] = {IN_SHAPEs};
    uint32_t out_shape[128] = {OUT_SHAPEs};
    size_t in_dim = IN_DIMs;
    size_t out_dim = OUT_DIMs;
//    size_t transpose_dim1 = TRANSPOSE_DIM1;
//    size_t transpose_dim0 = TRANSPOSE_DIM0;

    dtype input_buf[gIMs];
    dtype output_buf[gOMs];


    dtype* input  = input_buf;
    dtype* output = output_buf;


//    readBinaryFile("/remote/lms01/q50057645/jcore_project/JanusCoreBench/test/ascpp/transpose/src/data_2x128x64.bin", (uint8_t*)input, gIM * sizeof(dtype));
//    readBinaryFile("/remote/lms01/q50057645/jcore_project/JanusCoreBench/test/ascpp/transpose/src/data_8x1024x2048.bin", (uint8_t*)input, gIM * sizeof(dtype));
//    readBinaryFile("/remote/lms01/q50057645/jcore_project/JanusCoreBench/test/ascpp/transpose/src/data_64x8x4096x3.bin", (uint8_t*)input, gIM * sizeof(dtype)); 
//    readBinaryFile("/remote/lms01/q50057645/jcore_project/JanusCoreBench/test/ascpp/transpose/src/data_64x8x64x4x16x7.bin", (uint8_t*)input, gIMs * sizeof(dtype));
//    readBinaryFile("/remote/lms01/q50057645/jcore_project/JanusCoreBench/test/ascpp/transpose/src/data_2x2x4x4x8x4.bin", (uint8_t*)input, gIMs * sizeof(dtype)); 
//    readBinaryFile("/remote/lms01/q50057645/jcore_project/JanusCoreBench/test/ascpp/transpose/src/data_2x2x4x4x8.bin", (uint8_t*)input, gIMs * sizeof(dtype));           
//    readBinaryFile("/remote/lms01/q50057645/jcore_project/JanusCoreBench/test/ascpp/transpose/src/data_64x8x64x4x128.bin", (uint8_t*)input, gIM * sizeof(dtype));       
    transpose<dtype, MAX_DIMs, gIMs, gOMs, tMs, IN_DIMs, OUT_DIMs, TRANSPOSE_DIM1s, TRANSPOSE_DIM0s>(input, output, in_shape, out_shape);
//    transpose<dtype, MAX_DIM, gIM, gOM, tM>(input, output);
//    writeBinaryFile("/remote/lms01/q50057645/jcore_project/JanusCoreBench/test/ascpp/transpose/src/result_64x8x4096x3.bin", (uint8_t*)output, gOM * sizeof(dtype));
 //   writeBinaryFile("/remote/lms01/q50057645/jcore_project/JanusCoreBench/test/ascpp/transpose/src/result_2x2x4x4x8x4.bin", (uint8_t*)output, gOMs * sizeof(dtype));
//    writeBinaryFile("/remote/lms01/q50057645/jcore_project/JanusCoreBench/test/ascpp/transpose/src/result_2x2x4x4x8.bin", (uint8_t*)output, gOMs * sizeof(dtype));    
//    writeBinaryFile("/remote/lms01/q50057645/jcore_project/JanusCoreBench/test/ascpp/transpose/src/result_64x8x64x4x16x7.bin", (uint8_t*)output, gOMs * sizeof(dtype));
//    writeBinaryFile("/remote/lms01/q50057645/jcore_project/JanusCoreBench/test/ascpp/transpose/src/result_64x8x64x4x128.bin", (uint8_t*)output, gOM * sizeof(dtype));    
//    printf("output");


}

