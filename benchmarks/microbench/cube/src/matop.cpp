
#include <common/pto_tileop.hpp>

#ifndef tM
#define tM 16
#endif

#ifndef tN
#define tN 16
#endif

#ifndef tK
#define tK 16
#endif

#ifndef LOOP
#define LOOP 1000
#endif

#ifndef TEST_OP
#define TEST_OP "mul"
#endif

enum DataType {
    INT8,
    INT16,
    INT32,
    INT64,
    FP8_E4M3,
    FP8_E5M2,
    FP16,
    BF16,
    FP32
};

template<DataType DTYPE>
struct DTypeTraits;

template<>
struct DTypeTraits<INT8> {
    using src_dtype = int8_t;
    using dst_dtype = int32_t;
};

template<>
struct DTypeTraits<INT16> {
    using src_dtype = int16_t;
    using dst_dtype = int32_t;
};

template<>
struct DTypeTraits<INT32> {
    using src_dtype = int32_t;
    using dst_dtype = int32_t;
};

template<>
struct DTypeTraits<INT64> {
    using src_dtype = int64_t;
    using dst_dtype = int32_t;
};

template<>
struct DTypeTraits<FP8_E4M3> {
    using src_dtype = __fp8_e4m3;
    using dst_dtype = __fp32;
};

template<>
struct DTypeTraits<FP8_E5M2> {
    using src_dtype = __fp8_e5m2;
    using dst_dtype = __fp32;
};

template<>
struct DTypeTraits<FP16> {
    using src_dtype = __half;
    using dst_dtype = __fp32;
};

template<>
struct DTypeTraits<BF16> {
    using src_dtype = __bf16;
    using dst_dtype = __fp32;
};

template<>
struct DTypeTraits<FP32> {
    using src_dtype = __fp32;
    using dst_dtype = __fp32;
};

using src_dtype = typename DTypeTraits<DTYPE>::src_dtype;
using dst_dtype = typename DTypeTraits<DTYPE>::dst_dtype;

void matmul(const int loop, src_dtype *a, src_dtype *b) {
    using gm_shapeA = global_tensor<src_dtype, RowMajor<tM, tK>>;
    using gm_shapeB = global_tensor<src_dtype, RowMajor<tK, tN>>;

    using tile_shapeA = TileLeft<src_dtype, tM, tK>;
    using tile_shapeB = TileRight<src_dtype, tK, tN>;
    using tile_shapeACC = TileAcc<dst_dtype, tM, tN>;

    gm_shapeA gA(a);
    gm_shapeB gB(b);

    tile_shapeA tA;
    tile_shapeB tB;
    tile_shapeACC tACC;

    TCOPYIN(tA, gA);
    TCOPYIN(tB, gB);

    #pragma clang loop unroll(full)
    for(int i=0;i<LOOP;i++){
        MATMUL(tACC, tA, tB);
    }
}

void matmul_ch8(const int loop, src_dtype *a, src_dtype *b) {
    using gm_shapeA = global_tensor<src_dtype, RowMajor<tM, tK>>;
    using gm_shapeB = global_tensor<src_dtype, RowMajor<tK, tN>>;

    using tile_shapeA = TileLeft<src_dtype, tM, tK>;
    using tile_shapeB = TileRight<src_dtype, tK, tN>;
    using tile_shapeACC = TileAcc<dst_dtype, tM, tN>;

    gm_shapeA gA(a);
    gm_shapeB gB(b);

    tile_shapeACC tACC;

    tile_shapeA tA0;
    tile_shapeB tB0;

    tile_shapeA tA1;
    tile_shapeB tB1;

    tile_shapeA tA2;
    tile_shapeB tB2;

    tile_shapeA tA3;
    tile_shapeB tB3;

    tile_shapeA tA4;
    tile_shapeB tB4;

    tile_shapeA tA5;
    tile_shapeB tB5;

    tile_shapeA tA6;
    tile_shapeB tB6;

    tile_shapeA tA7;
    tile_shapeB tB7;

    TCOPYIN(tA0, gA);
    TCOPYIN(tB0, gB);
    TCOPYIN(tA1, gA);
    TCOPYIN(tB1, gB);
    TCOPYIN(tA2, gA);
    TCOPYIN(tB2, gB);
    TCOPYIN(tA3, gA);
    TCOPYIN(tB3, gB);
    TCOPYIN(tA4, gA);
    TCOPYIN(tB4, gB);
    TCOPYIN(tA5, gA);
    TCOPYIN(tB5, gB);
    TCOPYIN(tA6, gA);
    TCOPYIN(tB6, gB);
    TCOPYIN(tA7, gA);
    TCOPYIN(tB7, gB);

    #pragma clang loop unroll(full)
    for(int i=0;i<LOOP/2;i++){
        MATMUL(tACC, tA0, tB0);
        MATMUL(tACC, tA1, tB1);
        MATMUL(tACC, tA2, tB2);
        MATMUL(tACC, tA3, tB3);
        MATMUL(tACC, tA4, tB4);
        MATMUL(tACC, tA5, tB5);
        MATMUL(tACC, tA6, tB6);
        MATMUL(tACC, tA7, tB7);
    }
}

void matmacc(const int loop, src_dtype *a, src_dtype *b){
    using gm_shapeA = global_tensor<src_dtype, RowMajor<tM, tK>>;
    using gm_shapeB = global_tensor<src_dtype, RowMajor<tK, tN>>;

    using tile_shapeA = TileLeft<src_dtype, tM, tK>;
    using tile_shapeB = TileRight<src_dtype, tK, tN>;
    using tile_shapeACC = TileAcc<dst_dtype, tM, tN>;

    gm_shapeA gA(a);
    gm_shapeB gB(b);
    
    tile_shapeA tA;
    tile_shapeB tB;
    tile_shapeACC tACC;
    
    TCOPYIN(tA, gA);
    TCOPYIN(tB, gB);
    MATMUL(tACC, tA, tB);

    #pragma clang loop unroll(full)
    for(int i=0;i<LOOP;i++){
        MATMACC(tACC, tA, tB);
    }
}

void matmacc_ch8(const int loop, src_dtype *a, src_dtype *b) {
    using gm_shapeA = global_tensor<src_dtype, RowMajor<tM, tK>>;
    using gm_shapeB = global_tensor<src_dtype, RowMajor<tK, tN>>;

    using tile_shapeA = TileLeft<src_dtype, tM, tK>;
    using tile_shapeB = TileRight<src_dtype, tK, tN>;
    using tile_shapeACC = TileAcc<dst_dtype, tM, tN>;

    gm_shapeA gA(a);
    gm_shapeB gB(b);

    tile_shapeACC tACC;

    tile_shapeA tA0;
    tile_shapeB tB0;

    tile_shapeA tA1;
    tile_shapeB tB1;

    tile_shapeA tA2;
    tile_shapeB tB2;

    tile_shapeA tA3;
    tile_shapeB tB3;

    tile_shapeA tA4;
    tile_shapeB tB4;

    tile_shapeA tA5;
    tile_shapeB tB5;

    tile_shapeA tA6;
    tile_shapeB tB6;

    tile_shapeA tA7;
    tile_shapeB tB7;

    TCOPYIN(tA0, gA);
    TCOPYIN(tB0, gB);
    TCOPYIN(tA1, gA);
    TCOPYIN(tB1, gB);
    TCOPYIN(tA2, gA);
    TCOPYIN(tB2, gB);
    TCOPYIN(tA3, gA);
    TCOPYIN(tB3, gB);
    TCOPYIN(tA4, gA);
    TCOPYIN(tB4, gB);
    TCOPYIN(tA5, gA);
    TCOPYIN(tB5, gB);
    TCOPYIN(tA6, gA);
    TCOPYIN(tB6, gB);
    TCOPYIN(tA7, gA);
    TCOPYIN(tB7, gB);
    MATMUL(tACC, tA0, tB0);
    MATMUL(tACC, tA1, tB1);
    MATMUL(tACC, tA2, tB2);
    MATMUL(tACC, tA3, tB3);
    MATMUL(tACC, tA4, tB4);
    MATMUL(tACC, tA5, tB5);
    MATMUL(tACC, tA6, tB6);
    MATMUL(tACC, tA7, tB7);

    #pragma clang loop unroll(full)
    for(int i=0;i<LOOP/2;i++){
        MATMACC(tACC, tA0, tB0);
        MATMACC(tACC, tA1, tB1);
        MATMACC(tACC, tA2, tB2);
        MATMACC(tACC, tA3, tB3);
        MATMACC(tACC, tA4, tB4);
        MATMACC(tACC, tA5, tB5);
        MATMACC(tACC, tA6, tB6);
        MATMACC(tACC, tA7, tB7);
    }
}

int main(){
    src_dtype a[tM*tK];
    src_dtype b[tK*tN];

    if(!strcmp(TEST_OP, "mul")){
        matmul(LOOP, a, b);
    }
    else if(!strcmp(TEST_OP, "macc")){
        matmacc(LOOP, a, b);
    }
    else if(!strcmp(TEST_OP, "mul_ch8")){
        matmul_ch8(LOOP, a, b);
    }
    else if(!strcmp(TEST_OP, "macc_ch8")){
        matmacc_ch8(LOOP, a, b);
    }
    return 0;
}