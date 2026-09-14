/**
 * Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_mx_quant_tail_axis.h
 * \brief
 */

#ifndef DYNAMIC_MX_QUANT_TAIL_AXIS_H
#define DYNAMIC_MX_QUANT_TAIL_AXIS_H
#define FLOAT_OVERFLOW_MODE_CTRL 60
#include "kernel_operator.h"
#include "dynamic_mx_quant_common.h"
namespace DynamicMxQuant {
using namespace AscendC;

template <typename T, typename U, int64_t SCALE_ALG>
class DynamicMxQuantTailAxis {
public:
    __aicore__ inline DynamicMxQuantTailAxis() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR mxScale,
                                const DynamicMxQuantTailAxisTilingData* tilingData);
    __aicore__ inline void Process();

private:
    using intCalcType = typename std::conditional<IsSame<T, float>::value, uint32_t, uint16_t>::type;
    __aicore__ inline void ParseTilingData(const DynamicMxQuantTailAxisTilingData* tilingData);
    __aicore__ inline void GetGmParams();
    __aicore__ inline void GetUbParams();

    __aicore__ inline void CopyIn(int64_t dim0LoopIdx, int64_t dim1LoopIdx, int64_t ubFactorRowNum,
                                  int64_t ubFactorColNum, int64_t ubFactorColBlockNum);
    __aicore__ inline void Compute(int64_t ubFactorRowBlockNum, int64_t ubFactorColBlockNum);
    __aicore__ inline void CopyOut(int64_t dim0LoopIdx, int64_t dim1LoopIdx, int64_t ubFactorRowNum,
                                   int64_t ubFactorColNum, int64_t ubFactorColBlockNum);

    // SCALE_ALG==0: OCP实现
    __aicore__ inline void ComputeMaxExpOcpHalf(__ubuf__ half* xLocalAddr, __ubuf__ uint16_t* maxExpAddr,
                                                uint16_t loopNum2VF);
    __aicore__ inline void ComputeMaxExpOcpBf16(__ubuf__ bfloat16_t* xLocalAddr, __ubuf__ uint16_t* maxExpAddr,
                                                uint16_t loopNum2VF);
    __aicore__ inline void ComputeMaxExpOcpFp32(__ubuf__ float* xLocalAddr, __ubuf__ uint16_t* maxExpAddr,
                                                uint16_t loopNum2VF);

    // SCALE_ALG==1: Cublas实现
    __aicore__ inline void ComputeMaxExpCublasHalf(__ubuf__ half* xLocalAddr, __ubuf__ uint16_t* maxExpAddr,
                                                   uint16_t loopNum2VF);
    __aicore__ inline void ComputeMaxExpCublasBf16(__ubuf__ bfloat16_t* xLocalAddr, __ubuf__ uint16_t* maxExpAddr,
                                                   uint16_t loopNum2VF);
    __aicore__ inline void ComputeMaxExpCublasFp32(__ubuf__ float* xLocalAddr, __ubuf__ uint32_t* maxExpAddr,
                                                   uint16_t loopNum2VF);

    // SCALE_ALG==2: dst_value_max + Cublas实现
    __aicore__ inline void ComputeMaxExpDynDtypeRangeHalf(__ubuf__ half* xLocalAddr, __ubuf__ uint16_t* maxExpAddr,
                                                          uint16_t loopNum2VF);
    __aicore__ inline void ComputeMaxExpDynDtypeRangeBf16(__ubuf__ bfloat16_t* xLocalAddr,
                                                          __ubuf__ uint16_t* maxExpAddr, uint16_t loopNum2VF);
    __aicore__ inline void ComputeMaxExpDynDtypeRangeFp32(__ubuf__ float* xLocalAddr, __ubuf__ uint16_t* maxExpAddr,
                                                          uint16_t loopNum2VF);

    __aicore__ inline void ComputeScaleOcp(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr,
                                           __ubuf__ uint16_t* recipScaleLocalAddr, uint16_t loopNum1VF,
                                           uint32_t totalScaleInUB);
    __aicore__ inline void ComputeScaleDynamicDtypeRange(__ubuf__ uint16_t* maxExpAddr,
                                                         __ubuf__ uint16_t* mxScaleLocalAddr,
                                                         __ubuf__ uint16_t* recipScaleLocalAddr, uint16_t loopNum1VF,
                                                         uint32_t totalScaleInUB);
    __aicore__ inline void ComputeScaleCublas(__ubuf__ intCalcType* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr,
                                              __ubuf__ uint16_t* recipScaleLocalAddr, uint16_t loopNumHalfVF,
                                              uint32_t totalScaleInUB);

    template <RoundMode toBf16RoundMode, RoundMode roundMode>
    __aicore__ inline void ComputeDataHalf(__ubuf__ half* xLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr,
                                           __ubuf__ int8_t* yLocalAddr, uint16_t loopNum2VF);
    template <RoundMode toBf16RoundMode, RoundMode roundMode>
    __aicore__ inline void ComputeDataBf16(__ubuf__ bfloat16_t* xLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr,
                                           __ubuf__ int8_t* yLocalAddr, uint16_t loopNum2VF);
    template <RoundMode toBf16RoundMode, RoundMode roundMode>

    __aicore__ inline void ComputeDataOptimizeHalf(__ubuf__ half* xLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr,
                                                   __ubuf__ int8_t* yLocalAddr, uint16_t loopNum2VF);
    template <RoundMode toBf16RoundMode, RoundMode roundMode>
    __aicore__ inline void ComputeDataOptimizeBf16(__ubuf__ bfloat16_t* xLocalAddr,
                                                   __ubuf__ uint16_t* recipScaleLocalAddr, __ubuf__ int8_t* yLocalAddr,
                                                   uint16_t loopNum2VF);
    // FP32 FP4 tail-axis: reads float directly from xLocalAddr, bypasses bf16Scratch to avoid VEC_ERROR on non-aligned
    // colNum
    template <RoundMode toBf16RoundMode, RoundMode roundMode>
    __aicore__ inline void ComputeDataFloatToFP4(__ubuf__ float* xLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr,
                                                 __ubuf__ int8_t* yLocalAddr, uint16_t loopNum2VF);
    __aicore__ inline void ComputeFP4FromHalf(Reg::RegTensor<half>& output, Reg::RegTensor<half>& input,
                                              Reg::MaskReg& mask);
    template <RoundMode toBf16RoundMode, RoundMode roundMode>
    __aicore__ inline void ComputeFP4FromFp32(Reg::RegTensor<float>& Reg);

    __aicore__ inline void ComputeOcp(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* maxExpLocalAddr,
                                      __ubuf__ uint16_t* scaleLocalAddr, __ubuf__ int8_t* yLocalAddr,
                                      __ubuf__ uint16_t* recipScaleLocalAddr, uint16_t loopNum2VF, uint16_t loopNum1VF,
                                      uint16_t loopNumHalfVF, uint32_t totalBlockNum);

    __aicore__ inline void ComputeCublasFp4(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* maxExpLocalAddr,
                                            __ubuf__ uint16_t* scaleLocalAddr, __ubuf__ int8_t* yLocalAddr,
                                            __ubuf__ uint16_t* recipScaleLocalAddr, uint16_t loopNum2VF,
                                            uint16_t loopNum1VF, uint16_t loopNumHalfVF, uint32_t totalBlockNum);

    __aicore__ inline void ComputeCublasFp4Opti(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* maxExpLocalAddr,
                                                __ubuf__ uint16_t* scaleLocalAddr, __ubuf__ int8_t* yLocalAddr,
                                                __ubuf__ uint16_t* recipScaleLocalAddr, uint16_t loopNum2VF,
                                                uint16_t loopNum1VF, uint16_t loopNumHalfVF, uint32_t totalBlockNum);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, DB_BUFFER> inQueue_;
    GlobalTensor<T> xGm_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue_;
    GlobalTensor<uint8_t> yGm_;
    TQue<QuePosition::VECOUT, DB_BUFFER> mxScaleQueue_;
    GlobalTensor<uint8_t> scaleGm_;

    TBuf<QuePosition::VECCALC> maxExpBuffer_;
    TBuf<QuePosition::VECCALC> recipScaleBuffer_;

    int64_t roundMode_ = 0;
    int64_t blockSize_ = 0;
    int64_t totalCoreNum_ = 0;
    int64_t usedCoreNum_ = 0;

    int64_t rowTileNum_ = 0;        // row 方向上的切核数
    int64_t colTileNum_ = 0;        // col 方向上的切核数
    int64_t rowNum_ = 1;            // 合轴之后 -2 轴大小
    int64_t colNum_ = 1;            // 合轴之后 -1 轴大小
    int64_t colNormalBlockNum_ = 0; // 列方向头核处理的块数 (1 x 256)
    int64_t colTailLen_ = 0;        // 列方向尾块长度
    int64_t rowNormalBlockNum_ = 0; // 行方向头核处理的块数 (1 行)
    int64_t rowTailLen_ = 0;        // 行方向尾块长度
    int64_t maxUbBlockNum_ = 0;     // UB最大能放下的处理块数 (1 x 32) (8 的倍数)
    float dstTypeMax_ = 0.0;
    float invDstTypeMax_ = 0.0;

    // GM Params
    int64_t coreIdx_ = 0;       // Core ID
    int64_t coreColIdx_ = 0;    // Core 处理的 GM 数据块列方向的核ID数
    int64_t coreRowIdx_ = 0;    // Core 处理的 GM 数据块行方向的核ID数
    int64_t xGmOffset_ = 0;     // Core 处理的数据块在 GM 的起始地址偏移
    int64_t scaleGmOffset_ = 0; // Core 处理的数据块得到的Scale在 GM 的地址偏移
    int64_t scaleColNum_ = 0;   // 合轴后数据块在尾轴方向上的 scale 数量大小（偶数对齐）

    // UB Params
    int64_t ubFactorColNum_ = 0;            // Core 处理的 GM 数据块列方向元素个数
    int64_t ubFactorCol32BlockNum_ = 0;     // Core 处理的 GM 数据块列方向 32 长度块个数
    int64_t ubFactorRow1BlockNum_ = 0;      // Core 处理的 GM 数据块行方向 1 长度块个数
    int64_t ubFactorColLoopNum_ = 0;        // Core 处理的 GM 数据块列方向循环次数
    int64_t ubFactorRowLoopNum_ = 0;        // Core 处理的 GM 数据块行方向循环次数
    int64_t ubFactorColNormalBlockNum_ = 0; // Core 处理的 GM 数据块搬入 UB 列方向 Normal 32 长度数据块
    int64_t ubFactorColTailBlockNum_ = 0;   // Core 处理的 GM 数据块搬入 UB 列方向 Tail 32 长度数据块
    int64_t ubFactorColNormalBlockLen_ = 0; // Core 处理的 GM 数据块搬入 UB 列方向 Normal 数据块元素个数
    int64_t ubFactorColTailBlockLen_ = 0;   // Core 处理的 GM 数据块搬入 UB 列方向 Tail 数据块元素个数
    int64_t ubFactorRowNormalBlockNum_ = 0; // Core 处理的 GM 数据块搬入 UB 行方向 Normal 1 长度数据块
    int64_t ubFactorRowTailBlockNum_ = 0;   // Core 处理的 GM 数据块搬入 UB 行方向 Tail 1 长度数据块

    uint16_t f4Emax_ = 0;
    uint16_t addValueBit_ = 0;
};

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR mxScale,
                                                                     const DynamicMxQuantTailAxisTilingData* tilingData)
{
#if (__NPU_ARCH__ == 3510)
    SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    ParseTilingData(tilingData); // 获取TilingData数据

    GetGmParams(); // 计算核间GM地址偏移

    GetUbParams(); // 计算核内切分参数

    int64_t scaleBufferBlockNum = Ops::Base::CeilAlign(maxUbBlockNum_, static_cast<int64_t>(VF_LEN_16));
    pipe_.InitBuffer(inQueue_, DB_BUFFER, maxUbBlockNum_ * blockSize_ * sizeof(T));
    pipe_.InitBuffer(outQueue_, DB_BUFFER, maxUbBlockNum_ * blockSize_ * sizeof(uint8_t) / DIGIT_TWO);
    pipe_.InitBuffer(mxScaleQueue_, DB_BUFFER, scaleBufferBlockNum * sizeof(uint8_t));
    pipe_.InitBuffer(maxExpBuffer_, maxUbBlockNum_ * sizeof(T) * 2);
    pipe_.InitBuffer(recipScaleBuffer_, maxUbBlockNum_ * sizeof(uint16_t) * 2);

    xGm_.SetGlobalBuffer((__gm__ T*)x + xGmOffset_);
    yGm_.SetGlobalBuffer((__gm__ uint8_t*)y + xGmOffset_ / DIGIT_TWO);
    scaleGm_.SetGlobalBuffer((__gm__ uint8_t*)mxScale + scaleGmOffset_);

    if constexpr (IsSame<U, fp4x2_e2m1_t>::value) {
        f4Emax_ = FP4_E2M1_BF16_MAX_EXP;
    } else {
        f4Emax_ = FP4_E1M2_BF16_MAX_EXP;
    }

    if (dstTypeMax_ == DIGIT_ZERO_FLOAT || dstTypeMax_ == DIGIT_SIX_FLOAT) {
        addValueBit_ = BF16_ADD_VALUE_MAN1;
    } else if (dstTypeMax_ == DIGIT_SEVEN_FLOAT) {
        addValueBit_ = BF16_ADD_VALUE_MAN2;
    }
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::GetGmParams()
{
    coreIdx_ = GetBlockIdx();
    coreColIdx_ = coreIdx_ % colTileNum_;
    coreRowIdx_ = coreIdx_ / colTileNum_;
    xGmOffset_ = coreRowIdx_ * rowNormalBlockNum_ * DIGIT_ONE * colNum_ +
                 coreColIdx_ * colNormalBlockNum_ * DIGIT_EIGHT * blockSize_;
    scaleColNum_ = ops::CeilDiv(ops::CeilDiv(colNum_, blockSize_), DIGIT_TWO) * DIGIT_TWO;
    scaleGmOffset_ = coreRowIdx_ * rowNormalBlockNum_ * DIGIT_ONE * scaleColNum_ +
                     coreColIdx_ * colNormalBlockNum_ * DIGIT_EIGHT;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::GetUbParams()
{
    if (coreColIdx_ == colTileNum_ - 1) {
        ubFactorCol32BlockNum_ = ops::CeilDiv(colTailLen_, blockSize_);
        ubFactorColNum_ = colTailLen_;
    } else {
        ubFactorCol32BlockNum_ = colNormalBlockNum_ * DIGIT_EIGHT;
        ubFactorColNum_ = ubFactorCol32BlockNum_ * blockSize_;
    }

    if (coreRowIdx_ == rowTileNum_ - 1) {
        ubFactorRow1BlockNum_ = ops::CeilDiv(rowTailLen_, DIGIT_ONE);
    } else {
        ubFactorRow1BlockNum_ = rowNormalBlockNum_ * DIGIT_ONE;
    }

    ubFactorColLoopNum_ = ops::CeilDiv(ubFactorCol32BlockNum_,
                                       maxUbBlockNum_); // 列方向上UB循环次数（列方向上的切 UB 块数）
    ubFactorColNormalBlockNum_ = ops::CeilDiv(ubFactorCol32BlockNum_,
                                              ubFactorColLoopNum_); // 列方向上头块的Block数量（均衡处理） 用于VF内处理
    ubFactorColNormalBlockNum_ = ops::CeilDiv(ubFactorColNormalBlockNum_, DIGIT_TWO) *
                                 DIGIT_TWO; // 列方向上头块的Block数量补成偶数，只有尾块有可能需要补Pad
    ubFactorColTailBlockNum_ = ubFactorCol32BlockNum_ -
                               (ubFactorColLoopNum_ - DIGIT_ONE) *
                                   ubFactorColNormalBlockNum_; // 列方向上尾块的Block数量（只有一个尾块） 用于VF内处理
    ubFactorColNormalBlockLen_ = ubFactorColNormalBlockNum_ *
                                 blockSize_; // 列方向上头块的元素数量（均衡处理）           用于数据搬运
    ubFactorColTailBlockLen_ = ubFactorColNum_ -
                               (ubFactorColLoopNum_ - DIGIT_ONE) *
                                   ubFactorColNormalBlockLen_; // 列方向上尾块的元素数量（只有一个尾块） 用于数据搬运

    ubFactorRowNormalBlockNum_ = ops::FloorDiv(maxUbBlockNum_, ubFactorColNormalBlockNum_); // UB 一次至多载入的行数
    ubFactorRowLoopNum_ = ops::CeilDiv(ubFactorRow1BlockNum_,
                                       ubFactorRowNormalBlockNum_); // 行方向上 UB 循环次数（行方向上的切 UB 块数）
    ubFactorRowNormalBlockNum_ = ops::CeilDiv(ubFactorRow1BlockNum_,
                                              ubFactorRowLoopNum_); // 行方向上的均衡处理 用于VF内处理
    ubFactorRowTailBlockNum_ = ubFactorRow1BlockNum_ -
                               (ubFactorRowLoopNum_ - DIGIT_ONE) *
                                   ubFactorRowNormalBlockNum_; // 行方向上尾块的行数（只有一个尾块） 用于VF内处理
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ParseTilingData(
    const DynamicMxQuantTailAxisTilingData* tilingData)
{
    roundMode_ = tilingData->roundMode;
    blockSize_ = tilingData->blockSize;
    totalCoreNum_ = tilingData->totalCoreNum;
    usedCoreNum_ = tilingData->usedCoreNum;
    rowTileNum_ = tilingData->rowTileNum;
    colTileNum_ = tilingData->colTileNum;
    rowNum_ = tilingData->rowNum;
    colNum_ = tilingData->colNum;
    colNormalBlockNum_ = tilingData->colNormalBlockNum;
    colTailLen_ = tilingData->colTailLen;
    rowNormalBlockNum_ = tilingData->rowNormalBlockNum;
    rowTailLen_ = tilingData->rowTailLen;
    maxUbBlockNum_ = tilingData->maxUbBlockNum;
    dstTypeMax_ = tilingData->dstTypeMax;
    invDstTypeMax_ = tilingData->invDstTypeMax;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::Process()
{
    if (coreIdx_ >= usedCoreNum_) {
        return;
    }
    int64_t dim0LoopIdx = 0;
    int64_t dim1LoopIdx = 0;
    for (dim0LoopIdx = 0; dim0LoopIdx < ubFactorRowLoopNum_ - 1; dim0LoopIdx++) {
        for (dim1LoopIdx = 0; dim1LoopIdx < ubFactorColLoopNum_ - 1; dim1LoopIdx++) {
            CopyIn(dim0LoopIdx, dim1LoopIdx, ubFactorRowNormalBlockNum_, ubFactorColNormalBlockLen_,
                   ubFactorColNormalBlockNum_);
            Compute(ubFactorRowNormalBlockNum_, ubFactorColNormalBlockNum_);
            CopyOut(dim0LoopIdx, dim1LoopIdx, ubFactorRowNormalBlockNum_, ubFactorColNormalBlockLen_,
                    ubFactorColNormalBlockNum_);
        }
        CopyIn(dim0LoopIdx, dim1LoopIdx, ubFactorRowNormalBlockNum_, ubFactorColTailBlockLen_,
               ubFactorColTailBlockNum_);
        Compute(ubFactorRowNormalBlockNum_, ubFactorColTailBlockNum_);
        CopyOut(dim0LoopIdx, dim1LoopIdx, ubFactorRowNormalBlockNum_, ubFactorColTailBlockLen_,
                ubFactorColTailBlockNum_);
    }
    for (dim1LoopIdx = 0; dim1LoopIdx < ubFactorColLoopNum_ - 1; dim1LoopIdx++) {
        CopyIn(dim0LoopIdx, dim1LoopIdx, ubFactorRowTailBlockNum_, ubFactorColNormalBlockLen_,
               ubFactorColNormalBlockNum_);
        Compute(ubFactorRowTailBlockNum_, ubFactorColNormalBlockNum_);
        CopyOut(dim0LoopIdx, dim1LoopIdx, ubFactorRowTailBlockNum_, ubFactorColNormalBlockLen_,
                ubFactorColNormalBlockNum_);
    }
    CopyIn(dim0LoopIdx, dim1LoopIdx, ubFactorRowTailBlockNum_, ubFactorColTailBlockLen_, ubFactorColTailBlockNum_);
    Compute(ubFactorRowTailBlockNum_, ubFactorColTailBlockNum_);
    CopyOut(dim0LoopIdx, dim1LoopIdx, ubFactorRowTailBlockNum_, ubFactorColTailBlockLen_, ubFactorColTailBlockNum_);
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::CopyIn(int64_t dim0LoopIdx, int64_t dim1LoopIdx,
                                                                       int64_t ubFactorRowNum, int64_t ubFactorColNum,
                                                                       int64_t ubFactorColBlockNum)
{
    int64_t scaleIsNotOdd = ubFactorColBlockNum % DIGIT_TWO;

    LocalTensor<T> xLocal = inQueue_.AllocTensor<T>();

    DataCopyExtParams copyInParam;
    DataCopyPadExtParams<T> padParams;
    if constexpr (IsSame<T, float>::value) {
        Duplicate<T>(xLocal, static_cast<T>(0), maxUbBlockNum_ * blockSize_);
        event_t eventIDVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);

        int64_t blockSizeDouble = 64;
        int64_t padRightNum = (ubFactorColNum % FLOAT_PER_UB_BLOCK == 0) ?
                                  0 :
                                  FLOAT_PER_UB_BLOCK - ubFactorColNum % FLOAT_PER_UB_BLOCK;
        int64_t colNumUbBlockAligned = Ops::Base::CeilAlign(ubFactorColNum, FLOAT_PER_UB_BLOCK);
        int64_t dstStride = colNumUbBlockAligned % blockSizeDouble == 0 ?
                                0 :
                                (blockSizeDouble - colNumUbBlockAligned % blockSizeDouble) / FLOAT_PER_UB_BLOCK;
        copyInParam = {static_cast<uint16_t>(ubFactorRowNum), static_cast<uint32_t>(ubFactorColNum * sizeof(T)),
                       static_cast<uint32_t>((colNum_ - ubFactorColNum) * sizeof(T)), static_cast<uint32_t>(dstStride),
                       static_cast<uint32_t>(0)};
        padParams = {true, static_cast<uint8_t>(0), static_cast<uint8_t>(padRightNum), static_cast<T>(0)};
    } else {
        if (scaleIsNotOdd != 0) {
            // 当前 UB 块一行的scale数不是偶数时，需要将X的Buffer空间预先填充为0
            Duplicate<T>(xLocal, static_cast<T>(0), maxUbBlockNum_ * blockSize_);
            event_t eventIDVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
            WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
        }

        // 搬运X，补Pad示例
        // Eg1: dataLen = 1  => leftPad = [0 * sizeof(T)] Bytes , rightPad = [16 * sizeof(T)] Bytes  , dummy = [15 *
        // sizeof(T) + 2 * 16 * sizeof(T)] Bytes Eg2: dataLen = 17 => leftPad = [0 * sizeof(T)] Bytes , rightPad = [15 *
        // sizeof(T)] Bytes  , dummy = [ 0 * sizeof(T) + 2 * 16 * sizeof(T)] Bytes Eg3: dataLen = 32 => leftPad = [0 *
        // sizeof(T)] Bytes , rightPad = [ 0 * sizeof(T)] Bytes  , dummy = [ 0 * sizeof(T) + 2 * 16 * sizeof(T)] Bytes
        // Eg4: dataLen = 35 => leftPad = [0 * sizeof(T)] Bytes , rightPad = [16 * sizeof(T)] Bytes  , dummy = [13 *
        // sizeof(T) + 0 * 16 * sizeof(T)] Bytes
        // 当补的数(rightPad)是16时，一定能补到UB的下一个DataBlock，实现补到32个数对齐的功能
        int64_t padRightNumTmp = (ubFactorColNum % 32 > 16) ? (32 - ubFactorColNum % 32) : 16;
        int64_t padRightNum = (ubFactorColNum % 32 == 0) ? 0 : padRightNumTmp;
        copyInParam = {static_cast<uint16_t>(ubFactorRowNum), static_cast<uint32_t>(ubFactorColNum * sizeof(T)),
                       static_cast<uint32_t>((colNum_ - ubFactorColNum) * sizeof(T)),
                       static_cast<uint32_t>(scaleIsNotOdd * sizeof(T)), static_cast<uint32_t>(0)};
        padParams = {true, static_cast<uint8_t>(0), static_cast<uint8_t>(padRightNum), static_cast<T>(0)};
    }
    int64_t offset = dim0LoopIdx * ubFactorRowNormalBlockNum_ * colNum_ + dim1LoopIdx * ubFactorColNormalBlockLen_;

    DataCopyPad(xLocal, xGm_[offset], copyInParam, padParams);
    inQueue_.EnQue(xLocal);
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::CopyOut(int64_t dim0LoopIdx, int64_t dim1LoopIdx,
                                                                        int64_t ubFactorRowNum, int64_t ubFactorColNum,
                                                                        int64_t ubFactorColBlockNum)
{
    LocalTensor<uint8_t> scaleLocal = mxScaleQueue_.DeQue<uint8_t>();
    LocalTensor<uint8_t> yLocal = outQueue_.DeQue<uint8_t>();
    int64_t scaleIsNotOdd = ubFactorColBlockNum % DIGIT_TWO;

    DataCopyExtParams copyOutParamY = {
        static_cast<uint16_t>(ubFactorRowNum), static_cast<uint32_t>(ubFactorColNum / DIGIT_TWO),
        static_cast<uint32_t>(0), // 一个 BlockSize = 32，64个数对齐，输出 y 为 0.5 Bytes，占用 32 Bytes，srcStride = 0
        static_cast<uint32_t>((colNum_ - ubFactorColNum) / DIGIT_TWO), static_cast<uint32_t>(0)};

    int64_t offset = (dim0LoopIdx * ubFactorRowNormalBlockNum_ * colNum_ + dim1LoopIdx * ubFactorColNormalBlockLen_) /
                     DIGIT_TWO;
    DataCopyPad(yGm_[offset], yLocal, copyOutParamY);

    DataCopyExtParams copyOutParamScale = {0, 0, 0, 0, 0};
    copyOutParamScale.blockCount = ubFactorRowNum;
    copyOutParamScale.blockLen = ubFactorColBlockNum + scaleIsNotOdd;
    copyOutParamScale.srcStride = 0;
    copyOutParamScale.dstStride = scaleColNum_ - copyOutParamScale.blockLen;

    int64_t scaleOffset = dim0LoopIdx * ubFactorRowNormalBlockNum_ * scaleColNum_ +
                          dim1LoopIdx * ubFactorColNormalBlockNum_;
    DataCopyPad<uint8_t, PaddingMode::Compact>(scaleGm_[scaleOffset], scaleLocal, copyOutParamScale);

    outQueue_.FreeTensor(yLocal);
    mxScaleQueue_.FreeTensor(scaleLocal);
}

// ========================================== Compute Max Exp =====================================================
template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeMaxExpOcpHalf(__ubuf__ half* xLocalAddr,
                                                                                     __ubuf__ uint16_t* maxExpAddr,
                                                                                     uint16_t loopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> xExp0;
        Reg::RegTensor<T> xExp1;
        Reg::RegTensor<uint16_t> xMaxExp;
        Reg::RegTensor<bfloat16_t> xExp0BF16;
        Reg::RegTensor<bfloat16_t> xExp1BF16;
        Reg::RegTensor<uint16_t> xExpExtract0;
        Reg::RegTensor<uint16_t> xExpExtract1;
        Reg::RegTensor<uint16_t> xExpSelect0;
        Reg::RegTensor<uint16_t> xExpSelect1;

        Reg::RegTensor<uint16_t> expMaskBF16;
        Reg::Duplicate(expMaskBF16, BF16_MAX_EXP);
        Reg::RegTensor<uint16_t> invalidmaskfp16;
        Reg::Duplicate(invalidmaskfp16, FP16_INVALID);

        Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg invalidDataMask0;
        Reg::MaskReg invalidDataMask1;
        Reg::UnalignReg ureg;

        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                              Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};

        for (uint16_t i = 0; i < loopNum2VF; i++) {
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                xExp0, xExp1, xLocalAddr, VF_LEN_16_DOUBLE);

            Reg::And(xExpSelect0, (Reg::RegTensor<uint16_t>&)xExp0, invalidmaskfp16, Mask);
            Reg::And(xExpSelect1, (Reg::RegTensor<uint16_t>&)xExp1, invalidmaskfp16, Mask);
            Reg::Compare<uint16_t, CMPMODE::NE>(invalidDataMask0, xExpSelect0, invalidmaskfp16, Mask);
            Reg::Compare<uint16_t, CMPMODE::NE>(invalidDataMask1, xExpSelect1, invalidmaskfp16, Mask);
            Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(xExp0BF16, xExp0, Mask);
            Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(xExp1BF16, xExp1, Mask);
            Reg::And(xExpExtract0, (Reg::RegTensor<uint16_t>&)xExp0BF16, expMaskBF16, Mask);
            Reg::And(xExpExtract1, (Reg::RegTensor<uint16_t>&)xExp1BF16, expMaskBF16, Mask);
            Reg::Select<uint16_t>(xExpExtract0, xExpExtract0, expMaskBF16, invalidDataMask0);
            Reg::Select<uint16_t>(xExpExtract1, xExpExtract1, expMaskBF16, invalidDataMask1);

            Reg::Max(xMaxExp, xExpExtract0, xExpExtract1, Mask);
            Reg::ReduceMaxWithDataBlock(xMaxExp, xMaxExp, Mask);

            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(maxExpAddr, xMaxExp, ureg,
                                                                            ELEMENT_AFTER_REDUCE_);
        }
        Reg::StoreUnAlignPost(maxExpAddr, ureg, 0);
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeMaxExpOcpBf16(__ubuf__ bfloat16_t* xLocalAddr,
                                                                                     __ubuf__ uint16_t* maxExpAddr,
                                                                                     uint16_t loopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<bfloat16_t> xExp0;
        Reg::RegTensor<bfloat16_t> xExp1;
        Reg::RegTensor<uint16_t> xMaxExp;
        Reg::RegTensor<uint16_t> xExpExtract0;
        Reg::RegTensor<uint16_t> xExpExtract1;

        Reg::RegTensor<uint16_t> expMaskBF16;
        Reg::Duplicate(expMaskBF16, BF16_MAX_EXP);

        Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::UnalignReg ureg;

        for (uint16_t i = 0; i < loopNum2VF; i++) {
            Reg::LoadAlign<bfloat16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                xExp0, xExp1, xLocalAddr, VF_LEN_16_DOUBLE);
            Reg::And(xExpExtract0, (Reg::RegTensor<uint16_t>&)xExp0, expMaskBF16, Mask);
            Reg::And(xExpExtract1, (Reg::RegTensor<uint16_t>&)xExp1, expMaskBF16, Mask);
            Reg::Max(xMaxExp, xExpExtract0, xExpExtract1, Mask);
            Reg::ReduceMaxWithDataBlock(xMaxExp, xMaxExp, Mask);
            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(maxExpAddr, xMaxExp, ureg,
                                                                            ELEMENT_AFTER_REDUCE_);
        }
        Reg::StoreUnAlignPost(maxExpAddr, ureg, 0);
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeMaxExpOcpFp32(__ubuf__ float* xLocalAddr,
                                                                                     __ubuf__ uint16_t* maxExpAddr,
                                                                                     uint16_t loopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> x0ZeroFP32;
        Reg::RegTensor<float> x0OneFP32;
        Reg::RegTensor<float> x1ZeroFP32;
        Reg::RegTensor<float> x1OneFP32;
        Reg::RegTensor<uint32_t> x0MaxExpB32;
        Reg::RegTensor<uint32_t> x1MaxExpB32;
        Reg::RegTensor<uint32_t> xMaxExpB32;
        Reg::RegTensor<uint16_t> xMaxExpB16;

        Reg::RegTensor<uint32_t> expMask32Bit;
        Reg::Duplicate(expMask32Bit, FP32_MX_MAX_EXP);

        Reg::MaskReg MaskB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::UnalignReg ureg;
        static constexpr Reg::CastTrait castTraitFp32toBF16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                               Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
        for (uint16_t i = 0; i < loopNum2VF; i++) {
            // x0ZeroFP32: 1,3,5,7,9,...,127        x1ZeroFP32: 2,4,6,8,10,...,128
            Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                x0ZeroFP32, x1ZeroFP32, xLocalAddr, VF_LEN_32_DOUBLE);
            // x0OneFP32: 129,131,133,135,...,255  x1OneFP32: 130,132,134,136,...,256
            Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                x0OneFP32, x1OneFP32, xLocalAddr, VF_LEN_32_DOUBLE);
            // x0ZeroFP32: 1,5,9,13,...,253    x0OneFP32: 3,7,11,15,...,255
            Reg::DeInterleave(x0ZeroFP32, x0OneFP32, x0ZeroFP32, x0OneFP32);
            // x1ZeroFP32: 2,6,10,14,...,254    x1OneFP32: 4,8,12,16,...,256
            Reg::DeInterleave(x1ZeroFP32, x1OneFP32, x1ZeroFP32, x1OneFP32);
            Reg::And((Reg::RegTensor<uint32_t>&)x0ZeroFP32, (Reg::RegTensor<uint32_t>&)x0ZeroFP32, expMask32Bit,
                     MaskB32);
            Reg::And((Reg::RegTensor<uint32_t>&)x0OneFP32, (Reg::RegTensor<uint32_t>&)x0OneFP32, expMask32Bit, MaskB32);
            Reg::And((Reg::RegTensor<uint32_t>&)x1ZeroFP32, (Reg::RegTensor<uint32_t>&)x1ZeroFP32, expMask32Bit,
                     MaskB32);
            Reg::And((Reg::RegTensor<uint32_t>&)x1OneFP32, (Reg::RegTensor<uint32_t>&)x1OneFP32, expMask32Bit, MaskB32);
            Reg::Max(x0MaxExpB32, (Reg::RegTensor<uint32_t>&)x0ZeroFP32, (Reg::RegTensor<uint32_t>&)x0OneFP32, MaskB32);
            Reg::Max(x1MaxExpB32, (Reg::RegTensor<uint32_t>&)x1ZeroFP32, (Reg::RegTensor<uint32_t>&)x1OneFP32, MaskB32);
            Reg::Max(xMaxExpB32, x0MaxExpB32, x1MaxExpB32, MaskB32);
            Reg::ReduceMaxWithDataBlock(xMaxExpB32, xMaxExpB32, MaskB32);
            Reg::ShiftRights(xMaxExpB32, xMaxExpB32, FP32_PACK_SHR_NUM, MaskB32);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(xMaxExpB16, xMaxExpB32);
            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(maxExpAddr, xMaxExpB16, ureg,
                                                                            ELEMENT_AFTER_REDUCE_);
        }
        Reg::StoreUnAlignPost(maxExpAddr, ureg, 0);
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeMaxExpDynDtypeRangeHalf(
    __ubuf__ half* xLocalAddr, __ubuf__ uint16_t* maxExpAddr, uint16_t loopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> xExp0;
        Reg::RegTensor<T> xExp1;
        Reg::RegTensor<uint16_t> xMaxExp;
        Reg::RegTensor<bfloat16_t> xExp0BF16;
        Reg::RegTensor<bfloat16_t> xExp1BF16;
        Reg::RegTensor<uint16_t> xExpSelect0;
        Reg::RegTensor<uint16_t> xExpSelect1;
        Reg::RegTensor<uint16_t> xExpExtract0;
        Reg::RegTensor<uint16_t> xExpExtract1;

        Reg::RegTensor<uint16_t> expMaskBF16;
        Reg::Duplicate(expMaskBF16, BF16_MAX_EXP);
        Reg::RegTensor<uint16_t> invalidMaskFP16;
        Reg::Duplicate(invalidMaskFP16, FP16_INVALID);
        Reg::RegTensor<uint16_t> absMask16Bit;
        Reg::Duplicate(absMask16Bit, BF16_ABS_MASK);

        Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg invalidDataMask0;
        Reg::MaskReg invalidDataMask1;
        Reg::UnalignReg ureg;

        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                              Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

        for (uint16_t i = 0; i < loopNum2VF; i++) {
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                xExp0, xExp1, xLocalAddr, VF_LEN_16_DOUBLE);
            Reg::And(xExpSelect0, (Reg::RegTensor<uint16_t>&)xExp0, invalidMaskFP16, Mask);
            Reg::And(xExpSelect1, (Reg::RegTensor<uint16_t>&)xExp1, invalidMaskFP16, Mask);
            Reg::Compare<uint16_t, CMPMODE::NE>(invalidDataMask0, xExpSelect0, invalidMaskFP16, Mask);
            Reg::Compare<uint16_t, CMPMODE::NE>(invalidDataMask1, xExpSelect1, invalidMaskFP16, Mask);
            Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(xExp0BF16, xExp0, Mask);
            Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(xExp1BF16, xExp1, Mask);
            Reg::And(xExpExtract0, (Reg::RegTensor<uint16_t>&)xExp0BF16, absMask16Bit, Mask);
            Reg::And(xExpExtract1, (Reg::RegTensor<uint16_t>&)xExp1BF16, absMask16Bit, Mask);
            Reg::Select<uint16_t>(xExpExtract0, xExpExtract0, expMaskBF16, invalidDataMask0);
            Reg::Select<uint16_t>(xExpExtract1, xExpExtract1, expMaskBF16, invalidDataMask1);

            Reg::Max(xMaxExp, xExpExtract0, xExpExtract1, Mask);
            Reg::ReduceMaxWithDataBlock(xMaxExp, xMaxExp, Mask);

            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(maxExpAddr, xMaxExp, ureg,
                                                                            ELEMENT_AFTER_REDUCE_);
        }
        Reg::StoreUnAlignPost(maxExpAddr, ureg, 0);
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeMaxExpDynDtypeRangeBf16(
    __ubuf__ bfloat16_t* xLocalAddr, __ubuf__ uint16_t* maxExpAddr, uint16_t loopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<bfloat16_t> xExp0;
        Reg::RegTensor<bfloat16_t> xExp1;
        Reg::RegTensor<uint16_t> xMaxExp;
        Reg::RegTensor<uint16_t> xExpExtract0;
        Reg::RegTensor<uint16_t> xExpExtract1;

        Reg::RegTensor<uint16_t> absMask16Bit;
        Reg::Duplicate(absMask16Bit, BF16_ABS_MASK);

        Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::UnalignReg ureg;

        for (uint16_t i = 0; i < loopNum2VF; i++) {
            Reg::LoadAlign<bfloat16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                xExp0, xExp1, xLocalAddr, VF_LEN_16_DOUBLE);
            Reg::And(xExpExtract0, (Reg::RegTensor<uint16_t>&)xExp0, absMask16Bit, Mask);
            Reg::And(xExpExtract1, (Reg::RegTensor<uint16_t>&)xExp1, absMask16Bit, Mask);
            Reg::Max(xMaxExp, xExpExtract0, xExpExtract1, Mask);
            Reg::ReduceMaxWithDataBlock(xMaxExp, xMaxExp, Mask);
            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(maxExpAddr, xMaxExp, ureg,
                                                                            ELEMENT_AFTER_REDUCE_);
        }
        Reg::StoreUnAlignPost(maxExpAddr, ureg, 0);
    }
    return;
}
template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeMaxExpDynDtypeRangeFp32(
    __ubuf__ float* xLocalAddr, __ubuf__ uint16_t* maxExpAddr, uint16_t loopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> xExp0;
        Reg::RegTensor<float> xExp1;
        Reg::RegTensor<float> xExp2;
        Reg::RegTensor<float> xExp3;
        Reg::RegTensor<uint32_t> xMaxExpB32;
        Reg::RegTensor<uint16_t> xMaxExpB16;

        Reg::RegTensor<uint32_t> absMask32Bit;
        Reg::Duplicate(absMask32Bit, FP32_ABS_MASK);

        Reg::MaskReg MaskB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::UnalignReg ureg;

        for (uint16_t i = 0; i < loopNum2VF; i++) {
            Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                xExp0, xExp1, xLocalAddr, VF_LEN_32_DOUBLE);
            Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                xExp2, xExp3, xLocalAddr, VF_LEN_32_DOUBLE);
            Reg::And((Reg::RegTensor<uint32_t>&)xExp0, (Reg::RegTensor<uint32_t>&)xExp0, absMask32Bit, MaskB32);
            Reg::And((Reg::RegTensor<uint32_t>&)xExp1, (Reg::RegTensor<uint32_t>&)xExp1, absMask32Bit, MaskB32);
            Reg::And((Reg::RegTensor<uint32_t>&)xExp2, (Reg::RegTensor<uint32_t>&)xExp2, absMask32Bit, MaskB32);
            Reg::And((Reg::RegTensor<uint32_t>&)xExp3, (Reg::RegTensor<uint32_t>&)xExp3, absMask32Bit, MaskB32);
            Reg::DeInterleave(xExp0, xExp2, xExp0, xExp2);
            Reg::DeInterleave(xExp1, xExp3, xExp1, xExp3);
            Reg::Max(xMaxExpB32, (Reg::RegTensor<uint32_t>&)xExp0, (Reg::RegTensor<uint32_t>&)xExp1, MaskB32);
            Reg::Max(xMaxExpB32, (Reg::RegTensor<uint32_t>&)xMaxExpB32, (Reg::RegTensor<uint32_t>&)xExp2, MaskB32);
            Reg::Max(xMaxExpB32, (Reg::RegTensor<uint32_t>&)xMaxExpB32, (Reg::RegTensor<uint32_t>&)xExp3, MaskB32);
            Reg::ReduceMaxWithDataBlock(xMaxExpB32, xMaxExpB32, MaskB32);
            Reg::ShiftRights(xMaxExpB32, xMaxExpB32, FP32_PACK_SHR_NUM, MaskB32);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(xMaxExpB16, xMaxExpB32);
            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(maxExpAddr, xMaxExpB16, ureg,
                                                                            ELEMENT_AFTER_REDUCE_);
        }
        Reg::StoreUnAlignPost(maxExpAddr, ureg, 0);
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeMaxExpCublasHalf(__ubuf__ half* xLocalAddr,
                                                                                        __ubuf__ uint16_t* maxExpAddr,
                                                                                        uint16_t loopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> xExp0;
        Reg::RegTensor<T> xExp1;
        Reg::RegTensor<uint16_t> xMaxExp;

        Reg::RegTensor<uint16_t> absMask16Bit;
        Reg::Duplicate(absMask16Bit, BF16_ABS_MASK);

        Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::UnalignReg ureg;

        for (uint16_t i = 0; i < loopNum2VF; i++) {
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                xExp0, xExp1, xLocalAddr, VF_LEN_16_DOUBLE);
            Reg::And((Reg::RegTensor<uint16_t>&)xExp0, (Reg::RegTensor<uint16_t>&)xExp0, absMask16Bit, Mask);
            Reg::And((Reg::RegTensor<uint16_t>&)xExp1, (Reg::RegTensor<uint16_t>&)xExp1, absMask16Bit, Mask);
            Reg::Max(xMaxExp, (Reg::RegTensor<uint16_t>&)xExp0, (Reg::RegTensor<uint16_t>&)xExp1, Mask);
            Reg::ReduceMaxWithDataBlock(xMaxExp, xMaxExp, Mask);
            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(maxExpAddr, xMaxExp, ureg,
                                                                            ELEMENT_AFTER_REDUCE_);
        }
        Reg::StoreUnAlignPost(maxExpAddr, ureg, 0);
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeMaxExpCublasBf16(__ubuf__ bfloat16_t* xLocalAddr,
                                                                                        __ubuf__ uint16_t* maxExpAddr,
                                                                                        uint16_t loopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<bfloat16_t> xExp0;
        Reg::RegTensor<bfloat16_t> xExp1;
        Reg::RegTensor<uint16_t> xMaxExp;

        Reg::RegTensor<uint16_t> absMask16Bit;
        Reg::Duplicate(absMask16Bit, BF16_ABS_MASK);

        Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::UnalignReg ureg;

        for (uint16_t i = 0; i < loopNum2VF; i++) {
            Reg::LoadAlign<bfloat16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                xExp0, xExp1, xLocalAddr, VF_LEN_16_DOUBLE);
            Reg::And((Reg::RegTensor<uint16_t>&)xExp0, (Reg::RegTensor<uint16_t>&)xExp0, absMask16Bit, Mask);
            Reg::And((Reg::RegTensor<uint16_t>&)xExp1, (Reg::RegTensor<uint16_t>&)xExp1, absMask16Bit, Mask);
            Reg::Max(xMaxExp, (Reg::RegTensor<uint16_t>&)xExp0, (Reg::RegTensor<uint16_t>&)xExp1, Mask);
            Reg::ReduceMaxWithDataBlock(xMaxExp, xMaxExp, Mask);
            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(maxExpAddr, xMaxExp, ureg,
                                                                            ELEMENT_AFTER_REDUCE_);
        }
        Reg::StoreUnAlignPost(maxExpAddr, ureg, 0);
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeMaxExpCublasFp32(__ubuf__ float* xLocalAddr,
                                                                                        __ubuf__ uint32_t* maxExpAddr,
                                                                                        uint16_t loopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> xExp0;
        Reg::RegTensor<float> xExp1;
        Reg::RegTensor<float> xExp2;
        Reg::RegTensor<float> xExp3;
        Reg::RegTensor<uint32_t> xMaxExp;

        Reg::RegTensor<uint32_t> absMask32Bit;
        Reg::Duplicate(absMask32Bit, FP32_ABS_MASK);

        Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::UnalignReg ureg;

        for (uint16_t i = 0; i < loopNum2VF; i++) {
            Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                xExp0, xExp1, xLocalAddr, VF_LEN_32_DOUBLE);
            Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                xExp2, xExp3, xLocalAddr, VF_LEN_32_DOUBLE);
            Reg::And((Reg::RegTensor<uint32_t>&)xExp0, (Reg::RegTensor<uint32_t>&)xExp0, absMask32Bit, Mask);
            Reg::And((Reg::RegTensor<uint32_t>&)xExp1, (Reg::RegTensor<uint32_t>&)xExp1, absMask32Bit, Mask);
            Reg::And((Reg::RegTensor<uint32_t>&)xExp2, (Reg::RegTensor<uint32_t>&)xExp2, absMask32Bit, Mask);
            Reg::And((Reg::RegTensor<uint32_t>&)xExp3, (Reg::RegTensor<uint32_t>&)xExp3, absMask32Bit, Mask);
            Reg::DeInterleave(xExp0, xExp2, xExp0, xExp2);
            Reg::DeInterleave(xExp1, xExp3, xExp1, xExp3);
            Reg::Max(xMaxExp, (Reg::RegTensor<uint32_t>&)xExp0, (Reg::RegTensor<uint32_t>&)xExp1, Mask);
            Reg::Max(xMaxExp, (Reg::RegTensor<uint32_t>&)xMaxExp, (Reg::RegTensor<uint32_t>&)xExp2, Mask);
            Reg::Max(xMaxExp, (Reg::RegTensor<uint32_t>&)xMaxExp, (Reg::RegTensor<uint32_t>&)xExp3, Mask);
            Reg::ReduceMaxWithDataBlock(xMaxExp, xMaxExp, Mask);
            Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(maxExpAddr, xMaxExp, ureg,
                                                                            ELEMENT_AFTER_REDUCE_);
        }
        Reg::StoreUnAlignPost(maxExpAddr, ureg, 0);
    }
    return;
}

// ========================================== Compute Scale =====================================================
template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeScaleOcp(__ubuf__ uint16_t* maxExpAddr,
                                                                                __ubuf__ uint16_t* mxScaleLocalAddr,
                                                                                __ubuf__ uint16_t* recipScaleLocalAddr,
                                                                                uint16_t loopNum1VF,
                                                                                uint32_t totalScaleInUB)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> xExp0;
        Reg::RegTensor<T> xExp1;
        Reg::RegTensor<uint16_t> xMaxExp;
        Reg::RegTensor<uint16_t> sharedExp;
        Reg::RegTensor<uint16_t> scaleValue;
        Reg::RegTensor<uint16_t> halfScale;

        Reg::RegTensor<uint16_t> expMask;
        Reg::Duplicate(expMask, BF16_MAX_EXP);
        Reg::RegTensor<uint16_t> maxExpValue;
        Reg::Duplicate(maxExpValue, f4Emax_);
        Reg::RegTensor<uint16_t> scaleBias;
        Reg::Duplicate(scaleBias, BF16_EXP_BIAS);
        Reg::RegTensor<uint16_t> fp8NanU16;
        Reg::Duplicate(fp8NanU16, FP8_DEFAULT_MAX_EXP);
        Reg::RegTensor<uint16_t> zeroU16;
        Reg::Duplicate(zeroU16, 0);
        Reg::RegTensor<uint16_t> nanU16;
        Reg::Duplicate(nanU16, BF16_NAN_CUSTOM);
        Reg::RegTensor<uint16_t> specialExpU16;
        Reg::Duplicate(specialExpU16, BF16_SPECIAL_EXP_THRESHOLD);

        Reg::MaskReg cmpResult;
        Reg::MaskReg zeroMask;
        Reg::MaskReg preMaskScale;
        Reg::MaskReg invalidDataMask;
        Reg::MaskReg specialDataMask;

        for (uint16_t i = 0; i < loopNum1VF; i++) {
            preMaskScale = Reg::UpdateMask<uint16_t>(totalScaleInUB);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(xMaxExp, maxExpAddr, VF_LEN_16);
            Reg::Compare<uint16_t, CMPMODE::NE>(cmpResult, xMaxExp, expMask, preMaskScale); // INF/NAN
            Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, xMaxExp, maxExpValue, preMaskScale);

            Reg::Select<uint16_t>(xMaxExp, maxExpValue, xMaxExp, invalidDataMask);

            Reg::Sub(sharedExp, xMaxExp, maxExpValue, preMaskScale);
            Reg::ShiftRights(scaleValue, sharedExp, BF16_SHR_NUM, preMaskScale);

            Reg::Select<uint16_t>(scaleValue, scaleValue, fp8NanU16, cmpResult);

            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr, scaleValue, VF_LEN_32,
                preMaskScale); // 128 个scale，占用 128 * 1 Btyes = VF_LEN_32 * sizeof(uint16_t)

            Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, sharedExp, zeroU16, preMaskScale);
            Reg::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
            Reg::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            Reg::Select<uint16_t>(halfScale, halfScale, nanU16, cmpResult);
            Reg::Select<uint16_t>(halfScale, halfScale, zeroU16, zeroMask);
            Reg::Select<uint16_t>(halfScale, specialExpU16, halfScale, specialDataMask);

            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(recipScaleLocalAddr, halfScale, VF_LEN_16,
                                                                          preMaskScale);
        }
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeScaleDynamicDtypeRange(
    __ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr,
    uint16_t loopNum1VF, uint32_t totalScaleInUB)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> xMaxExp;
        Reg::RegTensor<uint16_t> sharedExp;
        Reg::RegTensor<uint16_t> scaleValue;
        Reg::RegTensor<uint16_t> halfScale;
        Reg::RegTensor<uint16_t> xMaxExpAdd;
        Reg::RegTensor<uint16_t> xMaxExpOnly;

        Reg::RegTensor<uint16_t> expMask;
        Reg::Duplicate(expMask, BF16_MAX_EXP);
        Reg::RegTensor<uint16_t> addValue;
        Reg::Duplicate(addValue, addValueBit_);
        Reg::RegTensor<uint16_t> maxExpValue;
        Reg::Duplicate(maxExpValue, FP4_E2M1_BF16_MAX_EXP);
        Reg::RegTensor<uint16_t> scaleBias;
        Reg::Duplicate(scaleBias, BF16_EXP_BIAS);
        Reg::RegTensor<uint16_t> fp8NanU16;
        Reg::Duplicate(fp8NanU16, FP8_DEFAULT_MAX_EXP);
        Reg::RegTensor<uint16_t> zeroU16;
        Reg::Duplicate(zeroU16, 0);
        Reg::RegTensor<uint16_t> nanU16;
        Reg::Duplicate(nanU16, BF16_NAN_CUSTOM);
        Reg::RegTensor<uint16_t> specialExpU16;
        Reg::Duplicate(specialExpU16, BF16_SPECIAL_EXP_THRESHOLD);

        Reg::MaskReg cmpResult;
        Reg::MaskReg zeroMask;
        Reg::MaskReg invalidDataMask;
        Reg::MaskReg specialDataMask;
        Reg::MaskReg preMaskScale;

        for (uint16_t i = 0; i < loopNum1VF; i++) {
            preMaskScale = Reg::UpdateMask<uint16_t>(totalScaleInUB);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(xMaxExp, maxExpAddr, VF_LEN_16);
            Reg::And(xMaxExpOnly, xMaxExp, expMask, preMaskScale);                              // 提取指数位
            Reg::Compare<uint16_t, CMPMODE::NE>(cmpResult, xMaxExpOnly, expMask, preMaskScale); // INF/NAN
            Reg::Compare<uint16_t, CMPMODE::LT>(invalidDataMask, xMaxExpOnly, maxExpValue, preMaskScale);

            Reg::Add(xMaxExpAdd, xMaxExp, addValue, preMaskScale);   // 进位后的结果
            Reg::And(xMaxExpAdd, xMaxExpAdd, expMask, preMaskScale); // 提取进位结果的指数位
            Reg::Select<uint16_t>(xMaxExpAdd, maxExpValue, xMaxExpAdd, invalidDataMask);
            Reg::Sub(sharedExp, xMaxExpAdd, maxExpValue, preMaskScale);

            Reg::ShiftRights(scaleValue, sharedExp, BF16_SHR_NUM, preMaskScale);
            Reg::Select<uint16_t>(scaleValue, scaleValue, fp8NanU16, cmpResult);

            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr, scaleValue, VF_LEN_32, preMaskScale);

            Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, sharedExp, zeroU16, preMaskScale);
            Reg::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
            Reg::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            Reg::Select<uint16_t>(halfScale, halfScale, nanU16, cmpResult);
            Reg::Select<uint16_t>(halfScale, halfScale, zeroU16, zeroMask);
            Reg::Select<uint16_t>(halfScale, specialExpU16, halfScale, specialDataMask);

            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM>(
                recipScaleLocalAddr, halfScale, VF_LEN_16, preMaskScale);
        }
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeScaleCublas(
    __ubuf__ intCalcType* maxExpAddr, __ubuf__ uint16_t* scaleLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr,
    uint16_t loopNumHalfVF, uint32_t totalScaleInUB)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> max16;
        Reg::RegTensor<uint32_t> max32;
        Reg::RegTensor<uint32_t> exp32;
        Reg::RegTensor<uint32_t> man32;
        Reg::RegTensor<uint32_t> normalExp32;
        Reg::RegTensor<uint32_t> expAddOne32;
        Reg::RegTensor<uint32_t> extractExp;
        Reg::RegTensor<uint16_t> expOut;
        Reg::RegTensor<uint32_t> halfScale;
        Reg::RegTensor<uint16_t> recExpOut;

        Reg::RegTensor<uint32_t> manMaskFP32;
        Reg::Duplicate(manMaskFP32, FP32_MX_MAN_MASK);
        Reg::RegTensor<uint32_t> expMask;
        Reg::Duplicate(expMask, FP32_MX_MAX_EXP);
        Reg::RegTensor<uint32_t> zeroU32;
        Reg::Duplicate(zeroU32, 0);
        Reg::RegTensor<uint32_t> scaleBias;
        Reg::Duplicate(scaleBias, FP32_MX_EXP_BIAS_CUBLAS);
        Reg::RegTensor<uint32_t> nanU32;
        Reg::Duplicate(nanU32, FP32_NAN_PACK);
        Reg::RegTensor<uint32_t> fp4NanFP32;
        Reg::Duplicate(fp4NanFP32, FP8_MAX_EXP_IN_FP32);
        Reg::RegTensor<float> invMax;
        Reg::Duplicate(invMax, invDstTypeMax_);

        Reg::MaskReg cmpResult;
        Reg::MaskReg zeroMask;
        Reg::MaskReg p0;
        Reg::MaskReg p1;
        Reg::MaskReg p2;
        Reg::MaskReg preMaskScale;
        uint32_t SixtyFour = 64;
        Reg::MaskReg dataMaskB16Half = Reg::UpdateMask<uint16_t>(SixtyFour);

        static constexpr Reg::CastTrait castTraitHalf2Float = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                               Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        for (uint16_t i = 0; i < loopNumHalfVF; i++) {
            preMaskScale = Reg::UpdateMask<uint32_t>(totalScaleInUB);
            if constexpr (IsSame<T, float>::value) {
                Reg::LoadAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_NORM>(
                    max32, maxExpAddr, VF_LEN_32);
            } else {
                Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(
                    max16, maxExpAddr, VF_LEN_32);
                Reg::Cast<float, T, castTraitHalf2Float>((Reg::RegTensor<float>&)max32, (Reg::RegTensor<T>&)max16,
                                                         preMaskScale);
            }

            Reg::Compare<uint32_t, CMPMODE::LT>(cmpResult, max32, expMask, preMaskScale);
            Reg::Compare<uint32_t, CMPMODE::NE>(zeroMask, max32, zeroU32, preMaskScale);

            Reg::Mul((Reg::RegTensor<float>&)max32, (Reg::RegTensor<float>&)max32, invMax, preMaskScale);
            Reg::ShiftRights(exp32, max32, FP32_SHR_NUM, preMaskScale);
            Reg::And(man32, max32, manMaskFP32, preMaskScale);

            Reg::CompareScalar<uint32_t, CMPMODE::GT>(p0, exp32, FP32_NUMBER_ZERO, preMaskScale);
            Reg::CompareScalar<uint32_t, CMPMODE::LT>(p1, exp32, FP32_NUMBER_254, preMaskScale);
            Reg::CompareScalar<uint32_t, CMPMODE::GT>(p2, man32, FP32_NUMBER_ZERO, preMaskScale);
            Reg::MaskAnd(p0, p0, p1, preMaskScale);
            Reg::MaskAnd(p0, p0, p2, preMaskScale);

            Reg::Adds(expAddOne32, exp32, 1, preMaskScale);
            Reg::Select(extractExp, expAddOne32, exp32, p0);
            Reg::Select<uint32_t>(extractExp, extractExp, fp4NanFP32, cmpResult);
            Reg::Select<uint32_t>(extractExp, extractExp, zeroU32, zeroMask);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(expOut, extractExp);

            Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_PACK_B16>(scaleLocalAddr + i * 32, expOut, dataMaskB16Half);

            Reg::ShiftLefts(extractExp, extractExp, BF16_SHR_NUM, preMaskScale);
            Reg::Sub(halfScale, scaleBias, extractExp, preMaskScale);
            Reg::Select<uint32_t>(halfScale, halfScale, nanU32, cmpResult);
            Reg::Select<uint32_t>(halfScale, halfScale, zeroU32, zeroMask);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(recExpOut, halfScale);

            Reg::StoreAlign<uint16_t>(recipScaleLocalAddr + i * VF_LEN_32, recExpOut, dataMaskB16Half);
        }
    }
    return;
}

// ================================================ Compute Data =======================================================
template <typename T, typename U, int64_t SCALE_ALG>
template <RoundMode toBf16RoundMode, RoundMode roundMode>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeDataHalf(__ubuf__ half* xLocalAddr,
                                                                                __ubuf__ uint16_t* recipScaleLocalAddr,
                                                                                __ubuf__ int8_t* yLocalAddr,
                                                                                uint16_t loopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> xExp0;
        Reg::RegTensor<T> xExp1;
        Reg::RegTensor<T> xExp0Convert;
        Reg::RegTensor<T> xExp1Convert;
        Reg::RegTensor<uint16_t> halfScaleForMul;
        Reg::RegTensor<bfloat16_t> xExp0BF16;
        Reg::RegTensor<bfloat16_t> xExp1BF16;
        Reg::RegTensor<bfloat16_t> xBF16Exp0FP4;
        Reg::RegTensor<bfloat16_t> xBF16Exp1FP4;
        Reg::RegTensor<U> xExp0FP4;
        Reg::RegTensor<U> xExp1FP4;

        Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();

        static constexpr Reg::CastTrait castTrait = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                     Reg::MaskMergeMode::ZEROING, roundMode};
        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                              Reg::MaskMergeMode::ZEROING, toBf16RoundMode};
        static constexpr Reg::CastTrait castTraitFp32toBF16Data = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                                   Reg::MaskMergeMode::ZEROING, roundMode};

        for (uint16_t i = 0; i < loopNum2VF; i++) {
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                xExp0, xExp1, xLocalAddr, VF_LEN_16_DOUBLE);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                halfScaleForMul, recipScaleLocalAddr, ELEMENT_AFTER_REDUCE_);

            if constexpr (roundMode == RoundMode::CAST_RINT) {
                ComputeFP4FromHalf(xExp0, xExp0, Mask);
                ComputeFP4FromHalf(xExp1, xExp1, Mask);
            }
            Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(xExp0BF16, xExp0, Mask);
            Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(xExp1BF16, xExp1, Mask);
            Reg::Mul(xExp0BF16, xExp0BF16, (Reg::RegTensor<bfloat16_t>&)halfScaleForMul, Mask);
            Reg::Mul(xExp1BF16, xExp1BF16, (Reg::RegTensor<bfloat16_t>&)halfScaleForMul, Mask);
            Reg::Interleave(xExp0BF16, xExp1BF16, xExp0BF16, xExp1BF16);
            Reg::Cast<U, bfloat16_t, castTrait>(xExp0FP4, xExp0BF16, Mask);
            Reg::Cast<U, bfloat16_t, castTrait>(xExp1FP4, xExp1BF16, Mask);

            Reg::StoreAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                yLocalAddr, (Reg::RegTensor<int8_t>&)xExp0FP4, FP4_OUT_ELE_PER_BLK, Mask);
            Reg::StoreAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                yLocalAddr, (Reg::RegTensor<int8_t>&)xExp1FP4, FP4_OUT_ELE_PER_BLK, Mask);
        }
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
template <RoundMode toBf16RoundMode, RoundMode roundMode>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeDataBf16(__ubuf__ bfloat16_t* xLocalAddr,
                                                                                __ubuf__ uint16_t* recipScaleLocalAddr,
                                                                                __ubuf__ int8_t* yLocalAddr,
                                                                                uint16_t loopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<bfloat16_t> xExp0;
        Reg::RegTensor<bfloat16_t> xExp1;
        Reg::RegTensor<uint16_t> halfScaleForMul;
        Reg::RegTensor<U> xExp0FP4;
        Reg::RegTensor<U> xExp1FP4;

        Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();

        static constexpr Reg::CastTrait castTrait = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                     Reg::MaskMergeMode::ZEROING, roundMode};
        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                              Reg::MaskMergeMode::ZEROING, toBf16RoundMode};
        static constexpr Reg::CastTrait castTraitFp32toBF16Data = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                                   Reg::MaskMergeMode::ZEROING, roundMode};

        for (uint16_t i = 0; i < loopNum2VF; i++) {
            Reg::LoadAlign<bfloat16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                xExp0, xExp1, xLocalAddr, VF_LEN_16_DOUBLE);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                halfScaleForMul, recipScaleLocalAddr, ELEMENT_AFTER_REDUCE_);

            Reg::Mul(xExp0, xExp0, (Reg::RegTensor<bfloat16_t>&)halfScaleForMul, Mask);
            Reg::Mul(xExp1, xExp1, (Reg::RegTensor<bfloat16_t>&)halfScaleForMul, Mask);
            Reg::Interleave(xExp0, xExp1, xExp0, xExp1);
            Reg::Cast<U, bfloat16_t, castTrait>(xExp0FP4, xExp0, Mask);
            Reg::Cast<U, bfloat16_t, castTrait>(xExp1FP4, xExp1, Mask);

            Reg::StoreAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                yLocalAddr, (Reg::RegTensor<int8_t>&)xExp0FP4, FP4_OUT_ELE_PER_BLK, Mask);
            Reg::StoreAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                yLocalAddr, (Reg::RegTensor<int8_t>&)xExp1FP4, FP4_OUT_ELE_PER_BLK, Mask);
        }
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
template <RoundMode toBf16RoundMode, RoundMode roundMode>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeDataOptimizeHalf(
    __ubuf__ half* xLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, __ubuf__ int8_t* yLocalAddr, uint16_t loopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> halfScaleForMul;
        Reg::RegTensor<T> xExp0;
        Reg::RegTensor<T> xExp1;
        Reg::RegTensor<T> xExp0Convert;
        Reg::RegTensor<T> xExp1Convert;
        Reg::RegTensor<float> halfScaleForMulFP32;
        Reg::RegTensor<float> xExp0ZeroFP32;
        Reg::RegTensor<float> xExp0OneFP32;
        Reg::RegTensor<float> xExp1ZeroFP32;
        Reg::RegTensor<float> xExp1OneFP32;
        Reg::RegTensor<bfloat16_t> xExp0ZeroBF16;
        Reg::RegTensor<bfloat16_t> xExp0OneBF16;
        Reg::RegTensor<bfloat16_t> xExp1ZeroBF16;
        Reg::RegTensor<bfloat16_t> xExp1OneBF16;

        Reg::RegTensor<bfloat16_t> xExp0BF16;
        Reg::RegTensor<bfloat16_t> xExp1BF16;

        Reg::RegTensor<U> xExp0FP4;
        Reg::RegTensor<U> xExp1FP4;

        Reg::RegTensor<bfloat16_t> xBF16Exp0FP4;
        Reg::RegTensor<bfloat16_t> xBF16Exp1FP4;

        Reg::MaskReg dataMaskB16 = Reg::CreateMask<half>();
        Reg::MaskReg dataMaskB32 = Reg::CreateMask<float>();

        static constexpr Reg::CastTrait castTrait = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                     Reg::MaskMergeMode::ZEROING, roundMode};
        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                              Reg::MaskMergeMode::ZEROING, toBf16RoundMode};
        static constexpr Reg::CastTrait castTraitF16toFp32Zero = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                                  Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitF16toFp32One = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                                 Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitFp32toBF16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                               Reg::MaskMergeMode::ZEROING, roundMode};

        for (uint16_t i = 0; i < loopNum2VF; i++) {
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                halfScaleForMul, recipScaleLocalAddr, ELEMENT_AFTER_REDUCE_);

            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                xExp0, xExp1, xLocalAddr, VF_LEN_16_DOUBLE);

            Reg::Cast<float, bfloat16_t, castTraitF16toFp32Zero>(
                halfScaleForMulFP32, (Reg::RegTensor<bfloat16_t>&)halfScaleForMul, dataMaskB16);

            // xExp0
            Reg::Cast<float, T, castTraitF16toFp32Zero>(xExp0ZeroFP32, xExp0, dataMaskB16);
            Reg::Cast<float, T, castTraitF16toFp32One>(xExp0OneFP32, xExp0, dataMaskB16);

            Reg::Mul(xExp0ZeroFP32, xExp0ZeroFP32, halfScaleForMulFP32, dataMaskB32);
            Reg::Mul(xExp0OneFP32, xExp0OneFP32, halfScaleForMulFP32, dataMaskB32);
            ComputeFP4FromFp32<toBf16RoundMode, roundMode>(xExp0ZeroFP32);
            ComputeFP4FromFp32<toBf16RoundMode, roundMode>(xExp0OneFP32);
            Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(xExp0ZeroBF16, xExp0ZeroFP32, dataMaskB32);
            Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(xExp0OneBF16, xExp0OneFP32, dataMaskB32);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)xExp0ZeroBF16,
                                                                    (Reg::RegTensor<uint32_t>&)xExp0ZeroBF16);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)xExp0OneBF16,
                                                                    (Reg::RegTensor<uint32_t>&)xExp0OneBF16);
            Reg::Interleave(xExp0ZeroBF16, xExp0OneBF16, xExp0ZeroBF16, xExp0OneBF16);

            // xExp1
            Reg::Cast<float, T, castTraitF16toFp32Zero>(xExp1ZeroFP32, xExp1, dataMaskB16);
            Reg::Cast<float, T, castTraitF16toFp32One>(xExp1OneFP32, xExp1, dataMaskB16);

            Reg::Mul(xExp1ZeroFP32, xExp1ZeroFP32, halfScaleForMulFP32, dataMaskB32);
            Reg::Mul(xExp1OneFP32, xExp1OneFP32, halfScaleForMulFP32, dataMaskB32);
            ComputeFP4FromFp32<toBf16RoundMode, roundMode>(xExp1ZeroFP32);
            ComputeFP4FromFp32<toBf16RoundMode, roundMode>(xExp1OneFP32);
            Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(xExp1ZeroBF16, xExp1ZeroFP32, dataMaskB32);
            Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(xExp1OneBF16, xExp1OneFP32, dataMaskB32);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)xExp1ZeroBF16,
                                                                    (Reg::RegTensor<uint32_t>&)xExp1ZeroBF16);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)xExp1OneBF16,
                                                                    (Reg::RegTensor<uint32_t>&)xExp1OneBF16);
            Reg::Interleave(xExp1ZeroBF16, xExp1OneBF16, xExp1ZeroBF16, xExp1OneBF16);

            Reg::Interleave(xExp0ZeroBF16, xExp1ZeroBF16, xExp0ZeroBF16, xExp1ZeroBF16);
            Reg::Cast<U, bfloat16_t, castTrait>(xExp0FP4, xExp0ZeroBF16, dataMaskB16);
            Reg::Cast<U, bfloat16_t, castTrait>(xExp1FP4, xExp1ZeroBF16, dataMaskB16);

            Reg::StoreAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                yLocalAddr, (Reg::RegTensor<int8_t>&)xExp0FP4, FP4_OUT_ELE_PER_BLK, dataMaskB16);
            Reg::StoreAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                yLocalAddr, (Reg::RegTensor<int8_t>&)xExp1FP4, FP4_OUT_ELE_PER_BLK, dataMaskB16);
        }
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
template <RoundMode toBf16RoundMode, RoundMode roundMode>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeDataOptimizeBf16(
    __ubuf__ bfloat16_t* xLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, __ubuf__ int8_t* yLocalAddr,
    uint16_t loopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> halfScaleForMul;
        Reg::RegTensor<bfloat16_t> xExp0;
        Reg::RegTensor<bfloat16_t> xExp1;
        Reg::RegTensor<bfloat16_t> xExp0Convert;
        Reg::RegTensor<bfloat16_t> xExp1Convert;
        Reg::RegTensor<float> halfScaleForMulFP32;
        Reg::RegTensor<float> xExp0ZeroFP32;
        Reg::RegTensor<float> xExp0OneFP32;
        Reg::RegTensor<float> xExp1ZeroFP32;
        Reg::RegTensor<float> xExp1OneFP32;
        Reg::RegTensor<bfloat16_t> xExp0ZeroBF16;
        Reg::RegTensor<bfloat16_t> xExp0OneBF16;
        Reg::RegTensor<bfloat16_t> xExp1ZeroBF16;
        Reg::RegTensor<bfloat16_t> xExp1OneBF16;

        Reg::RegTensor<bfloat16_t> xExp0BF16;
        Reg::RegTensor<bfloat16_t> xExp1BF16;

        Reg::RegTensor<U> xExp0FP4;
        Reg::RegTensor<U> xExp1FP4;

        Reg::RegTensor<bfloat16_t> xBF16Exp0FP4;
        Reg::RegTensor<bfloat16_t> xBF16Exp1FP4;

        Reg::MaskReg dataMaskB16 = Reg::CreateMask<half>();
        Reg::MaskReg dataMaskB32 = Reg::CreateMask<float>();

        static constexpr Reg::CastTrait castTrait = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                     Reg::MaskMergeMode::ZEROING, roundMode};
        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                              Reg::MaskMergeMode::ZEROING, toBf16RoundMode};
        static constexpr Reg::CastTrait castTraitF16toFp32Zero = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                                  Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitF16toFp32One = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                                 Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitFp32toBF16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                               Reg::MaskMergeMode::ZEROING, roundMode};

        for (uint16_t i = 0; i < loopNum2VF; i++) {
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                halfScaleForMul, recipScaleLocalAddr, ELEMENT_AFTER_REDUCE_);

            Reg::LoadAlign<bfloat16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                xExp0, xExp1, xLocalAddr, VF_LEN_16_DOUBLE);
            Reg::Mul(xExp0, xExp0, (Reg::RegTensor<bfloat16_t>&)halfScaleForMul, dataMaskB16);
            Reg::Mul(xExp1, xExp1, (Reg::RegTensor<bfloat16_t>&)halfScaleForMul, dataMaskB16);
            Reg::Interleave(xExp0, xExp1, xExp0, xExp1);
            Reg::Cast<U, bfloat16_t, castTrait>(xExp0FP4, xExp0, dataMaskB16);
            Reg::Cast<U, bfloat16_t, castTrait>(xExp1FP4, xExp1, dataMaskB16);

            Reg::StoreAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                yLocalAddr, (Reg::RegTensor<int8_t>&)xExp0FP4, FP4_OUT_ELE_PER_BLK, dataMaskB16);
            Reg::StoreAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                yLocalAddr, (Reg::RegTensor<int8_t>&)xExp1FP4, FP4_OUT_ELE_PER_BLK, dataMaskB16);
        }
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
template <RoundMode toBf16RoundMode, RoundMode roundMode>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeDataFloatToFP4(
    __ubuf__ float* xLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, __ubuf__ int8_t* yLocalAddr,
    uint16_t loopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> halfScaleForMul;
        Reg::RegTensor<float> floatScaleForMul;
        Reg::RegTensor<float> x0Zero;
        Reg::RegTensor<float> x0One;
        Reg::RegTensor<float> x1Zero;
        Reg::RegTensor<float> x1One;

        Reg::RegTensor<bfloat16_t> x0ZeroBF16;
        Reg::RegTensor<bfloat16_t> x0OneBF16;
        Reg::RegTensor<bfloat16_t> x1ZeroBF16;
        Reg::RegTensor<bfloat16_t> x1OneBF16;

        Reg::RegTensor<U> xExp0FP4;
        Reg::RegTensor<U> xExp1FP4;

        Reg::MaskReg maskFP32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskB16 = Reg::CreateMask<half, Reg::MaskPattern::ALL>();

        static constexpr Reg::CastTrait castTraitBf16ToFp32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                               Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTrait = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                     Reg::MaskMergeMode::ZEROING, roundMode};
        static constexpr Reg::CastTrait castTraitFp32toBF16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                               Reg::MaskMergeMode::ZEROING, toBf16RoundMode};

        Reg::RegTensor<int32_t> maxEleReg;
        Reg::Duplicate(maxEleReg, FP32_MX_MAX_EXP);

        for (uint16_t i = 0; i < loopNum2VF; i++) {
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                halfScaleForMul, recipScaleLocalAddr, ELEMENT_AFTER_REDUCE_);
            Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32>(floatScaleForMul,
                                                              (Reg::RegTensor<bfloat16_t>&)halfScaleForMul, maskFP32);

            Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                x0Zero, x1Zero, xLocalAddr, VF_LEN_32_DOUBLE);
            Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                x0One, x1One, xLocalAddr, VF_LEN_32_DOUBLE);
            Reg::DeInterleave(x0Zero, x0One, x0Zero, x0One);
            Reg::DeInterleave(x1Zero, x1One, x1Zero, x1One);

            Reg::Mul(x0Zero, x0Zero, floatScaleForMul, maskFP32);
            Reg::Mul(x1Zero, x1Zero, floatScaleForMul, maskFP32);
            Reg::Mul(x0One, x0One, floatScaleForMul, maskFP32);
            Reg::Mul(x1One, x1One, floatScaleForMul, maskFP32);

            ComputeFP4FromFp32<toBf16RoundMode, roundMode>(x0Zero);
            ComputeFP4FromFp32<toBf16RoundMode, roundMode>(x0One);
            ComputeFP4FromFp32<toBf16RoundMode, roundMode>(x1Zero);
            ComputeFP4FromFp32<toBf16RoundMode, roundMode>(x1One);

            Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(x0ZeroBF16, x0Zero, maskFP32);
            Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(x0OneBF16, x0One, maskFP32);
            Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(x1ZeroBF16, x1Zero, maskFP32);
            Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(x1OneBF16, x1One, maskFP32);

            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)x0ZeroBF16,
                                                                    (Reg::RegTensor<uint32_t>&)x0ZeroBF16);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)x0OneBF16,
                                                                    (Reg::RegTensor<uint32_t>&)x0OneBF16);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)x1ZeroBF16,
                                                                    (Reg::RegTensor<uint32_t>&)x1ZeroBF16);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)x1OneBF16,
                                                                    (Reg::RegTensor<uint32_t>&)x1OneBF16);

            // Merge zero+one bf16 per group
            Reg::Interleave(x0ZeroBF16, x0OneBF16, x0ZeroBF16, x0OneBF16);
            Reg::Interleave(x1ZeroBF16, x1OneBF16, x1ZeroBF16, x1OneBF16);

            // Merge group 0+1 bf16
            Reg::Interleave(x0ZeroBF16, x1ZeroBF16, x0ZeroBF16, x1ZeroBF16);

            Reg::Cast<U, bfloat16_t, castTrait>(xExp0FP4, x0ZeroBF16, maskB16);
            Reg::Cast<U, bfloat16_t, castTrait>(xExp1FP4, x1ZeroBF16, maskB16);

            // DIST_PACK4_B32 output (required for fp4x2 format)
            Reg::StoreAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                yLocalAddr, (Reg::RegTensor<int8_t>&)xExp0FP4, FP4_OUT_ELE_PER_BLK, maskB16);
            Reg::StoreAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                yLocalAddr, (Reg::RegTensor<int8_t>&)xExp1FP4, FP4_OUT_ELE_PER_BLK, maskB16);
        }
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeOcp(
    __ubuf__ T* xLocalAddr, __ubuf__ uint16_t* maxExpLocalAddr, __ubuf__ uint16_t* scaleLocalAddr,
    __ubuf__ int8_t* yLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, uint16_t loopNum2VF, uint16_t loopNum1VF,
    uint16_t loopNumHalfVF, uint32_t totalBlockNum)
{
    if constexpr (IsSame<T, float>::value) {
        ComputeMaxExpOcpFp32(xLocalAddr, maxExpLocalAddr, loopNum2VF);
        ComputeScaleOcp(maxExpLocalAddr, scaleLocalAddr, recipScaleLocalAddr, loopNum1VF, totalBlockNum);
        if (roundMode_ == MODE_RINT) {
            ComputeDataFloatToFP4<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(xLocalAddr, recipScaleLocalAddr,
                                                                               yLocalAddr, loopNum2VF);
        } else if (roundMode_ == MODE_ROUND) {
            ComputeDataFloatToFP4<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(xLocalAddr, recipScaleLocalAddr,
                                                                                yLocalAddr, loopNum2VF);
        } else if (roundMode_ == MODE_FLOOR) {
            ComputeDataFloatToFP4<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(xLocalAddr, recipScaleLocalAddr,
                                                                                yLocalAddr, loopNum2VF);
        }
    } else if constexpr (IsSame<T, bfloat16_t>::value) {
        ComputeMaxExpOcpBf16(xLocalAddr, maxExpLocalAddr, loopNum2VF);
        ComputeScaleOcp(maxExpLocalAddr, scaleLocalAddr, recipScaleLocalAddr, loopNum1VF, totalBlockNum);
        if (roundMode_ == MODE_RINT) {
            ComputeDataOptimizeBf16<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(xLocalAddr, recipScaleLocalAddr,
                                                                                 yLocalAddr, loopNum2VF);
        } else if (roundMode_ == MODE_ROUND) {
            ComputeDataOptimizeBf16<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(xLocalAddr, recipScaleLocalAddr,
                                                                                  yLocalAddr, loopNum2VF);
        } else if (roundMode_ == MODE_FLOOR) {
            ComputeDataBf16<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(xLocalAddr, recipScaleLocalAddr, yLocalAddr,
                                                                          loopNum2VF);
        }
    } else {
        ComputeMaxExpOcpHalf(xLocalAddr, maxExpLocalAddr, loopNum2VF);
        ComputeScaleOcp(maxExpLocalAddr, scaleLocalAddr, recipScaleLocalAddr, loopNum1VF, totalBlockNum);
        if (roundMode_ == MODE_RINT) {
            ComputeDataOptimizeHalf<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(xLocalAddr, recipScaleLocalAddr,
                                                                                 yLocalAddr, loopNum2VF);
        } else if (roundMode_ == MODE_ROUND) {
            ComputeDataOptimizeHalf<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(xLocalAddr, recipScaleLocalAddr,
                                                                                  yLocalAddr, loopNum2VF);
        } else if (roundMode_ == MODE_FLOOR) {
            ComputeDataHalf<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(xLocalAddr, recipScaleLocalAddr, yLocalAddr,
                                                                          loopNum2VF);
        }
    }
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeCublasFp4(
    __ubuf__ T* xLocalAddr, __ubuf__ uint16_t* maxExpLocalAddr, __ubuf__ uint16_t* scaleLocalAddr,
    __ubuf__ int8_t* yLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, uint16_t loopNum2VF, uint16_t loopNum1VF,
    uint16_t loopNumHalfVF, uint32_t totalBlockNum)
{
    if constexpr (IsSame<T, float>::value) {
        LocalTensor<uint32_t> maxExpLocal = maxExpBuffer_.Get<uint32_t>();
        auto maxExpLocalAddr = reinterpret_cast<__ubuf__ uint32_t*>(maxExpLocal.GetPhyAddr());
        ComputeMaxExpCublasFp32(xLocalAddr, maxExpLocalAddr, loopNum2VF);
        ComputeScaleCublas(maxExpLocalAddr, scaleLocalAddr, recipScaleLocalAddr, loopNumHalfVF, totalBlockNum);
        if (roundMode_ == MODE_RINT) {
            ComputeDataFloatToFP4<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(xLocalAddr, recipScaleLocalAddr,
                                                                               yLocalAddr, loopNum2VF);
        } else if (roundMode_ == MODE_ROUND) {
            ComputeDataFloatToFP4<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(xLocalAddr, recipScaleLocalAddr,
                                                                                yLocalAddr, loopNum2VF);
        } else if (roundMode_ == MODE_FLOOR) {
            ComputeDataFloatToFP4<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(xLocalAddr, recipScaleLocalAddr,
                                                                                yLocalAddr, loopNum2VF);
        }
    } else if constexpr (IsSame<T, bfloat16_t>::value) {
        ComputeMaxExpCublasBf16(xLocalAddr, maxExpLocalAddr, loopNum2VF);
        ComputeScaleCublas(maxExpLocalAddr, scaleLocalAddr, recipScaleLocalAddr, loopNumHalfVF, totalBlockNum);
        if (roundMode_ == MODE_RINT) {
            ComputeDataOptimizeBf16<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(xLocalAddr, recipScaleLocalAddr,
                                                                                 yLocalAddr, loopNum2VF);
        } else if (roundMode_ == MODE_ROUND) {
            ComputeDataOptimizeBf16<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(xLocalAddr, recipScaleLocalAddr,
                                                                                  yLocalAddr, loopNum2VF);
        } else if (roundMode_ == MODE_FLOOR) {
            ComputeDataBf16<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(xLocalAddr, recipScaleLocalAddr, yLocalAddr,
                                                                          loopNum2VF);
        }
    } else {
        ComputeMaxExpCublasHalf(xLocalAddr, maxExpLocalAddr, loopNum2VF);
        ComputeScaleCublas(maxExpLocalAddr, scaleLocalAddr, recipScaleLocalAddr, loopNumHalfVF, totalBlockNum);
        if (roundMode_ == MODE_RINT) {
            ComputeDataOptimizeHalf<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(xLocalAddr, recipScaleLocalAddr,
                                                                                 yLocalAddr, loopNum2VF);
        } else if (roundMode_ == MODE_ROUND) {
            ComputeDataOptimizeHalf<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(xLocalAddr, recipScaleLocalAddr,
                                                                                  yLocalAddr, loopNum2VF);
        } else if (roundMode_ == MODE_FLOOR) {
            ComputeDataHalf<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(xLocalAddr, recipScaleLocalAddr, yLocalAddr,
                                                                          loopNum2VF);
        }
    }
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeCublasFp4Opti(
    __ubuf__ T* xLocalAddr, __ubuf__ uint16_t* maxExpLocalAddr, __ubuf__ uint16_t* scaleLocalAddr,
    __ubuf__ int8_t* yLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, uint16_t loopNum2VF, uint16_t loopNum1VF,
    uint16_t loopNumHalfVF, uint32_t totalBlockNum)
{
    if constexpr (IsSame<T, float>::value) {
        ComputeMaxExpDynDtypeRangeFp32(xLocalAddr, maxExpLocalAddr, loopNum2VF);
        ComputeScaleDynamicDtypeRange(maxExpLocalAddr, scaleLocalAddr, recipScaleLocalAddr, loopNum1VF, totalBlockNum);
        if (roundMode_ == MODE_RINT) {
            ComputeDataFloatToFP4<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(xLocalAddr, recipScaleLocalAddr,
                                                                               yLocalAddr, loopNum2VF);
        } else if (roundMode_ == MODE_ROUND) {
            ComputeDataFloatToFP4<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(xLocalAddr, recipScaleLocalAddr,
                                                                                yLocalAddr, loopNum2VF);
        } else if (roundMode_ == MODE_FLOOR) {
            ComputeDataFloatToFP4<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(xLocalAddr, recipScaleLocalAddr,
                                                                                yLocalAddr, loopNum2VF);
        }
    } else if constexpr (IsSame<T, bfloat16_t>::value) {
        ComputeMaxExpDynDtypeRangeBf16(xLocalAddr, maxExpLocalAddr, loopNum2VF);
        ComputeScaleDynamicDtypeRange(maxExpLocalAddr, scaleLocalAddr, recipScaleLocalAddr, loopNum1VF, totalBlockNum);
        if (roundMode_ == MODE_RINT) {
            ComputeDataOptimizeBf16<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(xLocalAddr, recipScaleLocalAddr,
                                                                                 yLocalAddr, loopNum2VF);
        } else if (roundMode_ == MODE_ROUND) {
            ComputeDataOptimizeBf16<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(xLocalAddr, recipScaleLocalAddr,
                                                                                  yLocalAddr, loopNum2VF);
        } else if (roundMode_ == MODE_FLOOR) {
            ComputeDataBf16<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(xLocalAddr, recipScaleLocalAddr, yLocalAddr,
                                                                          loopNum2VF);
        }
    } else {
        ComputeMaxExpDynDtypeRangeHalf(xLocalAddr, maxExpLocalAddr, loopNum2VF);
        ComputeScaleDynamicDtypeRange(maxExpLocalAddr, scaleLocalAddr, recipScaleLocalAddr, loopNum1VF, totalBlockNum);
        if (roundMode_ == MODE_RINT) {
            ComputeDataOptimizeHalf<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(xLocalAddr, recipScaleLocalAddr,
                                                                                 yLocalAddr, loopNum2VF);
        } else if (roundMode_ == MODE_ROUND) {
            ComputeDataOptimizeHalf<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(xLocalAddr, recipScaleLocalAddr,
                                                                                  yLocalAddr, loopNum2VF);
        } else if (roundMode_ == MODE_FLOOR) {
            ComputeDataHalf<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(xLocalAddr, recipScaleLocalAddr, yLocalAddr,
                                                                          loopNum2VF);
        }
    }
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::Compute(int64_t ubFactorRowBlockNum,
                                                                        int64_t ubFactorColBlockNum)
{
    ubFactorColBlockNum = ops::CeilDiv(ubFactorColBlockNum, DIGIT_TWO) * DIGIT_TWO;
    uint32_t totalBlockNum = ubFactorRowBlockNum *
                             ubFactorColBlockNum; // 当前Ub内总的Block数量 = 当前Ub内总的计算出Scale数量

    uint16_t loopNum2VF = ops::CeilDiv(
        static_cast<uint16_t>(totalBlockNum),
        ELEMENT_AFTER_REDUCE_); // 当前Ub内总的Block数量向上整除8，BlockSize=32，即得到双搬256个X的循环次数
    uint16_t loopNum1VF = ops::CeilDiv(
        static_cast<uint16_t>(totalBlockNum),
        static_cast<uint16_t>(VF_LEN_16)); // 当前Ub内总的Block数量向上整除128，即得到单搬128个max的循环次数
    uint16_t loopNumHalfVF = ops::CeilDiv(
        static_cast<uint16_t>(totalBlockNum),
        static_cast<uint16_t>(VF_LEN_32)); // 当前Ub内总的Block数量向上整除64，即得到单搬64个max的循环次数

    LocalTensor<T> xLocal = inQueue_.DeQue<T>();
    LocalTensor<uint16_t> scaleLocal = mxScaleQueue_.AllocTensor<uint16_t>();
    LocalTensor<int8_t> yLocal = outQueue_.AllocTensor<int8_t>();
    LocalTensor<uint16_t> maxExpLocal = maxExpBuffer_.Get<uint16_t>();
    LocalTensor<uint16_t> recipScaleLocal = recipScaleBuffer_.Get<uint16_t>();

    auto xLocalAddr = reinterpret_cast<__ubuf__ T*>(xLocal.GetPhyAddr());
    auto scaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(scaleLocal.GetPhyAddr());
    auto yLocalAddr = reinterpret_cast<__ubuf__ int8_t*>(yLocal.GetPhyAddr());
    auto maxExpLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());
    auto recipScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(recipScaleLocal.GetPhyAddr());
    if constexpr (ops::IsSame<U, fp4x2_e1m2_t>::value ||
                  (ops::IsSame<U, fp4x2_e2m1_t>::value && SCALE_ALG == DIGIT_ZERO)) {
        ComputeOcp(xLocalAddr, maxExpLocalAddr, scaleLocalAddr, yLocalAddr, recipScaleLocalAddr, loopNum2VF, loopNum1VF,
                   loopNumHalfVF, totalBlockNum);
    } else if constexpr (ops::IsSame<U, fp4x2_e2m1_t>::value && SCALE_ALG == DIGIT_TWO) {
        if (dstTypeMax_ == DIGIT_ZERO_FLOAT || dstTypeMax_ == DIGIT_SIX_FLOAT || dstTypeMax_ == DIGIT_SEVEN_FLOAT) {
            ComputeCublasFp4Opti(xLocalAddr, maxExpLocalAddr, scaleLocalAddr, yLocalAddr, recipScaleLocalAddr,
                                 loopNum2VF, loopNum1VF, loopNumHalfVF, totalBlockNum);
        } else {
            ComputeCublasFp4(xLocalAddr, maxExpLocalAddr, scaleLocalAddr, yLocalAddr, recipScaleLocalAddr, loopNum2VF,
                             loopNum1VF, loopNumHalfVF, totalBlockNum);
        }
    }

    inQueue_.FreeTensor(xLocal);
    mxScaleQueue_.EnQue(scaleLocal);
    outQueue_.EnQue(yLocal);
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeFP4FromHalf(Reg::RegTensor<half>& output,
                                                                                   Reg::RegTensor<half>& input,
                                                                                   Reg::MaskReg& mask)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> specialValueTensor;
        Reg::RegTensor<uint16_t> newMantissa;
        Reg::RegTensor<uint16_t> andResult;
        Reg::RegTensor<uint16_t> newValue;
        Reg::MaskReg specialMask;
        Reg::MaskReg nonzeroMask;
        uint16_t specialValue = FP4_E1M2_SPECIAL_VALUE;
        if constexpr (IsSame<U, fp4x2_e2m1_t>::value) {
            specialValue = FP4_E2M1_SPECIAL_VALUE;
        }
        Reg::Duplicate(specialValueTensor, specialValue);
        Reg::Duplicate(newMantissa, FP4_NEW_MANTISSA);
        Reg::And(andResult, (Reg::RegTensor<uint16_t>&)input, specialValueTensor, mask);
        Reg::CompareScalar<uint16_t, CMPMODE::GT>(nonzeroMask, andResult, 0, mask);
        Reg::CompareScalar<uint16_t, CMPMODE::LT>(specialMask, andResult, FP4_NEW_MANTISSA, mask);
        Reg::MaskAnd(specialMask, specialMask, nonzeroMask, mask);
        Reg::Or(newValue, (Reg::RegTensor<uint16_t>&)input, newMantissa, mask);
        Reg::Select<uint16_t>((Reg::RegTensor<uint16_t>&)output, newValue, (Reg::RegTensor<uint16_t>&)input,
                              specialMask);
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
template <RoundMode toBf16RoundMode, RoundMode roundMode>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeFP4FromFp32(Reg::RegTensor<float>& Reg)
{
    Reg::MaskReg pregAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg zeroMask;
    Reg::MaskReg specialMask;
    Reg::MaskReg negInfMask;

    Reg::RegTensor<int32_t> negZero;
    Reg::RegTensor<int32_t> maxExpFP32;
    Reg::RegTensor<int32_t> exp0FP32;
    Reg::RegTensor<int32_t> exp1FP32;

    Reg::Duplicate(negZero, FP32_NEG_ZERO_BITS);
    Reg::Compare<int32_t, CMPMODE::EQ>(negInfMask, (Reg::RegTensor<int32_t>&)Reg, negZero, pregAll32);
    if constexpr (IsSameType<U, fp4x2_e1m2_t>::value) {
        Reg::Muls(Reg, Reg, FP4_SCALE_FACTOR, pregAll32);
        Reg::CompareScalar<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
        Reg::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        Reg::Muls(Reg, Reg, FP4_INV_SCALE_FACTOR, pregAll32);
    } else {
        Reg::Duplicate(maxExpFP32, FP32_MX_MAX_EXP);
        Reg::And(exp0FP32, (Reg::RegTensor<int32_t>&)Reg, maxExpFP32, pregAll32);
        Reg::ShiftRights(exp0FP32, exp0FP32, FP32_SHR_NUM, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, FP32_BIAS_NEG_VALUE, pregAll32);
        Reg::Maxs(exp0FP32, exp0FP32, 0, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, FP32_NEG_ONE, pregAll32);
        Reg::Muls(exp1FP32, exp0FP32, FP32_NEG_ONE, pregAll32);
        Reg::Adds(exp1FP32, exp1FP32, FP32_BIAS_VALUE, pregAll32);
        Reg::ShiftLefts(exp1FP32, exp1FP32, FP32_SHR_NUM, pregAll32);

        Reg::Mul(Reg, Reg, (Reg::RegTensor<float>&)exp1FP32, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, FP32_BIAS_VALUE, pregAll32);
        Reg::ShiftLefts(exp0FP32, exp0FP32, FP32_SHR_NUM, pregAll32);
        Reg::CompareScalar<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
        Reg::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        Reg::Mul(Reg, Reg, (Reg::RegTensor<float>&)exp0FP32, pregAll32);
    }
    Reg::CompareScalar<float, CMPMODE::EQ>(zeroMask, Reg, 0, pregAll32);
    Reg::MaskAnd(zeroMask, specialMask, zeroMask, pregAll32);
    Reg::MaskOr(zeroMask, negInfMask, zeroMask, pregAll32);
    Reg::Select<int32_t>((Reg::RegTensor<int32_t>&)Reg, negZero, (Reg::RegTensor<int32_t>&)Reg, zeroMask);
}

} // namespace DynamicMxQuant
#endif // DYNAMIC_MX_QUANT_TAIL_AXIS_H
