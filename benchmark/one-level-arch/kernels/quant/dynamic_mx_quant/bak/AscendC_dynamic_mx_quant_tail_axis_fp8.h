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
 * \file dynamic_mx_quant_tail_axis_fp8.h
 * \brief
 */

#ifndef DYNAMIC_MX_QUANT_TAIL_AXIS_FP8_H
#define DYNAMIC_MX_QUANT_TAIL_AXIS_FP8_H
#define FLOAT_OVERFLOW_MODE_CTRL 60
#include "kernel_operator.h"
#include "dynamic_mx_quant_common.h"
namespace DynamicMxQuant {
using namespace AscendC;

template <typename T, typename U, int64_t SCALE_ALG>
class DynamicMxQuantTailAxisFP8 {
public:
    __aicore__ inline DynamicMxQuantTailAxisFP8() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR mxScale,
                                const DynamicMxQuantTailAxisTilingData* tilingData);
    __aicore__ inline void Process();

private:
    using intCalcType = typename std::conditional<IsSame<T, float>::value && (SCALE_ALG == 1), uint32_t,
                                                  uint16_t>::type;
    __aicore__ inline void ParseTilingData(const DynamicMxQuantTailAxisTilingData* tilingData);
    __aicore__ inline void GetGmParams();
    __aicore__ inline void GetUbParams();

    __aicore__ inline void CopyIn(int64_t dim0LoopIdx, int64_t dim1LoopIdx, int64_t ubFactorRowNum,
                                  int64_t ubFactorColNum, int64_t ubFactorColBlockNum);
    __aicore__ inline void Compute(int64_t ubFactorRowBlockNum, int64_t ubFactorColBlockNum);
    __aicore__ inline void CopyOut(int64_t dim0LoopIdx, int64_t dim1LoopIdx, int64_t ubFactorRowNum,
                                   int64_t ubFactorColNum, int64_t ubFactorColBlockNum);
    __aicore__ inline void ComputeMaxExpOcpHalf(__ubuf__ half* xLocalAddr, __ubuf__ uint16_t* maxExpAddr,
                                                uint16_t loopNum2VF);
    __aicore__ inline void ComputeMaxExpOcpBf16(__ubuf__ bfloat16_t* xLocalAddr, __ubuf__ uint16_t* maxExpAddr,
                                                uint16_t loopNum2VF);
    __aicore__ inline void ComputeMaxExpOcpFp32(__ubuf__ float* xLocalAddr, __ubuf__ uint16_t* maxExpAddr,
                                                uint16_t loopNum2VF);
    __aicore__ inline void ComputeScaleOcp(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr,
                                           __ubuf__ uint16_t* recipScaleLocalAddr, uint16_t loopNum1VF,
                                           uint32_t totalScaleInUB);
    __aicore__ inline void ComputeMaxExpCublasHalf(__ubuf__ half* xLocalAddr, __ubuf__ uint16_t* maxExpAddr,
                                                   uint16_t loopNum2VF);
    __aicore__ inline void ComputeMaxExpCublasBf16(__ubuf__ bfloat16_t* xLocalAddr, __ubuf__ uint16_t* maxExpAddr,
                                                   uint16_t loopNum2VF);
    __aicore__ inline void ComputeMaxExpCublasFp32(__ubuf__ float* xLocalAddr, __ubuf__ uint32_t* maxExpAddr,
                                                   uint16_t loopNum2VF);
    __aicore__ inline void ComputeScaleCublas(__ubuf__ intCalcType* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr,
                                              __ubuf__ uint16_t* recipScaleLocalAddr, uint16_t loopNumHalfVF,
                                              uint32_t totalScaleInUB);
    __aicore__ inline void ComputeData(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr,
                                       __ubuf__ int8_t* yLocalAddr, uint16_t loopNum2VF);

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

    uint16_t f8Emax_ = 0;
    uint32_t dtypeMax_ = 0;
    float maxLowBound_ = 0.0f;
};

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, const DynamicMxQuantTailAxisTilingData* tilingData)
{
#if (__NPU_ARCH__ == 3510)
    SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    ParseTilingData(tilingData); // 获取TilingData数据

    GetGmParams(); // 计算核间GM地址偏移

    GetUbParams(); // 计算核内切分参数

    int64_t scaleBufferBlockNum = Ops::Base::CeilAlign(maxUbBlockNum_, static_cast<int64_t>(VF_LEN_16));
    pipe_.InitBuffer(inQueue_, DB_BUFFER, maxUbBlockNum_ * blockSize_ * sizeof(T));
    pipe_.InitBuffer(outQueue_, DB_BUFFER, maxUbBlockNum_ * blockSize_ * sizeof(uint8_t));
    pipe_.InitBuffer(mxScaleQueue_, DB_BUFFER, scaleBufferBlockNum * sizeof(uint8_t));
    pipe_.InitBuffer(maxExpBuffer_, maxUbBlockNum_ * sizeof(T) * 2);
    pipe_.InitBuffer(recipScaleBuffer_, maxUbBlockNum_ * sizeof(uint16_t) * 2);

    xGm_.SetGlobalBuffer((__gm__ T*)x + xGmOffset_);
    yGm_.SetGlobalBuffer((__gm__ uint8_t*)y + xGmOffset_);
    scaleGm_.SetGlobalBuffer((__gm__ uint8_t*)mxScale + scaleGmOffset_);

    if constexpr (IsSame<U, fp8_e4m3fn_t>::value) {
        f8Emax_ = FP8_E4M3_MAX_EXP;
        dtypeMax_ = FP8_E4M3_MAX_FLOAT_BITS;
    } else {
        f8Emax_ = FP8_E5M2_MAX_EXP;
        dtypeMax_ = FP8_E5M2_MAX_FLOAT_BITS;
    }
}

__aicore__ inline void CastBf16ScaleToFloat(Reg::RegTensor<float>& dst, Reg::RegTensor<uint16_t>& src,
                                            Reg::MaskReg& mask)
{
    // 规避utest GCC ICE bug
    static constexpr Reg::CastTrait trait = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                             RoundMode::UNKNOWN};
    Reg::Cast<float, bfloat16_t, trait>(dst, (Reg::RegTensor<bfloat16_t>&)src, mask);
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::GetGmParams()
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
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::GetUbParams()
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
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::ParseTilingData(
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
    maxLowBound_ = tilingData->maxLowBound;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::Process()
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
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::CopyIn(int64_t dim0LoopIdx, int64_t dim1LoopIdx,
                                                                          int64_t ubFactorRowNum,
                                                                          int64_t ubFactorColNum,
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
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::CopyOut(int64_t dim0LoopIdx, int64_t dim1LoopIdx,
                                                                           int64_t ubFactorRowNum,
                                                                           int64_t ubFactorColNum,
                                                                           int64_t ubFactorColBlockNum)
{
    LocalTensor<uint8_t> scaleLocal = mxScaleQueue_.DeQue<uint8_t>();
    LocalTensor<uint8_t> yLocal = outQueue_.DeQue<uint8_t>();
    int64_t scaleIsNotOdd = ubFactorColBlockNum % DIGIT_TWO;

    DataCopyExtParams copyOutParamY = {
        static_cast<uint16_t>(ubFactorRowNum), static_cast<uint32_t>(ubFactorColNum * sizeof(uint8_t)),
        static_cast<uint32_t>(scaleIsNotOdd), // 一个BlockSize=32，输出y为1 Bytes，搬入UB需要跳32Bytes，srcStride = 1
        static_cast<uint32_t>((colNum_ - ubFactorColNum) * sizeof(uint8_t)), static_cast<uint32_t>(0)};

    int64_t offset = dim0LoopIdx * ubFactorRowNormalBlockNum_ * colNum_ + dim1LoopIdx * ubFactorColNormalBlockLen_;
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

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::ComputeMaxExpOcpBf16(__ubuf__ bfloat16_t* xLocalAddr,
                                                                                        __ubuf__ uint16_t* maxExpAddr,
                                                                                        uint16_t loopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<bfloat16_t> x0;
        Reg::RegTensor<bfloat16_t> x1;
        Reg::RegTensor<uint16_t> xMaxExp;
        Reg::RegTensor<uint16_t> xExpExtract0;
        Reg::RegTensor<uint16_t> xExpExtract1;

        Reg::RegTensor<uint16_t> expMaskBF16;
        Reg::Duplicate(expMaskBF16, BF16_MAX_EXP);

        Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::UnalignReg ureg;

        for (uint16_t i = 0; i < loopNum2VF; i++) {
            Reg::LoadAlign<bfloat16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xLocalAddr, VF_LEN_16_DOUBLE);
            Reg::And(xExpExtract0, (Reg::RegTensor<uint16_t>&)x0, expMaskBF16, Mask);
            Reg::And(xExpExtract1, (Reg::RegTensor<uint16_t>&)x1, expMaskBF16, Mask);
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
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::ComputeMaxExpCublasBf16(
    __ubuf__ bfloat16_t* xLocalAddr, __ubuf__ uint16_t* maxExpAddr, uint16_t loopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<bfloat16_t> x0;
        Reg::RegTensor<bfloat16_t> x1;
        Reg::RegTensor<uint16_t> xMaxExp;

        Reg::RegTensor<uint16_t> absMask16Bit;
        Reg::Duplicate(absMask16Bit, BF16_ABS_MASK);

        Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::UnalignReg ureg;

        for (uint16_t i = 0; i < loopNum2VF; i++) {
            Reg::LoadAlign<bfloat16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xLocalAddr, VF_LEN_16_DOUBLE); // 解交织搬入，每个寄存器内256字节，对于B16输入是128元素，0,1,2,3...127,128,129,130...255 -> 0,2,4,6...254  1,3,5,7...255
            Reg::And((Reg::RegTensor<uint16_t>&)x0, (Reg::RegTensor<uint16_t>&)x0, absMask16Bit, Mask);
            Reg::And((Reg::RegTensor<uint16_t>&)x1, (Reg::RegTensor<uint16_t>&)x1, absMask16Bit, Mask); // 绝对值
            Reg::Max(xMaxExp, (Reg::RegTensor<uint16_t>&)x0, (Reg::RegTensor<uint16_t>&)x1, Mask); // 每相邻两个数的最大值->1,3,5,7...255
            Reg::ReduceMaxWithDataBlock(xMaxExp, xMaxExp, Mask);// 每个UBBlock(32Byte)内求最大值（相邻16个数的最大值）-> 得到32元素的最大值
            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(maxExpAddr, xMaxExp, ureg,
                                                                            ELEMENT_AFTER_REDUCE_);
        }
        Reg::StoreUnAlignPost(maxExpAddr, ureg, 0);
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::ComputeMaxExpOcpHalf(__ubuf__ half* xLocalAddr,
                                                                                        __ubuf__ uint16_t* maxExpAddr,
                                                                                        uint16_t loopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<half> x0;
        Reg::RegTensor<half> x1;
        Reg::RegTensor<uint16_t> xMaxExp;
        Reg::RegTensor<uint16_t> xExpSelect0;
        Reg::RegTensor<uint16_t> xExpSelect1;
        Reg::RegTensor<uint16_t> xExpExtract0;
        Reg::RegTensor<uint16_t> xExpExtract1;
        Reg::RegTensor<bfloat16_t> x0BF16;
        Reg::RegTensor<bfloat16_t> x1BF16;

        Reg::RegTensor<uint16_t> expMaskBF16;
        Reg::Duplicate(expMaskBF16, BF16_MAX_EXP);
        Reg::RegTensor<uint16_t> invalidMaskFP16;
        Reg::Duplicate(invalidMaskFP16, FP16_INVALID);

        Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg invalidDataMask0;
        Reg::MaskReg invalidDataMask1;
        Reg::UnalignReg ureg;

        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                              Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};

        for (uint16_t i = 0; i < loopNum2VF; i++) {
            Reg::LoadAlign<half, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(x0, x1, xLocalAddr,
                                                                                                     VF_LEN_16_DOUBLE);
            Reg::And(xExpSelect0, (Reg::RegTensor<uint16_t>&)x0, invalidMaskFP16, Mask);
            Reg::And(xExpSelect1, (Reg::RegTensor<uint16_t>&)x1, invalidMaskFP16, Mask);
            Reg::Compare<uint16_t, CMPMODE::NE>(invalidDataMask0, xExpSelect0, invalidMaskFP16, Mask);
            Reg::Compare<uint16_t, CMPMODE::NE>(invalidDataMask1, xExpSelect1, invalidMaskFP16, Mask);
            Reg::Cast<bfloat16_t, half, castTraitHalf2Bf16>(x0BF16, x0, Mask);
            Reg::Cast<bfloat16_t, half, castTraitHalf2Bf16>(x1BF16, x1, Mask);
            Reg::And(xExpExtract0, (Reg::RegTensor<uint16_t>&)x0BF16, expMaskBF16, Mask);
            Reg::And(xExpExtract1, (Reg::RegTensor<uint16_t>&)x1BF16, expMaskBF16, Mask);
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
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::ComputeMaxExpCublasHalf(
    __ubuf__ half* xLocalAddr, __ubuf__ uint16_t* maxExpAddr, uint16_t loopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<half> x0;
        Reg::RegTensor<half> x1;
        Reg::RegTensor<uint16_t> xMaxExp;

        Reg::RegTensor<uint16_t> absMask16Bit;
        Reg::Duplicate(absMask16Bit, BF16_ABS_MASK);

        Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::UnalignReg ureg;

        for (uint16_t i = 0; i < loopNum2VF; i++) {
            Reg::LoadAlign<half, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(x0, x1, xLocalAddr,
                                                                                                     VF_LEN_16_DOUBLE);
            Reg::And((Reg::RegTensor<uint16_t>&)x0, (Reg::RegTensor<uint16_t>&)x0, absMask16Bit, Mask);
            Reg::And((Reg::RegTensor<uint16_t>&)x1, (Reg::RegTensor<uint16_t>&)x1, absMask16Bit, Mask);
            Reg::Max(xMaxExp, (Reg::RegTensor<uint16_t>&)x0, (Reg::RegTensor<uint16_t>&)x1, Mask);
            Reg::ReduceMaxWithDataBlock(xMaxExp, xMaxExp, Mask);
            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(maxExpAddr, xMaxExp, ureg,
                                                                            ELEMENT_AFTER_REDUCE_);
        }
        Reg::StoreUnAlignPost(maxExpAddr, ureg, 0);
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::ComputeMaxExpOcpFp32(__ubuf__ float* xLocalAddr,
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
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::ComputeMaxExpCublasFp32(
    __ubuf__ float* xLocalAddr, __ubuf__ uint32_t* maxExpAddr, uint16_t loopNum2VF)
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

        Reg::RegTensor<uint32_t> absMask32Bit;
        Reg::Duplicate(absMask32Bit, FP32_ABS_MASK);

        Reg::MaskReg MaskB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::UnalignReg ureg;

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
            Reg::And((Reg::RegTensor<uint32_t>&)x0ZeroFP32, (Reg::RegTensor<uint32_t>&)x0ZeroFP32, absMask32Bit,
                     MaskB32);
            Reg::And((Reg::RegTensor<uint32_t>&)x0OneFP32, (Reg::RegTensor<uint32_t>&)x0OneFP32, absMask32Bit, MaskB32);
            Reg::And((Reg::RegTensor<uint32_t>&)x1ZeroFP32, (Reg::RegTensor<uint32_t>&)x1ZeroFP32, absMask32Bit,
                     MaskB32);
            Reg::And((Reg::RegTensor<uint32_t>&)x1OneFP32, (Reg::RegTensor<uint32_t>&)x1OneFP32, absMask32Bit, MaskB32);
            Reg::Max(x0MaxExpB32, (Reg::RegTensor<uint32_t>&)x0ZeroFP32, (Reg::RegTensor<uint32_t>&)x0OneFP32, MaskB32);
            Reg::Max(x1MaxExpB32, (Reg::RegTensor<uint32_t>&)x1ZeroFP32, (Reg::RegTensor<uint32_t>&)x1OneFP32, MaskB32);
            Reg::Max(xMaxExpB32, x0MaxExpB32, x1MaxExpB32, MaskB32);
            Reg::ReduceMaxWithDataBlock(xMaxExpB32, xMaxExpB32, MaskB32);
            Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(maxExpAddr, xMaxExpB32, ureg,
                                                                            ELEMENT_AFTER_REDUCE_);
        }
        Reg::StoreUnAlignPost(maxExpAddr, ureg, 0);
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::ComputeScaleOcp(
    __ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr,
    uint16_t loopNum1VF, uint32_t totalScaleInUB)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> x0;
        Reg::RegTensor<T> x1;
        Reg::RegTensor<uint16_t> xMaxExp;
        Reg::RegTensor<uint16_t> sharedExp;
        Reg::RegTensor<uint16_t> scaleValue;
        Reg::RegTensor<uint16_t> halfScale;

        Reg::RegTensor<uint16_t> expMask;
        Reg::Duplicate(expMask, BF16_MAX_EXP);
        Reg::RegTensor<uint16_t> maxExpValue;
        Reg::Duplicate(maxExpValue, f8Emax_);
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
        Reg::MaskReg cmpResultSub;
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
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::ComputeScaleCublas(
    __ubuf__ intCalcType* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr,
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

        Reg::RegTensor<uint32_t> invMax;
        Reg::Duplicate(invMax, dtypeMax_);
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
        Reg::RegTensor<uint32_t> fp8NanU32;
        Reg::Duplicate(fp8NanU32, FP8_MAX_EXP_IN_FP32);

        Reg::MaskReg cmpResult;
        Reg::MaskReg zeroMask;
        Reg::MaskReg p0;
        Reg::MaskReg p1;
        Reg::MaskReg p2;
        uint32_t SixtyFour = 64;
        Reg::MaskReg dataMaskB16Half = Reg::UpdateMask<uint16_t>(SixtyFour);
        Reg::MaskReg maskFloat = Reg::CreateMask<uint32_t>();

        static constexpr Reg::CastTrait castTrait2Float = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                           Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitBf162Float = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                               Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        for (uint16_t i = 0; i < loopNumHalfVF; i++) {
            if constexpr (IsSame<T, float>::value) {
                Reg::LoadAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_NORM>(
                    max32, maxExpAddr, VF_LEN_32);
            } else {
                Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(
                    max16, maxExpAddr, VF_LEN_32);
                Reg::Cast<float, T, castTrait2Float>((Reg::RegTensor<float>&)max32, (Reg::RegTensor<T>&)max16,
                                                     maskFloat);
            }
            Reg::Compare<uint32_t, CMPMODE::LT>(cmpResult, max32, expMask, maskFloat); // 小于inf
            Reg::Compare<uint32_t, CMPMODE::NE>(zeroMask, max32, zeroU32, maskFloat); // 非0

            Reg::Maxs((Reg::RegTensor<float>&)max32, (Reg::RegTensor<float>&)max32, maxLowBound_, maskFloat); // 下限钳位

            Reg::Mul((Reg::RegTensor<float>&)max32, (Reg::RegTensor<float>&)max32, (Reg::RegTensor<float>&)invMax,
                     maskFloat); // S^b_fp32
            Reg::ShiftRights(exp32, max32, FP32_SHR_NUM, maskFloat);// 无偏指数位
            Reg::And(man32, max32, manMaskFP32, maskFloat);// 无偏尾数位

            Reg::CompareScalar<uint32_t, CMPMODE::GT>(p0, exp32, FP32_NUMBER_ZERO, maskFloat);
            Reg::CompareScalar<uint32_t, CMPMODE::LT>(p1, exp32, FP32_NUMBER_254, maskFloat);
            Reg::CompareScalar<uint32_t, CMPMODE::GT>(p2, man32, FP32_NUMBER_ZERO, maskFloat);
            Reg::MaskAnd(p0, p0, p1, maskFloat);
            Reg::MaskAnd(p0, p0, p2, maskFloat); // 0<指数位<254 (正规数) 且 无偏尾数位>0

            Reg::CompareScalar<uint32_t, CMPMODE::EQ>(p1, exp32, FP32_NUMBER_ZERO, maskFloat);
            Reg::CompareScalar<uint32_t, CMPMODE::GT>(p2, man32, FP32_NUMBER_HALF, maskFloat);
            Reg::MaskAnd(p1, p1, p2, maskFloat);// p1:无偏指数=0（非正规数） 且 无偏尾数>0.5
            Reg::MaskOr(p0, p0, p1, maskFloat);//p0: （0<指数位<254 (正规数) 且 无偏尾数位>0） 或 （无偏指数=0（非正规数） 且 无偏尾数>0.5）

            Reg::Adds(expAddOne32, exp32, 1, maskFloat); // E^b_int + 1
            Reg::Select(extractExp, expAddOne32, exp32, p0); // 最终的E^b_int
            Reg::Select<uint32_t>(extractExp, extractExp, fp8NanU32, cmpResult);
            Reg::Select<uint32_t>(extractExp, extractExp, zeroU32, zeroMask);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(expOut, extractExp);

            // 64 个 max 计算得到 64 * 1 Bytes = 32 * sizeof(uint16_t) Bytes
            Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr + i * 32, expOut,
                                                                     dataMaskB16Half);

            Reg::ShiftLefts(extractExp, extractExp, BF16_SHR_NUM, maskFloat);
            Reg::Sub(halfScale, scaleBias, extractExp, maskFloat);
            Reg::Select<uint32_t>(halfScale, halfScale, nanU32, cmpResult);
            Reg::Select<uint32_t>(halfScale, halfScale, zeroU32, zeroMask);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(recExpOut, halfScale);

            Reg::StoreAlign<uint16_t>(recipScaleLocalAddr + i * VF_LEN_32, recExpOut, dataMaskB16Half);
        }
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::ComputeData(__ubuf__ T* xLocalAddr,
                                                                               __ubuf__ uint16_t* recipScaleLocalAddr,
                                                                               __ubuf__ int8_t* yLocalAddr,
                                                                               uint16_t loopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::MaskReg dataMask1 = Reg::CreateMask<T>();
        Reg::MaskReg dataMask2 = Reg::CreateMask<T>();
        Reg::MaskReg dataMask3 = Reg::CreateMask<T>();
        Reg::MaskReg dataMask4 = Reg::CreateMask<T>();
        Reg::MaskReg dataMask5 = Reg::CreateMask<U>();
        Reg::MaskReg maskB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskFP32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::RegTensor<uint16_t> halfScaleForMul;
        Reg::RegTensor<float> floatScaleForMul;
        Reg::RegTensor<T> x0;
        Reg::RegTensor<T> x1;
        Reg::RegTensor<float> x0ZeroFP32;
        Reg::RegTensor<float> x0OneFP32;
        Reg::RegTensor<float> x1ZeroFP32;
        Reg::RegTensor<float> x1OneFP32;
        Reg::RegTensor<U> x0ZeroFP8;
        Reg::RegTensor<U> x0OneFP8;
        Reg::RegTensor<U> x1ZeroFP8;
        Reg::RegTensor<U> x1OneFP8;

        static constexpr Reg::CastTrait castTraitZero = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitOne = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTrait32to80 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                           Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr Reg::CastTrait castTrait32to81 = {Reg::RegLayout::ONE, Reg::SatMode::SAT,
                                                           Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr Reg::CastTrait castTrait32to82 = {Reg::RegLayout::TWO, Reg::SatMode::SAT,
                                                           Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr Reg::CastTrait castTrait32to83 = {Reg::RegLayout::THREE, Reg::SatMode::SAT,
                                                           Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

        for (uint16_t i = 0; i < loopNum2VF; i++) {
            if constexpr (IsSame<T, float>::value) {
                Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                    halfScaleForMul, recipScaleLocalAddr, ELEMENT_AFTER_REDUCE_);
                CastBf16ScaleToFloat(floatScaleForMul, halfScaleForMul, maskFP32);
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
                Reg::Mul(x0ZeroFP32, x0ZeroFP32, floatScaleForMul, maskFP32);
                Reg::Mul(x1ZeroFP32, x1ZeroFP32, floatScaleForMul, maskFP32);
                Reg::Mul(x0OneFP32, x0OneFP32, floatScaleForMul, maskFP32);
                Reg::Mul(x1OneFP32, x1OneFP32, floatScaleForMul, maskFP32);
            } else {
                Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                    x0, x1, xLocalAddr, VF_LEN_16_DOUBLE);
                Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                    halfScaleForMul, recipScaleLocalAddr, ELEMENT_AFTER_REDUCE_);
                if constexpr (IsSame<T, half>::value) {
                    Reg::Cast<float, T, castTraitZero>(x0ZeroFP32, x0, dataMask1);
                    Reg::Cast<float, T, castTraitOne>(x0OneFP32, x0, dataMask1);
                    CastBf16ScaleToFloat(floatScaleForMul, halfScaleForMul, maskB16);
                    Reg::Mul(x0ZeroFP32, x0ZeroFP32, floatScaleForMul, dataMask3);
                    Reg::Mul(x0OneFP32, x0OneFP32, floatScaleForMul, dataMask4);

                    Reg::Cast<float, T, castTraitZero>(x1ZeroFP32, x1, dataMask1);
                    Reg::Cast<float, T, castTraitOne>(x1OneFP32, x1, dataMask1);
                    Reg::Mul(x1ZeroFP32, x1ZeroFP32, floatScaleForMul, dataMask3);
                    Reg::Mul(x1OneFP32, x1OneFP32, floatScaleForMul, dataMask4);
                } else {
                    Reg::Mul(x0, x0, (Reg::RegTensor<T>&)halfScaleForMul, dataMask1);
                    Reg::Mul(x1, x1, (Reg::RegTensor<T>&)halfScaleForMul, dataMask1);

                    Reg::Cast<float, T, castTraitZero>(x0ZeroFP32, x0, dataMask1);
                    Reg::Cast<float, T, castTraitOne>(x0OneFP32, x0, dataMask1);
                    Reg::Cast<float, T, castTraitZero>(x1ZeroFP32, x1, dataMask2);
                    Reg::Cast<float, T, castTraitOne>(x1OneFP32, x1, dataMask2);
                }
            }
            Reg::Cast<U, float, castTrait32to80>(x0ZeroFP8, x0ZeroFP32, dataMask3);
            Reg::Cast<U, float, castTrait32to81>(x1ZeroFP8, x1ZeroFP32, dataMask4);
            Reg::Cast<U, float, castTrait32to82>(x0OneFP8, x0OneFP32, dataMask3);
            Reg::Cast<U, float, castTrait32to83>(x1OneFP8, x1OneFP32, dataMask4);

            Reg::Add((Reg::RegTensor<uint8_t>&)x0ZeroFP8, (Reg::RegTensor<uint8_t>&)x0ZeroFP8,
                     (Reg::RegTensor<uint8_t>&)x0OneFP8, dataMask5);
            Reg::Add((Reg::RegTensor<uint8_t>&)x1ZeroFP8, (Reg::RegTensor<uint8_t>&)x1ZeroFP8,
                     (Reg::RegTensor<uint8_t>&)x1OneFP8, dataMask5);
            Reg::Add((Reg::RegTensor<uint8_t>&)x0ZeroFP8, (Reg::RegTensor<uint8_t>&)x0ZeroFP8,
                     (Reg::RegTensor<uint8_t>&)x1ZeroFP8, dataMask5);

            Reg::StoreAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM_B8>(
                yLocalAddr, (Reg::RegTensor<int8_t>&)x0ZeroFP8, VF_LEN_16_DOUBLE, dataMask5);
        }
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::Compute(int64_t ubFactorRowBlockNum,
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
    // float且scaleAlg=1时，保留位数位，用原始值计算，所以maxExpLocal是uint32_t
    LocalTensor<intCalcType> maxExpLocal = maxExpBuffer_.Get<intCalcType>();
    LocalTensor<uint16_t> recipScaleLocal = recipScaleBuffer_.Get<uint16_t>();

    auto xLocalAddr = reinterpret_cast<__ubuf__ T*>(xLocal.GetPhyAddr());
    auto scaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(scaleLocal.GetPhyAddr());
    auto yLocalAddr = reinterpret_cast<__ubuf__ int8_t*>(yLocal.GetPhyAddr());
    auto maxExpLocalAddr = reinterpret_cast<__ubuf__ intCalcType*>(maxExpLocal.GetPhyAddr());
    auto recipScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(recipScaleLocal.GetPhyAddr());

    if constexpr (SCALE_ALG == 0) {
        if constexpr (IsSame<T, float>::value) {
            ComputeMaxExpOcpFp32(xLocalAddr, maxExpLocalAddr, loopNum2VF);
        } else if constexpr (IsSame<T, bfloat16_t>::value) {
            ComputeMaxExpOcpBf16(xLocalAddr, maxExpLocalAddr, loopNum2VF);
        } else {
            ComputeMaxExpOcpHalf(xLocalAddr, maxExpLocalAddr, loopNum2VF);
        }
        ComputeScaleOcp(maxExpLocalAddr, scaleLocalAddr, recipScaleLocalAddr, loopNum1VF, totalBlockNum);
        ComputeData(xLocalAddr, recipScaleLocalAddr, yLocalAddr, loopNum2VF);
    } else if constexpr (SCALE_ALG == 1) {
        if constexpr (IsSame<T, float>::value) {
            ComputeMaxExpCublasFp32(xLocalAddr, maxExpLocalAddr, loopNum2VF);
        } else if constexpr (IsSame<T, bfloat16_t>::value) {
            ComputeMaxExpCublasBf16(xLocalAddr, maxExpLocalAddr, loopNum2VF);
        } else {
            ComputeMaxExpCublasHalf(xLocalAddr, maxExpLocalAddr, loopNum2VF);
        }
        ComputeScaleCublas(maxExpLocalAddr, scaleLocalAddr, recipScaleLocalAddr, loopNumHalfVF, totalBlockNum);
        ComputeData(xLocalAddr, recipScaleLocalAddr, yLocalAddr, loopNum2VF);
    }

    inQueue_.FreeTensor(xLocal);
    mxScaleQueue_.EnQue(scaleLocal);
    outQueue_.EnQue(yLocal);
    return;
}

} // namespace DynamicMxQuant
#endif // DYNAMIC_MX_QUANT_TAIL_AXIS_FP8_H
