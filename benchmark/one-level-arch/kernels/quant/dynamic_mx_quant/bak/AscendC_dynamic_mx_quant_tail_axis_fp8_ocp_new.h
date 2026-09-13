constexpr uint16_t BF16_ONE = 0x3f80;

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
    yMaxExpRecip = BF16_ONE - f8Emax_;
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
        Reg::RegTensor<uint16_t> fp8MaxExpRecipRegTensor;
        Reg::Duplicate(fp8MaxExpRecipRegTensor, yMaxExpRecip);
        Reg::RegTensor<uint16_t> specialExpU16;
        Reg::Duplicate(specialExpU16, BF16_SPECIAL_EXP_THRESHOLD);

        Reg::MaskReg cmpResult;
        Reg::MaskReg zeroMask;
        Reg::MaskReg cmpResultSub;
        Reg::MaskReg preMaskScale;
        Reg::MaskReg invalidDataMask;
        Reg::MaskReg specialDataMask;
        constexpr AscendC::MicroAPI::CastTrait castTraitBf16ToE8m0 = {
            Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
            Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};

        for (uint16_t i = 0; i < loopNum1VF; i++) {
            preMaskScale = Reg::UpdateMask<uint16_t>(totalScaleInUB);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(xMaxExp, maxExpAddr, VF_LEN_16);
            Reg::Compare<uint16_t, CMPMODE::NE>(cmpResult, xMaxExp, expMask, preMaskScale); // INF/NAN
            Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, xMaxExp, zeroU16, preMaskScale);

            Reg::Mul((Reg::RegTensor<bfloat16_t>&)sharedExp, (Reg::RegTensor<bfloat16_t>&)xMaxExp, (Reg::RegTensor<bfloat16_t>&)fp8MaxExpRecipRegTensor, preMaskScale);
            Reg::Cast<fp8_e8m0_t, bfloat16_t, castTraitBf16ToE8m0>((Reg::RegTensor<fp8_e8m0_t>&)scaleValue, (Reg::RegTensor<bfloat16_t>&)sharedExp, preMaskScale);
            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr, scaleValue, VF_LEN_32, preMaskScale);

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
