#ifndef QLI_PTO_OPT_DYNAMIC_HPP
#define QLI_PTO_OPT_DYNAMIC_HPP

// =============================================================================
// qli_pto_opt_dynamic.hpp — QLI 动态 shape + 多 PE 版本
// =============================================================================
// 与模板版 qli_pto_opt_simple.hpp 的差异：
//   - Sq/Skv/topK 为运行时参数（非模板参数）
//   - 多 PE 支持：每 PE 处理部分 token，get_thread_idx() 分配
//   - 固定 tile 尺寸（kTm=16, kTk=32, kD=128, MaxTileCol=2048）
//   - 末 chunk 不足 2048 列时用 0 填充（key=0 不会进入 TopK）
//
// 【内存契约】
//   - scores_gm: [Sq, paddedSkv]，paddedSkv = ceil(Skv/2048)*2048
//     （TopK 末 chunk 整块 2048 读取，越界读取由调用方保证安全）
//   - key_scratch: [Sq, paddedSkv] uint32，末 chunk 填充区由 kernel 清零
//   - indices_gm: [Sq, topK] int32
//   - hist/prefix scratch: 紧随 key_scratch 之后（1KB + 128B）
// =============================================================================

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <type_traits>

using namespace pto;

// ========== 编译期常量 ==========
constexpr int kD = 128;
constexpr int kTm = 16;
constexpr int kTk = 32;
constexpr int G = 64;
constexpr int Gb = G / kTm;
constexpr int MaxTileCol = 2048;
constexpr int MaxSkv = 16384;
constexpr int MaxSq = 8192;

// ========== tile 类型别名（Vec 链，TSTORE_CUBE 桥接后使用）==========
using tileS_t = Tile<Location::Vec, float, kTm, kTk, BLayout::RowMajor>;
using tileWb_t = Tile<Location::Vec, float, kTm, kTk, BLayout::RowMajor>;
using tileSum_t = Tile<Location::Vec, float, 1, kTk, BLayout::RowMajor>;
using tileSk_t = tileSum_t;

using itSk_t = global_iterator<global_tensor<float, RowMajor<1, MaxSkv>>, tileSk_t>;
using itOut_t = itSk_t;

// ========== Step 1-6: 动态 shape + 多 PE ==========
// 单 PE（numPEs==1）：local CUBE tile（CubeTileM16/N8 + TLOAD_CUBE），
//   TMATMUL 输出 CubeAccumulatorM16 → TSTORE_CUBE 桥接回 Vec tile 继续链
// 多 PE（numPEs>1）：cooperative SharedMatrix（GM→Shared 直载
//   TLOAD<Matrix,1>，kGroupM=16、kPeM=16），TMATMUL 每 PE 得 CUBE [16,32]
//   私有切片 → TSTORE_CUBE 桥接回 Vec。
// 约束：Sq % numPEs == 0（各 PE 迭代次数一致，保证集体指令同步）。
// TCOLSUMX — 列求和（本仓 b8669ce 模板 lb1 误绑 dst.ValidRow=1，只累加 1 行；
// 按上游 f00b928 (#101) 的正确编码：B.DIM 描述源几何，LB1=源 ValidRow）
template <typename tile_o, typename tile_i>
void TCOLSUMX(tile_o& dst, tile_i& src) {
  asm volatile(
    "BSTART.TEPL 80, %D1\n"
    "B.DIM zero, %c2, ->lb0\n"
    "B.DIM zero, %c3, ->lb1\n"
    "B.DIM zero, %c4, ->lb2\n"
    "B.IOT %5, mask=1111, last, ->%0<%Z6>\n"
    ""
    : "=Tr"(dst.data())
    : "i"(type_traits<typename tile_i::DType>::TypeCode),
      "i"(tile_i::ValidCol),
      "i"(tile_i::ValidRow),
      "i"(tile_i::Cols),
      "Tr"(src.data()),
      "i"(tile_type_traits<typename tile_o::TileDType>::TilesizeCode)
  );
}

// PTO v0.58.4 CUBE cell-layout：A=SharedMatrixLeft、B=SharedMatrixRight、
// D=CubeAccumulatorM16（kTm=16 → kGroupM=16、kPeM=16），TSTORE_CUBE 桥接回 Vec。
template <typename dtype>
inline void qli_pto_dynamic(float* scores_ptr, dtype* q_ptr, dtype* k_ptr,
                            float* wb_ptr, float* scale_k_ptr,
                            int Sq, int Skv, int numPEs, float* temp_gm = nullptr)
{
    const uint32_t tid = get_thread_idx();
    int Kb = Skv / kTk;
    int paddedSkv = ((Skv + MaxTileCol - 1) / MaxTileCol) * MaxTileCol;

    // 临时区缺省：紧随 scores 之后
    if (temp_gm == nullptr) {
        temp_gm = scores_ptr + (uint64_t)Sq * paddedSkv + 2048;
    }

    // CUBE tile 类型（单/多 PE 共用声明）
    using tileQCube_t = std::conditional_t<(kTm <= 16),
                          CubeTileM16<dtype, kTm, kD>, CubeTileM32<dtype, kTm, kD>>;
    using tileKCube_t = CubeTileN8<dtype, kD, kTk>;
    using tileSCube_t = std::conditional_t<(kTm <= 16),
                          CubeAccumulatorM16<float, kTm, kTk>,
                          CubeAccumulatorM32<float, kTm, kTk>>;
    using itQCube_t = global_iterator<global_tensor<dtype, RowMajor<MaxSq, kD>>, tileQCube_t>;
    using itKCube_t = global_iterator<global_tensor<dtype, RowMajor<kD, MaxSkv>>, tileKCube_t>;
    using gmTmp_t = global_tensor<float, RowMajor<kTm, kTk>>;

    if (numPEs == 1) {
        // ---- 单 PE：FP8 直通（local CUBE tile）----
        for (int i = tid; i < Sq; i += numPEs) {
            itQCube_t gQ(q_ptr + (uint64_t)i * G * kD);
            itOut_t gOut(scores_ptr + (uint64_t)i * paddedSkv);
            // K^T [kD, Skv] 行主序：每列块 j 用 [kD, kTk] 子矩阵（真实步长 kTk），
            // 避免 MaxSkv 模板步长与实际 Skv 不符导致跨行地址错
            using itKBlk_t = global_iterator<global_tensor<dtype, RowMajor<kD, kTk>>, tileKCube_t>;
            itSk_t gSk(scale_k_ptr);

            for (int j = 0; j < Kb; j++) {
                itKBlk_t gKblk(k_ptr + (uint64_t)j * kD * kTk);   // 块连续布局：块 j 首地址
                auto gKRef = gKblk(0, 0);
                tileKCube_t tK; TLOAD_CUBE(tK, gKRef);
                auto gSkRef = gSk(0, j);
                tileSk_t tSk; TLOAD(tSk, gSkRef);

                tileSum_t tSum, tZeroSum; TEXPANDS(tZeroSum, 0.0f);
                #pragma clang loop unroll(full)
                for (int gi = 0; gi < Gb; gi++) {
                    auto gQRef = gQ(gi, 0);
                    tileQCube_t tQ; TLOAD_CUBE(tQ, gQRef);
                    tileWb_t tWb;
                    {
                        using itWb_t = global_iterator<global_tensor<float, RowMajor<kTm, kTk>>, tileWb_t>;
                        itWb_t gWb(wb_ptr + (uint64_t)(i * G + gi * kTm) * kTk);
                        auto gWbRef = gWb(0, 0);
                        TLOAD(tWb, gWbRef);
                    }
                    // TMATMUL CUBE → TSTORE_CUBE 桥接回 Vec
                    tileSCube_t tSCube; TMATMUL(tSCube, tQ, tK);
                    gmTmp_t gTmp(temp_gm);
                    TSTORE_CUBE(gTmp, tSCube);
                    tileS_t tS; TLOAD(tS, gTmp);
                    tileS_t tZero; TEXPANDS(tZero, 0.0f); TMAX(tS, tS, tZero);
                    TMUL(tS, tS, tWb);
                    tileSum_t tPartial; TCOLSUMX(tPartial, tS);
                    if (gi == 0) { TADD(tSum, tZeroSum, tPartial); }
                    else { TADD(tSum, tSum, tPartial); }
                }
                TMUL(tSum, tSum, tSk);
                auto gOutRef = gOut(0, j);
                TSTORE(gOutRef, tSum);
            }
        }
        return;
    }

    // ---- 多 PE：cooperative SharedMatrix（GM→Shared 直载，kGroupM=kTm=16、kPeM=16）----
    // token 并行：Q/K 每 PE 私有 local CUBE tile（Shared B 的 rendezvous
    // 语义在 gfrun 独立 main 模型下未达成，回退 PE 私有加载）
    using tileQMtrx_t = CubeTileM16<dtype, kTm, kD>;
    using tileKMtrx_t = CubeTileN8<dtype, kD, kTk>;
    using tileCGrp_t = CubeAccumulatorM16<float, kTm, kTk>;
    using itQMtrx_t = global_iterator<global_tensor<dtype, RowMajor<MaxSq, kD>>, tileQMtrx_t>;
    using itKMtrx_t = global_iterator<global_tensor<dtype, RowMajor<kD, kTk>>, tileKMtrx_t>;
    // 每 PE 私有输出槽：temp_gm + tid * [kTm, kTk]
    using gmTmpPe_t = global_tensor<float, RowMajor<kTm, kTk>>;

    for (int i = tid; i < Sq; i += numPEs) {
        itQMtrx_t gQ(q_ptr + (uint64_t)i * G * kD);
        itOut_t gOut(scores_ptr + (uint64_t)i * paddedSkv);

        itSk_t gSk(scale_k_ptr);

        for (int j = 0; j < Kb; j++) {
            itKMtrx_t gKblk(k_ptr + (uint64_t)j * kD * kTk);
            auto gKRef = gKblk(0, 0);
            tileKMtrx_t tK;
            TLOAD_CUBE(tK, gKRef);

            auto gSkRef = gSk(0, j);
            tileSk_t tSk; TLOAD(tSk, gSkRef);

            tileSum_t tSum, tZeroSum; TEXPANDS(tZeroSum, 0.0f);

            #pragma clang loop unroll(full)
            for (int gi = 0; gi < Gb; gi++) {
                auto gQRef = gQ(gi, 0);
                tileQMtrx_t tQ;
                TLOAD_CUBE(tQ, gQRef);

                tileWb_t tWb;
                {
                    using itWb_t = global_iterator<global_tensor<float, RowMajor<kTm, kTk>>, tileWb_t>;
                    itWb_t gWb(wb_ptr + (uint64_t)(i * G + gi * kTm) * kTk);
                    auto gWbRef = gWb(0, 0);
                    TLOAD(tWb, gWbRef);
                }
                // cooperative TMATMUL：Shared A[16,128] × Shared B[128,32] → 每 PE CUBE [16,32]
                tileCGrp_t tCGrp;
                TMATMUL(tCGrp, tQ, tK);
                using gmTmpPe_t = global_tensor<float, RowMajor<kTm, kTk>>;
                gmTmpPe_t gTmpPe(temp_gm + (uint64_t)tid * kTm * kTk);
                TSTORE_CUBE(gTmpPe, tCGrp);
                tileS_t tS; TLOAD(tS, gTmpPe);
                tileS_t tZero; TEXPANDS(tZero, 0.0f); TMAX(tS, tS, tZero);
                TMUL(tS, tS, tWb);
                tileSum_t tPartial; TCOLSUMX(tPartial, tS);
                if (gi == 0) { TADD(tSum, tZeroSum, tPartial); }
                else { TADD(tSum, tSum, tPartial); }
            }
            TMUL(tSum, tSum, tSk);
            auto gOutRef = gOut(0, j);
            TSTORE(gOutRef, tSum);
        }
    }
}

// ========== Step 7: 动态 shape + 多 PE TopK ==========
namespace qli_radix_dyn {

using RU = uint32_t;
using TKey = Tile<Location::Vec, RU, 1, MaxTileCol, BLayout::RowMajor, 1, MaxTileCol>;
using gkRow = global_tensor<RU, RowMajor<1, MaxSkv>>;
using gk1 = global_tensor<RU, RowMajor<1, 1>>;

inline void MakeKey(RU* dst, const RU* src) {
    gkRow gs(const_cast<RU*>(src)); TKey bits; TLOAD(bits, gs);
    TKey sign; TANDS(sign, bits, 0x80000000u);
    TKey neg;  TNOT(neg, bits);
    TKey pos;  TORS(pos, bits, 0x80000000u);
    TKey s01;  TSHRS(s01, sign, 31u);
    TKey diff; TSUB(diff, neg, pos);
    TMUL(diff, diff, s01);
    TKey key;  TADD(key, pos, diff);
    gkRow gd(dst); TSTORE(gd, key);
}

inline void PopN(TKey& mv, RU chunkBase, int32_t* out, int n) {
    // v0.58.4 行归约契约：TROWARGMAX 目的须物理单列 [N,1]（Col==1）；
    // 物理行数 32 维持 128B tile 尺寸下限，valid 仍为 [1,1]
    using t1  = Tile<Location::Vec, RU, 32, 1, BLayout::RowMajor, 1, 1>;
    using t1i = Tile<Location::Vec, int32_t, 32, 1, BLayout::RowMajor, 1, 1>;
    TKey idxTile; TCI(idxTile, chunkBase);
    for (int k = 0; k < n; k++) {
        t1 best; TROWARGMAX(best, mv);
        t1 bestg; TADDS(bestg, best, chunkBase);
        t1i besti; TCVT(besti, bestg);
        { global_tensor<int32_t, RowMajor<1, 1>> gout(out + k); TSTORE(gout, besti); }
        RU bv = 0; gk1 gbv(&bv); TSTORE(gbv, bestg);
        TKey bbc; TEXPANDS(bbc, bv);
        // v0.58.4 predicate 契约：TCMP/TCMPS 结果为 packed predicate，
        // 不可参与算术；消零改用 TSEL 条件覆盖。
        // 用 TSUB 求差 + TCMPS==0 判等（tile-scalar，与 simple 版一致的
        // 已验证模式；避免 tile-tile TCMP 的额外 8KB 广播 tile）
        TKey diff; TSUB(diff, idxTile, bbc);
        TKey isp; TCMPS<CmpMode::EQ>(isp, diff, 0u);
        TKey zero; TEXPANDS(zero, 0u);
        TSEL(mv, isp, zero);
    }
}

inline void Extract(RU* key_ptr, RU kthVal, RU chunkBase, int32_t* outBase, int& outPos, int& needEq) {
    // v0.58.4 行归约契约：TROWSUM 目的须物理单列 [N,1]
    using t1 = Tile<Location::Vec, RU, 32, 1, BLayout::RowMajor, 1, 1>;
    gkRow g(key_ptr);
    // GT: key > kth（TCMPS tile-scalar，与 simple 版一致；避免 tile-tile
    // TCMP + kth 广播 tile 的额外 8KB 活 tile 触发后端 TMOV 尺寸错配）
    { TKey key; TLOAD(key, g);
      TKey isgt; TCMPS<CmpMode::GT>(isgt, key, kthVal);
      TKey one; TEXPANDS(one, 1u);
      TKey sel1; TEXPANDS(sel1, 0u); TSEL(sel1, isgt, one);
      t1 s; TROWSUM(s, sel1); RU cnt = 0; gk1 gcnt(&cnt); TSTORE(gcnt, s);
      TKey cand; TEXPANDS(cand, 0u); TSEL(cand, isgt, key);
      PopN(cand, chunkBase, outBase + outPos, (int)cnt);
      outPos += (int)cnt; }
    if (needEq > 0) {
      TKey key; TLOAD(key, g);
      TKey iseq; TCMPS<CmpMode::EQ>(iseq, key, kthVal);
      TKey one; TEXPANDS(one, 1u);
      TKey sel1; TEXPANDS(sel1, 0u); TSEL(sel1, iseq, one);
      t1 s; TROWSUM(s, sel1); RU cnt = 0; gk1 gcnt(&cnt); TSTORE(gcnt, s);
      int take = (int)cnt; if (take > needEq) take = needEq;
      TKey cand; TEXPANDS(cand, 0u); TSEL(cand, iseq, key);
      PopN(cand, chunkBase, outBase + outPos, take);
      outPos += take; needEq -= take; }
}

// ---- 计数原语：count(key CMP thr)，用于阈值二分找第 k 大（替代退休的 THISTOGRAM）----
// 与 Extract 同款在册算子（TCMPS/TSEL/TEXPANDS/TROWSUM）。填充区 key=0 对
// count_gt(T>=0) 恒不计入，不影响结果。
template <CmpMode M>
inline RU CountChunk(const RU* key_ptr, RU thr) {
    using t1 = Tile<Location::Vec, RU, 32, 1, BLayout::RowMajor, 1, 1>;
    gkRow g(const_cast<RU*>(key_ptr));
    TKey key; TLOAD(key, g);
    TKey ism; TCMPS<M>(ism, key, thr);
    TKey one; TEXPANDS(one, 1u);
    TKey sel1; TEXPANDS(sel1, 0u); TSEL(sel1, ism, one);
    t1 s; TROWSUM(s, sel1); RU cnt = 0; gk1 gc(&cnt); TSTORE(gc, s);
    return cnt;
}
template <CmpMode M>
inline RU CountAll(const RU* krow, int NumChunks, RU thr) {
    RU cnt = 0;
    for (int c = 0; c < NumChunks; c++)
        cnt += CountChunk<M>(krow + (uint64_t)c * MaxTileCol, thr);
    return cnt;
}

}  // namespace qli_radix_dyn

inline void qli_topk_radix_dynamic(float* scores_gm, int32_t* indices_gm,
                                   int Sq, int Skv, int topK, int numPEs,
                                   uint32_t* key_scratch)
{
    using namespace qli_radix_dyn;
    int NumChunks = (Skv + MaxTileCol - 1) / MaxTileCol;
    int paddedSkv = NumChunks * MaxTileCol;
    const uint32_t tid = get_thread_idx();

    for (int i = tid; i < Sq; i += numPEs) {
        const RU* row = (const RU*)(scores_gm) + (uint64_t)i * paddedSkv;
        RU* krow = key_scratch + (uint64_t)i * paddedSkv;

        // Step 1: sortable key（末 chunk 整块 2048 读取，填充区随后清零）
        for (int c = 0; c < NumChunks; c++)
            MakeKey(krow + c * MaxTileCol, row + c * MaxTileCol);

        // 末 chunk 填充区清零（key=0 为最小值，不进入 TopK）
        int padStart = Skv - (NumChunks - 1) * MaxTileCol;
        if (padStart < MaxTileCol) {
            volatile RU* pad = (volatile RU*)(krow + (NumChunks - 1) * MaxTileCol + padStart);
            for (int p = 0; p < MaxTileCol - padStart; p++) pad[p] = 0;
        }

        // Step 2: 阈值二分找第 topK 大 key（TCMPS<GT>+TROWSUM，替代已退休 THISTOGRAM）
        // kth_value = 最小 T 满足 count(key>T) < topK；仅用 Step3 同款在册算子，
        // 无 THISTOGRAM / 无 256 桶扫描 / 无 Idx 前缀。填充区 key=0 不影响 count_gt(T>=0)。
        RU lo = 0u, hi = 0xFFFFFFFFu;
        while (lo < hi) {
            RU mid = lo + (RU)((hi - lo) >> 1);
            RU gt = CountAll<CmpMode::GT>(krow, NumChunks, mid);
            if (gt < (RU)topK) hi = mid; else lo = mid + 1u;
        }
        RU kth_value = lo;
        RU gtCount = CountAll<CmpMode::GT>(krow, NumChunks, kth_value);
        int remaining = topK - (int)gtCount;   // 从 == kth_value 组补足的名额

        // Step 3: 提取 topK 索引
        int outPos = 0;
        int needEq = remaining;
        for (int c = 0; c < NumChunks; c++)
            Extract(krow + c * MaxTileCol, kth_value, (RU)c * MaxTileCol,
                    indices_gm + i * topK, outPos, needEq);
    }
}

#endif