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
                    tileSum_t tPartial; TCOLSUM(tPartial, tS);
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
                tileSum_t tPartial; TCOLSUM(tPartial, tS);
                if (gi == 0) { TADD(tSum, tZeroSum, tPartial); }
                else { TADD(tSum, tSum, tPartial); }
            }
            TMUL(tSum, tSum, tSk);
            auto gOutRef = gOut(0, j);
            TSTORE(gOutRef, tSum);
        }
    }
}

// ========== THISTOGRAMX（同标准 THISTOGRAM，放宽 Idx shape 约束）==========
template <typename tile_o, typename tile_s, typename tile_idx>
void THISTOGRAMX(tile_o& dst, tile_s& src, tile_idx& idx, int ByteId) {
#define THISTOGRAMX_ASM(BYTE_NAME)                                    \
  asm volatile(                                                        \
    "BSTART.TEPL 104, %D1\n"                                     \
    "B.DATR %D1, " BYTE_NAME ", Zero\n"                                \
    "B.DIM %3, 0, ->LB0\n"                                             \
    "B.DIM %4, 0, ->LB1\n"                                             \
    "B.DIM zero, %c5, ->LB2\n"                                         \
    "B.IOT %6, %7, mask=1111, last, ->%0<%Z8>\n"                       \
    ""                                                                  \
    : "=Tr"(dst.data())                                                \
    : "i"(type_traits<typename tile_s::DType>::TypeCode),              \
      "i"(type_traits<typename tile_o::DType>::TypeCode),              \
      "r"(dst.GetValidCol()),                                          \
      "r"(src.GetValidRow()),                                          \
      "i"(tile_o::Cols),                                               \
      "Tr"(src.data()),                                                \
      "Tr"(idx.data()),                                                \
      "i"(tile_type_traits<typename tile_o::TileDType>::TilesizeCode))
  switch (ByteId) {
    case 0: THISTOGRAMX_ASM("Byte0"); break;
    case 1: THISTOGRAMX_ASM("Byte1"); break;
    case 2: THISTOGRAMX_ASM("Byte2"); break;
    default: THISTOGRAMX_ASM("Byte3"); break;
  }
#undef THISTOGRAMX_ASM
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

inline void ChunkHist(RU* key_ptr, int byteId, RU* prefix, RU* hist) {
    using tidx = Tile<Location::Vec, RU, 4, 8, BLayout::RowMajor>;
    using gidx = global_tensor<RU, RowMajor<4, 8>>;
    using th = Tile<Location::Vec, RU, 1, 256, BLayout::RowMajor>;
    using gh = global_tensor<RU, RowMajor<1, 256>>;
    gkRow g(key_ptr); TKey key; TLOAD(key, g);
    tidx idxTile; { gidx gi(prefix); TLOAD(idxTile, gi); }
    th hist_tile; THISTOGRAMX(hist_tile, key, idxTile, byteId);
    gh gout(hist); TSTORE(gout, hist_tile);
}

inline void PopN(TKey& mv, RU chunkBase, int32_t* out, int n) {
    using t1  = Tile<Location::Vec, RU, 1, 32, BLayout::RowMajor, 1, 1>;
    using t1i = Tile<Location::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 1>;
    TKey idxTile; TCI(idxTile, chunkBase);
    for (int k = 0; k < n; k++) {
        t1 best; TROWARGMAX(best, mv);
        t1 bestg; TADDS(bestg, best, chunkBase);
        t1i besti; TCVT(besti, bestg);
        { global_tensor<int32_t, RowMajor<1, 1>> gout(out + k); TSTORE(gout, besti); }
        RU bv = 0; gk1 gbv(&bv); TSTORE(gbv, bestg);
        TKey bbc; TEXPANDS(bbc, bv);
        TKey isp; TCMP<CmpMode::EQ>(isp, idxTile, bbc);
        TKey one; TEXPANDS(one, 1u);
        TKey np; TSUB(np, one, isp);
        TMUL(mv, mv, np);
    }
}

inline void Extract(RU* key_ptr, RU kthVal, RU chunkBase, int32_t* outBase, int& outPos, int& needEq) {
    using t1 = Tile<Location::Vec, RU, 1, 32, BLayout::RowMajor, 1, 1>;
    gkRow g(key_ptr); TKey kthk; TEXPANDS(kthk, kthVal);
    { TKey key; TLOAD(key, g);
      TKey isgt; TCMP<CmpMode::GT>(isgt, key, kthk);
      t1 s; TROWSUM(s, isgt); RU cnt = 0; gk1 gcnt(&cnt); TSTORE(gcnt, s);
      TKey cand; TMUL(cand, key, isgt);
      PopN(cand, chunkBase, outBase + outPos, (int)cnt);
      outPos += (int)cnt; }
    if (needEq > 0) {
      TKey key; TLOAD(key, g);
      TKey iseq; TCMP<CmpMode::EQ>(iseq, key, kthk);
      t1 s; TROWSUM(s, iseq); RU cnt = 0; gk1 gcnt(&cnt); TSTORE(gcnt, s);
      int take = (int)cnt; if (take > needEq) take = needEq;
      TKey cand; TMUL(cand, key, iseq);
      PopN(cand, chunkBase, outBase + outPos, take);
      outPos += take; needEq -= take; }
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

    // 每 PE 独立的 hist/prefix scratch（256 + 32 uint32 = 1152B）
    // 多 PE 并行时共享同一 buffer 会互相覆盖（直方图数据竞争）
    uint32_t* hist_scratch = key_scratch + (uint64_t)Sq * paddedSkv
                           + (uint64_t)tid * 288;
    uint32_t* prefix_buf = hist_scratch + 256;

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

        // Step 2: 4 轮 MSD radix → kth_value
        RU kth_value = 0;
        int remaining = topK;
        for (int w = 0; w < 32; w++) prefix_buf[w] = 0;

        for (int r = 3; r >= 0; r--) {
            RU per_bin[256] = {0};
            if (r < 3) {
                volatile RU* pb = (volatile RU*)prefix_buf;
                RU b3 = (kth_value >> 24) & 0xFFu;
                RU b2 = (kth_value >> 16) & 0xFFu;
                RU b1 = (kth_value >> 8) & 0xFFu;
                for (int cc = 0; cc < 8; cc++) {
                    pb[0*8+cc] = b3; pb[1*8+cc] = b2;
                    pb[2*8+cc] = b1; pb[3*8+cc] = 0;
                }
            }
            for (int c = 0; c < NumChunks; c++) {
                ChunkHist(krow + c * MaxTileCol, r, prefix_buf, hist_scratch);
                volatile RU* rh = (volatile RU*)hist_scratch;
                RU prev = 0;
                for (int b = 0; b < 256; b++) { RU cum = rh[b]; per_bin[b] += (cum - prev); prev = cum; }
            }
            RU cum = 0; int kth_byte = 0;
            for (int b = 255; b >= 0; b--) {
                cum += per_bin[b];
                if (cum >= (RU)remaining) { kth_byte = b; break; }
            }
            remaining -= (int)(cum - per_bin[kth_byte]);
            kth_value |= ((RU)kth_byte) << (r * 8);
        }

        // Step 3: 提取 topK 索引
        int outPos = 0;
        int needEq = remaining;
        for (int c = 0; c < NumChunks; c++)
            Extract(krow + c * MaxTileCol, kth_value, (RU)c * MaxTileCol,
                    indices_gm + i * topK, outPos, needEq);
    }
}

#endif