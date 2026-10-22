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
#include "qli_element.hpp"
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
// 每个 PE 使用私有 local CUBE tile（M16/N8）和 accumulator；
// TSTORE_CUBE 桥接回 Vec，随后交替执行 element 循环与 Tile reduction。
// 查询行按 i=tid; i<Sq; i+=numPEs 分配，支持不均匀分配，无需集体同步。
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

                tileSum_t tSum;
                auto &sum = TPARTELEMENT(tSum);
#pragma pto element for
                for (uint32_t column = 0; column < kTk; ++column)
                    sum[column] = 0.0f;
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
                    auto &score = TPARTELEMENT(tS);
                    const auto &weight = TPARTELEMENT(tWb);
#pragma pto element for
                    for (uint32_t element = 0; element < kTm * kTk; ++element) {
                        const float positive = score[element] > 0.0f ? score[element] : 0.0f;
                        score[element] = positive * weight[element];
                    }
                    tileSum_t tPartial; TCOLSUM(tPartial, tS);
                    const auto &partial = TPARTELEMENT(tPartial);
#pragma pto element for
                    for (uint32_t column = 0; column < kTk; ++column)
                        sum[column] += partial[column];
                }
                const auto &scale = TPARTELEMENT(tSk);
#pragma pto element for
                for (uint32_t column = 0; column < kTk; ++column)
                    sum[column] *= scale[column];
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

            tileSum_t tSum;
            auto &sum = TPARTELEMENT(tSum);
#pragma pto element for
            for (uint32_t column = 0; column < kTk; ++column)
                sum[column] = 0.0f;

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
                auto &score = TPARTELEMENT(tS);
                const auto &weight = TPARTELEMENT(tWb);
#pragma pto element for
                for (uint32_t element = 0; element < kTm * kTk; ++element) {
                    const float positive = score[element] > 0.0f ? score[element] : 0.0f;
                    score[element] = positive * weight[element];
                }
                tileSum_t tPartial; TCOLSUM(tPartial, tS);
                const auto &partial = TPARTELEMENT(tPartial);
#pragma pto element for
                for (uint32_t column = 0; column < kTk; ++column)
                    sum[column] += partial[column];
            }
            const auto &scale = TPARTELEMENT(tSk);
#pragma pto element for
            for (uint32_t column = 0; column < kTk; ++column)
                sum[column] *= scale[column];
            auto gOutRef = gOut(0, j);
            TSTORE(gOutRef, tSum);
        }
    }
}

namespace qli_radix_dyn {

using RU = uint32_t;
using TKey = Tile<Location::Vec, RU, 1, MaxTileCol, BLayout::RowMajor, 1, MaxTileCol>;
using gkRow = global_tensor<RU, RowMajor<1, MaxSkv>>;
using gk1 = global_tensor<RU, RowMajor<1, 1>>;

inline void MakeKey(RU* dst, const RU* src) {
    qli_element::make_key<MaxTileCol>(dst, src);
}

inline void ChunkHist(RU* key_ptr, int byteId, RU* prefix, RU* hist) {
    const RU highPrefix = (prefix[0] << 24) | (prefix[8] << 16) | (prefix[16] << 8);
    qli_element::histogram<MaxTileCol>(hist, key_ptr, static_cast<RU>(byteId), highPrefix);
}

inline void PopN(TKey& mv, RU chunkBase, int32_t* out, int n) {
    // v0.58.4 行归约契约：TROWARGMAX 目的须物理单列 [N,1]（Col==1）；
    // 物理行数 32 维持 128B tile 尺寸下限，valid 仍为 [1,1]
    using Scalar = Tile<Location::Vec, RU, 32, 1, BLayout::RowMajor, 1, 1>;
    auto &values = TPARTELEMENT(mv);
    for (int k = 0; k < n; ++k) {
        Scalar best;
        TROWARGMAX(best, mv);
        RU selected = 0;
        gk1 index(&selected);
        TSTORE(index, best);
        out[k] = static_cast<int32_t>(chunkBase + selected);
#pragma pto element for
        for (uint32_t element = 0; element < MaxTileCol; ++element) {
            if (element == selected)
                values[element] = 0;
        }
    }
}

inline void Extract(RU* key_ptr, RU kthVal, RU chunkBase, int32_t* outBase,
                    int& outPos, int& needEq, uint32_t valid_columns) {
    // v0.58.4 行归约契约：TROWSUM 目的须物理单列 [N,1]
    using t1 = Tile<Location::Vec, RU, 32, 1, BLayout::RowMajor, 1, 1>;
    gkRow g(key_ptr);
    // GT/EQ 使用整行 NORM 元素视图，保留原计数归约与顺序提取。
    { TKey key; TLOAD(key, g);
      TKey sel1, cand;
      auto &keys = TPARTELEMENT(key);
      auto &counts = TPARTELEMENT(sel1);
      auto &values = TPARTELEMENT(cand);
#pragma pto element for
      for (uint32_t element = 0; element < MaxTileCol; ++element) {
          const bool take = element < valid_columns && keys[element] > kthVal;
          counts[element] = take ? 1u : 0u;
          values[element] = take ? element + 1u : 0u;
      }
      t1 s; TROWSUM(s, sel1);
      RU cnt = 0; gk1 gcnt(&cnt); TSTORE(gcnt, s);
      PopN(cand, chunkBase, outBase + outPos, (int)cnt);
      outPos += (int)cnt; }
    if (needEq > 0) {
      TKey key; TLOAD(key, g);
      TKey sel1, cand;
      auto &keys = TPARTELEMENT(key);
      auto &counts = TPARTELEMENT(sel1);
      auto &values = TPARTELEMENT(cand);
#pragma pto element for
      for (uint32_t element = 0; element < MaxTileCol; ++element) {
          const bool take = element < valid_columns && keys[element] == kthVal;
          counts[element] = take ? 1u : 0u;
          values[element] = take ? element + 1u : 0u;
      }
      t1 s; TROWSUM(s, sel1);
      RU cnt = 0; gk1 gcnt(&cnt); TSTORE(gcnt, s);
      int take = (int)cnt; if (take > needEq) take = needEq;
      PopN(cand, chunkBase, outBase + outPos, take);
      outPos += take; needEq -= take; }
}

}  // namespace qli_radix_dyn

inline void qli_topk_radix_dynamic(float* scores_gm, int32_t* indices_gm,
                                   int Sq, int Skv, int topK, int numPEs,
                                   uint32_t* key_scratch)
{
    using namespace qli_radix_dyn;
    if (Sq < 0 || Skv <= 0 || Skv > __INT_MAX__ - MaxTileCol ||
        topK < 0 || topK > Skv || numPEs <= 0 || numPEs > 4)
        __builtin_trap();
    if (Sq == 0 || topK == 0)
        return;
    int NumChunks = (Skv + MaxTileCol - 1) / MaxTileCol;
    int paddedSkv = NumChunks * MaxTileCol;
    if (static_cast<uint64_t>(Sq) * static_cast<uint32_t>(paddedSkv) > __INT_MAX__)
        __builtin_trap();
    const uint32_t tid = get_thread_idx();
    if (tid >= static_cast<uint32_t>(numPEs))
        return;

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

        // 末 chunk 填充区清零；提取时另以 valid_columns 排除填充位置，
        // 因而原始有效 key=0 仍可以参加 TopK。
        int padStart = Skv - (NumChunks - 1) * MaxTileCol;
        if (padStart < MaxTileCol) {
            const uint32_t first = static_cast<uint32_t>(padStart);
            RU *last_chunk = krow + (NumChunks - 1) * MaxTileCol;
            for (uint32_t base = first; base < MaxTileCol; base += 32) {
#pragma pto element for
                for (uint32_t element = 0; element < 32; ++element) {
                    const uint32_t column = base + element;
                    if (column < MaxTileCol)
                        last_chunk[column] = 0;
                }
            }
        }

        // Step 2: 4 轮 MSD radix → kth_value
        RU kth_value = 0;
        int remaining = topK;
#pragma pto element for
        for (uint32_t element = 0; element < 32; ++element)
            prefix_buf[element] = 0;

        for (int r = 3; r >= 0; r--) {
            RU per_bin[256] = {0};
            if (r < 3) {
                const RU b3 = (kth_value >> 24) & 0xFFu;
                const RU b2 = (kth_value >> 16) & 0xFFu;
                const RU b1 = (kth_value >> 8) & 0xFFu;
                // PE 私有前缀缓冲；下一次 histogram 调用按语句顺序读取。
#pragma pto element for
                for (uint32_t element = 0; element < 32; ++element) {
                    prefix_buf[element] = element < 8 ? b3 :
                                          element < 16 ? b2 :
                                          element < 24 ? b1 : 0u;
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
        for (int c = 0; c < NumChunks; c++) {
            const uint32_t valid_columns = static_cast<uint32_t>(
                c + 1 == NumChunks ? Skv - c * MaxTileCol : MaxTileCol);
            Extract(krow + c * MaxTileCol, kth_value, (RU)c * MaxTileCol,
                    indices_gm + i * topK, outPos, needEq, valid_columns);
        }
    }
}

#endif
