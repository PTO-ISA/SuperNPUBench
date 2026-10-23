#ifndef MATMUL_TEST_KERNEL_HPP
#define MATMUL_TEST_KERNEL_HPP

#include <common/pto_tileop.hpp>

#if !defined(Batch)
#define Batch 1
#endif

using namespace pto;

// ============================================================================
// matmul_test — 4-PE cooperative fp16 GEMM（A Local CellReg / B Shared）
//
// C[gM,gN] = A[gM,gK] × B[gK,gN]（fp16 in / fp32 acc / fp16 out）
// A 存储布局 [gM, gK]（行距 gK）；B 存储布局 [gK, gN]（行距 gN）。
//
// TCVT 适配（ISSUE-016 / GitHub #257）：K 链 CCTRL 四变体——
//   Kb==1 → CCTRL=0（单 op，结果直接写普通 tile）
//   Kb>1  → head=1(raw_acc) / mid=3(raw|hint) / tail=2(acc_hint)
// 末条 ACC 将累加结果物化到 CellReg tile，TCVT 从 CellReg 读取后转 fp16。
//
// 复用模式（kMode，REUSE_MODE 宏选择）：
//   0 = 不复用：每个 (条带, N 块) 独立 K 链，A/B 每 k 重装
//   1 = reuseA：A 每 k 装一次，跨 kReuse 个驻留累加器（N 块方向）复用
//   2 = reuseB：B 每 k 装一次，跨 kReuse 个驻留累加器（M 条带方向）复用；
//               仅完整 tM 条带参与分组，M 尾条带按模式 0/1 处理
// kReuse（REUSE_COUNT 宏）：驻留累加器个数，1..4，默认 4。
//
// gM/gN/gK 为编译期模板参数（与上游 reuseA 相同结构）：global_tensor 使用
// 静态 shape，避免运行时维度引入的额外标量与 tile 手分配漂移（R=4 时
// A tile 与累加器 dst 同手导致 cooperative 左操作数版本解析混淆）。
//
// 约束：tM ∈ {64,128}、tN/tK 为 2 的幂、kTailM ∈ {16,32,64}（M 尾组）。
// ============================================================================

enum MatmulTestReuseMode { kModeNone = 0, kModeReuseA = 1, kModeReuseB = 2 };

namespace matmul_test_detail {

template <typename TileT, int Count>
struct MatmulTestAccStorage {
    TileT tiles[Count];
    TileT &operator[](int index) { return tiles[index]; }
};

// K 链 CCTRL 四变体（ISSUE #257：末条物化到 CellReg 供 TCVT 读取）。
constexpr FixpAttr kOptBase = FixpAttr{}.transpose_b();
constexpr FixpAttr kOptSingle = kOptBase;                                       // CCTRL=0
constexpr FixpAttr kOptHead = kOptBase.with_cube_ctrl(CubeCtrlRawAccumulator);  // CCTRL=1
constexpr FixpAttr kOptMid = kOptBase.with_cube_ctrl(CubeCtrlRawAccAndHint);    // CCTRL=3
constexpr FixpAttr kOptTail = kOptBase.with_cube_ctrl(CubeCtrlInternalAccHint); // CCTRL=2

template <int kPeM, int tN, int tK, int gM, int gN, int gK>
struct MatmulTestTiles {
    using gmA = global_tensor<__half, RowMajor<gM, gK>>;
    using gmB = global_tensor<__half, RowMajor<gK, gN>>;
    using gmC = global_tensor<__half, RowMajor<gM, gN>>;
    using tileA = std::conditional_t<(kPeM <= 16),
        CubeTileM16<__half, kPeM, tK>, CubeTileM32<__half, kPeM, tK>>;
    using tileBMatrix = SharedMatrixRight<__half, tK, tN>;
    using tileB = SharedTile<tileBMatrix>;
    using tileAcc = std::conditional_t<(kPeM <= 16),
        CubeAccumulatorM16<float, kPeM, tN>, CubeAccumulatorM32<float, kPeM, tN>>;
    using tileAccF16 = std::conditional_t<(kPeM <= 16),
        CubeAccumulatorM16<__half, kPeM, tN>, CubeAccumulatorM32<__half, kPeM, tN>>;
};

template <int kPeM, int tN, int tK, int gM, int gN, int gK, int R>
using MatmulTestAccGroup = MatmulTestAccStorage<
    typename MatmulTestTiles<kPeM, tN, tK, gM, gN, gK>::tileAcc, R>;

// —— 单链块：一个 (条带, N 块) 的完整 K 链 + TCVT + TSTORE ——
template <int kPeM, int tN, int tK, int gM, int gN, int gK>
static void chain_block_(__half *a_base, __half *b_base, __half *c_base,
                         int row0, int j, int groupM, int Kb, uint32_t tid) {
    using Tiles = MatmulTestTiles<kPeM, tN, tK, gM, gN, gK>;
    using gmA = typename Tiles::gmA;
    using gmB = typename Tiles::gmB;
    using gmC = typename Tiles::gmC;
    using tileA = typename Tiles::tileA;
    using tileB = typename Tiles::tileB;
    using tileBMatrix = typename Tiles::tileBMatrix;
    using tileAcc = typename Tiles::tileAcc;
    using tileAccF16 = typename Tiles::tileAccF16;

    tileAcc tACC;
    for (int k = 0; k < Kb; ++k) {
        gmB gB(b_base + ((size_t)k * tK) * gN + (size_t)j * tN);
        tileB tB;
        TLOAD<tileBMatrix, 1>(tB, gB);

        gmA gA(a_base +
                   ((size_t)row0 + (size_t)tid * kPeM) * gK +
                   (size_t)k * tK);
        tileA tA;
        TLOAD_CUBE(tA, gA);

        if (Kb == 1) {
            TMATMUL<kOptSingle>(tACC, tA, tB, (size_t)groupM);
        } else if (k == 0) {
            TMATMUL<kOptHead>(tACC, tA, tB, (size_t)groupM);
        } else if (k == Kb - 1) {
            TMATMUL_ACC<kOptTail>(tACC, tACC, tA, tB, (size_t)groupM);
        } else {
            TMATMUL_ACC<kOptMid>(tACC, tACC, tA, tB, (size_t)groupM);
        }
    }

    tileAccF16 tCF16;
    TCVT(tCF16, tACC);
    gmC gC(c_base +
               ((size_t)row0 + (size_t)tid * kPeM) * gN +
               (size_t)j * tN);
    TSTORE_CUBE(gC, tCF16);
}

// —— reuseA 的单 k 步：A 装一次，喂 R 个驻留累加器（N 块方向）——
template <bool FirstK, bool LastK, int kPeM, int tN, int tK,
          int gM, int gN, int gK, int R>
static void reusea_k_(__half *a_base, __half *b_base,
                      MatmulTestAccGroup<kPeM, tN, tK, gM, gN, gK, R> &cs,
                      int row0, int jBase, int k, int groupM, uint32_t tid) {
    using Tiles = MatmulTestTiles<kPeM, tN, tK, gM, gN, gK>;
    using gmA = typename Tiles::gmA;
    using gmB = typename Tiles::gmB;
    using tileA = typename Tiles::tileA;
    using tileB = typename Tiles::tileB;
    using tileBMatrix = typename Tiles::tileBMatrix;
    constexpr int Nb = gN / tN;

    tileA tA;
    gmA gA(a_base +
               ((size_t)row0 + (size_t)tid * kPeM) * gK +
               (size_t)k * tK);
    TLOAD_CUBE(tA, gA);

#pragma clang loop unroll(full)
    for (int n = 0; n < R; ++n) {
        if (jBase + n < Nb) {
            tileB tB;
            gmB gB(b_base + ((size_t)k * tK) * gN +
                       (size_t)(jBase + n) * tN);
            TLOAD<tileBMatrix, 1>(tB, gB);
            if constexpr (FirstK && LastK) {
                TMATMUL<kOptSingle>(cs[n], tA, tB, (size_t)groupM);
            } else if constexpr (FirstK) {
                TMATMUL<kOptHead>(cs[n], tA, tB, (size_t)groupM);
            } else if constexpr (LastK) {
                TMATMUL_ACC<kOptTail>(cs[n], cs[n], tA, tB, (size_t)groupM);
            } else {
                TMATMUL_ACC<kOptMid>(cs[n], cs[n], tA, tB, (size_t)groupM);
            }
        }
    }
}

// —— reuseA 组：一个条带，jBase 按 R 分组，组内 R 个驻留累加器 ——
template <int kPeM, int tN, int tK, int gM, int gN, int gK, int R>
static void reusea_group_(__half *c_base, __half *a_base, __half *b_base,
                          int row0, int groupM, uint32_t tid) {
    using Tiles = MatmulTestTiles<kPeM, tN, tK, gM, gN, gK>;
    using gmC = typename Tiles::gmC;
    using tileAccF16 = typename Tiles::tileAccF16;
    constexpr int Nb = gN / tN;
    constexpr int Kb = gK / tK;

    static_assert(
        Tiles::tileA::LogicalTileBytes + R * Tiles::tileAcc::LogicalTileBytes <=
            256 * 1024,
        "A tile + R resident accumulators exceed CellReg; reduce tN or R");

#pragma clang loop unroll(full)
    for (int jBase = 0; jBase < Nb; jBase += R) {
        MatmulTestAccGroup<kPeM, tN, tK, gM, gN, gK, R> cs;
        // 首/中/尾分离成独立调用，避免链内 first/last-K 控制流 PHI 合并。
        if (Kb == 1) {
            reusea_k_<true, true, kPeM, tN, tK, gM, gN, gK, R>(
                a_base, b_base, cs, row0, jBase, 0, groupM, tid);
        } else {
            reusea_k_<true, false, kPeM, tN, tK, gM, gN, gK, R>(
                a_base, b_base, cs, row0, jBase, 0, groupM, tid);
            for (int k = 1; k < Kb - 1; ++k) {
                reusea_k_<false, false, kPeM, tN, tK, gM, gN, gK, R>(
                    a_base, b_base, cs, row0, jBase, k, groupM, tid);
            }
            reusea_k_<false, true, kPeM, tN, tK, gM, gN, gK, R>(
                a_base, b_base, cs, row0, jBase, Kb - 1, groupM, tid);
        }
#pragma clang loop unroll(full)
        for (int n = 0; n < R; ++n) {
            if (jBase + n < Nb) {
                tileAccF16 tCF16;
                TCVT(tCF16, cs[n]);
                gmC gC(c_base +
                           ((size_t)row0 + (size_t)tid * kPeM) * gN +
                           (size_t)(jBase + n) * tN);
                TSTORE_CUBE(gC, tCF16);
            }
        }
    }
}

// —— reuseB 的单 k 步：B 装一次，喂 R 个驻留累加器（M 条带方向）——
template <bool FirstK, bool LastK, int kGroupM, int kPeM, int tN, int tK,
          int gM, int gN, int gK, int R>
static void reuseb_k_(__half *a_base, __half *b_base,
                      MatmulTestAccGroup<kPeM, tN, tK, gM, gN, gK, R> &cs,
                      int stripBase, int stripCount, int j, int k,
                      uint32_t tid) {
    using Tiles = MatmulTestTiles<kPeM, tN, tK, gM, gN, gK>;
    using gmA = typename Tiles::gmA;
    using gmB = typename Tiles::gmB;
    using tileA = typename Tiles::tileA;
    using tileB = typename Tiles::tileB;
    using tileBMatrix = typename Tiles::tileBMatrix;

    tileB tB;
    gmB gB(b_base + ((size_t)k * tK) * gN + (size_t)j * tN);
    TLOAD<tileBMatrix, 1>(tB, gB);

#pragma clang loop unroll(full)
    for (int ms = 0; ms < R; ++ms) {
        if (ms < stripCount) {
            tileA tA;
            const int row0 = (stripBase + ms) * kGroupM;
            gmA gA(a_base +
                       ((size_t)row0 + (size_t)tid * kPeM) * gK +
                       (size_t)k * tK);
            TLOAD_CUBE(tA, gA);
            if constexpr (FirstK && LastK) {
                TMATMUL<kOptSingle>(cs[ms], tA, tB, (size_t)kGroupM);
            } else if constexpr (FirstK) {
                TMATMUL<kOptHead>(cs[ms], tA, tB, (size_t)kGroupM);
            } else if constexpr (LastK) {
                TMATMUL_ACC<kOptTail>(cs[ms], cs[ms], tA, tB, (size_t)kGroupM);
            } else {
                TMATMUL_ACC<kOptMid>(cs[ms], cs[ms], tA, tB, (size_t)kGroupM);
            }
        }
    }
}

// —— reuseB 组：R 个完整条带 × 一个 N 块 ——
template <int kGroupM, int kPeM, int tN, int tK, int gM, int gN, int gK, int R>
static void reuseb_group_(__half *c_base, __half *a_base, __half *b_base,
                          int stripBase, int stripCount, int j, uint32_t tid) {
    using Tiles = MatmulTestTiles<kPeM, tN, tK, gM, gN, gK>;
    using gmC = typename Tiles::gmC;
    using tileAccF16 = typename Tiles::tileAccF16;
    constexpr int Kb = gK / tK;

    MatmulTestAccGroup<kPeM, tN, tK, gM, gN, gK, R> cs;
    if (Kb == 1) {
        reuseb_k_<true, true, kGroupM, kPeM, tN, tK, gM, gN, gK, R>(
            a_base, b_base, cs, stripBase, stripCount, j, 0, tid);
    } else {
        reuseb_k_<true, false, kGroupM, kPeM, tN, tK, gM, gN, gK, R>(
            a_base, b_base, cs, stripBase, stripCount, j, 0, tid);
        for (int k = 1; k < Kb - 1; ++k) {
            reuseb_k_<false, false, kGroupM, kPeM, tN, tK, gM, gN, gK, R>(
                a_base, b_base, cs, stripBase, stripCount, j, k, tid);
        }
        reuseb_k_<false, true, kGroupM, kPeM, tN, tK, gM, gN, gK, R>(
            a_base, b_base, cs, stripBase, stripCount, j, Kb - 1, tid);
    }
#pragma clang loop unroll(full)
    for (int ms = 0; ms < R; ++ms) {
        if (ms < stripCount) {
            tileAccF16 tCF16;
            TCVT(tCF16, cs[ms]);
            const int row0 = (stripBase + ms) * kGroupM;
            gmC gC(c_base +
                       ((size_t)row0 + (size_t)tid * kPeM) * gN +
                       (size_t)j * tN);
            TSTORE_CUBE(gC, tCF16);
        }
    }
}

}  // namespace matmul_test_detail

template <int tM = 128, int tN = 64, int tK = 64, int kTailM = 0,
          int kMode = kModeNone, int kReuse = 4,
          int gM = 256, int gN = 256, int gK = 256>
void matmul_test(__half *c_ptr, __half *a_ptr, __half *b_ptr, float *scratch) {
    constexpr int kPeNum = 4;
    static_assert(tM == 64 || tM == 128,
                  "tM must be 64 or 128 (cooperative group_M contract)");
    static_assert((tN & (tN - 1)) == 0, "tN must be pow2 (model constraint)");
    static_assert((tK & (tK - 1)) == 0, "tK must be pow2 (model constraint)");
    static_assert(gK % tK == 0, "K must be divisible by tK");
    static_assert(gN % tN == 0, "N must be divisible by tN");
    static_assert(kTailM == 0 || kTailM == 16 || kTailM == 32 || kTailM == 64,
                  "tail group_M must be pow2 <= 64 (per-PE shard 16)");
    static_assert(kMode >= 0 && kMode <= 2, "kMode must be 0/1/2");
    static_assert(kReuse >= 1 && kReuse <= 4, "kReuse must be in 1..4");
    static_assert(kTailM == 0 ? (gM % tM == 0)
                              : (gM % tM == 0 ? false
                                              : (gM / tM * tM + 4 * 16 == gM)),
                  "gM must be tile-aligned or full-tiles + 64-row tail");
    constexpr int kFullPeM = tM / kPeNum;
    constexpr int kTailPeM = 16;
    constexpr int Mb = kTailM > 0 ? (gM - 4 * kTailPeM) / tM : gM / tM;
    constexpr int Nb = gN / tN;
    constexpr int Kb = gK / tK;
    (void)scratch;

    const uint32_t tid = get_thread_idx();

    for (int b = 0; b < Batch; ++b) {
        __half *a_base = a_ptr + (size_t)b * gM * gK;
        __half *b_base = b_ptr + (size_t)b * gK * gN;
        __half *c_base = c_ptr + (size_t)b * gM * gN;

        if constexpr (kMode == 2) {
            // reuseB：B 每 k 装一次跨 R 条带复用；尾条带按单链处理。
            for (int j = 0; j < Nb; ++j) {
#pragma clang loop unroll(full)
                for (int g = 0; g < Mb; g += kReuse) {
                    const int cnt = (Mb - g) < kReuse ? (Mb - g) : kReuse;
                    matmul_test_detail::reuseb_group_<
                        tM, kFullPeM, tN, tK, gM, gN, gK, kReuse>(
                        c_base, a_base, b_base, g, cnt, j, tid);
                }
                if constexpr (kTailM > 0) {
                    matmul_test_detail::chain_block_<
                        kTailPeM, tN, tK, gM, gN, gK>(
                        a_base, b_base, c_base, Mb * tM, j, kTailM, Kb, tid);
                }
            }
        } else {
            // 模式 0/1：按条带处理；尾条带同形态。
            for (int i = 0; i < Mb; ++i) {
                if constexpr (kMode == 1) {
                    matmul_test_detail::reusea_group_<
                        kFullPeM, tN, tK, gM, gN, gK, kReuse>(
                        c_base, a_base, b_base, i * tM, tM, tid);
                } else {
                    for (int j = 0; j < Nb; ++j) {
                        matmul_test_detail::chain_block_<
                            kFullPeM, tN, tK, gM, gN, gK>(
                            a_base, b_base, c_base, i * tM, j, tM, Kb, tid);
                    }
                }
            }
            if constexpr (kTailM > 0) {
                if constexpr (kMode == 1) {
                    matmul_test_detail::reusea_group_<
                        kTailPeM, tN, tK, gM, gN, gK, kReuse>(
                        c_base, a_base, b_base, Mb * tM, kTailM, tid);
                } else {
                    for (int j = 0; j < Nb; ++j) {
                        matmul_test_detail::chain_block_<
                            kTailPeM, tN, tK, gM, gN, gK>(
                            a_base, b_base, c_base, Mb * tM, j, kTailM, Kb, tid);
                    }
                }
            }
        }
    }
}

#endif  // MATMUL_TEST_KERNEL_HPP
