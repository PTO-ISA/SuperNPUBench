#pragma once

#include <common/pto_tileop.hpp>

using namespace pto;

namespace matmul_lowp_reuse_a_full_cache_detail {

// Compile-time contract for the full-A-cache schedule.
//
// The schedule keeps every A[m][k] slice in CellReg across the whole N axis.
// For each N tile it keeps one C chain per M128 strip and broadcasts each
// double-buffered B[k] tile to all of those chains.  It is profitable only
// when the complete A cache, all live C tiles and allocator headroom fit.
template <typename Input, typename Scale, int PackedFactor, bool UseMx,
          int ScaleGroup, int M, int N, int K, int TM, int TN, int TK>
struct Eligibility {
    static constexpr int KBlocks = K / TK;
    static constexpr int MBlocks = M / 128;
    static constexpr int ScaleCols = TK / ScaleGroup;
    static constexpr int PaddedScaleCols = (ScaleCols + 31) / 32 * 32;

    using ATile = CubeTileM32<Input, 32, TK>;
    using AScaleTile = CubeTileM32<Scale, 32, PaddedScaleCols, 32, ScaleCols>;
    using BMatrix = SharedMatrixRight<Input, TN, TK>;
    using BScaleMatrix = SharedMatrixRight<Scale, TN, PaddedScaleCols, TN, ScaleCols>;
    using CTile = CubeAccumulatorM32<float, 32, TN>;

    static constexpr int CellBytes =
        MBlocks * KBlocks *
            (ATile::LogicalTileBytes + AScaleTile::LogicalTileBytes) +
        MBlocks * CTile::LogicalTileBytes + 32 * 1024;
    static constexpr int BDataSharedBytes =
        (BMatrix::LogicalTileBytes + 4095) / 4096 * 4096;
    static constexpr int BScaleSharedBytes =
        (BScaleMatrix::LogicalTileBytes + 4095) / 4096 * 4096;
    static constexpr int SharedBytes =
        2 * (BDataSharedBytes + BScaleSharedBytes);

    // One M128 output consumes 16 KiB of internal-acc storage per Cube.
    // Current hardware/model capacity is 32 KiB, so at most two independent
    // M chains may stay raw across K without spilling or allocation stalls.
    static constexpr bool value =
        UseMx && TM == 128 && M % 128 == 0 && N % TN == 0 && K % TK == 0 &&
        MBlocks >= 1 && MBlocks <= 2 && KBlocks >= 2 && TK % ScaleGroup == 0 &&
        CellBytes <= 256 * 1024 && SharedBytes <= 256 * 1024;
};

} // namespace matmul_lowp_reuse_a_full_cache_detail

// Full-A-cache fast path. The public matmul_lowp_reuseA dispatcher calls this
// only when Eligibility::value is true; tails and capacity-limited shapes stay
// on the general implementation.
template <typename Input, typename Scale, int PackedFactor, bool UseMx,
          int ScaleGroup, int M, int N, int K, int TM, int TN, int TK>
void matmul_lowp_reuseA_full_cache(float *out, Input *ap, Input *bp,
                                   Scale *asp, Scale *bsp) {
    using FastPath = matmul_lowp_reuse_a_full_cache_detail::Eligibility<
        Input, Scale, PackedFactor, UseMx, ScaleGroup, M, N, K, TM, TN, TK>;
    static_assert(FastPath::value,
                  "matmul_lowp_reuseA_full_cache called for an ineligible shape");

    constexpr int KBlocks = FastPath::KBlocks;
    constexpr int MBlocks = FastPath::MBlocks;
    constexpr int LastK = KBlocks - 1;
    constexpr int ScaleCols = FastPath::ScaleCols;
    constexpr int PaddedScaleCols = FastPath::PaddedScaleCols;

    using GlobalA = global_tensor<Input, RowMajor<M, K / PackedFactor>>;
    using GlobalB = global_tensor<Input, RowMajor<N, K / PackedFactor>>;
    using GlobalAScale = global_tensor<Scale, RowMajor<M, K / ScaleGroup>>;
    using GlobalBScale = global_tensor<Scale, RowMajor<N, K / ScaleGroup>>;
    using GlobalC = global_tensor<float, RowMajor<M, N>>;
    using ATile = typename FastPath::ATile;
    using AScaleTile = typename FastPath::AScaleTile;
    using BMatrix = typename FastPath::BMatrix;
    using BScaleMatrix = typename FastPath::BScaleMatrix;
    using CTile = typename FastPath::CTile;

    const int tid = get_thread_idx();
    Input *a_cursor[MBlocks];
    Scale *a_scale_cursor[MBlocks];
#pragma clang loop unroll(full)
    for (int m = 0; m < MBlocks; ++m) {
        const int row = m * 128 + tid * 32;
        a_cursor[m] = ap + row * (K / PackedFactor);
        a_scale_cursor[m] = asp + row * (K / ScaleGroup);
    }

    ATile a[MBlocks][KBlocks];
    AScaleTile a_scale[MBlocks][KBlocks];
    CTile c[MBlocks];

#pragma clang loop unroll(full)
    for (int n = 0; n < N; n += TN) {
        SharedTile<BMatrix> b_even, b_odd;
        SharedTile<BScaleMatrix> b_scale_even, b_scale_odd;

        // Prologue: cache A[0] only for the first N tile and fill both B slots.
        if (n == 0) {
#pragma clang loop unroll(full)
            for (int m = 0; m < MBlocks; ++m) {
                GlobalA ga(a_cursor[m]);
                GlobalAScale gs(a_scale_cursor[m]);
                TLOAD_CUBE(a[m][0], ga);
                TLOAD_CUBE(a_scale[m][0], gs);
                a_cursor[m] += TK / PackedFactor;
                a_scale_cursor[m] += ScaleCols;
            }
        }
        GlobalB b0(bp + n * (K / PackedFactor));
        GlobalBScale bs0(bsp + n * (K / ScaleGroup));
        GlobalB b1(bp + n * (K / PackedFactor) + TK / PackedFactor);
        GlobalBScale bs1(bsp + n * (K / ScaleGroup) + ScaleCols);
        TLOAD<BMatrix, 1>(b_even, b0);
        TLOAD<BScaleMatrix, 1>(b_scale_even, bs0);
        TLOAD<BMatrix, 1>(b_odd, b1);
        TLOAD<BScaleMatrix, 1>(b_scale_odd, bs1);
#pragma clang loop unroll(full)
        for (int m = 0; m < MBlocks; ++m) {
            TMATMUL_MX<3>(c[m], a[m][0], a_scale[m][0], b_even, b_scale_even,
                          fixp::keep_acc().raw_acc(), 128);
        }

        // Steady state: each B tile is consumed by every M chain while the
        // alternate slot preloads k+1. Full unrolling folds the parity branch.
#pragma clang loop unroll(full)
        for (int k = 1; k < LastK; ++k) {
            if (n == 0) {
#pragma clang loop unroll(full)
                for (int m = 0; m < MBlocks; ++m) {
                    GlobalA ga(a_cursor[m]);
                    GlobalAScale gs(a_scale_cursor[m]);
                    TLOAD_CUBE(a[m][k], ga);
                    TLOAD_CUBE(a_scale[m][k], gs);
                    a_cursor[m] += TK / PackedFactor;
                    a_scale_cursor[m] += ScaleCols;
                }
            }
            GlobalB next_b(bp + n * (K / PackedFactor) +
                           (k + 1) * TK / PackedFactor);
            GlobalBScale next_scale(bsp + n * (K / ScaleGroup) +
                                    (k + 1) * ScaleCols);
            if (k & 1) {
                TLOAD<BMatrix, 1>(b_even, next_b);
                TLOAD<BScaleMatrix, 1>(b_scale_even, next_scale);
#pragma clang loop unroll(full)
                for (int m = 0; m < MBlocks; ++m) {
                    TMATMUL_MX_ACC<3>(
                        c[m], c[m], a[m][k], a_scale[m][k], b_odd, b_scale_odd,
                        fixp::keep_acc().raw_acc().acc_hint(), 128);
                }
            } else {
                TLOAD<BMatrix, 1>(b_odd, next_b);
                TLOAD<BScaleMatrix, 1>(b_scale_odd, next_scale);
#pragma clang loop unroll(full)
                for (int m = 0; m < MBlocks; ++m) {
                    TMATMUL_MX_ACC<3>(
                        c[m], c[m], a[m][k], a_scale[m][k], b_even, b_scale_even,
                        fixp::keep_acc().raw_acc().acc_hint(), 128);
                }
            }
        }

        // Epilogue: cache the final A slice and materialize C for TSTORE.
        if (n == 0) {
#pragma clang loop unroll(full)
            for (int m = 0; m < MBlocks; ++m) {
                GlobalA ga(a_cursor[m]);
                GlobalAScale gs(a_scale_cursor[m]);
                TLOAD_CUBE(a[m][LastK], ga);
                TLOAD_CUBE(a_scale[m][LastK], gs);
                a_cursor[m] += TK / PackedFactor;
                a_scale_cursor[m] += ScaleCols;
            }
        }
        if constexpr (LastK & 1) {
#pragma clang loop unroll(full)
            for (int m = 0; m < MBlocks; ++m) {
                TMATMUL_MX_ACC<3>(c[m], c[m], a[m][LastK], a_scale[m][LastK],
                                  b_odd, b_scale_odd,
                                  fixp::keep_acc().acc_hint(), 128);
            }
        } else {
#pragma clang loop unroll(full)
            for (int m = 0; m < MBlocks; ++m) {
                TMATMUL_MX_ACC<3>(c[m], c[m], a[m][LastK], a_scale[m][LastK],
                                  b_even, b_scale_even,
                                  fixp::keep_acc().acc_hint(), 128);
            }
        }

#pragma clang loop unroll(full)
        for (int m = 0; m < MBlocks; ++m) {
            GlobalC gc(out + (m * 128 + tid * 32) * N + n);
            TSTORE_CUBE(gc, c[m]);
        }
    }
}
