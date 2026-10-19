#pragma once

#include <common/pto_tileop.hpp>
#include <type_traits>
#include <utility>

using namespace pto;

namespace matmul_lowp_reuse_a_detail {

// Schedule: M block -> cache ALL A/ScaleA K slices -> N group -> K -> N.
// Cached A survives every N group. Only the resident C outputs are replaced.
// Each K step materializes C, which becomes Cin for the following K step.
// TailShared selects Shared vs Local A/ScaleA for M/K tails only; there is
// no reload fallback when the full-K cache exceeds the selected storage pool.
// B/ScaleB always use Shared tiles, including the N tail.
#ifndef MATMUL_LOWP_REUSE_A_MAX_RESIDENT_OUTPUTS
#define MATMUL_LOWP_REUSE_A_MAX_RESIDENT_OUTPUTS 2
#endif

template <typename Input, typename Scale, int PackedFactor, bool UseMx, int ScaleGroup,
          int gM, int gN, int gK, int tM, int tN, int tK, bool TailShared>
struct ReuseAKernel {
    // Preserve the lowp driver's ABI: HiF4 input rows are padded to tK.
    static constexpr int PhysicalK = UseMx && ScaleGroup == 64
        ? (gK + tK - 1) / tK * tK : gK;
    static constexpr int StoredK = PhysicalK / PackedFactor;
    static constexpr int PhysicalM = (gM + 63) / 64 * 64;
    static constexpr int GlobalScaleCols = (PhysicalK + ScaleGroup - 1) / ScaleGroup;
    static constexpr int TileScaleCols = (tK + ScaleGroup - 1) / ScaleGroup;
    static constexpr int PaddedScaleCols = (TileScaleCols + 31) / 32 * 32;
    static constexpr int FullKBlocks = gK / tK;
    static constexpr int TailK = gK % tK;
    static constexpr int MaxResidentOutputs = MATMUL_LOWP_REUSE_A_MAX_RESIDENT_OUTPUTS;
    static_assert(MaxResidentOutputs >= 2, "At least two C slots are needed for N tails");
    static constexpr int TilePoolBytes = 256 * 1024;

    struct Buffers {
        float *c;
        Input *a;
        Input *b;
        Scale *a_scale;
        Scale *b_scale;
    };

    using GlobalA = global_tensor<Input, RowMajor<PhysicalM, StoredK>>;
    using GlobalB = global_tensor<Input, RowMajor<gN, StoredK>>;
    using GlobalC = global_tensor<float, RowMajor<gM, gN>>;
    using GlobalAScale = global_tensor<Scale, RowMajor<PhysicalM, GlobalScaleCols>>;
    using GlobalBScale = global_tensor<Scale, RowMajor<gN, GlobalScaleCols>>;

    template <int ValidM, int ValidN>
    using Accumulator = std::conditional_t<ValidM <= 64,
        CubeAccumulatorM16<float, 16, tN, 16, ValidN>,
        CubeAccumulatorM32<float, 32, tN, 32, ValidN>>;

    // Descriptor choices are shared by cache allocation and each K consumer.
    template <int ValidM, int ValidK>
    struct CachedA {
        static constexpr int PeRows = ValidM <= 64 ? 16 : 32;
        static constexpr int GroupM = ValidM <= 64 ? 64 : 128;
        static constexpr int ScaleCols = (ValidK + ScaleGroup - 1) / ScaleGroup;
        static constexpr bool Shared = TailShared && (ValidM != tM || ValidK != tK);
        using Matrix = SharedMatrixLeft<Input, GroupM, tK, GroupM, ValidK>;
        using ScaleMatrix = SharedMatrixLeft<Scale, GroupM, PaddedScaleCols, GroupM, ScaleCols>;
        using Local = std::conditional_t<PeRows == 16,
            CubeTileM16<Input, PeRows, tK, PeRows, ValidK>,
            CubeTileM32<Input, PeRows, tK, PeRows, ValidK>>;
        using LocalScale = CubeTileM32<Scale, 32, PaddedScaleCols, PeRows, ScaleCols>;
        using Tile = std::conditional_t<Shared, SharedTile<Matrix>, Local>;
        using ScaleTile = std::conditional_t<Shared, SharedTile<ScaleMatrix>, LocalScale>;
        static constexpr int Bytes = Tile::LogicalTileBytes
            + (UseMx ? ScaleTile::LogicalTileBytes : 0);
    };

    template <int ValidM, int ResidentN, int TailN>
    static __attribute__((noinline)) void compute_m_cached(const Buffers &buffers, int m_begin) {
        using FullA = CachedA<ValidM, tK>;
        using LastA = CachedA<ValidM, TailK ? TailK : tK>;
        using FullC = Accumulator<ValidM, tN>;
        using TailC = Accumulator<ValidM, TailN ? TailN : tN>;
        using BMatrix = SharedMatrixRight<Input, tN, tK>;
        using BScale = SharedMatrixRight<Scale, tN, PaddedScaleCols>;
        constexpr int SharedCacheBytes = (FullA::Shared ? FullKBlocks * FullA::Bytes : 0)
            + (TailK && LastA::Shared ? LastA::Bytes : 0);
        constexpr int LocalCacheBytes = (!FullA::Shared ? FullKBlocks * FullA::Bytes : 0)
            + (TailK && !LastA::Shared ? LastA::Bytes : 0);
        constexpr int SharedBytes = SharedCacheBytes + BMatrix::LogicalTileBytes
            + (UseMx ? BScale::LogicalTileBytes : 0);
        constexpr int LocalBytes = LocalCacheBytes + ResidentN * FullC::LogicalTileBytes
            + (TailN ? TailC::LogicalTileBytes : 0);
        static_assert(SharedBytes <= TilePoolBytes,
            "Full-K A/scale cache plus B/scale exceeds SharedTile capacity; select local storage");
        static_assert(LocalBytes + 32 * 1024 <= TilePoolBytes,
            "Full-K A/scale cache plus C and 32 KiB headroom exceeds CellReg capacity; reduce resident C or tile size");

        typename FullA::Tile a[FullKBlocks ? FullKBlocks : 1];
        typename FullA::ScaleTile a_scale[FullKBlocks ? FullKBlocks : 1];
        typename LastA::Tile last_a;
        typename LastA::ScaleTile last_scale;
        const int tid = static_cast<int>(get_thread_idx());
        // Load the entire A K axis ONCE, before entering any N group.
#pragma clang loop unroll(full)
        for (int k = 0; k < FullKBlocks; ++k) {
            const int row = m_begin + (FullA::Shared ? 0 : tid * FullA::PeRows);
            GlobalA global_a(buffers.a + row * StoredK + k * tK / PackedFactor);
            if constexpr (FullA::Shared) TLOAD<typename FullA::Matrix, 1>(a[k], global_a);
            else TLOAD_CUBE(a[k], global_a);
            if constexpr (UseMx) {
                GlobalAScale global_scale(buffers.a_scale + row * GlobalScaleCols + k * tK / ScaleGroup);
                if constexpr (FullA::Shared) TLOAD<typename FullA::ScaleMatrix, 1>(a_scale[k], global_scale);
                else TLOAD_CUBE(a_scale[k], global_scale);
            }
        }
        if constexpr (TailK != 0) {
            const int row = m_begin + (LastA::Shared ? 0 : tid * LastA::PeRows);
            GlobalA global_a(buffers.a + row * StoredK + FullKBlocks * tK / PackedFactor);
            if constexpr (LastA::Shared) TLOAD<typename LastA::Matrix, 1>(last_a, global_a);
            else TLOAD_CUBE(last_a, global_a);
            if constexpr (UseMx) {
                GlobalAScale global_scale(buffers.a_scale + row * GlobalScaleCols + FullKBlocks * tK / ScaleGroup);
                if constexpr (LastA::Shared) TLOAD<typename LastA::ScaleMatrix, 1>(last_scale, global_scale);
                else TLOAD_CUBE(last_scale, global_scale);
            }
        }

        constexpr int FullNBlocks = gN / tN;
        // When N has no tail, retain the experiment's single N-group loop
        // and runtime final-group mask (no duplicated K-chain body).
        if constexpr (TailN == 0) {
#pragma clang loop unroll(disable)
            for (int n_begin = 0; n_begin < gN; n_begin += ResidentN * tN) {
                FullC c[ResidentN];
                TailC unused_tail;
                compute_cached_n_group<ValidM, ResidentN, 0>(buffers, c, unused_tail,
                    a, a_scale, last_a, last_scale, m_begin, n_begin,
                    std::make_integer_sequence<int, FullKBlocks>{});
            }
        } else {
            constexpr int LeadingGroups = FullNBlocks ? (FullNBlocks - 1) / ResidentN : 0;
#pragma clang loop unroll(disable)
            for (int group = 0; group < LeadingGroups; ++group) {
                FullC c[ResidentN];
                TailC unused_tail;
                compute_cached_n_group<ValidM, ResidentN, 0>(buffers, c, unused_tail,
                    a, a_scale, last_a, last_scale, m_begin, group * ResidentN * tN,
                    std::make_integer_sequence<int, FullKBlocks>{});
            }
            constexpr int FinalFullN = FullNBlocks - LeadingGroups * ResidentN;
            FullC c[FinalFullN ? FinalFullN : 1];
            TailC c_tail;
            compute_cached_n_group<ValidM, FinalFullN, TailN>(buffers, c, c_tail,
                a, a_scale, last_a, last_scale, m_begin, LeadingGroups * ResidentN * tN,
                std::make_integer_sequence<int, FullKBlocks>{});
        }
    }

    template <int ValidM, int FullN, int TailN, typename CTile, typename TailCTile,
              typename ATile, typename AScale, typename LastATile, typename LastScale, int... Ks>
    static PTO_SHARED_INLINE void compute_cached_n_group(
        const Buffers &buffers, CTile *c, TailCTile &c_tail,
        ATile *a, AScale *a_scale, LastATile &last_a, LastScale &last_scale,
        int m_begin, int n_begin, std::integer_sequence<int, Ks...>) {
        (compute_cached_k_step<ValidM, FullN, TailN, tK, Ks == 0>(
            buffers, c, c_tail, a[Ks], a_scale[Ks], n_begin, Ks * tK), ...);
        if constexpr (TailK != 0) {
            compute_cached_k_step<ValidM, FullN, TailN, TailK, FullKBlocks == 0>(
                buffers, c, c_tail, last_a, last_scale, n_begin, FullKBlocks * tK);
        }
#pragma clang loop unroll(full)
        for (int n = 0; n < FullN; ++n) {
            if (n_begin + n * tN < gN)
                store_output<ValidM>(buffers, c[n], m_begin, n_begin + n * tN);
        }
        if constexpr (TailN != 0)
            store_output<ValidM>(buffers, c_tail, m_begin, n_begin + FullN * tN);
    }

    template <int ValidM, int FullN, int TailN, int ValidK, bool FirstK,
              typename CTile, typename TailCTile, typename ATile, typename AScale>
    static PTO_SHARED_INLINE void compute_cached_k_step(
        const Buffers &buffers, CTile *c, TailCTile &c_tail,
        ATile &a, AScale &a_scale, int n_begin, int k_begin) {
        // LastK=true deliberately materializes each partial C. The next K
        // step consumes it as Cin, exactly as in the full-K-cache experiment.
#pragma clang loop unroll(full)
        for (int n = 0; n < FullN; ++n) {
            if (n_begin + n * tN < gN)
                load_b_and_accumulate<ValidM, tN, ValidK, FirstK, true>(
                    buffers, c[n], a, a_scale, n_begin + n * tN, k_begin);
        }
        if constexpr (TailN != 0)
            load_b_and_accumulate<ValidM, TailN, ValidK, FirstK, true>(
                buffers, c_tail, a, a_scale, n_begin + FullN * tN, k_begin);
    }


    // One B tile feeds one C chain. ValidN specializes full and tail N tiles;
    // the caller keeps A alive so both paths reuse the same A load.
    template <int ValidM, int ValidN, int ValidK, bool FirstK, bool LastK,
              typename CTile, typename ATile, typename AScaleTile>
    static PTO_SHARED_INLINE void load_b_and_accumulate(
        const Buffers &buffers, CTile &c, ATile &a, AScaleTile &a_scale,
        int n_begin, int k_begin) {
        using BMatrix = SharedMatrixRight<Input, tN, tK, ValidN, ValidK>;
        using BScale = SharedMatrixRight<Scale, tN, PaddedScaleCols,
                                         ValidN, (ValidK + ScaleGroup - 1) / ScaleGroup>;
        SharedTile<BMatrix> b;
        SharedTile<BScale> b_scale;
        GlobalB global_b(buffers.b + n_begin * StoredK + k_begin / PackedFactor);
        TLOAD<BMatrix, 1>(b, global_b);
        if constexpr (UseMx) {
            GlobalBScale global_scale(buffers.b_scale + n_begin * GlobalScaleCols + k_begin / ScaleGroup);
            TLOAD<BScale, 1>(b_scale, global_scale);
        }

        // K-chain state for this C tile (FirstK/LastK are compile-time flags):
        //   first + last: one K tile, initialize C and finish immediately;
        //   first only:   initialize C and keep a raw partial sum;
        //   neither:      read the preceding partial sum and keep accumulating;
        //   last only:    read the partial sum and make the final C available.
        // raw_acc() keeps the intermediate result in the raw accumulator;
        // acc_hint() tells the internal-accumulator path to consume that chain.
        // The last step omits raw_acc(), so store_output can read the final C.
        // The branches below also select MX/plain and Shared/Local A APIs;
        // Local A passes the padded group M; valid_row is a load/store mask,
        // not the matrix instruction's execution size.
        constexpr int ComputeM = ValidM <= 64 ? 64 : 128;
        constexpr auto keep = fixp::keep_acc();
        if constexpr (UseMx) {
            if constexpr (is_shared_tile_v<ATile>) {
                if constexpr (FirstK && LastK)
                    TMATMUL_MX<3>(c, a, a_scale, b, b_scale, keep);
                else if constexpr (FirstK)
                    TMATMUL_MX<3>(c, a, a_scale, b, b_scale, keep.raw_acc());
                else if constexpr (LastK)
                    TMATMUL_MX_ACC<3>(c, c, a, a_scale, b, b_scale, keep.acc_hint());
                else
                    TMATMUL_MX_ACC<3>(c, c, a, a_scale, b, b_scale, keep.raw_acc().acc_hint());
            } else {
                if constexpr (FirstK && LastK)
                    TMATMUL_MX<3>(c, a, a_scale, b, b_scale, keep, ComputeM);
                else if constexpr (FirstK)
                    TMATMUL_MX<3>(c, a, a_scale, b, b_scale, keep.raw_acc(), ComputeM);
                else if constexpr (LastK)
                    TMATMUL_MX_ACC<3>(c, c, a, a_scale, b, b_scale, keep.acc_hint(), ComputeM);
                else
                    TMATMUL_MX_ACC<3>(c, c, a, a_scale, b, b_scale, keep.raw_acc().acc_hint(), ComputeM);
            }
        } else {
            if constexpr (is_shared_tile_v<ATile>) {
                if constexpr (FirstK && LastK)
                    TMATMUL(c, a, b, keep);
                else if constexpr (FirstK)
                    TMATMUL(c, a, b, keep.raw_acc());
                else if constexpr (LastK)
                    TMATMUL_ACC(c, c, a, b, keep.acc_hint());
                else
                    TMATMUL_ACC(c, c, a, b, keep.raw_acc().acc_hint());
            } else {
                if constexpr (FirstK && LastK)
                    TMATMUL(c, a, b, keep, ComputeM);
                else if constexpr (FirstK)
                    TMATMUL(c, a, b, keep.raw_acc(), ComputeM);
                else if constexpr (LastK)
                    TMATMUL_ACC(c, c, a, b, keep.acc_hint(), ComputeM);
                else
                    TMATMUL_ACC(c, c, a, b, keep.raw_acc().acc_hint(), ComputeM);
            }
        }
    }

    template <int ValidM, typename CTile>
    static PTO_SHARED_INLINE void store_output(
        const Buffers &buffers, CTile &c, int m_begin, int n_begin) {
        constexpr int PeRows = CTile::Rows;
        const int local_row = static_cast<int>(get_thread_idx()) * PeRows;
        if (local_row < ValidM) {
            using OutputTile = std::conditional_t<PeRows == 16,
                CubeAccumulatorM16<float, PeRows, tN, -1, CTile::ValidCol>,
                CubeAccumulatorM32<float, PeRows, tN, -1, CTile::ValidCol>>;
            const int valid_rows = ValidM - local_row < PeRows ? ValidM - local_row : PeRows;
            OutputTile output(valid_rows);
            output.data() = c.data();
            GlobalC global_c(buffers.c + (m_begin + local_row) * gN + n_begin);
            TSTORE_CUBE(global_c, output);
        }
    }
};

} // namespace matmul_lowp_reuse_a_detail

// Public template argument order is unchanged for existing callers.
// Input contract: each A/ScaleA batch must allocate ceil(gM/64)*64 rows;
// rows [gM, ceil(gM/64)*64) must be zero-filled. C remains exactly gM x gN.
template <typename Input, typename Scale, int PackedFactor, bool UseMx, int ScaleGroup,
          int gM, int gN, int gK, int tM, int tN, int tK, bool TailShared = false>
void matmul_lowp_reuseA(float *c, Input *a, Input *b, Scale *a_scale, Scale *b_scale) {
    static_assert(gM > 0 && gN > 0 && gK > 0);
    static_assert(tM == 64 || tM == 128, "tM must be 64 or 128");
    static_assert(tN > 0 && (tN & (tN - 1)) == 0);
    static_assert(tK > 0 && (tK & (tK - 1)) == 0);
    static_assert(PackedFactor == 1 || PackedFactor == 2);
    static_assert(gK % PackedFactor == 0 && tK % PackedFactor == 0);
    static_assert(!UseMx || tK % ScaleGroup == 0,
                  "Each K tile must start on a scale-group boundary");

    using Kernel = matmul_lowp_reuse_a_detail::ReuseAKernel<
        Input, Scale, PackedFactor, UseMx, ScaleGroup, gM, gN, gK, tM, tN, tK, TailShared>;
    const typename Kernel::Buffers buffers{c, a, b, a_scale, b_scale};
    constexpr int FullNBlocks = gN / tN;
    constexpr int TailN = gN % tN;
    // Reserve one C slot for the N tail. All groups share the full-K A cache.
    constexpr int GroupWidth = Kernel::MaxResidentOutputs - (TailN != 0);

    // Full M blocks: cache A once, then finish one N group's K axis at a time.
#pragma clang loop unroll(disable)
    for (int m = 0; m < gM / tM; ++m) {
        Kernel::template compute_m_cached<tM, GroupWidth, TailN>(buffers, m * tM);
    }

    // M tails always execute as M64 strips. A 65..127-row tail has one
    // complete M64 strip and one padded M64 strip, both retaining N reuse.
    if constexpr (gM % tM != 0) {
        constexpr int TailM = gM % tM;
        constexpr int TailMBegin = gM / tM * tM;
        if constexpr (TailM > 64) {
            Kernel::template compute_m_cached<64, GroupWidth, TailN>(buffers, TailMBegin);
        }
        constexpr int LastM = TailM > 64 ? TailM - 64 : TailM;
        constexpr int LastMBegin = TailMBegin + (TailM > 64 ? 64 : 0);
        Kernel::template compute_m_cached<LastM, GroupWidth, TailN>(buffers, LastMBegin);
    }
}
