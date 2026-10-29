#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <type_traits>
#include <tuple>
#include <utility>

using namespace pto;

// Opt in after benchmarking: full caching can reduce C traffic but also
// reduce overlap between N outputs. The streaming paired path is the default.
#ifndef MATMUL_REUSE_A_FULL_CACHE
#define MATMUL_REUSE_A_FULL_CACHE 0
#endif
#ifndef MATMUL_REUSE_A_B_BUFFERS
#define MATMUL_REUSE_A_B_BUFFERS 2
#endif
#ifndef MATMUL_REUSE_A_N_GROUP
#define MATMUL_REUSE_A_N_GROUP 1
#endif
// Experimental: permit more live chains than the internal accumulator budget.
// CellReg and SharedTile capacity limits remain enforced.
#ifndef MATMUL_REUSE_A_ALLOW_ACC_SPILL
#define MATMUL_REUSE_A_ALLOW_ACC_SPILL 0
#endif
static_assert(MATMUL_REUSE_A_N_GROUP >= 1);
static_assert(MATMUL_REUSE_A_B_BUFFERS >= 2 && MATMUL_REUSE_A_B_BUFFERS <= 4);

template <typename TileT, int Count>
struct MatmulReuseAStorage {
    TileT tiles[Count];
    TileT &operator[](int index) { return tiles[index]; }
};

namespace matmul_reuse_a_detail {

constexpr int floor_power_of_two(int value) {
    int result = 1;
    while ((result << 1) <= value) result <<= 1;
    return result;
}

constexpr int compute_group_m(int valid_m) {
    // Current model requires power-of-two TMATMUL M.
    int rows = 1;
    while (rows < valid_m) rows <<= 1;
    return rows;
}

template <int ValidM, int PeM>
PTO_SHARED_INLINE int local_m_offset(int tid) {
    if constexpr (ValidM > 64 && ValidM < 128) {
        // Model requires full 32-row Local-A descriptors even when a load
        // zero-pads its CELL. Shift the last shard inside A instead. E.g.
        // M120 starts are 0,32,64,88; stores partition 32,32,24,32 rows.
        return tid * PeM < ValidM - PeM ? tid * PeM : ValidM - PeM;
    }
    return tid * PeM < ValidM ? tid * PeM : 0;
}

template <int ValidM, typename TileC, typename GmC>
PTO_SHARED_INLINE void store_c(GmC &gc, TileC &c, int tid) {
    constexpr int PeM = TileC::Rows;
    if constexpr (ValidM > 64 && ValidM < 128) {
        using StoreC = CubeAccumulatorM32<float, PeM, TileC::Cols,
                                         -1, TileC::ValidCol>;
        const int start = local_m_offset<ValidM, PeM>(tid);
        const int end = tid == 3 ? ValidM : local_m_offset<ValidM, PeM>(tid + 1);
        if (end > start) {
            StoreC stored(end - start);
            stored.data() = c.data();
            TSTORE_CUBE(gc, stored);
        }
    } else if constexpr (ValidM >= 4 * PeM) {
        TSTORE_CUBE(gc, c);
    } else if (tid * PeM < ValidM) {
        using StoreC = std::conditional_t<PeM == 16,
            CubeAccumulatorM16<float, PeM, TileC::Cols, -1, TileC::ValidCol>,
            CubeAccumulatorM32<float, PeM, TileC::Cols, -1, TileC::ValidCol>>;
        const int remaining = ValidM - tid * PeM;
        StoreC stored(remaining < PeM ? remaining : PeM);
        stored.data() = c.data();
        TSTORE_CUBE(gc, stored);
    }
}

// Represent N as full tiles followed by power-of-two tail chunks. Every
// chunk participates in the same resident grouping, including N < tN.
constexpr int n_chunk_width(int remaining, int tile_n) {
    return remaining >= tile_n ? tile_n : floor_power_of_two(remaining);
}
constexpr int n_chunk_count(int n, int tile_n) {
    int count = 0;
    while (n > 0) { n -= n_chunk_width(n, tile_n); ++count; }
    return count;
}
constexpr int n_chunk_begin(int n, int tile_n, int index) {
    int begin = 0;
    for (int i = 0; i < index; ++i)
        begin += n_chunk_width(n - begin, tile_n);
    return begin;
}

template <int PeM, int tN, int ValidN>
using NAccumulator = std::conditional_t<PeM == 16,
    CubeAccumulatorM16<float, PeM, tN, PeM, ValidN>,
    CubeAccumulatorM32<float, PeM, tN, PeM, ValidN>>;

template <bool FirstK, bool LastK, int Begin, typename dtype,
          int gM, int gN, int gK, int tN, int tK, int ValidM,
          typename Accumulators, size_t... Is>
PTO_SHARED_INLINE void run_n_group_k(
    dtype *a_ptr, dtype *b_ptr, Accumulators &cs,
    int m_begin, int k, int tid, std::index_sequence<Is...>) {
    constexpr int PeM = ValidM <= 64 ? 16 : 32;
    constexpr int ComputeM = compute_group_m(ValidM);
    using TileA = std::conditional_t<PeM == 16,
        CubeTileM16<dtype, PeM, tK>, CubeTileM32<dtype, PeM, tK>>;
    using GmA = global_tensor<dtype, RowMajor<gM, gK>>;
    using GmB = global_tensor<dtype, RowMajor<gN, gK>>;
    constexpr auto options = fixp::keep_acc();
    TileA a;
    GmA ga(a_ptr + (m_begin + local_m_offset<ValidM, PeM>(tid)) * gK + k * tK);
    TLOAD_CUBE(a, ga);
    // Compile-time expansion keeps each differently masked C chain distinct.
    ([&] {
        constexpr int NBegin = n_chunk_begin(gN, tN, Begin + Is);
        constexpr int ValidN = n_chunk_width(gN - NBegin, tN);
        using BMatrix = SharedMatrixRight<dtype, tN, tK, ValidN, tK>;
        SharedTile<BMatrix> b;
        GmB gb(b_ptr + NBegin * gK + k * tK);
        TLOAD<BMatrix, 1>(b, gb);
        auto &c = std::get<Is>(cs);
        if constexpr (FirstK && LastK)
            TMATMUL(c, a, b, options, ComputeM);
        else if constexpr (FirstK)
            TMATMUL(c, a, b, options.raw_acc(), ComputeM);
        else if constexpr (LastK)
            TMATMUL_ACC(c, c, a, b, options.acc_hint(), ComputeM);
        else
            TMATMUL_ACC(c, c, a, b, options.raw_acc().acc_hint(), ComputeM);
    }(), ...);
}

template <int Begin, typename dtype, int gM, int gN, int gK,
          int tN, int tK, int ValidM, size_t... Is>
PTO_SHARED_INLINE void run_n_group(float *c_ptr, dtype *a_ptr, dtype *b_ptr,
                                  int m_begin, std::index_sequence<Is...> seq) {
    constexpr int PeM = ValidM <= 64 ? 16 : 32;
    constexpr int Kb = gK / tK;
    std::tuple<NAccumulator<PeM, tN,
        n_chunk_width(gN - n_chunk_begin(gN, tN, Begin + Is), tN)>...> cs;
    const int tid = static_cast<int>(get_thread_idx());
    run_n_group_k<true, Kb == 1, Begin, dtype,
                  gM, gN, gK, tN, tK, ValidM>(a_ptr, b_ptr, cs, m_begin, 0, tid, seq);
#pragma clang loop unroll(disable)
    for (int k = 1; k < Kb - 1; ++k)
        run_n_group_k<false, false, Begin, dtype,
                      gM, gN, gK, tN, tK, ValidM>(a_ptr, b_ptr, cs, m_begin, k, tid, seq);
    if constexpr (Kb > 1)
        run_n_group_k<false, true, Begin, dtype,
                      gM, gN, gK, tN, tK, ValidM>(a_ptr, b_ptr, cs, m_begin, Kb - 1, tid, seq);
    ([&] {
        using GmC = global_tensor<float, RowMajor<gM, gN>>;
        constexpr int NBegin = n_chunk_begin(gN, tN, Begin + Is);
        GmC gc(c_ptr + (m_begin + local_m_offset<ValidM, PeM>(tid)) * gN + NBegin);
        store_c<ValidM>(gc, std::get<Is>(cs), tid);
    }(), ...);
}

template <typename dtype, int gM, int gN, int gK, int tN, int tK,
          int ValidM, int Begin = 0>
PTO_SHARED_INLINE void run_n_groups(float *c_ptr, dtype *a_ptr, dtype *b_ptr,
                                  int m_begin) {
    constexpr int PeM = ValidM <= 64 ? 16 : 32;
    using TileA = std::conditional_t<PeM == 16,
        CubeTileM16<dtype, PeM, tK>, CubeTileM32<dtype, PeM, tK>>;
    using TileC = NAccumulator<PeM, tN, tN>;
    using BMatrix = SharedMatrixRight<dtype, tN, tK>;
    constexpr int Capacity = (256 * 1024 - TileA::LogicalTileBytes) / TileC::LogicalTileBytes;
    constexpr int Resident = Capacity < 4 ? Capacity : 4;
    constexpr int Remaining = n_chunk_count(gN, tN) - Begin;
    constexpr int Count = Remaining < Resident ? Remaining : Resident;
    static_assert(ValidM >= 1 && ValidM <= 128);
    static_assert(Count > 0, "CellReg must hold A and at least one C tile");
    static_assert(BMatrix::LogicalTileBytes <= 256 * 1024,
                  "B tile exceeds the 256 KiB SharedTReg pool");
    static_assert(TileA::LogicalTileBytes + Count * TileC::LogicalTileBytes <= 256 * 1024,
                  "A and resident C group exceed CellReg capacity");
    run_n_group<Begin, dtype, gM, gN, gK, tN, tK, ValidM>(
        c_ptr, a_ptr, b_ptr, m_begin, std::make_index_sequence<Count>{});
    if constexpr (Remaining > Count)
        run_n_groups<dtype, gM, gN, gK, tN, tK, ValidM, Begin + Count>(
            c_ptr, a_ptr, b_ptr, m_begin);
}

// Pair two M strips in one K loop: each B tile feeds both independent
// accumulator chains before advancing N. A remains resident across N.
template <typename dtype, int ValidM, int tK>
using LocalA = std::conditional_t<(ValidM <= 64),
    CubeTileM16<dtype, 16, tK>, CubeTileM32<dtype, 32, tK>>;

template <bool FirstK, bool LastK, int ValidM, typename C, typename A, typename B>
PTO_SHARED_INLINE void update_chain(C &c, A &a, B &b) {
    constexpr auto options = fixp::keep_acc();
    constexpr int ComputeM = compute_group_m(ValidM);
    if constexpr (FirstK && LastK)
        TMATMUL(c, a, b, options, ComputeM);
    else if constexpr (FirstK)
        TMATMUL(c, a, b, options.raw_acc(), ComputeM);
    else if constexpr (LastK)
        TMATMUL_ACC(c, c, a, b, options.acc_hint(), ComputeM);
    else
        TMATMUL_ACC(c, c, a, b, options.raw_acc().acc_hint(), ComputeM);
}

template <bool FirstK, bool LastK, int Begin, typename dtype,
          int gM, int gN, int gK, int tN, int tK, int M0, int M1,
          typename Cs0, typename Cs1, size_t... Is>
PTO_SHARED_INLINE void run_m_pair_k(dtype *ap, dtype *bp, Cs0 &cs0, Cs1 &cs1,
                                   int m, int k, int tid, std::index_sequence<Is...>) {
    LocalA<dtype, M0, tK> a0;
    LocalA<dtype, M1, tK> a1;
    using GA = global_tensor<dtype, RowMajor<gM, gK>>;
    using GB = global_tensor<dtype, RowMajor<gN, gK>>;
    GA ga0(ap + (m + local_m_offset<M0, decltype(a0)::Rows>(tid)) * gK + k * tK);
    GA ga1(ap + (m + M0 + local_m_offset<M1, decltype(a1)::Rows>(tid)) * gK + k * tK);
    TLOAD_CUBE(a0, ga0);
    TLOAD_CUBE(a1, ga1);
    ([&] {
        constexpr int NBegin = n_chunk_begin(gN, tN, Begin + Is);
        constexpr int ValidN = n_chunk_width(gN - NBegin, tN);
        using BM = SharedMatrixRight<dtype, tN, tK, ValidN, tK>;
        SharedTile<BM> b;
        GB gb(bp + NBegin * gK + k * tK);
        TLOAD<BM, 1>(b, gb);
        update_chain<FirstK, LastK, M0>(std::get<Is>(cs0), a0, b);
        update_chain<FirstK, LastK, M1>(std::get<Is>(cs1), a1, b);
    }(), ...);
}

template <int Begin, typename dtype, int gM, int gN, int gK,
          int tN, int tK, int M0, int M1, size_t... Is>
PTO_SHARED_INLINE void run_m_pair_group(float *cp, dtype *ap, dtype *bp,
                                       int m, std::index_sequence<Is...> seq) {
    constexpr int P0 = M0 <= 64 ? 16 : 32;
    constexpr int P1 = M1 <= 64 ? 16 : 32;
    constexpr int Kb = gK / tK;
    std::tuple<NAccumulator<P0, tN,
        n_chunk_width(gN - n_chunk_begin(gN, tN, Begin + Is), tN)>...> cs0;
    std::tuple<NAccumulator<P1, tN,
        n_chunk_width(gN - n_chunk_begin(gN, tN, Begin + Is), tN)>...> cs1;
    const int tid = static_cast<int>(get_thread_idx());
    run_m_pair_k<true, Kb == 1, Begin, dtype, gM, gN, gK, tN, tK, M0, M1>(
        ap, bp, cs0, cs1, m, 0, tid, seq);
#pragma clang loop unroll(disable)
    for (int k = 1; k < Kb - 1; ++k)
        run_m_pair_k<false, false, Begin, dtype, gM, gN, gK, tN, tK, M0, M1>(
            ap, bp, cs0, cs1, m, k, tid, seq);
    if constexpr (Kb > 1)
        run_m_pair_k<false, true, Begin, dtype, gM, gN, gK, tN, tK, M0, M1>(
            ap, bp, cs0, cs1, m, Kb - 1, tid, seq);
    ([&] {
        using GC = global_tensor<float, RowMajor<gM, gN>>;
        constexpr int NBegin = n_chunk_begin(gN, tN, Begin + Is);
        GC gc0(cp + (m + local_m_offset<M0, P0>(tid)) * gN + NBegin);
        GC gc1(cp + (m + M0 + local_m_offset<M1, P1>(tid)) * gN + NBegin);
        store_c<M0>(gc0, std::get<Is>(cs0), tid);
        store_c<M1>(gc1, std::get<Is>(cs1), tid);
    }(), ...);
}

// Full-K A caching permits N-major scheduling with only two live C chains.
// Each A slice is loaded beside its first consumer, not in a preload barrier.
template <int KIndex, bool LoadA, typename dtype, int gM, int gN, int gK,
          int tN, int tK, int M0, int M1, typename C0, typename C1, typename BT>
PTO_SHARED_INLINE void cached_pair_step(dtype *ap, dtype *bp,
    LocalA<dtype, M0, tK> *a0, LocalA<dtype, M1, tK> *a1,
    C0 &c0, C1 &c1, BT *bs, int m, int n, int tid) {
    constexpr int NextK = KIndex + MATMUL_REUSE_A_B_BUFFERS - 1;
    if constexpr (NextK < gK / tK) {
        using BM = SharedMatrixRight<dtype, tN, tK, C0::ValidCol, tK>;
        using GB = global_tensor<dtype, RowMajor<gN, gK>>;
        GB gb(bp + n * gK + NextK * tK);
        TLOAD<BM, 1>(bs[NextK % MATMUL_REUSE_A_B_BUFFERS], gb);
    }
    if constexpr (LoadA) {
        using GA = global_tensor<dtype, RowMajor<gM, gK>>;
        GA ga0(ap + (m + local_m_offset<M0, LocalA<dtype, M0, tK>::Rows>(tid)) * gK + KIndex * tK);
        GA ga1(ap + (m + M0 + local_m_offset<M1, LocalA<dtype, M1, tK>::Rows>(tid)) * gK + KIndex * tK);
        TLOAD_CUBE(a0[KIndex], ga0);
        TLOAD_CUBE(a1[KIndex], ga1);
    }
    auto &b = bs[KIndex % MATMUL_REUSE_A_B_BUFFERS];
    update_chain<KIndex == 0, KIndex + 1 == gK / tK, M0>(c0, a0[KIndex], b);
    update_chain<KIndex == 0, KIndex + 1 == gK / tK, M1>(c1, a1[KIndex], b);
}

template <int NIndex, typename dtype, int gM, int gN, int gK,
          int tN, int tK, int M0, int M1, size_t... Ks>
PTO_SHARED_INLINE void cached_pair_n(float *cp, dtype *ap, dtype *bp,
    LocalA<dtype, M0, tK> *a0, LocalA<dtype, M1, tK> *a1,
    int m, int tid, std::index_sequence<Ks...> seq) {
    constexpr int NBegin = n_chunk_begin(gN, tN, NIndex);
    constexpr int ValidN = n_chunk_width(gN - NBegin, tN);
    constexpr int P0 = LocalA<dtype, M0, tK>::Rows;
    constexpr int P1 = LocalA<dtype, M1, tK>::Rows;
    NAccumulator<P0, tN, ValidN> c0;
    NAccumulator<P1, tN, ValidN> c1;
    using BM = SharedMatrixRight<dtype, tN, tK, ValidN, tK>;
    SharedTile<BM> bs[MATMUL_REUSE_A_B_BUFFERS];
    using GB = global_tensor<dtype, RowMajor<gN, gK>>;
#pragma clang loop unroll(full)
    for (int k = 0; k < MATMUL_REUSE_A_B_BUFFERS - 1 && k < gK / tK; ++k) {
        GB gb(bp + NBegin * gK + k * tK);
        TLOAD<BM, 1>(bs[k], gb);
    }
    (cached_pair_step<Ks, NIndex == 0, dtype, gM, gN, gK, tN, tK, M0, M1>(
        ap, bp, a0, a1, c0, c1, bs, m, NBegin, tid), ...);
    using GC = global_tensor<float, RowMajor<gM, gN>>;
    GC gc0(cp + (m + local_m_offset<M0, P0>(tid)) * gN + NBegin);
    GC gc1(cp + (m + M0 + local_m_offset<M1, P1>(tid)) * gN + NBegin);
    store_c<M0>(gc0, c0, tid);
    store_c<M1>(gc1, c1, tid);
    if constexpr (NIndex + 1 < n_chunk_count(gN, tN))
        cached_pair_n<NIndex + 1, dtype, gM, gN, gK, tN, tK, M0, M1>(
            cp, ap, bp, a0, a1, m, tid, seq);
}

// Group narrow N tiles while keeping their raw C chains inside the 32 KiB
// internal-accumulator budget. All groups share the full-K local A cache.
template <int KIndex, int Begin, typename dtype, int gM, int gN, int gK,
          int tN, int tK, int M0, int M1, typename Cs0, typename Cs1,
          typename Bs, size_t... Is>
PTO_SHARED_INLINE void cached_group_step(dtype *ap, dtype *bp,
    LocalA<dtype, M0, tK> *a0, LocalA<dtype, M1, tK> *a1,
    Cs0 &cs0, Cs1 &cs1, Bs &bs, int m, int tid, std::index_sequence<Is...>) {
    if constexpr (Begin == 0) {
        using GA = global_tensor<dtype, RowMajor<gM, gK>>;
        GA ga0(ap + (m + local_m_offset<M0, LocalA<dtype, M0, tK>::Rows>(tid)) * gK + KIndex * tK);
        GA ga1(ap + (m + M0 + local_m_offset<M1, LocalA<dtype, M1, tK>::Rows>(tid)) * gK + KIndex * tK);
        TLOAD_CUBE(a0[KIndex], ga0);
        TLOAD_CUBE(a1[KIndex], ga1);
    }
    (cached_pair_step<KIndex, false, dtype, gM, gN, gK, tN, tK, M0, M1>(
        ap, bp, a0, a1, std::get<Is>(cs0), std::get<Is>(cs1),
        std::get<Is>(bs).tiles, m, n_chunk_begin(gN, tN, Begin + Is), tid), ...);
}

template <int Begin, typename dtype, int gM, int gN, int gK,
          int tN, int tK, int M0, int M1, size_t... Is, size_t... Ks>
PTO_SHARED_INLINE void cached_group(float *cp, dtype *ap, dtype *bp,
    LocalA<dtype, M0, tK> *a0, LocalA<dtype, M1, tK> *a1, int m, int tid,
    std::index_sequence<Is...> ns, std::index_sequence<Ks...>) {
    constexpr int P0 = LocalA<dtype, M0, tK>::Rows;
    constexpr int P1 = LocalA<dtype, M1, tK>::Rows;
    std::tuple<NAccumulator<P0,tN,n_chunk_width(gN-n_chunk_begin(gN,tN,Begin+Is),tN)>...> cs0;
    std::tuple<NAccumulator<P1,tN,n_chunk_width(gN-n_chunk_begin(gN,tN,Begin+Is),tN)>...> cs1;
    std::tuple<MatmulReuseAStorage<SharedTile<SharedMatrixRight<dtype,tN,tK,
        n_chunk_width(gN-n_chunk_begin(gN,tN,Begin+Is),tN),tK>>,MATMUL_REUSE_A_B_BUFFERS>...> bs;
    ([&] {
        constexpr int NBegin = n_chunk_begin(gN,tN,Begin+Is);
        using BM = SharedMatrixRight<dtype,tN,tK,n_chunk_width(gN-NBegin,tN),tK>;
        using GB = global_tensor<dtype, RowMajor<gN,gK>>;
#pragma clang loop unroll(full)
        for (int k=0;k<MATMUL_REUSE_A_B_BUFFERS-1 && k<gK/tK;++k) {
            GB gb(bp+NBegin*gK+k*tK);
            TLOAD<BM,1>(std::get<Is>(bs)[k],gb);
        }
    }(), ...);
    (cached_group_step<Ks,Begin,dtype,gM,gN,gK,tN,tK,M0,M1>(
        ap,bp,a0,a1,cs0,cs1,bs,m,tid,ns), ...);
    ([&] {
        using GC = global_tensor<float,RowMajor<gM,gN>>;
        constexpr int NBegin = n_chunk_begin(gN,tN,Begin+Is);
        GC gc0(cp+(m+local_m_offset<M0,P0>(tid))*gN+NBegin);
        GC gc1(cp+(m+M0+local_m_offset<M1,P1>(tid))*gN+NBegin);
        store_c<M0>(gc0,std::get<Is>(cs0),tid);
        store_c<M1>(gc1,std::get<Is>(cs1),tid);
    }(), ...);
}

template <int Begin, typename dtype, int gM, int gN, int gK,
          int tN, int tK, int M0, int M1>
PTO_SHARED_INLINE void cached_groups(float *cp, dtype *ap, dtype *bp,
    LocalA<dtype,M0,tK> *a0, LocalA<dtype,M1,tK> *a1, int m, int tid) {
    constexpr int CBytes = NAccumulator<LocalA<dtype,M0,tK>::Rows,tN,tN>::LogicalTileBytes +
                          NAccumulator<LocalA<dtype,M1,tK>::Rows,tN,tN>::LogicalTileBytes;
    constexpr int ABytes = (gK/tK)*(LocalA<dtype,M0,tK>::LogicalTileBytes+LocalA<dtype,M1,tK>::LogicalTileBytes);
    constexpr int BBytes = MATMUL_REUSE_A_B_BUFFERS * ((SharedMatrixRight<dtype,tN,tK>::LogicalTileBytes+4095)/4096*4096);
    constexpr int AccLimit = 32768/CBytes;
    constexpr int CellLimit = (262144-32768-ABytes)/CBytes;
    constexpr int SharedLimit = 262144/BBytes;
    constexpr int L0 = MATMUL_REUSE_A_ALLOW_ACC_SPILL ? CellLimit :
        (AccLimit<CellLimit ? AccLimit:CellLimit);
    constexpr int L1 = L0<SharedLimit ? L0:SharedLimit;
    constexpr int Limit = L1<MATMUL_REUSE_A_N_GROUP ? L1:MATMUL_REUSE_A_N_GROUP;
    constexpr int Left = n_chunk_count(gN,tN)-Begin;
    constexpr int Count = Left<Limit ? Left:Limit;
    static_assert(Count>0);
    cached_group<Begin,dtype,gM,gN,gK,tN,tK,M0,M1>(cp,ap,bp,a0,a1,m,tid,
        std::make_index_sequence<Count>{},std::make_index_sequence<gK/tK>{});
    if constexpr (Left>Count)
        cached_groups<Begin+Count,dtype,gM,gN,gK,tN,tK,M0,M1>(cp,ap,bp,a0,a1,m,tid);
}

template <typename dtype, int gM, int gN, int gK, int tN, int tK,
          int M0, int M1, int Begin = 0>
PTO_SHARED_INLINE void run_m_pair(float *cp, dtype *ap, dtype *bp, int m) {
    constexpr int ABytes = LocalA<dtype, M0, tK>::LogicalTileBytes +
                           LocalA<dtype, M1, tK>::LogicalTileBytes;
    constexpr int CBytes = NAccumulator<(M0 <= 64 ? 16 : 32), tN, tN>::LogicalTileBytes +
                           NAccumulator<(M1 <= 64 ? 16 : 32), tN, tN>::LogicalTileBytes;
    constexpr int Capacity = (256 * 1024 - ABytes) / CBytes;
    constexpr int Resident = Capacity < 4 ? Capacity : 4;
    constexpr int Remaining = n_chunk_count(gN, tN) - Begin;
    constexpr int Count = Remaining < Resident ? Remaining : Resident;
    constexpr int CachedBytes = (gK / tK) * ABytes + CBytes + 32 * 1024;
    constexpr int BBytes = SharedMatrixRight<dtype, tN, tK>::LogicalTileBytes;
    constexpr int BufferedBBytes = MATMUL_REUSE_A_B_BUFFERS * ((BBytes + 4095) / 4096 * 4096);
    constexpr bool FullCache = MATMUL_REUSE_A_FULL_CACHE && Begin == 0 && gK / tK >= 2 &&
        CachedBytes <= 256 * 1024 && BufferedBBytes <= 256 * 1024 && CBytes <= 32 * 1024;
    if constexpr (FullCache) {
        LocalA<dtype, M0, tK> a0[gK / tK];
        LocalA<dtype, M1, tK> a1[gK / tK];
        if constexpr (MATMUL_REUSE_A_N_GROUP > 1)
            cached_groups<0,dtype,gM,gN,gK,tN,tK,M0,M1>(
                cp,ap,bp,a0,a1,m,static_cast<int>(get_thread_idx()));
        else
            cached_pair_n<0, dtype, gM, gN, gK, tN, tK, M0, M1>(
                cp, ap, bp, a0, a1, m, static_cast<int>(get_thread_idx()),
                std::make_index_sequence<gK / tK>{});
    } else if constexpr (Count > 0) {
        static_assert(ABytes + Count * CBytes <= 256 * 1024);
        static_assert(SharedMatrixRight<dtype, tN, tK>::LogicalTileBytes <= 256 * 1024);
        run_m_pair_group<Begin, dtype, gM, gN, gK, tN, tK, M0, M1>(
            cp, ap, bp, m, std::make_index_sequence<Count>{});
        if constexpr (Remaining > Count)
            run_m_pair<dtype, gM, gN, gK, tN, tK, M0, M1, Begin + Count>(cp, ap, bp, m);
    } else {
        // Large tiles that cannot hold both strips retain the single-strip path.
        run_n_groups<dtype, gM, gN, gK, tN, tK, M0>(cp, ap, bp, m);
        run_n_groups<dtype, gM, gN, gK, tN, tK, M1>(cp, ap, bp, m + M0);
    }
}

// Retained for reuseB edge paths; reuseA itself groups N tails with full tiles.
template <typename dtype, int gM, int gN, int gK, int tN, int tK,
          int ValidM, int ValidN>
__attribute__((noinline)) void run_tail_block(float *c_ptr, dtype *a_ptr,
                                      dtype *b_ptr, int m_begin,
                                      int n_begin) {
    constexpr int kPeM = ValidM <= 64 ? 16 : 32;
    constexpr int Kb = gK / tK;
    constexpr int ComputeM = compute_group_m(ValidM);
    constexpr auto matmulOptions = fixp::keep_acc();
    using gmA = global_tensor<dtype, RowMajor<gM, gK>>;
    using gmB = global_tensor<dtype, RowMajor<gN, gK>>;
    using gmC = global_tensor<float, RowMajor<gM, gN>>;
    using tileAM16 = CubeTileM16<dtype, kPeM, tK>;
    using tileAM32 = CubeTileM32<dtype, kPeM, tK>;
    using tileA = std::conditional_t<(kPeM <= 16), tileAM16, tileAM32>;
    using tileBMatrix = SharedMatrixRight<dtype, tN, tK, ValidN, tK>;
    using tileB = SharedTile<tileBMatrix>;
    using tileCM16 = CubeAccumulatorM16<float, kPeM, tN, kPeM, ValidN>;
    using tileCM32 = CubeAccumulatorM32<float, kPeM, tN, kPeM, ValidN>;
    using tileC = std::conditional_t<(kPeM <= 16), tileCM16, tileCM32>;

    static_assert(ValidM >= 1 && ValidM <= 128);
    static_assert(ValidN >= 1 && ValidN <= tN);
    static_assert(tileBMatrix::LogicalTileBytes <= 256 * 1024,
                  "B tail tile exceeds the 256 KiB SharedTReg pool");
    static_assert(tileA::LogicalTileBytes + tileC::LogicalTileBytes <=
                      256 * 1024,
                  "A and tail C tile exceed CellReg capacity");

    const int tid = static_cast<int>(get_thread_idx());
    tileC tC;
    // Tail blocks follow the same cross-K local-A buffer reuse as full N.
    tileA tA;
#pragma clang loop unroll(disable)
    for (int k = 0; k < Kb; ++k) {
        // Every PE must publish a Local-A primary before the cooperative
        // TMATMUL. Inactive tail PEs load a safe dummy shard; groupM masks
        // their rows and no C store is issued for them.
        const int local_m = m_begin + local_m_offset<ValidM, kPeM>(tid);
        gmA gA(a_ptr + local_m * gK + k * tK);
        TLOAD_CUBE(tA, gA);
        tileB tB;
        gmB gB(b_ptr + n_begin * gK + k * tK);
        TLOAD<tileBMatrix, 1>(tB, gB);
        if constexpr (Kb == 1) {
            TMATMUL(tC, tA, tB, matmulOptions, ComputeM);
        } else if (k == 0) {
            TMATMUL(tC, tA, tB, matmulOptions.raw_acc(), ComputeM);
        } else if (k == Kb - 1) {
            TMATMUL_ACC(tC, tC, tA, tB, matmulOptions.acc_hint(),
                        ComputeM);
        } else {
            TMATMUL_ACC(tC, tC, tA, tB,
                        matmulOptions.raw_acc().acc_hint(), ComputeM);
        }
    }
    gmC gC(c_ptr + (m_begin + local_m_offset<ValidM, kPeM>(tid)) * gN + n_begin);
    store_c<ValidM>(gC, tC, tid);
}

template <typename dtype, int gM, int gN, int gK, int tN, int tK,
          int ValidM, int RemainingN>
PTO_SHARED_INLINE void run_n_tail(float *c_ptr, dtype *a_ptr, dtype *b_ptr,
                                  int m_begin, int n_begin) {
    constexpr int kChunkN = floor_power_of_two(RemainingN);
    run_tail_block<dtype, gM, gN, gK, tN, tK, ValidM, kChunkN>(
        c_ptr, a_ptr, b_ptr, m_begin, n_begin);
    if constexpr (RemainingN > kChunkN) {
        run_n_tail<dtype, gM, gN, gK, tN, tK, ValidM,
                   RemainingN - kChunkN>(
            c_ptr, a_ptr, b_ptr, m_begin, n_begin + kChunkN);
    }
}

template <typename dtype, int gM, int gN, int gK, int tN, int tK,
          int ValidM>
PTO_SHARED_INLINE void run_m_strip(float *c_ptr, dtype *a_ptr, dtype *b_ptr,
                                   int m_begin) {
    run_n_groups<dtype, gM, gN, gK, tN, tK, ValidM>(
        c_ptr, a_ptr, b_ptr, m_begin);
}

template <typename dtype, int gM, int gN, int gK, int tN, int tK,
          int RemainingM>
__attribute__((noinline)) void run_m_tail(float *c_ptr, dtype *a_ptr,
                                          dtype *b_ptr, int m_begin) {
    constexpr int ChunkM = RemainingM > 64 ? RemainingM : floor_power_of_two(RemainingM);
    run_m_strip<dtype, gM, gN, gK, tN, tK, ChunkM>(
        c_ptr, a_ptr, b_ptr, m_begin);
    if constexpr (RemainingM > ChunkM) {
        run_m_tail<dtype, gM, gN, gK, tN, tK, RemainingM - ChunkM>(
            c_ptr, a_ptr, b_ptr, m_begin + ChunkM);
    }
}

}  // namespace matmul_reuse_a_detail

// Four-PE cooperative matmul with PE-local A tile storage reused across K:
// each K iteration refills the same logical CellReg tile with a different A
// slice. Full N blocks and power-of-two N tail chunks reuse that A slice across up to
// four resident C accumulators, including M16 tails. M tails >64 stay one
// M128 compute group using overlapping A shards and disjoint masked stores.
// Adjacent M strips (including the first M-tail chunk) share each B load
// when CellReg capacity permits. Smaller tails retain power-of-two chunks.
template <typename dtype, int gM, int gN, int gK, int tM, int tN, int tK>
void matmul_reuseA(float *c_ptr, dtype *a_ptr, dtype *b_ptr) {
    constexpr int kPeNum = 4;
    constexpr int kGroupM = tM <= 128 ? tM : 128;
    static_assert(gK % tK == 0, "K must be divisible by tK");
    static_assert(tM % kPeNum == 0,
                  "tM must be divisible by the PE count");
    static_assert(tM % kGroupM == 0,
                  "tM must be divisible by group_M");
    static_assert((kGroupM & (kGroupM - 1)) == 0,
                  "group_M must be a power of two");
    static_assert((tN & (tN - 1)) == 0,
                  "tN must be a power of two");

    constexpr int kFullMBlocks = gM / kGroupM;
    constexpr int kTailM = gM % kGroupM;
    // Reserve the final full strip for pairing with the first tail chunk.
    constexpr int PairTail = kTailM != 0 && kFullMBlocks != 0;
    constexpr int RegularFull = kFullMBlocks - PairTail;
#pragma clang loop unroll(full)
    for (int i = 0; i < RegularFull / 2; ++i)
        matmul_reuse_a_detail::run_m_pair<
            dtype, gM, gN, gK, tN, tK, kGroupM, kGroupM>(
            c_ptr, a_ptr, b_ptr, 2 * i * kGroupM);
    if constexpr (RegularFull % 2 != 0)
        matmul_reuse_a_detail::run_m_strip<dtype, gM, gN, gK, tN, tK, kGroupM>(
            c_ptr, a_ptr, b_ptr, (RegularFull - 1) * kGroupM);
    if constexpr (PairTail) {
        constexpr int TailChunk = kTailM > 64 ? kTailM :
            matmul_reuse_a_detail::floor_power_of_two(kTailM);
        matmul_reuse_a_detail::run_m_pair<
            dtype, gM, gN, gK, tN, tK, kGroupM, TailChunk>(
            c_ptr, a_ptr, b_ptr, (kFullMBlocks - 1) * kGroupM);
        if constexpr (kTailM > TailChunk)
            matmul_reuse_a_detail::run_m_tail<
                dtype, gM, gN, gK, tN, tK, kTailM - TailChunk>(
                c_ptr, a_ptr, b_ptr, kFullMBlocks * kGroupM + TailChunk);
    } else if constexpr (kTailM != 0) {
        matmul_reuse_a_detail::run_m_tail<dtype, gM, gN, gK, tN, tK, kTailM>(
            c_ptr, a_ptr, b_ptr, 0);
    }
}
