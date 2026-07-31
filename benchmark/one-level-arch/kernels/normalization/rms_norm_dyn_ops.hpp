// Shared helpers for rms_norm* when Tile ValidRow/Col are DYNAMIC (-1).
// TCOPYIN/OUT already honor GetValid* via blk_tload/tstore.
// TEPL B.DIM only accepts immediates, so compute uses jcore *_Impl / __vec__.
#ifndef SUPERNPU_RMS_NORM_DYN_OPS_HPP
#define SUPERNPU_RMS_NORM_DYN_OPS_HPP

#include <common/pto_tileop.hpp>

#include <cstdint>

namespace rms_dyn {

inline int64_t min64(int64_t a, int64_t b) { return a < b ? a : b; }

template <typename TileOut, typename TileIn0, typename TileIn1>
void __vec__ TRowExpandMul_Vec_RowMajor(
    typename TileOut::TileDType __out__ dst,
    const typename TileIn0::TileDType __in__ src0,
    const typename TileIn1::TileDType __in__ src1) {
    size_t i = blkv_get_index_x();
    size_t j = blkv_get_index_y();
    size_t idx = j * TileIn0::RowStride + i;
    size_t bidx = j * TileIn1::RowStride;
    blkv_get_tile_ptr(dst)[idx] =
        blkv_get_tile_ptr(src0)[idx] * blkv_get_tile_ptr(src1)[bidx];
}

template <typename TileOut, typename TileIn>
inline void tcvt_rm(TileOut &dst, TileIn &src) {
    static_assert(TileIn::isRowMajor && TileOut::isRowMajor,
                  "tcvt_rm expects RowMajor tiles");
    const size_t row = static_cast<size_t>(src.GetValidRow());
    const size_t col = static_cast<size_t>(src.GetValidCol());
    TCast_RowMajor_Imp<TileOut, TileIn>
        <<<col, row, 1>>>(dst.data(), src.data());
}

template <typename Tile>
inline void tmul(Tile &dst, Tile &src0, Tile &src1) {
    TMUL_Impl(dst, src0, src1);
}

template <typename Tile>
inline void tmuls(Tile &dst, Tile &src, typename Tile::DType s) {
    TMULS_Impl(dst, src, s);
}

template <typename Tile>
inline void tadd(Tile &dst, Tile &src0, Tile &src1) {
    TADD_Impl(dst, src0, src1);
}

template <typename Tile>
inline void tadds(Tile &dst, Tile &src, typename Tile::DType s) {
    const size_t row = static_cast<size_t>(src.GetValidRow());
    const size_t col = static_cast<size_t>(src.GetValidCol());
    if constexpr (Tile::isRowMajor) {
        TAdds_Vec_RowMajor<Tile><<<col, row, 1>>>(dst.data(), src.data(), s);
    } else {
        TAdds_Vec_ColMajor<Tile><<<row, col, 1>>>(dst.data(), src.data(), s);
    }
}

template <typename Tile>
inline void trecip(Tile &dst, Tile &src) {
    TRECIP_Impl(dst, src);
}

template <typename TileOut, typename TileIn>
inline void trowsum(TileOut &dst, TileIn &src) {
    TROWSUM_Impl(dst, src);
}

template <typename TileOut, typename TileIn0, typename TileIn1>
inline void trowexpandmul(TileOut &dst, TileIn0 &src0, TileIn1 &src1) {
    const size_t row = static_cast<size_t>(src0.GetValidRow());
    const size_t col = static_cast<size_t>(src0.GetValidCol());
    TRowExpandMul_Vec_RowMajor<TileOut, TileIn0, TileIn1>
        <<<col, row, 1>>>(dst.data(), src0.data(), src1.data());
}

template <typename TileVec>
inline void rsqrt_newton(TileVec &out, TileVec &a) {
    TileVec x(static_cast<size_t>(a.GetValidRow()));
    TileVec t1(static_cast<size_t>(a.GetValidRow()));
    TileVec t2(static_cast<size_t>(a.GetValidRow()));
    trecip(x, a);
    for (int64_t i = 0; i < 4; ++i) {
        tmul(t1, x, x);
        tmul(t2, t1, a);
        tmuls(t2, t2, -0.5f);
        tadds(t2, t2, 1.5f);
        tmul(x, x, t2);
    }
    tmuls(out, x, 1.0f);
}

} // namespace rms_dyn

#endif
