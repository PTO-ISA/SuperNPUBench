#pragma once

#include <common/pto_tileop.hpp>

namespace deepseek_v41::tile {

// Keep the same physical geometry through FP32/BF16/FP8 conversions. In
// particular even a single active row must fit the 128-byte minimum FP8 tile.
template <int R, int C, class T = float>
using Block = pto::Tile<pto::Location::Vec, T, (R * C < 128 ? (128 + C - 1) / C : R), C,
                        pto::BLayout::RowMajor, R, C>;

template <int R, int C, class T = float>
using ActiveRows = pto::Tile<pto::Location::Vec, T, Block<R, C>::Rows, C,
                             pto::BLayout::RowMajor, pto::DYNAMIC, C>;

template <class T, int R, int C, int Stride = C>
using GM = pto::global_tensor<T, pto::RowMajor<R, C, Stride>>;

template <class B>
[[gnu::always_inline]] inline void load(B &dst, const typename B::DType *src) {
    GM<typename B::DType, B::Rows, B::ValidCol> gm(src);
    TLOAD(dst, gm);
}

template <class B>
[[gnu::always_inline]] inline void store(typename B::DType *dst, B &src) {
    GM<typename B::DType, B::Rows, B::ValidCol> gm(dst);
    TSTORE(gm, src);
}

template <class B> [[gnu::always_inline]] inline void round_bf16(B &value) {
    using Half = pto::Tile<pto::Location::Vec, __bf16, B::Rows, B::Cols, pto::BLayout::RowMajor,
                           B::ValidRow, B::ValidCol>;
    if constexpr (B::ValidRow == pto::DYNAMIC) {
        Half rounded(value.GetValidRow());
        TCVT<LINX_RNE>(rounded, value);
        TCVT(value, rounded);
    } else {
        Half rounded;
        TCVT<LINX_RNE>(rounded, value);
        TCVT(value, rounded);
    }
}

template <class B> [[gnu::always_inline]] inline void sigmoid(B &result, B &x) {
    TNEG(result, x);
    TEXP(result, result);
    TADDS(result, result, 1.f);
    TRECIP(result, result);
}

} // namespace deepseek_v41::tile
