#ifndef _INCLUDE_ACCELERATOR_TYPE_H_
#define _INCLUDE_ACCELERATOR_TYPE_H_

enum class AS {
    GM,
    UB,
    L1,
    L0A,
    L0B,
    L0C,
    BIAS,
    SCALING,
};

template<typename AS, typename Element_>
struct Addr {
    using Type = Element_;
};

template<typename Element_>
struct Addr<GM, Element> {
    using Type = __gm__ Element_;
};

template<typename Element_>
struct Addr<UB, Element> {
    using Type = __ubuf__ Element_;
};

template<typename Element_>
struct Addr<L1, Element> {
    using Type = __cbuf__ Element_;
};

template<typename Element_>
struct Addr<L0A, Element> {
    using Type = __ca__ Element_;
};

template<typename Element_>
struct Addr<L0B, Element> {
    using Type = __cb__ Element_;
};

template<typename Element_>
struct Addr<L0C, Element> {
    using Type = __cc__ Element_;
};

template<typename Element_>
struct Addr<BIAS, Element> {
    using Type = __ubuf__ Element_;
};

template<typename Element_>
struct Addr<SCALING, Element> {
    using Type = __ubuf__ Element_;
};

enum class TileLayoutKind {
    kNoneBox,
    kRowMajor,
    kColMajor,
};

enum class FractalLayoutKind {
    kNoneBox,
    kRowMajor,
    kColMajor,
};

constexpr uint32_t BLOCK_BYTE_SIZE = 32;

#endif