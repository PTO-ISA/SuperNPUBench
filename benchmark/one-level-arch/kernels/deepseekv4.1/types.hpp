#pragma once

#include <stdint.h>

namespace deepseek_v41 {

// Single-request decode geometry. A group of 32 preserves the MX formats.
constexpr int Hidden = 32, HeadDim = 32, Heads = 2, Rank = 32, Groups = 2;
constexpr int OutRank = 32, Intermediate = 64, IndexHeads = 2, IndexDim = 32;
constexpr int Streams = 4, Mixes = 2 * Streams + Streams * Streams;
constexpr int Experts = 16, TopK = 6, Window = 4, History = 32;
constexpr int Page = 8, SparseK = 4, PoolBlocks = 2, RopeDim = 8;
constexpr int MxGroup = 32, CompGroup = 16, Sources = 4, IndexSources = 8;
constexpr float Eps = 1.e-6f;

enum class Kind { Swa, C2Full, C2Reuse, C1Full, C1Reindex, C1Reuse };

struct Schedule {
    Kind kind;
    int kv_bank, index_bank;
    bool engram;

    static Schedule for_layer(int layer) { // checked by decode_layer
        if (layer < 2)
            return {Kind::Swa, -1, -1, layer == 1};
        if (layer < 20) {
            int bank = (layer - 2) / 6;
            return {(layer - 2) % 6 == 0 ? Kind::C2Full : Kind::C2Reuse, bank, bank, layer == 14};
        }
        int bank = 3 + (layer - 20) / 4;
        Kind kind = layer == 20 ? Kind::C1Full : (layer % 4 == 0 ? Kind::C1Reindex : Kind::C1Reuse);
        return {kind, 3, bank, false};
    }
};

template <int Out, int In, int Bits> struct MxWeight {
    static_assert(In % MxGroup == 0 && (Bits == 4 || Bits == 8));
    uint8_t data[Out][In * Bits / 8];
    uint8_t scale[Out][In / MxGroup]; // UE8M0, output-major
};

template <int Bits> struct Expert {
    MxWeight<Intermediate, Hidden, Bits> gate, up;
    MxWeight<Hidden, Intermediate, Bits> down;
};

struct HCWeights {
    float fn[Mixes][Streams * Hidden], scale[3], base[Mixes], norm[Hidden];
};

struct Weights {
    HCWeights attn, ffn;
    MxWeight<Rank, Hidden, 8> qa;
    MxWeight<Heads * HeadDim, Rank, 8> qb;
    MxWeight<HeadDim, Hidden, 8> kv;
    MxWeight<OutRank, Heads * HeadDim / Groups, 8> oa[Groups];
    MxWeight<Hidden, Groups * OutRank, 8> ob;
    float q_norm[Rank], kv_norm[HeadDim], sink[Heads];
    float compressor[HeadDim][Hidden], compressor_gate[HeadDim][Hidden], comp_norm[HeadDim];
    float ik[IndexDim][HeadDim], ik_norm[IndexDim], iw[IndexHeads][Hidden];
    MxWeight<IndexHeads * IndexDim, Rank, 8> iq;
    MxWeight<(Streams + 1) * Hidden, Hidden, 8> engram_kv;
    float engram_weight[Streams][Hidden], router[Experts][Hidden], bias[Experts];
    Expert<8> shared;
    Expert<4> routed[Experts];
};

struct HiddenState {
    float x[Streams][Hidden], pre[Streams];
};
struct Rotary {
    float cosine[RopeDim / 2], sine[RopeDim / 2];
};
struct Token {
    HiddenState hidden;
    float lookup[Hidden]; // flattened, host-looked-up Engram row
    Rotary rope, compressed_rope;
    int position, window_slot, window_indices[Window];
};
struct WindowRow {
    uint8_t data[HeadDim];
    uint16_t scale[HeadDim / MxGroup]; // BF16 powers of two
};
struct CompressedRow {
    uint8_t data[HeadDim / 2];
    uint16_t scale[HeadDim / CompGroup];
};
struct IndexRow {
    uint8_t data[IndexDim / 2], scale[IndexDim / MxGroup];
};
struct CacheBank {
    CompressedRow kv[History];
    IndexRow index[History];
};
struct LayerCache {
    WindowRow window[Window];
    float pair_kv[2][HeadDim], pair_gate[2][HeadDim]; // layer-local C2A ring
    CacheBank banks[Sources];
    int pages[History / Page]; // permutation shared by KV and index

    bool valid_pages() const {
        bool seen[History / Page]{};
        for (int page : pages) {
            if (page < 0 || page >= History / Page || seen[page])
                return false;
            seen[page] = true;
        }
        return true;
    }
};
struct Selection {
    int rows[SparseK]; // physical slots; -1 denotes padding
};
struct Candidates {
    int blocks[PoolBlocks]; // sorted logical block IDs; -1 denotes padding
};
struct Published {
    Selection indices[IndexSources];
    Candidates candidates; // layer 20 per-query pool
};
struct Route {
    int expert, slot;
    float weight;
};
struct Contribution {
    float value[Hidden];
};
struct LayerOutput {
    HiddenState attention, next;
    Selection selected;
    Candidates candidates;
    Route routes[TopK];
    Contribution shared, routed[TopK];
};

} // namespace deepseek_v41
