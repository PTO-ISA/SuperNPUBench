// =============================================================================
// engram_hash_pto.hpp — n-gram 哈希嵌入索引（PTO 一层 tile 版）
// =============================================================================
//
// 【功能】
//   对每 token 算 n-gram 哈希：hash = XOR_{n}(id[n] * mult[n])；
//   对每个 ngram_idx>0 的每个嵌入表 j：
//     out[col] = (hash % vocab[n-1, j]) + offset[col]
//   哈希累加跨 max_ngram_size 步是每 token 串行，但每步运算可跨 token tile 并行。
//
// 【源端】TileKernels/tile_kernels/engram/engram_hash_kernel.py
//
// 【迁移映射】
//   id*mult（标量乘 int32）→ TMULS
//   哈希异或累加          → TXOR
//   hash % vocab           → TREMS（tile-标量取余）
//   + offset              → TADDS
//   取列 n 的 id           → global_iterator(tn, n) 取 (TileN×8 valid1) 子块（每 token 一个 id）
//
// 【约束】MaxNgramSize/TileN/num_out_cols 须 8 的倍数（int32 32B 对齐）；int32 hash
//   （真实 int64 需 4 宽 tile，此处编译可用 int32）。
//
// 【算法步骤（每 tile tn）】
//   hash=0; for n in 0..MaxNgramSize-1:
//     1) TLOAD id 列 n（TileN×8 valid1）
//     2) TMULS(prod, id, mult[n])
//     3) TXOR(hash, hash, prod)
//     4) if n>0: for j: TREMS(mod, hash, vocab); TADDS(out, mod, offset); TSTORE out 列 col
// =============================================================================
#ifndef SUPERNPU_ENGRAM_HASH_PTO_HPP
#define SUPERNPU_ENGRAM_HASH_PTO_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {

template <int NumTokens, int MaxNgramSize, int NumEmbedPerNgram, int TileN = 8>
void engram_hash_layer(std::int32_t *ngram_token_ids,
                       const std::int64_t *multipliers,
                       const std::int32_t *vocab_sizes,
                       const std::int32_t *offsets,
                       std::int32_t *output) {
    static_assert(NumTokens > 0 && MaxNgramSize > 0 && NumEmbedPerNgram > 0, "dim must be positive");
    static_assert(MaxNgramSize % 8 == 0, "MaxNgramSize must be multiple of 8");
    static_assert(TileN % 8 == 0, "TileN must be multiple of 8");
    constexpr int kNumOutCols = (MaxNgramSize - 1) * NumEmbedPerNgram;
    static_assert(kNumOutCols % 8 == 0, "num_out_cols must be multiple of 8");
    constexpr int kTN = NumTokens / TileN;
    using namespace pto;
    using gm_id  = global_tensor<std::int32_t, RowMajor<NumTokens, MaxNgramSize>>;
    using gm_out = global_tensor<std::int32_t, RowMajor<NumTokens, kNumOutCols>>;
    using tile_v = Tile<Location::Vec, std::int32_t, TileN, 8, BLayout::RowMajor, TileN, 1>; // 每 token 一个值
    using it_id  = global_iterator<gm_id,  tile_v>;
    using it_out = global_iterator<gm_out, tile_v>;
    it_id id_iter(ngram_token_ids); it_out out_iter(output);

    for (int tn = 0; tn < kTN; ++tn) {
        tile_v hash, idn, prod, mod, outc;
        TEXPANDS(hash, 0);                              // hash 初值 0（每 token 一个）
        for (int n = 0; n < MaxNgramSize; ++n) {        // ngram 步串行（状态依赖上步）
            auto gid = id_iter(tn, n);                  // 取列 n 的 id（每 token 一个 id）
            TLOAD(idn, gid);                             // 1) id[n]
            TMULS(prod, idn, static_cast<std::int32_t>(multipliers[n])); // 2) id*mult
            TXOR(hash, hash, prod);                     // 3) hash ^= prod
            if (n > 0) {
                for (int j = 0; j < NumEmbedPerNgram; ++j) {
                    int col = (n - 1) * NumEmbedPerNgram + j;
                    std::int32_t vocab = vocab_sizes[(n - 1) * NumEmbedPerNgram + j];
                    std::int32_t off = offsets[col];
                    TREMS(mod, hash, vocab);            // 4a) hash % vocab
                    TADDS(outc, mod, off);              // 4b) + offset
                    auto gout = out_iter(tn, col);
                    TSTORE(gout, outc);                 //    写输出列 col
                }
            }
        }
    }
}

} // namespace supernpu::tile_isa
#endif
