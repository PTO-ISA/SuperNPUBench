#include <common/pto_tileop.hpp>
#include <cstdint>
#include "single_thread/deepseek/engram/engram_hash.hpp"
using namespace pto;
using namespace supernpu::tile_isa;

static std::int32_t ngram_token_ids[256] __attribute__((aligned(4096))) = {};
static std::int64_t multipliers[16] __attribute__((aligned(4096))) = {};
static std::int32_t vocab_sizes[64] __attribute__((aligned(4096))) = {};
static std::int32_t offsets[64] __attribute__((aligned(4096))) = {};
static std::int32_t output[2048] __attribute__((aligned(4096))) = {};

int main() {
    engram_hash_layer<16, 8, 8, 8>(ngram_token_ids, multipliers, vocab_sizes, offsets, output);
    return 0;
}
