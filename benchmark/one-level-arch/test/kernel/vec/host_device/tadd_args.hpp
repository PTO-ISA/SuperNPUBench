#pragma once

#include <stdint.h>

struct TaddArgs {
    uint64_t a, b, out;
    uint64_t geometry;
    uint64_t tiles;
};
static_assert(sizeof(TaddArgs) == 5 * sizeof(uint64_t));
constexpr unsigned kTileRows = 4;
constexpr unsigned kTileCols = 16;
constexpr unsigned kTileElements = kTileRows * kTileCols;
