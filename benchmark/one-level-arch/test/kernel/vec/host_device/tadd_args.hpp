#pragma once

#include <stdint.h>

constexpr uint32_t kTaddKernelSchema = 0x54414444u;  // "TADD"
constexpr unsigned kTileRows = 4;
constexpr unsigned kTileCols = 16;
constexpr unsigned kTileElements = kTileRows * kTileCols;
