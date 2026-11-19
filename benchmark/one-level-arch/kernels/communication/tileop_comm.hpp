#pragma once

#include <common/pto_tileop.hpp>
#include "communication/comm_types.hpp"

namespace supernpu::comm::tileop {

constexpr std::size_t kTileElements = 32;
using F32Tile = pto::VecTileM32<float, 32, 1>;
using F32Global = pto::global_tensor<float, pto::RowMajor<32, 1>>;

// Direct TileOP kernels. The caller supplies accessible, non-overlapping mapped
// ranges and owns publication/completion. Status::ok means valid arguments and
// issued tile work; it does not establish remote visibility or acknowledge it.
// All ranges contain full FP32 tiles. A partial tail must be padded explicitly.
inline __attribute__((always_inline)) Status put(
    float* mapped_destination, const float* source, std::size_t elements) {
    if (!mapped_destination || !source || elements == 0 || elements % kTileElements != 0)
        return Status::invalid_argument;
    for (std::size_t i = 0; i < elements; i += kTileElements) {
        F32Global local(source + i);
        F32Global remote(mapped_destination + i);
        F32Tile data;
        TLOAD(data, local);
        TSTORE(remote, data);
    }
    return Status::ok;
}

inline __attribute__((always_inline)) Status get(
    float* destination, const float* mapped_source, std::size_t elements) {
    if (!destination || !mapped_source || elements == 0 || elements % kTileElements != 0)
        return Status::invalid_argument;
    for (std::size_t i = 0; i < elements; i += kTileElements) {
        F32Global remote(mapped_source + i);
        F32Global local(destination + i);
        F32Tile data;
        TLOAD(data, remote);
        TSTORE(local, data);
    }
    return Status::ok;
}

// Sum mapped source windows in source order into one destination. This is a
// gather-and-reduce payload, not remote atomic add or an AllReduce collective.
inline __attribute__((always_inline)) Status reduce_sum(
    float* destination, const float* const* sources, std::size_t source_count,
    std::size_t elements) {
    if (!destination || !sources || source_count == 0 || source_count > kMaxSources ||
        elements == 0 || elements % kTileElements != 0)
        return Status::invalid_argument;
    for (std::size_t r = 0; r < source_count; ++r)
        if (!sources[r]) return Status::invalid_argument;
    for (std::size_t i = 0; i < elements; i += kTileElements) {
        F32Tile accumulator;
        TEXPANDS(accumulator, 0.0f);
        for (std::size_t r = 0; r < source_count; ++r) {
            F32Global remote(sources[r] + i);
            F32Tile input;
            TLOAD(input, remote);
            TADD(accumulator, accumulator, input);
        }
        F32Global output(destination + i);
        TSTORE(output, accumulator);
    }
    return Status::ok;
}

// Fused mapped-source load -> weight -> sum -> output store. Each source must
// already be ready and remain immutable until its reads complete. The separate
// host protocol prototype does not supply device-side readiness to this kernel.
inline __attribute__((always_inline)) Status weighted_combine(
    float* destination, const float* const* sources, const float* weights,
    std::size_t source_count, std::size_t elements) {
    if (!destination || !sources || source_count == 0 || source_count > kMaxSources ||
        elements == 0 || elements % kTileElements != 0)
        return Status::invalid_argument;
    for (std::size_t r = 0; r < source_count; ++r)
        if (!sources[r]) return Status::invalid_argument;
    for (std::size_t i = 0; i < elements; i += kTileElements) {
        F32Tile accumulator;
        TEXPANDS(accumulator, 0.0f);
        for (std::size_t r = 0; r < source_count; ++r) {
            F32Global remote(sources[r] + i);
            F32Tile input, scaled;
            TLOAD(input, remote);
            TMULS(scaled, input, weights ? weights[r] : 1.0f);
            TADD(accumulator, accumulator, scaled);
        }
        F32Global output(destination + i);
        TSTORE(output, accumulator);
    }
    return Status::ok;
}

}  // namespace supernpu::comm::tileop
