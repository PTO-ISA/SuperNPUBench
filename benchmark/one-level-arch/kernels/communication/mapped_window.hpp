#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include "communication/comm_types.hpp"

namespace supernpu::comm {

// Payload operations require accessible, non-overlapping mapped ranges. Address
// registration/mapping and peer discovery belong to the caller. A mapping in a
// host process alone does not make the range accessible to a tile load engine.
// CPU protocol/reference path: Ops supplies kWidth, Tile, load/store/zero/scale/add.
// Device payload kernels call TileOP directly in tileop_comm.hpp.
template <class Ops>
inline bool valid_elements(std::size_t elements) {
    return elements != 0 && elements % Ops::kWidth == 0;
}

template <class Ops>
Status put(float* mapped_destination, const float* source, std::size_t elements) {
    if (!mapped_destination || !source || !valid_elements<Ops>(elements))
        return Status::invalid_argument;
    for (std::size_t i = 0; i < elements; i += Ops::kWidth) {
        typename Ops::Tile tile;
        Ops::load(tile, source + i);
        Ops::store(mapped_destination + i, tile);
    }
    return Status::ok;
}

template <class Ops>
Status get(float* destination, const float* mapped_source, std::size_t elements) {
    return put<Ops>(destination, mapped_source, elements);
}

// Each source occupies a distinct window. Reduce in source order, avoiding a
// dependence on remote floating-point atomic-add support or arrival order.
// weights == nullptr selects SUM; otherwise weights contains one scale/source.
template <class Ops>
inline __attribute__((always_inline)) Status weighted_combine(float* destination, const float* const* sources,
                        const float* weights, std::size_t source_count,
                        std::size_t elements) {
    if (!destination || !sources || source_count == 0 ||
        source_count > kMaxSources || !valid_elements<Ops>(elements))
        return Status::invalid_argument;
    for (std::size_t r = 0; r < source_count; ++r)
        if (!sources[r]) return Status::invalid_argument;
    for (std::size_t i = 0; i < elements; i += Ops::kWidth) {
        typename Ops::Tile acc;
        Ops::zero(acc);
        for (std::size_t r = 0; r < source_count; ++r) {
            typename Ops::Tile input, scaled;
            Ops::load(input, sources[r] + i);
            Ops::scale(scaled, input, weights ? weights[r] : 1.0f);
            Ops::add(acc, acc, scaled);
        }
        Ops::store(destination + i, acc);
    }
    return Status::ok;
}

template <class Ops>
Status reduce_sum(float* destination, const float* const* sources,
                  std::size_t source_count, std::size_t elements) {
    return weighted_combine<Ops>(destination, sources, nullptr, source_count, elements);
}

// One producer and one consumer per slot. Construct/zero before either starts.
// ready/consumed start at zero; both peers use epochs 1,2,... without wraparound.
// An array of slots supports ping-pong; each slot has its own epoch sequence.
template <std::size_t Capacity>
struct Window {
    static_assert(Capacity > 0, "A mapped window needs payload storage");
    alignas(64) std::uint64_t ready = 0;
    alignas(64) std::uint64_t consumed = 0;
    std::uint64_t elements = 0;
    alignas(64) float data[Capacity]{};
};

static_assert(sizeof(float) == 4, "Mapped payloads use FP32");
static_assert(std::is_standard_layout<Window<32>>::value &&
              std::is_trivially_copyable<Window<32>>::value,
              "Mapped windows must have a plain data representation");
static_assert(alignof(Window<32>) == 64, "Mapped window alignment is part of the ABI");
static_assert(offsetof(Window<32>, ready) == 0 && offsetof(Window<32>, consumed) == 64 &&
              offsetof(Window<32>, elements) == 72 && offsetof(Window<32>, data) == 128,
              "Mapped window fields must have fixed offsets");

// Sync is deliberately mandatory. There is no volatile/compiler-barrier
// fallback. Its contract covers both tile and scalar accesses:
//   wait_equal(word, epoch): bounded wait, acquire visibility on success;
//   publish(word, epoch): finish earlier payload reads/writes, then release
//                         publish at the scope containing both peers.
// A remote backend must also supply the required cache maintenance. Host CPU
// atomics cannot establish completion of asynchronous device tile operations.
template <class Ops, class Sync, std::size_t Capacity>
Status send(Window<Capacity>& remote, const float* source, std::size_t elements,
            std::uint64_t epoch, Sync& sync) {
    if (!source || epoch == 0 || !valid_elements<Ops>(elements) || elements > Capacity)
        return Status::invalid_argument;
    if (!sync.wait_equal(remote.ready, epoch - 1) ||
        !sync.wait_equal(remote.consumed, epoch - 1)) return Status::timeout;
    put<Ops>(remote.data, source, elements);
    remote.elements = elements;
    sync.publish(remote.ready, epoch);
    return Status::ok;
}

template <class Ops, class Sync, std::size_t Capacity>
Status receive(float* destination, Window<Capacity>& mapped_source,
               std::size_t elements, std::uint64_t epoch, Sync& sync) {
    if (!destination || epoch == 0 || !valid_elements<Ops>(elements) || elements > Capacity)
        return Status::invalid_argument;
    if (!sync.wait_equal(mapped_source.consumed, epoch - 1) ||
        !sync.wait_equal(mapped_source.ready, epoch)) return Status::timeout;
    if (mapped_source.elements != elements) return Status::invalid_argument;
    get<Ops>(destination, mapped_source.data, elements);
    sync.publish(mapped_source.consumed, epoch);
    return Status::ok;
}

// All contributors must be ready for this chunk, but no global barrier is used
// between chunks. Acknowledge only after the result and all source reads finish.
template <class Ops, class Sync, std::size_t Capacity>
Status receive_combine(float* destination, Window<Capacity>* const* windows,
                       const float* weights, std::size_t source_count,
                       std::size_t elements, std::uint64_t epoch, Sync& sync) {
    if (!destination || !windows || source_count == 0 || source_count > kMaxSources ||
        epoch == 0 || !valid_elements<Ops>(elements) || elements > Capacity)
        return Status::invalid_argument;
    for (std::size_t r = 0; r < source_count; ++r) {
        if (!windows[r]) return Status::invalid_argument;
        for (std::size_t previous = 0; previous < r; ++previous)
            if (windows[previous] == windows[r]) return Status::invalid_argument;
    }
    const float* sources[kMaxSources];
    for (std::size_t r = 0; r < source_count; ++r) {
        if (!sync.wait_equal(windows[r]->consumed, epoch - 1) ||
            !sync.wait_equal(windows[r]->ready, epoch)) return Status::timeout;
        if (windows[r]->elements != elements) return Status::invalid_argument;
        sources[r] = windows[r]->data;
    }
    weighted_combine<Ops>(destination, sources, weights, source_count, elements);
    for (std::size_t r = 0; r < source_count; ++r)
        sync.publish(windows[r]->consumed, epoch);
    return Status::ok;
}

}  // namespace supernpu::comm
