# Mapped-window communication operators

These operators express communication through caller-supplied mapped addresses.
The first implementation contains FP32 Put/Get, source-ordered ReduceSum and
weighted combine, plus a per-chunk ready/ack protocol. A receiver can combine a
completed chunk while producers prepare later chunks, without a whole-tensor
barrier.

**Support boundary:** the POSIX test exercises real shared mappings between CPU
processes. The PTO test executes the payload operations on one simulated PE.
There is no Linx/NPU cross-device synchronization backend in this change, and
neither test measures network bandwidth, RTT or device communication overlap.

## Operators

Include `communication/tileop_comm.hpp` for the directly implemented PTO kernels
in `supernpu::comm::tileop`. Their bodies contain the TileOP calls themselves;
there is no `PtoPayload` or `Ops::load/store` forwarding layer on this path.

| Entry point | Operation |
| --- | --- |
| `tileop::put(mapped_dst, src, elements)` | TLOAD local input, TSTORE mapped peer destination |
| `tileop::get(dst, mapped_src, elements)` | TLOAD mapped peer source, TSTORE local output |
| `tileop::reduce_sum(dst, sources, source_count, elements)` | TLOAD each source, TADD into accumulator, TSTORE result |
| `tileop::weighted_combine(dst, sources, weights, source_count, elements)` | TLOAD each source, TMULS by its weight, TADD, TSTORE result |

The weighted-combine tile loop is explicit in the kernel:

```cpp
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
```

The separate `communication/mapped_window.hpp` contains the generic CPU protocol
prototype tested with `CpuPayload`. It is not connected to the device kernels:

| Host protocol entry point | Operation |
| --- | --- |
| `send<Ops>(window, src, elements, epoch, sync)` | Wait for slot credit, copy payload and publish ready |
| `receive<Ops>(dst, window, elements, epoch, sync)` | Wait for ready, copy payload and acknowledge consumption |
| `receive_combine<Ops>(dst, windows, weights, source_count, elements, epoch, sync)` | Wait for this chunk from every source, combine it and acknowledge each slot |

There is no remote floating-point atomic
reduction: each producer owns a distinct source window and reduction follows
source order. `weights == nullptr` selects unit weights. ReduceSum has a single
destination; it is not an AllReduce or ReduceScatter collective.

The first PTO profile uses full 32-element FP32 M32 tiles. `elements` must be
positive and divisible by 32. Explicitly pad a partial final tile before
publishing it; the implementation rejects unsupported tails instead of reading
past a caller's buffer. The source count is 1..32. Callers supply sufficiently
large, non-overlapping ranges; destination storage must not alias inputs,
weights, pointer tables or protocol metadata. Source ranges remain immutable
until the corresponding reads have completed.

Payload functions report argument acceptance with `Status::ok`, **not remote
completion**. Their publication and completion requirements belong to the
backend. Only `send`/`receive`/`receive_combine` invoke the synchronization
contract below. The API does not register memory, create an MMAP, discover
peers, establish fabric connectivity or acquire device access permissions.

## Window protocol and backend contract

`Window<Capacity>` has fixed-width ready/consumed epochs and payload length,
64-byte alignment, and an FP32 payload at byte offset 128. Participating peers
must agree on capacity, layout, endianness and element representation. Mapping
addresses may differ between peers; the structure contains no pointers.

Each slot has exactly one producer and one consumer. Initialize it to zero
before participants start. For each slot, use epochs 1, 2, 3, ... without
wraparound; a two-slot ping-pong array has **two independent epoch sequences**.
Do not reset a live window.

1. The producer waits for both `ready == epoch-1` and `consumed == epoch-1`.
2. It writes the payload and its length, then publishes `ready = epoch`.
3. The consumer acquires that epoch, checks length and reads/computes the chunk.
4. After those accesses complete, it publishes `consumed = epoch`.

Checking both epochs prevents a duplicate send from overwriting an unconsumed
slot and prevents a duplicate receive from rereading a slot already released
to its producer. Combine waits for all contributors to the current chunk, but
never introduces a global barrier between chunks.

The required `Sync` interface is:

```cpp
bool wait_equal(uint64_t& word, uint64_t expected);
void publish(uint64_t& word, uint64_t epoch);
```

- `wait_equal` must be bounded and return true only after an acquire at the
  scope containing both peers. It must make subsequent payload reads observe
  the producer's writes, including any required device-cache maintenance.
- `publish` must finish earlier payload reads and writes **before** release
  publication. In particular, `consumed` cannot become visible while a tile
  load is still reading the slot. Payload lifetime may end when `send` returns
  only because this completion contract has been satisfied.
- All concurrent signal accesses must be atomic. The CPU test uses native
  lock-free `std::atomic_ref<uint64_t>` acquire/release operations over shared
  anonymous MMAP memory. This test backend requires C++20 and POSIX fork/MMAP.
  It must not be paired with asynchronous device payload operations.

No default `volatile`, empty assembly barrier or scalar atomic substitute is
provided for device completion. A device protocol implementation must supply
the missing tile/scalar/fabric ordering guarantees before publishing or
consuming results from `tileop::put/get/reduce_sum/weighted_combine`.

Invalid arguments, including unsupported shapes and duplicate combine
windows, return `invalid_argument`. A failed wait returns `timeout` without
publishing ready/ack or consuming the output buffer. A published length
mismatch is detected after acquire and is not acknowledged. On error, stop
the participating ranks and quiesce outstanding device work before unmapping,
resetting or reusing windows; this API does not implement distributed recovery.

## Build and verify

From the repository root:

```sh
cd benchmark/one-level-arch/test/solution/communication
make host-check
make host-sanitize

python3 src/run_payload_check.py \
  --compiler-dir /path/to/linx_blockisa_llvm_musl/bin \
  --gfrun /path/to/compatible/gfrun
```

The host check uses two processes for Put/Get and four for combine/reduction.
It covers delayed arrivals, backpressure, two-slot reuse, 32/64/96-element
messages, untouched output tails, stale/duplicate epochs, length disagreement,
invalid arguments and bounded timeout/child cleanup. Golden arithmetic is
independent of the payload implementation. ASan/UBSan covers the host code;
it is not a proof of a remote device's memory ordering.

The PTO checker builds an ELF and checks the disassembly for decoded TLOAD,
TSTORE, TEXPANDS, TMULS and TADD instructions, then compares all output bytes
for four operations at three sizes with independent host-generated FP32
goldens. It also checks untouched tails, status exports and the simulator
finisher. Missing tools, output files, malformed output, nonzero status and
timeouts fail the check. Outputs, logs, source/artifact hashes and `report.json`
are written under `benchmark/one-level-arch/compare/solution_communication_comm_payload/`.
The instruction counts are static disassembly counts, not runtime performance
counters. Artifacts are ignored by Git. Inspect `worktree_status` before attributing a report
to its recorded revision.

The initial payload run passed with clang 15.0.4
`1f3f6a3fb2f24da3734bfed41c8032f901202bc1` (Linx musl toolchain) and the
`pto-element-functional-model` gfrun checkout at
`0a237ccfe8404d73d92dd144a6af539b2b0ebc5a`. This is focused functional
evidence, not a full toolchain/model release gate. An older local gfrun failed
before running the payload; selecting an arbitrary installed binary is not
sufficient compatibility evidence.

## CANN reference points

The implementation is original and does not embed Ascend SQE layouts. Useful
reference mechanisms are the synchronous
[TPUT load/store path](https://gitcode.com/cann/pto-isa/blob/9f7bb9cbc49fb577c9835c3eac936f9442f30d9d/include/pto/comm/a2a3/TPut.hpp#L259),
[notify publication](https://gitcode.com/cann/pto-isa/blob/9f7bb9cbc49fb577c9835c3eac936f9442f30d9d/include/pto/comm/a2a3/TNotify.hpp#L38)
and [wait/cache handling](https://gitcode.com/cann/pto-isa/blob/9f7bb9cbc49fb577c9835c3eac936f9442f30d9d/include/pto/comm/a2a3/TWait.hpp#L51).
Those implementations do not establish that a Linx mapped address or atomic
operation has the same completion or visibility semantics.
