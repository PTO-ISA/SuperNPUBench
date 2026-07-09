#ifndef SCALAR_BENCH_HPP
#define SCALAR_BENCH_HPP

// Scalar ALU micro-bench templates (plain C + volatile, no inline asm).
// Measures ordinary GPR scalar execution unit throughput / latency, distinct
// from the tile families (CUBE/TEPL/TLSU). Intrinsic naming is N/A here: the
// compiler emits scalar micro-ISA (misa_g/l/f) from plain C; volatile sinks
// prevent dead-code elimination.
//
// Metrics:
//   latency     : chain dependency acc = op(acc, b[i])   -> cycles / N
//   throughput  : 8 independent accumulators            -> cycles / (8*N)
//
// Per-iter b index is masked to a 16-entry window so inputs stay in registers.

#include <cstdint>
#include <cmath>
#include "benchmark.h"

constexpr int SCALAR_N = 1024;    // loop trip count
constexpr int SCALAR_MASK = 15;   // b index window (16 entries)

// latency: serial dependency chain
template <typename T, int N = SCALAR_N>
inline T bench_latency(const T *a, const T *b, auto op) {
    T acc = a[0];
    for (int i = 0; i < N; ++i) {
        acc = op(acc, b[i & SCALAR_MASK]);
    }
    return acc;
}

// throughput: 8 independent chains to expose ILP
template <typename T, int N = SCALAR_N>
inline T bench_throughput(const T *a, const T *b, auto op) {
    T a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
    T a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
    for (int i = 0; i < N; ++i) {
        a0 = op(a0, b[(i * 8 + 0) & SCALAR_MASK]);
        a1 = op(a1, b[(i * 8 + 1) & SCALAR_MASK]);
        a2 = op(a2, b[(i * 8 + 2) & SCALAR_MASK]);
        a3 = op(a3, b[(i * 8 + 3) & SCALAR_MASK]);
        a4 = op(a4, b[(i * 8 + 4) & SCALAR_MASK]);
        a5 = op(a5, b[(i * 8 + 5) & SCALAR_MASK]);
        a6 = op(a6, b[(i * 8 + 6) & SCALAR_MASK]);
        a7 = op(a7, b[(i * 8 + 7) & SCALAR_MASK]);
    }
    return (T)(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7);
}

// store throughput: stream val to out[i]
template <typename T, int N = SCALAR_N>
inline void bench_store(T *out, T val) {
    for (int i = 0; i < N; ++i) {
        out[i & SCALAR_MASK] = val;
    }
}

// conversion throughput: IN -> OUT, 8 independent conversions
template <typename IN, typename OUT, int N = SCALAR_N>
inline OUT bench_cv(const IN *b) {
    OUT a0 = (OUT)b[0], a1 = (OUT)b[1], a2 = (OUT)b[2], a3 = (OUT)b[3];
    OUT a4 = (OUT)b[4], a5 = (OUT)b[5], a6 = (OUT)b[6], a7 = (OUT)b[7];
    for (int i = 0; i < N; ++i) {
        a0 = (OUT)b[(i * 8 + 0) & SCALAR_MASK];
        a1 = (OUT)b[(i * 8 + 1) & SCALAR_MASK];
        a2 = (OUT)b[(i * 8 + 2) & SCALAR_MASK];
        a3 = (OUT)b[(i * 8 + 3) & SCALAR_MASK];
        a4 = (OUT)b[(i * 8 + 4) & SCALAR_MASK];
        a5 = (OUT)b[(i * 8 + 5) & SCALAR_MASK];
        a6 = (OUT)b[(i * 8 + 6) & SCALAR_MASK];
        a7 = (OUT)b[(i * 8 + 7) & SCALAR_MASK];
    }
    return (OUT)(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7);
}

#endif
