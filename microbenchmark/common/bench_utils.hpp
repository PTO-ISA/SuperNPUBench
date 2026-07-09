#ifndef BENCH_UTILS_HPP
#define BENCH_UTILS_HPP

#include <cstdint>
#include <cmath>

template <typename T>
void fill_seq(T *p, int n, T base = (T)0) {
    for (int i = 0; i < n; ++i) {
        p[i] = (T)(base + (T)i * (T)0.1);
    }
}

template <typename T>
void fill_const(T *p, int n, T v) {
    for (int i = 0; i < n; ++i) p[i] = v;
}

template <typename T>
void zero(T *p, int n) { fill_const(p, n, (T)0); }

template <typename T>
void fill_idx(T *p, int n, T base = (T)0) {
    for (int i = 0; i < n; ++i) p[i] = (T)((i * 7) % n) + base;
}

template <typename T>
bool verify(const T *got, const T *ref, int n, T eps = (T)1e-3) {
    for (int i = 0; i < n; ++i) {
        if (std::fabs((double)(got[i] - ref[i])) > (double)eps) return false;
    }
    return true;
}

#endif
