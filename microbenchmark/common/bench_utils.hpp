#ifndef BENCH_UTILS_HPP
#define BENCH_UTILS_HPP

#include <cstdint>
#include <cmath>

template <typename T>
void fill_seq(T *p, int n, T base = (T)0) {
    for (int i = 0; i < n; ++i) {
        p[i] = (T)((float)base + (float)i * 0.1f);
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
    for (int i = 0; i < n; ++i) p[i] = (T)((float)((i * 7) % n) + (float)base);
}

template <typename T>
bool verify(const T *got, const T *ref, int n, T eps = (T)1e-3,
            T rel_eps = (T)1e-3) {
    for (int i = 0; i < n; ++i) {
        const double g = (double)got[i];
        const double r = (double)ref[i];
        if (!std::isfinite(g) || !std::isfinite(r)) {
            if (g != r) return false;
            continue;
        }
        const double abs_err = std::fabs(g - r);
        const double limit = (double)eps + (double)rel_eps * std::fabs(r);
        if (abs_err > limit) return false;
    }
    return true;
}

#endif
