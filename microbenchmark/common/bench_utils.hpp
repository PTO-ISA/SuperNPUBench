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
__attribute__((noinline, optnone)) void fill_const(T *p, int n, T v) {
    // optnone prevents the freestanding target build from lowering this loop
    // to an unresolved hosted-library memset call.
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
        if (g == r) continue;
        if (g != g || r != r) return false;
        const double abs_err = g > r ? g - r : r - g;
        const double abs_ref = r < 0 ? -r : r;
        const double limit = (double)eps + (double)rel_eps * abs_ref;
        if (abs_err > limit) return false;
    }
    return true;
}

template <typename T>
constexpr double verify_epsilon() {
    return sizeof(T) <= 2 ? 2e-2 : 1e-4;
}

template <typename T>
bool verify_scalar(T got, T ref, double eps = verify_epsilon<T>()) {
    const double g = (double)got;
    const double r = (double)ref;
    if (g == r) return true;
    if (g != g || r != r) return false;
    const double abs_err = g > r ? g - r : r - g;
    const double abs_ref = r < 0 ? -r : r;
    return abs_err <= eps + eps * abs_ref;
}

template <typename T>
bool verify_zero(const T *got, int n) {
    for (int i = 0; i < n; ++i)
        if ((double)got[i] != 0.0) return false;
    return true;
}

#endif
