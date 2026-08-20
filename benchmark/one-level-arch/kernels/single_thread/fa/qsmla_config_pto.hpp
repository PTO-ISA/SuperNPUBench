#ifndef QSMLA_CONFIG_PTO_HPP
#define QSMLA_CONFIG_PTO_HPP

#include <cstddef>

struct QsmlaWorkItem {
    int batch;
    int q_token;
    int kv_head;
    int g_slice;
    int q_head_begin;
    int m_real;
};

template <int B_, int S1_, int S2_, int N1_, int N2_, int D_, int K_,
          int Tm_, int Tk_, int Td_, int GSliceMax_ = 64>
struct QsmlaConfig {
    static_assert(B_ > 0 && S1_ > 0 && S2_ > 0, "B/S1/S2 must be positive");
    static_assert(N1_ > 0 && N2_ > 0 && N1_ % N2_ == 0,
                  "N1 must be positive and divisible by N2");
    static_assert(D_ > 0 && K_ >= 0, "D must be positive and K non-negative");
    static_assert(Tm_ > 0 && Tk_ > 0 && Td_ > 0, "tile sizes must be positive");
    static_assert(GSliceMax_ > 0 && GSliceMax_ <= 64,
                  "gSlice must be positive and no larger than the MM1 M block");

    static constexpr int B = B_;
    static constexpr int S1 = S1_;
    static constexpr int S2 = S2_;
    static constexpr int N1 = N1_;
    static constexpr int N2 = N2_;
    static constexpr int D = D_;
    static constexpr int K = K_;
    static constexpr int TileM = Tm_;
    static constexpr int TileK = Tk_;
    static constexpr int TileD = Td_;
    static constexpr int GSliceMax = GSliceMax_;

    static constexpr int G = N1 / N2;
    static constexpr int GSliceCount = (G + GSliceMax - 1) / GSliceMax;
    static constexpr int WorkCount = B * S1 * N2 * GSliceCount;
    static constexpr int DBlockCount = (D + TileD - 1) / TileD;
    static constexpr int KvBlockCount = (S2 + TileK - 1) / TileK;

    static constexpr int g_head_begin(int kv_head, int g_slice) {
        return kv_head * G + g_slice * GSliceMax;
    }

    static constexpr int g_slice_size(int g_slice) {
        const int remaining = G - g_slice * GSliceMax;
        return remaining < GSliceMax ? remaining : GSliceMax;
    }

    static constexpr std::size_t q_offset(int batch, int token, int q_head, int dim) {
        return (((static_cast<std::size_t>(batch) * S1 + token) * N1 + q_head) * D + dim);
    }

    static constexpr std::size_t kv_offset(int batch, int token, int kv_head, int dim) {
        return (((static_cast<std::size_t>(batch) * S2 + token) * N2 + kv_head) * D + dim);
    }

    static constexpr std::size_t out_offset(int batch, int token, int q_head, int dim) {
        return q_offset(batch, token, q_head, dim);
    }

    static constexpr std::size_t q_work_offset(const QsmlaWorkItem& work) {
        return q_offset(work.batch, work.q_token, work.q_head_begin, 0);
    }

    static constexpr std::size_t kv_work_offset(const QsmlaWorkItem& work) {
        return kv_offset(work.batch, 0, work.kv_head, 0);
    }

    static constexpr std::size_t out_work_offset(const QsmlaWorkItem& work) {
        return out_offset(work.batch, work.q_token, work.q_head_begin, 0);
    }

    static constexpr QsmlaWorkItem decode_work(int work_id) {
        const int g_slice = work_id % GSliceCount;
        work_id /= GSliceCount;
        const int kv_head = work_id % N2;
        work_id /= N2;
        const int q_token = work_id % S1;
        const int batch = work_id / S1;
        return {batch, q_token, kv_head, g_slice,
                g_head_begin(kv_head, g_slice), g_slice_size(g_slice)};
    }
};

#endif
