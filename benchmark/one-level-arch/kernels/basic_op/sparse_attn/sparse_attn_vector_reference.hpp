#pragma once
#include <common/pto_tileop.hpp>
#include <cstddef>

namespace deepseek_v41 {
// Contiguous Q/O [B,S,H,D], shared KV [B,N,D], indices [B,S,K], sink [H].
// Invoke on each of four PEs. Finite inputs/sinks, positive scale required.
// Vector baseline: streaming one selected KV row; no SharedTile allocation.
template<class T, int D>
void sparse_attn_vector_reference(T* out, T* q, T* kv, float* sink, const int* indices,
                 int B, int S, int H, int N, int K, float scale) {
    using namespace pto;
    static_assert(D > 0 && D <= 1024, "supported head dimension: 1..1024");
    using V = Tile<Location::Vec, float, 1, D, BLayout::RowMajor>;
    using R = Tile<Location::Vec, float, 1, 1, BLayout::RowMajor>;
    using IO = Tile<Location::Vec, T, 1, D, BLayout::RowMajor>;
    using G = global_tensor<T, RowMajor<1,D>>;
    using I = global_iterator<G, IO>;
    using SG = global_tensor<float, RowMajor<1,1>>;
    using SI = global_iterator<SG,R>;
    const int pe = static_cast<int>(get_thread_idx());
    for (std::size_t row = pe; row < std::size_t(B)*S*H; row += 4) {
        const int h = row % H;
        const std::size_t token = row / H;
        const int b = token / S;
        IO qi, ki, oi;
        V fq, fk, product, acc, weighted;
        R m, sum, score, next_m, alpha, prob, tmp, inv;
        I iq(q + row*D), io(out + row*D);
        SI is(sink + h);
        auto gq = iq(0,0); auto go = io(0,0); auto gs = is(0,0);
        TLOAD(qi, gq); TCVT(fq, qi);
        // Sink is a zero-valued virtual token: stable even for an empty list.
        TLOAD(m, gs); TEXPANDS(sum, 1.0f); TEXPANDS(acc, 0.0f);
        for (int j=0; j<K; ++j) {
            int idx = indices[token*K+j];
            if (idx < 0 || idx >= N) continue;
            I ik(kv + (std::size_t(b)*N+idx)*D);
            auto gk = ik(0,0); TLOAD(ki, gk); TCVT(fk, ki);
            TMUL(product, fq, fk); TROWSUM(score, product);
            TMULS(score, score, scale); TMAX(next_m, m, score);
            TSUB(alpha, m, next_m); TEXP(alpha, alpha);
            TSUB(prob, score, next_m); TEXP(prob, prob);
            TMUL(tmp, sum, alpha); TADD(sum, tmp, prob);
            TROWEXPANDMUL(acc, acc, alpha);
            TROWEXPANDMUL(weighted, fk, prob); TADD(acc, acc, weighted);
            m = next_m;
        }
        TRECIP(inv, sum); TROWEXPANDMUL(acc, acc, inv);
        TCVT(oi, acc); TSTORE(go, oi);
    }
}
} // namespace deepseek_v41
