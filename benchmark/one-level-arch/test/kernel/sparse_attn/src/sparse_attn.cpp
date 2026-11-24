#include "basic_op/sparse_attn/sparse_attn.hpp"
// Q=0, KV=1, sink=0: each output is valid_count/(valid_count+1).
// Covers middle -1, duplicate indices, K tail, two KV blocks, head tail,
// multiple queries, D accumulation, and an entirely empty selection.
#ifndef SPARSE_NUM_HEADS
#define SPARSE_NUM_HEADS 65
#endif
#ifndef SPARSE_HEADS_PER_GROUP
#define SPARSE_HEADS_PER_GROUP 64
#endif
static constexpr int H=SPARSE_NUM_HEADS, GroupHeads=SPARSE_HEADS_PER_GROUP;
static constexpr int D=128, S=2, K=35;
static _Float16 q[S*H*D] = {}, kv[2*D], out[S*H*D];
static float sink[H] = {};
static int indices[S*K];
static deepseek_v41::SparseAttnWorkspace<_Float16,D,GroupHeads> workspace;
int main() {
    for(int i=0;i<2*D;++i) kv[i]=1.0f;
    for(int i=0;i<K;++i) {
        indices[i]=(i==1 ? -1 : i%2);
        indices[K+i]=-1;
    }
    deepseek_v41::sparse_attn<_Float16,D,2,K,GroupHeads>(
        out,q,kv,sink,indices,1,S,H,1.0f,workspace);
    const int pe=static_cast<int>(get_thread_idx());
    for(int h=0;h<H;++h) if ((h%GroupHeads)/(GroupHeads/4)==pe) {
        for(int d=0;d<D;++d) {
            float v=out[h*D+d];
            if (!(v>34.0f/35.0f-0.003f && v<34.0f/35.0f+0.003f)) return 1;
            if (out[(H+h)*D+d]!=0) return 2;
        }
    }
    return 0;
}
