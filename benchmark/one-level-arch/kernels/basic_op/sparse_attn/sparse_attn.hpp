#pragma once
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <type_traits>

namespace deepseek_v41 {
using namespace pto;

// One query token, HeadsPerGroup padded heads; four PEs split the rows equally.
// Single pass over selected KV blocks: QK -> online softmax -> PV.
template<class T, int HeadDim, int KVLength, int TopK, int HeadsPerGroup=64>
void sparse_attn_cube_group(T* out, T* q, T* kv, float* sink,
    const int* indices, float softmax_scale,
    T* prob_scratch) {
    // Matrix M = query heads; score columns = selected KV slots.
    // HeadDim is the QK reduction dimension and the output feature dimension.
    static_assert(HeadsPerGroup==64 || HeadsPerGroup==128,
                  "four-PE Cube groups support 64 or 128 heads");
    constexpr int kHeadsPerGroup=HeadsPerGroup, kHeadsPerPE=HeadsPerGroup/4;
    constexpr int kKeysPerBlock=32, kDimsPerBlock=64;
    constexpr int kDimBlocks=HeadDim/kDimsPerBlock;
    const int pe_id=static_cast<int>(get_thread_idx());
    using tileQMatrix = SharedMatrixLeft<T, kHeadsPerGroup, kDimsPerBlock>;
    // Shared B interprets rows as QK output columns by default.
    // PV toggles transpose_b() to consume this same [32,64] layout as V.
    using tileKMatrix = SharedMatrixRight<T, kKeysPerBlock, kDimsPerBlock>;
    using tilePMatrix = SharedMatrixLeft<T, kHeadsPerGroup, kKeysPerBlock>;
    using tileVMatrix = SharedMatrixRight<T, kKeysPerBlock, kDimsPerBlock>;
    using tileQShared = SharedTile<tileQMatrix>;
    using tileKShared = SharedTile<tileKMatrix>;
    using tilePShared = SharedTile<tilePMatrix>;
    using tileVShared = SharedTile<tileVMatrix>;
    using tileScoreCube = std::conditional_t<HeadsPerGroup==64,
        CubeAccumulatorM16<float, kHeadsPerPE, kKeysPerBlock>, CubeAccumulatorM32<float, kHeadsPerPE, kKeysPerBlock>>;
    using tilePVCube = std::conditional_t<HeadsPerGroup==64,
        CubeAccumulatorM16<float, kHeadsPerPE, kDimsPerBlock>, CubeAccumulatorM32<float, kHeadsPerPE, kDimsPerBlock>>;
    using tileW = tileScoreCube;
    using tilePShard = std::conditional_t<HeadsPerGroup==64,
        CubeTileM16<T,kHeadsPerPE,kKeysPerBlock>, CubeTileM32<T,kHeadsPerPE,kKeysPerBlock>>;
    using tileO = tilePVCube;
    using tileOCast = std::conditional_t<HeadsPerGroup==64,
        CubeTileM16<T,kHeadsPerPE,kDimsPerBlock>, CubeTileM32<T,kHeadsPerPE,kDimsPerBlock>>;
    using tileMax = std::conditional_t<HeadsPerGroup==64,
        VecTileM16<float,kHeadsPerPE,1,kHeadsPerPE,1>, VecTileM32<float,kHeadsPerPE,1,kHeadsPerPE,1>>;
    using tileSum = tileMax;
    using tileReduce = std::conditional_t<HeadsPerGroup==64,
        VecTileM16<float,kHeadsPerPE,kKeysPerBlock,kHeadsPerPE,1>, VecTileM32<float,kHeadsPerPE,kKeysPerBlock,kHeadsPerPE,1>>;

    using gmQ = global_tensor<T, RowMajor<kHeadsPerGroup, HeadDim>>;
    using gmGatherKV = global_tensor<T, RowMajor<kKeysPerBlock, HeadDim>>;
    using gmO = global_tensor<T, RowMajor<kHeadsPerGroup, HeadDim>>;
    using itQ = global_iterator<gmQ, tileQMatrix>;
    using itK = global_iterator<gmGatherKV, tileKMatrix>;
    using itV = global_iterator<gmGatherKV, tileVMatrix>;
    using itO = global_iterator<gmO, tileOCast>;

    using gmProbScratch = global_tensor<T, RowMajor<kHeadsPerGroup, kKeysPerBlock>>;
    using gmRowState = global_tensor<float, RowMajor<kHeadsPerPE, 1>>;
    using itProbShard = global_iterator<gmProbScratch, tilePShard>;
    using itPShared = global_iterator<gmProbScratch, tilePMatrix>;
    using itRowState = global_iterator<gmRowState, tileMax>;
    itProbShard gIterProb(prob_scratch);
    itPShared gIterP(prob_scratch);


    // Gathered KV [32, head_dim]; the same tensor supplies both K and V.
    T kv_block[kKeysPerBlock*HeadDim];
    float mask[kHeadsPerPE*kKeysPerBlock]; // [heads per PE, 32 selected keys]
    itQ gIterQ(q);
    itRowState gSink(sink + pe_id*kHeadsPerPE);
    auto sinkView=gSink(0,0);

    // The sink is a virtual token with logit=sink and value=0.
    // Starting here avoids -inf - -inf for empty/all-masked index lists.
    // Per-head FP32 state: max logit, exponential sum, weighted-value numerator.
    tileMax running_max;
    tileSum running_sum;
    TLOAD(running_max,sinkView);
    TEXPANDS(running_sum,1.0f);
    // Compile-time output tiles stay live across KV blocks; no GM state buffer.
    tileO output_numerator[kDimBlocks];
#pragma clang loop unroll(full)
    for(int dim_block=0;dim_block<kDimBlocks;++dim_block) TEXPANDS(output_numerator[dim_block],0.0f);

    for(int key_begin=0;key_begin<TopK;key_begin+=kKeysPerBlock) {
        // 1. Gather exactly one KV block. -1 is padding anywhere in the list;
        // duplicate entries keep their multiplicity. No extra causal mask.
        for(int key_slot=0;key_slot<kKeysPerBlock;++key_slot) {
            const int kv_index=key_begin+key_slot<TopK ? indices[key_begin+key_slot] : -1;
            const bool valid_index=kv_index>=0 && kv_index<KVLength;
            for(int dim=0;dim<HeadDim;++dim)
                kv_block[key_slot*HeadDim+dim]=valid_index ? kv[std::size_t(kv_index)*HeadDim+dim] : T(0);
            for(int head_row=0;head_row<kHeadsPerPE;++head_row)
                mask[head_row*kKeysPerBlock+key_slot]=valid_index ? 0.0f : -__builtin_inff();
        }
        itK gIterK(kv_block);
        itV gIterV(kv_block);

        // 2. Score = Q * KV^T. Reduce HeadDim in 64-element chunks.
        tileW scores;
#pragma clang loop unroll(full)
        for(int dim_block=0;dim_block<kDimBlocks;++dim_block) {
            tileQShared tq;
            tileKShared tk;
            auto gq=gIterQ(0,dim_block);
            auto gk=gIterK(0,dim_block);
            TLOAD<tileQMatrix,1>(tq,gq);
            TLOAD<tileKMatrix,1>(tk,gk);
            if(dim_block==0) TMATMUL(scores,tq,tk,fixp::keep_acc());
            else TMATMUL_ACC(scores,scores,tq,tk,fixp::keep_acc());
        }
        tileW maskTile;
        TMULS(scores,scores,softmax_scale);
        using gmMask=global_tensor<float,RowMajor<kHeadsPerPE,kKeysPerBlock>>;
        using itMask=global_iterator<gmMask,tileW>;
        itMask maskIter(mask);
        auto maskView=maskIter(0,0);
        TLOAD(maskTile,maskView);
        TADD(scores,scores,maskTile);

        // 3. Online softmax: m'=max(m,max(score)); rescale=exp(m-m').
        // P=exp(score-m'); l'=rescale*l+sum(P).
        tileMax new_max, rescale;
        tileReduce reduce;
        TROWMAX(reduce,scores);
        auto localMax=TREDUCEPREFIXVIEW<tileMax>(reduce);
        TMAX(new_max,running_max,localMax);
        TSUB(rescale,running_max,new_max);
        TEXP(rescale,rescale);
        TROWEXPANDSUB(scores,scores,new_max);
        TEXP(scores,scores); // scores now holds unnormalized P, not logits.
        TROWSUM(reduce,scores);
        auto blockSum=TREDUCEPREFIXVIEW<tileSum>(reduce);
        TMUL(running_sum,running_sum,rescale);
        TADD(running_sum,running_sum,blockSum);
        running_max=new_max;

        // Publish the four per-PE P shards for the cooperative PV matmul.
        tilePShard probability;
        TCVT(probability,scores);
        auto probabilityShard=gIterProb(pe_id,0);
        TSTORE(probabilityShard,probability);

        // 4. O'=rescale*O + P*KV. Reuse each persistent output tile.
#pragma clang loop unroll(full)
        for(int dim_block=0;dim_block<kDimBlocks;++dim_block) {
            TROWEXPANDMUL(output_numerator[dim_block],output_numerator[dim_block],rescale);
            tilePShared tp;
            tileVShared tv;
            auto gp=gIterP(0,0);
            auto gv=gIterV(0,dim_block);
            TLOAD<tilePMatrix,1>(tp,gp);
            TLOAD<tileVMatrix,1>(tv,gv);
            tilePVCube pvCube;
            TMATMUL(pvCube,tp,tv,fixp::keep_acc().transpose_b());
            TADD(output_numerator[dim_block],output_numerator[dim_block],pvCube);
        }
    }

    // 5. Normalize once, including the sink contribution already in l.
    TRECIP(running_sum,running_sum); // Reuse storage: running_sum now holds 1 / denominator.
    itO outputIter(out + pe_id*kHeadsPerPE*HeadDim);
#pragma clang loop unroll(full)
    for(int dim_block=0;dim_block<kDimBlocks;++dim_block) {
        auto outputView=outputIter(0,dim_block);
        tileOCast castResult;
        TROWEXPANDMUL(output_numerator[dim_block],output_numerator[dim_block],running_sum);
        TCVT(castResult,output_numerator[dim_block]);
        TSTORE(outputView,castResult);
    }
}

// Caller owns one workspace, shared by all four PEs. Separate invocations
// need separate workspaces. Scratch may not alias inputs or output.
template<class T, int HeadDim, int HeadsPerGroup=64> struct SparseAttnWorkspace {
    static_assert(HeadsPerGroup==64 || HeadsPerGroup==128);
    alignas(64) T q[HeadsPerGroup*HeadDim], out[HeadsPerGroup*HeadDim], probability[HeadsPerGroup*32];
    alignas(64) float sink[HeadsPerGroup];
};

// 算子接口：所有张量按下列维度顺序连续存储，最后一维连续。
// 张量     类型       shape
// q        T          [batch_size, query_length, num_heads, HeadDim]
// kv       T          [batch_size, KVLength, HeadDim]
// sink     float      [num_heads]
// indices  int32      [batch_size, query_length, TopK]
// out      T          [batch_size, query_length, num_heads, HeadDim]
// T 使用 FP16/BF16；内部 softmax 统计量和输出累加使用 FP32。
// HeadDim = head_dim，每个注意力头的特征维度，也是输出的最后一维。
// KVLength 是 KV 总行数；TopK 是每个 query 的索引槽位数，不是 head_dim。
// kv 同时作为 K 和 V；同一 query 的所有 heads 共享 indices 和 KV。
// indices 中有效行号范围为 [0, KVLength)，-1 在任意位置均表示 padding。
// 重复索引重复参与计算；其他越界索引也会跳过，不额外施加 causal mask。
// sink 是每个 head 的 logit，只贡献 softmax 分母，不贡献输出分子。
//
// 例：batch_size=1, query_length=4, num_heads=96,
//     HeadDim=128, KVLength=4096, TopK=64：
//     q/out=[1,4,96,128], kv=[1,4096,128], sink=[96], indices=[1,4,64]。
//
// HeadDim/KVLength/TopK 为编译期尺寸；batch/query/head 数为运行期参数。
// num_heads 可为任意正数，不要求等于或整除 HeadsPerGroup。
// HeadsPerGroup=64/128 仅控制内部 Cube 分组，不改变输入输出 shape。
// HeadDim 必须是正的 64 倍数；KVLength>0，TopK>=0。
// 四个 PE 使用相同参数和共享 workspace 调用；输入、输出、workspace 不得别名。
template<class T, int HeadDim, int KVLength, int TopK, int HeadsPerGroup=64>
void sparse_attn(T* out, T* q, T* kv, float* sink, const int* indices,
                 int batch_size, int query_length, int num_heads, float softmax_scale, SparseAttnWorkspace<T,HeadDim,HeadsPerGroup>& w) {
    static_assert(HeadDim>0 && HeadDim%64==0, "HeadDim must be a multiple of 64");
    static_assert(sizeof(T)==2, "Cube baseline requires FP16/BF16 storage");
    static_assert(KVLength>0 && TopK>=0, "invalid KV or index count");
    constexpr int heads_per_pe=HeadsPerGroup/4;
    // Conservative upper bound for Q/K/P/V simultaneously live (bytes).
    static_assert((HeadsPerGroup*64+32*64+HeadsPerGroup*32+32*64)*sizeof(T)<=262144,
                  "SharedTile capacity exceeded");
    const int pe=static_cast<int>(get_thread_idx());
    for(int batch=0;batch<batch_size;++batch) for(int query=0;query<query_length;++query) for(int head_begin=0;head_begin<num_heads;head_begin+=HeadsPerGroup) {
        const std::size_t offset=((std::size_t(batch)*query_length+query)*num_heads+head_begin)*HeadDim;
        const int* selected=indices+(std::size_t(batch)*query_length+query)*TopK;
        T* batchKV=kv+std::size_t(batch)*KVLength*HeadDim;
        if(head_begin+HeadsPerGroup<=num_heads) {
            // Complete groups need no Q/sink staging or output copy.
            sparse_attn_cube_group<T,HeadDim,KVLength,TopK,HeadsPerGroup>(out+offset,q+offset,batchKV,
                sink+head_begin,selected,softmax_scale,w.probability);
        } else {
            // Only the head tail needs padding. All PEs take the same branch.
            for(int head=0;head<HeadsPerGroup;++head) {
                w.sink[head]=head_begin+head<num_heads?sink[head_begin+head]:0.0f;
                for(int dim=0;dim<HeadDim;++dim)
                    w.q[head*HeadDim+dim]=head_begin+head<num_heads?q[offset+head*HeadDim+dim]:T(0);
            }
            sparse_attn_cube_group<T,HeadDim,KVLength,TopK,HeadsPerGroup>(w.out,w.q,batchKV,
                w.sink,selected,softmax_scale,w.probability);
            for(int head=pe*heads_per_pe;head<(pe+1)*heads_per_pe && head_begin+head<num_heads;++head)
                for(int dim=0;dim<HeadDim;++dim) out[offset+head*HeadDim+dim]=w.out[head*HeadDim+dim];
        }
    }
}
} // namespace deepseek_v41
