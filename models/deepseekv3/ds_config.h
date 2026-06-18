#ifndef DS_CONFIG_H
#define DS_CONFIG_H

#include <string>

enum class DataType {
    FP8,
    BF16,
    FP32,
    FP64
};

enum class ScoreFunc {
    SOFTMAX,
    SIGMOID
};

struct ModelArgs {
    static constexpr bool attention_bias = false;
    static constexpr bool attention_dropout = false; // training
    static constexpr int max_batch_size = 8;
    static constexpr int max_seq_len = 4096 * 4;
    static constexpr DataType dtype = DataType::FP32;
    static constexpr int vocab_size = 102400;
    static constexpr int dim = 1024;//2048;
    static constexpr int inter_dim = 4096;//10944;
    static constexpr int moe_inter_dim = 64;//1024 change//1408;
    static constexpr int n_layers = 1;
    static constexpr int n_dense_layers = 0;
    static constexpr int n_heads = 16;
    // moe
    static constexpr int n_routed_experts = 64;
    static constexpr int n_shared_experts = 2;
    static constexpr int n_activated_experts = 8;//6;
    static constexpr int n_expert_groups = 8;
    static constexpr int n_limited_groups = 4;
    static constexpr ScoreFunc score_func = ScoreFunc::SIGMOID;
    static constexpr float route_scale = 1.0f;
    // mla
    static constexpr int q_lora_rank = 64;
    static constexpr int kv_lora_rank = 512;
    static constexpr int qk_nope_head_dim = 128;
    static constexpr int qk_rope_head_dim = 128; //64 FIXME flash attention layout
    static constexpr int qk_head_dim = qk_nope_head_dim+qk_rope_head_dim;
    static constexpr int v_head_dim = 128;
    // yarn
    static constexpr int original_seq_len = 4096;
    static constexpr float rope_theta = 10000.0f;
    static constexpr float rope_factor = 40.0f;
    static constexpr int beta_fast = 32;
    static constexpr int beta_slow = 1;
    static constexpr float mscale = 1.0f;
};

template<DataType T>
struct DType;

#ifdef _linx
template<>
struct DType<DataType::BF16> {
    using type = __half;
};
#else
template<>
struct DType<DataType::BF16> {
    using type = float;
};
#endif

template<>
struct DType<DataType::FP32> {
    using type = float;
};

template<>
struct DType<DataType::FP64> {
    using type = double;
};

// 使用 dtype
using dtype = typename DType<ModelArgs::dtype>::type;

#endif