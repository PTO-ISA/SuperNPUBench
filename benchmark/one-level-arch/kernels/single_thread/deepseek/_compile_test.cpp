// 编译测试驱动：实例化全部 19 个迁移 kernel，用于 linx 工具链编译验证。
#include <common/pto_tileop.hpp>
#include <cstdint>
#include "single_thread/deepseek/engram/fused_weight.hpp"
#include "single_thread/deepseek/engram/engram_hash.hpp"
#include "single_thread/deepseek/mhc/expand_to_mhc.hpp"
#include "single_thread/deepseek/mhc/expand_to_mhc_bwd.hpp"
#include "single_thread/deepseek/mhc/sinkhorn.hpp"
#include "single_thread/deepseek/mhc/norm_fn.hpp"
#include "single_thread/deepseek/mhc/multilayer_recompute.hpp"
#include "single_thread/deepseek/moe/normalize_weight.hpp"
#include "single_thread/deepseek/moe/topk_gate.hpp"
#include "single_thread/deepseek/moe/group_count_aux_fi.hpp"
#include "single_thread/deepseek/moe/expand_to_fused.hpp"
#include "single_thread/deepseek/moe/reduce_fused.hpp"
#include "single_thread/deepseek/moe/inplace_unique_group_indices.hpp"
#include "single_thread/deepseek/moe/mask_indices_by_tp.hpp"
#include "single_thread/deepseek/moe/get_fused_mapping.hpp"
#include "single_thread/deepseek/quant/cast_back.hpp"
#include "single_thread/deepseek/quant/per_token_cast.hpp"
#include "single_thread/deepseek/quant/swiglu_fused_cast.hpp"
#include "single_thread/deepseek/transpose/batched_transpose.hpp"

using namespace supernpu::tile_isa;
int main() {
    [[maybe_unused]] auto a01 = &fused_weight<2, 64, 64>;
    [[maybe_unused]] auto a02 = &engram_hash_layer<16, 8, 8, 8>;
    [[maybe_unused]] auto a03 = &expand_to_mhc_fwd<16, 64, 4>;
    [[maybe_unused]] auto a04 = &expand_to_mhc_bwd<16, 64, 16, 64>;
    [[maybe_unused]] auto a05 = &sinkhorn_fwd<2, 16, 1>;
    [[maybe_unused]] auto a06 = &rms_norm<16, 8>;
    [[maybe_unused]] auto a07 = &fn_normw_merge_fwd<16, 32, 16, 32>;
    [[maybe_unused]] auto a08 = &multilayer_recompute<16, 16, 2>;
    [[maybe_unused]] auto a09 = &normalize_weight<16, 8>;
    [[maybe_unused]] auto a10 = &topk_gate<16, 32, 4>;
    [[maybe_unused]] auto a11 = &group_count<16, 8, 32>;
    [[maybe_unused]] auto a12 = &aux_fi<16, 8, 32>;
    [[maybe_unused]] auto a13 = &expand_to_fused<__bf16, 16, 64, 4, 32, 64>;
    [[maybe_unused]] auto a14 = &reduce_fused<__bf16, float, 16, 64, 4, 32, 64>;
    [[maybe_unused]] auto a15 = &inplace_unique_group_indices<16, 8, 8>;
    [[maybe_unused]] auto a16 = &mask_indices_by_tp<16, 8>;
    [[maybe_unused]] auto a17 = &get_fused_mapping<16, 8, 32>;
    [[maybe_unused]] auto a18 = &cast_back_per_token<__bf16, 16, 16>;
    [[maybe_unused]] auto a19 = &cast_back_per_channel<__bf16, 16, 32, 16, 32>;
    [[maybe_unused]] auto a20 = &per_token_cast<16, 16>;
    [[maybe_unused]] auto a21 = &per_channel_cast<16, 32, 32>;
    [[maybe_unused]] auto a22 = &swiglu_forward_and_per_token_cast<16, 16, 16>;
    [[maybe_unused]] auto a23 = &batched_transpose<float, 2, 16, 16>;
    return 0;
}
