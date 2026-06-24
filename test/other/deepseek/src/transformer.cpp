#include <common/pto_tileop.hpp>
#include "ds_config.h"
#include "tensor.h"
#include "mla.hpp"
#include "moe.hpp"
#include "tensorwrite.hpp"

int main(){
    Tensor<dtype, 2, 128, ModelArgs::dim> mla_in;
    Tensor<dtype, 2, 128, ModelArgs::dim> mla_out;
    Tensor<dtype, 2, 128, ModelArgs::dim> moe_out;
    Tensor<dtype, 128, ModelArgs::qk_rope_head_dim> freqs_cis;

    int tmp = 0;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 128; ++j) {
            for (int k = 0; k < ModelArgs::dim; ++k) {
                mla_in(i, j, k) = static_cast<dtype>(1);
            }
        }
    }

    for (int i = 0; i < 128; ++i) {
        for (int j = 0; j < ModelArgs::qk_rope_head_dim; ++j) {
            freqs_cis(i, j) = static_cast<dtype>(j%2+1);
        }
    }

    MLA<dtype, 2, 128, ModelArgs>(mla_out, mla_in, freqs_cis);
    rmsnorm<dtype, 2*128, ModelArgs::dim,32,32>(mla_out.data(), mla_out.data());
    MoE<dtype, 2, 128, ModelArgs>(moe_out, mla_out);
    rmsnorm<dtype, 2*128, ModelArgs::dim,32,32>(moe_out.data(), moe_out.data());
}