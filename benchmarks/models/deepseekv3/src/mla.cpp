#include "ds_config.h"
#include "tensor.h"
#include "tensorwrite.hpp"
#include "mla.hpp"


int main(){
    Tensor<dtype, 2, 128, ModelArgs::dim> mla_input;
    Tensor<dtype, 2, 128, ModelArgs::dim> mla_output;
    Tensor<dtype, 128, ModelArgs::qk_rope_head_dim> freqs_cis;

    int tmp = 0;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 128; ++j) {
            for (int k = 0; k < ModelArgs::dim; ++k) {
                mla_input(i, j, k) = static_cast<dtype>(1);
            }
        }
    }

    for (int i = 0; i < 128; ++i) {
        for (int j = 0; j < ModelArgs::qk_rope_head_dim; ++j) {
            freqs_cis(i, j) = static_cast<dtype>(j%2+1);
        }
    }

    MLA<dtype, 2, 128, ModelArgs>(mla_output, mla_input, freqs_cis);
}