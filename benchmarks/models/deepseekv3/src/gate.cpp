#include "ds_config.h"
#include "tensor.h"
#include "moe.hpp"
#include "tensorwrite.hpp"

#ifndef BSZ
#define BSZ 2
#endif

#ifndef SLEN
#define SLEN 128
#endif

int main(){
    Tensor<dtype, BSZ, SLEN, ModelArgs::dim> gate_input;
    Tensor<dtype, BSZ, SLEN, ModelArgs::n_activated_experts> gate_weight;
    Tensor<dtype, BSZ, SLEN, ModelArgs::n_activated_experts> gate_indices;

    int tmp = 0;
    for (int i = 0; i < BSZ; ++i) {
        for (int j = 0; j < SLEN; ++j) {
            for (int k = 0; k < ModelArgs::dim; ++k) {
                gate_input(i, j, k) = static_cast<dtype>(1);
            }
        }
    }

    Gate<dtype, BSZ, SLEN, ModelArgs>(gate_weight.data(), gate_indices.data(), gate_input.data(), nullptr);
}