#include "ds_config.h"
#include "tensor.h"
#include "tensorwrite.hpp"
#include "moe.hpp"


int main(){
    Tensor<dtype, 2, 128, ModelArgs::dim> moe_input;
    Tensor<dtype, 2, 128, ModelArgs::dim> moe_output;

    int tmp = 0;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 128; ++j) {
            for (int k = 0; k < ModelArgs::dim; ++k) {
                moe_input(i, j, k) = static_cast<dtype>(1);
            }
        }
    }

    #ifdef __cpu_sim__
    writeTensorToFile<dtype, 2*128, ModelArgs::dim>(moe_input.data(), "moe_input_cpp.txt");
    #endif

    MoE<dtype, 2, 128, ModelArgs>(moe_output, moe_input);

    #ifdef __cpu_sim__
    writeTensorToFile<dtype, 2*128, ModelArgs::dim>(moe_output.data(), "moe_output_cpp.txt");
    #endif
    // printf("moe_output:\n");
    // for(int i=0;i<2*128;i++){
    //     for(int j=0;j<ModelArgs::dim;j++){
    //         printf("%.8f ", moe_output.data()[i*ModelArgs::dim+j]);
    //     }
    //     printf("\n");
    // }
}