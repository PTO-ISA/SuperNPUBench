#include "ds_config.h"
#include <common/pto_tileop.hpp>
#include "tensor.h"
#include "mla.hpp"
#include "tensorwrite.hpp"

#ifndef ROW
#define ROW 320
#endif

#ifndef PE_DIM
#define PE_DIM 128
#endif



int main() {
    // 定义输入和输出张量
    Tensor<dtype, ROW, PE_DIM> x;
    Tensor<dtype, ROW, PE_DIM> freqs;

    // 给 x 赋值为 1, 2, 1, 2, ... 循环
    for (int i = 0; i < ROW; ++i) {
        for (int j = 0; j < PE_DIM; ++j) {
            x(i, j) = static_cast<dtype>((j % 2) + 1); // 1, 2, 1, 2, ...
        }
    }

    // 给 freqs 赋值为 1, 2, 1, 2, ... 循环
    for (int i = 0; i < ROW; ++i) {
        for (int j = 0; j < PE_DIM; ++j) {
          if(j<PE_DIM/2)
            freqs(i, j) = static_cast<dtype>((j % 2) + 1); // 1, 2, 1, 2, ...
          else
            freqs(i, j) = static_cast<dtype>((j % 2)*2+2);
        }
    }

    #ifdef __cpu_sim__
    // 将输入数据写入文件
    writeTensorToFile<dtype, ROW, PE_DIM>(x.data(), "x_before.txt");
    writeTensorToFile<dtype, ROW, PE_DIM>(freqs.data(), "freqs_before.txt");
    #endif

    // 执行 apply_rotary_emb 操作
    apply_rotary_emb<dtype, ROW, PE_DIM>(x.data(), freqs.data());

    #ifdef __cpu_sim__
    // 将输出数据写入文件
    writeTensorToFile<dtype, ROW, PE_DIM>(x.data(), "x_after.txt");
    std::cout << "数据已写入文件 x_before.txt, freqs_before.txt, freqs_after.txt" << std::endl;
    #endif
    
    return 0;
}