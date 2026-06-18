添加新 CASE 的步骤

1. 修改 HPP 文件

步骤说明：

 · 文件路径：PTOTileLib/include/cpu_sim/

 · 操作说明：
   
   1. 如果是添加一个新的运算方式（如 texp），则需要新建一个 HPP 文件。
   2. 如果是同一运算方式的不同属性（如不同的矩阵尺寸或 tile 大小），则直接在对应的 HPP 文件中添加。

 · 标准函数格式：
   
   · 文件头需要包含必要的头文件。
   · 声明变量和函数名称时，需注意命名规范。
   · 如果有一个综合函数记得写清条件
   . 引用的TAdd_Impl<tile_shape>(dst.data(), src0.data(), src1.data())需要用.data的形式。
   . 添加之后会统一进行编译。

示例代码：
```
#ifndef TADD_HPP
#define TADD_HPP

#include "common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape>
void TAdd_Impl(typename tile_shape::TileDType dst,
               typename tile_shape::TileDType src0,
               typename tile_shape::TileDType src1) {
  for (uint16_t i = 0; i < tile_shape::ValidRow; ++i)
    for (uint16_t j = 0; j < tile_shape::ValidCol; ++j) {
      dst[i * tile_shape::RowStride + j] =
          src0[i * tile_shape::RowStride + j] +
          src1[i * tile_shape::RowStride + j];
    }
}

template <is_tile_data_v tile_shape>
void TADD(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  if constexpr (tile_shape::isRowMajor) {
    TAdd_Impl<tile_shape>(dst.data(), src0.data(), src1.data());
  } else {
    static_assert(tile_shape::isRowMajor,
                  "Storage layout type not supported");
  }
}

#endif

```
----------------------------------------

2. 新建调用函数与python接口的hpp文件

文件路径：PTOTileLib/test/py_api/src

步骤说明：

 1. 新建文件：
    
    · 添加固定文件头。
    . 声明要传入的参数
    . 声明矩阵的形状与layout
    . 进行矩阵操作，并且使用TCOPYIN,TCOPYOUT函数以及上一个步骤声明的函数来进行操作。注意满足TCOPYIN,TCOPYOUT对于矩阵layout的要求。
    . 对函数进行绑定，注意在绑定时需要进行接口的转换以及参数的传入。
    . 之后在tileop_py.cpp中加入要编译的文件名
    ```
#pragma once

#include <common/pto_tileop.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row,
          uint16_t tile_col>
void tadd_py(float* dst, float* src0, float* src1){
    using gm_shape = global_tensor<float, RowMajor<gm_row, gm_col>>;
    using tile_shape = tile_tensor<float, RowMajor<tile_row, tile_col>>;
    uint16_t block_row = gm_row / tile_row;
    uint16_t block_col = gm_col / tile_col;
    for (int i = 0; i < block_row; ++i) {
        for (int j = 0; j < block_col; ++j) {
            int offset = i * (tile_row * gm_col) + j * tile_col;
            gm_shape s0(src0 + offset);
            gm_shape s1(src1 + offset);
            gm_shape res(dst + offset);  

            tile_shape d0, d1, d2;
            TCOPYIN(d0, s0);
            TCOPYIN(d1, s1);
            TADD(d2, d0, d1);
            TCOPYOUT(res, d2);
        }
    }
}

#ifdef __cpu_sim__ 
    void bind_tadd(py::module_& m) {
        m.def("tadd", [](py::array_t<float> dst_py, py::array_t<float> src0_py, py::array_t<float> src1_py){
            float* dst = static_cast<float*>(dst_py.request().ptr);
            float* src0 = static_cast<float*>(src0_py.request().ptr);
            float* src1 = static_cast<float*>(src1_py.request().ptr);
            tadd_py<16,16,8,8>(dst, src0, src1);}
        );
        return;
    }
#endif
    ```


----------------------------------------

3. 修改 CONFIG.JSON 文件和 ref_func_lib.py 文件

文件路径：/PTOTileLib/test/py_api/golden_cmp/

步骤说明：

 1. 在config.json文件中的 cases 中添加新测试用例：
    
    · 按照以下格式添加新函数的属性。
        ```
        {
            "name": "tadd",
            "group": "tadd",
            "input_shapes": [[16, 16], [16, 16]],
            "output_shape": [16, 16],
            "ref_func":"lambda input: tadd(input[0], input[1])", 
            "test_func":"lambda res, input: tileop_py.tadd_api.tadd(res, input[0], input[1])"
        }
        ```
    · **name**：函数名称（与绑定函数名称一致）。
    · **group**：操作组名称（如 texp）。
    · **input_shapes**：输入矩阵的形状，注意即使是一个也要用大括号包括进来。
    · **output_shapes**：输出矩阵的形状。
    . **ref_func**: 该函数在ref_func_lib中有定义。
    . **test_func**: 该函数调用的是绑定的函数。

 2. 在 ref_func_lib.py 中添加需要进行的python操作：
    
    · 按照以下格式添加python操作。
    ```
    def tadd(a, b):
    return a + b
    ```

----------------------------------------

4. 运行验证

添加新的测试用例
按照以下步骤完成修改后，即可成功添加新的测试用例。确保所有文件的修改内容与 config.json 中的定义一致，以避免运行时错误。

在 /PTOTileLib/test/py_api/ 路径下，执行以下命令：
```
make clean  
make TESTCASE=tileop_py  
python3 golden_cmp/golden_cmp.py -i tadd  
```
其中 -i 后面跟着的是函数的名称，具体的函数名可以参考 config.json 文件中的内容。  
之后print出的对比结果中，在最后两行会显示loss（误差）以及是否pass or fail

