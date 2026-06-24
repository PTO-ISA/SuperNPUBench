添加新 CASE 的步骤

1. 修改 HPP 文件

步骤说明：

 · 文件路径：JanusCoreBench/test/golden_cmp/py_api/src/

 · 操作说明：
   
   1. 如果是添加一个新的运算方式（如 texp），则需要新建一个 HPP 文件。
   2. 如果是同一运算方式的不同属性（如不同的矩阵尺寸或 tile 大小），则直接在对应的 HPP 文件中添加。

 · 标准函数格式：
   
   · 文件头需要包含必要的头文件。
   · 声明变量和函数名称时，需注意命名规范。
   · 函数声明后，需将函数与模块绑定（m.def）。
   · 注意：传递的参数需与 config.json 文件中的定义一致。

示例代码：
```
#pragma once

#include <common/pto_tileop.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template <uint64_t gm_row, uint64_t gm_col, uint64_t tile_row, uint64_t tile_col>
void texp1_py(py::array_t<float> dst_py, py::array_t<float> src_py) {
    float* dst = static_cast<float*>(dst_py.request().ptr);
    float* src = static_cast<float*>(src_py.request().ptr);

    using gm_shape = global_tensor<float, RowMajor<gm_row, gm_col>>;
    using tile_shape = Tile<Location::Vec, float, tile_row, tile_col, BLayout::RowMajor>;

    uint16_t block_row = gm_row / tile_row;
    uint16_t block_col = gm_col / tile_col;

    for (int i = 0; i < block_row; ++i) {
        for (int j = 0; j < block_col; ++j) {
            int offset = i * (tile_row * gm_col) + j * tile_col;
            gm_shape s0(src + offset);
            gm_shape res(dst + offset);

            tile_shape d0, d1;

            TCOPYIN(d0, s0);
            TEXP(d1, d0);
            TCOPYOUT(res, d1);
        }
    }
}

// 绑定函数
#ifdef __cpu_sim__
void bind_texp(py::module_& m) {
    m.def("texp1", &texp1_py<8, 8, 2, 2>);
    return;
}
#endif

```
----------------------------------------

2. 修改 TILEOP_PY.CPP 文件

文件路径：JanusCoreBench/test/golden_cmp/py_api/tileop_py.cpp

步骤说明：

 1. 添加文件头：
    
    · 在文件开头添加包含新 HPP 文件的头文件路径。
    ```
    #include "src/texp1.hpp"
    ```

 2. 添加绑定内容：
    
    · 在模块中绑定新函数。
    ```
    py::module_ _api = m.def_submodule("_api", "API module");
    bind_texp(_api);
    ```

----------------------------------------

3. 修改 CONFIG.JSON 文件

步骤说明：

 1. 在 cases 中添加新测试用例：
    
    · 按照以下格式添加新函数的属性。
        ```
        {
            "name": "texp1",
            "group": "texp",
            "input_shapes": [[8, 8]],
            "output_shape": [8, 8]
        }
        ```
    · **name**：函数名称（与绑定函数名称一致）。
    · **group**：操作组名称（如 texp）。
    · **input_shapes**：输入矩阵的形状。
    · **output_shapes**：输出矩阵的形状。

 2. 在 op_map 中添加操作映射：
    
    · 按照以下格式添加新操作的映射。
    ```
    "texp": [
            "lambda src0: src0.exp()",
            "lambda case: getattr(tileop_py.texp_api, case)"
        ]
    ```
    · **lambda src0: src0.exp()**：Python 中的参考实现。
    · **lambda case: getattr(tileop_py.texp_api, case)**：调用对应的 HPP 文件中的函数。

----------------------------------------

4. 运行验证

添加新的测试用例
按照以下步骤完成修改后，即可成功添加新的测试用例。确保所有文件的修改内容与 config.json 中的定义一致，以避免运行时错误。

在 /JanusCoreBench/test 路径下，执行以下命令：
```
make clean  
make TESTCASE=tileop_py PLAT=cpu PY_LIB=on  
python3 golden_cmp/golden_cmp.py -i tadd1  
```
其中，PLAT 和 PY_LIB 的值可以根据需要进行修改。具体的可选项可参考 common 文件夹下的 Makefile 。其中 -i 后面跟着的是函数的名称，具体的函数名可以参考 config.json 文件中的内容。  
之后print出的对比结果中，在最后两行会显示loss（误差）以及是否pass or fail

