# Element-wise 原子直方图

本测试在改用公共 `ElementTile` API 的同时，保持原始 128-element 原子直方图
workload 不变。输入 value 是 128 个值为 1 的 S32 元素，S32 index 为
`element & 7`，8 个 S32 bin 初始为 0，128 个 byte 谓词全为 1，128 个 U32
观测值初始为 0。

kernel 将 value 和 index 加载到 `ElementTile<int32_t, 128>`，通过
`TPARTVIEW<32>` 将其分为四个 part，再用 `TPARTELEMENT` 取得各 part 的逻辑
element 引用。前四个 `#pragma pto element for` 循环执行普通 relaxed
`__atomic_fetch_add`。只有四个原子 part 全部完成后，后四个独立的标记循环才会
读取 bin。激活的 element 读取其 index 指定的 bin；禁用的 element 返回零
padding seed，且不发起内存读取。gather 结果通过 `TSTORE` 写入
`elementwise_atomic_histogram_observed`。

原始 byte 谓词仍是对外导出的输入。公共 element API 的 element carrier 支持
U32、S32 和 F32，因此 `main` 在计时的 kernel 调用前，用普通 C++ 将 byte
谓词拓宽到私有 U32 transport 数组。

标量检查要求每个 bin 和每个观测值都等于 16，同时确认 value、index 和 byte
谓词保持初始化值。任一检查不一致时，`main` 通过自然返回值报告失败。
