#include <cstdint>

#include "benchmark.h"
#include "fileop.h"

// 最小复现：linx clang 后端对"均匀（全同且非 0）常量数组"的错误物化。
//   期望：uniform_out = {8,8,8,8}        nonuniform_out = {1,2,3,4}
//   实测：uniform_out = {8,16,8,16}（奇数下标元素 ×2）  nonuniform_out 正确
//
// 参见 issue: https://github.com/LinxISA/SuperScalarModel/issues/1038
// 注意：必须用 res_check=on 构建（否则 CHK_DIR 未定义）。
int main() {
    constexpr std::uint32_t kUniform[4] = {8, 8, 8, 8};     // 触发 bug
    constexpr std::uint32_t kNonUniform[4] = {1, 2, 3, 4};  // 对照组（正确）

    writeBinaryFile(CHK_DIR "/uniform_out.bin",
                    reinterpret_cast<const uint8_t *>(kUniform), sizeof(kUniform));
    writeBinaryFile(CHK_DIR "/nonuniform_out.bin",
                    reinterpret_cast<const uint8_t *>(kNonUniform), sizeof(kNonUniform));
    return 0;
}
