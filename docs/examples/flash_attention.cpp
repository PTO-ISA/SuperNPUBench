#include <cstdint>

#include <pto/linx/AutoModeKernels.hpp>

extern "C" void flash_attention_probe(
    std::int32_t *out,
    const std::int32_t *q,
    const std::int32_t *k,
    const std::int32_t *v) {
    pto::linx::auto_mode::flash_attention_kernel_i32(q, k, v, out);
}
