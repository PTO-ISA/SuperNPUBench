#pragma once

#include <cstdint>

// Uniform result carrier for ASL/QEMU/cross-model validation.  Every
// microbenchmark publishes its architecture-visible result bytes here before
// returning to the common _end stop label.  The fixed size keeps the sidecar
// ABI stable across dtypes while unused bytes remain zero.
constexpr unsigned long kCrossModelResultBytes = 8192;

extern "C" alignas(64) unsigned char
    cross_model_result[kCrossModelResultBytes] = {};

#ifndef CROSS_MODEL_GOLDEN_HOST
asm(".globl cross_model_result_size\n"
    ".set cross_model_result_size, 8192\n");
#endif

template <typename T>
inline void publish_cross_model_result(const T *source,
                                       unsigned long element_count) {
  const auto *bytes = reinterpret_cast<const unsigned char *>(source);
  unsigned long byte_count = element_count * sizeof(T);
  if (byte_count > kCrossModelResultBytes)
    byte_count = kCrossModelResultBytes;
  for (unsigned long index = 0; index < byte_count; ++index)
    cross_model_result[index] = bytes[index];
}

template <typename T>
inline void publish_cross_model_scalar(T value) {
  publish_cross_model_result(&value, 1);
}
