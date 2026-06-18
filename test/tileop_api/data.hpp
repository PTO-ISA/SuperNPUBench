#ifndef DATA_H
#define DATA_H

#include <iostream>
#include <cmath>
#include "common/type.hpp"

float s_fp32 = 0.1;
__half s_fp16 = 0.1;
int8_t s_i8 = 1;
int16_t s_i16 = 1;
int32_t s_i32 = 1;
int64_t s_i64 = 1;

template <typename T> void init_src_uint(T *aar, uint16_t size) {
  for (uint16_t i = 0; i < size; i++) {
    aar[i] = i + 1;
  }
}

template <typename T> void init_src_int(T *aar, uint16_t size) {
  for (uint16_t i = 0; i < size; i++) {
    aar[i] = -(i + 1);
  }
}
void init_src_int8(int8_t *aar, uint16_t size) {
  for (uint16_t i = 0; i < size; i++) {
    uint16_t val = i % 256;
    if (val != 128) {
      aar[i] = val - 128;
    } else {
      aar[i] = -128;
    }
  }
}

template <typename T> void init_src_fp(T *aar, uint16_t size) {
  for (uint16_t i = 0; i < size; i++) {
    aar[i] = sin((i + 1) / 100.0f);
  }
}

template <typename T> void init_dst(T *aar, uint16_t size) {
  for (uint16_t i = 0; i < size; i++) {
    aar[i] = 0.0;
  }
}

template <typename T> void init_dst_no_zero(T *aar, uint16_t size) {
  for (uint16_t i = 0; i < size; i++) {
    aar[i] = 1.0;
  }
}

template <typename T> void init_index(T *aar, uint16_t row, uint16_t col) {
  for (uint16_t i = 0; i < row; ++i) {
    for (uint16_t j = 0; j < col; ++j) {
      aar[i * col + j] = 0;
    }
  }
}

template <typename T> void init_01(T *aar, uint16_t row, uint16_t col) {
  for (uint16_t i = 0; i < row; ++i) {
    for (uint16_t j = 0; j < col; ++j) {
      if ((i * j) % 2 == 0) {
        aar[i * col + j] = 1;
      } else {
        aar[i * col + j] = 0;
      }
    }
  }
}

template <typename T> void init_rows_fp(T *aar, uint16_t row, uint16_t col) {
  for (uint16_t i = 0; i < row; ++i) {
    for (uint16_t j = 0; j < col; ++j) {
        aar[i * col + j] = (i * col + j) / 100.0f;
    }
  }
}

template <typename T> void OutArray(const T *aar, size_t size) {
  for (uint16_t i = 0; i < size; i++) {
    std::cout << aar[i] << " ";
  }
  std::cout << std::endl;
}
void OutArray(const int8_t *aar, size_t size) {
  for (uint16_t i = 0; i < size; i++) {
    std::cout << static_cast<int32_t>(aar[i]) << " ";
  }
  std::cout << std::endl;
}
void OutArray(const __half *aar, size_t size) {
  for (uint16_t i = 0; i < size; i++) {
    std::cout << static_cast<__fp16>(aar[i]) << " ";
  }
  std::cout << std::endl;
}

// check memory allocation
template <typename T> void check_mem_alloc(const T *p) {
  if (p == nullptr) {
    printf("Memory allocation failed!\n");
    exit(-1);
  }
}

#define INIT_FP32_D0(size)                                                     \
  float *d0 = (float *)malloc(size * sizeof(float));                           \
  check_mem_alloc(d0);                                                         \
  init_dst(d0, size);

#define INIT_FP32_D0_D1(size)                                                  \
  float *d0 = (float *)malloc(size * sizeof(float));                           \
  check_mem_alloc(d0);                                                         \
  init_dst(d0, size);                                                          \
  float *d1 = (float *)malloc(size * sizeof(float));                           \
  check_mem_alloc(d1);                                                         \
  init_dst(d1, size);

#define INIT_FP32_D0_D1_D2(size)                                               \
  float *d0 = (float *)malloc(size * sizeof(float));                           \
  check_mem_alloc(d0);                                                         \
  init_dst(d0, size);                                                          \
  float *d1 = (float *)malloc(size * sizeof(float));                           \
  check_mem_alloc(d1);                                                         \
  init_dst(d1, size);                                                          \
  float *d2 = (float *)malloc(size * sizeof(float));                           \
  check_mem_alloc(d2);                                                         \
  init_dst(d2, size);

#define INIT_FP32_D0_D1_DIFF(size0, size1)                                     \
  float *d0 = (float *)malloc(size0 * sizeof(float));                          \
  check_mem_alloc(d0);                                                         \
  init_dst(d0, size0);                                                         \
  float *d1 = (float *)malloc(size1 * sizeof(float));                          \
  check_mem_alloc(d1);                                                         \
  init_dst(d1, size1);

#define INIT_FP32_D0_D1_D2_DIFF(size0, size1, size2)                           \
  float *d0 = (float *)malloc(size0 * sizeof(float));                          \
  check_mem_alloc(d0);                                                         \
  init_dst(d0, size0);                                                         \
  float *d1 = (float *)malloc(size1 * sizeof(float));                          \
  check_mem_alloc(d1);                                                         \
  init_dst(d1, size1);                                                         \
  float *d2 = (float *)malloc(size2 * sizeof(float));                          \
  check_mem_alloc(d2);                                                         \
  init_dst(d2, size2);

#define INIT_FP32_D0_D1_D2_D3_DIFF(size0, size1, size2, size3)                 \
  float *d0 = (float *)malloc(size0 * sizeof(float));                          \
  check_mem_alloc(d0);                                                         \
  init_dst(d0, size0);                                                         \
  float *d1 = (float *)malloc(size1 * sizeof(float));                          \
  check_mem_alloc(d1);                                                         \
  init_dst(d1, size1);                                                         \
  float *d2 = (float *)malloc(size2 * sizeof(float));                          \
  check_mem_alloc(d2);                                                         \
  init_dst(d2, size2);                                                         \
  float *d3 = (float *)malloc(size3 * sizeof(float));                          \
  check_mem_alloc(d3);                                                         \
  init_dst(d3, size3);

#define FREE_D0 free(d0);

#define FREE_D0_D1                                                             \
  free(d0);                                                                    \
  free(d1);

#define FREE_D0_D1_D2                                                          \
  free(d0);                                                                    \
  free(d1);                                                                    \
  free(d2);

#define FREE_D0_D1_D2_D3                                                       \
  free(d0);                                                                    \
  free(d1);                                                                    \
  free(d2);                                                                    \
  free(d3);

#endif