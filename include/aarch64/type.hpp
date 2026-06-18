#ifndef __ARM_SME_TYPE_HPP__
#define __ARM_SME_TYPE_HPP__

#include <stdint.h>

#ifdef __clang__
#if __clang_major__ >= 15
typedef _Float16 __half;
#endif
#endif

#if defined(__GNUC__) && !defined(__clang__)
#if defined(__aarch64__) && defined(__ARM_NEON)
#include "arm_fp16.h"
typedef __fp16 __half;
#endif
#endif

#ifdef __ARM_FEATURE_BF16
#include "arm_bf16.h"
#endif

typedef float __fp32;

#endif