#ifndef _INCLUDE_TYPE_H_
#define _INCLUDE_TYPE_H_

template <typename E> struct type_traits {
  static constexpr int bits = sizeof(E) * 8;
};

#ifdef __linx
#include <jcore/type.hpp>
#elif defined(__cpu_sim__)
#include <cpu_sim/type.hpp>
#elif defined(__ARM_FEATURE_SME)
#include <aarch64/type.hpp>
#elif defined(__accelerator)
#include <accelerator/v220/type.hpp>
#else
#error "__linx, __cpu_sim__, __ARM_FEATURE_SME, or __accelerator must be defined"
#endif

#endif
