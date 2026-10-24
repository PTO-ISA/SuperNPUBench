#pragma once
#include "ctcloss_params.hpp"

namespace ctcloss_v2 {
enum class Reduction { none, sum, mean };
struct PaddedI32Arguments {
  const float *log_probs;
  const int32_t *targets, *input_lengths, *target_lengths;
  float *log_alpha, *negative_log_likelihood;
  uint32_t time, batch, classes, max_target_length, blank;
  Reduction reduction = Reduction::none;
  bool zero_infinity = false;
};
enum class InputError { ok, attributes, shape, pointer, storage, negative_integer, kernel_contract };

// Adapt contiguous [T,N,C] / padded [N,U] signed INT32 inputs without changing
// their stored dtype. C++ permits access through the corresponding unsigned
// type; signed values are checked BEFORE passing views to the U32 core.
// Caller owns N offset words, retained until the kernel has completed.
inline InputError prepare(const PaddedI32Arguments &a,
    uint64_t lp_capacity, uint64_t target_capacity,
    uint64_t input_length_capacity, uint64_t target_length_capacity,
    uint64_t alpha_capacity, uint64_t loss_capacity,
    uint32_t *offsets, uint64_t offset_capacity,
    ctc_loss_forward::Arguments &result) {
  if (a.reduction != Reduction::none || a.zero_infinity)
    return InputError::attributes;
  if (!a.classes || a.blank>=a.classes || a.max_target_length>(UINT32_MAX-1)/2)
    return InputError::shape;
  const uint64_t states=uint64_t(a.max_target_length)*2+1;
  const uint64_t lp_row=uint64_t(a.batch)*a.classes;
  const uint64_t alpha_row=uint64_t(a.time)*states;
  // Bound factors before multiplication, retaining the core's 32-bit offsets.
  if (lp_row>UINT32_MAX || alpha_row>UINT32_MAX ||
      uint64_t(a.time)*lp_row>UINT32_MAX ||
      uint64_t(a.batch)*alpha_row>UINT32_MAX ||
      uint64_t(a.batch)*a.max_target_length>UINT32_MAX)
    return InputError::shape;
  if (lp_capacity<uint64_t(a.time)*lp_row ||
      target_capacity<uint64_t(a.batch)*a.max_target_length ||
      input_length_capacity<a.batch || target_length_capacity<a.batch ||
      alpha_capacity<uint64_t(a.batch)*alpha_row ||
      loss_capacity<a.batch || offset_capacity<a.batch)
    return InputError::storage;
  if (a.batch && (!a.input_lengths || !a.target_lengths || !offsets ||
      !a.negative_log_likelihood ||
      (a.max_target_length && !a.targets) ||
      (a.time && (!a.log_probs || !a.log_alpha))))
    return InputError::pointer;
  for (uint32_t b=0;b<a.batch;++b) {
    if (a.input_lengths[b]<0 || a.target_lengths[b]<0)
      return InputError::negative_integer;
    if (uint32_t(a.input_lengths[b])>a.time ||
        uint32_t(a.target_lengths[b])>a.max_target_length)
      return InputError::shape;
    const uint32_t offset=b*a.max_target_length;
    for (int32_t i=0;i<a.target_lengths[b];++i) {
      const int32_t label=a.targets[offset+uint32_t(i)];
      if (label<0) return InputError::negative_integer;
      if (uint32_t(label)>=a.classes) return InputError::shape;
    }
    offsets[b]=offset;
  }
  ctc_loss_forward::Arguments core{
    a.log_probs,reinterpret_cast<const uint32_t *>(a.targets),
    reinterpret_cast<const uint32_t *>(a.input_lengths),
    reinterpret_cast<const uint32_t *>(a.target_lengths),offsets,
    a.log_alpha,a.negative_log_likelihood,a.time,a.batch,a.classes,
    uint32_t(states),a.blank,uint32_t(lp_row),a.classes,1,1,
    uint32_t(alpha_row),uint32_t(states),1};
  if (ctc_loss_forward::validate(core,lp_capacity,target_capacity,alpha_capacity,loss_capacity)
      !=ctc_loss_forward::InputError::ok) return InputError::kernel_contract;
  result=core;
  return InputError::ok;
}
}
