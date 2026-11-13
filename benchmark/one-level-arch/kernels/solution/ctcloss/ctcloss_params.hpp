#ifndef PTO_CTC_LOSS_FORWARD_PARAMS_HPP
#define PTO_CTC_LOSS_FORWARD_PARAMS_HPP
#include <cstddef>
#include <cstdint>

namespace ctc_loss_forward {
struct Arguments {
  const float *log_probs;
  const uint32_t *targets, *input_lengths, *target_lengths, *target_offsets;
  float *log_alpha, *negative_log_likelihood;
  uint32_t time, batch, classes, max_states, blank;
  uint32_t lp_time_stride, lp_batch_stride, lp_class_stride, target_stride;
  uint32_t alpha_batch_stride, alpha_time_stride, alpha_state_stride;
};
enum class InputError { ok, dimensions, pointer, lengths, label, storage, stride, offset_overflow };

// Runs before launch, on the host (or once in a standalone target test).
// All capacities count elements, including stride padding. Output tensors
// must not alias inputs or each other. No data is copied and no DP is done.
inline InputError validate(const Arguments &a, uint64_t lp_capacity,
                           uint64_t target_capacity, uint64_t alpha_capacity,
                           uint64_t loss_capacity) {
  if (!a.classes || a.blank >= a.classes || !a.max_states)
    return InputError::dimensions;
  if (!a.batch) return InputError::ok;
  if (!a.input_lengths || !a.target_lengths || !a.target_offsets ||
      !a.negative_log_likelihood || (a.time && (!a.log_probs || !a.log_alpha)))
    return InputError::pointer;
  if (loss_capacity < a.batch) return InputError::storage;
  if (a.time) {
    const uint64_t state_span = uint64_t(a.max_states-1)*a.alpha_state_stride+1;
    if (state_span>UINT32_MAX) return InputError::offset_overflow;
    const uint64_t sample_span = uint64_t(a.time-1)*a.alpha_time_stride+state_span;
    if (sample_span>UINT32_MAX) return InputError::offset_overflow;
    if (!a.alpha_state_stride || (a.time>1 && a.alpha_time_stride<state_span) ||
        (a.batch>1 && a.alpha_batch_stride<sample_span)) return InputError::stride;
    const uint64_t alpha_span = uint64_t(a.batch-1)*a.alpha_batch_stride+sample_span;
    const uint64_t lp_t = uint64_t(a.time-1)*a.lp_time_stride;
    const uint64_t lp_b = uint64_t(a.batch-1)*a.lp_batch_stride;
    const uint64_t lp_c = uint64_t(a.classes-1)*a.lp_class_stride;
    // Check products before adding: three valid uint64 products can wrap
    // their sum and otherwise make an out-of-range tensor look small.
    if (lp_t>UINT32_MAX || lp_b>UINT32_MAX || lp_c>UINT32_MAX)
      return InputError::offset_overflow;
    const uint64_t lp_span = lp_t+lp_b+lp_c+1;
    if (alpha_span>UINT32_MAX || lp_span>UINT32_MAX) return InputError::offset_overflow;
    if (alpha_span>alpha_capacity || lp_span>lp_capacity) return InputError::storage;
  }
  for (uint32_t b=0;b<a.batch;++b) {
    const uint32_t u=a.target_lengths[b];
    if (a.input_lengths[b]>a.time || uint64_t(u)*2+1>a.max_states)
      return InputError::lengths;
    if (u) {
      if (!a.targets) return InputError::pointer;
      const uint64_t end=uint64_t(a.target_offsets[b])+uint64_t(u-1)*a.target_stride+1;
      if (end>UINT32_MAX) return InputError::offset_overflow;
      if (end>target_capacity) return InputError::storage;
      for (uint32_t i=0;i<u;++i)
        if (a.targets[a.target_offsets[b]+i*a.target_stride]>=a.classes)
          return InputError::label;
    }
  }
  return InputError::ok;
}

// Packed targets: prefix sum of lengths times target_stride. Padded targets:
// b*batch_stride. Both layouts use offsets in physical element units.
inline bool make_target_offsets(const uint32_t *lengths, uint32_t batch,
                                bool packed, uint32_t target_stride,
                                uint32_t batch_stride, uint32_t *offsets) {
  if (batch && (!lengths || !offsets)) return false;
  uint64_t prefix=0;
  for (uint32_t b=0;b<batch;++b) {
    const uint64_t offset=packed ? prefix : uint64_t(b)*batch_stride;
    if (offset>UINT32_MAX) return false;
    offsets[b]=uint32_t(offset);
    if (packed) {
      prefix+=uint64_t(lengths[b])*target_stride;
      if (prefix>UINT32_MAX) return false;
    }
  }
  return true;
}
} // namespace ctc_loss_forward
#endif
