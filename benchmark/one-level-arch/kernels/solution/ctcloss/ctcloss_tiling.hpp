#pragma once
#include "ctcloss_params.hpp"
#include "ctcloss_partition.hpp"

// Host-computed layout and PE ownership, serialized as 24 little-endian U32s.
// The kernel applies this metadata; validation belongs before launch.
struct CtcLossTilingData {
  uint32_t time, batch, classes, max_target_length, max_states, blank, pe_num, reserved0;
  uint32_t lp_time_stride, lp_batch_stride, lp_class_stride, target_stride;
  uint32_t alpha_batch_stride, alpha_time_stride, alpha_state_stride, reserved1;
  uint32_t batch_begin[4], batch_end[4];
};
static_assert(sizeof(CtcLossTilingData)==24*sizeof(uint32_t), "tiling_info ABI");

inline CtcLossTilingData make_ctcloss_tiling(
    const ctc_loss_forward::Arguments &a, uint32_t target_width, uint32_t pe_num) {
  CtcLossTilingData t{a.time,a.batch,a.classes,target_width,a.max_states,a.blank,pe_num,0,
    a.lp_time_stride,a.lp_batch_stride,a.lp_class_stride,a.target_stride,
    a.alpha_batch_stride,a.alpha_time_stride,a.alpha_state_stride,0,{},{}};
  for (uint32_t pe=0;pe<4;++pe) {
    const auto r=pe_num==1
        ? ctc_loss_forward_4pe::BatchRange{pe==0?0u:a.batch,a.batch}
        : ctc_loss_forward_4pe::partition_batch(a.batch,pe);
    t.batch_begin[pe]=r.begin; t.batch_end[pe]=r.end;
  }
  return t;
}

// Caller must validate signed inputs with ctcloss_v2::prepare before launch;
// the corresponding unsigned views retain the original INT32 storage.
inline ctc_loss_forward::Arguments bind_ctcloss_arguments(
    const float *log_probs, const int32_t *targets,
    const int32_t *input_lengths, const int32_t *target_lengths,
    const CtcLossTilingData *t, float *loss, float *alpha,
    const uint32_t *target_offsets) {
  return {log_probs,reinterpret_cast<const uint32_t *>(targets),
    reinterpret_cast<const uint32_t *>(input_lengths),
    reinterpret_cast<const uint32_t *>(target_lengths),target_offsets,
    alpha,loss,t->time,t->batch,t->classes,t->max_states,t->blank,
    t->lp_time_stride,t->lp_batch_stride,t->lp_class_stride,t->target_stride,
    t->alpha_batch_stride,t->alpha_time_stride,t->alpha_state_stride};
}
