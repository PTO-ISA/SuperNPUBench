#pragma once
#include "ctcloss_single_pe.hpp"
#include "ctcloss_partition.hpp"
namespace ctc_loss_forward_4pe {
// SPMD: launch on four real contexts with independent stacks and Local Tile
// state. Validate full tensors before launch; inputs remain immutable.
// Each PE owns whole samples: no cross-PE DP reads or per-step barrier.
// The launcher must join all contexts before consumers access the outputs.
// Calling this function four times on PE0 is not a four-PE launch.
inline void forward(const ctc_loss_forward::Arguments &args) {
  const auto range=partition_batch(args.batch,get_thread_idx());
  if (range.begin==range.end) return;
  const auto local=slice_batch(args,range);
  // Each underlying invocation remains single-issuer, including its transfer.
  ctc_loss_forward::forward<16,2>(local);
}
}

inline void ctcloss_v2_four_pe(
    const float *log_probs, const int32_t *targets,
    const int32_t *input_lengths, const int32_t *target_lengths,
    const CtcLossTilingData *tiling_info, float *loss, float *log_alpha,
    const uint32_t *target_offsets) {
  const uint32_t pe=get_thread_idx();
  if (pe>=tiling_info->pe_num) return;
  const ctc_loss_forward_4pe::BatchRange range{
      tiling_info->batch_begin[pe],tiling_info->batch_end[pe]};
  if (range.begin==range.end) return;
  const auto args=bind_ctcloss_arguments(log_probs,targets,input_lengths,
      target_lengths,tiling_info,loss,log_alpha,target_offsets);
  const auto local=ctc_loss_forward_4pe::slice_batch(args,range);
  ctc_loss_forward::forward(local);
}
