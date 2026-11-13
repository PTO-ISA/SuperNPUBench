#pragma once
#include "ctcloss_params.hpp"
namespace ctc_loss_forward_4pe {
constexpr uint32_t pe_count = 4;
struct BatchRange { uint32_t begin, end; };
// Split sample pairs into disjoint contiguous ranges, including idle PEs.
inline BatchRange partition_batch(uint32_t batch, uint32_t pe) {
  if (pe>=pe_count) return {batch,batch};
  const uint64_t groups=(uint64_t(batch)+1)/2, q=groups/4, r=groups%4;
  const uint64_t first=q*pe+(pe<r ? pe : r), last=first+q+(pe<r ? 1:0);
  return {uint32_t(first*2<batch ? first*2:batch),
          uint32_t(last*2<batch ? last*2:batch)};
}
// Requires validated full tensors and a nonempty range. Offsets still refer
// to the original targets base; other batch-dependent pointers are rebased.
inline ctc_loss_forward::Arguments slice_batch(
    const ctc_loss_forward::Arguments &a, BatchRange range) {
  auto local=a;
  local.batch=range.end-range.begin;
  local.input_lengths+=range.begin;
  local.target_lengths+=range.begin;
  local.target_offsets+=range.begin;
  local.negative_log_likelihood+=range.begin;
  if (a.time) {
    local.log_probs+=uint64_t(range.begin)*a.lp_batch_stride;
    local.log_alpha+=uint64_t(range.begin)*a.alpha_batch_stride;
  }
  return local;
}
}
