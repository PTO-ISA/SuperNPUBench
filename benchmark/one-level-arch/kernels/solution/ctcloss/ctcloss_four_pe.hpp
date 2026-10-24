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
