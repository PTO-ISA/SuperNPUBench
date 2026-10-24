#include "solution/ctcloss/ctcloss_v2_i32.hpp"
#include "solution/ctcloss/ctcloss_four_pe.hpp"
#include "data.hpp"
#ifndef CTC_PE_COUNT
#define CTC_PE_COUNT 1
#endif
static_assert(CTC_PE_COUNT==1 || CTC_PE_COUNT==4);
extern "C" { alignas(128) uint32_t ctc_status[CTC_PE_COUNT*4]; }
int main() {
  const uint32_t pe=get_thread_idx();
  if (pe>=CTC_PE_COUNT) return 1;
  uint32_t offsets[kBatch];
  ctcloss_v2::PaddedI32Arguments input{
    ctc_lp,ctc_targets,ctc_lengths,ctc_target_lengths,
    ctc_alpha+4,ctc_loss+4,kTime,kBatch,kClasses,kTargetWidth,0,
    ctcloss_v2::Reduction::none,false};
  ctc_loss_forward::Arguments core{};
  const auto error=ctcloss_v2::prepare(input,kInputElements,kTargetElements,
      kBatch,kBatch,kAlpha-8,kLoss-8,offsets,kBatch,core);
  if (error!=ctcloss_v2::InputError::ok) {
    ctc_status[pe*4+2]=uint32_t(error);
    return 1;
  }
  uint32_t samples=kBatch;
  if constexpr (CTC_PE_COUNT==4) {
    ctc_loss_forward_4pe::forward(core);
    const auto range=ctc_loss_forward_4pe::partition_batch(kBatch,pe);
    samples=range.end-range.begin;
  } else {
    ctc_loss_forward::forward(core);
  }
  ctc_status[pe*4]=1;
  ctc_status[pe*4+1]=pe+1;
  ctc_status[pe*4+3]=samples;
  return 0;
}
