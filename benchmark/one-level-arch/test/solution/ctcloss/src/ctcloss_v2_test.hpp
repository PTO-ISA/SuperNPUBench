#pragma once
#include <cstdint>
#include <cstring>
#include <type_traits>
#include "fileop.h"
#include "benchmark.h"
#include "multi_thread_res_check.h"
#pragma push_macro("DType")
#undef DType
#include "solution/ctcloss/ctcloss_v2_i32.hpp"
#include "solution/ctcloss/ctcloss_four_pe.hpp"
#pragma pop_macro("DType")

#ifndef DType
#define DType float
#endif
#ifndef PE_NUM
#error "PE_NUM must be supplied by the testcase Makefile"
#endif
#ifndef CHK_DIR
#error "CHK_DIR must identify the generated network inputs"
#endif
static_assert(std::is_same<DType,float>::value, "Ctcloss_v2 profile is FP32");
static_assert(PE_NUM==1 || PE_NUM==4, "single or four PE only");
constexpr uint32_t kTime=195,kBatch=256,kClasses=41,kTargetWidth=74,kStates=149;
constexpr uint32_t kInputElements=kTime*kBatch*kClasses;
constexpr uint32_t kTargetElements=kBatch*kTargetWidth;
constexpr uint32_t kAlphaElements=kBatch*kTime*kStates;
constexpr float kGuard=-987.25f;

namespace {
alignas(32) float log_probs_buf[kInputElements];
alignas(32) int32_t targets_buf[kTargetElements];
alignas(32) int32_t input_lengths_buf[kBatch],target_lengths_buf[kBatch];
alignas(32) uint32_t target_offsets_buf[kBatch];
alignas(32) CtcLossTilingData tiling_info;
MultiThreadResCheckSync res_sync{};
volatile uint32_t input_error=0;
}
extern "C" {
alignas(4096) float ctcloss_alpha[kAlphaElements+8];
alignas(4096) float ctcloss_loss[kBatch+8];
alignas(128) uint32_t ctcloss_status[PE_NUM][4];
}

int main() {
  const uint32_t tid=get_thread_idx();
  if (tid>=PE_NUM) return 1;
  float *alpha=ctcloss_alpha+4,*loss=ctcloss_loss+4;
  if (tid==0) {
    // Lengths and labels affect work; load the same network inputs even in
    // timing builds. All I/O and launch validation precede TRACE.begin.
    const bool loaded=
      readBinaryFile(CHK_DIR "/log_probs.bin",reinterpret_cast<uint8_t *>(log_probs_buf),sizeof(log_probs_buf)) &&
      readBinaryFile(CHK_DIR "/targets.bin",reinterpret_cast<uint8_t *>(targets_buf),sizeof(targets_buf)) &&
      readBinaryFile(CHK_DIR "/input_lengths.bin",reinterpret_cast<uint8_t *>(input_lengths_buf),sizeof(input_lengths_buf)) &&
      readBinaryFile(CHK_DIR "/target_lengths.bin",reinterpret_cast<uint8_t *>(target_lengths_buf),sizeof(target_lengths_buf)) &&
      readBinaryFile(CHK_DIR "/tiling_info.bin",reinterpret_cast<uint8_t *>(&tiling_info),sizeof(tiling_info));
    if (!loaded) input_error=1;
    if (!input_error) {
      ctcloss_v2::PaddedI32Arguments input{
        log_probs_buf,targets_buf,input_lengths_buf,target_lengths_buf,alpha,loss,
        kTime,kBatch,kClasses,kTargetWidth,0,ctcloss_v2::Reduction::none,false};
      ctc_loss_forward::Arguments args{};
      if (ctcloss_v2::prepare(input,kInputElements,kTargetElements,kBatch,kBatch,
          kAlphaElements,kBatch,target_offsets_buf,kBatch,args)!=ctcloss_v2::InputError::ok) {
        input_error=2;
      } else {
        const auto expected=make_ctcloss_tiling(args,kTargetWidth,PE_NUM);
        if (std::memcmp(&expected,&tiling_info,sizeof(expected))!=0) input_error=3;
      }
    }
#ifdef RES_CHECK
    if (!input_error) {
      for (uint32_t i=0;i<kAlphaElements+8;++i) ctcloss_alpha[i]=kGuard;
      for (uint32_t i=0;i<kBatch+8;++i) ctcloss_loss[i]=kGuard;
    }
#endif
  }
  res_check_publish_inputs(res_sync,tid);
  if (input_error) return 1;

  BENCHSTART;
  if constexpr (PE_NUM==1)
    ctcloss_v2_single_pe(log_probs_buf,targets_buf,input_lengths_buf,
        target_lengths_buf,&tiling_info,loss,alpha,target_offsets_buf);
  else
    ctcloss_v2_four_pe(log_probs_buf,targets_buf,input_lengths_buf,
        target_lengths_buf,&tiling_info,loss,alpha,target_offsets_buf);
  BENCHEND;

  ctcloss_status[tid][0]=1;
  ctcloss_status[tid][1]=tid+1;
  ctcloss_status[tid][2]=0;
  ctcloss_status[tid][3]=tiling_info.batch_end[tid]-tiling_info.batch_begin[tid];
  if constexpr (PE_NUM==4) res_check_wait_for_all(res_sync,tid);
#ifdef RES_CHECK
  if (tid==0) {
    float guards[16];
    for (uint32_t i=0;i<4;++i) {
      guards[i]=ctcloss_alpha[i]; guards[4+i]=ctcloss_alpha[kAlphaElements+4+i];
      guards[8+i]=ctcloss_loss[i]; guards[12+i]=ctcloss_loss[kBatch+4+i];
    }
    const bool written=
      writeBinaryFile(CHK_DIR "/loss.bin",reinterpret_cast<const uint8_t *>(loss),kBatch*sizeof(float)) &&
      writeBinaryFile(CHK_DIR "/log_alpha.bin",reinterpret_cast<const uint8_t *>(alpha),kAlphaElements*sizeof(float)) &&
      writeBinaryFile(CHK_DIR "/status.bin",reinterpret_cast<const uint8_t *>(ctcloss_status),sizeof(ctcloss_status)) &&
      writeBinaryFile(CHK_DIR "/output_guards.bin",reinterpret_cast<const uint8_t *>(guards),sizeof(guards));
    if (!written) return 1;
    puts("CTCLOSS_OUTPUT_WRITTEN");
  }
#endif
  // Hosted workers return to the common park entry; only PE0 exits libc.
  return 0;
}
