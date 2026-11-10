#include "pipeline.hpp"
#include "split.hpp"
#include "benchmark.h"
#ifdef RES_CHECK
#include "fileop.h"
#include "multi_thread_res_check.h"
#endif

namespace {
#ifdef RES_CHECK
alignas(4096) __bf16 input[512*256];
alignas(4096) __fp8_e4m3 output[512*256];
alignas(4096) __fp8_e8m0 scales[512*8];
MultiThreadResCheckSync res_sync;
#endif
}
#ifndef RES_CHECK
// Embed host-generated nonzero input, so exactly the timed ELF can be checked
// in both engines without scalar stores, hosted syscalls or barrier loops.
extern "C" const __bf16 input[512*256];
asm(".pushsection .mx_input,\"aw\",@progbits\n.balign 4096\n.global input\ninput:\n.incbin \"" INPUT_FILE "\"\n.popsection\n");
extern "C" {
alignas(4096) __attribute__((section(".mx_output"))) __fp8_e4m3 output[512*256];
alignas(4096) __attribute__((section(".mx_scales"))) __fp8_e8m0 scales[512*8];
}
#endif
int main() {
  const auto tid=get_thread_idx();
#ifdef RES_CHECK
  if(tid==0)readBinaryFile(CHK_DIR "/input.bin",reinterpret_cast<uint8_t*>(input),sizeof(input));
  res_check_publish_inputs(res_sync,tid);
#endif
  BENCHSTART;
#if VARIANT == 0
  mxquant::run(output,scales,input);
#elif VARIANT == 1
  mxquant_bulk::run<false>(output,scales,input);
#elif VARIANT == 2
  mxquant_bulk::run<true>(output,scales,input);
#elif VARIANT == 3
  mxquant_bulk::run<false,true>(output,scales,input);
#elif VARIANT == 4
  mxquant_bulk::run<true,true>(output,scales,input);
#elif VARIANT == 7
  mxquant_split::run<2>(output,scales,input);
#elif VARIANT == 8
  mxquant_split::run<4>(output,scales,input);
#else
  mxquant_pipeline::run<VARIANT>(output,scales,input);
#endif
  BENCHEND;
#ifdef RES_CHECK
  res_check_wait_for_all(res_sync,tid);
  if(tid==0) {
    writeBinaryFile(CHK_DIR "/input_readback.bin",reinterpret_cast<uint8_t*>(input),sizeof(input));
    writeBinaryFile(CHK_DIR "/output.bin",reinterpret_cast<uint8_t*>(output),sizeof(output));
    writeBinaryFile(CHK_DIR "/scale_output.bin",reinterpret_cast<uint8_t*>(scales),sizeof(scales));
  }
#endif
  return 0;
}
