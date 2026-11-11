// SSM worker callback for existing int main(void) benchmark entry points.
// Link with the SSM adapter and group worker, not common/_start.s.
// Each PE calls main; shared initialization must happen before the callback.

  .text
  .section .text.__linx_group_pe_main,"ax",@progbits
  .globl __linx_group_pe_main
  .p2align 1
  .type __linx_group_pe_main,@function
__linx_group_pe_main:
  .cfi_startproc
  .pto_pe_domain 15
  // Preserve the worker's return address in this PE's stack frame.
  FENTRY [ra ~ ra], sp!, 256
  .cfi_def_cfa_offset 256
  .cfi_offset ra, -8

  L.BSTART.STD CALL, main
  setret .Lmain_return, ->ra
.Lmain_return:
  C.BSTART.STD
  // Restore the worker return address; preserve main's status in a0.
  FRET.STK [ra ~ ra], sp!, 256
  C.BSTART.STD
  .pto_pe_domain 15
.Lbridge_end:
  .size __linx_group_pe_main, .Lbridge_end-__linx_group_pe_main
  .cfi_endproc
  .section .note.GNU-stack,"",@progbits
