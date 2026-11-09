.global  _start
  .type   _start,@function
  .text
_start:
  bstart.std call main
  # C.SETRET encodes halfwords: +1 reaches the next 2-byte instruction.
  c.setret 1, ->ra
_end:
  # ACRC is a SYS-only operation under the current PTO ISA.
  c.bstart.sys fall
  addi zero, 0x5e, ->x1
  acrc 1
  c.bstop
