.global  _start
  .type   _start,@function
  .text
_start:
  bstart.std call main
  c.setret 2, ->ra
_end:
  bstart.sys fall
  addi zero, 0x5e, ->x1
  acrc 1
  c.bstop
