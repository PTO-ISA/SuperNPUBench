.global _start
.global _start_c
.type _start,@function
.type _start_c,@function
.set _start_c, _start
.text
_start:
  BSTART CALL, main, ra=_end
_end:
  bstart.std fall
  addi zero, 0x5e, ->a7
  acrc 1
  c.bstop
