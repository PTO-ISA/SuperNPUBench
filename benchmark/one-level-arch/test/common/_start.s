.global  _start
  .type   _start,@function
  .text
_start:
  bstart.std call main
  c.setret 2, ->ra
.global cross_model_stop
.global _end
.set cross_model_stop, _end
_end:
  bstart.std fall
  addi zero, 0x5e, ->x1
  acrc 1
  c.bstop
