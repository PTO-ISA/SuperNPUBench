.global  _start
  .type   _start,@function
  .text
_start:
  bstart.std call main
  c.setret 2, ->ra
_end:
  bstart.aux fall
  acrc 0
  c.bstop