.global  _start
  .type   _start,@function
  .text
_start:
  # The model initializes every PE with the same SP. Reserve 1 MiB per PE.
  bstart.std fall
  ssrget 2050, ->t
  or zero, t#1.sw, ->a0
  slli a0, 20, ->t
  sub sp, t#1, ->sp
  bstart.std call main
  c.setret 2, ->ra
_end:
  bstart.std fall
  addi zero, 0x5e, ->x1
  acrc 1
  c.bstop
