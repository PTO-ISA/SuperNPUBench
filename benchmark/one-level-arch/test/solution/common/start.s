# Direct-entry four-PE test startup. The gfrun ELF loader provides a stack
# arena larger than 128 MiB but this version gives all PEs the same initial SP.
# Reserve disjoint 1 MiB slices before any C++ prologue uses the stack.
.global _start
.type _start,@function
.text
_start:
  bstart.std fall
  ssrget 2050, ->a0
  slli a0, 20, ->a0
  sub sp, a0, ->sp
  c.bstop
  bstart.std call main
  c.setret 2, ->ra
_end:
  bstart.std fall
  addi zero, 0x5e, ->x1
  acrc 1
  c.bstop
