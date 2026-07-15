
.section linxboot_text, "ax"

.extern _linx_start

_start:
  bstart.std call _linx_start
  c.setret 2, ->ra
_end:
  bstart.std fall
  addi zero, 0x5e, ->x1
  acrc 1
  c.bstop
