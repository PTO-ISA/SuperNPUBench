ENTRY(_start)
SECTIONS {
  . = 0x10000;
  .text : { *(.text*) }
  . = ALIGN(4096);
  .rodata : { *(.rodata*) }
  . = ALIGN(4096);
  .mx_input : { *(.mx_input) }
  . = ALIGN(4096);
  .mx_output : { *(.mx_output) }
  . = ALIGN(4096);
  .mx_scales : { *(.mx_scales) }
  . = ALIGN(4096);
  .data : { *(.data*) *(.sdata*) }
  .bss : { *(.bss*) *(.sbss*) *(COMMON) }
}
