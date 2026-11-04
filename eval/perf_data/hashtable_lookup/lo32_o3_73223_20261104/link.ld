OUTPUT_ARCH(linxv5)
ENTRY(_start)

PHDRS
{
  text PT_LOAD FLAGS(5);
  rodata PT_LOAD FLAGS(4);
  data PT_LOAD FLAGS(6);
}

SECTIONS
{
  . = 0x10000 + SIZEOF_HEADERS;
  .note.pto.isa : ALIGN(4)
  {
    KEEP(*(.note.pto.isa))
  } :text

  .text : ALIGN(16)
  {
    *(.text.startup .text.startup.*)
    *(.text .text.*)
  } :text

  . = ALIGN(4096);
  .rodata :
  {
    *(.rodata .rodata.*)
    *(.srodata .srodata.*)
  } :rodata

  . = ALIGN(4096);
  .data :
  {
    __DATA_BEGIN__ = .;
    *(.data .data.*)
    *(.sdata .sdata.*)
    *(.data.rel.ro .data.rel.ro.*)
  } :data

  .bss (NOLOAD) :
  {
    __bss_start = .;
    *(.bss .bss.*)
    *(.sbss .sbss.*)
    *(COMMON)
    . = ALIGN(64);
    __BSS_END__ = .;
  } :data

  __global_pointer$ = __DATA_BEGIN__ + 0x800;
  _end = .;
  .comment 0 : { *(.comment) }
  /DISCARD/ : { *(.eh_frame*) }
}
