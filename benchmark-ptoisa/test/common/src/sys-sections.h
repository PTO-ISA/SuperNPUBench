#ifndef _SYS_SECTIONS_H_
#define _SYS_SECTIONS_H_

#ifdef SVE
#define BOOT_TEXT(...)
#else
#define BOOT_TEXT(...) __VA_ARGS__ __attribute__((section("boot_text")))
#endif

#define BOOT_DATA(...) __VA_ARGS__ __attribute__((section("boot_data")))
// No special BSS section for boot code since BSS is initialized by the startup code based on linker-generated labels
#define BOOT_BSS(...) __VA_ARGS__

#endif