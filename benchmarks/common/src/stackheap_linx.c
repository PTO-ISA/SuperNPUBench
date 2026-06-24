#include <stdint.h>
#include "chip_def.h"
#include "sys-sections.h"

char bottom_of_heap __attribute__ ((aligned(32768))) __attribute__ ((section("Heap")));

BOOT_TEXT(uintptr_t sys_all_stacks_size(void));
uintptr_t sys_all_stacks_size(void) {
    return STACK_SIZE * CPU_NUM;
}

BOOT_TEXT(uintptr_t sys_stack_base(void));
uintptr_t sys_stack_base(void) {
    return STACK_BASE;
}

BOOT_TEXT(uintptr_t sys_heap_limit(void));
uintptr_t sys_heap_limit(void) {
    return STACK_BASE - sys_all_stacks_size();
}

BOOT_TEXT(uintptr_t sys_cpu_stack_base(int cpu));
uintptr_t sys_cpu_stack_base(int cpu) {
    return sys_stack_base() - cpu * STACK_SIZE;
}