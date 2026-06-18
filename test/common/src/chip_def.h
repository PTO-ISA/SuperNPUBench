#ifndef CHIP_DEF_H
#define CHIP_DEF_H

#ifndef CPU_NUM
#define CPU_NUM 0x1
#endif

#ifndef TASK_EL
#define TASK_EL 3
#endif
#define MEM_BASE 0x60000000
#define FIRMWARE_BASE 0x70000000
#define MEM_END 0xb0000000
#define PERCPU_OFFSET 0x10000
#define PERCPU_BASE (MEM_END - (PERCPU_OFFSET*CPU_NUM))
#define STACK_BASE (MEM_END - (PERCPU_OFFSET*CPU_NUM))
#define STACK_SIZE 0x00900000
#define VECTOR_BASE 0x70001000
#define CODE_BASE 0x70001000
#define PAGE_TABLE_BASE 0x71000000
#define HEAP_HEAD 0x72000000
#define HEAP_SIZE 0x1000000000

#define APP_BASE MEM_BASE
#define APP_STACK_BASE (FIRMWARE_BASE - 0x10000)

#define TUBE_ADDRESS 0x13000000
#define L3_SYS_REG_BASE 0x1f010000

/************************************************************/
#define L1_CACHE_BYTES 64
#endif