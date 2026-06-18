#ifndef ENTRY
#define ENTRY(name) \
    .globl name ; \
    .align 4 ; \
    name:
#endif

/* If symbol ‘name’ is treated as a subroutine (gets called, and returns)
 * then please use ENDPROC to mark ‘name’ as STT_FUNC for the benefit of
 * static analysis tools such as stack depth analyzer.
 */
#ifndef ENDPROC
#define ENDPROC(name) \
    .type name, @function ; \
    .size name, .-name
#endif

#ifndef END
#define END(name) \
    .size name, .-name
#endif

/* 
 * Select code when configured for BE.
 */
#ifdef CONFIG_BIG_ENDIAN
#define CPU_BE(code...) code
#else
#define CPU_BE(code...)
#endif

/* 
 * Select code when configured for LE.
 */
#ifdef CONFIG_BIG_ENDIAN
#define CPU_LE(code...)
#else
#define CPU_LE(code...) code
#endif

#define MASTER_READY          0xaaaaaaaa
#define SECURE_MONITOR        0xfed78943

#define SYNC_SP0              0
#define IRQ_SP0               1
#define FIQ_SP0               2
#define SERR_SP0              3

#define SYNC_SPX              4
#define IRQ_SPX               5
#define FIQ_SPX               6
#define SERR_SPX              7

#define SYNC_L64              8
#define IRQ_L64               9
#define FIQ_L64               10
#define SERR_L64              11

#define SYNC_L32              12
#define IRQ_L32               13
#define FIQ_L32               14
#define SERR_L32              15

#define RBOS_SLV_DEBUG_ADDR 0xe1000000

#define RBOS_STACK_SIZE 0x00000008000000
#define S1_TTBR0_MEM_SIZE 0x00000008000000
#define S2_TTBR0_MEM_SIZE 0x00000008000000
#define RBOS_HEAP_MEM_SIZE 0x00000007000000
#define RBOS_STATIC_MEM_SIZE (RBOS_STACK_SIZE+S1_TTBR0_MEM_SIZE \
                              +S2_TTBR0_MEM_SIZE)
#define RBOS_SYS_MEM_TOTAL_SIZE (RBOS_STATIC_MEM_SIZE+RBOS_HEAP_MEM_SIZE)

#define MONITOR_IMAGE_OFST (RBOS_SYS_MEM_TOTAL_SIZE)
#define MONITOR_TTBR_MEM_SIZE 0x00000008000000

#define PER_CORE_SP_SIZE 0x0000000020000
#define EL3_SP_OFST 0x00000000080000
#define EL2_SP_OFST 0x00000000080000
#define EL1_SP_OFST 0x0000000020000
#define ELO_SP_OFST 0x0000000010000