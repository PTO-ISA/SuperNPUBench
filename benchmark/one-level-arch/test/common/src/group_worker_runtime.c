/*
 * Lightweight worker entry for gfrun hosted SMT4 multi-PE execution.
 *
 * gfrun (commit accc09b9) sets PE0's PC to the ELF entry (musl _start) and
 * PE1..PE3's PC to this symbol's address.  Each PE also receives an
 * independent 128 MB stack bank, so this function must NOT adjust SP.
 *
 * Workers must not call _exit / exit_group: under the hosted exit semantics
 * any exit_group terminates the entire group, which would kill PE0 before it
 * finishes writing the result file.  Instead, workers park in an infinite
 * loop after main() returns; PE0's libc exit (SYS_exit_group) releases them.
 */
#ifdef __cplusplus
extern "C" {
#endif

int main(void);

__attribute__((noreturn))
void __linx_group_worker_start(void)
{
    main();
    for (;;) {
        __asm__ volatile("" : : : "memory");
    }
}

#ifdef __cplusplus
}
#endif
