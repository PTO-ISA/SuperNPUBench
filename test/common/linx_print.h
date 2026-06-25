#ifndef LINX_PRINT_H
#define LINX_PRINT_H

/*
 * Minimal print utility for QEMU virt (UART @ 0x10000000).
 * No syscalls, no buffering — direct MMIO writes.
 * Works with both freestanding (-nostdlib) and hosted builds.
 */

#define LINX_VIRT_UART_BASE 0x10000000u

static inline void linx_putc(char c)
{
    volatile unsigned char *uart = (volatile unsigned char *)LINX_VIRT_UART_BASE;
    *uart = (unsigned char)c;
}

static inline void linx_puts(const char *s)
{
    while (*s) {
        linx_putc(*s++);
    }
    linx_putc('\n');
}

static inline void linx_print(const char *s)
{
    while (*s) {
        linx_putc(*s++);
    }
}

#endif /* LINX_PRINT_H */
