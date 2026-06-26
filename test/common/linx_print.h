#ifndef LINX_PRINT_H
#define LINX_PRINT_H

/*
 * Minimal, syscall-free dump utility for Linx targets ("linxi_put").
 *
 * All output goes straight to the virt UART data register (MMIO @ 0x10000000).
 * That single channel is observed by BOTH back-ends we run benchmarks on:
 *
 *   - qemu-system-linx64 -M virt        -> bytes appear on the console.
 *   - gfsim (LinxCoreModel CA model)    -> SimSys::observeUartWrite() buffers a
 *                                          line and prints "linx_uart: <line>"
 *                                          on each '\n' (model/SimSys.cpp).
 *
 * Why not printf?  printf pulls in musl libc, whose write()/exit() lower to the
 * `acrc 1` (SCT_SYS) syscall path. That path is not usable under gfsim, and the
 * freestanding _start.s exit uses `acrc 3` (== ACRC_REQUEST_TYPE::SCT_INVAL,
 * isa/MInst.h) which trips a model assertion. The raw UART MMIO write below
 * needs no syscalls, so it works in freestanding / baremetal builds as well.
 *
 * NOTE: gfsim only flushes a line on '\n', so every helper that emits a record
 * terminates its line with '\n'. Use linxi_put() for fragments and finish the
 * line yourself with linxi_putc('\n') (or use the *_kv / *s helpers).
 */

#define LINX_VIRT_UART_BASE 0x10000000u

static inline void linxi_putc(char c)
{
    volatile unsigned char *uart =
        (volatile unsigned char *)(unsigned long)LINX_VIRT_UART_BASE;
    *uart = (unsigned char)c;
}

/* Raw string, no trailing newline. */
static inline void linxi_put(const char *s)
{
    while (*s) {
        linxi_putc(*s++);
    }
}

/* String followed by newline (flushes the line on gfsim). */
static inline void linxi_puts(const char *s)
{
    linxi_put(s);
    linxi_putc('\n');
}

/* Signed decimal integer, no trailing newline. */
static inline void linxi_put_i64(long long v)
{
    char buf[24];
    int i = 0;
    unsigned long long u;

    if (v < 0) {
        linxi_putc('-');
        u = (unsigned long long)(-(v + 1)) + 1ULL; /* avoid INT64_MIN overflow */
    } else {
        u = (unsigned long long)v;
    }

    do {
        buf[i++] = (char)('0' + (int)(u % 10ULL));
        u /= 10ULL;
    } while (u);

    while (i) {
        linxi_putc(buf[--i]);
    }
}

/* Unsigned hex with 0x prefix, no trailing newline. */
static inline void linxi_put_hex(unsigned long long v)
{
    static const char hexd[] = "0123456789abcdef";
    char buf[16];
    int i = 0;

    linxi_put("0x");
    do {
        buf[i++] = hexd[v & 0xFULL];
        v >>= 4;
    } while (v);

    while (i) {
        linxi_putc(buf[--i]);
    }
}

/* "label = <dec>\n" convenience dump. */
static inline void linxi_put_kv(const char *label, long long v)
{
    linxi_put(label);
    linxi_put(" = ");
    linxi_put_i64(v);
    linxi_putc('\n');
}

/* Back-compat aliases for earlier callers that used the linx_* names. */
static inline void linx_putc(char c)         { linxi_putc(c); }
static inline void linx_print(const char *s) { linxi_put(s); }
static inline void linx_puts(const char *s)  { linxi_puts(s); }

#endif /* LINX_PRINT_H */
