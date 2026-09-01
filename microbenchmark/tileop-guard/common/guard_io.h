#ifndef TILEOP_GUARD_IO_H
#define TILEOP_GUARD_IO_H
// Host-side input generation + raw binary dump for the res_check precision path.
//
// These live in guard_io.c, compiled WITHOUT the matrix flags
// (-fenable-matrix / -enable-all-vector-as-tilereg). That isolation is load-
// bearing: when the fill loops are compiled with the matrix flags they get
// promoted to tile registers and the host arrays degenerate (e.g. a[] collapses
// to a single non-zero element), producing weak/degenerate test vectors. The
// dumper likewise avoids libc printf (writeBinaryFile's trailing printf/fflush
// hangs gfrun); raw open/write/close via the emulator syscall passthrough works.
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

// Deterministic, varied fills (stay scalar; safe to dump + recompute host-side).
void guard_fill_seq_f32(float* p, int n, float base, float step);
void guard_fill_const_f32(float* p, int n, float v);
void guard_fill_seq_i32(int32_t* p, int n, int32_t base, int32_t step);
void guard_fill_seq_f16(uint16_t* p, int n, float base, float step);  // raw __half bits

// Raw binary dump (no printf). Silently no-ops on open failure.
void guard_dump_bin(const char* path, const void* p, size_t bytes);

// Raw binary read (host-generated inputs). Silently no-ops on open failure.
void guard_read_bin(const char* path, void* p, size_t bytes);

#ifdef __cplusplus
}
#endif
#endif  // TILEOP_GUARD_IO_H
