#ifndef TILEOP_GUARD_CASE_HPP
#define TILEOP_GUARD_CASE_HPP
// res_check case macros. Each macro emits the whole demo body: file-scope
// buffers (globals resist tile-register promotion) + main() that reads
// host-generated inputs (CHK_DIR/in_*.bin) via read() syscall, runs the tile op
// through a guard_common.hpp driver, and dumps CHK_DIR/out.bin for golden.py.
// The op is passed as a lambda in __VA_ARGS__ (internal commas are fine).
// See guard_io.h / project memory for why inputs are host-owned and dumps are
// printf-free.
#include "guard_common.hpp"
#include "guard_io.h"

#define GUARD_BINARY(DT, M, N, ...) \
  static DT gA[(M) * (N)], gB[(M) * (N)], gC[(M) * (N)]; \
  int main() { \
    guard_read_bin(CHK_DIR "/in_a.bin", gA, sizeof(gA)); \
    guard_read_bin(CHK_DIR "/in_b.bin", gB, sizeof(gB)); \
    BENCHSTART; g_binary<DT, M, N>(gC, gA, gB, __VA_ARGS__); BENCHEND; \
    guard_dump_bin(CHK_DIR "/out.bin", gC, sizeof(gC)); \
    return 0; }

#define GUARD_SCALAR(DT, M, N, SVAL, ...) \
  static DT gA[(M) * (N)], gC[(M) * (N)]; \
  int main() { \
    guard_read_bin(CHK_DIR "/in_a.bin", gA, sizeof(gA)); \
    BENCHSTART; g_scalar<DT, M, N>(gC, gA, (DT)(SVAL), __VA_ARGS__); BENCHEND; \
    guard_dump_bin(CHK_DIR "/out.bin", gC, sizeof(gC)); \
    return 0; }

#define GUARD_UNARY(DT, M, N, ...) \
  static DT gA[(M) * (N)], gC[(M) * (N)]; \
  int main() { \
    guard_read_bin(CHK_DIR "/in_a.bin", gA, sizeof(gA)); \
    BENCHSTART; g_unary<DT, M, N>(gC, gA, __VA_ARGS__); BENCHEND; \
    guard_dump_bin(CHK_DIR "/out.bin", gC, sizeof(gC)); \
    return 0; }

#define GUARD_UNARY_CVT(DIN, DOUT, M, N, ...) \
  static DIN gA[(M) * (N)]; static DOUT gC[(M) * (N)]; \
  int main() { \
    guard_read_bin(CHK_DIR "/in_a.bin", gA, sizeof(gA)); \
    BENCHSTART; g_unary_cvt<DIN, DOUT, M, N>(gC, gA, __VA_ARGS__); BENCHEND; \
    guard_dump_bin(CHK_DIR "/out.bin", gC, sizeof(gC)); \
    return 0; }

#define GUARD_TERNARY(DT, M, N, ...) \
  static DT gA[(M) * (N)], gB[(M) * (N)], gD[(M) * (N)], gC[(M) * (N)]; \
  int main() { \
    guard_read_bin(CHK_DIR "/in_a.bin", gA, sizeof(gA)); \
    guard_read_bin(CHK_DIR "/in_b.bin", gB, sizeof(gB)); \
    guard_read_bin(CHK_DIR "/in_c.bin", gD, sizeof(gD)); \
    BENCHSTART; g_ternary<DT, M, N>(gC, gA, gB, gD, __VA_ARGS__); BENCHEND; \
    guard_dump_bin(CHK_DIR "/out.bin", gC, sizeof(gC)); \
    return 0; }

// reduce: src M x N -> dst physical M x N with valid axis = 1 (row: ValidCol=1,
// col: ValidRow=1). golden.py compares only the valid footprint.
#define GUARD_ROWREDUCE(DT, M, N, ...) \
  static DT gA[(M) * (N)], gC[(M) * (N)]; \
  int main() { \
    guard_read_bin(CHK_DIR "/in_a.bin", gA, sizeof(gA)); \
    BENCHSTART; g_rowreduce<DT, M, N>(gC, gA, __VA_ARGS__); BENCHEND; \
    guard_dump_bin(CHK_DIR "/out.bin", gC, sizeof(gC)); \
    return 0; }

#define GUARD_COLREDUCE(DT, M, N, ...) \
  static DT gA[(M) * (N)], gC[(M) * (N)]; \
  int main() { \
    guard_read_bin(CHK_DIR "/in_a.bin", gA, sizeof(gA)); \
    BENCHSTART; g_colreduce<DT, M, N>(gC, gA, __VA_ARGS__); BENCHEND; \
    guard_dump_bin(CHK_DIR "/out.bin", gC, sizeof(gC)); \
    return 0; }

// expand-arith: dst = op(src0 MxN, src1 broadcast). row: src1 is Mx1 col-bcast;
// col: src1 is 1xN row-bcast (physical MxN, ValidRow=1). golden.py broadcasts.
#define GUARD_ROWEXPAND(DT, M, N, ...) \
  static DT gA[(M) * (N)], gB[(M) * 1], gC[(M) * (N)]; \
  int main() { \
    guard_read_bin(CHK_DIR "/in_a.bin", gA, sizeof(gA)); \
    guard_read_bin(CHK_DIR "/in_b.bin", gB, sizeof(gB)); \
    BENCHSTART; g_rowexpand<DT, M, N>(gC, gA, gB, __VA_ARGS__); BENCHEND; \
    guard_dump_bin(CHK_DIR "/out.bin", gC, sizeof(gC)); \
    return 0; }

#define GUARD_COLEXPAND(DT, M, N, ...) \
  static DT gA[(M) * (N)], gB[(M) * (N)], gC[(M) * (N)]; \
  int main() { \
    guard_read_bin(CHK_DIR "/in_a.bin", gA, sizeof(gA)); \
    guard_read_bin(CHK_DIR "/in_b.bin", gB, sizeof(gB)); \
    BENCHSTART; g_colexpand<DT, M, N>(gC, gA, gB, __VA_ARGS__); BENCHEND; \
    guard_dump_bin(CHK_DIR "/out.bin", gC, sizeof(gC)); \
    return 0; }

#endif  // TILEOP_GUARD_CASE_HPP
