#pragma once

// gfsim does not support guest stdout/stderr syscalls. Keep the checks and
// return status in both builds; the host runner validates the same quiet ELF
// with gfrun and an independent golden before accepting gfsim timing.
#ifdef GFSIM
#define SOLUTION_TEST_PRINTF(...) ((void)0)
#define SOLUTION_TEST_PUTS(...) ((void)0)
#else
#define SOLUTION_TEST_PRINTF(...) printf(__VA_ARGS__)
#define SOLUTION_TEST_PUTS(...) puts(__VA_ARGS__)
#endif
