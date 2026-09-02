// Temporary gfsim diagnostic variant. It keeps the QK/scale/rowmax/exp/rowsum
// path, but suppresses the P*V and final-output sections containing
// CUBE_M16/M32-to-ND stores unsupported by the current timing model. Use this
// testcase only for model/trace validation; it does not produce an FA result.
#define FA_DISABLE_CUBE_TSTORE 1
#include "fa_2d_unroll_gmma.cpp"
