// Dedicated translation unit for TESTCASE=fa_lowp_recip.
//
// The -DFA_LOWP_RECIP=1 define (fa Makefile) selects the TRECIP-based
// reciprocal path inside fa_lowp.cpp. A dedicated TU is required because
// make tracks only source mtimes: sharing fa_lowp.o between the fa_lowp
// and fa_lowp_recip TESTCASEs silently reuses an object built with the
// other define set, producing an ELF named fa_lowp_recip that actually
// contains the default TSUB path (observed in the 09-18 regression).
#include "fa_lowp.cpp"
