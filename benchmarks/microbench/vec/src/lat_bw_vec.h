#define KEY_CODE(id) \
                "l.add vt#"#id".reuse.uh, vt#"#id".reuse.uh, ->vt.h\n"
TEST_FUNC(VADD)
#undef KEY_CODE
#if defined(VADD)
    MAIN_FUNC(VADD)
#endif

#define KEY_CODE(id) \
                "l.add vt#"#id".reuse.uh, vt#"#id".reuse.uh, ->vt.h\n" \
                "l.add vu#"#id".reuse.uh, vu#"#id".reuse.uh, ->vu.h\n" \
                "l.add vm#"#id".reuse.uh, vm#"#id".reuse.uh, ->vm.h\n" \
                "l.add vn#"#id".reuse.uh, vn#"#id".reuse.uh, ->vn.h\n"
TEST_FUNC(VADD_TUMN)
#undef KEY_CODE
#if defined(VADD_TUMN)
    MAIN_FUNC(VADD_TUMN)
#endif

#define KEY_CODE(id) \
                "l.sra vt#"#id".reuse.uh, vt#"#id".reuse.uh, ->vt.h\n"
TEST_FUNC(VSRA)
#undef KEY_CODE
#if defined(VSRA)
    MAIN_FUNC(VSRA)
#endif

#define KEY_CODE(id) \
                "l.sra vt#"#id".reuse.uh, vt#"#id".reuse.uh, ->vt.h\n" \
                "l.sra vu#"#id".reuse.uh, vu#"#id".reuse.uh, ->vu.h\n" \
                "l.sra vm#"#id".reuse.uh, vm#"#id".reuse.uh, ->vm.h\n" \
                "l.sra vn#"#id".reuse.uh, vn#"#id".reuse.uh, ->vn.h\n"
TEST_FUNC(VSRA_TUMN)
#undef KEY_CODE
#if defined(VSRA_TUMN)
    MAIN_FUNC(VSRA_TUMN)
#endif

#define KEY_CODE(id) \
                "l.cmp.eq vt#"#id".reuse.uh, vt#"#id".reuse.uh, ->vt.h\n"
TEST_FUNC(VCMPEQ)
#undef KEY_CODE
#if defined(VCMPEQ)
    MAIN_FUNC(VCMPEQ)
#endif

#define KEY_CODE(id) \
                "l.cmp.eq vt#"#id".reuse.uh, vt#"#id".reuse.uh, ->vt.h\n" \
                "l.cmp.eq vu#"#id".reuse.uh, vu#"#id".reuse.uh, ->vu.h\n" \
                "l.cmp.eq vm#"#id".reuse.uh, vm#"#id".reuse.uh, ->vm.h\n" \
                "l.cmp.eq vn#"#id".reuse.uh, vn#"#id".reuse.uh, ->vn.h\n"
TEST_FUNC(VCMPEQ_TUMN)
#undef KEY_CODE
#if defined(VCMPEQ_TUMN)
    MAIN_FUNC(VCMPEQ_TUMN)
#endif

#define KEY_CODE(id) \
                "l.madd vt#"#id".reuse.uh, vt#"#id".reuse.uh, vt#"#id".reuse.uh, ->vt.h\n"
TEST_FUNC(VMADD)
#undef KEY_CODE
#if defined(VMADD)
    MAIN_FUNC(VMADD)
#endif

#define KEY_CODE(id) \
                "l.madd vt#"#id".reuse.uh, vt#"#id".reuse.uh, vt#"#id".reuse.uh, ->vt.h\n" \
                "l.madd vu#"#id".reuse.uh, vu#"#id".reuse.uh, vu#"#id".reuse.uh, ->vu.h\n" \
                "l.madd vm#"#id".reuse.uh, vm#"#id".reuse.uh, vm#"#id".reuse.uh, ->vm.h\n" \
                "l.madd vn#"#id".reuse.uh, vn#"#id".reuse.uh, vn#"#id".reuse.uh, ->vn.h\n"
TEST_FUNC(VMADD_TUMN)
#undef KEY_CODE
#if defined(VMADD_TUMN)
    MAIN_FUNC(VMADD_TUMN)
#endif

#define KEY_CODE(id) \
                "l.rem vt#"#id".reuse.uh, vt#"#id".reuse.uh, ->vt.h\n"
TEST_FUNC(VREM)
#undef KEY_CODE
#if defined(VREM)
    MAIN_FUNC(VREM)
#endif

#define KEY_CODE(id) \
                "l.rem vt#"#id".reuse.uh, vt#"#id".reuse.uh, ->vt.h\n" \
                "l.rem vu#"#id".reuse.uh, vu#"#id".reuse.uh, ->vu.h\n" \
                "l.rem vm#"#id".reuse.uh, vm#"#id".reuse.uh, ->vm.h\n" \
                "l.rem vn#"#id".reuse.uh, vn#"#id".reuse.uh, ->vn.h\n"
TEST_FUNC(VREM_TUMN)
#undef KEY_CODE
#if defined(VREM_TUMN)
    MAIN_FUNC(VREM_TUMN)
#endif

#define KEY_CODE(id) \
                "l.mul vt#"#id".reuse.uh, vt#"#id".reuse.uh, ->vt.h\n"
TEST_FUNC(VMUL)
#undef KEY_CODE
#if defined(VMUL)
    MAIN_FUNC(VMUL)
#endif

#define KEY_CODE(id) \
                "l.mul vt#"#id".reuse.uh, vt#"#id".reuse.uh, ->vt.h\n" \
                "l.mul vu#"#id".reuse.uh, vu#"#id".reuse.uh, ->vu.h\n" \
                "l.mul vm#"#id".reuse.uh, vm#"#id".reuse.uh, ->vm.h\n" \
                "l.mul vn#"#id".reuse.uh, vn#"#id".reuse.uh, ->vn.h\n"
TEST_FUNC(VMUL_TUMN)
#undef KEY_CODE
#if defined(VMUL_TUMN)
    MAIN_FUNC(VMUL_TUMN)
#endif

#define KEY_CODE(id) \
                "l.bic vt#"#id".reuse.uh, 0, 10, ->vt.h\n"
TEST_FUNC(VBIC)
#undef KEY_CODE
#if defined(VBIC)
    MAIN_FUNC(VBIC)
#endif

#define KEY_CODE(id) \
                "l.bic vt#"#id".reuse.uh, 0, 10, ->vt.h\n" \
                "l.bic vu#"#id".reuse.uh, 0, 10, ->vu.h\n" \
                "l.bic vm#"#id".reuse.uh, 0, 10, ->vm.h\n" \
                "l.bic vn#"#id".reuse.uh, 0, 10, ->vn.h\n"
TEST_FUNC(VBIC_TUMN)
#undef KEY_CODE
#if defined(VBIC_TUMN)
    MAIN_FUNC(VBIC_TUMN)
#endif

#define KEY_CODE(id) \
                "l.fmadd vt#"#id".reuse.fh, vt#"#id".reuse.fh, vt#"#id".reuse.fh, ->vt.h\n"
TEST_FUNC(VFMADD)
#undef KEY_CODE
#if defined(VFMADD)
    MAIN_FUNC(VFMADD)
#endif

#define KEY_CODE(id) \
                "l.fmadd vt#"#id".reuse.fh, vt#"#id".reuse.fh, vt#"#id".reuse.fh, ->vt.h\n" \
                "l.fmadd vu#"#id".reuse.fh, vu#"#id".reuse.fh, vu#"#id".reuse.fh, ->vu.h\n" \
                "l.fmadd vm#"#id".reuse.fh, vm#"#id".reuse.fh, vm#"#id".reuse.fh, ->vm.h\n" \
                "l.fmadd vn#"#id".reuse.fh, vn#"#id".reuse.fh, vn#"#id".reuse.fh, ->vn.h\n"
TEST_FUNC(VFMADD_TUMN)
#undef KEY_CODE
#if defined(VFMADD_TUMN)
    MAIN_FUNC(VFMADD_TUMN)
#endif

#define KEY_CODE(id) \
                "l.fdiv vt#"#id".reuse.fh, vt#"#id".reuse.fh, ->vt.h\n"
TEST_FUNC(VFDIV)
#undef KEY_CODE
#if defined(VFDIV)
    MAIN_FUNC(VFDIV)
#endif

#define KEY_CODE(id) \
                "l.fdiv vt#"#id".reuse.fh, vt#"#id".reuse.fh, ->vt.h\n" \
                "l.fdiv vu#"#id".reuse.fh, vu#"#id".reuse.fh, ->vu.h\n" \
                "l.fdiv vm#"#id".reuse.fh, vm#"#id".reuse.fh, ->vm.h\n" \
                "l.fdiv vn#"#id".reuse.fh, vn#"#id".reuse.fh, ->vn.h\n"
TEST_FUNC(VFDIV_TUMN)
#undef KEY_CODE
#if defined(VFDIV_TUMN)
    MAIN_FUNC(VFDIV_TUMN)
#endif

#define KEY_CODE(id) \
                "l.feq vt#"#id".reuse.fh, vt#"#id".reuse.fh, ->vt.h\n"
TEST_FUNC(VFEQ)
#undef KEY_CODE
#if defined(VFEQ)
    MAIN_FUNC(VFEQ)
#endif

#define KEY_CODE(id) \
                "l.feq vt#"#id".reuse.fh, vt#"#id".reuse.fh, ->vt.h\n" \
                "l.feq vu#"#id".reuse.fh, vu#"#id".reuse.fh, ->vu.h\n" \
                "l.feq vm#"#id".reuse.fh, vm#"#id".reuse.fh, ->vm.h\n" \
                "l.feq vn#"#id".reuse.fh, vn#"#id".reuse.fh, ->vn.h\n"
TEST_FUNC(VFEQ_TUMN)
#undef KEY_CODE
#if defined(VFEQ_TUMN)
    MAIN_FUNC(VFEQ_TUMN)
#endif

#define KEY_CODE(id) \
                "l.fcvt.fp322fp16 vt#"#id".reuse.fh, ->vt.h\n"
TEST_FUNC(VFCVT)
#undef KEY_CODE
#if defined(VFCVT)
    MAIN_FUNC(VFCVT)
#endif

#define KEY_CODE(id) \
                "l.fcvt.fp322fp16 vt#"#id".reuse.fh, ->vt.h\n" \
                "l.fcvt.fp322fp16 vu#"#id".reuse.fh, ->vu.h\n" \
                "l.fcvt.fp322fp16 vm#"#id".reuse.fh, ->vm.h\n" \
                "l.fcvt.fp322fp16 vn#"#id".reuse.fh, ->vn.h\n"
TEST_FUNC(VFCVT_TUMN)
#undef KEY_CODE
#if defined(VFCVT_TUMN)
    MAIN_FUNC(VFCVT_TUMN)
#endif

#define KEY_CODE(id) \
                "l.rdadd vt#"#id".reuse.uh, ->t.d\n"
TEST_FUNC(VRDADD)
#undef KEY_CODE
#if defined(VRDADD)
    MAIN_FUNC(VRDADD)
#endif

#define KEY_CODE(id) \
                "l.rdadd vt#"#id".reuse.uh, ->t.d\n" \
                "l.rdadd vu#"#id".reuse.uh, ->u.d\n" \
                "l.rdadd vm#"#id".reuse.uh, ->t.d\n" \
                "l.rdadd vn#"#id".reuse.uh, ->u.d\n"
TEST_FUNC(VRDADD_TUMN)
#undef KEY_CODE
#if defined(VRDADD_TUMN)
    MAIN_FUNC(VRDADD_TUMN)
#endif

#define KEY_CODE(id) \
                "l.rdmax vt#"#id".reuse.uh, ->t.d\n"
TEST_FUNC(VRDMAX)
#undef KEY_CODE
#if defined(VRDMAX)
    MAIN_FUNC(VRDMAX)
#endif

#define KEY_CODE(id) \
                "l.rdmax vt#"#id".reuse.uh, ->t.d\n" \
                "l.rdmax vu#"#id".reuse.uh, ->u.d\n" \
                "l.rdmax vm#"#id".reuse.uh, ->t.d\n" \
                "l.rdmax vn#"#id".reuse.uh, ->u.d\n"
TEST_FUNC(VRDMAX_TUMN)
#undef KEY_CODE
#if defined(VRDMAX_TUMN)
    MAIN_FUNC(VRDMAX_TUMN)
#endif

// #define KEY_CODE(id) \
//                 "l.rdfmax vt#"#id".reuse.fh, ->t.d\n"
// TEST_FUNC(VRDFMAX)
// #undef KEY_CODE
// #if defined(VRDFMAX)
//     MAIN_FUNC(VRDFMAX)
// #endif

// #define KEY_CODE(id) \
//                 "l.rdfmax vt#"#id".reuse.fh, ->t.d\n" \
//                 "l.rdfmax vu#"#id".reuse.fh, ->u.d\n" \
//                 "l.rdfmax vm#"#id".reuse.fh, ->t.d\n" \
//                 "l.rdfmax vn#"#id".reuse.fh, ->u.d\n"
// TEST_FUNC(VRDFMAX_TUMN)
// #undef KEY_CODE
// #if defined(VRDFMAX_TUMN)
//     MAIN_FUNC(VRDFMAX_TUMN)
// #endif

#define KEY_CODE(id) \
                "l.shfl.bfly vt#"#id".reuse.uh, vt#"#id".reuse.uh, ->vt.h\n"
TEST_FUNC(VSHFL_BFLY)
#undef KEY_CODE
#if defined(VSHFL_BFLY)
    MAIN_FUNC(VSHFL_BFLY)
#endif

#define KEY_CODE(id) \
                "l.shfl.bfly vt#"#id".reuse.uh, vt#"#id".reuse.uh, ->vt.h\n" \
                "l.shfl.bfly vu#"#id".reuse.uh, vu#"#id".reuse.uh, ->vu.h\n" \
                "l.shfl.bfly vm#"#id".reuse.uh, vm#"#id".reuse.uh, ->vm.h\n" \
                "l.shfl.bfly vn#"#id".reuse.uh, vn#"#id".reuse.uh, ->vn.h\n"
TEST_FUNC(VSHFL_BFLY_TUMN)
#undef KEY_CODE
#if defined(VSHFL_BFLY_TUMN)
    MAIN_FUNC(VSHFL_BFLY_TUMN)
#endif

#define KEY_CODE(id) \
                "l.shfli.idx vt#"#id".reuse.uh, vt#"#id".reuse.uh, 1->vt.h\n"
TEST_FUNC(VSHFLI_IDX)
#undef KEY_CODE
#if defined(VSHFLI_IDX)
    MAIN_FUNC(VSHFLI_IDX)
#endif

#define KEY_CODE(id) \
                "l.shfli.idx vt#"#id".reuse.uh, vt#"#id".reuse.uh, 1->vt.h\n" \
                "l.shfli.idx vu#"#id".reuse.uh, vu#"#id".reuse.uh, 1->vu.h\n" \
                "l.shfli.idx vm#"#id".reuse.uh, vm#"#id".reuse.uh, 1->vm.h\n" \
                "l.shfli.idx vn#"#id".reuse.uh, vn#"#id".reuse.uh, 1->vn.h\n"
TEST_FUNC(VSHFLI_IDX_TUMN)
#undef KEY_CODE
#if defined(VSHFLI_IDX_TUMN)
    MAIN_FUNC(VSHFLI_IDX_TUMN)
#endif



