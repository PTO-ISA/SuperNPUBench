#if (CHAIN_NUM==1)
#define TEST_FUNC(name) \
    void __vec__ test_##name(int loop, tile_shape::TileDType __out__ placeholder)\
    { \
        __asm__ __volatile__( \
        "l.addi zero.sh, 1, ->vt.h\n" \
        "l.addi zero.sh, 1, ->vu.h\n" \
        "l.addi zero.sh, 1, ->vm.h\n" \
        "l.addi zero.sh, 1, ->vn.h\n" \
    "1: \n" \
        ".rept 50\n" \
        KEY_CODE(1) KEY_CODE(1) KEY_CODE(1) KEY_CODE(1) \
        KEY_CODE(1) KEY_CODE(1) KEY_CODE(1) KEY_CODE(1) \
        KEY_CODE(1) KEY_CODE(1) KEY_CODE(1) KEY_CODE(1) \
        KEY_CODE(1) KEY_CODE(1) KEY_CODE(1) KEY_CODE(1) \
        KEY_CODE(1) KEY_CODE(1) KEY_CODE(1) KEY_CODE(1) \
        KEY_CODE(1) KEY_CODE(1) KEY_CODE(1) KEY_CODE(1) \
        KEY_CODE(1) KEY_CODE(1) KEY_CODE(1) KEY_CODE(1) \
        KEY_CODE(1) KEY_CODE(1) KEY_CODE(1) KEY_CODE(1) \
        ".endr\n" \
        "l.bstop\n" \
        : \
        : \
        :"memory" \
        ); \
    }
#endif

// "subi ri0=, 1, ->ri0\n"
// "b.ne ri0, 0, 1b\n" 

#if (CHAIN_NUM==2)
#define TEST_FUNC(name) \
    void __vec__ test_##name(int loop, tile_shape::TileDType __out__ placeholder)\
    { \
        __asm__ __volatile__( \
        "l.addi zero.sh, 1, ->vt.h\n" \
        "l.addi zero.sh, 1, ->vt.h\n" \
        "l.addi zero.sh, 1, ->vu.h\n" \
        "l.addi zero.sh, 1, ->vu.h\n" \
        "l.addi zero.sh, 1, ->vm.h\n" \
        "l.addi zero.sh, 1, ->vm.h\n" \
        "l.addi zero.sh, 1, ->vn.h\n" \
        "l.addi zero.sh, 1, ->vn.h\n" \
    "1: \n" \
        ".rept 50\n" \
        KEY_CODE(1) KEY_CODE(2) KEY_CODE(1) KEY_CODE(2) \
        KEY_CODE(1) KEY_CODE(2) KEY_CODE(1) KEY_CODE(2) \
        KEY_CODE(1) KEY_CODE(2) KEY_CODE(1) KEY_CODE(2) \
        KEY_CODE(1) KEY_CODE(2) KEY_CODE(1) KEY_CODE(2) \
        KEY_CODE(1) KEY_CODE(2) KEY_CODE(1) KEY_CODE(2) \
        KEY_CODE(1) KEY_CODE(2) KEY_CODE(1) KEY_CODE(2) \
        KEY_CODE(1) KEY_CODE(2) KEY_CODE(1) KEY_CODE(2) \
        KEY_CODE(1) KEY_CODE(2) KEY_CODE(1) KEY_CODE(2) \
        ".endr\n" \
        "l.bstop\n" \
        : \
        : \
        :"memory" \
        ); \
    }
#endif

#if (CHAIN_NUM==4)
#define TEST_FUNC(name) \
    void __vec__ test_##name( int loop, tile_shape::TileDType __out__ placeholder)\
    { \
        __asm__ __volatile__( \
        "l.addi zero.sh, 1, ->vt.h\n" \
        "l.addi zero.sh, 1, ->vt.h\n" \
        "l.addi zero.sh, 1, ->vt.h\n" \
        "l.addi zero.sh, 1, ->vt.h\n" \
        "l.addi zero.sh, 1, ->vu.h\n" \
        "l.addi zero.sh, 1, ->vu.h\n" \
        "l.addi zero.sh, 1, ->vu.h\n" \
        "l.addi zero.sh, 1, ->vu.h\n" \
        "l.addi zero.sh, 1, ->vm.h\n" \
        "l.addi zero.sh, 1, ->vm.h\n" \
        "l.addi zero.sh, 1, ->vm.h\n" \
        "l.addi zero.sh, 1, ->vm.h\n" \
        "l.addi zero.sh, 1, ->vn.h\n" \
        "l.addi zero.sh, 1, ->vn.h\n" \
        "l.addi zero.sh, 1, ->vn.h\n" \
        "l.addi zero.sh, 1, ->vn.h\n" \
    "1: \n" \
        ".rept 50\n" \
        KEY_CODE(1) KEY_CODE(2) KEY_CODE(3) KEY_CODE(4) \
        KEY_CODE(1) KEY_CODE(2) KEY_CODE(3) KEY_CODE(4) \
        KEY_CODE(1) KEY_CODE(2) KEY_CODE(3) KEY_CODE(4) \
        KEY_CODE(1) KEY_CODE(2) KEY_CODE(3) KEY_CODE(4) \
        KEY_CODE(1) KEY_CODE(2) KEY_CODE(3) KEY_CODE(4) \
        KEY_CODE(1) KEY_CODE(2) KEY_CODE(3) KEY_CODE(4) \
        KEY_CODE(1) KEY_CODE(2) KEY_CODE(3) KEY_CODE(4) \
        KEY_CODE(1) KEY_CODE(2) KEY_CODE(3) KEY_CODE(4) \
        ".endr\n" \
        "l.bstop\n" \
        : \
        : \
        :"memory" \
        ); \
    }
#endif