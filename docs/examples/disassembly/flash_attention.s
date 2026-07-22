# Linx LLVM v0.57 block-template excerpt based on docs/examples/flash_attention.cpp.
# Repeated Q/K/V loads and QK/PV products are condensed to representative blocks.
# C.B.IOS shows the required shared-RHS lowering; compiler support is pending.
BSTART.TLOAD    INT32
B.DIM           a5, 0, ->lb0
B.DIM           a5, 0, ->lb1
B.DIM           a5, 0, ->lb2
B.ARG           NORM.normal
B.IOR           [a1,a4],[]
B.IOT           last, ->t<4KB>

BSTART.TLOAD    INT32
B.DIM           a5, 0, ->lb0
B.DIM           a5, 0, ->lb1
B.DIM           a5, 0, ->lb2
B.ARG           NORM.normal
B.IOR           [a2,a4],[]
C.B.IOS         S#12

BSTART.TMATMUL  INT32
C.B.DIMI        8, ->lb0
C.B.DIMI        8, ->lb1
C.B.DIMI        8, ->lb2
C.B.IOS         S#12
B.IOT           t#8, last, ->acc<4KB>

BSTART.ACCCVT   INT32
B.IOT           last, ->m<4KB>

BSTART.TLOAD    INT32
B.DIM           a5, 0, ->lb0
B.DIM           a5, 0, ->lb1
B.DIM           a5, 0, ->lb2
B.ARG           NORM.normal
B.IOR           [a3,a4],[]
C.B.IOS         S#13

BSTART.TMATMUL  INT32
C.B.DIMI        8, ->lb0
C.B.DIMI        8, ->lb1
C.B.DIMI        8, ->lb2
C.B.IOS         S#13
B.IOT           m#2, last, ->acc<4KB>

BSTART.ACCCVT   INT32
B.IOT           last, ->m<4KB>

BSTART.TSTORE   INT32
B.DIM           a5, 0, ->lb0
B.DIM           a5, 0, ->lb1
B.DIM           a5, 0, ->lb2
B.ARG           NORM.normal
B.IOR           [a0,a4],[]
B.IOT           m#1, last, ->m<4KB>
