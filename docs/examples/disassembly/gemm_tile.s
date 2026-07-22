# Compiler v0.57 excerpt generated from docs/examples/gemm_tile.cpp.
BSTART.TLOAD    INT32
B.ARG           NORM.normal
B.IOR           [a1,a3],[]
B.IOT           last, ->t<4KB>

BSTART.TLOAD    INT32
B.ARG           NORM.normal
B.IOR           [a2,a3],[]
B.IOT           last, ->t<4KB>

BSTART.TMATMUL  INT32
C.B.DIMI        8, ->lb0
C.B.DIMI        8, ->lb1
C.B.DIMI        8, ->lb2
B.IOT           t#2, t#1, last, ->acc<4KB>

BSTART.ACCCVT   INT32
B.IOT           last, ->m<4KB>

BSTART.TSTORE   INT32
B.ARG           NORM.normal
B.IOR           [a0,a3],[]
B.IOT           m#1, last, ->m<4KB>
