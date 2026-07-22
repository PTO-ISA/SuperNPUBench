# Compiler v0.57 excerpt generated from docs/examples/tile_add.cpp.
BSTART.TLOAD   FP32
B.DATR         canon, FP64, Null, cmode0, rmode0
B.IOT          last, ->t<4KB>
B.IOR          [a1,a3],[]

BSTART.TLOAD   FP32
B.DATR         canon, FP64, Null, cmode0, rmode0
B.IOT          last, ->t<4KB>
B.IOR          [a2,a3],[]

BSTART.TADD    FP32
B.DATR         canon, FP64, Null, cmode0, rmode0
B.IOT          t#1, t#2, last, ->t<4KB>

BSTART.TSTORE  FP32
B.DATR         canon, FP64, Null, cmode0, rmode0
B.IOT          t#1, last, ->t<4KB>
B.IOR          [a0,a3],[]
