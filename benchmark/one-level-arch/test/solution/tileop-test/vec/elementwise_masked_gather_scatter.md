# Element-wise masked gather/scatter

`src/elementwise_masked_gather_scatter.cpp` exercises the LLVM LinxV5
`ew_mgather_masked` and `ew_mscatter_masked` builtins.  The offset tile uses
the PTO logical element-index contract; the predicate tile is emitted as the
final `B.IOT` source in each masked TLSU bundle.
