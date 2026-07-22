# Glossary

| Term | Meaning |
| --- | --- |
| PTO | Typed tile programming and instruction model used by one-level NPU kernels. |
| PTO intrinsic | Operation with defined operands, types, shape/layout constraints, value semantics, and memory effects. |
| PE | One of four independent processing elements in a core. |
| PE ID | Immutable value in `[0, 3]` returned by `get_thread_idx()`. |
| Core ID | Launch-defined value returned by `get_block_idx()`. |
| Logical tile | Typed value with shape, valid region, layout, location, and distribution. |
| PE-local tile | Tile-register version directly visible to one PE. |
| Shared tile register | Core-local, versioned tile value visible to all four PEs. |
| Shared region | Fixed quarter of a shared tile associated with one PE. |
| Defined mask | Immutable four-bit mask naming valid shared regions. |
| Tile ID | Compiler-assigned logical identifier for a tile value. |
| Tile version | One definition of a tile ID, bound immutably by consumers. |
| GlobalTensor | Non-owning typed view of a global-memory region. |
| Valid region | Logical rows and columns for which an operation defines values. |
| Group operation | Dynamic operation requiring compatible participation from all four PEs. |
| M-axis sharding | Matrix rows divided into four PE-local quarters. |
| TLSU | Tile load/store unit for global/local/shared movement forms. |
| PTO-AS | Target-independent textual form of a PTO operation. |
| Linx block | Target block template carrying tile IDs, dimensions, attributes, and memory operands. |
| `COMPILER_DIR` | Directory containing Linx Clang and LLVM binary tools. |
| `LINX_SYSROOT` | Linx target libc/runtime sysroot used for hosted links. |
