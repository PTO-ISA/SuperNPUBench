# Glossary

| Term | Meaning |
| --- | --- |
| Block | Programmer-visible group of threads that can cooperate through block-shared tiles. |
| Thread | One logical execution instance inside a block. |
| Tile intrinsic | C++ operation with defined operands, types, shape/layout constraints, value semantics, and memory effects. |
| PE | Physical processing element to which a logical thread is mapped. |
| Thread index | Immutable value in `[0, threads_per_block)` returned by `get_thread_idx()`. |
| Block index | Launch-defined value returned by `get_block_idx()`. |
| Logical tile | Typed value with shape, valid region, layout, location, and distribution. |
| Thread-local tile | Tile-register version directly visible to one thread. |
| Shared tile register | Block-local, versioned tile value visible to participating threads. |
| Shared region | Thread-owned fragment of a shared tile. |
| Defined mask | Immutable mask naming valid shared regions. |
| Tile ID | Compiler-assigned logical identifier for a tile value. |
| Tile version | One definition of a tile ID, bound immutably by consumers. |
| GlobalTensor | Non-owning typed view of a global-memory region. |
| Valid region | Logical rows and columns for which an operation defines values. |
| Group operation | Operation requiring compatible participation from a defined thread group. |
| M-axis sharding | Matrix rows divided into non-overlapping thread-local fragments. |
| TLSU | Tile load/store unit for global/local/shared movement forms. |
| Generated block | Compiler output block carrying tile IDs, dimensions, attributes, and memory operands. |
| `COMPILER_DIR` | Directory containing the target Clang and LLVM binary tools. |
| `LINX_SYSROOT` | Target libc/runtime sysroot used for hosted links. |
