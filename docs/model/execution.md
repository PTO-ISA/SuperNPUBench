# Tile-ID Superscalar Execution

Tile operations form a dependency graph. The compiler assigns tile IDs, every
definition creates a version, and every consumer binds to the exact versions it
uses.

## Definitions and Uses

```text
T#2.v1 = TLOAD(A)
T#3.v1 = TLOAD(B)
T#4.v1 = TADD(T#2.v1, T#3.v1)
T#5.v1 = TMULS(T#4.v1, alpha)
```

The two loads are independent. The add depends on both load versions. The
scale depends on the add version. A later write to `T#2` cannot change the
version already bound by `TADD`.

## Superscalar Issue

The implementation may issue independent tile operations out of source order
when this preserves:

- source-version dependencies;
- destination-version allocation;
- grouped-operation dynamic order;
- required global-memory conflict order;
- architectural exception and retirement rules.

Correct code therefore expresses ordering through values and memory effects,
not through spacing between instructions.

Fine-grained kernels often expose several short, independent chains at once.
See [Fine-grained 128-byte tile
kernels](../tutorials/fine-grained-tiles.md#expose-independent-tile-chains) for
an unrolled example whose loads and arithmetic are related only by tile
versions.

## Local Versions

Each thread tracks local tile versions independently. A branch may define a
tile only on one thread path. A merge point may use that tile only if the value
is defined on every path that reaches the consumer.

```cpp
if (thread == 0) {
  TLOAD(control_tile, control_view);
}
```

The example defines `control_tile` only on one path. Other paths cannot consume
it as a local value unless another definition covers those paths.

## Shared Versions

Shared definitions bind all participating threads to the same shared version.
A shared `TLOAD`, shared-destination `TMOV`, or shared-producing operation
creates a new version. A consumer must name the version through normal C++
dataflow.

Grouped operations that consume shared values wait for:

- every required local source version;
- the required shared source version;
- destination capacity for the new local or shared version;
- the dynamic participant set required by the operation.

## Relative Tile References

The target block encoding may refer to recent tile producers with compact
relative IDs. That encoding is a compiler and object-code detail. Source
correctness must not depend on physical tile-register numbers or on how long a
relative producer remains encodable.

## Tile-Version Ordering

A consumer is ordered after a producer when it consumes the producer's
tile version. Cross-block visibility requires a launch or runtime protocol
outside the kernel.
