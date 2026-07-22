# Tile-ID Superscalar Execution

## Logical Tile IDs

The compiler assigns logical IDs to PE-local and shared tile values. A
destination is a definition of a new version; a source is a use of one exact
version.

```text
program order
  T#2.v1 = TLOAD(A)
  T#3.v1 = TLOAD(B)
  T#4.v1 = TADD(T#2.v1, T#3.v1)
  T#5.v1 = TMUL(T#2.v1, T#3.v1)
```

The two arithmetic operations are independent after both loads are ready and
may execute in either order or overlap.

## Rename

Rename maps each logical version to a physical tile-register version. Source
bindings are immutable for that instruction. A newer definition of the same
logical ID receives a different physical version.

```text
old reader -> T#4.v1 -> physical p12
new writer -> T#4.v2 -> physical p29
```

This removes write-after-write and write-after-read hazards. Read-after-write
is enforced by source-version readiness.

## Issue Rules

An operation may issue when:

1. all bound source versions are ready;
2. its destination physical storage is allocated;
3. the required functional unit accepts the operation;
4. any group-participation requirement is satisfied;
5. memory conflict and ordering rules permit issue.

Program order alone does not force issue order for independent operations.

## PE-Local Versions

Each PE renames and tracks its local tile versions independently. A branch may
create a new version on only a static subset of PEs. Merge points carry
per-region definedness so undefined quarters cannot flow into a full-tile
consumer.

## Shared Versions

All four PE contexts must bind a source-level shared definition to the same
physical shared version. A core definition sequencer coordinates the mapping.
Shared consumers wait on the bound shared version, not on a dynamically chosen
"latest" value.

## Group Matrix Issue

A group `TMATMUL` waits for:

- four matching dynamic participants;
- each PE's local A quarter;
- one fully defined shared B version;
- local destination capacity in each PE.

After group formation, the four matrix engines may complete their quarters at
different cycles.

## Producer-Age Windows

The one-level block encoding may refer to recent tile producers with relative
IDs. Compiler allocation must keep every source within the target's live
producer window. A consumer does not pop a producer. Physical reclamation
occurs only after all bound readers finish.

## Source-Level Rule

Express dependencies by passing tile values. Do not encode correctness through
instruction spacing, assumed latency, or physical tile-register numbers.
