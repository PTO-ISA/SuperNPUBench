# Generic kernel launch ABI

The host runtime and resident device image exchange one descriptor address in
the existing HSCB LAUNCH argument word. The descriptor uses a little-endian,
fixed-width layout. This keeps the scheduler protocol independent of C++ struct
padding and lets one ELF export kernels with different argument lists.

The descriptor begins with a 16-byte `KernelArgumentHeader`:

| Offset | Type | Field |
| --- | --- | --- |
| 0 | `uint32_t` | Magic `0x534e5055` |
| 4 | `uint16_t` | ABI version, currently 1 |
| 6 | `uint16_t` | Header size, currently 16 |
| 8 | `uint32_t` | Argument payload size in bytes |
| 12 | `uint32_t` | Kernel-owned schema ID |

Each argument is one 16-byte record with a 32-bit type, a zero reserved field
and a 64-bit value. Version 1 defines buffer address, signed and unsigned 32-
and 64-bit integer, and 32- and 64-bit floating-point records. Narrow scalar
records require zero upper bits. A kernel validates the header and its exact
record count before its typed implementation runs.

Host code builds the list with `KernelCall<Schema>`. Calling `buffer()` records
the current GM address when the launch is submitted and automatically retains
that `Buffer` through completion. Scalar methods accept explicit fixed-width
types. The descriptor bytes are copied into runtime-owned storage at submission,
uploaded to GM in stream order and freed after matching DONE.

```cpp
supernpu::KernelCall<kSchema> call;
call.buffer(input).buffer(output).u32(elements).i32(-3).u64(0);
device.enqueueLaunch(device.kernel("my_kernel"), blockDim, call);
```

Device code uses the checked adapter and declares an exported entry:

```cpp
using Args = supernpu::device::KernelArguments<kSchema, 5>;

uint64_t body(const Args& args) {
    if (!args.is(0, supernpu::KernelArgumentType::Buffer))
        return uint64_t(supernpu::KernelArgumentStatus::TypeMismatch);
    return 0;
}

SUPERNPU_DEFINE_KERNEL(my_kernel, kSchema, 5, body)
```

`DEVICE_KERNEL_SOURCES` may list multiple source files. Each exported C entry
is discovered from the exact ELF loaded by gfrun and becomes a session-owned
`Kernel` handle. The older trivially-copyable `enqueueLaunch` overload remains
available for existing device images that own a private descriptor contract.
