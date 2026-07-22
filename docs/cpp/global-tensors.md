# Global Tensor Views

A global tensor view is a typed description of external memory. It carries
enough metadata for tile operations to load and store the intended elements.

## Descriptor Fields

A view should describe:

- base address;
- element type;
- logical shape;
- stride;
- layout;
- valid region;
- padding behavior when applicable.

## Slicing

Slice by block first, then by thread:

```cpp
const uint32_t block = get_block_idx();
const uint32_t thread = get_thread_idx();

RowSlice rows = partition_rows(total_rows, block, block_count, thread);
auto fragment_view = tensor.slice_rows(rows.begin, rows.count);
```

The slice must stay inside the valid region. Padding can exist in storage, but
it is not valid data unless an operation explicitly defines it as such.

## Iterating a Tile Grid

A global iterator converts tile coordinates to a typed tensor region:

```cpp
using Tile = LocalTile<float, 32, 32>;
using Tensor = global_tensor<float, RowMajor<128, 256>>;
using Tiles = global_iterator<Tensor, Tile>;

Tiles tiles(base);
TLOAD(local, tiles(tile_row, tile_col));
```

`tile_row` and `tile_col` count `32 x 32` regions. They are not element
coordinates. For tensors with outer dimensions, offset the base to one matrix
or construct a view for that outer slice before creating the tile iterator.

## Loads and Stores

```cpp
TLOAD(local_fragment, fragment_view);
TSTORE(output.slice_rows(rows.begin, rows.count), local_fragment);
```

Avoid overlapping stores between participants. If an algorithm requires many
writers for one output range, use a defined reduction or scatter operation
rather than relying on memory timing.
