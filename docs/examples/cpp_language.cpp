template <int Rows, int Cols>
concept StaticTileShape = (Rows > 0) && (Cols > 0);

template <int Rows, int Cols>
  requires StaticTileShape<Rows, Cols>
struct TileShape {
  static constexpr int rows = Rows;
  static constexpr int cols = Cols;
  static constexpr int elements = Rows * Cols;
};

template <typename Shape>
constexpr int vector_count() {
  if constexpr (Shape::elements <= 256) {
    return 1;
  } else {
    return (Shape::elements + 255) / 256;
  }
}

struct WorkSplit {
  int begin;
  int end;
};

constexpr WorkSplit split_rows(int core_id, int core_count, int rows) {
  const int begin = rows * core_id / core_count;
  const int end = rows * (core_id + 1) / core_count;
  return {begin, end};
}

using ExampleTile = TileShape<16, 32>;
static_assert(ExampleTile::elements == 512);
static_assert(vector_count<ExampleTile>() == 2);

extern "C" int cpp_language_probe(int core_id) {
  const auto [begin, end] = split_rows(core_id, 4, 128);
  const auto extent = [=]() constexpr { return end - begin; };
  return extent() + vector_count<ExampleTile>();
}
