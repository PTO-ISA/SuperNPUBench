#include <pto_kernel/tile.hpp>

using namespace pto;

namespace {

constexpr int kSequence = 16;
constexpr int kHead = 8;
constexpr int kQueryRows = 8;
constexpr int kKeyRows = 8;

using QueryTile = MatrixLeftTile<int, kQueryRows, kHead>;
using KeyTile = MatrixRightTile<int, kHead, kKeyRows>;
using ValueTile = MatrixRightTile<int, kKeyRows, kHead>;
using ScoreTile = AccumulatorTile<int, kQueryRows, kKeyRows>;
using ScoreValues = LocalTile<int, kQueryRows, kKeyRows>;
using ScoreLeft = MatrixLeftTile<int, kQueryRows, kKeyRows>;
using OutputTile = AccumulatorTile<int, kQueryRows, kHead>;
using OutputValues = LocalTile<int, kQueryRows, kHead>;

using Queries = global_tensor<int, RowMajor<kSequence, kHead>>;
using Keys = global_tensor<int, ColMajor<kHead, kSequence>>;
using Values = global_tensor<int, ColMajor<kSequence, kHead>>;
using Output = global_tensor<int, RowMajor<kSequence, kHead>>;

} // namespace

extern "C" void flash_attention_probe(int *out, int *q, int *k, int *v) {
  global_iterator<Queries, QueryTile> query_tiles(q);
  global_iterator<Keys, KeyTile> key_tiles(k);
  global_iterator<Values, ValueTile> value_tiles(v);
  global_iterator<Output, OutputValues> output_tiles(out);

  for (int query_block = 0; query_block < kSequence / kQueryRows;
       ++query_block) {
    QueryTile query;
    OutputValues running;
    TLOAD(query, query_tiles(query_block, 0));

    for (int key_block = 0; key_block < kSequence / kKeyRows; ++key_block) {
      KeyTile key;
      ValueTile value;
      ScoreTile score_acc;
      ScoreValues score;
      ScoreLeft score_left;
      OutputTile output_acc;
      OutputValues contribution;

      TLOAD(key, key_tiles(0, key_block));
      TLOAD(value, value_tiles(key_block, 0));
      TMATMUL(score_acc, query, key);
      TCVT(score, score_acc);
      TCVT(score_left, score);
      TMATMUL(output_acc, score_left, value);
      TCVT(contribution, output_acc);

      if (key_block == 0) {
        TMOV(running, contribution);
      } else {
        TADD(running, running, contribution);
      }
    }

    TSTORE(output_tiles(query_block, 0), running);
  }
}
