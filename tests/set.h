#pragma once
#include "benchmark/benchmark.h"
#include "utils.h"

namespace ibbv::test {
template <typename T> static void BM_set_seq(benchmark::State &state) {
  const auto max_num_blocks = state.range(0), ele_per_block = state.range(1);
  for (auto _ : state) {
    T vec{};
    for (auto i = 0; i < max_num_blocks; i++)
      for (auto j = 0; j < ele_per_block; j++)
        vec.set(i * BlockSize + j);
  }
  state.SetItemsProcessed(max_num_blocks * ele_per_block * state.iterations());
}

static const inline auto set_seq_args = {
    benchmark::CreateRange(1, 1 << 16, 2),
    concat_vec({{1}, benchmark::CreateRange(2, 128, 4)}),
};
BENCHMARK(BM_set_seq<IBBV>)->ArgsProduct(set_seq_args);
BENCHMARK(BM_set_seq<SBV>)->ArgsProduct(set_seq_args);
} // namespace ibbv::test