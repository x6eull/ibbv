#pragma once
#include "utils.h"
#include <benchmark/benchmark.h>

namespace ibbv::test {
template <typename T> static void BM_copy(benchmark::State& state) {
  const auto num_blocks = state.range(0), ele_per_block = state.range(1);
  T bv_base{};
  for (int i = 0; i < num_blocks; i++)
    for (int j = 0; j < ele_per_block; j++)
      bv_base.set(i * BlockSize + j);
  for (auto _ : state) {
    auto copy = bv_base;
    benchmark::DoNotOptimize(copy);
  }
  state.SetItemsProcessed(state.iterations());
}

static const inline auto copy_args = {
    concat_vec({{0}, benchmark::CreateRange(1, 1 << 16, 8)}),
    benchmark::CreateRange(1, 128, 8),
};
BENCHMARK(BM_copy<IBBV>)->ArgsProduct(copy_args);
BENCHMARK(BM_copy<SBV>)->ArgsProduct(copy_args);
} // namespace ibbv::test