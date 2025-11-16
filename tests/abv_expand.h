#pragma once
#include "utils.h"
#include <benchmark/benchmark.h>

namespace ibbv::test {
static void BM_abv_expand(benchmark::State& state) {
  const auto num_eles = state.range(0);
  uint32_t items[8];
  for (int i = 0; i < num_eles; i++) {
    items[i] = BlockSize * i + i;
  }
  for (auto _ : state) {
    ibbv::IndexedBlockBitVector<> v(items, num_eles);
    benchmark::DoNotOptimize(v);
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_abv_expand)->DenseRange(1, 8, 1);
} // namespace ibbv::test