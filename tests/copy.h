#pragma once
#include "benchmark/benchmark.h"
#include "utils.h"

namespace ibbv::test {
template <typename T> static void BM_copy(benchmark::State &state) {
  const auto max_num_blocks = state.range(0), occupied_percent = state.range(1);
  const auto max_ele_idx = BlockSize * max_num_blocks - 1,
             num_ele = BlockSize * max_num_blocks * occupied_percent / 100;
  const auto rep = stable_random_dist(num_ele, max_ele_idx);
  const auto bv_base = from_common<T>(rep);
  for (auto _ : state) {
    auto copy = bv_base;
    benchmark::DoNotOptimize(copy);
  }
}

static const inline auto copy_args = {
    concat_vec({{0}, benchmark::CreateRange(1, 1 << 16, 2)}),
    benchmark::CreateDenseRange(10, 90, 20),
};
BENCHMARK(BM_copy<IBBV>)->ArgsProduct(copy_args);
BENCHMARK(BM_copy<SBV>)->ArgsProduct(copy_args);
} // namespace ibbv::test