#pragma once
#include <benchmark/benchmark.h>
#include "utils.h"

namespace ibbv::test {
template <typename T> static void BM_union(benchmark::State &state) {
  const auto max_num_blocks = state.range(0), occupied_percent = state.range(1);
  const auto max_ele_idx = BlockSize * max_num_blocks - 1,
             num_ele = BlockSize * max_num_blocks * occupied_percent / 100;
  const auto rep = stable_random_dist(num_ele, max_ele_idx),
             rep_rhs = stable_random_dist(num_ele, max_ele_idx);
  const auto bv_base = from_common<T>(rep),
             bv_rhs = from_common<T>(rep_rhs);
  for (auto _ : state) {
    auto bv = bv_base;
    bv |= bv_rhs;
  }
}

static const inline auto union_args = {
    concat_vec({benchmark::CreateDenseRange(1, 15, 2),
                benchmark::CreateRange(32, 6400, 8)}),
    benchmark::CreateDenseRange(10, 90, 20),
};
BENCHMARK(BM_union<IBBV>)->ArgsProduct(union_args);
BENCHMARK(BM_union<SBV>)->ArgsProduct(union_args);
} // namespace ibbv::test