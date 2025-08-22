#include "utils.h"

namespace ibbv::test {
template <typename T> static void BM_union(benchmark::State &state) {
  const auto max_num_blocks = state.range(0), occupied_percent = state.range(1);
  const auto max_ele_idx = 128 * max_num_blocks - 1,
             num_ele = 128 * max_num_blocks * occupied_percent / 100;
  const auto rep_scalar = stable_random_dist(num_ele, max_ele_idx),
             rep_rhs_scalar = stable_random_dist(num_ele, max_ele_idx);
  const auto bv_base = from_common<T>(rep_scalar),
             bv_rhs = from_common<T>(rep_rhs_scalar);
  for (auto _ : state) {
    auto bv = bv_base;
    bv |= bv_rhs;
  }
}

const std::vector<std::vector<int64_t>> arg_full_scalar = {
    benchmark::CreateDenseRange(1, 15, 2),
    benchmark::CreateDenseRange(10, 90, 20),
};
const std::vector<std::vector<int64_t>> arg_ext = {
    benchmark::CreateRange(32, 6400, 8),
    benchmark::CreateDenseRange(10, 90, 20),
};
BENCHMARK(BM_union<IBBV>)->ArgsProduct(arg_full_scalar)->ArgsProduct(arg_ext);
BENCHMARK(BM_union<SBV>)->ArgsProduct(arg_full_scalar)->ArgsProduct(arg_ext);
} // namespace ibbv::test