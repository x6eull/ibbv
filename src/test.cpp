#include "IndexedBlockBitVector.h"
#include "SparseBitVector.h"
#include <benchmark/benchmark.h>
#include <cstdint>
#include <random>
#include <unordered_set>

using ele_idx = uint32_t;
using common_rep = std::unordered_set<ele_idx>;
using IBBV = ibbv::IndexedBlockBitVector<>;
using SBV = llvm::SparseBitVector<>;

static std::random_device rd;
static const auto dev_seed = rd();
static common_rep stable_random_dist(ele_idx n, ele_idx max) {
  constexpr ele_idx min = 0;
  if (n <= 0 || min > max || max - min + 1 < n)
    throw std::invalid_argument("Invalid parameters");

  static common_rep unique_numbers;
  std::mt19937 gen(dev_seed ^ n ^ min ^ max);
  std::uniform_int_distribution<ele_idx> dis(min, max);

  unique_numbers.reserve(n);

  for (ele_idx i = max - n + 1; i <= max; ++i) {
    ele_idx random_num = dis(gen) % (i + 1);
    if (unique_numbers.find(random_num) == unique_numbers.end())
      unique_numbers.insert(random_num);
    else
      unique_numbers.insert(i);
  }

  return unique_numbers;
}

template <typename T> T from_common(const common_rep &rep) {
  T vec;
  for (const auto i : rep)
    vec.set(i);
  return vec;
}
template <typename T> common_rep to_common(const T &bv) {
  common_rep rep;
  for (const auto i : bv)
    rep.emplace(i);
  return rep;
}

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
BENCHMARK(BM_union<IBBV>)
    ->ArgsProduct(arg_full_scalar)
    ->ArgsProduct(arg_ext);
BENCHMARK(BM_union<SBV>) 
    ->ArgsProduct(arg_full_scalar)
    ->ArgsProduct(arg_ext);

// Run the benchmark
BENCHMARK_MAIN();