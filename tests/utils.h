#pragma once

#include "AdaptiveBitVector.h"
#include "IndexedBlockBitVector.h"
#include "SparseBitVector.h"
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <random>
#include <set>
#include <unordered_set>

namespace ibbv::test {
using ele_idx = int32_t;
using common_rep = std::unordered_set<ele_idx>;
/// block size in bits
constexpr inline unsigned short BlockSize = 128;
using ABV = ibbv::AdaptiveBitVector<>;
using SBV = llvm::SparseBitVector<BlockSize>;

static const auto global_seed = []() -> long int {
  const auto result = std::random_device{}();
  std::cout << "random seed: " << result << std::endl;
  return result;
}();
static inline std::mt19937 mix_seed(std::initializer_list<uint32_t> seeds) {
  auto s = global_seed;
  for (const auto t : seeds)
    s ^= t;
  return std::mt19937(s);
}
static inline common_rep stable_random_dist(std::mt19937 gen, ele_idx n,
                                            ele_idx max) {
  constexpr ele_idx min = 0;
  if (n <= 0) return common_rep{};

  common_rep unique_numbers;
  std::uniform_int_distribution<ele_idx> dis(min, max);

  unique_numbers.reserve(n);

  for (ele_idx i = max - n + 1; i <= max; ++i) {
    ele_idx random_num = dis(gen) % (i + 1);
    if (unique_numbers.find(random_num) == unique_numbers.end())
      unique_numbers.insert(random_num);
    else unique_numbers.insert(i);
  }

  return unique_numbers;
}
static inline common_rep stable_random_dist(ele_idx n, ele_idx max) {
  return stable_random_dist(mix_seed({(uint32_t)n, (uint32_t)max}), n, max);
}
static inline std::mt19937 shared_gen(global_seed);
static inline std::set<ele_idx> random_dist(ele_idx n, ele_idx max) {
  constexpr ele_idx min = 0;
  if (n <= 0) return std::set<ele_idx>{};

  std::set<ele_idx> nums;
  std::uniform_int_distribution<ele_idx> dis(min, max);

  for (int i = 0; i < n; ++i)
    nums.insert(dis(shared_gen));

  return nums;
}

template <typename T> T from_common(const common_rep& rep) {
  T vec;
  for (const auto i : rep)
    vec.set(i);
  return vec;
}
template <typename T> common_rep to_common(const T& bv) {
  common_rep rep;
  for (const auto i : bv)
    rep.emplace(i);
  return rep;
}
template <typename T> std::set<ele_idx> to_common_sorted(const T& bv) {
  std::set<ele_idx> rep;
  for (const auto i : bv)
    rep.insert(i);
  return rep;
}

template <typename V>
std::vector<V> concat_vec(std::initializer_list<std::vector<V>> lists) {
  std::vector<V> result;
  size_t total_size = 0;
  for (const auto& lst : lists)
    total_size += lst.size();
  result.reserve(total_size);
  for (const auto& lst : lists)
    result.insert(result.end(), lst.begin(), lst.end());
  return result;
}
} // namespace ibbv::test