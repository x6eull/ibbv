#include "IndexedBlockBitVector.h"
#include "SparseBitVector.h"
#include <benchmark/benchmark.h>
#include <cstdint>
#include <random>
#include <unordered_set>

namespace ibbv::test {
using ele_idx = uint32_t;
using common_rep = std::unordered_set<ele_idx>;
using IBBV = ibbv::IndexedBlockBitVector<>;
using SBV = llvm::SparseBitVector<>;

static std::random_device rd;
static const auto dev_seed = rd();
static inline common_rep stable_random_dist(ele_idx n, ele_idx max) {
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
} // namespace ibbv::test