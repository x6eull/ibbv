#include "IndexedBlockBitVector.h"
#include "utils.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <iterator>

namespace ibbv::test {
using set = std::set<ele_idx>;
static constexpr size_t max_ele = 128 * (32 + 8);
TEST(TrivialTest, Set) {
  ABV vec;
  const auto bits = random_dist(max_ele * 0.3, max_ele);
  set ordered_bits{bits.begin(), bits.end()};
  for (const auto b : ordered_bits)
    vec.set(b);
  common_rep result = to_common(vec);
  set ordered_result{result.begin(), result.end()};
  ASSERT_EQ(ordered_bits, ordered_result);
}

TEST(ReferenceTest, Union) {
  ABV v1, v2;
  const auto b1 = random_dist(max_ele * 0.3, max_ele),
             b2 = random_dist(max_ele * 0.3, max_ele);
  for (const auto b : b1)
    v1.set(b);
  for (const auto b : b2)
    v2.set(b);
  ABV v3 = v1 | v2;
  std::vector<ele_idx> std_result;
  std::set_union(b1.cbegin(), b1.cend(), b2.cbegin(), b2.cend(),
                 std::back_inserter(std_result));
  ASSERT_EQ(to_common(v3), to_common(std_result));
}

TEST(ReferenceTest, Intersection) {
  ABV v1, v2;
  const auto b1 = random_dist(max_ele * 0.3, max_ele),
             b2 = random_dist(max_ele * 0.3, max_ele);
  for (const auto b : b1)
    v1.set(b);
  for (const auto b : b2)
    v2.set(b);
  ABV v3 = v1;
  v3 &= v2;
  std::vector<ele_idx> std_result;
  std::set_intersection(b1.cbegin(), b1.cend(), b2.cbegin(), b2.cend(),
                        std::back_inserter(std_result));

  const auto sorted_result = to_common_sorted(v3),
             sorted_std_result = to_common_sorted(std_result);
  ASSERT_EQ(sorted_result, sorted_std_result);
}

TEST(ReferenceTest, Difference) {
  ABV v1, v2;
  const auto b1 = random_dist(max_ele * .3, max_ele),
             b2 = random_dist(max_ele * .3, max_ele);
  for (const auto b : b1)
    v1.set(b);
  for (const auto b : b2)
    v2.set(b);
  ABV v3 = v1;
  v3 -= v2;
  std::vector<ele_idx> std_result;
  std::set_difference(b1.cbegin(), b1.cend(), b2.cbegin(), b2.cend(),
                      std::back_inserter(std_result));

  const auto sorted_result = to_common_sorted(v3),
             sorted_std_result = to_common_sorted(std_result);
  ASSERT_EQ(sorted_result, sorted_std_result);
  ASSERT_EQ(v3.count(), sorted_std_result.size());
}

TEST(ReferenceTest, DifferenceAsReset) {
  ABV v1;
  const auto b1 = random_dist(max_ele * .3, max_ele),
             b2 = random_dist(max_ele * .3, max_ele);
  for (const auto b : b1)
    v1.set(b);
  for (const auto b : b2)
    v1.reset(b);
  std::vector<ele_idx> std_result;
  std::set_difference(b1.cbegin(), b1.cend(), b2.cbegin(), b2.cend(),
                      std::back_inserter(std_result));

  const auto sorted_result = to_common_sorted(v1),
             sorted_std_result = to_common_sorted(std_result);
  ASSERT_EQ(sorted_result, sorted_std_result);
  ASSERT_EQ(v1.count(), sorted_std_result.size());
}

TEST(ReferenceTest, Intersects) {
  for (int i = 0; i < 10000; i++) {
    ABV v1, v2;
    const auto b1 = random_dist(64, 128 * 10), b2 = random_dist(64, 128 * 10);
    for (const auto b : b1)
      v1.set(b);
    for (const auto b : b2)
      v2.set(b);
    std::vector<ele_idx> std_result;
    std::set_intersection(b1.cbegin(), b1.cend(), b2.cbegin(), b2.cend(),
                          std::back_inserter(std_result));

    ASSERT_EQ(v1.intersects(v2), !std_result.empty());
  }
}
} // namespace ibbv::test