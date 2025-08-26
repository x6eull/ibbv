#include "IndexedBlockBitVector.h"
#include "utils.h"
#include <gtest/gtest.h>
#include <iterator>

namespace ibbv::test {
using set = std::set<ele_idx>;
TEST(TrivialTest, Set) {
  IBBV vec;
  common_rep bits = stable_random_dist(128 * 10 * 0.3, 128 * 10);
  set ordered_bits{bits.begin(), bits.end()};
  for (const auto b : ordered_bits)
    vec.set(b);
  common_rep result = to_common(vec);
  set ordered_result{result.begin(), result.end()};
  ASSERT_EQ(ordered_bits, ordered_result);
}

TEST(ReferenceTest, Union) {
  IBBV v1, v2;
  common_rep b1 = stable_random_dist(128 * 10 * 0.3, 128 * 10),
             b2 = stable_random_dist(128 * 10 * 0.3, 128 * 10);
  for (const auto b : b1)
    v1.set(b);
  for (const auto b : b2)
    v2.set(b);
  IBBV v3 = v1 | v2;
  std::vector<ele_idx> std_result;
  std::set_union(b1.cbegin(), b1.cend(), b2.cbegin(), b2.cend(),
                 std::back_inserter(std_result));
  ASSERT_EQ(to_common(v3), to_common(std_result));
}
} // namespace ibbv::test