#pragma once

#include "IndexedBlockBitVector.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>

namespace ibbv {
/// A sorted vector that contains several numbers. Represents a sparse bitset
/// with few bits set.
/// Mutation functions that can grow the vector requires an output argument for
/// the expanded representation. NOTE: If the representation is expanded, the
/// original representation is no longer valid.
template <typename index_t, size_t MAX_SIZE> class TinyBitVector {
protected:
  using data_container = std::array<index_t, MAX_SIZE>;
  data_container data{};
  size_t _count = 0;

public:
  using iterator = typename data_container::const_iterator;
  auto begin() const noexcept {
    return data.cbegin();
  }
  auto end() const noexcept {
    return data.cbegin() + _count;
  }
  /// Returns true if no bits are set.
  auto empty() const noexcept {
    return begin() == end();
  }
  /// Returns the count of bits set.
  size_t count() const noexcept {
    return _count;
  };
  /// Empty the set.
  void clear() noexcept {
    _count = 0;
  }

  template <typename ForwardIt>
  static inline std::optional<IndexedBlockBitVector<>> expand(
      std::optional<IndexedBlockBitVector<>>& expanded, const ForwardIt& begin,
      const ForwardIt& end) noexcept {
    expanded.emplace();
    for (auto it = begin; it != end; ++it)
      expanded->set(*it);
    return expanded;
  }
  template <typename ForwardIt>
  static inline std::optional<IndexedBlockBitVector<>> expand(
      const ForwardIt& begin, const ForwardIt& end) noexcept {
    std::optional<IndexedBlockBitVector<>> expanded;
    return expand(expanded, begin, end);
  }
  inline IndexedBlockBitVector<> expand() const noexcept {
    return expand(begin(), end()).value();
  }
  inline std::optional<IndexedBlockBitVector<>> expand(
      std::optional<IndexedBlockBitVector<>>& expanded) const noexcept {
    return expand(expanded, begin(), end());
  }

public:
  int32_t find_first() const {
    if (empty()) return -1;
    return *begin();
  }
  int32_t find_last() const {
    if (empty()) return -1;
    return *std::prev(end());
  }
  /// Returns true if bit `n` is set.
  bool test(index_t n) const noexcept {
    return std::binary_search(begin(), end(), n);
  }
  /// Set bit `n` (zero-based).
  void set(index_t n,
           std::optional<IndexedBlockBitVector<>>& expanded) noexcept {
    const auto it = std::lower_bound(begin(), end(), n);
    if (it == end() || *it != n) {
      if (count() >= MAX_SIZE) {
        expand(expanded);
        expanded->set(n);
      } else {
        typename data_container::iterator mit = data.begin() + (it - begin());
        typename data_container::iterator mend = data.begin() + count();
        std::copy_backward(it, end(), mend + 1);
        ++_count;
        *mit = n;
      }
    } // found in the vector, noop
  }
  /// Check if bit `n` is set. If it is, returns false.
  /// Otherwise, set it and return true.
  bool test_and_set(index_t n,
                    std::optional<IndexedBlockBitVector<>>& expanded) noexcept {
    const auto it = std::lower_bound(begin(), end(), n);
    if (it == end() || *it != n) {
      if (count() >= MAX_SIZE) {
        expand(expanded);
        expanded->set(n);
      } else {
        typename data_container::iterator mit = data.begin() + (it - begin());
        typename data_container::iterator mend = data.begin() + count();
        std::copy_backward(it, end(), mend + 1);
        ++_count;
        *mit = n;
      }
      return true; // new bit is set
    }
    return false;
  }
  /// Unset bit `n`.
  void reset(index_t n) noexcept {
    const auto it = std::lower_bound(begin(), end(), n);
    if (it != end() && *it == n) {
      std::copy(it + 1, end(), data.begin() + (it - begin()));
      --_count;
    }
  }
  /// Returns true if `this` contains all bits of rhs.
  bool contains(const TinyBitVector& rhs) const noexcept {
    if (count() < rhs.count()) return false;
    auto lhs_it = begin(), rhs_it = rhs.begin();
    while (lhs_it < end() && rhs_it < rhs.end()) {
      if (*lhs_it > *rhs_it) return false;
      else if (*lhs_it < *rhs_it) ++lhs_it;
      else ++lhs_it, ++rhs_it;
    }
    return rhs_it == rhs.end();
  }
  /// Returns true if `this` contains some bits of rhs.
  bool intersects(const TinyBitVector& rhs) const noexcept {
    auto lhs_it = begin(), rhs_it = rhs.begin();
    while (lhs_it < end() && rhs_it < rhs.end()) {
      if (*lhs_it > *rhs_it) ++rhs_it;
      else if (*lhs_it < *rhs_it) ++lhs_it;
      else return true;
    }
    return false;
  }
  bool operator==(const TinyBitVector& rhs) const noexcept {
    if (count() != rhs.count()) return false;
    return std::memcmp(begin(), rhs.begin(), count() * sizeof(index_t)) == 0;
  }
  bool operator!=(const TinyBitVector& rhs) const noexcept {
    return !(*this == rhs);
  }
  /// Inplace union with rhs.
  /// Returns true if `this` changed.
  bool inplace_union(
      const TinyBitVector& rhs,
      std::optional<IndexedBlockBitVector<>>& expanded) noexcept {
    const auto prev_count = count();
    std::array<index_t, 2 * MAX_SIZE> tmp;
    const auto tmp_end = static_cast<typename data_container::const_iterator>(
        std::set_union(begin(), end(), rhs.begin(), rhs.end(), tmp.begin()));
    const auto new_count = tmp_end - tmp.cbegin();
    if (new_count > (decltype(new_count))MAX_SIZE) {
      expand(expanded, tmp.cbegin(), tmp_end);
      return true;
    } else {
      std::copy(tmp.cbegin(), tmp_end, data.begin());
      _count = new_count;
      return new_count != prev_count;
    }
  }
  /// Inplace intersect with rhs.
  /// Returns true if `this` changed.
  bool inplace_intersect(const TinyBitVector& rhs) noexcept {
    auto prev_count = count();
    _count = std::set_intersection(begin(), end(), rhs.begin(), rhs.end(),
                                   data.begin()) -
             begin();
    return count() != prev_count;
  }
  /// Inplace intersect with rhs.
  /// Returns true if `this` changed.
  bool inplace_intersect(const IndexedBlockBitVector<>& rhs) noexcept {
    auto prev_count = count();
    _count = std::set_intersection(begin(), end(), rhs.begin(), rhs.end(),
                                   data.begin()) -
             begin();
    return count() != prev_count;
  }
  /// Inplace difference with rhs.
  /// Returns true if `this` changed.
  bool inplace_diff(const TinyBitVector& rhs) noexcept {
    auto prev_count = count();
    _count = std::set_difference(begin(), end(), rhs.begin(), rhs.end(),
                                 data.begin()) -
             begin();
    return count() != prev_count;
  }
  /// Inplace difference with rhs.
  /// Returns true if `this` changed.
  bool inplace_diff(const IndexedBlockBitVector<>& rhs) noexcept {
    auto prev_count = count();
    _count = std::remove_if(data.begin(), data.begin() + count(),
                            [&](const index_t v) { return rhs.test(v); }) -
             begin();
    return count() != prev_count;
  }

  friend struct std::hash<TinyBitVector>;
};
} // namespace ibbv

namespace std {
template <typename T, size_t MaxSize>
struct hash<ibbv::TinyBitVector<T, MaxSize>> {
  std::size_t operator()(const ibbv::TinyBitVector<T, MaxSize>& v) const {
    return v.count() ^ v.find_first();
  }
};
} // namespace std