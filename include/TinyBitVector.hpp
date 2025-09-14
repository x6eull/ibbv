#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>

#include "IndexedBlockBitVector.hpp"

namespace ibbv {
/// A sorted vector that contains several numbers. Represents a sparse bitset
/// with few bits set.
/// Mutation functions that can grow the vector requires an output argument for
/// the expanded representation. NOTE: If the representation is expanded, the
/// original representation is no longer valid.
template <typename index_t, size_t MAX_SIZE> class TinyBitVector {
protected:
#if IBBV_ENABLE_ABV
  template <size_t MaxTbvSize> friend class AdaptiveBitVector;
#endif
  using data_container = std::array<index_t, MAX_SIZE>;
  data_container data;
  typename data_container::iterator data_end = data.begin();
  void resize_to(const typename data_container::iterator new_end) noexcept {
    data_end = new_end;
  }
  void inc_size() noexcept { ++data_end; }
  void dec_size() noexcept { --data_end; }
  inline typename data_container::const_iterator cast_const(
      const typename data_container::iterator it) const noexcept {
    return data.cbegin() + (it - data.begin());
  }
  inline typename data_container::iterator cast_nonconst(
      const typename data_container::const_iterator it) noexcept {
    return data.begin() + (it - data.cbegin());
  }

public:
  TinyBitVector() noexcept = default;
  /// Construct TinyBitVector by copying a range.
  /// UB when number of elements > MAX_SIZE.
  template <typename ForwardIt>
  TinyBitVector(ForwardIt begin_it, ForwardIt end_it) noexcept
      : data_end{std::copy(begin_it, end_it, data.begin())} {}
  TinyBitVector(const TinyBitVector& rhs) noexcept
      : data(rhs.data), data_end(data.begin() + rhs.count()) {}
  TinyBitVector& operator=(const TinyBitVector& rhs) noexcept {
    data = rhs.data;
    data_end = data.begin() + rhs.count();
    return *this;
  }
  // moving std::array is quivalent to copy (don't change state)
  TinyBitVector(TinyBitVector&& rhs) noexcept
      : data(std::move(rhs.data)), data_end(data.begin() + rhs.count()) {}
  TinyBitVector& operator=(TinyBitVector&& rhs) noexcept {
    data = std::move(rhs.data);
    data_end = data.begin() + rhs.count();
    return *this;
  }
  // Trivially destructible
  ~TinyBitVector() noexcept = default;

  // Modifying with iterator is forbidden to ensure data sorted
  using iterator = typename data_container::const_iterator;
  using const_iterator = typename data_container::const_iterator;
  inline auto begin() const noexcept { return data.begin(); }
  inline auto cbegin() const noexcept { return data.cbegin(); }
  inline auto begin() noexcept { return data.begin(); }
  inline auto end() const noexcept { return cast_const(data_end); }
  inline auto cend() const noexcept { return cast_const(data_end); }
  inline auto end() noexcept { return data_end; }
  /// Returns true if no bits are set.
  inline auto empty() const noexcept { return begin() == end(); }
  /// Returns the count of bits set.
  inline size_t count() const noexcept { return end() - begin(); };
  /// Empty the set.
  inline void clear() noexcept { resize_to(begin()); }
  /// Expand tbv to ibbv.
  inline auto expand() const noexcept {
    return IndexedBlockBitVector<>(
        static_cast<const index_t*>(data.cbegin()),
        static_cast<size_t>(data_end - data.cbegin()));
  }

  /// Expand tbv to ibbv.
  inline auto expand(
      std::optional<IndexedBlockBitVector<>>& target) const noexcept {
    target.emplace(static_cast<const index_t*>(data.cbegin()),
                   static_cast<size_t>(data_end - data.cbegin()));
  }

public:
  inline int32_t find_first() const noexcept {
    if (empty()) return -1;
    return *begin();
  }
  inline int32_t find_last() const noexcept {
    if (empty()) return -1;
    return *std::prev(end());
  }
  /// Returns true if bit `n` is set.
  inline bool test(index_t n) const noexcept {
    return std::binary_search(begin(), end(), n);
  }
  /// Set bit `n` (zero-based).
  inline void set(index_t n,
                  std::optional<IndexedBlockBitVector<>>& expanded) noexcept {
    const auto it = std::lower_bound(begin(), end(), n);
    if (it == end() || *it != n) {
      if (count() >= MAX_SIZE) {
        expand(expanded);
        expanded->set(n);
      } else {
        std::copy_backward(it, end(), end() + 1);
        *it = n;
        inc_size();
      }
    } // found in the vector, noop
  }
  /// Check if bit `n` is set. If it is, returns false.
  /// Otherwise, set it and return true.
  inline bool test_and_set(
      index_t n, std::optional<IndexedBlockBitVector<>>& expanded) noexcept {
    const auto it = std::lower_bound(begin(), end(), n);
    if (it == end() || *it != n) {
      if (count() >= MAX_SIZE) {
        expand(expanded);
        expanded->set(n);
      } else {
        std::copy_backward(it, end(), end() + 1);
        *it = n;
        inc_size();
      }
      return true; // new bit is set
    }
    return false;
  }
  /// Unset bit `n`.
  inline void reset(index_t n) noexcept {
    const auto it = std::lower_bound(begin(), end(), n);
    if (it != end() && *it == n) {
      std::copy(it + 1, end(), it);
      dec_size();
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
    const size_t new_count = tmp_end - tmp.cbegin();
    if (new_count > MAX_SIZE) {
      expanded.emplace(static_cast<const index_t*>(tmp.cbegin()), new_count);
      return true;
    } else {
      resize_to(std::copy(tmp.cbegin(), tmp_end, begin()));
      return new_count != prev_count;
    }
  }
  /// Inplace intersect with rhs.
  /// Returns true if `this` changed.
  bool inplace_intersect(const TinyBitVector& rhs) noexcept {
    auto prev_count = count();
    resize_to(
        std::set_intersection(begin(), end(), rhs.begin(), rhs.end(), begin()));
    return count() != prev_count;
  }
  /// Inplace intersect with rhs.
  /// Returns true if `this` changed.
  bool inplace_intersect(const IndexedBlockBitVector<>& rhs) noexcept {
    auto prev_count = count();
    resize_to(
        std::set_intersection(begin(), end(), rhs.begin(), rhs.end(), begin()));
    return count() != prev_count;
  }
  void diff_into_this(const TinyBitVector& lhs,
                      const TinyBitVector& rhs) noexcept {
    resize_to(std::set_difference(lhs.begin(), lhs.end(), rhs.begin(),
                                  rhs.end(), begin()));
  }
  void diff_into_this(const TinyBitVector& lhs,
                      const IndexedBlockBitVector<>& rhs) noexcept {
    resize_to(std::copy_if(lhs.begin(), lhs.end(), begin(),
                           [&](const index_t v) { return !rhs.test(v); }));
  }
  /// Inplace difference with rhs.
  /// Returns true if `this` changed.
  bool inplace_diff(const TinyBitVector& rhs) noexcept {
    auto prev_count = count();
    diff_into_this(*this, rhs);
    return count() != prev_count;
  }
  /// Inplace difference with rhs.
  /// Returns true if `this` changed.
  bool inplace_diff(const IndexedBlockBitVector<>& rhs) noexcept {
    auto prev_count = count();
    diff_into_this(*this, rhs);
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