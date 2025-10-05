#pragma once

#include <cstdint>
#include <limits>

#include "ewah.h"

namespace ibbv {
class EWAHBitVector {
protected:
  ewah::EWAHBoolArray<uint32_t> rep;

public:
  EWAHBitVector() = default;
  EWAHBitVector(const EWAHBitVector&) = default;
  EWAHBitVector& operator=(const EWAHBitVector&) = default;
  EWAHBitVector(EWAHBitVector&&) = default;
  EWAHBitVector& operator=(EWAHBitVector&&) = default;

  using index_t = uint32_t;
  using iterator = typename decltype(rep)::const_iterator;

  auto begin() const noexcept { return rep.begin(); }
  auto end() const noexcept { return rep.end(); }
  auto find_first() const noexcept {
    return rep.empty() ? std::numeric_limits<index_t>::max() : *rep.begin();
  }
  auto empty() const noexcept { return rep.empty(); }
  auto count() const noexcept { return rep.numberOfOnes(); }
  void clear() noexcept { rep = ewah::EWAHBoolArray<index_t>{}; }

  bool test(index_t n) const noexcept { return rep.get(n); }
  void reset(index_t n) noexcept {
    // inefficient (rarely used)
    rep = rep.logicalandnot(decltype(rep)::bitmapOf(1, n));
  }
  void set(index_t n) noexcept {
    if (n < rep.sizeInBits())
      // by design EWAH is not an updatable data structure in the sense that
      // once bit 1000 is set, you cannot change the value of bits 0 to 1000.
      rep = rep.logicalor(decltype(rep)::bitmapOf(1, n));
    else rep.set(n);
  }
  bool test_and_set(index_t n) noexcept {
    if (n < rep.sizeInBits()) {
      if (test(n)) return false;

      set(n);
      return true;
    }

    return rep.set(n);
  }

  auto contains(const EWAHBitVector& rhs) const noexcept {
    return rhs.count() == rep.logicalandcount(rhs.rep);
  }
  auto intersects(const EWAHBitVector& rhs) const noexcept {
    return rep.logicalandcount(rhs.rep) != 0;
  }

  auto operator==(const EWAHBitVector& rhs) const noexcept {
    return rep == rhs.rep;
  }
  auto operator!=(const EWAHBitVector& rhs) const noexcept {
    return !(*this == rhs);
  }

  bool operator|=(const EWAHBitVector& rhs) noexcept {
    const auto old_count = count();
    rep = rep.logicalor(rhs.rep);
    return count() != old_count;
  }
  EWAHBitVector operator|(const EWAHBitVector& rhs) const noexcept {
    EWAHBitVector result;
    rep.logicalor(rhs.rep, result.rep);
    return result;
  }

  bool operator&=(const EWAHBitVector& rhs) noexcept {
    const auto old_count = count();
    rep = rep.logicaland(rhs.rep);
    return count() != old_count;
  }
  EWAHBitVector operator&(const EWAHBitVector& rhs) const noexcept {
    EWAHBitVector result;
    rep.logicaland(rhs.rep, result.rep);
    return result;
  }

  bool operator-=(const EWAHBitVector& rhs) noexcept {
    const auto old_count = count();
    rep = rep.logicalandnot(rhs.rep);
    return count() != old_count;
  }
  EWAHBitVector operator-(const EWAHBitVector& rhs) const noexcept {
    EWAHBitVector result;
    rep.logicalandnot(rhs.rep, result.rep);
    return result;
  }

  bool intersectWithComplement(const EWAHBitVector& rhs) noexcept {
    return *this -= rhs;
  }
  void intersectWithComplement(const EWAHBitVector& lhs,
                               const EWAHBitVector& rhs) noexcept {
    lhs.rep.logicalandnot(rhs.rep, rep);
  }

  friend struct std::hash<EWAHBitVector>;
};
} // namespace ibbv

namespace std {
template <> struct hash<ibbv::EWAHBitVector> {
  size_t operator()(const ibbv::EWAHBitVector& bv) const noexcept {
    return bv.count() ^ bv.find_first();
  }
};
} // namespace std