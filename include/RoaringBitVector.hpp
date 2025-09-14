#pragma once

#include "roaring.hh"

namespace ibbv {
class RoaringBitVector {
protected:
  roaring::Roaring rep;

public:
  // All constructors & destructors & assignments are default
  RoaringBitVector() = default;
  RoaringBitVector(roaring::Roaring&& r) : rep{std::move(r)} {}
  using index_t = uint32_t;
  using iterator = roaring::Roaring::const_iterator;

  auto begin() const noexcept { return rep.begin(); }
  auto end() const noexcept { return rep.end(); }
  auto find_first() const noexcept { return rep.minimum(); }
  auto empty() const noexcept { return rep.isEmpty(); }
  auto count() const noexcept { return rep.cardinality(); }
  void clear() noexcept { rep.clear(); }

  auto test(index_t n) const noexcept { return rep.contains(n); }
  auto reset(index_t n) noexcept { return rep.remove(n); }
  auto set(index_t n) noexcept { return rep.add(n); }
  auto test_and_set(index_t n) noexcept {
    if (test(n)) return false;
    set(n);
    return true;
  }

  auto contains(const RoaringBitVector& rhs) const noexcept {
    return rhs.rep.isSubset(rep);
  }
  auto intersects(const RoaringBitVector& rhs) const noexcept {
    return rep.and_cardinality(rhs.rep) != 0;
  }

  auto operator==(const RoaringBitVector& rhs) const noexcept {
    return rep == rhs.rep;
  }
  auto operator!=(const RoaringBitVector& rhs) const noexcept {
    return !(*this == rhs);
  }

  bool operator|=(const RoaringBitVector& rhs) noexcept {
    const auto old_count = count();
    rep |= rhs.rep;
    return count() != old_count;
  }
  RoaringBitVector operator|(const RoaringBitVector& rhs) const noexcept {
    return RoaringBitVector{rep | rhs.rep};
  }

  bool operator&=(const RoaringBitVector& rhs) noexcept {
    const auto old_count = count();
    rep &= rhs.rep;
    return count() != old_count;
  }
  RoaringBitVector operator&(const RoaringBitVector& rhs) const noexcept {
    return RoaringBitVector{rep & rhs.rep};
  }

  bool operator-=(const RoaringBitVector& rhs) noexcept {
    const auto old_count = count();
    rep -= rhs.rep;
    return count() != old_count;
  }
  RoaringBitVector operator-(const RoaringBitVector& rhs) const noexcept {
    return RoaringBitVector{rep - rhs.rep};
  }
  bool intersectWithComplement(const RoaringBitVector& rhs) noexcept {
    return *this -= rhs;
  }
  void intersectWithComplement(const RoaringBitVector& lhs,
                               const RoaringBitVector& rhs) noexcept {
    rep = lhs.rep - rhs.rep;
  }

  friend struct std::hash<RoaringBitVector>;
};
} // namespace ibbv

namespace std {
template <> struct hash<ibbv::RoaringBitVector> {
  std::size_t operator()(const ibbv::RoaringBitVector& s) const noexcept {
    return s.count() ^ s.find_first();
  }
};
} // namespace std