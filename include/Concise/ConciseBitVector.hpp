#pragma once

#include <limits>

#include "concise.h"

#if IBBV_IMPL == IBBV_IMPL_CONCISE
static inline constexpr bool wah_mode = false;
#elif IBBV_IMPL == IBBV_IMPL_WAH
static inline constexpr bool wah_mode = true;
#else
#  error Invalid IBBV_IMPL
#endif
namespace ibbv {
class ConciseBitVector {
protected:
  ConciseSet<wah_mode> rep;

public:
  ConciseBitVector() = default;
  ConciseBitVector(const ConciseBitVector&) = default;
  ConciseBitVector& operator=(const ConciseBitVector&) = default;
  ConciseBitVector(ConciseBitVector&&) = default;
  ConciseBitVector& operator=(ConciseBitVector&&) = default;

  using index_t = uint32_t;
  using iterator = typename decltype(rep)::const_iterator;

  auto begin() const noexcept { return rep.begin(); }
  auto end() const noexcept { return rep.end(); }
  auto find_first() const noexcept {
    return rep.isEmpty() ? std::numeric_limits<index_t>::max() : *rep.begin();
  }
  auto empty() const noexcept { return rep.isEmpty(); }
  auto count() const noexcept { return rep.size(); }
  void clear() noexcept { rep.clear(); }

  auto test(index_t n) const noexcept { return rep.contains(n); }
  void reset(index_t n) noexcept {
    // TODO: inefficient (rarely used)
    decltype(rep) tmp{};
    tmp.append(n);
    rep = std::move(rep.logicalandnot(tmp));
  }
  void set(index_t n) noexcept { rep.add(n); }
  bool test_and_set(index_t n) noexcept {
    if (test(n)) return false;
    set(n);
    return true;
  }

  auto contains(const ConciseBitVector& rhs) const noexcept {
    return rhs.count() == rep.logicalandCount(rhs.rep);
  }
  auto intersects(const ConciseBitVector& rhs) const noexcept {
    return rep.logicalandCount(rhs.rep) != 0;
  }

  auto operator==(const ConciseBitVector& rhs) const noexcept {
    return rep.equals(rhs.rep);
  }
  auto operator!=(const ConciseBitVector& rhs) const noexcept {
    return !(*this == rhs);
  }

  bool operator|=(const ConciseBitVector& rhs) noexcept {
    const auto old_count = count();
    rep = std::move(rep.logicalor(rhs.rep));
    return count() != old_count;
  }
  ConciseBitVector operator|(const ConciseBitVector& rhs) const noexcept {
    ConciseBitVector result;
    rep.logicalorToContainer(rhs.rep, result.rep);
    return result;
  }

  bool operator&=(const ConciseBitVector& rhs) noexcept {
    const auto old_count = count();
    rep = std::move(rep.logicaland(rhs.rep));
    return count() != old_count;
  }
  ConciseBitVector operator&(const ConciseBitVector& rhs) const noexcept {
    ConciseBitVector result;
    rep.logicalandToContainer(rhs.rep, result.rep);
    return result;
  }

  bool operator-=(const ConciseBitVector& rhs) noexcept {
    const auto old_count = count();
    rep = std::move(rep.logicalandnot(rhs.rep));
    return count() != old_count;
  }
  ConciseBitVector operator-(const ConciseBitVector& rhs) const noexcept {
    ConciseBitVector result;
    rep.logicalandnotToContainer(rhs.rep, result.rep);
    return result;
  }

  bool intersectWithComplement(const ConciseBitVector& rhs) noexcept {
    return *this -= rhs;
  }
  void intersectWithComplement(const ConciseBitVector& lhs,
                               const ConciseBitVector& rhs) noexcept {
    lhs.rep.logicalandnotToContainer(rhs.rep, rep);
  }

  friend struct std::hash<ConciseBitVector>;
};
} // namespace ibbv

namespace std {
template <> struct hash<ibbv::ConciseBitVector> {
  size_t operator()(const ibbv::ConciseBitVector& bv) const noexcept {
    return bv.count() ^ bv.find_first();
  }
};
} // namespace std