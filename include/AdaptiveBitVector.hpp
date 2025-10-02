#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <variant>

#if IBBV_IMPL == IBBV_IMPL_CONCISE || IBBV_IMPL == IBBV_IMPL_WAH
#  include "Concise/ConciseBitVector.hpp"
namespace ibbv {
template <size_t _ = 0> using AdaptiveBitVector = ibbv::ConciseBitVector;
}
#elif IBBV_IMPL == IBBV_IMPL_ROARING
#  include "Roaring/RoaringBitVector.hpp"
namespace ibbv {
template <size_t _ = 0> using AdaptiveBitVector = ibbv::RoaringBitVector;
}
#elif IBBV_IMPL == IBBV_IMPL_ABV
#  include "TinyBitVector.hpp"

namespace ibbv {
static inline constexpr size_t DEFAULT_TBV_LIMIT =
    (sizeof(IndexedBlockBitVector<>) - 8) / sizeof(uint32_t);
static_assert(sizeof(TinyBitVector<uint32_t, DEFAULT_TBV_LIMIT>) ==
              sizeof(IndexedBlockBitVector<>));
template <size_t MaxTbvSize = DEFAULT_TBV_LIMIT> class AdaptiveBitVector {
public:
  using index_t = uint32_t;
  using tbv = TinyBitVector<index_t, MaxTbvSize>;
  using ibbv = IndexedBlockBitVector<>;

protected:
  std::variant<tbv, ibbv> rep{};
#  define IFE_TBV(ptrvarname, variant, then_block, else_block)                 \
    if (auto* ptrvarname = std::get_if<tbv>(&variant)) {                       \
      then_block                                                               \
    } else {                                                                   \
      {                                                                        \
        auto* ptrvarname = std::get_if<ibbv>(&variant);                        \
        else_block                                                             \
      }                                                                        \
    }
/// 4 cases
#  define IFE_CASE(variant_lhs, variant_rhs, t_t_block, t_i_block, i_t_block,  \
                   i_i_block)                                                  \
    IFE_TBV(                                                                   \
        lptr, variant_lhs,                                                     \
        { IFE_TBV(rptr, variant_rhs, t_t_block, t_i_block); },                 \
        { IFE_TBV(rptr, variant_rhs, i_t_block, i_i_block); })

/// 3 cases: matched types are processed in the same way
#  define IFE_MATCH_SYM(variant_lhs, variant_rhs, match_block, t_i_block,      \
                        i_t_block)                                             \
    IFE_TBV(lptr, variant_lhs,                                                 \
            IFE_TBV(rptr, variant_rhs, match_block, t_i_block),                \
            IFE_TBV(rptr, variant_rhs, i_t_block, match_block))

/// 2 cases: unmatched types are also processed in the same way
#  define IFE_SYM(variant_lhs, variant_rhs, match_block, unmatch_block)        \
    IFE_MATCH_SYM(variant_lhs, variant_rhs, match_block, unmatch_block,        \
                  unmatch_block)

public:
  /// For debugging. Use with caution.
  __attribute__((used)) auto impl_type() const noexcept { return rep.index(); }
  /// For debugging. Use with caution.
  __attribute__((used)) ibbv* get_ibbv() noexcept {
    return std::get_if<ibbv>(&rep);
  }
  /// For debugging. Use with caution.
  __attribute__((used)) tbv* get_tbv() noexcept {
    return std::get_if<tbv>(&rep);
  }
  class AdaptiveBitVectorIterator {
  protected:
    std::variant<typename tbv::iterator, typename ibbv::iterator> it_rep;

  public:
    using iterator_category = std::input_iterator_tag;
    using value_type = index_t;
    using difference_type = std::ptrdiff_t;
    using pointer = const index_t*;
    using reference = const index_t&;
    AdaptiveBitVectorIterator(const std::variant<tbv, ibbv>& vec,
                              std::false_type) noexcept
        : it_rep(std::visit(
              [=](auto&& arg) {
                return std::variant<typename tbv::iterator,
                                    typename ibbv::iterator>{arg.begin()};
              },
              vec)) {}
    AdaptiveBitVectorIterator(const std::variant<tbv, ibbv>& vec,
                              std::true_type) noexcept
        : it_rep(std::visit(
              [=](auto&& arg) {
                return std::variant<typename tbv::iterator,
                                    typename ibbv::iterator>{arg.end()};
              },
              vec)) {}

    value_type operator*() const noexcept {
      return std::visit([](auto&& arg) -> value_type { return (*arg); },
                        it_rep);
    }
    AdaptiveBitVectorIterator& operator++() noexcept {
      std::visit([](auto&& arg) { arg++; }, it_rep);
      return *this;
    }
    AdaptiveBitVectorIterator operator++(int) noexcept {
      AdaptiveBitVectorIterator temp = *this;
      ++*this;
      return temp;
    }
    /// Compare whether two iterators are equal.
    /// UB if comparing iterators with different representations.
    bool operator==(const AdaptiveBitVectorIterator& other) const noexcept {
      static_assert(
          std::is_convertible_v<typename tbv::iterator, index_t const*>);
      static_assert(offsetof(ibbv::iterator, idx_it) == 0);
      return *reinterpret_cast<index_t const* const*>(&it_rep) ==
             *reinterpret_cast<index_t const* const*>(&other.it_rep);
    }
    bool operator!=(const AdaptiveBitVectorIterator& other) const noexcept {
      return !(*this == other);
    }
  };
  using iterator = AdaptiveBitVectorIterator;
  auto begin() const noexcept { return iterator(rep, std::false_type()); }
  auto end() const noexcept { return iterator(rep, std::true_type()); }
  auto find_first() const noexcept {
    return std::visit([](auto&& arg) { return arg.find_first(); }, rep);
  }
  auto empty() const noexcept {
    return std::visit([](auto&& arg) { return arg.empty(); }, rep);
  }
  auto count() const noexcept {
    return std::visit([](auto&& arg) { return (size_t)arg.count(); }, rep);
  }
  auto clear() noexcept { rep.template emplace<tbv>(); }
  auto test(index_t n) const noexcept {
    return std::visit([=](auto&& arg) { return arg.test(n); }, rep);
  }
  auto reset(index_t n) noexcept {
    return std::visit([=](auto&& arg) { return arg.reset(n); }, rep);
  }
  // set, test_and_set could expand representation
  auto set(index_t n) noexcept {
    IFE_TBV(
        ptr, rep,
        {
          std::optional<ibbv> expanded;
          ptr->set(n, expanded);
          if (expanded.has_value())
            rep.template emplace<ibbv>(std::move(expanded.value()));
        },
        { ptr->set(n); });
  }
  auto test_and_set(index_t n) noexcept {
    IFE_TBV(
        ptr, rep,
        {
          std::optional<ibbv> expanded;
          const auto result = ptr->test_and_set(n, expanded);
          if (expanded.has_value())
            rep.template emplace<ibbv>(std::move(expanded.value()));
          return result;
        },
        { return ptr->test_and_set(n); });
  }

  auto contains(const AdaptiveBitVector& rhs) const noexcept {
    IFE_SYM(
        rep, rhs.rep, { return lptr->contains(*rptr); },
        {
          if (lptr->count() < rptr->count()) return false;
          for (const auto i : *rptr)
            if (!lptr->test(i)) return false;
          return true;
        })
  }
  auto intersects(const AdaptiveBitVector& rhs) const noexcept {
    IFE_MATCH_SYM(
        rep, rhs.rep, { return lptr->intersects(*rptr); },
        // always iterate over tbv as it's usually smaller
        {
          return std::any_of(lptr->begin(), lptr->end(),
                             [=](auto&& i) { return rptr->test(i); });
        },
        {
          return std::any_of(rptr->begin(), rptr->end(),
                             [=](auto&& i) { return lptr->test(i); });
        });
  }

  auto operator==(const AdaptiveBitVector& rhs) const noexcept {
    IFE_SYM(
        rep, rhs.rep, { return *lptr == *rptr; },
        {
          if (lptr->count() != rhs.count()) return false;
          auto lhs_it = lptr->begin();
          auto rhs_it = rptr->begin();
          for (; lhs_it != lptr->end(); ++lhs_it, ++rhs_it)
            if (*lhs_it != *rhs_it) return false;
          return true;
        });
  }
  auto operator!=(const AdaptiveBitVector& rhs) const noexcept {
    return !(*this == rhs);
  }

  /// Inplace union with rhs.
  /// Returns true if `this` changed.
  auto operator|=(const AdaptiveBitVector& rhs) noexcept {
    IFE_CASE(
        rep, rhs.rep,
        {
          std::optional<ibbv> expanded;
          const auto changed = lptr->inplace_union(*rptr, expanded);
          if (expanded.has_value())
            rep.template emplace<ibbv>(std::move(expanded.value()));
          return changed;
        },
        {
          // tbv |= ibbv
          if (lptr->empty()) {
            if (rptr->empty()) return false;
            rep.template emplace<ibbv>(*rptr);
            return true;
          }
          auto& new_ibbv =
              rep.template emplace<ibbv>(std::move(lptr->expand()));
          return new_ibbv |= *rptr;
        },
        {
          // ibbv |= tbv
          bool changed = false;
          for (const auto i : *rptr)
            changed |= lptr->test_and_set(i);
          return changed;
        },
        { return *lptr |= *rptr; });
  }
  AdaptiveBitVector operator|(const AdaptiveBitVector& rhs) const noexcept {
    auto copy = *this;
    copy |= rhs;
    return copy;
  }

  /// Inplace intersection with rhs.
  /// Returns true if `this` changed.
  auto operator&=(const AdaptiveBitVector& rhs) noexcept {
    IFE_CASE(
        rep, rhs.rep, { return lptr->inplace_intersect(*rptr); },
        {
          // tbv &= ibbv
          return lptr->inplace_intersect(*rptr);
        },
        {
          // ibbv &= tbv
          ibbv temp_ibbv = std::move(*lptr); // save *lptr
          return rep.template emplace<tbv>(*rptr).inplace_intersect(temp_ibbv);
        },
        { return *lptr &= *rptr; })
  }

  /// Inplace difference with rhs.
  /// Returns true if `this` changed.
  bool operator-=(const AdaptiveBitVector& rhs) noexcept {
    IFE_CASE(
        rep, rhs.rep, { return lptr->inplace_diff(*rptr); },
        { return lptr->inplace_diff(*rptr); },
        { // ibbv -= tbv
          bool changed = false;
          for (const auto i : *rptr)
            changed |= lptr->test_and_reset(i);
          return changed;
        },
        { return *lptr -= *rptr; })
  }
  bool intersectWithComplement(const AdaptiveBitVector& rhs) noexcept {
    return *this -= rhs;
  }
  void intersectWithComplement(const AdaptiveBitVector& lhs,
                               const AdaptiveBitVector& rhs) noexcept {
    IFE_CASE(
        lhs.rep, rhs.rep,
        {
          auto& new_rep = rep.template emplace<tbv>();
          new_rep.diff_into_this(*lptr, *rptr);
        },
        {
          auto& new_rep = rep.template emplace<tbv>();
          new_rep.diff_into_this(*lptr, *rptr);
        },
        {
          auto& new_rep = rep.template emplace<ibbv>(*lptr);
          for (const auto i : *rptr)
            new_rep.reset(i);
        },
        { // ibbv -= ibbv (may produce tbv)
          auto& new_rep = rep.template emplace<ibbv>(*lptr);
          new_rep -= *rptr;
          const auto new_count = new_rep.count();
          if (new_count <= MaxTbvSize) { // shrink to tbv
            ibbv temp_ibbv = std::move(new_rep);
            rep.template emplace<tbv>(temp_ibbv.begin(), temp_ibbv.end());
          }
        });
  }

  friend struct std::hash<AdaptiveBitVector>;
};
} // namespace ibbv

namespace std {
template <> struct hash<ibbv::AdaptiveBitVector<>> {
  std::size_t operator()(const ibbv::AdaptiveBitVector<>& v) const noexcept {
    return std::visit(
        [](auto&& arg) {
          return std::hash<std::decay_t<decltype(arg)>>{}(arg);
        },
        v.rep);
  }
};
} // namespace std
#else
#  include "IndexedBlockBitVector.hpp"

namespace ibbv {
template <size_t _ = 0> using AdaptiveBitVector = ibbv::IndexedBlockBitVector<>;
}
#endif
