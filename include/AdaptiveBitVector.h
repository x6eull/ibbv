#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <variant>

#include "IndexedBlockBitVector.h"
#include "TinyBitVector.h"

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
#define IFE_TBV(ptrvarname, variant, then_block, else_block)                   \
  if (auto* ptrvarname = std::get_if<tbv>(&variant)) {                         \
    then_block                                                                 \
  } else {                                                                     \
    {                                                                          \
      auto* ptrvarname = std::get_if<ibbv>(&variant);                          \
      else_block                                                               \
    }                                                                          \
  }
/// 4 cases
#define IFE_CASE(variant_lhs, variant_rhs, t_t_block, t_i_block, i_t_block,    \
                 i_i_block)                                                    \
  IFE_TBV(                                                                     \
      lptr, variant_lhs,                                                       \
      { IFE_TBV(rptr, variant_rhs, t_t_block, t_i_block); },                   \
      { IFE_TBV(rptr, variant_rhs, i_t_block, i_i_block); })

/// 3 cases: matched types are processed in the same way
#define IFE_MATCH_SYM(variant_lhs, variant_rhs, match_block, t_i_block,        \
                      i_t_block)                                               \
  IFE_TBV(lptr, variant_lhs,                                                   \
          IFE_TBV(rptr, variant_rhs, match_block, t_i_block),                  \
          IFE_TBV(rptr, variant_rhs, i_t_block, match_block))

/// 2 cases: unmatched types are also processed in the same way
#define IFE_SYM(variant_lhs, variant_rhs, match_block, unmatch_block)          \
  IFE_MATCH_SYM(variant_lhs, variant_rhs, match_block, unmatch_block,          \
                unmatch_block)

public:
  auto impl_type() const noexcept {
    return rep.index();
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
    AdaptiveBitVectorIterator(const decltype(rep)& vec, bool end = false)
        : it_rep(std::visit(
              [=](auto&& arg) {
                return decltype(it_rep){end ? arg.end() : arg.begin()};
              },
              vec)) {}

    reference operator*() const {
      return std::visit([](auto&& arg) -> reference { return (*arg); }, it_rep);
    }
    pointer operator->() const {
      return std::visit([](auto&& arg) -> pointer { return (&*arg); }, it_rep);
    }
    AdaptiveBitVectorIterator& operator++() {
      std::visit([](auto&& arg) { arg++; }, it_rep);
      return *this;
    }
    AdaptiveBitVectorIterator operator++(int) {
      AdaptiveBitVectorIterator temp = *this;
      ++*this;
      return temp;
    }
    bool operator==(const AdaptiveBitVectorIterator& other) const {
      return it_rep == other.it_rep;
    }
    bool operator!=(const AdaptiveBitVectorIterator& other) const {
      return !(*this == other);
    }
  };
  using iterator = AdaptiveBitVectorIterator;
  auto begin() const {
    return iterator(rep);
  }
  auto end() const {
    return iterator(rep, true);
  }
  auto find_first() const noexcept {
    return std::visit([](auto&& arg) { return arg.find_first(); }, rep);
  }
  auto empty() const noexcept {
    return std::visit([](auto&& arg) { return arg.empty(); }, rep);
  }
  auto count() const noexcept {
    return std::visit([](auto&& arg) { return (size_t)arg.count(); }, rep);
  }
  auto clear() noexcept {
    rep.template emplace<tbv>();
  }
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
          tbv new_rep{*rptr};
          const auto result = new_rep.inplace_intersect(*lptr);
          rep.template emplace<tbv>(std::move(new_rep));
          return result;
        },
        { return *lptr &= *rptr; })
  }

  /// Inplace difference with rhs.
  /// Returns true if `this` changed.
  bool operator-=(const AdaptiveBitVector& rhs) noexcept {
    IFE_CASE(
        rep, rhs.rep, { return lptr->inplace_diff(*rptr); },
        { return lptr->inplace_diff(*rptr); },
        {
          bool changed = false;
          for (const auto i : *rptr) {
            changed |= lptr->test(i);
            lptr->reset(i);
          }
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
        {
          auto& new_rep = rep.template emplace<ibbv>(*lptr);
          new_rep -= *rptr;
          const auto new_count = new_rep.count();
          if (new_count <= MaxTbvSize) { // shrink to tbv
            index_t tmp[MaxTbvSize];
            const auto tmp_end = std::copy(new_rep.begin(), new_rep.end(),
                                           static_cast<index_t*>(tmp));
            auto& tbv_rep = rep.template emplace<tbv>();
            std::copy(static_cast<index_t*>(tmp), tmp_end, tbv_rep.begin());
            tbv_rep.resize_to(tbv_rep.begin() + new_count);
          }
        });
  }

  friend struct std::hash<AdaptiveBitVector>;
};
} // namespace ibbv

namespace std {
template <> struct hash<ibbv::AdaptiveBitVector<>> {
  std::size_t operator()(const ibbv::AdaptiveBitVector<>& v) const {
    return std::visit(
        [](auto&& arg) {
          return std::hash<std::decay_t<decltype(arg)>>{}(arg);
        },
        v.rep);
  }
};
} // namespace std