#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <memory>
#include <vector>

#include "SIMDHelper.h"

#define DEFAULT_COPY(T)                                                        \
  T(const T &) = default;                                                      \
  T &operator=(const T &) = default;
#define DEFAULT_MOVE(T)                                                        \
  T(T &&) = default;                                                           \
  T &operator=(T &&) = default;
#define DEFAULT_COPY_MOVE(T) DEFAULT_COPY(T) DEFAULT_MOVE(T)
#define DO_WHILE0(code)                                                        \
  do {                                                                         \
    code                                                                       \
  } while (0);
#define REPEAT_2(identifier, start_from, block)                                \
  {                                                                            \
    constexpr int identifier = start_from;                                     \
    DO_WHILE0(block)                                                           \
  }                                                                            \
  {                                                                            \
    constexpr int identifier = start_from + 1;                                 \
    DO_WHILE0(block)                                                           \
  }
#define REPEAT_4(identifier, start_from, block)                                \
  REPEAT_2(identifier, start_from, block)                                      \
  REPEAT_2(identifier, (start_from) + 2, block)
#define REPEAT_8(identifier, start_from, block)                                \
  REPEAT_4(identifier, start_from, block)                                      \
  REPEAT_4(identifier, (start_from) + 4, block)
#define REPEAT_i_2(block) REPEAT_2(i, 0, block)
#define REPEAT_i_4(block) REPEAT_4(i, 0, block)

/// A 512-bit vector contains 32*16-bit integers in ascending order,
/// that is, {0, 1, 2, ..., 31} (from e0 to e31).
static const auto asc_indexes =
    _mm512_set_epi16(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
                     16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

static inline size_t szudzik(size_t a, size_t b) {
  return a > b ? b * b + a : a * a + a + b;
}

#define ADV_COUNT                                                              \
  /**the maximum index in current range [this_i..=this_i + 15], spread to      \
   * vector register */                                                        \
  const auto rangemax_this = _mm512_set1_epi32(index_at(this_i + 15)),         \
             rangemax_rhs = _mm512_set1_epi32(rhs.index_at(rhs_i + 15));       \
  /**whether each u32 index is less than or equal to the maximum index in      \
   * current range of the other vector */                                      \
  const uint16_t lemask_this =                                                 \
                     _mm512_cmple_epu32_mask(v_this_idx, rangemax_rhs),        \
                 lemask_rhs =                                                  \
                     _mm512_cmple_epu32_mask(v_rhs_idx, rangemax_this);        \
  /**the number to increase for this_i / rhs_i, <= 16 */                       \
  const auto advance_this = 16 - ibbv::utils::lzcnt(lemask_this),              \
             advance_rhs = 16 - ibbv::utils::lzcnt(lemask_rhs);

namespace ibbv {
using namespace ibbv::utils;
template <uint16_t BlockSize = 128> class IndexedBlockBitVector {
  static_assert(BlockSize == 128,
                "BlockSize other than 128 is unsupported currently");

  template <typename T, size_t Align, size_t Threshold = Align>
  class AlignedAllocatorWithThreshold : public std::allocator<T> {
  public:
    AlignedAllocatorWithThreshold() noexcept {}
    template <class U> struct rebind {
      using other = AlignedAllocatorWithThreshold<U, Align, Threshold>;
    };

    /// Removes default initialization behaviour.
    template <class U> void construct(U *p) noexcept {}

    [[nodiscard]] T *allocate(std::size_t n) {
      const auto nbytes = n * sizeof(T);
      if (nbytes >= Threshold)
        return reinterpret_cast<T *>(
            ::operator new(nbytes, std::align_val_t{Align}));
      else
        return reinterpret_cast<T *>(::operator new(nbytes));
    }

    void deallocate(T *p, std::size_t n) noexcept {
      const auto nbytes = n * sizeof(T);
      if (nbytes >= Threshold)
        ::operator delete(p, std::align_val_t{Align});
      else
        ::operator delete(p);
    }

    bool operator==(const AlignedAllocatorWithThreshold &) const noexcept {
      return true;
    }
    bool operator!=(const AlignedAllocatorWithThreshold &) const noexcept {
      return false;
    }
  };

public:
  using UnitType = uint64_t;
  static constexpr uint16_t UnitBits = sizeof(UnitType) * 8;
  static constexpr uint16_t UnitsPerBlock = BlockSize / UnitBits;

  struct Block {
    UnitType data[UnitsPerBlock];

    /// Default constructor. data is uninitialized.
    Block() noexcept {}
    /// Initialize block with only one bit set.
    Block(size_t index) noexcept : data{} { set(index); }

    DEFAULT_COPY_MOVE(Block);

    /// Returns true if all bits are zero.
    bool empty() const noexcept { return testz<BlockSize>(data); }

    bool test(size_t index) const noexcept {
      const size_t unit_index = index / UnitBits;
      const size_t bit_index = index % UnitBits;
      return data[unit_index] & (static_cast<UnitType>(1) << bit_index);
    }

    void set(size_t index) noexcept {
      const size_t unit_index = index / UnitBits;
      const size_t bit_index = index % UnitBits;
      data[unit_index] |= (static_cast<UnitType>(1) << bit_index);
    }

    /// Returns true if the bit was not set before, and set it.
    bool test_and_set(size_t index) noexcept {
      const size_t unit_index = index / UnitBits;
      const size_t bit_index = index % UnitBits;
      const auto mask = static_cast<UnitType>(1) << bit_index;
      const bool prev_set = data[unit_index] & mask;
      data[unit_index] |= mask;
      return !prev_set; // return true if it was not set before
    }

    void reset(size_t index) noexcept {
      const size_t unit_index = index / UnitBits;
      const size_t bit_index = index % UnitBits;
      data[unit_index] &= ~(static_cast<UnitType>(1) << bit_index);
    }

    bool contains(const Block &rhs) const noexcept {
      return ibbv::utils::contains<BlockSize>(data, rhs.data);
    }

    bool intersects(const Block &rhs) const noexcept {
      return ibbv::utils::intersects<BlockSize>(data, rhs.data);
    }

    bool operator==(const Block &rhs) const noexcept {
      return ibbv::utils::cmpeq<BlockSize>(data, rhs.data);
    }

    bool operator!=(const Block &rhs) const noexcept { return !(*this == rhs); }

    bool operator|=(const Block &rhs) noexcept {
      return ibbv::utils::or_inplace<BlockSize>(data, rhs.data);
    }

    ibbv::utils::ComposedChangeResult operator&=(const Block &rhs) noexcept {
      return ibbv::utils::and_inplace<BlockSize>(data, rhs.data);
    }

    ibbv::utils::ComposedChangeResult operator-=(const Block &rhs) noexcept {
      return ibbv::utils::diff_inplace<BlockSize>(data, rhs.data);
    }
  };
  static_assert(sizeof(Block) * 8 == BlockSize);

  using index_t = int32_t;

protected:
  using index_container =
      std::vector<index_t, AlignedAllocatorWithThreshold<index_t, 64>>;
  using block_container =
      std::vector<Block, AlignedAllocatorWithThreshold<Block, 64>>;
  index_container indexes;
  block_container blocks;

  /// Returns # of blocks.
  _inline size_t size() const noexcept { return indexes.size(); }
  _inline index_t index_at(size_t i) const noexcept { return indexes[i]; }
  _inline Block &block_at(size_t i) noexcept { return blocks[i]; }
  _inline const Block &block_at(size_t i) const noexcept { return blocks[i]; }
  _inline void erase_at(size_t i) noexcept {
    indexes.erase(indexes.begin() + i);
    blocks.erase(blocks.begin() + i);
  }
  _inline void truncate(size_t keep_count) noexcept {
    indexes.resize(keep_count);
    blocks.resize(keep_count);
  }
  template <typename... Args>
  _inline void emplace_at(size_t i, const index_t &ind,
                          Args &&...blkArgs) noexcept {
    indexes.emplace(indexes.begin() + i, ind);
    blocks.emplace(blocks.begin() + i, std::forward<Args>(blkArgs)...);
  }

public:
  class IndexedBlockBitVectorIterator {
  public:
    using iterator_category = std::input_iterator_tag;
    using value_type = index_t;
    using difference_type = std::ptrdiff_t;
    using pointer = const index_t *;
    using reference = const index_t &;
    index_t cur_pos;

  protected:
    typename index_container::const_iterator indexIt;
    typename index_container::const_iterator indexEnd;
    typename block_container::const_iterator blockIt;
    unsigned char unit_index; // unit index in the current block
    unsigned char bit_index;  // bit index in the current unit
    bool end;
    /// move to next unit, returns false if reached the end
    void incr_unit() {
      bit_index = 0;
      if (++unit_index == UnitsPerBlock) {
        // forward to next block
        unit_index = 0;
        ++indexIt, ++blockIt;
        if (indexIt == indexEnd)
          // reached the end
          end = true;
      }
    }
    /// Increment the bit index (mark the current bit as visited).
    void incr_bit() {
      if (++bit_index == UnitBits)
        // forward to next unit
        incr_unit();
    }
    /// Start from current position and search for the next set bit
    void search() {
      while (!end) {
        auto mask = ~((static_cast<UnitType>(1) << bit_index) - 1);
        auto masked_unit = blockIt->data[unit_index] & mask;
        auto tz_count = ibbv::utils::tzcnt(masked_unit);
        if (tz_count < UnitBits) {
          // found a set bit
          bit_index = tz_count;
          cur_pos = *indexIt + unit_index * UnitBits + bit_index;
          return;
        } else // move to next unit
          incr_unit();
      }
    }
    /// Step out from the current position and search for the next set bit.
    void forward() {
      incr_bit();
      search();
    }

  public:
    IndexedBlockBitVectorIterator() = delete;
    IndexedBlockBitVectorIterator(const IndexedBlockBitVector &vec,
                                  bool end = false)
        : // must be init to identify raw vector
          indexEnd(vec.indexes.end()), end(end | vec.empty()) {
      if (end)
        return;
      indexIt = vec.indexes.begin();
      blockIt = vec.blocks.begin();
      unit_index = 0;
      bit_index = 0;
      search();
    }
    DEFAULT_COPY_MOVE(IndexedBlockBitVectorIterator);

    reference operator*() const { return cur_pos; }
    pointer operator->() const { return &cur_pos; }
    IndexedBlockBitVectorIterator &operator++() {
      forward();
      return *this;
    }
    IndexedBlockBitVectorIterator operator++(int) {
      IndexedBlockBitVectorIterator temp = *this;
      ++*this;
      return temp;
    }
    bool operator==(const IndexedBlockBitVectorIterator &other) const {
      return
          // created from the same vector, and
          indexEnd == other.indexEnd &&
          (( // both ended, or
               end && other.end) ||
           ( // both not ended and pointing to the same position
               end == other.end && cur_pos == other.cur_pos));
    }
    bool operator!=(const IndexedBlockBitVectorIterator &other) const {
      return !(*this == other);
    }
  };
  using iterator = IndexedBlockBitVectorIterator;

  /// Returns an iterator to the beginning of this vector.
  /// NOTE: If you modify the vector after creating an iterator, the iterator
  /// is not stable and may cause UB if used.
  IndexedBlockBitVectorIterator begin() const {
    return IndexedBlockBitVectorIterator(*this);
  }
  IndexedBlockBitVectorIterator end() const {
    return IndexedBlockBitVectorIterator(*this, true);
  }

  /// Return the first set bit in the bitmap.  Return -1 if no bits are set.
  index_t find_first() const {
    if (empty())
      return -1;
    return *begin();
  }

  /// Return the last set bit in the bitmap.  Return -1 if no bits are set.
  index_t find_last() const {
    if (empty())
      return -1;
    const auto &last_blk = blocks.back();
    index_t last_index = indexes.back() + BlockSize - 1;
    for (auto i = UnitsPerBlock - 1; i >= 0; i--) {
      const auto cnt = ibbv::utils::lzcnt(last_blk.data[i]);
      last_index -= cnt;
      if (cnt < UnitBits)
        return last_index;
    }
    __builtin_unreachable(); // all bits are zero ?
  }

  /// Construct empty vector
  IndexedBlockBitVector(void) {}

  /// Copy constructor
  IndexedBlockBitVector(const IndexedBlockBitVector &other) = default;

  /// Move constructor
  IndexedBlockBitVector(IndexedBlockBitVector &&other) noexcept = default;

  /// Copy assignment
  IndexedBlockBitVector &
  operator=(const IndexedBlockBitVector &other) = default;

  /// Move assignment
  IndexedBlockBitVector &
  operator=(IndexedBlockBitVector &&other) noexcept = default;

  /// Returns true if no bits are set.
  bool empty() const noexcept { return indexes.empty(); }

  /// Returns the count of set bits.
  uint32_t count() const noexcept {
    // TODO: improve
    if (size() == 0)
      return 0;

#if __AVX512VPOPCNTDQ__ && __AVX512VL__
    auto it = blocks.begin();
    const auto v0 = avx_vec<BlockSize>::load(&(it->data));
    auto c = avx_vec<BlockSize>::popcnt(v0);
    ++it;
    for (; it != blocks.end(); ++it) {
      const auto curv = avx_vec<BlockSize>::load(&(it->data));
      const auto curc = avx_vec<BlockSize>::popcnt(curv);
      c = avx_vec<BlockSize>::add_op(c, curc);
    }
    return avx_vec<BlockSize>::reduce_add(c);
#else
    uint32_t result = 0;
    auto arr = reinterpret_cast<const uint32_t *>(this->blocks.data());
    for (size_t i = 0; i < size() * (sizeof(Block) / 8); ++i, ++arr)
      result += ibbv::utils::popcnt(*arr);
    return result;
#endif
  }

  /// Empty the set.
  void clear() noexcept {
    indexes.clear();
    blocks.clear();
  }

  /// Returns true if n is in this set.
  bool test(index_t n) const noexcept {
    const auto target_ind = n - (n % BlockSize);
    const auto low_pos =
        std::lower_bound(indexes.begin(), indexes.end(), target_ind);
    const auto i = std::distance(indexes.begin(), low_pos);
    if (low_pos == indexes.end() || *low_pos != target_ind) // not found
      return false;
    else
      return block_at(i).test(n % BlockSize);
  }

  void set(index_t n) noexcept {
    const auto target_ind = n - (n % BlockSize);
    const auto low_pos =
        std::lower_bound(indexes.begin(), indexes.end(), target_ind);
    const auto i = std::distance(indexes.begin(), low_pos);
    if (low_pos == indexes.end() || *low_pos != target_ind) // not found
      emplace_at(i, target_ind, n % BlockSize);
    else
      return block_at(i).set(n % BlockSize);
  }

  /// Check if bit is set. If it is, returns false.
  /// Otherwise, sets bit and returns true.
  bool test_and_set(index_t n) noexcept {
    const auto target_ind = n - (n % BlockSize);
    const auto low_pos =
        std::lower_bound(indexes.begin(), indexes.end(), target_ind);
    const auto i = std::distance(indexes.begin(), low_pos);
    if (low_pos == indexes.end() || *low_pos != target_ind) { // not found
      emplace_at(i, target_ind, n % BlockSize);
      return true;
    } else
      return block_at(i).test_and_set(n % BlockSize);
  }

  void reset(index_t n) noexcept {
    const auto target_ind = n - (n % BlockSize);
    const auto low_pos =
        std::lower_bound(indexes.begin(), indexes.end(), target_ind);
    if (low_pos == indexes.end() || *low_pos != target_ind)
      return; // not found
    const auto i = std::distance(indexes.begin(), low_pos);
    auto &d = block_at(i);
    d.reset(n % BlockSize);
    if (d.empty())
      erase_at(i); // this block is empty, remove it
  }

  /// Returns true if this set contains all bits of rhs.
  bool contains_simd(const IndexedBlockBitVector &rhs) const noexcept {
    const auto this_size = size(), rhs_size = rhs.size();
    if (this_size < rhs_size)
      return false;

    size_t this_i = 0, rhs_i = 0;
    const auto all_zero = _mm512_setzero_si512();
    while (this_i + 16 <= this_size && rhs_i + 16 <= rhs_size) {
      /// indexes[this_i..this_i + 16]
      const auto v_this_idx = _mm512_loadu_epi32(indexes.data() + this_i),
                 v_rhs_idx = _mm512_loadu_epi32(rhs.indexes.data() + rhs_i);
      /// whether each u32 matches (exist in both vectors)
      uint16_t match_this, match_rhs;
      ne_mm512_2intersect_epi32(v_this_idx, v_rhs_idx, match_this, match_rhs);

      ADV_COUNT;

      /// count of matched indexes
      const auto n_matched = ibbv::utils::popcnt((uint32_t)match_this);

      if (advance_rhs > n_matched)
        return false;

      /// compress the intersected data's offset(/8bytes).
      const auto gather_this_offset_u16x32 =
                     _mm512_maskz_compress_epi32(match_this, asc_indexes),
                 gather_rhs_offset_u16x32 =
                     _mm512_maskz_compress_epi32(match_rhs, asc_indexes);
      const auto gather_this_base_addr = block_at(this_i).data;
      const auto gather_rhs_base_addr = rhs.block_at(rhs_i).data;

      const uint32_t n_matched_bits_dup = ((uint64_t)1 << (n_matched * 2)) - 1;

      REPEAT_i_4({ // required for `i` to be const
        const auto cur_gather_this_offset_u16x8 =
            _mm512_extracti64x2_epi64(gather_this_offset_u16x32, i);
        const auto cur_gather_rhs_offset_u16x8 =
            _mm512_extracti64x2_epi64(gather_rhs_offset_u16x32, i);

        const auto cur_gather_this_offset_u64x8 =
            _mm512_cvtepu16_epi64(cur_gather_this_offset_u16x8);
        const auto cur_gather_rhs_offset_u64x8 =
            _mm512_cvtepu16_epi64(cur_gather_rhs_offset_u16x8);

        const auto intersect_this = _mm512_mask_i64gather_epi64(
            all_zero, n_matched_bits_dup >> i * 8, cur_gather_this_offset_u64x8,
            gather_this_base_addr, 8);
        const auto intersect_rhs = _mm512_mask_i64gather_epi64(
            all_zero, n_matched_bits_dup >> i * 8, cur_gather_rhs_offset_u64x8,
            gather_rhs_base_addr, 8);

        const auto and_result = _mm512_and_epi64(intersect_this, intersect_rhs);

        if (!avx_vec<512>::eq_cmp(and_result, intersect_rhs))
          return false;
      });
      this_i += advance_this, rhs_i += advance_rhs;
    }
    while (this_i < size() && rhs_i < rhs.size()) {
      const auto this_ind = index_at(this_i);
      const auto rhs_ind = rhs.index_at(rhs_i);
      if (this_ind > rhs_ind)
        return false;
      if (this_ind < rhs_ind)
        ++this_i;
      else {
        if (!block_at(this_i).contains(rhs.block_at(rhs_i)))
          return false;
        ++this_i, ++rhs_i;
      }
    }
    return rhs_i == rhs.size();
  }

  bool contains_loop(const IndexedBlockBitVector &rhs) const noexcept {
    size_t this_i = 0, rhs_i = 0;
    while (this_i < size() && rhs_i < rhs.size()) {
      const auto this_ind = index_at(this_i);
      const auto rhs_ind = rhs.index_at(rhs_i);
      if (this_ind > rhs_ind)
        return false;
      if (this_ind < rhs_ind)
        ++this_i;
      else {
        if (!block_at(this_i).contains(rhs.block_at(rhs_i)))
          return false;
        ++this_i, ++rhs_i;
      }
    }
    return rhs_i == rhs.size();
  }

  bool contains(const IndexedBlockBitVector &rhs) const noexcept {
    return contains_simd(rhs);
    // auto simd_result = contains_simd(rhs), loop_result =
    // contains_loop(rhs); assert(simd_result == loop_result); return
    // simd_result;
  }

  // TODO: use SIMD to improve perf
  /// Returns true if this set and rhs share any bits.
  bool intersects(const IndexedBlockBitVector &rhs) const noexcept {
    size_t this_i = 0, rhs_i = 0;
    while (this_i < size() && rhs_i < rhs.size()) {
      const auto this_ind = index_at(this_i);
      const auto rhs_ind = rhs.index_at(rhs_i);
      if (this_ind > rhs_ind)
        ++rhs_i;
      else if (this_ind < rhs_ind)
        ++this_i;
      else {
        if (this->block_at(this_i).intersects(rhs.block_at(rhs_i)))
          return true;
        ++this_i, ++rhs_i;
      }
    }
    return false;
  }

  bool operator==(const IndexedBlockBitVector &rhs) const noexcept {
    if (size() != rhs.size())
      return false;
    return std::memcmp(indexes.data(), rhs.indexes.data(),
                       sizeof(index_t) * size()) == 0 &&
           std::memcmp(blocks.data(), rhs.blocks.data(),
                       sizeof(Block) * size()) == 0;
  }

  bool operator!=(const IndexedBlockBitVector &rhs) const noexcept {
    return !(*this == rhs);
  }

  bool union_simd3(const IndexedBlockBitVector &rhs) {
    // Update `this` inplace, save extra blocks to temp
    const auto this_size = size(), rhs_size = rhs.size();
    size_t this_i = 0, rhs_i = 0;
    bool changed = false;
    size_t extra_count = 0;
    index_t *extra_indexes = (index_t *)std::malloc(sizeof(index_t) * rhs_size);
    Block *extra_blocks = (Block *)std::malloc(sizeof(Block) * rhs_size);

    alignas(64) Block rhs_block_temp[512 / (sizeof(index_t) * 8)];
    while (this_i + 16 <= this_size && rhs_i + 16 <= rhs_size) {
      /// indexes[this_i..this_i + 16]
      const auto v_this_idx = _mm512_loadu_epi32(indexes.data() + this_i),
                 v_rhs_idx = _mm512_loadu_epi32(rhs.indexes.data() + rhs_i);
      /// whether each u32 matches (exist in both vectors)
      uint16_t match_this, match_rhs;
      ne_mm512_2intersect_epi32(v_this_idx, v_rhs_idx, match_this, match_rhs);

      ADV_COUNT;

      auto match_this_temp = match_this;
      Block *store_blk_base = rhs_block_temp;
      for (auto i = 0; i < advance_rhs; ++i) {
        const auto &rhs_block_cur = rhs.block_at(rhs_i + i);
        if (match_rhs & (1 << i)) { // check bits[i]
          // copy current rhs block into right pos
          const auto this_pad = ibbv::utils::tzcnt(match_this_temp);
          store_blk_base += this_pad;
          *store_blk_base = rhs_block_cur;
          ++store_blk_base;
          match_this_temp >>= this_pad + 1;
        } else { // current rhs block is extra
          extra_indexes[extra_count] = rhs.index_at(rhs_i + i);
          extra_blocks[extra_count] = rhs_block_cur;
          ++extra_count;
        }
      }

      /// each u32 index match => 2*u64 (data) to store & compute
      auto dup_match_this = duplicate_bits(match_this);

      // compute OR result of matched blocks
      REPEAT_i_4({
        /// matched & ordered 4 blocks (8 u64) from memory. zero in
        /// case of out of bounds
        const auto v_this = _mm512_loadu_epi64(&block_at(this_i + i * 4));
        const auto v_rhs = _mm512_maskz_loadu_epi64(dup_match_this >> (i * 8),
                                                    &rhs_block_temp[i * 4]);
        const auto or_result = _mm512_or_epi64(v_this, v_rhs);

        if (!changed) // compute `changed` if not already set
          changed = !avx_vec<512>::eq_cmp(v_this, or_result);

        _mm512_storeu_epi64(&block_at(this_i + i * 4), or_result);
      });
      this_i += advance_this, rhs_i += advance_rhs;
    }

    // deal with remaining elements
    while (this_i < this_size && rhs_i < rhs_size) {
      const auto this_ind = index_at(this_i);
      const auto rhs_ind = rhs.index_at(rhs_i);
      if (this_ind < rhs_ind)
        ++this_i;
      else if (this_ind > rhs_ind) {
        // copy current rhs block to temp
        extra_indexes[extra_count] = rhs_ind;
        extra_blocks[extra_count] = rhs.block_at(rhs_i);
        ++extra_count;
        ++rhs_i;
      } else { // this_ind == rhs_ind
        changed |= (block_at(this_i) |= rhs.block_at(rhs_i));
        ++this_i, ++rhs_i;
      }
    }
    /// whether any extra blocks are added (requires merge sort)
    const bool any_extra = extra_count > 0;
    /// whether any remaining blocks in rhs (extra & largest)
    const bool rhs_remaining = rhs_i < rhs_size;
    changed |= any_extra || rhs_remaining;
    const auto new_size = this_size + extra_count + (rhs_size - rhs_i);
    this->indexes.resize(new_size);
    this->blocks.resize(new_size);
    if (any_extra) {
      // inplace merge sort (also arrange blocks).
      signed int dest_i = this_size + extra_count - 1, src_i = this_size - 1,
                 extra_i = extra_count - 1;
      while (src_i >= 0 && extra_i >= 0) {
        if (index_at(src_i) > extra_indexes[extra_i]) {
          indexes[dest_i] = index_at(src_i);
          blocks[dest_i] = block_at(src_i);
          --src_i;
        } else {
          indexes[dest_i] = extra_indexes[extra_i];
          blocks[dest_i] = extra_blocks[extra_i];
          --extra_i;
        }
        --dest_i;
      }
      if (extra_i >= 0) {
        // copy remaining extra blocks
        std::memcpy(&indexes[0], &extra_indexes[0],
                    sizeof(index_t) * (extra_i + 1));
        std::memcpy(&blocks[0], &extra_blocks[0],
                    sizeof(Block) * (extra_i + 1));
      }
      // else if src_i >= 0, they are already in place
    }
    std::free(extra_indexes);
    std::free(extra_blocks);
    if (rhs_remaining) { // remaining elements in rhs are always largest
      std::memcpy(&indexes[this_size + extra_count], &rhs.indexes[rhs_i],
                  sizeof(index_t) * (rhs_size - rhs_i));
      std::memcpy(&blocks[this_size + extra_count], &rhs.blocks[rhs_i],
                  sizeof(Block) * (rhs_size - rhs_i));
    }
    return changed;
  }

  /// Inplace union with rhs.
  /// Returns true if this set changed.
  bool operator|=(const IndexedBlockBitVector &rhs) { return union_simd3(rhs); }

  IndexedBlockBitVector operator|(const IndexedBlockBitVector &rhs) const {
    IndexedBlockBitVector copy(*this);
    copy |= rhs;
    return copy;
  }

  /// Inplace intersection with rhs.
  /// Returns true if this set changed.
  /// Optimized using AVX512 intrinsics, requires AVX512F inst set.
  bool intersect_simd(const IndexedBlockBitVector &rhs) {
    const auto this_size = size(), rhs_size = rhs.size();
    size_t valid_count = 0, this_i = 0, rhs_i = 0;
    bool changed = false;
    const auto all_zero = _mm512_setzero_si512();
    while (this_i + 16 <= this_size && rhs_i + 16 <= rhs_size) {
      /// indexes[this_i..this_i + 16]
      const auto v_this_idx = _mm512_loadu_epi32(indexes.data() + this_i),
                 v_rhs_idx = _mm512_loadu_epi32(rhs.indexes.data() + rhs_i);
      /// whether each u32 matches (exist in both vectors)
      uint16_t match_this, match_rhs;
      ne_mm512_2intersect_epi32(v_this_idx, v_rhs_idx, match_this, match_rhs);

      ADV_COUNT;

      /// compress the intersected data's offset(/8bytes).
      const auto gather_this_offset_u16x32 =
                     _mm512_maskz_compress_epi32(match_this, asc_indexes),
                 gather_rhs_offset_u16x32 =
                     _mm512_maskz_compress_epi32(match_rhs, asc_indexes);
      const auto gather_this_base_addr = block_at(this_i).data;
      const auto gather_rhs_base_addr = rhs.block_at(rhs_i).data;

      const auto ordered_indexes =
          _mm512_maskz_compress_epi32(match_this, v_this_idx);

      const auto n_matched = _mm_popcnt_u32(match_this);
      const uint32_t n_matched_bits_dup = ((uint64_t)1 << (n_matched * 2)) - 1;

      // compute AND result of matched blocks
      REPEAT_i_4({
        const auto cur_gather_this_offset_u16x8 =
            _mm512_extracti64x2_epi64(gather_this_offset_u16x32, i);
        const auto cur_gather_rhs_offset_u16x8 =
            _mm512_extracti64x2_epi64(gather_rhs_offset_u16x32, i);

        const auto cur_gather_this_offset_u64x8 =
            _mm512_cvtepu16_epi64(cur_gather_this_offset_u16x8);
        const auto cur_gather_rhs_offset_u64x8 =
            _mm512_cvtepu16_epi64(cur_gather_rhs_offset_u16x8);

        /// matched & ordered 4 blocks (8 u64) from memory. zero
        /// in case of out of bounds
        const auto intersect_this = _mm512_mask_i64gather_epi64(
            all_zero, n_matched_bits_dup >> i * 8, cur_gather_this_offset_u64x8,
            gather_this_base_addr, 8);
        const auto intersect_rhs = _mm512_mask_i64gather_epi64(
            all_zero, n_matched_bits_dup >> i * 8, cur_gather_rhs_offset_u64x8,
            gather_rhs_base_addr, 8);
        const auto and_result = _mm512_and_epi64(intersect_this, intersect_rhs);

        if (!changed) // compute `changed` if not already set
          changed = !avx_vec<512>::eq_cmp(intersect_this, and_result);

        /// zero-test for each u64, 1 means nonzero
        const auto nzero_mask = _mm512_test_epi64_mask(and_result, and_result);
        // _bit[2k] := bit[2k] | bit[2k+1]
        uint8_t nzero_mask_by_block =
            nzero_mask | ((nzero_mask & 0b10101010) >> 1);
        // _bit[k] := bit[2k] | bit[2k+1]
        const auto nzero_compressed =
            _pext_u32(nzero_mask_by_block, 0b01010101);
        // _bit[2k+1] := bit[2k] | bit[2k+1],
        // compute here to improve pipeline perf
        nzero_mask_by_block |= ((nzero_mask & 0b01010101) << 1);
        // store new index & data for 4 blocks (remove empty
        // blocks)
        _mm512_mask_compressstoreu_epi32(indexes.data() + valid_count,
                                         nzero_compressed << (i * 4),
                                         ordered_indexes);
        _mm512_mask_compressstoreu_epi64(blocks.data() + valid_count,
                                         nzero_mask_by_block, and_result);

        const auto nzero_count = _mm_popcnt_u32(nzero_compressed);
        valid_count += nzero_count;
      });
      this_i += advance_this, rhs_i += advance_rhs;
    }
    // use trival loop for the rest
    // TODO: improve
    while (this_i < this_size && rhs_i < rhs_size) {
      const auto this_ind = index_at(this_i);
      const auto rhs_ind = rhs.index_at(rhs_i);
      if (this_ind < rhs_ind) {
        // ignore this block (not copied to valid position)
        ++this_i;
        changed = true;
      } else if (this_ind > rhs_ind)
        ++rhs_i;
      else { // this_ind == rhs_ind
        const auto vcur_this = avx_vec<BlockSize>::load(block_at(this_i).data),
                   vcur_rhs =
                       avx_vec<BlockSize>::load(rhs.block_at(rhs_i).data);
        const auto and_result = avx_vec<BlockSize>::and_op(vcur_this, vcur_rhs);
        if (avx_vec<BlockSize>::is_zero(and_result))
          // no bits in common, skip this block
          changed = true;
        else {
          if (!changed) // compute `changed` if not already set
            changed = !avx_vec<BlockSize>::eq_cmp(vcur_this, and_result);

          // store the result
          avx_vec<BlockSize>::store(block_at(valid_count).data, and_result);
          indexes[valid_count] = this_ind;
          ++valid_count; // increment valid count
        }
        ++this_i, ++rhs_i;
      }
    }
    truncate(valid_count);
    changed |= (valid_count != this_size);
    return changed;
  }
  /// Inplace intersection with rhs.
  /// Returns true if this set changed.
  bool operator&=(const IndexedBlockBitVector &rhs) {
    return intersect_simd(rhs);
  }

  /// Inplace difference with rhs.
  /// Returns true if this set changed.
  bool diff_simd(const IndexedBlockBitVector &rhs) {
    const auto this_size = size(), rhs_size = rhs.size();
    size_t valid_count = 0, this_i = 0, rhs_i = 0;
    bool changed = false;
    alignas(64) Block rhs_block_temp[512 / (sizeof(index_t) * 8)];
    while (this_i + 16 <= this_size && rhs_i + 16 <= rhs_size) {
      /// indexes[this_i..this_i + 16]
      const auto v_this_idx = _mm512_loadu_epi32(indexes.data() + this_i),
                 v_rhs_idx = _mm512_loadu_epi32(rhs.indexes.data() + rhs_i);
      /// whether each u32 matches (exist in both vectors)
      uint16_t match_this, match_rhs;
      ne_mm512_2intersect_epi32(v_this_idx, v_rhs_idx, match_this, match_rhs);

      ADV_COUNT;

      // align matched data of rhs to the shape of this.
      // we don't care unmatched position
      auto matched_this_temp = match_this, matched_rhs_temp = match_rhs;
      auto rhs_block_temp_addr = rhs_block_temp,
           rhs_block_addr = &rhs.block_at(rhs_i);
      while (matched_this_temp) {
        const auto this_pad = ibbv::utils::tzcnt(matched_this_temp),
                   rhs_pad = ibbv::utils::tzcnt(matched_rhs_temp);
        rhs_block_temp_addr += this_pad, rhs_block_addr += rhs_pad;
        *rhs_block_temp_addr = *rhs_block_addr;
        matched_this_temp >>= this_pad + 1, matched_rhs_temp >>= rhs_pad + 1;
        ++rhs_block_temp_addr, ++rhs_block_addr;
      }

      /// each u32 index match => 2*u64 (data) to store & compute
      auto dup_this = duplicate_bits(match_this);
      const uint16_t advance_this_to_bits = ((uint32_t)1 << advance_this) - 1;
      const auto dup_advance_this = duplicate_bits(advance_this_to_bits);

      // compute AND result of matched blocks
      REPEAT_i_4({
        /// matched & ordered 4 blocks (8 u64) from memory. zero
        /// in case of out of bounds
        const auto v_this = _mm512_maskz_loadu_epi64(
            dup_advance_this >> (i * 8), &block_at(this_i) + i * 4);
        const auto v_rhs = _mm512_maskz_loadu_epi64(dup_this >> (i * 8),
                                                    &rhs_block_temp[i * 4]);
        // _mm512_andnot_epi64 intrinsic: NOT of 512 bits (composed
        // of packed 64-bit integers) in a and then AND with b
        const auto andnot_result = _mm512_andnot_epi64(v_rhs, v_this);

        if (!changed) // compute `changed` if not already set
          changed = !avx_vec<512>::eq_cmp(v_this, andnot_result);

        /// zero-test for each u64, 1 means nonzero
        const auto nzero_mask =
            _mm512_test_epi64_mask(andnot_result, andnot_result);
        // _bit[2k] := bit[2k] | bit[2k+1]
        uint8_t nzero_mask_by_block =
            nzero_mask | ((nzero_mask & 0b10101010) >> 1);
        // _bit[k] := bit[2k] | bit[2k+1]
        const auto nzero_compressed =
            _pext_u32(nzero_mask_by_block, 0b01010101);
        // _bit[2k+1] := bit[2k] | bit[2k+1],
        // compute here to improve pipeline perf
        nzero_mask_by_block |= ((nzero_mask & 0b01010101) << 1);
        // store new index & data for 4 blocks (remove empty
        // blocks)
        _mm512_mask_compressstoreu_epi32(indexes.data() + valid_count,
                                         nzero_compressed << (i * 4),
                                         v_this_idx);
        _mm512_mask_compressstoreu_epi64(blocks.data() + valid_count,
                                         nzero_mask_by_block, andnot_result);

        const auto nzero_count = _mm_popcnt_u32(nzero_compressed);
        valid_count += nzero_count;
      });
      this_i += advance_this, rhs_i += advance_rhs;
    }
    while (this_i < this_size && rhs_i < rhs_size) {
      const auto this_ind = index_at(this_i);
      const auto rhs_ind = rhs.index_at(rhs_i);
      if (this_ind < rhs_ind) { // keep this block
        indexes[valid_count] = this_ind;
        block_at(valid_count) = block_at(this_i);
        ++this_i;
        ++valid_count;
      } else if (this_ind > rhs_ind)
        ++rhs_i;
      else { // compute ANDNOT
        const auto v_this = avx_vec<BlockSize>::load(&block_at(this_i));
        const auto v_rhs = avx_vec<BlockSize>::load(&rhs.block_at(rhs_i));
        const auto andnot_result = avx_vec<BlockSize>::andnot_op(v_this, v_rhs);
        if (avx_vec<BlockSize>::is_zero(andnot_result)) // changed to zero
          changed = true;
        else {
          if (!changed)
            changed = !avx_vec<BlockSize>::eq_cmp(v_this, andnot_result);
          indexes[valid_count] = this_ind;
          avx_vec<BlockSize>::store(&block_at(valid_count), andnot_result);
          valid_count++;
        }
        ++this_i, ++rhs_i;
      }
    }
    // the rest element is kept
    const auto rest_count = this_size - this_i;
    std::memmove(&indexes[valid_count], &indexes[this_i],
                 sizeof(index_t) * rest_count);
    std::memmove(&block_at(valid_count), &block_at(this_i),
                 sizeof(Block) * rest_count);
    valid_count += rest_count;
    truncate(valid_count);
    changed |= (valid_count != this_size);
    return changed;
  }
  bool operator-=(const IndexedBlockBitVector &rhs) { return diff_simd(rhs); }
  bool intersectWithComplement(const IndexedBlockBitVector &rhs) {
    return *this -= rhs;
  }
  void intersectWithComplement(const IndexedBlockBitVector &lhs,
                               const IndexedBlockBitVector &rhs) {
    // TODO: inefficient!
    *this = lhs;
    intersectWithComplement(rhs);
  }

  size_t hash() const noexcept {
    return szudzik(count(), szudzik(size(), size() > 0 ? *begin() : -1));
  }
};
} // namespace ibbv
