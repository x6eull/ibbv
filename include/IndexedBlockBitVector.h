#pragma once
static_assert(__AVX512F__, "AVX512F is required for IndexedBlockBitVector");
// The least requirement for IIBV is AVX512F instruction set.
// Certain operation is more efficient if corresponding instruction set
//  is available:
// AVX512_VP2INTERSECT: speed up set union/intersect/subset testing, and
//  other algorithms using vectorized index matching.
// AVX512VPOPCNTDQ: speed up count() method.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <iterator>
#include <mimalloc.h>
#include <utility>

#if IBBV_COUNT_OP
#  include "Counter.h"
#endif
#include "bit_utils.h"

#define DEFAULT_COPY_MOVE(T)                                                   \
  T(const T&) noexcept = default;                                              \
  T& operator=(const T&) noexcept = default;                                   \
  T(T&&) noexcept = default;                                                   \
  T& operator=(T&&) noexcept = default;

namespace ibbv {
using namespace ibbv::utils;
/// A variant of sparse bit vector that utilizes SIMD technology to acclerate
/// computation.
/// The implementation is not thread-safe.
template <uint16_t BlockBits = 128> class IndexedBlockBitVector {
  static_assert(BlockBits == 128,
                "BlockSize other than 128 is unsupported currently");

public:
  template <size_t MaxTbvSize> friend class AdaptiveBitVector;
  using UnitType = uint64_t;
  static inline constexpr uint16_t UnitBits = sizeof(UnitType) * 8;
  static inline constexpr uint16_t UnitsPerBlock = BlockBits / UnitBits;
  template <typename... Args>
  IndexedBlockBitVector(Args&&... args)
      : storage{std::forward<Args>(args)...} {}

  struct Block {
    UnitType data[UnitsPerBlock];

    /// Default constructor. data is uninitialized.
    Block() noexcept = default;
    /// Initialize block with only one bit set.
    Block(size_t index) noexcept : data{} {
      set(index);
    }

    DEFAULT_COPY_MOVE(Block);

    /// Returns true if all bits are zero.
    bool empty() const noexcept {
      return testz<BlockBits>(data);
    }

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

    bool contains(const Block& rhs) const noexcept {
      return ibbv::utils::contains<BlockBits>(data, rhs.data);
    }

    bool intersects(const Block& rhs) const noexcept {
      return ibbv::utils::intersects<BlockBits>(data, rhs.data);
    }

    bool operator==(const Block& rhs) const noexcept {
      return ibbv::utils::cmpeq<BlockBits>(data, rhs.data);
    }

    bool operator!=(const Block& rhs) const noexcept {
      return !(*this == rhs);
    }

    bool operator|=(const Block& rhs) noexcept {
      return ibbv::utils::or_inplace<BlockBits>(data, rhs.data);
    }

    ibbv::utils::ComposedChangeResult operator&=(const Block& rhs) noexcept {
      return ibbv::utils::and_inplace<BlockBits>(data, rhs.data);
    }

    ibbv::utils::ComposedChangeResult operator-=(const Block& rhs) noexcept {
      return ibbv::utils::diff_inplace<BlockBits>(data, rhs.data);
    }
  };

  using index_t = uint32_t;

protected:
  static inline constexpr auto IndexReservedBits = log2int<BlockBits>;
  static inline constexpr index_t IndexValidBitsMask =
      ~((1 << IndexReservedBits) - 1);
  struct IBBVStorage {
  public:
    static inline constexpr auto IndexSize = sizeof(index_t);
    static inline constexpr auto BlockSize = sizeof(Block);

    std::byte* start;
    size_t num_block;

    static inline auto bytes_needed(size_t num_block) noexcept {
      return num_block * (IndexSize + BlockSize);
    }
    /// Init storage. Ignore any data already exists.
    /// All indexes and blocks are zero-inited.
    inline void init_small_storage(size_t num_block) noexcept {
      start = reinterpret_cast<std::byte*>(
          mi_zalloc_small(bytes_needed(num_block)));
      this->num_block = num_block;
    }

    /// Init storage. Ignore any data already exists.
    /// All indexes and blocks are zero-inited.
    inline void init_storage(size_t num_block) noexcept {
      start = reinterpret_cast<std::byte*>(mi_zalloc(bytes_needed(num_block)));
      this->num_block = num_block;
    }

    static inline index_t* idx_at(std::byte* base, size_t pos) noexcept {
      return reinterpret_cast<index_t*>(base) + pos;
    }
    inline index_t* idx_at(size_t pos) noexcept {
      return idx_at(start, pos);
    }
    inline const index_t* idx_at(size_t pos) const noexcept {
      return idx_at(start, pos);
    }
    static inline Block* blk_at(std::byte* base, size_t num_block,
                                size_t pos) noexcept {
      return reinterpret_cast<Block*>(idx_at(base, num_block)) + pos;
    }
    inline Block* blk_at(size_t pos) noexcept {
      return blk_at(start, num_block, pos);
    }
    inline const Block* blk_at(size_t pos) const noexcept {
      return blk_at(start, num_block, pos);
    }

    /// Shrink to specified num of blocks.
    /// If new_num_block >= num_block, do nothing (don't throw).
    inline void truncate(size_t new_num_block) noexcept {
      if (new_num_block >= num_block) return;
      std::memmove(idx_at(new_num_block), blk_at(0), new_num_block * BlockSize);
      start = reinterpret_cast<std::byte*>(
          mi_expand(start, bytes_needed(new_num_block)));
      num_block = new_num_block;
    }
    /// Extend to specified num of blocks. New indexes and blocks are uninited.
    /// If new_num_block <= num_block, do nothing (don't throw).
    inline void extend(size_t new_num_block) noexcept {
      if (new_num_block <= num_block) return;
      if (mi_expand(start, bytes_needed(new_num_block))) {
        std::memmove(blk_at(start, new_num_block, 0), blk_at(0),
                     num_block * BlockSize);
        num_block = new_num_block;
      } else {
        std::byte* new_start = reinterpret_cast<std::byte*>(
            mi_malloc(bytes_needed(new_num_block)));
        std::memcpy(idx_at(new_start, 0), idx_at(0), num_block * IndexSize);
        std::memcpy(blk_at(new_start, new_num_block, 0), blk_at(0),
                    num_block * BlockSize);
        start = new_start;
        num_block = new_num_block;
      }
    }

    inline size_t find_lower_bound(index_t target_index) const noexcept {
      return std::lower_bound(idx_at(0), idx_at(num_block), target_index) -
             idx_at(0);
    }
    /// Insert a new block with one bit set.
    /// Requires: 0 <= pos <= num_block, and the block mustn't exists.
    inline void insert(size_t pos, index_t value) noexcept {
      if (mi_expand(start, bytes_needed(num_block + 1))) {
        std::memmove(blk_at(start, num_block + 1, pos + 1), blk_at(pos),
                     (num_block - pos) * BlockSize);
        ::new (blk_at(start, num_block + 1, pos))(Block)(value % BlockBits);
        std::memmove(idx_at(pos + 1), idx_at(pos),
                     (num_block - pos) * IndexSize + pos * BlockSize);
        *idx_at(pos) = value & IndexValidBitsMask;
        ++num_block;
      } else {
        std::byte* new_start = reinterpret_cast<std::byte*>(
            mi_malloc(bytes_needed(num_block + 1)));
        std::memcpy(new_start, start, pos * IndexSize);
        *idx_at(new_start, pos) = value & IndexValidBitsMask;
        std::memcpy(idx_at(new_start, pos + 1), idx_at(pos),
                    (num_block - pos) * IndexSize + pos * BlockSize);
        ::new (blk_at(new_start, num_block + 1, pos))(Block)(value % BlockBits);
        std::memcpy(blk_at(new_start, num_block + 1, pos + 1), blk_at(pos),
                    (num_block - pos) * BlockSize);
        mi_free(start);
        start = new_start;
        ++num_block;
      }
    }
    /// Remove a block.
    inline void remove(size_t pos) noexcept {
      std::memmove(idx_at(pos), idx_at(pos + 1),
                   (num_block - pos - 1) * IndexSize + pos * BlockSize);
      std::memmove(reinterpret_cast<std::byte*>(idx_at(pos)) +
                       (num_block - pos - 1) * IndexSize + pos * BlockSize,
                   blk_at(pos + 1), (num_block - pos - 1) * BlockSize);
      --num_block;
      start = reinterpret_cast<std::byte*>( // the pointer shouldn't change
          mi_expand(start, bytes_needed(num_block)));
    }
    inline void clear() noexcept {
      mi_free(start);
      start = nullptr;
      num_block = 0;
    }

    /// Convert a sorted array to IBBVStorage. Require: 1 <= value_count <= 8
    /// (UB otherwise)
    __attribute__((no_sanitize("address"))) IBBVStorage(
        const index_t* value_start, const size_t value_count) noexcept {
      const auto v_raw = _mm256_loadu_epi32(value_start);
      /// clear lowest IndexReservedBits bits
      const auto v_lowcleared = _mm256_slli_epi32(
          _mm256_srli_epi32(v_raw, IndexReservedBits), IndexReservedBits);
      alignas(__m256i) index_t idx_low_cleared[8];
      _mm256_store_si256(reinterpret_cast<__m256i*>(idx_low_cleared),
                         v_lowcleared);
      /// rotate-shift right the whole vector by 1 element
      const auto v_sr1 = _mm256_alignr_epi32(v_lowcleared, v_lowcleared, 1);
      /// whether each idx is equal to the next idx
      const auto m_eq = _mm256_mask_cmpeq_epi32_mask(
          // the last valid idx shouldn't be compared (its next idx is unknown)
          (1 << (value_count - 1)) - 1, v_lowcleared, v_sr1);
      /// num of idx that is equal to the next idx
      const auto num_eq = popcnt(m_eq);
      const auto num_unique_idx = value_count - num_eq;

      init_small_storage(num_unique_idx);
      index_t* cur_idx = idx_at(0) - 1;
      Block* cur_blk = blk_at(0) - 1;
      /// whether each idx is equal to the previous idx
      const auto m_eq_sl1 = m_eq << 1;
      for (size_t i = 0; i < value_count; i++) {
        // if cur idx == prev idx, don't move to next idx/blk
        cur_idx += m_eq_sl1 & (1 << i) ? 0 : 1;
        cur_blk += m_eq_sl1 & (1 << i) ? 0 : 1;
        *cur_idx = idx_low_cleared[i];
        cur_blk->set(value_start[i] % BlockBits);
      }
    }
    ~IBBVStorage() noexcept {
      mi_free(start);
    }
    IBBVStorage(const IBBVStorage& rhs) noexcept {
      init_storage(rhs.num_block);
      std::memcpy(idx_at(0), rhs.idx_at(0), bytes_needed(rhs.num_block));
    }
    IBBVStorage& operator=(const IBBVStorage& rhs) noexcept {
      if (this == &rhs) return *this;
      mi_free(start);
      init_storage(rhs.num_block);
      std::memcpy(idx_at(0), rhs.idx_at(0), bytes_needed(rhs.num_block));
      return *this;
    }
    /// Move constructor. rhs is unstable after moving
    IBBVStorage(IBBVStorage&& rhs) noexcept
        : start(rhs.start), num_block(rhs.num_block) {
      rhs.start = nullptr;
    }
    /// Move assignment. rhs is unstable after moving if this != &rhs
    IBBVStorage& operator=(IBBVStorage&& rhs) noexcept {
      if (this == &rhs) return *this;
      mi_free(start);
      start = rhs.start;
      num_block = rhs.num_block;
      rhs.start = nullptr;
      return *this;
    }

    inline bool operator==(const IBBVStorage& rhs) const noexcept {
      if (num_block != rhs.num_block) return false;
      return std::memcmp(start, rhs.start, bytes_needed(num_block)) == 0;
    }
    inline bool operator!=(const IBBVStorage& rhs) const noexcept {
      return !(*this == rhs);
    }
  };

  IBBVStorage storage;

  /// Returns # of blocks.
  _inline size_t size() const noexcept {
    return storage.num_block;
  }
  _inline const index_t& index_at(size_t i) const noexcept {
    return *storage.idx_at(i);
  }
  _inline index_t& index_at(size_t i) noexcept {
    return *storage.idx_at(i);
  }
  _inline Block& block_at(size_t i) noexcept {
    return *storage.blk_at(i);
  }
  _inline const Block& block_at(size_t i) const noexcept {
    return *storage.blk_at(i);
  }
  mutable size_t last_used_pos{0};

  // Static helper methods

#define unroll_loop(var, times, ...)                                           \
  static_assert(times == 4);                                                   \
  {                                                                            \
    constexpr int var = 0;                                                     \
    __VA_ARGS__;                                                               \
  }                                                                            \
  {                                                                            \
    constexpr int var = 1;                                                     \
    __VA_ARGS__;                                                               \
  }                                                                            \
  {                                                                            \
    constexpr int var = 2;                                                     \
    __VA_ARGS__;                                                               \
  }                                                                            \
  {                                                                            \
    constexpr int var = 3;                                                     \
    __VA_ARGS__;                                                               \
  }

  static _inline std::pair<uint8_t, uint8_t> adv_count(
      const IndexedBlockBitVector<>& lhs, const IndexedBlockBitVector<>& rhs,
      const __m512i& lhs_idx, const __m512i& rhs_idx, const size_t lhs_i,
      const size_t rhs_i) {
    /**the maximum index in current range [lhs_i..=lhs_i + 15], spread to \
     * vector register */
    const auto rangemax_lhs = _mm512_set1_epi32(lhs.index_at(lhs_i + 15)),
               rangemax_rhs = _mm512_set1_epi32(rhs.index_at(rhs_i + 15));
    /**whether each u32 index is less than or equal to the maximum index in
     * current range of the other vector */
    const uint16_t lemask_lhs = _mm512_cmple_epu32_mask(lhs_idx, rangemax_rhs),
                   lemask_rhs = _mm512_cmple_epu32_mask(rhs_idx, rangemax_lhs);
    /**the number to increase for lhs_i / rhs_i, <= 16 */
    const auto advance_lhs = 16 - ibbv::utils::lzcnt(lemask_lhs),
               advance_rhs = 16 - ibbv::utils::lzcnt(lemask_rhs);
    return {advance_lhs, advance_rhs};
  }
  static inline size_t szudzik(size_t a, size_t b) {
    return a > b ? b * b + a : a * a + a + b;
  }
  /// A 512-bit vector contains 32*16-bit integers in ascending order,
  /// that is, {0, 1, 2, ..., 31} (from e0 to e31).
  static inline const __m512i asc_indexes = _mm512_set_epi16(
      31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14,
      13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

  // SIMD algorithms.
  bool union_simd(const IndexedBlockBitVector& rhs) noexcept {
#if IBBV_COUNT_OP
    // this_count, this_size, rhs_count, rhs_size, result_count, result_size
    static Counter<int, int, int, int, int, int> ctr{"union_simd"};
    const auto this_count = count(), rhs_count = rhs.count();
#endif
    // Update `this` inplace, save extra blocks to temp
    const auto this_size = size(), rhs_size = rhs.size();
    size_t lhs_i = 0, rhs_i = 0;
    bool changed = false;
    size_t extra_count = 0;
    index_t* extra_indexes = (index_t*)mi_mallocn(sizeof(index_t), rhs_size);
    Block* extra_blocks = (Block*)mi_mallocn(sizeof(Block), rhs_size);

    alignas(64) Block rhs_block_temp[512 / (sizeof(index_t) * 8)];
    while (lhs_i + 16 <= this_size && rhs_i + 16 <= rhs_size) {
      /// indexes[lhs_i..lhs_i + 16]
      const auto v_lhs_idx = _mm512_loadu_epi32(storage.idx_at(lhs_i)),
                 v_rhs_idx = _mm512_loadu_epi32(rhs.storage.idx_at(rhs_i));
      /// whether each u32 matches (exist in both vectors)
      uint16_t match_this, match_rhs;
      ne_mm512_2intersect_epi32(v_lhs_idx, v_rhs_idx, match_this, match_rhs);

      const auto [advance_lhs, advance_rhs] =
          adv_count(*this, rhs, v_lhs_idx, v_rhs_idx, lhs_i, rhs_i);

      auto match_this_temp = match_this;
      Block* store_blk_base = rhs_block_temp;
      for (auto i = 0; i < advance_rhs; ++i) {
        const auto& rhs_block_cur = rhs.block_at(rhs_i + i);
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
      unroll_loop(i, 4, {
        /// matched & ordered 4 blocks (8 u64) from memory.
        /// zero in case of out of bounds
        const auto v_this = _mm512_loadu_epi64(&block_at(lhs_i + i * 4));
        const auto v_rhs = _mm512_maskz_loadu_epi64(dup_match_this >> (i * 8),
                                                    &rhs_block_temp[i * 4]);
        const auto or_result = _mm512_or_epi64(v_this, v_rhs);

        if (!changed) // compute `changed` if not already set
          changed = !avx_vec<512>::eq_cmp(v_this, or_result);

        _mm512_storeu_epi64(&block_at(lhs_i + i * 4), or_result);
      });
      lhs_i += advance_lhs, rhs_i += advance_rhs;
    }

    // deal with remaining elements
    while (lhs_i < this_size && rhs_i < rhs_size) {
      const auto lhs_ind = index_at(lhs_i);
      const auto rhs_ind = rhs.index_at(rhs_i);
      if (lhs_ind < rhs_ind) ++lhs_i;
      else if (lhs_ind > rhs_ind) {
        // copy current rhs block to temp
        extra_indexes[extra_count] = rhs_ind;
        extra_blocks[extra_count] = rhs.block_at(rhs_i);
        ++extra_count;
        ++rhs_i;
      } else { // lhs_ind == rhs_ind
        changed |= (block_at(lhs_i) |= rhs.block_at(rhs_i));
        ++lhs_i, ++rhs_i;
      }
    }
    /// whether any extra blocks are added (requires merge sort)
    const bool any_extra = extra_count > 0;
    /// whether any remaining blocks in rhs (extra & largest)
    const bool rhs_remaining = rhs_i < rhs_size;
    changed |= any_extra || rhs_remaining;
    const auto new_size = this_size + extra_count + (rhs_size - rhs_i);
    storage.extend(new_size);
    if (any_extra) {
      // inplace merge sort (also arrange blocks).
      signed int dest_i = this_size + extra_count - 1, src_i = this_size - 1,
                 extra_i = extra_count - 1;
      while (src_i >= 0 && extra_i >= 0) {
        if (index_at(src_i) > extra_indexes[extra_i]) {
          index_at(dest_i) = index_at(src_i);
          block_at(dest_i) = block_at(src_i);
          --src_i;
        } else {
          index_at(dest_i) = extra_indexes[extra_i];
          block_at(dest_i) = extra_blocks[extra_i];
          --extra_i;
        }
        --dest_i;
      }
      if (extra_i >= 0) {
        // copy remaining extra blocks
        std::memcpy(&index_at(0), &extra_indexes[0],
                    sizeof(index_t) * (extra_i + 1));
        std::memcpy(&block_at(0), &extra_blocks[0],
                    sizeof(Block) * (extra_i + 1));
      }
      // else if src_i >= 0, they are already in place
    }
    mi_free(extra_indexes);
    mi_free(extra_blocks);
    if (rhs_remaining) { // remaining elements in rhs are always largest
      std::memcpy(&index_at(this_size + extra_count), &rhs.index_at(rhs_i),
                  sizeof(index_t) * (rhs_size - rhs_i));
      std::memcpy(&block_at(this_size + extra_count), &rhs.block_at(rhs_i),
                  sizeof(Block) * (rhs_size - rhs_i));
    }
#if IBBV_COUNT_OP
    ctr.inc({this_count, this_size, rhs_count, rhs_size, count(), size()});
#endif
    return changed;
  }

  bool intersect_simd(const IndexedBlockBitVector& rhs) noexcept {
    const auto this_size = size(), rhs_size = rhs.size();
    size_t valid_count = 0, lhs_i = 0, rhs_i = 0;
    bool changed = false;
    const auto all_zero = _mm512_setzero_si512();
    while (lhs_i + 16 <= this_size && rhs_i + 16 <= rhs_size) {
      /// indexes[lhs_i..lhs_i + 16]
      const auto v_lhs_idx = _mm512_loadu_epi32(storage.idx_at(lhs_i)),
                 v_rhs_idx = _mm512_loadu_epi32(rhs.storage.idx_at(rhs_i));
      /// whether each u32 matches (exist in both vectors)
      uint16_t match_this, match_rhs;
      ne_mm512_2intersect_epi32(v_lhs_idx, v_rhs_idx, match_this, match_rhs);

      const auto [advance_lhs, advance_rhs] =
          adv_count(*this, rhs, v_lhs_idx, v_rhs_idx, lhs_i, rhs_i);

      /// compress the intersected data's offset(/8bytes).
      const auto gather_this_offset_u16x32 =
                     _mm512_maskz_compress_epi32(match_this, asc_indexes),
                 gather_rhs_offset_u16x32 =
                     _mm512_maskz_compress_epi32(match_rhs, asc_indexes);
      const auto gather_this_base_addr = block_at(lhs_i).data;
      const auto gather_rhs_base_addr = rhs.block_at(rhs_i).data;

      const auto ordered_indexes =
          _mm512_maskz_compress_epi32(match_this, v_lhs_idx);

      const auto n_matched = _mm_popcnt_u32(match_this);
      const uint32_t n_matched_bits_dup = ((uint64_t)1 << (n_matched * 2)) - 1;

      // compute AND result of matched blocks
      unroll_loop(i, 4, {
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
        _mm512_mask_compressstoreu_epi32(&index_at(valid_count),
                                         nzero_compressed << (i * 4),
                                         ordered_indexes);
        _mm512_mask_compressstoreu_epi64(&block_at(valid_count),
                                         nzero_mask_by_block, and_result);

        const auto nzero_count = ibbv::utils::popcnt(nzero_compressed);
        valid_count += nzero_count;
      });
      lhs_i += advance_lhs, rhs_i += advance_rhs;
    }
    // use trival loop for the rest
    // TODO: improve
    while (lhs_i < this_size && rhs_i < rhs_size) {
      const auto lhs_ind = index_at(lhs_i);
      const auto rhs_ind = rhs.index_at(rhs_i);
      if (lhs_ind < rhs_ind) {
        // ignore this block (not copied to valid position)
        ++lhs_i;
        changed = true;
      } else if (lhs_ind > rhs_ind) ++rhs_i;
      else { // lhs_ind == rhs_ind
        const auto vcur_this = avx_vec<BlockBits>::load(block_at(lhs_i).data),
                   vcur_rhs =
                       avx_vec<BlockBits>::load(rhs.block_at(rhs_i).data);
        const auto and_result = avx_vec<BlockBits>::and_op(vcur_this, vcur_rhs);
        if (avx_vec<BlockBits>::is_zero(and_result))
          // no bits in common, skip this block
          changed = true;
        else {
          if (!changed) // compute `changed` if not already set
            changed = !avx_vec<BlockBits>::eq_cmp(vcur_this, and_result);

          // store the result
          avx_vec<BlockBits>::store(block_at(valid_count).data, and_result);
          index_at(valid_count) = lhs_ind;
          ++valid_count; // increment valid count
        }
        ++lhs_i, ++rhs_i;
      }
    }
    storage.truncate(valid_count);
    changed |= (valid_count != this_size);
    return changed;
  }

  bool diff_simd(const IndexedBlockBitVector& rhs) noexcept {
    const auto this_size = size(), rhs_size = rhs.size();
    size_t valid_count = 0, lhs_i = 0, rhs_i = 0;
    bool changed = false;
    alignas(64) Block rhs_block_temp[512 / (sizeof(index_t) * 8)];
    while (lhs_i + 16 <= this_size && rhs_i + 16 <= rhs_size) {
      /// indexes[lhs_i..lhs_i + 16]
      const auto v_lhs_idx = _mm512_loadu_epi32(storage.idx_at(lhs_i)),
                 v_rhs_idx = _mm512_loadu_epi32(rhs.storage.idx_at(rhs_i));
      /// whether each u32 matches (exist in both vectors)
      uint16_t match_this, match_rhs;
      ne_mm512_2intersect_epi32(v_lhs_idx, v_rhs_idx, match_this, match_rhs);

      const auto [advance_lhs, advance_rhs] =
          adv_count(*this, rhs, v_lhs_idx, v_rhs_idx, lhs_i, rhs_i);

      // align matched data of rhs to the shape of this.
      // we don't care unmatched position
      auto matched_this_temp = match_this, matched_rhs_temp = match_rhs;
      auto rhs_block_temp_addr = rhs_block_temp;
      auto rhs_block_addr = &rhs.block_at(rhs_i);
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
      const uint16_t advance_lhs_to_bits = ((uint32_t)1 << advance_lhs) - 1;
      const auto dup_advance_lhs = duplicate_bits(advance_lhs_to_bits);

      // compute AND result of matched blocks
      unroll_loop(i, 4, {
        /// matched & ordered 4 blocks (8 u64) from memory. zero
        /// in case of out of bounds
        const auto v_this = _mm512_maskz_loadu_epi64(dup_advance_lhs >> (i * 8),
                                                     &block_at(lhs_i) + i * 4);
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
        _mm512_mask_compressstoreu_epi32(
            &index_at(valid_count), nzero_compressed << (i * 4), v_lhs_idx);
        _mm512_mask_compressstoreu_epi64(&block_at(valid_count),
                                         nzero_mask_by_block, andnot_result);

        const auto nzero_count = _mm_popcnt_u32(nzero_compressed);
        valid_count += nzero_count;
      });
      lhs_i += advance_lhs, rhs_i += advance_rhs;
    }
    while (lhs_i < this_size && rhs_i < rhs_size) {
      const auto lhs_ind = index_at(lhs_i);
      const auto rhs_ind = rhs.index_at(rhs_i);
      if (lhs_ind < rhs_ind) { // keep this block
        index_at(valid_count) = lhs_ind;
        block_at(valid_count) = block_at(lhs_i);
        ++lhs_i;
        ++valid_count;
      } else if (lhs_ind > rhs_ind) ++rhs_i;
      else { // compute ANDNOT
        const auto v_this = avx_vec<BlockBits>::load(&block_at(lhs_i));
        const auto v_rhs = avx_vec<BlockBits>::load(&rhs.block_at(rhs_i));
        const auto andnot_result = avx_vec<BlockBits>::andnot_op(v_this, v_rhs);
        if (avx_vec<BlockBits>::is_zero(andnot_result)) // changed to zero
          changed = true;
        else {
          if (!changed)
            changed = !avx_vec<BlockBits>::eq_cmp(v_this, andnot_result);
          index_at(valid_count) = lhs_ind;
          avx_vec<BlockBits>::store(&block_at(valid_count), andnot_result);
          valid_count++;
        }
        ++lhs_i, ++rhs_i;
      }
    }
    // the rest element is kept
    const auto rest_count = this_size - lhs_i;
    std::memmove(&index_at(valid_count), &index_at(lhs_i),
                 sizeof(index_t) * rest_count);
    std::memmove(&block_at(valid_count), &block_at(lhs_i),
                 sizeof(Block) * rest_count);
    valid_count += rest_count;
    storage.truncate(valid_count);
    changed |= (valid_count != this_size);
    return changed;
  }

  bool contains_simd(const IndexedBlockBitVector& rhs) const noexcept {
    const auto this_size = size(), rhs_size = rhs.size();
    if (this_size < rhs_size) return false;

    size_t lhs_i = 0, rhs_i = 0;
    const auto all_zero = _mm512_setzero_si512();
    while (lhs_i + 16 <= this_size && rhs_i + 16 <= rhs_size) {
      /// indexes[lhs_i..lhs_i + 16]
      const auto v_lhs_idx = _mm512_loadu_epi32(storage.idx_at(lhs_i)),
                 v_rhs_idx = _mm512_loadu_epi32(rhs.storage.idx_at(rhs_i));
      /// whether each u32 matches (exist in both vectors)
      uint16_t match_this, match_rhs;
      ne_mm512_2intersect_epi32(v_lhs_idx, v_rhs_idx, match_this, match_rhs);

      const auto [advance_lhs, advance_rhs] =
          adv_count(*this, rhs, v_lhs_idx, v_rhs_idx, lhs_i, rhs_i);

      /// count of matched indexes
      const auto n_matched = ibbv::utils::popcnt((uint32_t)match_this);

      if (advance_rhs > n_matched) return false;

      /// compress the intersected data's offset(/8bytes).
      const auto gather_this_offset_u16x32 =
                     _mm512_maskz_compress_epi32(match_this, asc_indexes),
                 gather_rhs_offset_u16x32 =
                     _mm512_maskz_compress_epi32(match_rhs, asc_indexes);
      const auto gather_this_base_addr = block_at(lhs_i).data;
      const auto gather_rhs_base_addr = rhs.block_at(rhs_i).data;

      const uint32_t n_matched_bits_dup = ((uint64_t)1 << (n_matched * 2)) - 1;

      unroll_loop(i, 4, {
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

        if (!avx_vec<512>::eq_cmp(and_result, intersect_rhs)) return false;
      });
      lhs_i += advance_lhs, rhs_i += advance_rhs;
    }
    while (lhs_i < size() && rhs_i < rhs.size()) {
      const auto lhs_ind = index_at(lhs_i);
      const auto rhs_ind = rhs.index_at(rhs_i);
      if (lhs_ind > rhs_ind) return false;
      else if (lhs_ind < rhs_ind) ++lhs_i;
      else {
        if (!block_at(lhs_i).contains(rhs.block_at(rhs_i))) return false;
        ++lhs_i, ++rhs_i;
      }
    }
    return rhs_i == rhs.size();
  }

  bool intersects_simd(const IndexedBlockBitVector& rhs) const noexcept {
    const auto this_size = size(), rhs_size = rhs.size();
    size_t lhs_i = 0, rhs_i = 0;
    const auto all_zero = _mm512_setzero_si512();
    while (lhs_i + 16 <= this_size && rhs_i + 16 <= rhs_size) {
      /// indexes[lhs_i..lhs_i + 16]
      const auto v_lhs_idx = _mm512_loadu_epi32(storage.idx_at(lhs_i)),
                 v_rhs_idx = _mm512_loadu_epi32(rhs.storage.idx_at(rhs_i));
      /// whether each u32 matches (exist in both vectors)
      uint16_t match_this, match_rhs;
      ne_mm512_2intersect_epi32(v_lhs_idx, v_rhs_idx, match_this, match_rhs);

      const auto [advance_lhs, advance_rhs] =
          adv_count(*this, rhs, v_lhs_idx, v_rhs_idx, lhs_i, rhs_i);

      /// count of matched indexes
      const auto n_matched = ibbv::utils::popcnt((uint32_t)match_this);

      /// compress the intersected data's offset(/8bytes).
      const auto gather_this_offset_u16x32 =
                     _mm512_maskz_compress_epi32(match_this, asc_indexes),
                 gather_rhs_offset_u16x32 =
                     _mm512_maskz_compress_epi32(match_rhs, asc_indexes);
      const auto gather_this_base_addr = block_at(lhs_i).data;
      const auto gather_rhs_base_addr = rhs.block_at(rhs_i).data;

      const uint32_t n_matched_bits_dup = ((uint64_t)1 << (n_matched * 2)) - 1;

      unroll_loop(i, 4, {
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

        // not zero => share any bits in common => return true
        if (!avx_vec<512>::is_zero(and_result)) return true;
      });
      lhs_i += advance_lhs, rhs_i += advance_rhs;
    }
    while (lhs_i < size() && rhs_i < rhs.size()) {
      const auto lhs_ind = index_at(lhs_i);
      const auto rhs_ind = rhs.index_at(rhs_i);
      if (lhs_ind > rhs_ind) ++rhs_i;
      else if (lhs_ind < rhs_ind) ++lhs_i;
      else {
        if (block_at(lhs_i).intersects(rhs.block_at(rhs_i))) return true;
        ++lhs_i, ++rhs_i;
      }
    }
    return false;
  }

  // Public interfaces
public:
  class IndexedBlockBitVectorIterator {
  public:
    using iterator_category = std::input_iterator_tag;
    using value_type = index_t;
    using difference_type = std::ptrdiff_t;
    using pointer = const index_t*;
    using reference = const index_t&;
    index_t cur_pos;

  protected:
    const index_t* indexIt;
    const index_t* indexEnd;
    const Block* blockIt;
    unsigned char unit_index; // unit index in the current block
    unsigned char bit_index;  // bit index in the current unit
    bool end;
    /// move to next unit, returns false if reached the end
    void incr_unit() noexcept {
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
    void incr_bit() noexcept {
      if (++bit_index == UnitBits)
        // forward to next unit
        incr_unit();
    }
    /// Start from current position and search for the next set bit
    void search() noexcept {
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
    void forward() noexcept {
      incr_bit();
      search();
    }

  public:
    IndexedBlockBitVectorIterator() = delete;
    IndexedBlockBitVectorIterator(const IndexedBlockBitVector& vec,
                                  bool end = false) noexcept
        : // must be init to identify raw vector
          indexEnd(vec.storage.idx_at(vec.storage.num_block)),
          end(end || vec.empty()) {
      if (end) return;
      indexIt = vec.storage.idx_at(0);
      blockIt = vec.storage.blk_at(0);
      unit_index = 0;
      bit_index = 0;
      search();
    }
    DEFAULT_COPY_MOVE(IndexedBlockBitVectorIterator);

    reference operator*() const noexcept {
      return cur_pos;
    }
    pointer operator->() const noexcept {
      return &cur_pos;
    }
    IndexedBlockBitVectorIterator& operator++() noexcept {
      forward();
      return *this;
    }
    IndexedBlockBitVectorIterator operator++(int) noexcept {
      IndexedBlockBitVectorIterator temp = *this;
      ++*this;
      return temp;
    }
    bool operator==(const IndexedBlockBitVectorIterator& other) const noexcept {
      return
          // created from the same vector, and
          indexEnd == other.indexEnd &&
          (( // both ended, or
               end && other.end) ||
           ( // both not ended and pointing to the same position
               end == other.end && cur_pos == other.cur_pos));
    }
    bool operator!=(const IndexedBlockBitVectorIterator& other) const noexcept {
      return !(*this == other);
    }
  };
  using iterator = IndexedBlockBitVectorIterator;

  /// Returns an iterator to the beginning of this vector.
  /// NOTE: If you modify the vector after creating an iterator, the iterator
  /// is not stable and may cause UB if used.
  iterator begin() const noexcept {
    return iterator(*this);
  }
  iterator end() const noexcept {
    return iterator(*this, true);
  }

  /// Return the first set bit in the bitmap.  Return -1 if no bits are set.
  int32_t find_first() const noexcept {
    if (empty()) return -1;
    return *begin();
  }

  /// Returns true if no bits are set.
  bool empty() const noexcept {
    return storage.num_block == 0;
  }

  /// Returns the count of bits set.
  uint32_t count() const noexcept {
    if (size() == 0) return 0;

    if constexpr (__AVX512VPOPCNTDQ__ && __AVX512VL__) {
      const Block* it = storage.blk_at(0);
      const auto v0 = avx_vec<BlockBits>::load(&(it->data));
      auto c = avx_vec<BlockBits>::popcnt(v0);
      ++it;
      for (; it != storage.blk_at(size()); ++it) {
        const auto curv = avx_vec<BlockBits>::load(&(it->data));
        const auto curc = avx_vec<BlockBits>::popcnt(curv);
        c = avx_vec<BlockBits>::add_op(c, curc);
      }
      return avx_vec<BlockBits>::reduce_add(c);
    } else {
      uint32_t result = 0;
      auto arr = reinterpret_cast<const uint64_t*>(storage.blk_at(0));
      for (size_t i = 0; i < size() * sizeof(Block) / 8; ++i, ++arr)
        result += ibbv::utils::popcnt(*arr);
      return result;
    }
  }

  /// Empty the set and release memory holded.
  void clear() noexcept {
    storage.clear();
  }

  /// Returns true if bit `n` is set.
  bool test(index_t n) const noexcept {
    const auto target_ind = n & IndexValidBitsMask;
    const auto pos = storage.find_lower_bound(target_ind);
    if (pos == storage.num_block ||
        *storage.idx_at(pos) != target_ind) // not found
      return false;
    else return storage.blk_at(pos)->test(n % BlockBits);
  }

  /// Set bit `n` (zero-based).
  void set(index_t n) noexcept {
    const auto target_ind = n & IndexValidBitsMask;
    const auto pos = storage.find_lower_bound(target_ind);
    if (pos == storage.num_block ||
        *storage.idx_at(pos) != target_ind) // not found
      storage.insert(pos, n);
    else storage.blk_at(pos)->set(n % BlockBits);
  }

  /// Check if bit `n` is set. If it is, returns false.
  /// Otherwise, set it and return true.
  bool test_and_set(index_t n) noexcept {
    const auto target_ind = n & IndexValidBitsMask;
    const auto pos = storage.find_lower_bound(target_ind);
    if (pos == storage.num_block ||
        *storage.idx_at(pos) != target_ind) { // not found
      storage.insert(pos, n);
      return true;
    } else return storage.blk_at(pos)->test_and_set(n % BlockBits);
  }

  /// Unset bit `n`.
  void reset(index_t n) noexcept {
    const auto target_ind = n & IndexValidBitsMask;
    const auto pos = storage.find_lower_bound(target_ind);
    if (pos == storage.num_block ||
        *storage.idx_at(pos) != target_ind) // not found
      return;
    Block* d = storage.blk_at(pos);
    d->reset(n % BlockBits);
    if (d->empty()) storage.remove(pos);
  }

  /// Returns true if `this` contains all bits of rhs.
  bool contains(const IndexedBlockBitVector& rhs) const noexcept {
    return contains_simd(rhs);
  }

  /// Returns true if `this` contains some bits of rhs.
  bool intersects(const IndexedBlockBitVector& rhs) const noexcept {
    return intersects_simd(rhs);
  }

  bool operator==(const IndexedBlockBitVector& rhs) const noexcept {
    return storage == rhs.storage;
  }

  bool operator!=(const IndexedBlockBitVector& rhs) const noexcept {
    return !(*this == rhs);
  }

  /// Inplace union with rhs.
  /// Returns true if `this` changed.
  bool operator|=(const IndexedBlockBitVector& rhs) noexcept {
    return union_simd(rhs);
  }

  IndexedBlockBitVector operator|(const IndexedBlockBitVector& rhs) const {
    IndexedBlockBitVector copy(*this);
    copy |= rhs;
    return copy;
  }

  /// Inplace intersection with rhs.
  /// Returns true if `this` changed.
  bool operator&=(const IndexedBlockBitVector& rhs) noexcept {
    return intersect_simd(rhs);
  }

  /// Inplace difference with rhs.
  /// Returns true if `this` changed.
  bool operator-=(const IndexedBlockBitVector& rhs) noexcept {
    return diff_simd(rhs);
  }

  friend struct std::hash<ibbv::IndexedBlockBitVector<>>;
};
} // namespace ibbv

namespace std {
template <> struct hash<ibbv::IndexedBlockBitVector<>> {
  std::size_t operator()(
      const ibbv::IndexedBlockBitVector<>& s) const noexcept {
    return ibbv::IndexedBlockBitVector<>::szudzik(
        s.count(),
        ibbv::IndexedBlockBitVector<>::szudzik(s.size(), s.find_first()));
  }
};
} // namespace std