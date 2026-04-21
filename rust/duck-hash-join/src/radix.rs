//! Port of DuckDB's radix partitioning bit-extraction logic from
//! `src/include/duckdb/common/radix_partitioning.hpp`.
//!
//! Layout of a 64-bit hash in DuckDB:
//! ```text
//!   bit 63 ─────────────────── bit 0
//!   ┌──────────┬──────────────┬──────────────────────────┐
//!   │ 16b salt │ 16b headroom │      32 bits rest         │
//!   │  [63:48] │   [47:32]    │       [31:0]              │
//!   └──────────┴──────────────┴──────────────────────────┘
//!     ^^^^^^^ used as salt in HT entries (ht_entry.hpp)
//!              ^^^^^^ `MAX_RADIX_BITS = 12` live here, starting at bit 36
//! ```
//!
//! `Shift(radix_bits) = 48 - radix_bits` places the radix field flush against
//! the top of the 48-bit pointer region. Mirrors the C++ `Shift` /
//! `Mask` / `NumberOfPartitions` inline functions.

/// 4096 partitions — enough for external joins to go out-of-core.
pub const MAX_RADIX_BITS: u32 = 12;

/// `NumberOfPartitions(radix_bits)` — `1 << radix_bits`.
#[inline(always)]
pub const fn num_partitions(radix_bits: u32) -> usize {
    debug_assert!(radix_bits <= MAX_RADIX_BITS);
    1usize << radix_bits
}

/// `Shift(radix_bits)` — bit offset of the radix field within a 64-bit hash.
///
/// Salt is the top 16 bits, so radix bits are packed below salt:
/// `(sizeof(u64) - sizeof(u16)) * 8 - radix_bits = 48 - radix_bits`.
#[inline(always)]
pub const fn shift(radix_bits: u32) -> u32 {
    48 - radix_bits
}

/// `Mask(radix_bits)` — mask that isolates the radix bits when anded with a hash.
#[inline(always)]
pub const fn mask(radix_bits: u32) -> u64 {
    ((1u64 << radix_bits) - 1) << shift(radix_bits)
}

/// Extract the radix partition index for a given hash.
#[inline(always)]
pub const fn partition_of(hash: u64, radix_bits: u32) -> usize {
    ((hash >> shift(radix_bits)) & ((1u64 << radix_bits) - 1)) as usize
}

/// DuckDB's `HashJoinGlobalSinkState::initial_radix_bits` heuristic
/// (physical_hash_join.cpp, near line 335): use 4 bits under 100 threads, else 5.
///
/// This is the starting partition count for the parallel build. External-join
/// repartitioning may increase it after the fact.
#[inline]
pub fn initial_radix_bits(thread_count: usize) -> u32 {
    if thread_count < 100 {
        4
    } else {
        5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_partitions_basic() {
        assert_eq!(num_partitions(0), 1);
        assert_eq!(num_partitions(4), 16);
        assert_eq!(num_partitions(12), 4096);
    }

    #[test]
    fn shift_layout() {
        // radix bits sit just below the 16 salt bits.
        assert_eq!(shift(0), 48);
        assert_eq!(shift(4), 44);
        assert_eq!(shift(12), 36);
    }

    #[test]
    fn mask_isolates_radix_field() {
        // 4 radix bits → bits 44..48 set in the mask.
        assert_eq!(mask(4), 0x0000_F000_0000_0000);
        // 12 radix bits → bits 36..48 set.
        assert_eq!(mask(12), 0x0000_FFF0_0000_0000);
        // 0 radix bits → no bits set.
        assert_eq!(mask(0), 0);
    }

    #[test]
    fn partition_of_extracts_correct_field() {
        // 0xABCD_1234_5678_9ABC bit layout:
        //   top 16 (salt)  = 0xABCD
        //   next 12 (bits 36..47, radix for MAX) = 0x123
        //   next 4  (bits 44..47, radix for 4)  = 0x1
        let h = 0xABCD_1234_5678_9ABCu64;
        assert_eq!(partition_of(h, 12), 0x123);
        assert_eq!(partition_of(h, 4), 0x1);
        assert_eq!(partition_of(h, 0), 0);
    }

    #[test]
    fn radix_and_salt_do_not_overlap() {
        // With 12 radix bits, the mask should not touch the top 16 (salt) bits.
        let salt_mask: u64 = 0xFFFF_0000_0000_0000;
        assert_eq!(mask(12) & salt_mask, 0);
    }

    #[test]
    fn initial_radix_bits_heuristic() {
        assert_eq!(initial_radix_bits(1), 4);
        assert_eq!(initial_radix_bits(4), 4);
        assert_eq!(initial_radix_bits(99), 4);
        assert_eq!(initial_radix_bits(100), 5);
        assert_eq!(initial_radix_bits(1024), 5);
    }
}
