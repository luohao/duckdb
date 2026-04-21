//! Port of `src/include/duckdb/execution/ht_entry.hpp`.
//!
//! A single hash-table bucket packs 16 bits of salt (hash fingerprint) and
//! 48 bits of pointer into one `u64`. Linear probing reads this word first
//! and rejects non-matching salts before dereferencing the pointer — that
//! early-out keeps the hot probe loop off the row data until salts match.
//!
//! Layout (unchanged from DuckDB — same masks, same semantics):
//! ```text
//!   bit 63 ──────── bit 48 bit 47 ─────────────── bit 0
//!   ┌──────────────┬─────────────────────────────┐
//!   │   16b salt   │        48b pointer           │
//!   └──────────────┴─────────────────────────────┘
//! ```
//!
//! `value == 0` marks an empty entry. Non-empty entries MUST have a non-null
//! pointer (48-bit zero means empty), which is always true for heap-allocated
//! row arenas on x86-64 / ARM64. Zero-page mappings don't happen in row
//! arenas.

/// Salt occupies the top 16 bits of a 64-bit word.
pub const SALT_MASK: u64 = 0xFFFF_0000_0000_0000;

/// Pointer occupies the bottom 48 bits.
pub const POINTER_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

/// A packed `(salt, ptr)` hash-table entry. Binary layout matches DuckDB.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct HtEntry(pub u64);

impl HtEntry {
    #[inline(always)]
    pub const fn empty() -> Self {
        HtEntry(0)
    }

    /// Build an entry from a 48-bit row pointer and a 64-bit hash (only the
    /// top 16 bits are used as salt).
    #[inline(always)]
    pub fn new(hash: u64, ptr: *const u8) -> Self {
        let p = ptr as usize as u64;
        debug_assert!(p & SALT_MASK == 0, "pointer uses top 16 bits");
        HtEntry(p | (hash & SALT_MASK))
    }

    #[inline(always)]
    pub const fn is_occupied(self) -> bool {
        self.0 != 0
    }

    /// Recover the row pointer. Safe to call on an empty entry — returns null.
    #[inline(always)]
    pub const fn pointer(self) -> *const u8 {
        (self.0 & POINTER_MASK) as usize as *const u8
    }

    /// Extract the salt bits in a form directly comparable to
    /// `extract_salt(probe_hash)` — bottom 48 bits set to 1.
    #[inline(always)]
    pub const fn salt(self) -> u64 {
        self.0 | POINTER_MASK
    }

    /// Turn a full 64-bit hash into the salt form used for comparison against
    /// `HtEntry::salt()`. DuckDB's trick: OR in all 1s below the salt bits so
    /// a single equality compares against the stored entry word without
    /// masking.
    #[inline(always)]
    pub const fn extract_salt(hash: u64) -> u64 {
        hash | POINTER_MASK
    }
}

/// Modulo-like wrap via mask. Only correct when `capacity` is a power of two
/// and `mask = capacity - 1`. Same idiom as DuckDB's `IncrementAndWrap`.
#[inline(always)]
pub fn increment_and_wrap(offset: &mut usize, capacity_mask: usize) {
    *offset = (*offset + 1) & capacity_mask;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_zero() {
        let e = HtEntry::empty();
        assert_eq!(e.0, 0);
        assert!(!e.is_occupied());
        assert!(e.pointer().is_null());
    }

    #[test]
    fn packs_and_unpacks_pointer() {
        let buf = [0u8; 16];
        let p = buf.as_ptr();
        let hash: u64 = 0xDEAD_BEEF_1234_5678;
        let e = HtEntry::new(hash, p);
        assert!(e.is_occupied());
        assert_eq!(e.pointer(), p);
    }

    #[test]
    fn salt_comparable_form() {
        let buf = [0u8; 16];
        let p = buf.as_ptr();
        let hash: u64 = 0xDEAD_BEEF_1234_5678;
        let e = HtEntry::new(hash, p);
        // The salt of the stored entry must equal extract_salt(original hash).
        assert_eq!(e.salt(), HtEntry::extract_salt(hash));
        // And must NOT equal extract_salt of a different-in-salt hash.
        let other_hash: u64 = 0x1234_BEEF_1234_5678;
        assert_ne!(e.salt(), HtEntry::extract_salt(other_hash));
    }

    #[test]
    fn salt_ignores_low_bits() {
        // Two hashes with identical top 16 bits but different everywhere else
        // must produce the same extract_salt().
        let h1: u64 = 0xABCD_0000_0000_0000;
        let h2: u64 = 0xABCD_FFFF_FFFF_FFFF;
        assert_eq!(HtEntry::extract_salt(h1), HtEntry::extract_salt(h2));
    }

    #[test]
    fn increment_and_wrap_is_power_of_two_mod() {
        let mask = 7; // capacity 8
        let mut off = 6;
        increment_and_wrap(&mut off, mask);
        assert_eq!(off, 7);
        increment_and_wrap(&mut off, mask);
        assert_eq!(off, 0);
        increment_and_wrap(&mut off, mask);
        assert_eq!(off, 1);
    }
}
