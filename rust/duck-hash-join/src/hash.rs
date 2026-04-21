//! Port of DuckDB's MurmurHash64 / CombineHash from
//! `src/include/duckdb/common/types/hash.hpp:38-45`.
//!
//! The multiplicative constant and shift pattern come from
//! <https://nullprogram.com/blog/2018/07/31/>. Bit-for-bit identical to DuckDB
//! so that salt bits (top 16) and radix bits extracted from the result match
//! the C++ side.

/// DuckDB `MurmurHash64` (hash.hpp:38).
#[inline(always)]
pub const fn murmur_hash64(mut x: u64) -> u64 {
    x ^= x >> 32;
    x = x.wrapping_mul(0xd6e8_feb8_6659_fd93);
    x ^= x >> 32;
    x = x.wrapping_mul(0xd6e8_feb8_6659_fd93);
    x ^= x >> 32;
    x
}

/// DuckDB `MurmurHash32` (hash.hpp:47) — zero-extend then 64-bit mix.
#[inline(always)]
pub const fn murmur_hash32(x: u32) -> u64 {
    murmur_hash64(x as u64)
}

/// DuckDB `CombineHash` (hash.hpp:23) — XOR. Used to fold multi-key hashes.
///
/// Note: XOR is commutative, so hash order doesn't affect the result. DuckDB
/// relies on this for arbitrary-order equality joins.
#[inline(always)]
pub const fn combine_hash(left: u64, right: u64) -> u64 {
    left ^ right
}

/// Hash a slice of `i64` keys into `out`. Mirrors DuckDB's vectorized
/// `VectorOperations::Hash` for int64 flat vectors with no nulls.
#[inline]
pub fn hash_i64_slice(keys: &[i64], out: &mut [u64]) {
    assert_eq!(keys.len(), out.len());
    for (k, o) in keys.iter().zip(out.iter_mut()) {
        *o = murmur_hash64(*k as u64);
    }
}

/// Hash a slice of `i32` keys into `out`. DuckDB's behavior for int32 is to
/// `MurmurHash32` (zero-extend through the 64-bit mix).
#[inline]
pub fn hash_i32_slice(keys: &[i32], out: &mut [u64]) {
    assert_eq!(keys.len(), out.len());
    for (k, o) in keys.iter().zip(out.iter_mut()) {
        *o = murmur_hash32(*k as u32);
    }
}

/// Fold a second column's hash into `out` via XOR-combine. Used when a join
/// has multiple key columns — hash each column independently, combine in place.
#[inline]
pub fn combine_into(out: &mut [u64], other: &[u64]) {
    assert_eq!(out.len(), other.len());
    for (o, r) in out.iter_mut().zip(other.iter()) {
        *o = combine_hash(*o, *r);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Fixed-point checks — these pairs came out of `./build/tools/hash_dump`
    // (source: `tools/utils/hash_dump/hash_dump.cpp`), which calls DuckDB's
    // actual `MurmurHash64` from `src/include/duckdb/common/types/hash.hpp`.
    // If this test ever fails, either the Rust port drifted or DuckDB
    // changed its hash function — either way, rerun hash_dump and update.
    #[test]
    fn murmur_known_vectors() {
        let cases: &[(u64, u64)] = &[
            (0x0000000000000000, 0x0000000000000000),
            (0x0000000000000001, 0x4179b061e0c0e0d0),
            (0x0000000000000002, 0x1c9963305febc252),
            (0x0000000000000007, 0x793f9bd73383384a),
            (0x0000000000000042, 0x559d8f6fa7b0996b),
            (0x00000000deadbeef, 0x288344a3928b2668),
            (0x00000000ffffffff, 0x41c6aec41b1456d1),
            (0x0000000100000000, 0xa04a6aae1526eda6),
            (0x8000000000000000, 0x72446bbcf6c799d7),
            (0xffffffffffffffff, 0x448e29ced4103459),
            (0xabcd123456789abc, 0xbf52d583ca69aa1f),
            (0x0123456789abcdef, 0x3efdea49c590f4ec),
        ];
        for &(input, expected) in cases {
            assert_eq!(
                murmur_hash64(input),
                expected,
                "input=0x{:016x}",
                input
            );
        }
    }

    #[test]
    fn murmur_nonzero_for_nonzero_input() {
        for i in 1u64..1000 {
            assert_ne!(murmur_hash64(i), 0, "hash should not be zero for i={}", i);
        }
    }

    #[test]
    fn combine_hash_is_xor() {
        assert_eq!(combine_hash(0xaaaa, 0x5555), 0xffff);
        assert_eq!(combine_hash(0x1234, 0x0), 0x1234);
        // symmetry
        assert_eq!(combine_hash(0x42, 0xdeadbeef), combine_hash(0xdeadbeef, 0x42));
    }

    #[test]
    fn vectorized_matches_scalar() {
        let keys: Vec<i64> = (0..32).collect();
        let mut out = vec![0u64; keys.len()];
        hash_i64_slice(&keys, &mut out);
        for (i, &k) in keys.iter().enumerate() {
            assert_eq!(out[i], murmur_hash64(k as u64));
        }
    }
}
