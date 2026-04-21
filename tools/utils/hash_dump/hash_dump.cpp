// hash_dump: dump MurmurHash64 outputs for a fixed set of inputs so the Rust
// port in rust/duck-hash-join/ can cross-validate bit-for-bit against the
// same function DuckDB actually uses at runtime.
//
// Output format: one line per case, space-separated:
//   <input_hex> <output_hex>
// Meant to be copy-pasted into rust/duck-hash-join/src/hash.rs test vectors.

#include "duckdb/common/types/hash.hpp"
#include <cstdint>
#include <cstdio>

using namespace duckdb; // NOLINT

int main() {
	// Inputs that together exercise: zero, small ints, negative-as-unsigned,
	// powers of two, a hash-like value, and a couple of pseudorandom u64s.
	const uint64_t inputs[] = {
	    0ULL,
	    1ULL,
	    2ULL,
	    0x7ULL,
	    0x42ULL,
	    0xDEADBEEFULL,
	    0xFFFFFFFFULL, // u32 max, zero-extended
	    0x1'0000'0000ULL,
	    0x8000'0000'0000'0000ULL, // high bit set
	    0xFFFF'FFFF'FFFF'FFFFULL, // u64 max
	    0xABCD'1234'5678'9ABCULL, // arbitrary pattern
	    0x0123'4567'89AB'CDEFULL,
	};

	printf("// paste into rust/duck-hash-join/src/hash.rs:murmur_known_vectors\n");
	for (auto x : inputs) {
		auto h = MurmurHash64(x);
		printf("(0x%016llx, 0x%016llx),\n", (unsigned long long)x, (unsigned long long)h);
	}
	return 0;
}
