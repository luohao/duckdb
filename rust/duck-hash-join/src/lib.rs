//! Port of DuckDB's parallel hash join to Rust.
//!
//! Reference impls this port tracks (paths relative to DuckDB repo root):
//! - `src/include/duckdb/common/types/hash.hpp`              — MurmurHash64
//! - `src/include/duckdb/common/radix_partitioning.hpp`     — radix bit extraction
//! - `src/include/duckdb/execution/ht_entry.hpp`            — salt + ptr entry layout
//! - `src/execution/join_hashtable.cpp`                     — HT core
//! - `src/execution/operator/join/physical_hash_join.cpp`   — parallel build/probe protocol
//!
//! Correctness is validated in `tests/` by comparing outputs to values produced
//! by DuckDB's actual implementation (see `tools/utils/hash_dump`).

pub mod hash;
pub mod ht;
pub mod ht_entry;
pub mod io;
pub mod radix;
pub mod row;
