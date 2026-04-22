//! `JoinHashTable` — port of `src/execution/join_hashtable.cpp`.
//!
//! MVP scope:
//! * Single-threaded build and probe. Parallelism comes in a later pass.
//! * INNER JOIN only (FOJ handling requires match-tracking bitmaps, a post-MVP
//!   concern).
//! * i32 key only (the `k` column in the bench parquet). Multi-key / non-i32
//!   keys are a RowMatcher concern — plug in later.
//! * No residual predicates, no bloom filters, no perfect-hash fast path.
//!
//! Layout / algorithm choices follow DuckDB verbatim where they matter for
//! perf:
//! * Power-of-two pointer table sized at `next_power_of_two(n_rows * 2)`
//!   (load factor 0.5 → 25–50% occupancy; matches
//!   `join_hashtable.cpp:438-440`).
//! * Linear probing with salt fast-reject: a probe that lands in a slot whose
//!   stored salt doesn't match the probe hash's top 16 bits moves on without
//!   dereferencing the row pointer.
//! * Chains are built during insert. New rows are prepended: the entry's
//!   pointer becomes the new row's `next_ptr`, and the bucket points at the
//!   new row. Probing walks the chain via `next_ptr`.

use crate::ht_entry::{HtEntry, POINTER_MASK};
use crate::row::{
    row_next_ptr, set_row_hash, set_row_next_ptr, RowArena, RowLayout, HEADER_SIZE,
};

/// Minimum directory size — matches DuckDB's 16384 floor
/// (`join_hashtable.cpp:442`).
pub const MIN_CAPACITY: usize = 16384;

/// Returns the next power-of-two ≥ `n`. Uses DuckDB's `max(16k, pot(n_rows*2))`
/// convention for the pointer-table size.
#[inline]
fn directory_capacity(n_rows: usize) -> usize {
    let target = n_rows.saturating_mul(2).max(MIN_CAPACITY);
    target.next_power_of_two()
}

/// Public alias of `directory_capacity` for use by `pipeline.rs`.
pub fn directory_capacity_for(n_rows: usize) -> usize {
    directory_capacity(n_rows)
}

/// Hash-table directory + row arena. Build in three steps:
/// 1. `reserve(n)` to pre-size the arena.
/// 2. Append rows + their key hashes via `insert_rows`.
/// 3. `finalize()` to allocate the pointer table and link chains.
///
/// After `finalize`, call `probe_i32_inner` to enumerate matches.
pub struct JoinHashTable {
    arena: RowArena,
    /// Cached hashes, one per row, in insertion order. Consumed by `finalize`.
    hashes: Vec<u64>,
    /// Power-of-two directory. Length = `capacity_mask + 1`. Empty until
    /// `finalize()`.
    directory: Vec<HtEntry>,
    capacity_mask: usize,
    /// Byte offset of the join key column within each row.
    key_offset: usize,
    finalized: bool,
}

impl JoinHashTable {
    /// Create an empty HT. `layout`'s first column must be the i32 join key.
    pub fn new(layout: RowLayout) -> Self {
        assert!(
            matches!(
                layout.column_type(0),
                crate::row::PhysType::I32
            ),
            "MVP only supports i32 join key at column 0"
        );
        let key_offset = layout.column_offset(0);
        JoinHashTable {
            arena: RowArena::new(layout),
            hashes: Vec::new(),
            directory: Vec::new(),
            capacity_mask: 0,
            key_offset,
            finalized: false,
        }
    }

    #[inline]
    pub fn layout(&self) -> &RowLayout {
        self.arena.layout()
    }

    /// Pre-size the arena and hash buffer for exactly `n` rows and expose
    /// parallel-write access. Parallel ingesters (one per row group) can
    /// compute row addresses from `arena_base_addr()` + `i * row_size` and
    /// write hash values into `hashes_base_addr()[i]`. Caller must set every
    /// row before `finalize`.
    pub fn set_len(&mut self, n: usize) {
        assert!(!self.finalized, "cannot set_len after finalize");
        self.arena.set_len(n);
        self.hashes.resize(n, 0);
    }

    #[inline]
    pub fn arena_base_addr(&self) -> usize {
        self.arena.base_addr()
    }

    #[inline]
    pub fn hashes_base_addr(&mut self) -> usize {
        self.hashes.as_mut_ptr() as usize
    }

    #[inline]
    pub fn hashes_base_addr_const(&self) -> usize {
        self.hashes.as_ptr() as usize
    }

    #[inline]
    pub fn key_offset(&self) -> usize {
        self.key_offset
    }

    #[inline]
    pub fn is_finalized(&self) -> bool {
        self.finalized
    }

    #[inline]
    pub fn directory_ptr(&self) -> *const HtEntry {
        self.directory.as_ptr()
    }

    #[inline]
    pub fn capacity_mask(&self) -> usize {
        self.capacity_mask
    }

    /// Pre-size both the arena and hash buffer for an expected row count.
    /// Skip only if you're willing to pay Vec realloc cost mid-insert (which
    /// also invalidates pointers handed out earlier — so just don't skip).
    pub fn reserve(&mut self, expected_rows: usize) {
        self.arena.reserve(expected_rows);
        self.hashes.reserve(expected_rows);
    }

    /// Append `n` rows of zeroed bytes and return the slice. Caller fills in
    /// column values and calls `record_hashes` with the matching hashes.
    ///
    /// Kept separate from hash ingestion so parquet batches can write columns
    /// in bulk (one column at a time) before hashing the key column.
    pub fn append_rows(&mut self, n: usize) -> &mut [u8] {
        assert!(!self.finalized, "cannot append after finalize");
        self.arena.append_blank(n)
    }

    /// Record per-row hashes. Must be called once per `append_rows` call,
    /// with `n` hashes in the same order. The arena writes the hash into
    /// each row's header slot so probe chains can be verified later.
    pub fn record_hashes(&mut self, hashes: &[u64]) {
        let start_row = self.arena.len() - hashes.len();
        for (i, &h) in hashes.iter().enumerate() {
            unsafe {
                set_row_hash(self.arena.row_ptr_mut(start_row + i), h);
            }
        }
        self.hashes.extend_from_slice(hashes);
    }

    /// Convenience for fixed-layout ingestion: write one row's worth of i32
    /// key + (layout-defined) payload bytes. Testing helper; the hot path
    /// goes through `append_rows` + bulk column writes.
    #[cfg(test)]
    pub fn insert_test_row(&mut self, key: i32, payload: &[u8]) {
        let layout = self.layout().clone();
        let slice = self.append_rows(1);
        let key_off = layout.column_offset(0);
        slice[key_off..key_off + 4].copy_from_slice(&key.to_ne_bytes());
        let payload_off = key_off + 4;
        slice[payload_off..payload_off + payload.len()].copy_from_slice(payload);
        self.record_hashes(&[crate::hash::murmur_hash32(key as u32)]);
    }

    /// Build the pointer table and link collision chains. Must be called
    /// before any probe. O(n_rows) plus O(capacity) directory initialization.
    ///
    /// Single-threaded. For parallel finalize see `finalize_parallel`.
    pub fn finalize(&mut self) {
        self.finalize_impl(false);
    }

    /// Parallel finalize via partition-aware CAS inserts.
    ///
    /// Rows are pre-grouped by hash partition (top `radix_bits` of the bucket
    /// index), then each rayon task inserts its partition's rows with
    /// `compare_exchange_weak` on the full directory. Because each task's
    /// rows almost always land in its partition's slice of the directory,
    /// cache-line contention is rare. The CAS lets occasional cross-slice
    /// wrap-arounds stay correct without a separate fallback path.
    pub fn finalize_parallel(&mut self) {
        self.finalize_impl(true);
    }

    fn finalize_impl(&mut self, parallel: bool) {
        assert!(!self.finalized, "finalize called twice");
        let n = self.arena.len();
        assert_eq!(self.hashes.len(), n, "hashes/rows count mismatch");

        let capacity = directory_capacity(n);
        self.directory = vec![HtEntry::empty(); capacity];
        self.capacity_mask = capacity - 1;

        if !parallel || n < 65_536 {
            for i in 0..n {
                let h = self.hashes[i];
                let row_ptr = self.arena.row_ptr_mut(i);
                insert_row_serial(&mut self.directory, self.capacity_mask, h, row_ptr);
            }
        } else {
            // Partition-aware CAS insert. Each rayon task handles exactly
            // one partition's rows. Contention is low because each task
            // primarily writes to its partition's slice of the directory —
            // the full directory is atomic, so rare cross-slice wraparounds
            // stay correct without separate fallback.
            use rayon::prelude::*;
            use std::sync::atomic::{AtomicU64, Ordering};
            let log_capacity = capacity.trailing_zeros();
            let num_threads = rayon::current_num_threads();
            // Match DuckDB's partition-count heuristic: 4× threads, capped.
            let min_parts = (num_threads * 4).next_power_of_two();
            let radix_bits = (min_parts.trailing_zeros()).min(log_capacity.saturating_sub(10));
            let num_partitions = 1usize << radix_bits;
            let partition_shift = log_capacity - radix_bits;

            // Pass 1 (parallel): each thread scans a chunk of rows, buckets
            // into its own per-partition row-index vecs. Avoids the shared
            // mutation of a single Vec<Vec<u32>>.
            let per_thread_chunks: Vec<Vec<Vec<u32>>> = {
                let chunk_size = (n + num_threads - 1) / num_threads;
                let hashes = self.hashes.as_slice();
                (0..num_threads)
                    .into_par_iter()
                    .map(|t| {
                        let start = t * chunk_size;
                        let end = (start + chunk_size).min(n);
                        let mut local: Vec<Vec<u32>> = (0..num_partitions)
                            .map(|_| Vec::with_capacity((end - start) / num_partitions + 8))
                            .collect();
                        for i in start..end {
                            let h = hashes[i];
                            let p = ((h >> partition_shift) as usize) & (num_partitions - 1);
                            local[p].push(i as u32);
                        }
                        local
                    })
                    .collect()
            };

            // Pass 2: per-partition CAS inserts. Each task merges rows from
            // all thread-local vecs for its partition, then inserts. The
            // full directory is atomic; CAS retry handles rare cross-slice
            // wraparound.
            let directory_atomic: &[AtomicU64] = unsafe {
                std::slice::from_raw_parts(
                    self.directory.as_ptr() as *const AtomicU64,
                    self.directory.len(),
                )
            };
            let mask = self.capacity_mask;
            let row_size = self.arena.layout().row_size();
            let arena_addr: usize = self.arena.row_ptr(0) as usize;
            let hashes_addr = self.hashes.as_ptr() as usize;

            (0..num_partitions).into_par_iter().for_each(|p| {
                let hashes_ptr = hashes_addr as *const u64;
                for t in 0..num_threads {
                    for &i in &per_thread_chunks[t][p] {
                        let i = i as usize;
                        let h = unsafe { *hashes_ptr.add(i) };
                        let row_ptr = (arena_addr + i * row_size) as *mut u8;
                        insert_row_atomic(
                            directory_atomic,
                            mask,
                            h,
                            row_ptr,
                            Ordering::Release,
                        );
                    }
                }
            });
        }

        self.finalized = true;
    }

    #[inline]
    pub fn n_rows(&self) -> usize {
        self.arena.len()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.directory.len()
    }

    /// Parallel probe that collects probe-side rows *without* a build match.
    /// Used to grow a running accumulator across a k-way FOJ chain: at each
    /// step, unmatched probe keys become new output rows, then feed the next
    /// intermediate HT.
    pub fn probe_i32_collect_unmatched(
        &self,
        probe_hashes: &[u64],
        probe_keys: &[i32],
        chunk_size: usize,
    ) -> (Vec<i32>, Vec<u64>) {
        assert!(self.finalized, "probe before finalize");
        assert_eq!(probe_hashes.len(), probe_keys.len());

        use rayon::prelude::*;
        let n = probe_keys.len();
        let chunks: Vec<(usize, usize)> = (0..n)
            .step_by(chunk_size)
            .map(|start| (start, (start + chunk_size).min(n)))
            .collect();
        let mask = self.capacity_mask;
        let key_off = self.key_offset;
        let directory_addr = self.directory.as_ptr() as usize;

        let parts: Vec<(Vec<i32>, Vec<u64>)> = chunks
            .into_par_iter()
            .map(|(start, end)| {
                let directory_ptr = directory_addr as *const HtEntry;
                let mut ks = Vec::with_capacity(end - start);
                let mut hs = Vec::with_capacity(end - start);
                for i in start..end {
                    let h = probe_hashes[i];
                    let probe_key = probe_keys[i];
                    let salt = HtEntry::extract_salt(h);
                    let mut bucket = (h as usize) & mask;
                    let mut found = false;
                    loop {
                        let entry = unsafe { *directory_ptr.add(bucket) };
                        if !entry.is_occupied() {
                            break;
                        }
                        if entry.salt() == salt {
                            let mut p = entry.pointer();
                            while !p.is_null() {
                                let row_key = unsafe {
                                    std::ptr::read_unaligned(p.add(key_off) as *const i32)
                                };
                                if row_key == probe_key {
                                    found = true;
                                    break;
                                }
                                let next = unsafe { crate::row::row_next_ptr(p) };
                                p = (next & POINTER_MASK) as usize as *const u8;
                            }
                            break;
                        }
                        bucket = (bucket + 1) & mask;
                    }
                    if !found {
                        ks.push(probe_key);
                        hs.push(h);
                    }
                }
                (ks, hs)
            })
            .collect();

        // Merge thread-local buffers.
        let total: usize = parts.iter().map(|(k, _)| k.len()).sum();
        let mut keys = Vec::with_capacity(total);
        let mut hashes = Vec::with_capacity(total);
        for (k, h) in parts {
            keys.extend(k);
            hashes.extend(h);
        }
        (keys, hashes)
    }

    /// Parallel FULL OUTER JOIN probe — i32 key variant.
    ///
    /// Returns `(matches, probe_unmatched, build_unmatched)`, the three
    /// components of the FOJ output row count (`sum == |build| + |probe| -
    /// intersection|` for distinct-key tables).
    ///
    /// Internally:
    /// 1. Parallel probe tracks which build rows matched via an atomic
    ///    bitmap (1 bit per build row).
    /// 2. Probe-unmatched rows (no hash match) are counted per chunk.
    /// 3. After probe completes, scan the bitmap to count build rows that
    ///    never got a match — those are the right-outer output.
    pub fn probe_i32_full_outer_count(
        &self,
        probe_hashes: &[u64],
        probe_keys: &[i32],
        chunk_size: usize,
    ) -> (usize, usize, usize) {
        assert!(self.finalized, "probe before finalize");
        assert_eq!(probe_hashes.len(), probe_keys.len());

        use rayon::prelude::*;
        use std::sync::atomic::{AtomicU64, Ordering};

        let n_build = self.n_rows();
        // Bitmap: 1 bit per build row, packed into u64 words.
        let words = (n_build + 63) / 64;
        let matched: Vec<AtomicU64> = (0..words).map(|_| AtomicU64::new(0)).collect();

        // Each build row's bit index = its position in the arena (insertion
        // order). The probe callback receives `build_row_ptr`, which we
        // convert back to an index via `(ptr - arena_base) / row_size`.
        let arena_base = self.arena.row_ptr(0) as usize;
        let row_size = self.arena.layout().row_size();

        let n_probe = probe_keys.len();
        let chunks: Vec<(usize, usize)> = (0..n_probe)
            .step_by(chunk_size)
            .map(|start| (start, (start + chunk_size).min(n_probe)))
            .collect();

        let mask = self.capacity_mask;
        let key_off = self.key_offset;
        // Capture as usize to cross thread boundaries (raw ptrs aren't Sync).
        let directory_addr = self.directory.as_ptr() as usize;

        // Sum (matches, probe_unmatched) per chunk, reduce.
        let (matches, probe_unmatched) = chunks
            .into_par_iter()
            .map(|(start, end)| {
                let directory_ptr = directory_addr as *const HtEntry;
                let mut matches = 0usize;
                let mut unmatched = 0usize;
                for i in start..end {
                    let h = probe_hashes[i];
                    let probe_key = probe_keys[i];
                    let salt = HtEntry::extract_salt(h);
                    let mut bucket = (h as usize) & mask;
                    let mut found_any = false;
                    loop {
                        let entry = unsafe { *directory_ptr.add(bucket) };
                        if !entry.is_occupied() {
                            break;
                        }
                        if entry.salt() == salt {
                            let mut p = entry.pointer();
                            while !p.is_null() {
                                let row_key = unsafe {
                                    std::ptr::read_unaligned(p.add(key_off) as *const i32)
                                };
                                if row_key == probe_key {
                                    matches += 1;
                                    found_any = true;
                                    // Mark build row matched.
                                    let row_idx =
                                        (p as usize - arena_base) / row_size;
                                    let word = row_idx / 64;
                                    let bit = row_idx % 64;
                                    matched[word].fetch_or(1u64 << bit, Ordering::Relaxed);
                                }
                                let next = unsafe { crate::row::row_next_ptr(p) };
                                p = (next & POINTER_MASK) as usize as *const u8;
                            }
                            break;
                        }
                        bucket = (bucket + 1) & mask;
                    }
                    if !found_any {
                        unmatched += 1;
                    }
                }
                (matches, unmatched)
            })
            .reduce(|| (0usize, 0usize), |a, b| (a.0 + b.0, a.1 + b.1));

        // Build-unmatched: count zero bits in the bitmap that lie within
        // `[0, n_build)`. Parallel.
        let build_unmatched: usize = matched
            .par_iter()
            .enumerate()
            .map(|(w, word)| {
                let bits = word.load(Ordering::Relaxed);
                let mut zeros = (!bits).count_ones() as usize;
                // Mask off padding bits in the last word.
                if w == words - 1 {
                    let used_bits = n_build - w * 64;
                    let padding = 64 - used_bits;
                    zeros = zeros.saturating_sub(padding);
                }
                zeros
            })
            .sum();

        (matches, probe_unmatched, build_unmatched)
    }

    /// Parallel inner-join probe. Splits the input into chunks and runs
    /// `probe_i32_inner` on each chunk in the current rayon pool. Per-thread
    /// accumulators are combined at the end.
    ///
    /// `partial` is called once per thread with `(thread_local_state, row_ptr,
    /// probe_row_idx)` — identical signature to the emit callback in
    /// `probe_i32_inner`, but the state lets the caller accumulate matches
    /// without sharing a Mutex across threads.
    pub fn probe_i32_inner_parallel<S, F, I, R>(
        &self,
        probe_hashes: &[u64],
        probe_keys: &[i32],
        chunk_size: usize,
        init: I,
        partial: F,
        reduce: R,
    ) -> S
    where
        S: Send,
        F: Fn(&mut S, *const u8, usize) + Send + Sync,
        I: Fn() -> S + Send + Sync,
        R: Fn(S, S) -> S + Send + Sync,
    {
        assert!(self.finalized, "probe before finalize");
        assert_eq!(probe_hashes.len(), probe_keys.len());
        assert!(chunk_size > 0);

        use rayon::prelude::*;
        // Chunk once — faster than rayon's default splitting for this workload
        // because chunks are naturally aligned to cache-line-friendly sizes.
        let n = probe_keys.len();
        let chunks: Vec<(usize, usize)> = (0..n)
            .step_by(chunk_size)
            .map(|start| (start, (start + chunk_size).min(n)))
            .collect();

        chunks
            .into_par_iter()
            .map(|(start, end)| {
                let mut state = init();
                let hashes = &probe_hashes[start..end];
                let keys = &probe_keys[start..end];
                let mask = self.capacity_mask;
                let key_off = self.key_offset;
                let directory_ptr = self.directory.as_ptr();

                // Prefetch lookahead: DuckDB's probe batches ~2048 rows and
                // prefetches directory entries a few steps ahead. We do the
                // single-row equivalent: for probe[i], prefetch the bucket
                // for probe[i+LOOKAHEAD] so the load lands in L1 by the time
                // we need it. 16 is a sweet spot on M-series cores — enough
                // to cover a full memory stall but not so much we prefetch
                // past the chunk end.
                const LOOKAHEAD: usize = 16;
                let n = hashes.len();

                for i in 0..n {
                    // Prefetch the directory slot for a later probe.
                    if i + LOOKAHEAD < n {
                        let ph = hashes[i + LOOKAHEAD];
                        let pb = (ph as usize) & mask;
                        unsafe {
                            // Arm64 uses the same __builtin_prefetch intrinsic;
                            // on x86 it's prefetcht0. The arrow crate's
                            // prefetch helper isn't stable so we do it raw.
                            prefetch_read(directory_ptr.add(pb) as *const u8);
                        }
                    }

                    let h = hashes[i];
                    let probe_key = keys[i];
                    let salt = HtEntry::extract_salt(h);
                    let mut bucket = (h as usize) & mask;
                    loop {
                        let entry = unsafe { *directory_ptr.add(bucket) };
                        if !entry.is_occupied() {
                            break;
                        }
                        if entry.salt() == salt {
                            let mut p = entry.pointer();
                            while !p.is_null() {
                                let row_key = unsafe {
                                    std::ptr::read_unaligned(p.add(key_off) as *const i32)
                                };
                                if row_key == probe_key {
                                    partial(&mut state, p, start + i);
                                }
                                let next = unsafe { crate::row::row_next_ptr(p) };
                                p = (next & POINTER_MASK) as usize as *const u8;
                            }
                            break;
                        }
                        bucket = (bucket + 1) & mask;
                    }
                }
                state
            })
            .reduce(&init, &reduce)
    }

    /// Inner-join probe over a batch of i32 keys.
    ///
    /// Calls `emit(build_row_ptr, probe_row_idx)` once per match. A single
    /// probe key may produce multiple matches when the build side has
    /// duplicate keys.
    pub fn probe_i32_inner<F: FnMut(*const u8, usize)>(
        &self,
        probe_hashes: &[u64],
        probe_keys: &[i32],
        mut emit: F,
    ) {
        assert!(self.finalized, "probe before finalize");
        assert_eq!(probe_hashes.len(), probe_keys.len());
        let mask = self.capacity_mask;
        let key_off = self.key_offset;

        for (i, (&h, &probe_key)) in probe_hashes.iter().zip(probe_keys.iter()).enumerate() {
            let salt = HtEntry::extract_salt(h);
            let mut bucket = (h as usize) & mask;
            // Linear probe to the chain head (or an empty slot → no match).
            loop {
                let entry = self.directory[bucket];
                if !entry.is_occupied() {
                    break;
                }
                if entry.salt() == salt {
                    // Walk the chain. Each row's next_ptr is a 48-bit packed
                    // pointer (same POINTER_MASK layout as HtEntry).
                    let mut p = entry.pointer();
                    while !p.is_null() {
                        let row_key = unsafe {
                            std::ptr::read_unaligned(p.add(key_off) as *const i32)
                        };
                        if row_key == probe_key {
                            emit(p, i);
                        }
                        let next = unsafe { row_next_ptr(p) };
                        p = (next & POINTER_MASK) as usize as *const u8;
                    }
                    break;
                }
                bucket = (bucket + 1) & mask;
            }
        }
    }
}

// Ensure HEADER_SIZE is the offset the first column sits at.
const _: () = {
    assert!(HEADER_SIZE == 16);
};

#[inline]
fn insert_row_serial(directory: &mut [HtEntry], mask: usize, h: u64, row_ptr: *mut u8) {
    let mut bucket = (h as usize) & mask;
    let salt = HtEntry::extract_salt(h);
    loop {
        let entry = directory[bucket];
        if !entry.is_occupied() {
            unsafe {
                set_row_next_ptr(row_ptr, 0);
            }
            directory[bucket] = HtEntry::new(h, row_ptr);
            return;
        }
        if entry.salt() == salt {
            let old_ptr_bits = entry.0 & POINTER_MASK;
            unsafe {
                set_row_next_ptr(row_ptr, old_ptr_bits);
            }
            directory[bucket] = HtEntry::new(h, row_ptr);
            return;
        }
        bucket = (bucket + 1) & mask;
    }
}

#[inline]
fn insert_row_atomic(
    directory: &[std::sync::atomic::AtomicU64],
    mask: usize,
    h: u64,
    row_ptr: *mut u8,
    success: std::sync::atomic::Ordering,
) {
    use std::sync::atomic::Ordering;
    let mut bucket = (h as usize) & mask;
    let salt = HtEntry::extract_salt(h);
    loop {
        let current = directory[bucket].load(Ordering::Acquire);
        let entry = HtEntry(current);
        let same_chain = current == 0 || entry.salt() == salt;
        if same_chain {
            // Try to claim this bucket: link our row to the existing chain
            // (if any) then CAS. On failure another thread wrote first — retry
            // the same bucket with the observed `current`.
            let old_ptr_bits = current & POINTER_MASK;
            unsafe {
                set_row_next_ptr(row_ptr, old_ptr_bits);
            }
            let new_value = HtEntry::new(h, row_ptr).0;
            match directory[bucket].compare_exchange_weak(
                current,
                new_value,
                success,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(_) => continue, // retry same bucket
            }
        }
        bucket = (bucket + 1) & mask;
    }
}

/// Cross-arch prefetch. Uses `core::arch` intrinsics on x86-64 and
/// `std::intrinsics::prefetch_read_data` elsewhere (stable via `core::arch`
/// asm on aarch64). Falls back to a plain load otherwise.
#[inline(always)]
unsafe fn prefetch_read(p: *const u8) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        _mm_prefetch(p as *const i8, _MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    {
        // arm64 prfm pldl1keep on the pointer.
        std::arch::asm!("prfm pldl1keep, [{x}]", x = in(reg) p, options(nostack, preserves_flags));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // Force a load so the compiler can't hoist the call away.
        let _ = std::ptr::read_volatile(p);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::murmur_hash32;
    use crate::row::PhysType;

    fn make_ht(payload_types: &[PhysType]) -> JoinHashTable {
        let mut tys = vec![PhysType::I32]; // key
        tys.extend_from_slice(payload_types);
        JoinHashTable::new(RowLayout::new(tys))
    }

    #[test]
    fn empty_ht_finalizes() {
        let mut ht = make_ht(&[]);
        ht.finalize();
        assert_eq!(ht.n_rows(), 0);
        assert_eq!(ht.capacity(), MIN_CAPACITY);
    }

    #[test]
    fn capacity_is_power_of_two_min_16k() {
        assert_eq!(directory_capacity(0), MIN_CAPACITY);
        assert_eq!(directory_capacity(7), MIN_CAPACITY);
        assert_eq!(directory_capacity(100_000), 262_144);
        assert_eq!(directory_capacity(1_000_000), 2_097_152);
    }

    #[test]
    fn single_row_roundtrip() {
        let mut ht = make_ht(&[]);
        ht.reserve(1);
        ht.insert_test_row(42, &[]);
        ht.finalize();
        let mut hits = 0usize;
        let probe_key = 42i32;
        let probe_hash = murmur_hash32(42u32);
        ht.probe_i32_inner(&[probe_hash], &[probe_key], |_row, _idx| hits += 1);
        assert_eq!(hits, 1);
    }

    #[test]
    fn nonmatching_probe_zero_hits() {
        let mut ht = make_ht(&[]);
        ht.reserve(1);
        ht.insert_test_row(42, &[]);
        ht.finalize();
        let probe_key = 99i32;
        let probe_hash = murmur_hash32(99u32);
        let mut hits = 0;
        ht.probe_i32_inner(&[probe_hash], &[probe_key], |_, _| hits += 1);
        assert_eq!(hits, 0);
    }

    #[test]
    fn duplicate_build_keys_all_emit() {
        let mut ht = make_ht(&[]);
        ht.reserve(3);
        for _ in 0..3 {
            ht.insert_test_row(42, &[]);
        }
        ht.finalize();
        let mut hits = 0;
        let probe_hash = murmur_hash32(42u32);
        ht.probe_i32_inner(&[probe_hash], &[42i32], |_, _| hits += 1);
        assert_eq!(hits, 3);
    }

    #[test]
    fn batch_probe_mixed_hits() {
        let mut ht = make_ht(&[]);
        ht.reserve(4);
        for k in [1, 2, 3, 2] {
            ht.insert_test_row(k, &[]);
        }
        ht.finalize();
        let probe_keys: Vec<i32> = vec![1, 2, 3, 4, 5];
        let probe_hashes: Vec<u64> = probe_keys.iter().map(|&k| murmur_hash32(k as u32)).collect();
        let mut per_key_hits = vec![0usize; probe_keys.len()];
        ht.probe_i32_inner(&probe_hashes, &probe_keys, |_row, idx| {
            per_key_hits[idx] += 1;
        });
        assert_eq!(per_key_hits, vec![1, 2, 1, 0, 0]);
    }

    #[test]
    fn larger_sweep_no_false_negatives() {
        // Insert 10 000 distinct keys; probe every one — each must hit exactly once.
        let n = 10_000i32;
        let mut ht = make_ht(&[]);
        ht.reserve(n as usize);
        for k in 0..n {
            ht.insert_test_row(k, &[]);
        }
        ht.finalize();
        let keys: Vec<i32> = (0..n).collect();
        let hashes: Vec<u64> = keys.iter().map(|&k| murmur_hash32(k as u32)).collect();
        let mut hits = 0usize;
        ht.probe_i32_inner(&hashes, &keys, |_, _| hits += 1);
        assert_eq!(hits as i32, n);
    }
}
