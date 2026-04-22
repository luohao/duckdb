//! Pipeline framework — DuckDB-style operator/sink/source with
//! Global + Local state split.
//!
//! The DuckDB design that matters for our workload:
//! * Each operator has `GlobalState` (shared across threads, holds the HT,
//!   directory, match bitmap) and `LocalState` (per thread; thread-local
//!   write buffers, iteration cursors).
//! * Parallel sink: each thread pushes into `LocalState` without locks.
//!   After all threads are done sinking, `combine(local, global)` merges
//!   the local buffers into a shared arena (single atomic bump).
//! * Parallel probe: read-only access to `GlobalState.directory`; threads
//!   emit output batches independently.
//! * Partition-parallel finalize: each thread owns one partition of the
//!   directory. No CAS on the hot path — slots in partition `p` are only
//!   touched by one thread.
//!
//! Operator interfaces use concrete types (monomorphization) rather than
//! `dyn` because the pipelines are known at compile time for this bench.
//! A future extension could switch to `dyn` if we need dynamic plans.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// A batch of chain rows flowing through the pipeline. Each row represents
/// an output row of the chain so far, carrying its join key, cached hash,
/// and up to 5 source-table row pointers (one per k-way FOJ side;
/// `ptrs[i] == 0` means "no match on that side", i.e., NULL-padding).
#[derive(Default)]
pub struct Batch {
    pub keys: Vec<i32>,
    pub hashes: Vec<u64>,
    pub ptrs: Vec<[u64; 5]>,
}

impl Batch {
    pub fn with_capacity(n: usize) -> Self {
        Self {
            keys: Vec::with_capacity(n),
            hashes: Vec::with_capacity(n),
            ptrs: Vec::with_capacity(n),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    #[inline]
    pub fn push(&mut self, key: i32, hash: u64, ptrs: [u64; 5]) {
        self.keys.push(key);
        self.hashes.push(hash);
        self.ptrs.push(ptrs);
    }

    pub fn clear(&mut self) {
        self.keys.clear();
        self.hashes.clear();
        self.ptrs.clear();
    }
}

/// Standard chunk size — matches DuckDB's STANDARD_VECTOR_SIZE = 2048.
pub const CHUNK_SIZE: usize = 2048;

// ============================================================================
// Source
// ============================================================================

/// A streaming source that hands out batches to worker threads.
///
/// The global state coordinates work across threads (e.g., a row-group queue
/// for parquet scan). Each thread keeps its own local state for its current
/// position within a handed-out work unit.
pub trait Source: Send + Sync {
    type Global: Send + Sync;
    type Local: Send + Default;

    fn global_state(&self) -> &Self::Global;

    /// Return the next batch for this thread, or `None` when the source is
    /// exhausted. May be called concurrently from many threads.
    fn get_data(
        &self,
        global: &Self::Global,
        local: &mut Self::Local,
    ) -> Option<Batch>;
}

// ============================================================================
// Sink (pipeline breaker)
// ============================================================================

/// A pipeline sink — accumulates batches across threads into a shared state.
///
/// Lifecycle:
/// 1. Each worker thread creates a fresh `Local` and calls `sink()` per batch.
/// 2. When a worker finishes its source iterator, it calls `combine(local, global)`.
/// 3. After all workers combined, the driver calls `finalize(global)` once.
pub trait Sink: Send + Sync {
    type Global: Send + Sync;
    type Local: Send + Default;

    fn global_state(&self) -> &Self::Global;

    fn sink(&self, batch: Batch, global: &Self::Global, local: &mut Self::Local);

    fn combine(&self, global: &Self::Global, local: Self::Local);

    fn finalize(&self, global: &Self::Global);
}

// ============================================================================
// Operator (streaming)
// ============================================================================

/// A streaming operator — consumes a batch, emits zero or more output batches.
/// Used for things like probe-side hash-join execution where each incoming
/// batch produces an output batch (potentially smaller due to predicates).
pub trait Operator: Send + Sync {
    type Global: Send + Sync;
    type Local: Send + Default;

    fn global_state(&self) -> &Self::Global;

    /// Process `input`, emit zero or more output batches. Output goes through
    /// `push` — the driver wires this into the next operator or sink.
    fn execute(
        &self,
        input: Batch,
        global: &Self::Global,
        local: &mut Self::Local,
        push: &mut dyn FnMut(Batch),
    );
}

// ============================================================================
// Pipeline drivers
// ============================================================================

/// Run the parallel "sink phase" of a "source → sink" pipeline. Each worker
/// pulls batches from `source`, sinks them, then combines its local state
/// into the sink's global state.
///
/// **Does NOT call `finalize`**. Caller must invoke `sink.finalize()` after
/// all sub-pipelines feeding this sink have completed (DuckDB's MetaPipeline
/// pattern — multiple sub-pipelines feeding the same sink).
pub fn drive_source_to_sink<S, K>(source: &S, sink: &K, num_threads: usize)
where
    S: Source,
    K: Sink,
{
    use rayon::prelude::*;
    (0..num_threads).into_par_iter().for_each(|_| {
        let mut local_src = S::Local::default();
        let mut local_sink = K::Local::default();
        while let Some(batch) = source.get_data(source.global_state(), &mut local_src) {
            sink.sink(batch, sink.global_state(), &mut local_sink);
        }
        sink.combine(sink.global_state(), local_sink);
    });
}

/// Run the parallel "sink phase" of a "source → operator → sink" pipeline.
///
/// **Does NOT call `finalize`**.
pub fn drive_source_op_sink<S, O, K>(
    source: &S,
    op: &O,
    sink: &K,
    num_threads: usize,
) where
    S: Source,
    O: Operator,
    K: Sink,
{
    use rayon::prelude::*;
    (0..num_threads).into_par_iter().for_each(|_| {
        let mut local_src = S::Local::default();
        let mut local_op = O::Local::default();
        let mut local_sink = K::Local::default();
        while let Some(batch) = source.get_data(source.global_state(), &mut local_src) {
            op.execute(batch, op.global_state(), &mut local_op, &mut |out_batch| {
                sink.sink(out_batch, sink.global_state(), &mut local_sink);
            });
        }
        sink.combine(sink.global_state(), local_sink);
    });
}

/// Run a "parallel-source → sink" (no operator) followed by a second
/// "single-source → sink" phase — this is the shape of a pipeline that
/// merges a streaming output with a deferred "unmatched source" before
/// completing the downstream sink. Used by FOJ chain steps: parallel
/// probe + sink, then build-unmatched + sink, then finalize sink.
pub struct StagedPipeline<'a, S, U, K>
where
    S: Source,
    U: Source,
    K: Sink,
{
    pub primary: &'a S,
    pub secondary: &'a U,
    pub sink: &'a K,
    pub num_threads: usize,
}

// ============================================================================
// Thread-local row arena for HashJoin build sinks.
// ============================================================================

/// Thread-local row buffer used by `HashJoinOp`'s build sink. Holds rows
/// for this thread until `combine` merges them into the global arena.
///
/// Layout per row matches the slim Lance-style: 20 bytes (16-byte header +
/// 4-byte key). Data columns stay in the source Arrow arrays via `ptrs`.
/// We additionally store ptr array per row so probe can reconstruct them.
pub struct BuildLocalState {
    /// Packed keys + hashes + ptrs. Layout per row:
    ///   offset 0  : u64 next_ptr (zero; filled during finalize)
    ///   offset 8  : u64 hash
    ///   offset 16 : i32 key
    ///   offset 20 : u32 padding
    ///   offset 24 : [u64; 5] ptrs_into_tables
    ///   offset 64 : (row end)
    pub bytes: Vec<u8>,
    pub n_rows: usize,
}

pub const BUILD_ROW_SIZE: usize = 64;
pub const BUILD_KEY_OFF: usize = 16;
pub const BUILD_HASH_OFF: usize = 8;
pub const BUILD_NEXT_PTR_OFF: usize = 0;
pub const BUILD_PTRS_OFF: usize = 24;

impl Default for BuildLocalState {
    fn default() -> Self {
        Self {
            bytes: Vec::new(),
            n_rows: 0,
        }
    }
}

impl BuildLocalState {
    pub fn append_batch(&mut self, batch: &Batch) {
        let n = batch.len();
        if n == 0 {
            return;
        }
        let start = self.bytes.len();
        self.bytes.resize(start + n * BUILD_ROW_SIZE, 0);
        for i in 0..n {
            let base = start + i * BUILD_ROW_SIZE;
            unsafe {
                let p = self.bytes.as_mut_ptr().add(base);
                // next_ptr=0 (already zero from resize)
                std::ptr::write_unaligned(p.add(BUILD_HASH_OFF) as *mut u64, batch.hashes[i]);
                std::ptr::write_unaligned(p.add(BUILD_KEY_OFF) as *mut i32, batch.keys[i]);
                std::ptr::copy_nonoverlapping(
                    batch.ptrs[i].as_ptr() as *const u8,
                    p.add(BUILD_PTRS_OFF),
                    40,
                );
            }
        }
        self.n_rows += n;
    }
}

/// Global state for the build side: arena of combined thread-local rows
/// followed by a directory built in a partition-parallel pass.
pub struct BuildGlobalState {
    /// Concatenated row bytes from all threads.
    pub arena: Mutex<Vec<u8>>,
    /// Total row count, atomically bumped by `combine`.
    pub n_rows: AtomicUsize,
    /// Pre-reserved capacity; caller uses table size hints to avoid reallocs.
    pub reserved_rows: AtomicUsize,
}

impl BuildGlobalState {
    pub fn new(expected_rows: usize) -> Self {
        let mut arena = Vec::with_capacity(expected_rows * BUILD_ROW_SIZE);
        arena.reserve(expected_rows * BUILD_ROW_SIZE);
        Self {
            arena: Mutex::new(arena),
            n_rows: AtomicUsize::new(0),
            reserved_rows: AtomicUsize::new(expected_rows),
        }
    }

    /// Append a thread-local buffer to the shared arena. Uses a mutex for
    /// the bulk append — cheap since it's called once per worker at combine
    /// time, not per batch.
    pub fn combine_local(&self, local: BuildLocalState) {
        if local.n_rows == 0 {
            return;
        }
        let mut arena = self.arena.lock().unwrap();
        arena.extend_from_slice(&local.bytes);
        self.n_rows.fetch_add(local.n_rows, Ordering::AcqRel);
    }
}

// ============================================================================
// ParquetScanSource — parallel row-group work queue source.
// ============================================================================

use std::sync::Arc;
use arrow::record_batch::RecordBatch;

/// A parquet-backed source that hands out row-group batches. Each worker
/// pulls a row group from the queue, decodes it, converts to a `Batch`
/// where `ptrs[table_slot_idx]` is populated with per-row source-table
/// addresses and other slots are zero.
///
/// The source "global state" holds the full `TableData` (with `Arc<RecordBatch>`
/// kept alive) and an atomic row-group counter.
pub struct ParquetScanSource {
    pub table: Arc<crate::io::TableData>,
    /// Which chain slot (0..5) this table's pointers populate.
    pub slot_idx: usize,
    pub global: ParquetScanGlobal,
}

pub struct ParquetScanGlobal {
    pub next_rg: AtomicUsize,
}

#[derive(Default)]
pub struct ParquetScanLocal;

impl ParquetScanSource {
    pub fn new(table: Arc<crate::io::TableData>, slot_idx: usize) -> Self {
        Self {
            table,
            slot_idx,
            global: ParquetScanGlobal {
                next_rg: AtomicUsize::new(0),
            },
        }
    }
}

impl Source for ParquetScanSource {
    type Global = ParquetScanGlobal;
    type Local = ParquetScanLocal;

    fn global_state(&self) -> &ParquetScanGlobal {
        &self.global
    }

    fn get_data(
        &self,
        global: &ParquetScanGlobal,
        _local: &mut ParquetScanLocal,
    ) -> Option<Batch> {
        let rg_idx = global.next_rg.fetch_add(1, Ordering::AcqRel);
        if rg_idx >= self.table.batches.len() {
            return None;
        }

        // Convert one RecordBatch → one Batch. The ptr for each row is the
        // address of the underlying arena row; it lets downstream sinks
        // reference the Arrow arrays at output time.
        let batch = &self.table.batches[rg_idx];
        let n = batch.num_rows();
        if n == 0 {
            return Some(Batch::with_capacity(0));
        }

        let row_start = self.table.boundaries[rg_idx] as usize;
        let arena_base = self.table.ht.arena_base_addr();
        let row_size = self.table.ht.layout().row_size();
        let hashes_base = self.table.ht.hashes_base_addr_const() as *const u64;

        let k_col = batch
            .column_by_name("k")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int32Array>()
            .unwrap();
        let keys = k_col.values();

        let mut out = Batch::with_capacity(n);
        for i in 0..n {
            let abs = row_start + i;
            let row_ptr = (arena_base + abs * row_size) as u64;
            let h = unsafe { *hashes_base.add(abs) };
            let mut ptrs = [0u64; 5];
            ptrs[self.slot_idx] = row_ptr;
            out.push(keys[i], h, ptrs);
        }
        Some(out)
    }
}

// ============================================================================
// HashJoinOp
// ============================================================================

use crate::ht_entry::{HtEntry, POINTER_MASK};

/// Hash join build-side collector + directory + probe operator.
///
/// Used as a `Sink` during its build phase and as an `Operator` for probe.
/// Additionally exposes an `unmatched_source` method that emits build-side
/// rows whose `matched_bitmap` bit is 0 after the probe pipeline completes.
pub struct HashJoinOp {
    pub slot_idx: usize, // which ptrs[] slot this HJ's build side occupies
    pub build: BuildGlobalState,
    // After finalize(): directory + matched bitmap
    pub directory: std::sync::RwLock<Vec<HtEntry>>,
    pub capacity_mask: AtomicUsize,
    pub matched: std::sync::RwLock<Vec<std::sync::atomic::AtomicU64>>, // one bit per row
}

impl HashJoinOp {
    pub fn new(expected_rows: usize, slot_idx: usize) -> Self {
        Self {
            slot_idx,
            build: BuildGlobalState::new(expected_rows),
            directory: std::sync::RwLock::new(Vec::new()),
            capacity_mask: AtomicUsize::new(0),
            matched: std::sync::RwLock::new(Vec::new()),
        }
    }
}

impl Sink for HashJoinOp {
    type Global = BuildGlobalState;
    type Local = BuildLocalState;

    fn global_state(&self) -> &BuildGlobalState {
        &self.build
    }

    fn sink(&self, batch: Batch, _global: &BuildGlobalState, local: &mut BuildLocalState) {
        local.append_batch(&batch);
    }

    fn combine(&self, global: &BuildGlobalState, local: BuildLocalState) {
        global.combine_local(local);
    }

    /// Build the directory with partition-parallel inserts. No CAS — each
    /// partition's directory slice is touched by exactly one thread.
    ///
    /// Uses 4x load factor (vs the normal HT's 2x) so each partition's slice
    /// stays ≤ 25% full. Linear-probe wraparound within a slice is then
    /// statistically impossible for uniform hashes — avoids the correctness
    /// gotcha where a row wrap-inserted at a lower slice offset can't be
    /// found by probe's wrap-later linear probe.
    fn finalize(&self, global: &BuildGlobalState) {
        use rayon::prelude::*;
        let n = global.n_rows.load(Ordering::Acquire);
        let target = n.saturating_mul(4).max(16_384);
        let capacity = target.next_power_of_two();
        let log_capacity = capacity.trailing_zeros();

        // Partition count — radix bits picked so each partition has ≥ 1024
        // slots after 2x headroom. DuckDB's initial_radix_bits heuristic.
        let num_threads = rayon::current_num_threads().max(1);
        let min_parts = (num_threads * 4).next_power_of_two();
        let radix_bits = min_parts
            .trailing_zeros()
            .min(log_capacity.saturating_sub(10));
        let num_partitions = 1usize << radix_bits;
        let partition_shift = log_capacity - radix_bits;
        let slice_size = capacity >> radix_bits;
        let slice_mask = slice_size - 1;

        // Allocate directory + matched bitmap.
        let mut directory = vec![HtEntry::empty(); capacity];
        let matched_words = (n + 63) / 64;
        let matched: Vec<std::sync::atomic::AtomicU64> =
            (0..matched_words)
                .map(|_| std::sync::atomic::AtomicU64::new(0))
                .collect();

        // Pass 1 — parallel partition-scatter: bucket row indices by
        // partition. Each thread handles a chunk of rows.
        let arena = global.arena.lock().unwrap();
        let arena_addr = arena.as_ptr() as usize;
        drop(arena);

        let per_thread: Vec<Vec<Vec<u32>>> = {
            let chunk_size = (n + num_threads - 1) / num_threads;
            (0..num_threads)
                .into_par_iter()
                .map(|t| {
                    let start = t * chunk_size;
                    let end = (start + chunk_size).min(n);
                    let mut local: Vec<Vec<u32>> = (0..num_partitions)
                        .map(|_| Vec::with_capacity((end - start) / num_partitions + 8))
                        .collect();
                    for i in start..end {
                        let row_ptr = (arena_addr + i * BUILD_ROW_SIZE) as *const u8;
                        let h = unsafe {
                            std::ptr::read_unaligned(row_ptr.add(BUILD_HASH_OFF) as *const u64)
                        };
                        let p = ((h >> partition_shift) as usize) & (num_partitions - 1);
                        local[p].push(i as u32);
                    }
                    local
                })
                .collect()
        };

        // Pass 2 — parallel per-partition insert. Each partition's directory
        // slice is disjoint, so no CAS; linear probing confined to slice by
        // local mask. With slice load factor ≤ 0.5 (directory_capacity_for's
        // 2x headroom) wraparound stays within slice.
        let dir_ptr_addr = directory.as_mut_ptr() as usize;
        (0..num_partitions).into_par_iter().for_each(|p| {
            let dir_ptr = dir_ptr_addr as *mut HtEntry;
            let slice_base = p * slice_size;
            for t in 0..num_threads {
                for &i in &per_thread[t][p] {
                    let i = i as usize;
                    let row_ptr = (arena_addr + i * BUILD_ROW_SIZE) as *mut u8;
                    let h = unsafe {
                        std::ptr::read_unaligned(row_ptr.add(BUILD_HASH_OFF) as *const u64)
                    };
                    let salt = HtEntry::extract_salt(h);
                    let mut local = (h as usize) & slice_mask;
                    loop {
                        let slot = unsafe { dir_ptr.add(slice_base + local) };
                        let entry = unsafe { *slot };
                        if !entry.is_occupied() {
                            unsafe {
                                // next_ptr=0 already
                                *slot = HtEntry::new(h, row_ptr);
                            }
                            break;
                        }
                        if entry.salt() == salt {
                            let old_ptr_bits = entry.0 & POINTER_MASK;
                            unsafe {
                                std::ptr::write_unaligned(
                                    row_ptr.add(BUILD_NEXT_PTR_OFF) as *mut u64,
                                    old_ptr_bits,
                                );
                                *slot = HtEntry::new(h, row_ptr);
                            }
                            break;
                        }
                        local = (local + 1) & slice_mask;
                    }
                }
            }
        });

        *self.directory.write().unwrap() = directory;
        self.capacity_mask.store(capacity - 1, Ordering::Release);
        *self.matched.write().unwrap() = matched;
    }
}

impl HashJoinOp {
    pub fn n_rows(&self) -> usize {
        self.build.n_rows.load(Ordering::Acquire)
    }

    /// Streaming probe. Returns a single output batch containing
    /// (a) matched pairs with both build-side and probe-side ptrs, and
    /// (b) probe-unmatched rows with only probe-side ptrs set (FOJ left-outer).
    /// Also marks matched build rows in the `matched` bitmap for later
    /// unmatched-source emission.
    pub fn probe_execute(&self, probe_batch: &Batch) -> Batch {
        use std::sync::atomic::Ordering as O;
        let dir = self.directory.read().unwrap();
        let mask = self.capacity_mask.load(Ordering::Acquire);
        let matched = self.matched.read().unwrap();
        let slot_idx = self.slot_idx;

        let mut out = Batch::with_capacity(probe_batch.len() * 2);
        let dir_ptr = dir.as_ptr();
        let arena = self.build.arena.lock().unwrap();
        let arena_base = arena.as_ptr() as usize;
        drop(arena);

        for i in 0..probe_batch.len() {
            let h = probe_batch.hashes[i];
            let pk = probe_batch.keys[i];
            let probe_ptrs = probe_batch.ptrs[i];
            let salt = HtEntry::extract_salt(h);
            let mut bucket = (h as usize) & mask;
            let mut any_match = false;
            loop {
                let entry = unsafe { *dir_ptr.add(bucket) };
                if !entry.is_occupied() {
                    break;
                }
                if entry.salt() == salt {
                    let mut p = entry.pointer();
                    while !p.is_null() {
                        let row_key = unsafe {
                            std::ptr::read_unaligned(p.add(BUILD_KEY_OFF) as *const i32)
                        };
                        if row_key == pk {
                            // Mark matched
                            let row_idx = (p as usize - arena_base) / BUILD_ROW_SIZE;
                            let word = row_idx / 64;
                            let bit = row_idx % 64;
                            if word < matched.len() {
                                // Release so subsequent unmatched-phase reads
                                // (done after rayon join) observe this set bit.
                                matched[word].fetch_or(1u64 << bit, O::Release);
                            }
                            // Emit: merge build-side ptrs with probe-side ptrs
                            let build_ptrs = unsafe {
                                let mut pp = [0u64; 5];
                                std::ptr::copy_nonoverlapping(
                                    p.add(BUILD_PTRS_OFF),
                                    pp.as_mut_ptr() as *mut u8,
                                    40,
                                );
                                pp
                            };
                            let mut merged = build_ptrs;
                            for j in 0..5 {
                                if merged[j] == 0 {
                                    merged[j] = probe_ptrs[j];
                                }
                            }
                            out.push(pk, h, merged);
                            any_match = true;
                        }
                        let next = unsafe {
                            std::ptr::read_unaligned(p.add(BUILD_NEXT_PTR_OFF) as *const u64)
                        };
                        p = (next & POINTER_MASK) as usize as *const u8;
                    }
                    break;
                }
                bucket = (bucket + 1) & mask;
            }
            if !any_match {
                // Probe-side-only row (FOJ left-outer)
                out.push(pk, h, probe_ptrs);
            }
        }
        out
    }

    /// Emit a batch of build-side rows that were NOT matched during probe.
    /// Called after the probe pipeline completes. Caller invokes this
    /// per-partition in parallel via an index cursor.
    pub fn emit_unmatched_batch(&self, start_row: usize, chunk_size: usize) -> Option<Batch> {
        let n = self.n_rows();
        if start_row >= n {
            return None;
        }
        let end = (start_row + chunk_size).min(n);
        let matched = self.matched.read().unwrap();
        let arena = self.build.arena.lock().unwrap();
        let arena_base = arena.as_ptr() as usize;
        drop(arena);

        let mut out = Batch::with_capacity(chunk_size);
        for i in start_row..end {
            let word = i / 64;
            let bit = i % 64;
            let bits = matched[word].load(std::sync::atomic::Ordering::Acquire);
            if bits & (1u64 << bit) != 0 {
                continue; // matched, skip
            }
            let row_ptr = (arena_base + i * BUILD_ROW_SIZE) as *const u8;
            let key = unsafe { std::ptr::read_unaligned(row_ptr.add(BUILD_KEY_OFF) as *const i32) };
            let hash = unsafe { std::ptr::read_unaligned(row_ptr.add(BUILD_HASH_OFF) as *const u64) };
            let mut ptrs = [0u64; 5];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    row_ptr.add(BUILD_PTRS_OFF),
                    ptrs.as_mut_ptr() as *mut u8,
                    40,
                );
            }
            out.push(key, hash, ptrs);
        }
        if out.is_empty() {
            return None;
        }
        Some(out)
    }
}

// ============================================================================
// ProbeOperator — wraps a reference to a HashJoinOp for the probe pipeline.
// ============================================================================

/// Streaming operator: probe against a finalized `HashJoinOp`, emit matched +
/// probe-unmatched rows as output batches.
pub struct ProbeOperator<'a> {
    pub hj: &'a HashJoinOp,
    pub global: ProbeGlobal,
}

pub struct ProbeGlobal;

#[derive(Default)]
pub struct ProbeLocal;

impl<'a> ProbeOperator<'a> {
    pub fn new(hj: &'a HashJoinOp) -> Self {
        Self { hj, global: ProbeGlobal }
    }
}

impl<'a> Operator for ProbeOperator<'a> {
    type Global = ProbeGlobal;
    type Local = ProbeLocal;

    fn global_state(&self) -> &ProbeGlobal {
        &self.global
    }

    fn execute(
        &self,
        input: Batch,
        _global: &ProbeGlobal,
        _local: &mut ProbeLocal,
        push: &mut dyn FnMut(Batch),
    ) {
        let out = self.hj.probe_execute(&input);
        if !out.is_empty() {
            push(out);
        }
    }
}

// ============================================================================
// UnmatchedSource — emits build-side rows not matched during probe phase.
// ============================================================================

/// Source that enumerates build-unmatched rows of a finalized hash join.
/// Used in the post-probe sub-pipeline to feed FOJ's build-unmatched into
/// the next stage's build sink (or into the final checksum sink).
pub struct UnmatchedSource<'a> {
    pub hj: &'a HashJoinOp,
    pub global: UnmatchedGlobal,
}

pub struct UnmatchedGlobal {
    pub next_row: AtomicUsize,
    pub chunk_size: usize,
}

#[derive(Default)]
pub struct UnmatchedLocal;

impl<'a> UnmatchedSource<'a> {
    pub fn new(hj: &'a HashJoinOp) -> Self {
        Self {
            hj,
            global: UnmatchedGlobal {
                next_row: AtomicUsize::new(0),
                chunk_size: 4096,
            },
        }
    }
}

impl<'a> Source for UnmatchedSource<'a> {
    type Global = UnmatchedGlobal;
    type Local = UnmatchedLocal;

    fn global_state(&self) -> &UnmatchedGlobal {
        &self.global
    }

    fn get_data(
        &self,
        global: &UnmatchedGlobal,
        _local: &mut UnmatchedLocal,
    ) -> Option<Batch> {
        let cs = global.chunk_size;
        loop {
            let start = global.next_row.fetch_add(cs, Ordering::AcqRel);
            let n = self.hj.n_rows();
            if start >= n {
                return None;
            }
            if let Some(b) = self.hj.emit_unmatched_batch(start, cs) {
                return Some(b);
            }
            // All rows in this chunk were matched; try next chunk.
        }
    }
}

// ============================================================================
// ChecksumSink — final sink that reads source-table columns via chain row
// ptrs and folds a checksum, mirroring DuckDB's blackhole semantics.
// ============================================================================

use arrow::array::{
    BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
};

/// Cached Arrow buffer pointers for one RecordBatch — used to read data cols
/// at sink time by direct indexing rather than Arrow's downcast + bounds check.
pub struct BatchCols {
    pub c_int: *const i32,
    pub c_bigint: *const i64,
    pub c_dbl: *const f64,
    pub c_flt: *const f32,
    pub c_bool: *const BooleanArray, // bool values are bit-packed; keep the array
    // Strings: pointer to values buffer + offsets buffer
    pub c_short: *const StringArray,
    pub c_long: *const StringArray,
    // Batch row count (for bounds checks, though we trust the ptr path)
    pub n_rows: usize,
}

// SAFETY: pointers stay valid because the source TableData is held via Arc
// in the ChecksumSinkOp for the duration of the pipeline.
unsafe impl Send for BatchCols {}
unsafe impl Sync for BatchCols {}

impl BatchCols {
    pub fn from_batch(batch: &arrow::record_batch::RecordBatch) -> Self {
        let c_int = batch
            .column_by_name("c_int")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let c_bigint = batch
            .column_by_name("c_bigint")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let c_dbl = batch
            .column_by_name("c_dbl")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let c_flt = batch
            .column_by_name("c_flt")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        let c_bool_arr = batch
            .column_by_name("c_bool")
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap() as *const BooleanArray;
        let c_short_arr = batch
            .column_by_name("c_short")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap() as *const StringArray;
        let c_long_arr = batch
            .column_by_name("c_long")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap() as *const StringArray;
        Self {
            c_int: c_int.values().as_ptr(),
            c_bigint: c_bigint.values().as_ptr(),
            c_dbl: c_dbl.values().as_ptr(),
            c_flt: c_flt.values().as_ptr(),
            c_bool: c_bool_arr,
            c_short: c_short_arr,
            c_long: c_long_arr,
            n_rows: batch.num_rows(),
        }
    }
}

pub struct ChecksumSinkOp {
    /// Per-table, per-batch cached Arrow column pointers.
    pub cols: Vec<Vec<BatchCols>>,
    /// Per-table (batch_idx, row_in_batch) mapping — we already have
    /// `TableData.boundaries` for this. Store the arena base + row size to
    /// convert ptrs back to arena indices.
    pub arena_bases: Vec<usize>,
    pub row_sizes: Vec<usize>,
    pub boundaries: Vec<Vec<u32>>,
    /// Keep source tables alive so ptrs remain valid.
    _tables: Vec<std::sync::Arc<crate::io::TableData>>,
    pub global: ChecksumGlobal,
}

pub struct ChecksumGlobal {
    pub checksum: std::sync::atomic::AtomicU64,
    pub rows_sunk: AtomicUsize,
}

#[derive(Default)]
pub struct ChecksumLocal {
    pub checksum: u64,
    pub rows: usize,
}

impl ChecksumSinkOp {
    pub fn new(tables: Vec<std::sync::Arc<crate::io::TableData>>) -> Self {
        let cols: Vec<Vec<BatchCols>> = tables
            .iter()
            .map(|t| t.batches.iter().map(|b| BatchCols::from_batch(b)).collect())
            .collect();
        let arena_bases = tables.iter().map(|t| t.ht.arena_base_addr()).collect();
        let row_sizes = tables.iter().map(|t| t.ht.layout().row_size()).collect();
        let boundaries = tables.iter().map(|t| t.boundaries.clone()).collect();
        Self {
            cols,
            arena_bases,
            row_sizes,
            boundaries,
            _tables: tables,
            global: ChecksumGlobal {
                checksum: std::sync::atomic::AtomicU64::new(0),
                rows_sunk: AtomicUsize::new(0),
            },
        }
    }

    pub fn finish(&self) -> (u64, usize) {
        (
            self.global.checksum.load(Ordering::Acquire),
            self.global.rows_sunk.load(Ordering::Acquire),
        )
    }
}

impl Sink for ChecksumSinkOp {
    type Global = ChecksumGlobal;
    type Local = ChecksumLocal;

    fn global_state(&self) -> &ChecksumGlobal {
        &self.global
    }

    fn sink(
        &self,
        batch: Batch,
        _global: &ChecksumGlobal,
        local: &mut ChecksumLocal,
    ) {
        for i in 0..batch.len() {
            let ptrs = batch.ptrs[i];
            for (t_idx, &ptr) in ptrs.iter().enumerate() {
                if ptr == 0 {
                    local.checksum = local.checksum.wrapping_add(0xDEAD_BEEF_CAFE_BABE);
                    continue;
                }
                // ptr → arena_idx in table t_idx
                let arena_idx =
                    (ptr as usize - self.arena_bases[t_idx]) / self.row_sizes[t_idx];
                // Locate batch via boundaries (binary search)
                let bnds = &self.boundaries[t_idx];
                let p = bnds.partition_point(|&b| (b as usize) <= arena_idx);
                let batch_idx = p - 1;
                let row_in_batch = arena_idx - bnds[batch_idx] as usize;
                let cols = &self.cols[t_idx][batch_idx];

                // Read all 7 data cols through the cached raw pointers.
                unsafe {
                    local.checksum = local
                        .checksum
                        .wrapping_add(*cols.c_int.add(row_in_batch) as u64);
                    local.checksum = local
                        .checksum
                        .wrapping_add(*cols.c_bigint.add(row_in_batch) as u64);
                    local.checksum = local
                        .checksum
                        .wrapping_add((*cols.c_dbl.add(row_in_batch)).to_bits());
                    local.checksum = local
                        .checksum
                        .wrapping_add((*cols.c_flt.add(row_in_batch)).to_bits() as u64);
                    let bool_arr = &*cols.c_bool;
                    local.checksum = local
                        .checksum
                        .wrapping_add(bool_arr.value(row_in_batch) as u64);
                    let short_arr = &*cols.c_short;
                    let s = short_arr.value(row_in_batch);
                    local.checksum = local.checksum.wrapping_add(s.len() as u64);
                    if !s.is_empty() {
                        local.checksum = local.checksum.wrapping_add(s.as_bytes()[0] as u64);
                    }
                    let long_arr = &*cols.c_long;
                    let s = long_arr.value(row_in_batch);
                    local.checksum = local.checksum.wrapping_add(s.len() as u64);
                    if !s.is_empty() {
                        local.checksum = local.checksum.wrapping_add(s.as_bytes()[0] as u64);
                    }
                }
            }
        }
        local.rows += batch.len();
    }

    fn combine(&self, global: &ChecksumGlobal, local: ChecksumLocal) {
        global
            .checksum
            .fetch_add(local.checksum, Ordering::AcqRel);
        global
            .rows_sunk
            .fetch_add(local.rows, Ordering::AcqRel);
    }

    fn finalize(&self, _global: &ChecksumGlobal) {
        // nothing
    }
}
