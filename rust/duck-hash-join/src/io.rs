//! Parquet ingress — read a parquet file, extract the `k` (i32) join key
//! plus a subset of numeric data columns, write them into a `RowArena`.
//!
//! MVP: hardcoded to the `kway_bench` schema (see `tools/utils/kway_bench.cpp`
//! line 192: `k, c_int, c_bigint, c_dbl, c_flt, c_short, c_long, c_date,
//! c_ts, c_bool, c_dec`). Only the numeric subset is ingested — strings get
//! skipped. That's enough to reproduce the hot path (hash + row copy +
//! probe + output).
//!
//! The numeric subset used by the bench: `k`, `c_int` (i32), `c_bigint`
//! (i64), `c_dbl` (f64), `c_flt` (f32), `c_bool` (bool).

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    Array, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array,
};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;

use crate::hash::hash_i32_slice;
use crate::ht::JoinHashTable;
use crate::row::{PhysType, RowLayout};

/// Column set ingested into rows. Index in this slice is the column index in
/// the row layout (the join key always sits at index 0).
pub const BENCH_COLUMNS: &[(&str, PhysType)] = &[
    ("k", PhysType::I32),
    ("c_int", PhysType::I32),
    ("c_bigint", PhysType::I64),
    ("c_dbl", PhysType::F64),
    ("c_flt", PhysType::F32),
    ("c_bool", PhysType::Bool),
];

pub fn bench_layout() -> RowLayout {
    RowLayout::new(BENCH_COLUMNS.iter().map(|(_, t)| *t).collect())
}

/// Key-only layout: just the `k` column. Matches DuckDB's projection-pushdown
/// when the join output is `count(*)` — no data columns are read.
pub fn keys_only_layout() -> RowLayout {
    RowLayout::new(vec![PhysType::I32])
}

/// Parallel all-columns ingest — one rayon task per row group. Reads the
/// full `BENCH_COLUMNS` set into pre-sized rows. Hashes computed from `k`
/// alongside the row write.
pub fn read_parquet_into_ht_parallel(
    path: &Path,
    ht: &mut JoinHashTable,
) -> parquet::errors::Result<usize> {
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use rayon::prelude::*;

    let metadata_file = File::open(path).map_err(|e| {
        parquet::errors::ParquetError::General(format!("open {:?}: {}", path, e))
    })?;
    let meta_reader = SerializedFileReader::new(metadata_file)?;
    let meta = meta_reader.metadata();
    let rg_counts: Vec<usize> = (0..meta.num_row_groups())
        .map(|i| meta.row_group(i).num_rows() as usize)
        .collect();
    let total: usize = rg_counts.iter().sum();
    let mut offsets = Vec::with_capacity(rg_counts.len());
    let mut acc = 0usize;
    for &c in &rg_counts {
        offsets.push(acc);
        acc += c;
    }

    ht.set_len(total);
    let layout = ht.layout().clone();
    let row_size = layout.row_size();
    let arena_addr = ht.arena_base_addr();
    let hashes_addr = ht.hashes_base_addr();

    let path_arc: std::sync::Arc<Path> = std::sync::Arc::from(path);

    (0..rg_counts.len()).into_par_iter().try_for_each(|rg_idx| -> parquet::errors::Result<()> {
        let file = File::open(&*path_arc).map_err(|e| {
            parquet::errors::ParquetError::General(format!("open rg{}: {}", rg_idx, e))
        })?;
        let builder = ParquetRecordBatchReaderBuilder::try_new_with_options(
            file,
            parquet::arrow::arrow_reader::ArrowReaderOptions::new(),
        )?;
        let builder = builder.with_row_groups(vec![rg_idx]);
        let schema_desc = builder.parquet_schema().clone();
        let mut col_indices = Vec::with_capacity(BENCH_COLUMNS.len());
        for (name, _) in BENCH_COLUMNS {
            let idx = schema_desc
                .columns()
                .iter()
                .position(|c| c.name() == *name)
                .expect("column missing");
            col_indices.push(idx);
        }
        let mask = ProjectionMask::leaves(&schema_desc, col_indices.iter().copied());
        let reader = builder.with_projection(mask).with_batch_size(8192).build()?;

        let mut row_offset = offsets[rg_idx];
        for batch in reader {
            let batch = batch?;
            let n = batch.num_rows();
            if n == 0 {
                continue;
            }
            // Build a slice [row_offset..row_offset+n] of the global arena.
            let slice_ptr = (arena_addr + row_offset * row_size) as *mut u8;
            let slice = unsafe { std::slice::from_raw_parts_mut(slice_ptr, n * row_size) };

            // Key column 0.
            let key_arr = downcast::<Int32Array>(batch.column(0), "k");
            write_i32_column(slice, row_size, layout.column_offset(0), key_arr);
            // Remaining data columns.
            for (col_idx, (name, ty)) in BENCH_COLUMNS.iter().enumerate().skip(1) {
                let off = layout.column_offset(col_idx);
                match ty {
                    PhysType::I32 => {
                        let arr = downcast::<Int32Array>(batch.column(col_idx), name);
                        write_i32_column(slice, row_size, off, arr);
                    }
                    PhysType::I64 => {
                        let arr = downcast::<Int64Array>(batch.column(col_idx), name);
                        write_i64_column(slice, row_size, off, arr);
                    }
                    PhysType::F32 => {
                        let arr = downcast::<Float32Array>(batch.column(col_idx), name);
                        write_f32_column(slice, row_size, off, arr);
                    }
                    PhysType::F64 => {
                        let arr = downcast::<Float64Array>(batch.column(col_idx), name);
                        write_f64_column(slice, row_size, off, arr);
                    }
                    PhysType::Bool => {
                        let arr = downcast::<BooleanArray>(batch.column(col_idx), name);
                        write_bool_column(slice, row_size, off, arr);
                    }
                }
            }

            // Hashes: compute from key column into global hash buffer.
            let vals = key_arr.values();
            let hashes_ptr = hashes_addr as *mut u64;
            for (i, &v) in vals.iter().enumerate() {
                unsafe {
                    *hashes_ptr.add(row_offset + i) = crate::hash::murmur_hash32(v as u32);
                }
            }

            row_offset += n;
        }
        Ok(())
    })?;

    Ok(total)
}

/// Parallel key-only ingest: one rayon task per row group, each writing into
/// a pre-sized region of the arena. Returns the total row count. Much faster
/// than the serial variant on files with ≥4 row groups.
pub fn read_parquet_keys_into_ht_parallel(
    path: &Path,
    ht: &mut JoinHashTable,
) -> parquet::errors::Result<usize> {
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use rayon::prelude::*;
    use std::sync::Arc;

    // Pass 1: metadata → per-row-group counts + start offsets.
    let metadata_file = File::open(path).map_err(|e| {
        parquet::errors::ParquetError::General(format!("open {:?}: {}", path, e))
    })?;
    let meta_reader = SerializedFileReader::new(metadata_file)?;
    let meta = meta_reader.metadata();
    let rg_counts: Vec<usize> = (0..meta.num_row_groups())
        .map(|i| meta.row_group(i).num_rows() as usize)
        .collect();
    let total: usize = rg_counts.iter().sum();

    let mut offsets = Vec::with_capacity(rg_counts.len());
    let mut acc = 0usize;
    for &c in &rg_counts {
        offsets.push(acc);
        acc += c;
    }

    // Pre-size arena + hash buffer. Single allocation, no later resize.
    ht.set_len(total);
    let layout = ht.layout().clone();
    let row_size = layout.row_size();
    let key_off = layout.column_offset(0);
    let arena_addr = ht.arena_base_addr();
    let hashes_addr = ht.hashes_base_addr();

    // Share metadata across threads so each reader doesn't re-read the footer.
    let meta_arc = Arc::new(meta.clone());
    let path_arc: Arc<Path> = Arc::from(path);

    (0..rg_counts.len()).into_par_iter().try_for_each(|rg_idx| -> parquet::errors::Result<()> {
        let file = File::open(&*path_arc).map_err(|e| {
            parquet::errors::ParquetError::General(format!("open rg{}: {}", rg_idx, e))
        })?;
        let builder = ParquetRecordBatchReaderBuilder::try_new_with_options(
            file,
            parquet::arrow::arrow_reader::ArrowReaderOptions::new(),
        )?;
        // Use the shared metadata to avoid re-reading the footer per task.
        let builder = builder.with_row_groups(vec![rg_idx]);
        let schema_desc = builder.parquet_schema().clone();
        let k_idx = schema_desc
            .columns()
            .iter()
            .position(|c| c.name() == "k")
            .expect("k column missing");
        let mask = ProjectionMask::leaves(&schema_desc, std::iter::once(k_idx));
        let reader = builder.with_projection(mask).with_batch_size(8192).build()?;

        let mut row_offset = offsets[rg_idx];
        for batch in reader {
            let batch = batch?;
            let arr = downcast::<Int32Array>(batch.column(0), "k");
            let vals = arr.values();
            for (i, &v) in vals.iter().enumerate() {
                let idx = row_offset + i;
                let row_ptr = (arena_addr + idx * row_size) as *mut u8;
                unsafe {
                    std::ptr::write_unaligned(row_ptr.add(key_off) as *mut i32, v);
                    let h = crate::hash::murmur_hash32(v as u32);
                    *((hashes_addr as *mut u64).add(idx)) = h;
                }
            }
            row_offset += vals.len();
        }
        let _ = meta_arc; // keep Arc alive (future use once we pre-parse).
        Ok(())
    })?;

    Ok(total)
}

/// Key-only ingest: reads only the `k` column, appends rows containing just k.
pub fn read_parquet_keys_into_ht(
    path: &Path,
    ht: &mut JoinHashTable,
) -> parquet::errors::Result<usize> {
    let file = File::open(path).map_err(|e| {
        parquet::errors::ParquetError::General(format!("open {:?}: {}", path, e))
    })?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema_desc = builder.parquet_schema().clone();
    let k_idx = schema_desc
        .columns()
        .iter()
        .position(|c| c.name() == "k")
        .expect("k column missing");
    let mask = ProjectionMask::leaves(&schema_desc, std::iter::once(k_idx));
    let reader = builder.with_projection(mask).with_batch_size(8192).build()?;

    let layout = ht.layout().clone();
    let key_off = layout.column_offset(0);
    let row_size = layout.row_size();

    let mut rows_total = 0usize;
    for batch in reader {
        let batch = batch?;
        let n = batch.num_rows();
        if n == 0 {
            continue;
        }
        let key_arr = downcast::<Int32Array>(batch.column(0), "k");
        let slice = ht.append_rows(n);
        write_i32_column(slice, row_size, key_off, key_arr);
        let mut hashes = vec![0u64; n];
        crate::hash::hash_i32_slice(key_arr.values(), &mut hashes);
        ht.record_hashes(&hashes);
        rows_total += n;
    }
    Ok(rows_total)
}

/// Stream `path` into `ht`, batch by batch, hashing the key column and
/// writing row bytes directly into the arena.
///
/// Returns the total number of rows read. `ht` must be freshly created
/// (caller reserves). Call `ht.finalize()` after all build-side files are
/// ingested.
pub fn read_parquet_into_ht(path: &Path, ht: &mut JoinHashTable) -> parquet::errors::Result<usize> {
    let file = File::open(path).map_err(|e| {
        parquet::errors::ParquetError::General(format!("open {:?}: {}", path, e))
    })?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema_desc = builder.parquet_schema().clone();

    // Project only the columns we care about (by name), in bench_columns order.
    let mut indices = Vec::with_capacity(BENCH_COLUMNS.len());
    for (name, _) in BENCH_COLUMNS {
        let idx = schema_desc
            .columns()
            .iter()
            .position(|c| c.name() == *name)
            .ok_or_else(|| {
                parquet::errors::ParquetError::General(format!(
                    "column '{}' not in parquet schema",
                    name
                ))
            })?;
        indices.push(idx);
    }
    let mask = ProjectionMask::leaves(&schema_desc, indices.iter().copied());

    let reader = builder
        .with_projection(mask)
        .with_batch_size(8192)
        .build()?;

    let mut rows_total = 0usize;
    for batch in reader {
        let batch = batch?;
        append_batch_into_ht(ht, &batch);
        rows_total += batch.num_rows();
    }
    Ok(rows_total)
}

/// Copy one `RecordBatch` into the row arena and record hashes. Assumes the
/// batch columns are in `BENCH_COLUMNS` order — which is guaranteed by the
/// projection applied upstream.
fn append_batch_into_ht(ht: &mut JoinHashTable, batch: &RecordBatch) {
    let n = batch.num_rows();
    if n == 0 {
        return;
    }
    let layout = ht.layout().clone();
    let key_arr = downcast::<Int32Array>(batch.column(0), "k");
    // Reserve and write every column into the arena slice in one shot.
    let row_size = layout.row_size();
    let slice = ht.append_rows(n);

    // Key at column 0 — i32.
    write_i32_column(slice, row_size, layout.column_offset(0), key_arr);

    // Remaining data columns, in BENCH_COLUMNS order.
    for (col_idx, (name, ty)) in BENCH_COLUMNS.iter().enumerate().skip(1) {
        let off = layout.column_offset(col_idx);
        match ty {
            PhysType::I32 => {
                let arr = downcast::<Int32Array>(batch.column(col_idx), name);
                write_i32_column(slice, row_size, off, arr);
            }
            PhysType::I64 => {
                let arr = downcast::<Int64Array>(batch.column(col_idx), name);
                write_i64_column(slice, row_size, off, arr);
            }
            PhysType::F32 => {
                let arr = downcast::<Float32Array>(batch.column(col_idx), name);
                write_f32_column(slice, row_size, off, arr);
            }
            PhysType::F64 => {
                let arr = downcast::<Float64Array>(batch.column(col_idx), name);
                write_f64_column(slice, row_size, off, arr);
            }
            PhysType::Bool => {
                let arr = downcast::<BooleanArray>(batch.column(col_idx), name);
                write_bool_column(slice, row_size, off, arr);
            }
        }
    }

    // Hash the key column and record into each row's header.
    let mut hashes = vec![0u64; n];
    hash_i32_slice(key_arr.values(), &mut hashes);
    ht.record_hashes(&hashes);
}

fn downcast<'a, T: Array + 'static>(
    col: &'a std::sync::Arc<dyn Array>,
    name: &str,
) -> &'a T {
    col.as_any()
        .downcast_ref::<T>()
        .unwrap_or_else(|| panic!("column {} has wrong type (got {:?})", name, col.data_type()))
}

#[inline]
fn write_i32_column(slice: &mut [u8], row_size: usize, col_off: usize, arr: &Int32Array) {
    let vals = arr.values();
    for (i, &v) in vals.iter().enumerate() {
        let off = i * row_size + col_off;
        slice[off..off + 4].copy_from_slice(&v.to_ne_bytes());
    }
}

#[inline]
fn write_i64_column(slice: &mut [u8], row_size: usize, col_off: usize, arr: &Int64Array) {
    let vals = arr.values();
    for (i, &v) in vals.iter().enumerate() {
        let off = i * row_size + col_off;
        slice[off..off + 8].copy_from_slice(&v.to_ne_bytes());
    }
}

#[inline]
fn write_f32_column(slice: &mut [u8], row_size: usize, col_off: usize, arr: &Float32Array) {
    let vals = arr.values();
    for (i, &v) in vals.iter().enumerate() {
        let off = i * row_size + col_off;
        slice[off..off + 4].copy_from_slice(&v.to_ne_bytes());
    }
}

#[inline]
fn write_f64_column(slice: &mut [u8], row_size: usize, col_off: usize, arr: &Float64Array) {
    let vals = arr.values();
    for (i, &v) in vals.iter().enumerate() {
        let off = i * row_size + col_off;
        slice[off..off + 8].copy_from_slice(&v.to_ne_bytes());
    }
}

#[inline]
fn write_bool_column(slice: &mut [u8], row_size: usize, col_off: usize, arr: &BooleanArray) {
    for i in 0..arr.len() {
        slice[i * row_size + col_off] = arr.value(i) as u8;
    }
}

/// Read a parquet file and produce two parallel vectors — key values + hashes —
/// for the probe side. No row arena: the probe side just streams through,
/// looking up in the build-side HT.
pub fn read_parquet_keys_for_probe(
    path: &Path,
) -> parquet::errors::Result<(Vec<i32>, Vec<u64>)> {
    let file = File::open(path).map_err(|e| {
        parquet::errors::ParquetError::General(format!("open {:?}: {}", path, e))
    })?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema_desc = builder.parquet_schema().clone();

    let k_idx = schema_desc
        .columns()
        .iter()
        .position(|c| c.name() == "k")
        .expect("k column missing");
    let mask = ProjectionMask::leaves(&schema_desc, std::iter::once(k_idx));

    let reader = builder.with_projection(mask).with_batch_size(8192).build()?;

    let mut keys: Vec<i32> = Vec::new();
    for batch in reader {
        let batch = batch?;
        let arr = downcast::<Int32Array>(batch.column(0), "k");
        keys.extend_from_slice(arr.values());
    }
    let mut hashes = vec![0u64; keys.len()];
    hash_i32_slice(&keys, &mut hashes);
    Ok((keys, hashes))
}

// ============================================================================
// Lance-style ingest: slim arena (keys only) + retained Arrow RecordBatches.
// ============================================================================
//
// Inspired by lance's `HashJoiner` (rust/lance/src/dataset/hash_joiner.rs):
// the join's hash table only needs the key for probe/compare; every data
// column stays in its original Arrow buffer, held alive via Arc<RecordBatch>.
// At output time we look up `(batch_idx, row_in_batch)` from the arena row's
// index and read the data directly from the Arrow array.
//
// Benefits vs the self-contained row arena:
//  * Build-side ingest is essentially "read parquet + compute key hash" —
//    no per-column copy into the arena for data columns.
//  * Row size drops from 40 B (6-col bench) to 20 B — more rows per L2 line
//    means faster linear probing.
//  * Strings come for free: the source Arrow StringArray already holds them.
//
// Cost: the sink has to do `arena_idx → (batch_idx, row_in_batch)` for each
// read, which is a binary search over the boundaries vec. O(log n_batches) —
// cheap. Arrow's values arrays give O(1) access after that.

pub struct TableData {
    pub ht: JoinHashTable,
    pub batches: Vec<Arc<RecordBatch>>,
    /// Cumulative row counts per batch. `boundaries[i]` = first arena-row
    /// index in batch `i`. `boundaries[batches.len()]` = total rows.
    pub boundaries: Vec<u32>,
}

impl TableData {
    #[inline]
    pub fn n_rows(&self) -> usize {
        *self.boundaries.last().unwrap() as usize
    }

    /// O(log n_batches) mapping from arena row index to `(batch, row_in_batch)`.
    #[inline]
    pub fn locate(&self, arena_idx: usize) -> (usize, usize) {
        // `partition_point` finds the first boundary strictly > arena_idx.
        // The batch index is `p - 1`.
        let probe = arena_idx as u32;
        let p = self.boundaries.partition_point(|&b| b <= probe);
        let batch_idx = p - 1;
        let row_in_batch = arena_idx - self.boundaries[batch_idx] as usize;
        (batch_idx, row_in_batch)
    }
}

/// Parallel parquet ingest with Arrow retention. Reads each row group on a
/// rayon task, returning the RecordBatches intact. The HT arena is slim
/// (key + header only), populated per batch in parallel.
pub fn read_parquet_lance_parallel(path: &Path) -> parquet::errors::Result<TableData> {
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use rayon::prelude::*;

    // --- Stage 1: metadata ---
    let meta_file = File::open(path).map_err(|e| {
        parquet::errors::ParquetError::General(format!("open {:?}: {}", path, e))
    })?;
    let meta_reader = SerializedFileReader::new(meta_file)?;
    let meta = meta_reader.metadata();
    let num_row_groups = meta.num_row_groups();
    let path_arc: Arc<Path> = Arc::from(path);

    // --- Stage 2: parallel per-row-group read, full schema retained ---
    // Each row group becomes one or more RecordBatches (capped by batch_size).
    // Setting batch_size large enough so each row group typically produces a
    // single batch keeps `boundaries` short and `locate` binary searches fast.
    let per_rg_batches: Vec<Vec<Arc<RecordBatch>>> = (0..num_row_groups)
        .into_par_iter()
        .map(|rg_idx| -> parquet::errors::Result<Vec<Arc<RecordBatch>>> {
            let file = File::open(&*path_arc).map_err(|e| {
                parquet::errors::ParquetError::General(format!("open rg{}: {}", rg_idx, e))
            })?;
            let builder = ParquetRecordBatchReaderBuilder::try_new_with_options(
                file,
                parquet::arrow::arrow_reader::ArrowReaderOptions::new(),
            )?
            .with_row_groups(vec![rg_idx])
            .with_batch_size(1_000_000); // big: one batch per row group normally
            let reader = builder.build()?;
            let mut out = Vec::new();
            for b in reader {
                out.push(Arc::new(b?));
            }
            Ok(out)
        })
        .collect::<parquet::errors::Result<Vec<_>>>()?;

    // --- Stage 3: flatten + compute boundaries ---
    let mut batches: Vec<Arc<RecordBatch>> = Vec::new();
    let mut boundaries: Vec<u32> = vec![0];
    for rg_batches in per_rg_batches {
        for b in rg_batches {
            let nr = b.num_rows() as u32;
            let next = *boundaries.last().unwrap() + nr;
            boundaries.push(next);
            batches.push(b);
        }
    }
    let total_rows = *boundaries.last().unwrap() as usize;

    // --- Stage 4: populate slim HT arena in parallel (key + cached hash) ---
    let mut ht = JoinHashTable::new(keys_only_layout());
    ht.set_len(total_rows);
    let layout = ht.layout().clone();
    let key_off = layout.column_offset(0);
    let row_size = layout.row_size();
    let arena_addr = ht.arena_base_addr();
    let hashes_addr = ht.hashes_base_addr();

    batches
        .par_iter()
        .enumerate()
        .for_each(|(batch_idx, batch)| {
            let start = boundaries[batch_idx] as usize;
            let k_col = batch
                .column_by_name("k")
                .expect("missing k column");
            let arr = k_col
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("k is not Int32");
            let vals = arr.values();
            let hashes_ptr = hashes_addr as *mut u64;
            for (i, &v) in vals.iter().enumerate() {
                let idx = start + i;
                unsafe {
                    let row_ptr = (arena_addr + idx * row_size) as *mut u8;
                    std::ptr::write_unaligned(row_ptr.add(key_off) as *mut i32, v);
                    *hashes_ptr.add(idx) = crate::hash::murmur_hash32(v as u32);
                }
            }
        });

    Ok(TableData {
        ht,
        batches,
        boundaries,
    })
}
