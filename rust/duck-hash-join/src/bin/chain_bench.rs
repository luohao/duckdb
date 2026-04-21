// k-way chain bench: left-deep INNER JOIN chain t0 ⋈ t1 ⋈ ... ⋈ t_{k-1}.
//
// Mirrors kway_bench.cpp's INNER variant but runs against the Rust port. Each
// binary join produces an intermediate keys-only hash table that feeds the
// next join. Intermediate rows carry just the join key `k` (not the 10 data
// columns) so this measures the core chained hash-join algorithm without
// apples-to-apples column materialization — use `--full-cols` to include
// reading build-side data columns (matching what DuckDB's blackhole sink
// forces the parquet reader to decode).
//
// Workload from kway_bench's 5-way / 50%-overlap config:
//   total keys ≈ 3M distributed: 500k shared across all tables,
//   500k unique per table. At o=0.5 and k=5, INNER 5-way output = 500k.

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use duck_hash_join::ht::JoinHashTable;
use duck_hash_join::io::{
    bench_layout, keys_only_layout, read_parquet_into_ht_parallel, read_parquet_lance_parallel,
    read_parquet_keys_into_ht_parallel, TableData,
};
use duck_hash_join::row::PhysType;

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn usage_and_exit() -> ! {
    eprintln!(
        "usage: chain_bench <config_dir> [k] [repeats] [--threads=N] [--full-cols] [--scan-only]\n\
         --scan-only: only time reading joined.parquet (denormalized baseline)\n\
         --full-cols: read all 6 bench columns into arena (matches DuckDB blackhole work)"
    );
    std::process::exit(1);
}

fn main() {
    let mut args = env::args().skip(1);
    let dir = args.next().unwrap_or_else(|| usage_and_exit());
    let k: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let repeats: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let threads: Option<usize> = env::args()
        .find_map(|a| a.strip_prefix("--threads=").map(|s| s.parse().unwrap()));
    let full_cols = env::args().any(|a| a == "--full-cols");
    let scan_only = env::args().any(|a| a == "--scan-only");
    let foj = env::args().any(|a| a == "--foj");
    let sink_checksum = env::args().any(|a| a == "--sink-checksum");
    let pipelined = env::args().any(|a| a == "--pipelined");
    let lance = env::args().any(|a| a == "--lance");

    let dir = PathBuf::from(dir);
    if let Some(t) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .expect("build rayon pool");
    }

    println!(
        "[chain_bench] dir={:?} k={} repeats={} threads={} full_cols={} foj={} sink_checksum={} scan_only={}",
        dir,
        k,
        repeats,
        rayon::current_num_threads(),
        full_cols,
        foj,
        sink_checksum,
        scan_only,
    );

    if scan_only {
        run_scan(&dir, repeats);
        return;
    }

    if pipelined {
        run_chain_pipelined(&dir, k, repeats);
        return;
    }

    if lance {
        run_chain_lance(&dir, k, repeats);
        return;
    }

    run_chain(&dir, k, repeats, full_cols, foj, sink_checksum);
}

fn run_scan(dir: &std::path::Path, repeats: usize) {
    let path = dir.join("joined.parquet");
    if !path.exists() {
        panic!("missing {:?}", path);
    }
    // "Scan" baseline: read joined.parquet's 6 bench columns into a throwaway
    // hash table arena (no probe, no finalize). Mirrors the I/O + column
    // decode + copy cost of DuckDB's scan_joined / blackhole sink.
    let mut samples = Vec::new();
    let mut rows_out = 0;
    for rep in 0..=repeats {
        let mut ht = JoinHashTable::new(bench_layout());
        let t0 = Instant::now();
        rows_out = read_parquet_into_ht_parallel(&path, &mut ht).unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if rep == 0 {
            println!("[warmup] scan rows={} {:.1}ms", rows_out, ms);
            continue;
        }
        println!("[rep {}] scan rows={} {:.1}ms", rep, rows_out, ms);
        samples.push(ms);
    }
    println!();
    println!(
        "[scan_denorm median] {:.1} ms (rows={})",
        median(&mut samples),
        rows_out
    );
}

fn run_chain(
    dir: &std::path::Path,
    k: usize,
    repeats: usize,
    full_cols: bool,
    foj: bool,
    sink_checksum: bool,
) {
    let paths: Vec<PathBuf> = (0..k).map(|i| dir.join(format!("t{}.parquet", i))).collect();
    for p in &paths {
        if !p.exists() {
            panic!("missing {:?}", p);
        }
    }

    let mut read_ms = Vec::new();
    let mut build_ms = Vec::new();
    let mut chain_ms = Vec::new();
    let mut total_ms = Vec::new();
    let mut last_rows_out = 0usize;
    let mut sink_ms_samples: Vec<f64> = Vec::new();
    let mut checksum_accum: u64 = 0;

    for rep in 0..=repeats {
        let t_start = Instant::now();

        // Phase 1: read all k parquets into JoinHashTables.
        // `tables[0]` starts the chain; `tables[1..]` are the probe sides.
        let mut tables: Vec<JoinHashTable> = (0..k)
            .map(|_| {
                JoinHashTable::new(if full_cols {
                    bench_layout()
                } else {
                    keys_only_layout()
                })
            })
            .collect();
        for (i, path) in paths.iter().enumerate() {
            if full_cols {
                read_parquet_into_ht_parallel(path, &mut tables[i]).unwrap();
            } else {
                read_parquet_keys_into_ht_parallel(path, &mut tables[i]).unwrap();
            }
        }
        let t_read = Instant::now();

        // Phase 2: build HT for tables[0] (first build side in the chain).
        tables[0].finalize_parallel();
        let t_build0 = Instant::now();

        // Phase 3: chained probes. After probing table[i], we have a
        // `matched_keys` vector — keys that passed join step i. Build a new
        // HT over those keys to feed the next probe.
        //
        // For INNER chain, step i's output keys = keys that are in every
        // table 0..=i.
        let mut current_keys: Vec<i32> = Vec::new();
        let mut current_hashes: Vec<u64> = Vec::new();

        // First probe: tables[1] against tables[0].
        //
        // For INNER chain: accumulate keys that matched.
        // For FOJ chain: accumulator = union of all keys seen so far. At step 1
        //   this is union(t0, t1). We start with t0's keys (already in tables[0])
        //   and add t1's keys that AREN'T already in tables[0].
        {
            let build_ht = &tables[0];
            let probe = &tables[1];
            let (probe_keys, probe_hashes) = extract_keys(probe);
            if foj {
                // Rust accumulator = t0's keys + t1's unmatched keys.
                let (un_keys, un_hashes) =
                    build_ht.probe_i32_collect_unmatched(&probe_hashes, &probe_keys, 16384);
                current_keys = extract_keys(build_ht).0;
                current_hashes = extract_keys(build_ht).1;
                current_keys.extend(un_keys);
                current_hashes.extend(un_hashes);
            } else {
                build_ht
                    .probe_i32_inner_parallel(
                        &probe_hashes,
                        &probe_keys,
                        16384,
                        || (Vec::<i32>::with_capacity(16384), Vec::<u64>::with_capacity(16384)),
                        |s, _row, idx| {
                            let k = probe_keys[idx];
                            let h = probe_hashes[idx];
                            s.0.push(k);
                            s.1.push(h);
                        },
                        |mut a, b| {
                            a.0.extend(b.0);
                            a.1.extend(b.1);
                            a
                        },
                    )
                    .pipe(|(keys, hashes)| {
                        current_keys = keys;
                        current_hashes = hashes;
                    });
            }
        }

        // Subsequent probes: build HT over current_keys, probe tables[i].
        for i in 2..k {
            let mut intermediate = JoinHashTable::new(keys_only_layout());
            intermediate.set_len(current_keys.len());
            let layout = intermediate.layout().clone();
            let key_off = layout.column_offset(0);
            let row_size = layout.row_size();
            let arena_addr = intermediate.arena_base_addr();
            let hashes_addr = intermediate.hashes_base_addr();
            // Fill intermediate: write keys + hashes in parallel.
            use rayon::prelude::*;
            let ck = &current_keys;
            let ch = &current_hashes;
            (0..ck.len()).into_par_iter().for_each(|idx| {
                let row_ptr = (arena_addr + idx * row_size) as *mut u8;
                unsafe {
                    std::ptr::write_unaligned(row_ptr.add(key_off) as *mut i32, ck[idx]);
                    *((hashes_addr as *mut u64).add(idx)) = ch[idx];
                }
            });
            intermediate.finalize_parallel();

            let probe = &tables[i];
            let (probe_keys, probe_hashes) = extract_keys(probe);
            if foj {
                // FOJ: union grows by probe-side unmatched keys. Intermediate
                // is already the accumulator (set of keys seen so far).
                let (un_keys, un_hashes) =
                    intermediate.probe_i32_collect_unmatched(&probe_hashes, &probe_keys, 16384);
                current_keys.extend(un_keys);
                current_hashes.extend(un_hashes);
            } else {
                intermediate
                    .probe_i32_inner_parallel(
                        &probe_hashes,
                        &probe_keys,
                        16384,
                        || (Vec::<i32>::with_capacity(16384), Vec::<u64>::with_capacity(16384)),
                        |s, _row, idx| {
                            s.0.push(probe_keys[idx]);
                            s.1.push(probe_hashes[idx]);
                        },
                        |mut a, b| {
                            a.0.extend(b.0);
                            a.1.extend(b.1);
                            a
                        },
                    )
                    .pipe(|(keys, hashes)| {
                        current_keys = keys;
                        current_hashes = hashes;
                    });
            }
        }

        let t_chain = Instant::now();

        let rows_out = current_keys.len();
        last_rows_out = rows_out;

        // Sink-checksum phase: simulate blackhole output materialization.
        // For each final output key, look up in each of the 5 tables' HTs.
        // If matched, read all 5 numeric data columns (c_int, c_bigint, c_dbl,
        // c_flt, c_bool) through the pointer and fold into a checksum. If not
        // matched, fold in zeros (NULL-padding). This forces every output
        // column byte to actually be read, mirroring DuckDB's blackhole sink.
        //
        // Note: this does *more* work than DuckDB's pipelined approach
        // (DuckDB carries column pointers through the chain; we re-probe).
        // If Rust still wins, parity is genuine.
        let mut sink_ms = 0.0f64;
        if sink_checksum && full_cols {
            // Finalize every HT (tables[1..] weren't finalized during chain).
            for i in 1..k {
                if !tables[i].is_finalized() {
                    tables[i].finalize_parallel();
                }
            }
            let t_sink_start = Instant::now();
            let checksum = materialize_sink(&tables, &current_keys, &current_hashes);
            let checksum = std::hint::black_box(checksum);
            sink_ms = t_sink_start.elapsed().as_secs_f64() * 1000.0;
            eprintln!("[rep {}] sink_checksum=0x{:016x}", rep, checksum);
            checksum_accum = checksum_accum.wrapping_add(checksum);
        }

        let read = t_read.duration_since(t_start).as_secs_f64() * 1000.0;
        let build = t_build0.duration_since(t_read).as_secs_f64() * 1000.0;
        let chain = t_chain.duration_since(t_build0).as_secs_f64() * 1000.0;
        let total = t_chain.duration_since(t_start).as_secs_f64() * 1000.0 + sink_ms;

        if rep == 0 {
            println!(
                "[warmup] read={:.1} build0={:.1} chain={:.1} sink={:.1} total={:.1} rows_out={}",
                read, build, chain, sink_ms, total, rows_out
            );
            continue;
        }
        read_ms.push(read);
        build_ms.push(build);
        chain_ms.push(chain);
        total_ms.push(total);
        if sink_checksum {
            sink_ms_samples.push(sink_ms);
        }
        println!(
            "[rep {}]  read={:.1} build0={:.1} chain={:.1} sink={:.1} total={:.1} rows_out={}",
            rep, read, build, chain, sink_ms, total, rows_out
        );
    }

    println!();
    println!("[median over {} reps]", repeats);
    println!("  parquet read (all k)  : {:>7.1} ms", median(&mut read_ms));
    println!("  first HT finalize     : {:>7.1} ms", median(&mut build_ms));
    println!("  chained probes        : {:>7.1} ms", median(&mut chain_ms));
    if sink_checksum {
        println!("  output materialize    : {:>7.1} ms", median(&mut sink_ms_samples));
    }
    println!("  total                 : {:>7.1} ms", median(&mut total_ms));
    println!("  rows_out              : {}", last_rows_out);
    if sink_checksum {
        // Print checksum so the compiler can't elide the reads.
        println!("  checksum              : 0x{:016x}", checksum_accum);
    }
}

// Iterate the final output keys, probe each of the 5 tables' HTs for each
// key, and fold every numeric data column value into a running XOR. Mirrors
// what a blackhole sink does — force every output column byte to be read.
// Runs in parallel across chunks of the output key range.
fn materialize_sink(
    tables: &[JoinHashTable],
    keys: &[i32],
    hashes: &[u64],
) -> u64 {
    use rayon::prelude::*;
    let k = tables.len();
    assert_eq!(keys.len(), hashes.len());

    // Per-table probe context — offsets for the 5 data columns within a row.
    let col_offsets: Vec<Vec<usize>> = tables
        .iter()
        .map(|t| {
            let layout = t.layout();
            // Data columns are at indices 1..=5 (after the key at index 0).
            (1..layout.num_columns())
                .map(|c| layout.column_offset(c))
                .collect()
        })
        .collect();

    let chunk_size = 16_384;
    let n = keys.len();
    let chunks: Vec<(usize, usize)> = (0..n)
        .step_by(chunk_size)
        .map(|s| (s, (s + chunk_size).min(n)))
        .collect();

    chunks
        .into_par_iter()
        .map(|(start, end)| {
            let mut checksum: u64 = 0;
            for i in start..end {
                let k_val = keys[i];
                let h = hashes[i];
                for (t_idx, table) in tables.iter().enumerate() {
                    // Probe table[t_idx] for key; if found, read data cols.
                    // Use a tight inline probe rather than the callback form
                    // so we can fold data directly into the checksum.
                    let row_ptr_opt = probe_lookup(table, h, k_val);
                    if let Some(row_ptr) = row_ptr_opt {
                        for &off in &col_offsets[t_idx] {
                            // Read 8 bytes of whatever column this is;
                            // c_int (4B) and c_bool (1B) will over-read
                            // harmlessly into the row's next column or
                            // padding — still forces the read.
                            let v = unsafe {
                                std::ptr::read_unaligned(row_ptr.add(off) as *const u64)
                            };
                            checksum = checksum.wrapping_add(v);
                        }
                    } else {
                        // NULL-padded side: fold a sentinel so the branch
                        // is not fully optimized away.
                        checksum = checksum.wrapping_add(0xDEAD_BEEF_CAFE_BABE);
                    }
                }
            }
            checksum
        })
        .reduce(|| 0u64, |a, b| a.wrapping_add(b))
}

// Direct probe returning the first matching row pointer (or None).
#[inline]
fn probe_lookup(table: &JoinHashTable, hash: u64, key: i32) -> Option<*const u8> {
    let directory_ptr = table.directory_ptr();
    let mask = table.capacity_mask();
    let key_off = table.key_offset();
    let salt = duck_hash_join::ht_entry::HtEntry::extract_salt(hash);
    let mut bucket = (hash as usize) & mask;
    loop {
        let entry = unsafe { *directory_ptr.add(bucket) };
        if !entry.is_occupied() {
            return None;
        }
        if entry.salt() == salt {
            let mut p = entry.pointer();
            while !p.is_null() {
                let row_key =
                    unsafe { std::ptr::read_unaligned(p.add(key_off) as *const i32) };
                if row_key == key {
                    return Some(p);
                }
                let next = unsafe { duck_hash_join::row::row_next_ptr(p) };
                p = (next & duck_hash_join::ht_entry::POINTER_MASK) as usize as *const u8;
            }
            return None;
        }
        bucket = (bucket + 1) & mask;
    }
}

// Helper: extract (keys, hashes) from a finalized or unfinalized HT. The
// layout's first column is the i32 key; hashes are the per-row cached hash
// in the arena header. We pull them directly from internal state.
fn extract_keys(ht: &JoinHashTable) -> (Vec<i32>, Vec<u64>) {
    let layout = ht.layout();
    assert!(matches!(layout.column_type(0), PhysType::I32));
    let key_off = layout.column_offset(0);
    let row_size = layout.row_size();
    let n = ht.n_rows();
    let mut keys = Vec::with_capacity(n);
    let mut hashes = Vec::with_capacity(n);

    // Hash values are cached by the ingest functions into `ht.hashes` but
    // that field is private. We have `hashes_base_addr()` which returns the
    // pointer, but only mut. For read-only access we need a helper.
    let arena_base = ht.arena_base_addr();
    let hashes_addr = ht.hashes_base_addr_const();
    unsafe {
        let hashes_ptr = hashes_addr as *const u64;
        for i in 0..n {
            let row_ptr = (arena_base + i * row_size) as *const u8;
            let k = std::ptr::read_unaligned(row_ptr.add(key_off) as *const i32);
            let h = *hashes_ptr.add(i);
            keys.push(k);
            hashes.push(h);
        }
    }
    (keys, hashes)
}

// Small .pipe() helper on tuple results so we can chain side effects.
trait Pipe: Sized {
    fn pipe<F: FnOnce(Self)>(self, f: F) {
        f(self)
    }
}
impl<T> Pipe for T {}

// ============================================================================
// Pipelined FOJ chain — DuckDB's approach.
// ============================================================================
//
// Each output row carries one pointer slot per source table. After step i the
// slots for tables 0..=i are meaningful (or zero for the unmatched side).
// The final sink then just dereferences each non-null ptr — no re-probing,
// no keyed lookups at the end.
//
// Step i (1..k):
//   1. Build an HT from the current `chain` vec, keyed on the coalesced `k`.
//      Row index in the HT matches the chain vec index (since we ingest in
//      order), so a probe match recovers `chain_idx` via pointer arithmetic.
//   2. Probe tables[i]'s rows against the chain HT.
//      Match → update `chain[chain_idx].ptrs[i]` with the probe-side row ptr.
//      Miss  → append a new `ChainRow` (t_i-only; slots 0..=i-1 and i+1..
//              stay zero).
//   3. Drop the chain HT before the next step.

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct ChainRow {
    key: i32,
    _pad: u32,
    hash: u64,
    // Row pointer into tables[i]'s arena, or 0 for "no match on this side".
    ptrs: [u64; 5],
}

fn run_chain_pipelined(dir: &std::path::Path, k: usize, repeats: usize) {
    assert!(k <= 5, "pipelined chain hard-codes 5 slots per ChainRow");
    let paths: Vec<PathBuf> = (0..k).map(|i| dir.join(format!("t{}.parquet", i))).collect();
    for p in &paths {
        if !p.exists() {
            panic!("missing {:?}", p);
        }
    }

    let mut read_ms = Vec::new();
    let mut chain_ms = Vec::new();
    let mut sink_ms = Vec::new();
    let mut total_ms = Vec::new();
    let mut last_rows_out = 0usize;
    let mut checksum_accum: u64 = 0;

    for rep in 0..=repeats {
        let t_start = Instant::now();

        // Phase 1: read all k parquets into JoinHashTables (all 6 bench cols).
        let mut tables: Vec<JoinHashTable> = (0..k)
            .map(|_| JoinHashTable::new(bench_layout()))
            .collect();
        for (i, path) in paths.iter().enumerate() {
            read_parquet_into_ht_parallel(path, &mut tables[i]).unwrap();
        }
        let t_read = Instant::now();

        // Phase 2: initial chain from tables[0] — one ChainRow per t0 row.
        let t0 = &tables[0];
        let n0 = t0.n_rows();
        let t0_arena = t0.arena_base_addr();
        let t0_row_size = t0.layout().row_size();
        let t0_key_off = t0.key_offset();
        // Capture pointer as usize — raw pointers aren't Send/Sync.
        let t0_hashes_addr = t0.hashes_base_addr_const() as usize;

        let mut chain: Vec<ChainRow> = Vec::with_capacity(n0 * 4);
        chain.resize(n0, ChainRow::default());
        let chain_addr = chain.as_mut_ptr() as usize;
        (0..n0).into_par_iter_fill(move |i| unsafe {
            let row_ptr_i = (t0_arena + i * t0_row_size) as *const u8;
            let k = std::ptr::read_unaligned(row_ptr_i.add(t0_key_off) as *const i32);
            let h = *((t0_hashes_addr as *const u64).add(i));
            let dst = (chain_addr as *mut ChainRow).add(i);
            std::ptr::write(
                dst,
                ChainRow {
                    key: k,
                    _pad: 0,
                    hash: h,
                    ptrs: [row_ptr_i as u64, 0, 0, 0, 0],
                },
            );
        });

        // Phase 3: chain t1..t{k-1} onto the accumulator.
        for i in 1..k {
            chain_step(&mut chain, &tables[i], i);
        }

        let t_chain = Instant::now();
        let rows_out = chain.len();
        last_rows_out = rows_out;

        // Phase 4: sink — dereference every non-null ptr for all chain rows,
        // read 5 data cols per table. No more HT probes.
        let checksum = pipelined_sink(&chain, &tables);
        let checksum = std::hint::black_box(checksum);
        checksum_accum = checksum_accum.wrapping_add(checksum);
        let t_sink = Instant::now();

        let read = t_read.duration_since(t_start).as_secs_f64() * 1000.0;
        let chain_time = t_chain.duration_since(t_read).as_secs_f64() * 1000.0;
        let sink = t_sink.duration_since(t_chain).as_secs_f64() * 1000.0;
        let total = t_sink.duration_since(t_start).as_secs_f64() * 1000.0;

        if rep == 0 {
            println!(
                "[warmup] read={:.1} chain={:.1} sink={:.1} total={:.1} rows_out={}",
                read, chain_time, sink, total, rows_out
            );
            continue;
        }
        read_ms.push(read);
        chain_ms.push(chain_time);
        sink_ms.push(sink);
        total_ms.push(total);
        println!(
            "[rep {}]  read={:.1} chain={:.1} sink={:.1} total={:.1} rows_out={}",
            rep, read, chain_time, sink, total, rows_out
        );
    }

    println!();
    println!("[median over {} reps — pipelined FOJ chain]", repeats);
    println!("  parquet read (all k)   : {:>7.1} ms", median(&mut read_ms));
    println!("  chained probes + build : {:>7.1} ms", median(&mut chain_ms));
    println!("  sink (ptr-deref only)  : {:>7.1} ms", median(&mut sink_ms));
    println!("  total                  : {:>7.1} ms", median(&mut total_ms));
    println!("  rows_out               : {}", last_rows_out);
    println!("  checksum               : 0x{:016x}", checksum_accum);
}

// Build an HT over the chain's current keys, probe tables[i], then either
// update existing chain rows (match) or append new chain rows (miss).
fn chain_step(chain: &mut Vec<ChainRow>, table_i: &JoinHashTable, i: usize) {
    use duck_hash_join::ht_entry::{HtEntry, POINTER_MASK};
    use duck_hash_join::row::row_next_ptr;
    use rayon::prelude::*;

    let chain_len = chain.len();

    // 1. Build a fresh HT over `chain` keyed on `.key`. Row index == chain idx.
    let mut chain_ht = JoinHashTable::new(keys_only_layout());
    chain_ht.set_len(chain_len);
    let ht_layout = chain_ht.layout().clone();
    let ht_key_off = ht_layout.column_offset(0);
    let ht_row_size = ht_layout.row_size();
    let ht_arena = chain_ht.arena_base_addr();
    let ht_hashes = chain_ht.hashes_base_addr();

    let chain_addr = chain.as_ptr() as usize;
    (0..chain_len).into_par_iter().for_each(|idx| {
        let cr_ptr = (chain_addr as *const ChainRow).wrapping_add(idx);
        let cr_key = unsafe { (*cr_ptr).key };
        let cr_hash = unsafe { (*cr_ptr).hash };
        let row_ptr = (ht_arena + idx * ht_row_size) as *mut u8;
        unsafe {
            std::ptr::write_unaligned(row_ptr.add(ht_key_off) as *mut i32, cr_key);
            *((ht_hashes as *mut u64).add(idx)) = cr_hash;
        }
    });
    chain_ht.finalize_parallel();

    // 2. Probe tables[i].
    let t_i_arena = table_i.arena_base_addr();
    let t_i_row_size = table_i.layout().row_size();
    let t_i_key_off = table_i.key_offset();
    let t_i_hashes = table_i.hashes_base_addr_const() as usize;
    let n_i = table_i.n_rows();

    let ht_directory = chain_ht.directory_ptr() as usize;
    let ht_mask = chain_ht.capacity_mask();
    let ht_arena_base = ht_arena;

    let chunk_size = 16_384;
    let chunks: Vec<(usize, usize)> = (0..n_i)
        .step_by(chunk_size)
        .map(|s| (s, (s + chunk_size).min(n_i)))
        .collect();

    let results: Vec<(Vec<(u32, u64)>, Vec<(i32, u64, u64)>)> = chunks
        .into_par_iter()
        .map(|(start, end)| {
            let t_i_hashes_ptr = t_i_hashes as *const u64;
            let directory_ptr = ht_directory as *const HtEntry;
            let mut matches: Vec<(u32, u64)> = Vec::with_capacity((end - start) / 2);
            let mut misses: Vec<(i32, u64, u64)> = Vec::with_capacity((end - start) / 4);

            for probe_idx in start..end {
                let t_i_row_ptr = (t_i_arena + probe_idx * t_i_row_size) as *const u8;
                let k = unsafe {
                    std::ptr::read_unaligned(t_i_row_ptr.add(t_i_key_off) as *const i32)
                };
                let h = unsafe { *t_i_hashes_ptr.add(probe_idx) };

                let salt = HtEntry::extract_salt(h);
                let mut bucket = (h as usize) & ht_mask;
                let mut found_chain_idx: Option<u32> = None;
                loop {
                    let entry = unsafe { *directory_ptr.add(bucket) };
                    if !entry.is_occupied() {
                        break;
                    }
                    if entry.salt() == salt {
                        let mut p = entry.pointer();
                        while !p.is_null() {
                            let chain_key = unsafe {
                                std::ptr::read_unaligned(p.add(ht_key_off) as *const i32)
                            };
                            if chain_key == k {
                                let chain_idx =
                                    (p as usize - ht_arena_base) / ht_row_size;
                                found_chain_idx = Some(chain_idx as u32);
                                break;
                            }
                            let next = unsafe { row_next_ptr(p) };
                            p = (next & POINTER_MASK) as usize as *const u8;
                        }
                        break;
                    }
                    bucket = (bucket + 1) & ht_mask;
                }

                match found_chain_idx {
                    Some(ci) => matches.push((ci, t_i_row_ptr as u64)),
                    None => misses.push((k, h, t_i_row_ptr as u64)),
                }
            }
            (matches, misses)
        })
        .collect();

    // 3a. Apply matches. Disjoint chain indices (chain has unique keys, t_i
    // has unique keys → 1:1 at most), so parallel writes to different slots
    // are safe.
    let chain_mut_addr = chain.as_mut_ptr() as usize;
    results
        .par_iter()
        .for_each(|(matches, _)| {
            let chain_mut_ptr = chain_mut_addr as *mut ChainRow;
            for &(chain_idx, ptr) in matches {
                unsafe {
                    (*chain_mut_ptr.add(chain_idx as usize)).ptrs[i] = ptr;
                }
            }
        });

    // 3b. Append misses — sequential (can't parallel-push to a Vec).
    for (_, misses) in results {
        for (k, h, ptr) in misses {
            let mut ptrs = [0u64; 5];
            ptrs[i] = ptr;
            chain.push(ChainRow {
                key: k,
                _pad: 0,
                hash: h,
                ptrs,
            });
        }
    }
}

fn pipelined_sink(chain: &[ChainRow], tables: &[JoinHashTable]) -> u64 {
    use rayon::prelude::*;

    // Pre-compute data-column offsets for each table — the 5 cols after `k`.
    let col_offsets: Vec<Vec<usize>> = tables
        .iter()
        .map(|t| {
            (1..t.layout().num_columns())
                .map(|c| t.layout().column_offset(c))
                .collect()
        })
        .collect();

    chain
        .par_chunks(16_384)
        .map(|chunk| {
            let mut sum: u64 = 0;
            for cr in chunk {
                for (i, &ptr) in cr.ptrs.iter().enumerate() {
                    if ptr != 0 {
                        let row_ptr = ptr as *const u8;
                        for &off in &col_offsets[i] {
                            let v = unsafe {
                                std::ptr::read_unaligned(row_ptr.add(off) as *const u64)
                            };
                            sum = sum.wrapping_add(v);
                        }
                    } else {
                        sum = sum.wrapping_add(0xDEAD_BEEF_CAFE_BABE);
                    }
                }
            }
            sum
        })
        .reduce(|| 0u64, |a, b| a.wrapping_add(b))
}

// Rayon helper: run a closure for each index in parallel (indexed fill).
trait ParFill: Sized {
    fn into_par_iter_fill<F>(self, f: F)
    where
        F: Fn(usize) + Send + Sync;
}

impl ParFill for std::ops::Range<usize> {
    fn into_par_iter_fill<F>(self, f: F)
    where
        F: Fn(usize) + Send + Sync,
    {
        use rayon::prelude::*;
        self.into_par_iter().for_each(f);
    }
}

// ============================================================================
// Lance-style FOJ chain: slim arena + Arrow RecordBatch retention + sink via
// direct Arrow buffer access.
// ============================================================================

fn run_chain_lance(dir: &std::path::Path, k: usize, repeats: usize) {
    assert!(k <= 5, "ChainRow hardcodes 5 slots");
    let paths: Vec<PathBuf> = (0..k).map(|i| dir.join(format!("t{}.parquet", i))).collect();
    for p in &paths {
        if !p.exists() {
            panic!("missing {:?}", p);
        }
    }

    let mut read_ms = Vec::new();
    let mut chain_ms = Vec::new();
    let mut sink_ms = Vec::new();
    let mut total_ms = Vec::new();
    let mut last_rows_out = 0usize;
    let mut checksum_accum: u64 = 0;

    for rep in 0..=repeats {
        let t_start = Instant::now();

        // Phase 1: Lance-style read — parallel per row group, retain batches.
        let mut tables: Vec<TableData> = paths
            .iter()
            .map(|p| read_parquet_lance_parallel(p).unwrap())
            .collect();
        let t_read = Instant::now();

        // Phase 2: initial chain from tables[0] — ptrs index into the slim HT arena.
        let t0 = &tables[0];
        let n0 = t0.ht.n_rows();
        let t0_arena = t0.ht.arena_base_addr();
        let t0_row_size = t0.ht.layout().row_size();
        let t0_key_off = t0.ht.key_offset();
        let t0_hashes_addr = t0.ht.hashes_base_addr_const() as usize;

        let mut chain: Vec<ChainRow> = Vec::with_capacity(n0 * 4);
        chain.resize(n0, ChainRow::default());
        let chain_addr = chain.as_mut_ptr() as usize;
        (0..n0).into_par_iter_fill(move |i| unsafe {
            let row_ptr_i = (t0_arena + i * t0_row_size) as *const u8;
            let key = std::ptr::read_unaligned(row_ptr_i.add(t0_key_off) as *const i32);
            let h = *((t0_hashes_addr as *const u64).add(i));
            let dst = (chain_addr as *mut ChainRow).add(i);
            std::ptr::write(
                dst,
                ChainRow {
                    key,
                    _pad: 0,
                    hash: h,
                    ptrs: [row_ptr_i as u64, 0, 0, 0, 0],
                },
            );
        });

        // Phase 3: chain t1..t{k-1}.
        for i in 1..k {
            chain_step(&mut chain, &tables[i].ht, i);
        }
        let t_chain = Instant::now();
        let rows_out = chain.len();
        last_rows_out = rows_out;

        // Phase 4: sink — read data cols directly from Arrow via boundary lookup.
        let checksum = lance_sink(&tables, &chain);
        let checksum = std::hint::black_box(checksum);
        checksum_accum = checksum_accum.wrapping_add(checksum);
        let t_sink = Instant::now();

        let read = t_read.duration_since(t_start).as_secs_f64() * 1000.0;
        let chain_time = t_chain.duration_since(t_read).as_secs_f64() * 1000.0;
        let sink = t_sink.duration_since(t_chain).as_secs_f64() * 1000.0;
        let total = t_sink.duration_since(t_start).as_secs_f64() * 1000.0;

        if rep == 0 {
            println!(
                "[warmup] read={:.1} chain={:.1} sink={:.1} total={:.1} rows_out={}",
                read, chain_time, sink, total, rows_out
            );
            continue;
        }
        read_ms.push(read);
        chain_ms.push(chain_time);
        sink_ms.push(sink);
        total_ms.push(total);
        println!(
            "[rep {}]  read={:.1} chain={:.1} sink={:.1} total={:.1} rows_out={}",
            rep, read, chain_time, sink, total, rows_out
        );
        let _ = tables; // keep alive until end of rep
    }

    println!();
    println!("[median over {} reps — Lance-style FOJ chain]", repeats);
    println!("  parquet read (all k)   : {:>7.1} ms", median(&mut read_ms));
    println!("  chained probes + build : {:>7.1} ms", median(&mut chain_ms));
    println!("  sink (Arrow gather)    : {:>7.1} ms", median(&mut sink_ms));
    println!("  total                  : {:>7.1} ms", median(&mut total_ms));
    println!("  rows_out               : {}", last_rows_out);
    println!("  checksum               : 0x{:016x}", checksum_accum);
}

// Data-column set we "read" at sink time. Mirrors DuckDB's blackhole work:
// force every column byte to be touched, fold into a checksum.
const LANCE_NUMERIC_COLS: &[&str] =
    &["c_int", "c_bigint", "c_dbl", "c_flt", "c_bool"];
// Two string columns the numeric-only pipeline ignores — include only when
// we extend to the full 10-column output.
const LANCE_STRING_COLS: &[&str] = &["c_short", "c_long"];

fn lance_sink(tables: &[TableData], chain: &[ChainRow]) -> u64 {
    use rayon::prelude::*;

    // Per-table, per-batch: downcast every column we'll read. Done once up
    // front so the hot loop just does indexed reads.
    let numeric_views: Vec<Vec<BatchNumericView>> = tables
        .iter()
        .map(|t| t.batches.iter().map(|b| BatchNumericView::new(b)).collect())
        .collect();
    let string_views: Vec<Vec<BatchStringView>> = tables
        .iter()
        .map(|t| t.batches.iter().map(|b| BatchStringView::new(b)).collect())
        .collect();

    // Per-table arena bases + row sizes — needed to convert row_ptr → arena_idx.
    let arena_bases: Vec<usize> = tables.iter().map(|t| t.ht.arena_base_addr()).collect();
    let row_sizes: Vec<usize> = tables.iter().map(|t| t.ht.layout().row_size()).collect();
    let boundaries: Vec<&Vec<u32>> = tables.iter().map(|t| &t.boundaries).collect();

    chain
        .par_chunks(16_384)
        .map(|chunk| {
            let mut sum: u64 = 0;
            for cr in chunk {
                for (i, &ptr) in cr.ptrs.iter().enumerate() {
                    if ptr == 0 {
                        // FOJ NULL-side — fold a sentinel.
                        sum = sum.wrapping_add(0xDEAD_BEEF_CAFE_BABE);
                        continue;
                    }
                    // Convert raw ptr → arena index → (batch_idx, row_in_batch).
                    let arena_idx = (ptr as usize - arena_bases[i]) / row_sizes[i];
                    let probe = arena_idx as u32;
                    let p = boundaries[i].partition_point(|&b| b <= probe);
                    let batch_idx = p - 1;
                    let row = arena_idx - boundaries[i][batch_idx] as usize;

                    let nv = &numeric_views[i][batch_idx];
                    sum = sum.wrapping_add(nv.c_int.values()[row] as u64);
                    sum = sum.wrapping_add(nv.c_bigint.values()[row] as u64);
                    sum = sum.wrapping_add(nv.c_dbl.values()[row].to_bits());
                    sum = sum.wrapping_add(nv.c_flt.values()[row].to_bits() as u64);
                    sum = sum.wrapping_add(nv.c_bool.value(row) as u64);

                    // Strings: read length bytes, fold.
                    let sv = &string_views[i][batch_idx];
                    // c_short
                    let s = sv.c_short.value(row);
                    sum = sum.wrapping_add(s.len() as u64);
                    if !s.is_empty() {
                        sum = sum.wrapping_add(s.as_bytes()[0] as u64);
                    }
                    // c_long
                    let s = sv.c_long.value(row);
                    sum = sum.wrapping_add(s.len() as u64);
                    if !s.is_empty() {
                        sum = sum.wrapping_add(s.as_bytes()[0] as u64);
                    }
                }
            }
            sum
        })
        .reduce(|| 0u64, |a, b| a.wrapping_add(b))
}

struct BatchNumericView<'a> {
    c_int: &'a arrow::array::Int32Array,
    c_bigint: &'a arrow::array::Int64Array,
    c_dbl: &'a arrow::array::Float64Array,
    c_flt: &'a arrow::array::Float32Array,
    c_bool: &'a arrow::array::BooleanArray,
}

impl<'a> BatchNumericView<'a> {
    fn new(batch: &'a arrow::record_batch::RecordBatch) -> Self {
        use arrow::array::*;
        Self {
            c_int: batch
                .column_by_name("c_int")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap(),
            c_bigint: batch
                .column_by_name("c_bigint")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap(),
            c_dbl: batch
                .column_by_name("c_dbl")
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap(),
            c_flt: batch
                .column_by_name("c_flt")
                .unwrap()
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap(),
            c_bool: batch
                .column_by_name("c_bool")
                .unwrap()
                .as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap(),
        }
    }
}

struct BatchStringView<'a> {
    c_short: &'a arrow::array::StringArray,
    c_long: &'a arrow::array::StringArray,
}

impl<'a> BatchStringView<'a> {
    fn new(batch: &'a arrow::record_batch::RecordBatch) -> Self {
        use arrow::array::StringArray;
        Self {
            c_short: batch
                .column_by_name("c_short")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap(),
            c_long: batch
                .column_by_name("c_long")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap(),
        }
    }
}
