# duck-hash-join

A Rust port of DuckDB's parallel hash join, evaluated against the 5-way FULL
OUTER JOIN benchmark in [`bench/fullouter5/`](../../bench/fullouter5/README.md).

## Tl;dr

| variant | 4-thread total | 8-thread total | vs DuckDB |
|---|---:|---:|---:|
| DuckDB `kway_bench hash_join` (5-way FOJ, 11 cols, blackhole) | 275.8 ms | 193.4 ms | — |
| **Rust `--lance`** (this crate, preferred) | **240.8 ms** | **182.3 ms** | **13% / 6% faster** |
| Rust `--operator` (DuckDB-style framework) | 299.5 ms | 208.6 ms | 9% slower / 8% slower |
| Rust `--streaming` | ~204 ms (4t) | ~172 ms (8t) | best at 4t |

Workload: `k=5, overlap=0.5, 1M rows/table, 11 columns per table (incl. 2
strings), blackhole sink materializing every output byte`. Output is 3 M FOJ
rows, checksum verified non-zero across runs.

The Lance-style variant is the best-performing implementation in this crate.

## Motivation

DuckDB's parallel hash join is fast. We wanted to know if porting it to Rust
— staying within Arrow's zero-copy columnar model rather than DuckDB's
internal `TupleDataCollection` row format — could match or beat it on a
representative analytical benchmark.

Answer: yes, and by a meaningful margin on the 5-way FOJ workload — provided
you keep the implementation simple (Lance-style phased chain) rather than
layering on a full DuckDB-style operator framework.

## How the benches are organized

| Binary | What it does |
|---|---|
| [`inner_join_bench`](src/bin/inner_join_bench.rs) | 2-way INNER join baseline |
| [`chain_bench`](src/bin/chain_bench.rs) | k-way chain with multiple execution variants |

`chain_bench` exposes all the variants we tried, selected by flag:

| Flag | Execution strategy |
|---|---|
| (default) | Keys-only chain, minimal — useful for isolating hash-join algorithm cost |
| `--full-cols` | Read 6 numeric columns into arena, count output rows |
| `--foj` | FULL OUTER JOIN semantics (keys-only) |
| `--pipelined` | FOJ chain carrying per-row pointer arrays through intermediate `Vec<ChainRow>` |
| `--lance` | **Best variant.** Slim 20 B rows + `Arc<RecordBatch>` retention + direct Arrow buffer sink |
| `--streaming` | `--lance` plus parquet reads on dedicated threads overlapped with chain work |
| `--operator` | Full DuckDB-style operator framework (Source/Sink/Operator traits, Global/Local state, MetaPipeline deps) |
| `--scan-only` | Read `joined.parquet`'s 6 numeric cols — lower-bound scan baseline |
| `--scan-lance` | Read `joined.parquet`'s full 11 cols — apples-to-apples with DuckDB `scan_joined` |

Run it against the parquet files `kway_bench` generates. Example:
```bash
./build/tools/kway_bench --rows 1000000 --ks "5" --overlaps "0.5" --threads 4 --dir /tmp/kway_4t --keep
cd rust/duck-hash-join
cargo build --release
./target/release/chain_bench /tmp/kway_4t/k5_o050 5 5 --threads=4 --lance
```

## What got built (bottom-up)

### 1. [`hash.rs`](src/hash.rs) — MurmurHash64
Port of DuckDB's `src/include/duckdb/common/types/hash.hpp:38-45`. Cross-validated
bit-for-bit via [`tools/utils/hash_dump/hash_dump.cpp`](../../tools/utils/hash_dump/hash_dump.cpp),
which dumps DuckDB's actual hash values for known inputs. Lives in a unit test
(`murmur_known_vectors`).

### 2. [`radix.rs`](src/radix.rs) — RadixPartitioning
Port of `src/include/duckdb/common/radix_partitioning.hpp`. Extracts the radix
bits from a 64-bit hash immediately below the 16 salt bits (DuckDB's
convention so salt and radix don't overlap).

### 3. [`ht_entry.rs`](src/ht_entry.rs) — HtEntry
Port of `src/include/duckdb/execution/ht_entry.hpp`. A `u64` packing 16 bits of
salt (hash fingerprint) into the top and a 48-bit pointer into the bottom.
Linear probing uses the salt for fast rejection before dereferencing the row.

### 4. [`row.rs`](src/row.rs) — Fixed-width row arena
Simplified `TupleDataCollection`. A single `Vec<u8>` arena with `RowLayout`
for column offsets. Fixed-width columns only (no varlen) — strings are
handled differently; see `io.rs` below.

### 5. [`ht.rs`](src/ht.rs) — JoinHashTable
Linear-probed hash table with salt fast-reject and chain walking. Supports
both a serial `finalize()` and a parallel `finalize_parallel()` that
partitions inserts by the top radix bits of the bucket.

### 6. [`io.rs`](src/io.rs) — Parquet ingress and TableData
Multiple readers:
- `read_parquet_into_ht_parallel` — Classic full-column ingest (data copied into arena).
- `read_parquet_keys_into_ht_parallel` — Keys-only ingest for pure hash-join experiments.
- `read_parquet_lance_parallel` — **Lance-style.** Retains `Arc<Vec<RecordBatch>>` + a `boundaries` vec so rows' data columns stay in Arrow's buffers. The arena only holds the key + hash + header. This is the design that produces the best performance.

### 7. [`pipeline.rs`](src/pipeline.rs) — DuckDB-style operator framework
`Source` / `Sink` / `Operator` traits with the Global/Local state split.
Concrete operators: `ParquetScanSource`, `HashJoinOp`, `ProbeOperator`,
`UnmatchedSource`, `ChecksumSinkOp`. MetaPipeline-style orchestration in
`run_chain_operator` — two sub-pipelines feed each FOJ's build sink (the
probe stream + the prior FOJ's build-unmatched source).

Partition-parallel finalize (`HashJoinOp::finalize`) uses a 4× load factor
so partition slices stay ≤ 25% full, which eliminates slice-boundary
wraparound and lets each partition's inserts skip CAS entirely.

## Key findings

### 1. Lance-style is the performance sweet spot

On the 5-way FOJ chain with 11-column blackhole sink (the `kway_bench
hash_join` workload):

| | 4 threads | 8 threads |
|---|---:|---:|
| DuckDB hash_join (scan + join) | 275.8 ms | 193.4 ms |
| DuckDB scan_joined (scan only) | 170.3 ms | 91.6 ms |
| **DuckDB join overhead** | **+106 ms** | **+102 ms** |
| Rust `--lance` (scan + join) | 240.8 ms | 182.3 ms |
| Rust `--scan-lance` (scan only) | 165.3 ms | 101.9 ms |
| **Rust `--lance` join overhead** | **+76 ms** | **+80 ms** |

Rust `--lance`'s **pure join work is ~25 ms cheaper per chain** than DuckDB.
The relative overhead is 1.45×/1.78× scan vs DuckDB's 1.61×/2.11× scan.

### 2. The operator framework doesn't beat simple straight-line code

We built a full DuckDB-style operator framework (pipelines, Global/Local
state split, partition-parallel finalize with no CAS). Correctness-verified
at 3,000,000 exact FOJ output rows with stable checksums.

Result: **consistently 25-60 ms slower than `--lance`** at both thread counts.

Why:
- Per-batch `Batch { keys: Vec<_>, hashes: Vec<_>, ptrs: Vec<_> }`
  allocations — ~2000 per chain.
- `Mutex<Vec<u8>>` on the shared arena during `combine`.
- Two-pass partition scatter in finalize.
- No SIMD for per-row ptr merging.

These are all fixable, but the fix list tells the real story: **DuckDB's
framework wins only because they've spent years inlining the abstractions
away**. A first-pass Rust port of that framework carries real per-batch
overhead. For a fixed-plan benchmark like this, straight-line code wins.

### 3. String columns via Arrow references cost essentially zero

DuckDB's `TupleDataCollection` copies string bytes into a per-chunk heap
and caches a 4-byte prefix in the row slot for fast equality. This is
expensive at ingest time — a `memcpy` per non-inlined string.

Our Lance-style approach just keeps `Arc<RecordBatch>` alive and reads
strings through Arrow's `StringArray::value(i)` at the sink. Because strings
are *payload only* in this workload (join key is `i32`), the cached-prefix
optimization doesn't apply — the sink reads each string once. Net: zero
string-copy work at ingest, same sink cost.

This design is inspired by Lance's `HashJoiner`
(`rust/lance/src/dataset/hash_joiner.rs`), which similarly keeps source
batches alive and uses Arrow's `interleave` for gather.

### 4. Operator-level pipelining is oversold for this workload

A left-deep chain of 4 FULL OUTER JOINs has 4 pipeline breakers —
each FOJ's build side must complete before the next can finalize. Streaming
batches through the chain saves peak memory but doesn't speed up the
critical path.

What *did* help: overlapping parquet reads with chain work
(`--streaming`). At 4 threads it wins 15% vs the phased Lance variant
because the 4 cores don't saturate during chain-phase serial portions, so
background reads get free cycles. At 8 threads this advantage disappears
(phased Lance already saturates 8 cores).

### 5. Rust beats DuckDB because it avoids one particular copy

The core reason Rust wins by 13% at 4 threads: **we skip the parquet-decode-to-arena copy for data columns**. DuckDB's hash join materializes every input column into its TupleDataCollection arena during build. Our Lance-style keeps them in Arrow. Everything else (probe, finalize, sink cost) is within a few percent of DuckDB.

## What we did not build (future work)

- **String join keys.** Our HT only supports i32 keys. Adding string keys
  requires either porting DuckDB's inline-prefix `string_t` or using an
  Arrow-row canonical binary format like Lance's `HashJoiner`.
- **Out-of-core (spill to disk).** DuckDB spills when the build side
  exceeds memory; we just fail.
- **Multi-column join keys.** Port of `VectorOperations::CombineHash` is
  in `hash.rs::combine_hash` but not wired through.
- **More operator types.** No sort, aggregate, or filter operators — just
  what's needed for a k-way FOJ benchmark.
- **DataFrame API / SQL front end.** Caller passes paths + key column name.

## Benchmark harnesses

- [`tools/utils/kway_bench.cpp`](../../tools/utils/kway_bench.cpp) —
  DuckDB-side k-way FOJ workload, generates parquet input files with the
  `k + 10 data column` schema and runs all three variants (scan_joined,
  hash_join, kway_op).
- [`tools/utils/hash_dump/hash_dump.cpp`](../../tools/utils/hash_dump/hash_dump.cpp) —
  Cross-validates MurmurHash64 output between Rust and DuckDB.
- [`tools/utils/inner_bench/inner_bench.cpp`](../../tools/utils/inner_bench/inner_bench.cpp) —
  DuckDB-side 2-way / k-way INNER or FOJ harness with apples-to-apples
  projection (6 cols blackhole or count, matching the Rust variants).
- [`tools/utils/datafusion_bench/`](../../tools/utils/datafusion_bench/) —
  DataFusion comparison (Phase 0 — used to decide whether a port was
  justified; DataFusion was 2-34× slower than DuckDB depending on config,
  which motivated the port).

## Method for gathering numbers in this README

All "median of N runs" numbers use N=5 unless noted. Each run does 3 internal
repeats with the first discarded as warmup (page-cache warming), then
reports the median of the remaining. Files are read through OS page cache
after the first kway_bench run, so measurements don't include cold-start
parquet I/O — they're comparable because DuckDB's kway_bench and Rust's
chain_bench both benefit equally from the cache.

Host: Apple M2 Max (8 cores: 4P+4E), macOS 26.4.

## Bench data: where it lives and how to make it

- **Files live** in `$DIR/k{K}_o{OVERLAP*100}/` — per-config subdirs with
  `t0.parquet`..`t{K-1}.parquet` (inputs) + `joined.parquet` (pre-materialized
  FOJ output used as the `scan_joined` baseline). At `rows=1000000, k=5,
  overlap=0.5` each table is ~67 MB on disk (11 cols incl. 2 strings) and
  `joined.parquet` is ~332 MB.
- **Regenerate with** [`scripts/gen_bench_data.sh`](scripts/gen_bench_data.sh).
  It builds `kway_bench` if needed, then writes the full `{k=2,5,6} × {0.0,
  0.5, 1.0}` matrix by default. Override via `DIR=`, `KS=`, `OVERLAPS=`,
  `ROWS=`, `THREADS=`, `REPEATS=` env vars. Example:
  ```bash
  DIR=/tmp/my_bench KS=5 OVERLAPS=0.5 rust/duck-hash-join/scripts/gen_bench_data.sh
  ```
- **Why kway_bench doubles as the generator:** the DuckDB-side bench in
  [`tools/utils/kway_bench.cpp`](../../tools/utils/kway_bench.cpp) emits the
  parquet files as a side effect of timing its own `scan_joined`,
  `hash_join`, and `kway_op` variants. Using the same tool for both sides
  guarantees byte-identical inputs — no generation drift between the DuckDB
  and Rust measurements.
- **Key distribution** (kway_bench.cpp:173-217): seeded-random shuffled
  keyspace (`setseed(0.42)`), then each table gets `shared + unique` keys
  where `shared = rows * overlap` is the same across all tables and
  `unique = rows - shared` is per-table disjoint. Each table sorted by `k`
  before writing.

## Quick commands

```bash
# 1. Generate parquet data (one-time)
rust/duck-hash-join/scripts/gen_bench_data.sh
# or, equivalent manual:
./build/tools/kway_bench --rows 1000000 --ks "5" --overlaps "0.5" \
    --repeats 3 --threads 4 --dir /tmp/kway_4t --keep

# 2. Build this crate
cd rust/duck-hash-join && cargo build --release

# 3. Run the winning variant
./target/release/chain_bench /tmp/kway_4t/k5_o050 5 5 --threads=4 --lance

# 4. Compare to DuckDB
./build/tools/kway_bench --rows 1000000 --ks "5" --overlaps "0.5" \
    --repeats 5 --threads 4 --dir /tmp/kway_4t --keep

# 5. Apples-to-apples scan baseline
./target/release/chain_bench /tmp/kway_4t/k5_o050 5 5 --threads=4 --scan-lance

# 6. (Optional) try the operator framework
./target/release/chain_bench /tmp/kway_4t/k5_o050 5 5 --threads=4 --operator
```
