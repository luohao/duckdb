# 5-way FULL OUTER JOIN benchmark

This directory ships a small self-contained benchmark harness that compares
reading five parquet files and joining them on a shared key `k` against
reading the same data already pre-joined into a single denormalized parquet
file. It is intentionally **ad-hoc and separate** from the formal
`benchmark/` runner at the repo root.

It comes in two shapes so we can measure the same thing from two angles:

| bench | where it runs | what it measures |
|---|---|---|
| `bench/fullouter5/bench_fullouter5.py` | Python, via the `duckdb` module | end-to-end: SQL parse → plan → optimize → execute → drain |
| `tools/utils/plan_bench.cpp`           | C++, directly against `libduckdb` | just execute: plan is built once, then a `LogicalPlanStatement` is rerun in a hot loop (no parser/binder per iteration) |

Both use the same five input parquet files, both drain the result with the
same sink, and both produce byte-identical logical plans. So the gap
between the two numbers is an upper bound on the parse/bind/optimize
overhead DuckDB pays per SQL execution.

---

## What's new

### 1. `blackhole` aggregate (`src/function/aggregate/distributive/blackhole.cpp`)

A built-in aggregate that accepts any number of arguments of any type,
forces them to be read from the scan, does **no per-row work**, and always
returns `BIGINT 0`. It's the sink used by both benches.

Why we need it:

* `COUNT(*)` on parquet triggers a metadata-only fast path — no values
  are read, no joins run. Useless as a benchmark sink.
* `SUM(HASH(_t))` works but adds real per-row hashing cost on top of the
  plan we care about.
* `blackhole(*COLUMNS(*))` evaluates every column (so projection pushdown
  can't drop them) but does nothing with the values. Only a single
  `BIGINT 0` row crosses the C++/Python boundary.

Registered in `functions.json`, `function_list.cpp`,
`distributive_functions.hpp`, and the distributive CMakeLists.

```sql
duckdb> SELECT blackhole(x, y, z) FROM big_table;   -- forces read, does nothing
0
```

### 2. `bench/fullouter5/bench_fullouter5.py`

Python bench with full DuckDB JSON profiling (latency, CPU time, peak
buffer/temp memory, per-operator timing, cardinality, rows scanned).

### 3. `tools/utils/plan_bench.cpp`

C++ bench that builds each logical plan once and reruns it. Tests six
variants: the same 5-way join in natural, left-deep, right-deep, bushy
shapes (to show optimizer effect on join ordering), plus two scan
references.

---

## Building

### C++ (`plan_bench`)

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make plan_bench -j
./tools/plan_bench --help
```

### Python (`bench_fullouter5.py`)

The Python bench uses the `blackhole` aggregate, so the stock pip wheel
of `duckdb` **won't work** — it doesn't have the patch. Build the
bindings from source against this tree (run from the repo root):

```bash
python -m venv .venv-bench
source .venv-bench/bin/activate

git clone https://github.com/duckdb/duckdb-python.git ../duckdb-python
OVERRIDE_GIT_DESCRIBE=v1.5.2 \
  pip install \
    --config-settings=cmake.define.DUCKDB_SOURCE_PATH=$PWD \
    ../duckdb-python
```

The script fails loudly with a clear message if the imported `duckdb`
doesn't expose `blackhole`.

---

## Running

### Shared data dir

Point both benches at the same `--dir` and they share the parquet files
and the pre-materialized `joined.parquet`. This is the supported way to
get apples-to-apples timings. Run from the repo root:

```bash
DATA=/tmp/bench_fo5
mkdir -p $DATA

# C++: build-and-run six variants
./build/tools/plan_bench \
    --rows 1000000 --overlap 0.5 \
    --repeats 3 --threads 4 \
    --dir $DATA --keep

# Python: run the same two comparisons with profiling
python bench/fullouter5/bench_fullouter5.py \
    --rows 1000000 --overlap 0.5 \
    --repeats 3 --threads 4 \
    --dir $DATA --keep \
    --perf-dir ./perf_out
```

### Useful flags

`plan_bench`:
* `--variant fo5_optimized` — only run one variant
* `--print-plan` — dump the logical plan before executing
* `--keep` — don't delete the temp data dir on exit

`bench_fullouter5.py`:
* `--explain` — print `EXPLAIN ANALYZE` for the join
* `--perf-dir DIR` — keep the JSON profiles (one per run; fastest run is reported)
* `--summary-json FILE` — dump a single-file summary
* `--memory-limit 8GB` — cap RAM

---

## Sample results

Host: Apple M2 Max, macOS 26.4.1, DuckDB built in `Release` mode.
Settings: `rows=1,000,000` per file, `overlap=0.5`, `threads=4`,
`repeats=3`. Shared data dir, so both benches touched the exact same
parquet bytes.

### `plan_bench` (C++, plan-only, no SQL in the hot loop)

```
[bench] repeats=3  threads=4  sink=blackhole
        variant                         min(ms)   median(ms)      opt
        ------------------------------------------------------------------
        fo5_optimized                     216.7        218.0       on
        fo5_leftdeep_raw                  219.6        219.9      off
        fo5_rightdeep_raw                 217.3        217.3      off
        fo5_bushy_raw                     239.0        239.7      off
        scan_t0                            15.6         15.7      off
        scan_joined                       132.9        133.5      off
```

### `bench_fullouter5.py` (end-to-end)

```
[timing] query                              wall_min_ms   wall_med_ms
         5-way FULL OUTER JOIN                    222.0         224.3
         SCAN materialized parquet                133.6         134.8
[ratio ] join / scan = 1.66x  (join overhead on top of scan)
```

### Takeaways

* **Parse/bind/plan overhead is ~5 ms** per execution: C++
  `fo5_optimized` 216.7 ms vs Python `fullouter_join` 222.0 ms on the
  same plan, same data. That's the full SQL front-end cost.
* **Scan bound**: `scan_joined` and `scan_materialized` are within noise
  (132.9 vs 133.6 ms) — no surprise, same bytes through the same parquet
  reader. These are both ~8× the single-file `scan_t0` (15.6 ms), which
  tracks the 5× input volume plus full-column materialization of a
  6M-row result.
* **Optimizer doesn't move the needle here**: left-deep, right-deep,
  and "optimized" (whatever DuckDB picks) all land within 3 ms of each
  other. The bushy shape is measurably worse (~22 ms), which tells us
  DuckDB's default ordering is already near-optimal for this query.
* **Join / scan ~1.64×**: a 5-way FULL OUTER HASH_JOIN over these
  payload widths costs about 64 % more than just reading the
  pre-joined parquet. That's the number the whole benchmark exists to
  produce.

---

## Layout reference

```
bench/fullouter5/
    README.md                     # this file
    bench_fullouter5.py           # Python bench
src/function/aggregate/distributive/
    blackhole.cpp                 # the UDF
    CMakeLists.txt                # adds blackhole.cpp
    functions.json                # registers `blackhole` aggregate set
src/function/function_list.cpp    # hooks BlackholeFun::GetFunctions()
src/include/duckdb/function/aggregate/distributive_functions.hpp
                                  # declares BlackholeFun
tools/utils/plan_bench.cpp        # C++ bench
tools/CMakeLists.txt              # builds plan_bench
```

---

# K-way merge join (port of ClickHouse `SortingQueueImpl`)

On top of the `fullouter5` infrastructure, we built a k-way merge join as a
real DuckDB `PhysicalOperator` and benchmarked it against the parallel hash
join over a range of k values and overlap fractions. This section documents
what was built, the optimizations that shipped, and the measured results.

## What was built

### 1. `PhysicalKWayMergeJoin` operator

New files:

- `src/include/duckdb/execution/operator/join/physical_kway_merge_join.hpp`
- `src/execution/operator/join/physical_kway_merge_join.cpp`

A multi-input `PhysicalOperator` (k children, k ≥ 2) that implements FULL
OUTER JOIN over k sorted inputs via a priority-queue merge. The heap + batch
logic is a direct port of ClickHouse's `SortingQueueImpl`
(`src/Core/SortCursor.h:378–621`), adapted for DuckDB's `ColumnDataCollection`
storage and `DataChunk` output.

Three merge paths:

- **BATCH path** — when the top cursor's run of rows has keys strictly less
  than the next-best cursor's current key, emit them all with one
  `VectorOperations::Copy` per column. Hot path for unique-to-one-table keys.
- **ALL-AGREE path** — when all k cursors are at the same key and hold
  consecutive matching keys (`K, K+1, K+2, …`), emit the run as a bulk batch
  per cursor. Hot path for shared-key prefixes.
- **GROUP path (generic)** — per-row copies for each contributing cursor at
  the current key; NULLs for non-contributing children fall out of the
  pre-pass `SetAllInvalid` on the output columns.

### 2. Planner integration

- `src/main/settings.hpp` / `config.cpp` — new session setting
  `force_kway_merge_join` (default `false`).
- `src/execution/physical_plan/plan_comparison_join.cpp` — when the setting
  is on and we encounter a `LogicalComparisonJoin` that's FULL OUTER with a
  single equality condition, recursively flatten the chain (`FlattenFOJChain`)
  and emit a single `PhysicalKWayMergeJoin` with all k leaves as children.
  When off, the existing hash-join path is used.
- `src/include/duckdb/common/enums/physical_operator_type.hpp` /
  `src/common/enums/physical_operator_type.cpp` — new `KWAY_MERGE_JOIN`
  enum value + string mapping.

### 3. `kway_bench`

New C++ bench harness: `tools/utils/kway_bench.cpp`. It sweeps (k, overlap)
pairs and times three variants per config, all with a blackhole sink:

- **`scan_joined`** — `SELECT blackhole(*COLUMNS(*)) FROM joined.parquet`.
  Lower bound: reads the denormalized answer, does no join work.
- **`hash_join`** — `SELECT blackhole(*COLUMNS(*)) FROM t0 FULL OUTER JOIN
  ... FULL OUTER JOIN t{k-1}`. Parallel hash join (default DuckDB path).
- **`kway_op`** — same SQL but with `SET force_kway_merge_join=true`, so
  the planner collapses the chain into `PhysicalKWayMergeJoin`.

Data is generated via a shuffled permutation of the keyspace, then sorted
per file — so each file is sorted but key *values* are scattered
(representative of real workloads, not the optimistic contiguous-integer
case).

## Optimizations

### Operator integration with DuckDB's pipeline

The operator uses the `IEJoin` pattern for pipeline construction:

```cpp
void PhysicalKWayMergeJoin::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
    meta_pipeline.GetState().SetPipelineSource(current, *this);
    auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);

    // Child 0 → base pipeline
    pipeline_to_child.emplace(*child_meta_pipeline.GetBasePipeline(), 0);
    children[0].get().BuildPipelines(*child_meta_pipeline.GetBasePipeline(), child_meta_pipeline);

    // Children 1..k-1 → new pipelines in the same child_meta_pipeline,
    // each with its own finish event (fires Finalize on this operator).
    for (idx_t i = 1; i < children.size(); i++) {
        auto &pipe = child_meta_pipeline.CreatePipeline();
        pipeline_to_child.emplace(pipe, i);
        children[i].get().BuildPipelines(pipe, child_meta_pipeline);
        child_meta_pipeline.AddFinishEvent(pipe);
    }
}
```

Key trick: **`pipeline_to_child` maps each sink-side `Pipeline*` to its
child index**, so `GetLocalSinkState(ExecutionContext &)` can look up the
current pipeline (`context.pipeline`) and tag the per-thread
`LocalSinkState` with the right `child_idx`. This lets chunks route
deterministically to the correct per-child `ColumnDataCollection`
regardless of scheduling order.

### Parallel source via merge-path partitioning

The source splits the key space into P = `thread_count` partitions, with
each thread running an independent k-way merge over its slice:

1. **Build a per-child `ChunkIndex` once** in `GetGlobalSourceState()`:
   for each child's CDC, record `(first_key, last_key, start_row)` per
   chunk. O(chunks) `FetchChunk` calls; enables O(log chunks) binary search
   by key.
2. **Sample P−1 boundary keys** from cursor 0's row distribution (rows at
   positions `n/P, 2n/P, …`).
3. **For each partition p**, each thread claims it atomically
   (`gstate.next_partition.fetch_add(1)`), then positions its k cursors
   at `LowerBoundRow(boundary_keys[p-1])` on each child's CDC, bounded by
   `end_key = boundary_keys[p]`. The last partition uses `INT_MAX` as
   `end_key`.
4. The per-partition merge is the **same** algorithm — the priority queue
   + BATCH/ALL-AGREE/GROUP paths — just on a bounded slice of each input.

**The `ChunkIndex` build is skipped when `num_partitions == 1`** (single
thread). That keeps the single-threaded path free of partitioning overhead
and falls straight through to the classic one-cursor-per-child merge.

## Results — shuffled-keyspace bench

Host: Apple M2 Max, macOS, DuckDB `Release` build. 1 M rows per input file.
All variants read parquet from disk and use the `blackhole` sink. Ratios
are against `scan_joined` (= reading the denormalized answer with
blackhole — the theoretical minimum).

### 1 thread (`--threads 1`)

| k | overlap | scan_joined | hash_join | kway_op | hash/scan | kway/scan |
|---|---------|------------:|----------:|--------:|----------:|----------:|
| 2 | 0.00 |  153 |  197 |   431 | 1.29x | 2.82x |
| 2 | 0.50 |  140 |  217 |   469 | 1.55x | 3.35x |
| 2 | 1.00 |  114 |  201 |   197 | 1.76x | **1.72x** |
| 3 | 0.00 |  300 |  384 |   728 | 1.28x | 2.43x |
| 3 | 0.50 |  246 |  401 |   771 | 1.63x | 3.13x |
| 3 | 1.00 |  164 |  332 |   291 | 2.03x | **1.78x** |
| 4 | 0.00 |  465 |  579 | 1 022 | 1.24x | 2.20x |
| 4 | 0.50 |  365 |  599 | 1 078 | 1.64x | 2.95x |
| 4 | 1.00 |  221 |  482 |   375 | 2.18x | **1.70x** |
| 5 | 0.00 |  700 |  828 | 1 351 | 1.18x | 1.93x |
| 5 | 0.50 |  514 |  803 | 1 347 | 1.56x | 2.62x |
| 5 | 1.00 |  280 |  638 |   485 | 2.28x | **1.73x** |
| 6 | 0.00 |  956 | 1 126 | 1 685 | 1.18x | 1.76x |
| 6 | 0.50 |  668 | 1 029 | 1 641 | 1.54x | 2.46x |
| 6 | 1.00 |  326 |  800 |   570 | 2.45x | **1.75x** |

### 4 threads (`--threads 4`)

| k | overlap | scan_joined | hash_join | kway_op | hash/scan | kway/scan |
|---|---------|------------:|----------:|--------:|----------:|----------:|
| 2 | 0.00 |  44 |  68 |   250 | 1.54x | 5.70x |
| 2 | 0.50 |  39 |  71 |   256 | 1.82x | 6.55x |
| 2 | 1.00 |  32 |  67 |   177 | 2.11x | 5.61x |
| 3 | 0.00 |  81 | 114 |   321 | 1.40x | 3.97x |
| 3 | 0.50 |  66 | 119 |   338 | 1.80x | 5.11x |
| 3 | 1.00 |  46 | 103 |   205 | 2.22x | 4.41x |
| 4 | 0.00 | 131 | 168 |   402 | 1.28x | **3.07x** |
| 4 | 0.50 | 100 | 172 |   409 | 1.72x | 4.10x |
| 4 | 1.00 |  62 | 151 |   233 | 2.41x | 3.73x |
| 5 | 0.00 | 188 | 243 |   482 | 1.29x | 2.57x |
| 5 | 0.50 | 141 | 232 |   484 | 1.64x | 3.43x |
| 5 | 1.00 |  75 | 187 |   254 | 2.50x | 3.40x |
| 6 | 0.00 | 255 | 330 |   630 | 1.29x | **2.47x** |
| 6 | 0.50 | 184 | 308 |   612 | 1.67x | 3.32x |
| 6 | 1.00 |  91 | 242 |   341 | 2.68x | 3.77x |

## Takeaways

1. **Overlap=1.00 is `kway_op`'s best regime.** At 1 thread, when every
   table has the same keyset, `kway_op` beats `hash_join` across all k.
   The margin grows with k — near-tied at k=2 (197 vs. 201 ms), to 29%
   faster at k=6 (570 vs. 800 ms).

2. **`kway_op`'s relative cost narrows with k.** At 4 threads, `kway/scan`
   drops from 5.6–6.6× at k=2 to 2.5–3.8× at k=6, while `hash/scan` stays
   in a 1.3–2.7× band across all k. The single-pass merge amortizes better
   than chained binary hash joins as table count grows — though `hash_join`
   still wins on wall time across the 4-thread sweep.

3. **1-thread range.** `kway_op` completes the join in 1.7–3.4× the time
   of `scan_joined` (reading the pre-materialized answer); `hash_join`
   needs 1.2–2.5× over the same set of configs.

4. **All times are full pipeline** — parquet decode in each sink + k-way
   merge in source + blackhole — so the numbers are apples-to-apples with
   `hash_join`.

## Running the k-way bench

```bash
# Build the operator + bench
cd build && make kway_bench -j

# Full 1t matrix
./tools/kway_bench \
    --rows 1000000 \
    --ks "2,3,4,5,6" \
    --overlaps "0.0,0.25,0.5,0.75,1.0" \
    --repeats 3 --threads 1 \
    --dir /tmp/kway_1t --keep

# Full 4t matrix
./tools/kway_bench \
    --rows 1000000 \
    --ks "2,3,4,5,6" \
    --overlaps "0.0,0.25,0.5,0.75,1.0" \
    --repeats 3 --threads 4 \
    --dir /tmp/kway_4t --keep
```

Each (k, overlap) pair goes in its own subdir (`k{K}_o{O*100}`) so
directories can be reused across runs.

## Layout reference (k-way additions)

```
src/include/duckdb/execution/operator/join/
    physical_kway_merge_join.hpp      # operator header
src/execution/operator/join/
    physical_kway_merge_join.cpp      # operator impl (Sink/Source/BuildPipelines)
    CMakeLists.txt                    # adds physical_kway_merge_join.cpp
src/include/duckdb/common/enums/
    physical_operator_type.hpp        # adds KWAY_MERGE_JOIN enum
src/common/enums/
    physical_operator_type.cpp        # ToString mapping
src/include/duckdb/main/settings.hpp  # ForceKWayMergeJoinSetting
src/main/config.cpp                   # registers the setting
src/execution/physical_plan/
    plan_comparison_join.cpp          # detects FOJ chains, emits KWayMergeJoin
tools/utils/kway_bench.cpp            # matrix bench harness
```
