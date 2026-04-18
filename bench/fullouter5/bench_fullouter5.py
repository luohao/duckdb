#!/usr/bin/env python3
"""
Micro-bench: 5-way FULL OUTER JOIN on the same key across Parquet files vs.
reading the pre-materialized denormalized Parquet.

Each input file has a join key `k` plus 10 value columns of mixed types
(int, bigint, double, float, varchar short/long, date, timestamp, bool, decimal).

Sink
----
The measured queries are wrapped with the built-in `blackhole` aggregate:

    SELECT blackhole(*COLUMNS(*)) FROM (<query>) _t

`blackhole` is a do-nothing aggregate (see
src/function/aggregate/distributive/blackhole.cpp): it forces every projected
column to be evaluated (defeating metadata-only fast paths like the parquet
COUNT(*) shortcut) but performs no per-row work, so nothing meaningful is
added on top of the plan we actually want to measure. Only a single `BIGINT 0`
row crosses the C++/Python boundary. This matches `plan_bench --output
blackhole` byte-for-byte (same logical plan).

Requirement
-----------
The `duckdb` Python module needs to have been built against this source tree
so that it exposes the `blackhole` aggregate. The stock pip wheel does not.
See the CONTRIBUTING notes in ../duckdb-python for building a local wheel.

Example
-------
    python bench_fullouter5.py --rows 1000000 --overlap 0.5 --threads 4 \\
        --perf-dir ./perf_out
"""
import argparse
import json
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import duckdb

N_FILES = 5

VALUE_COLUMNS = [
    ("c_int",   "((k * 7  + {i} * 13) % 100000)::INTEGER"),
    ("c_bigint","(k::BIGINT * 1000 + {i})::BIGINT"),
    ("c_dbl",   "((k + {i}) * 3.14159)::DOUBLE"),
    ("c_flt",   "((k * 0.01 + {i})::FLOAT)"),
    ("c_short", "('tag_' || ((k + {i}) % 100))::VARCHAR"),
    ("c_long",  "md5(k::VARCHAR || '_' || {i}::VARCHAR)"),
    ("c_date",  "(DATE '2020-01-01' + ((k + {i}) % 1000) * INTERVAL 1 DAY)::DATE"),
    ("c_ts",    "(TIMESTAMP '2020-01-01' + ((k + {i}) % 86400) * INTERVAL 1 SECOND)::TIMESTAMP"),
    ("c_bool",  "((k + {i}) % 2 = 0)::BOOLEAN"),
    ("c_dec",   "(((k + {i}) * 1.23)::DECIMAL(18,2))"),
]
assert len(VALUE_COLUMNS) == 10


# ---------------------------------------------------------------------------
# data generation / queries
# ---------------------------------------------------------------------------

def gen_parquet(con, out_dir: Path, rows: int, overlap: float) -> None:
    """Create 5 Parquet files sharing `overlap` fraction of their keys."""
    shared = int(rows * overlap)
    unique = rows - shared
    for i in range(N_FILES):
        start = shared + i * unique
        end = start + unique
        col_sql = ",\n                       ".join(
            f"{expr.format(i=i)} AS {name}" for (name, expr) in VALUE_COLUMNS
        )
        con.execute(f"""
            COPY (
                SELECT k::INTEGER AS k,
                       {col_sql}
                FROM (
                    SELECT range AS k FROM range(0, {shared})
                    UNION ALL
                    SELECT range AS k FROM range({start}, {end})
                )
            ) TO '{out_dir}/t{i}.parquet' (FORMAT PARQUET);
        """)


def materialize_join(con, out_dir: Path) -> None:
    con.execute(f"""
        COPY (
            SELECT * FROM '{out_dir}/t0.parquet' t0
            FULL OUTER JOIN '{out_dir}/t1.parquet' t1 USING (k)
            FULL OUTER JOIN '{out_dir}/t2.parquet' t2 USING (k)
            FULL OUTER JOIN '{out_dir}/t3.parquet' t3 USING (k)
            FULL OUTER JOIN '{out_dir}/t4.parquet' t4 USING (k)
        ) TO '{out_dir}/joined.parquet' (FORMAT PARQUET);
    """)


def join_query(d: Path) -> str:
    return f"""
        SELECT * FROM '{d}/t0.parquet' t0
        FULL OUTER JOIN '{d}/t1.parquet' t1 USING (k)
        FULL OUTER JOIN '{d}/t2.parquet' t2 USING (k)
        FULL OUTER JOIN '{d}/t3.parquet' t3 USING (k)
        FULL OUTER JOIN '{d}/t4.parquet' t4 USING (k)
    """


def scan_query(d: Path) -> str:
    return f"SELECT * FROM '{d}/joined.parquet'"


def _blackhole_wrap(sql: str) -> str:
    # Mirrors `plan_bench --output blackhole`. Bare `*` is stripped by the
    # Postgres grammar inside aggregate calls (that is why COUNT(*) is
    # rewritten to count_star), so we use DuckDB's *COLUMNS(*) unpack to
    # forward every column as a separate argument.
    return f"SELECT blackhole(*COLUMNS(*)) FROM ({sql}) _t"


def _require_blackhole(con) -> None:
    n = con.execute(
        "SELECT COUNT(*) FROM duckdb_functions() "
        "WHERE function_name='blackhole' AND function_type='aggregate'"
    ).fetchone()[0]
    if not n:
        raise RuntimeError(
            "the imported `duckdb` module does not expose the `blackhole` "
            "aggregate. Build the Python bindings against this source tree "
            "(see ../duckdb-python) so the local patch is baked in."
        )


# ---------------------------------------------------------------------------
# profiling
# ---------------------------------------------------------------------------

def _flatten_operators(node: dict, depth: int = 0, out: Optional[List[dict]] = None) -> List[dict]:
    """Walk the profile JSON tree and return a flat list of operator dicts."""
    if out is None:
        out = []
    if "operator_name" in node or "operator_type" in node:
        out.append({
            "depth": depth,
            "name": node.get("operator_name") or node.get("operator_type") or "?",
            "type": node.get("operator_type"),
            "timing_s": float(node.get("operator_timing", 0.0) or 0.0),
            "cardinality": int(node.get("operator_cardinality", 0) or 0),
            "rows_scanned": int(node.get("operator_rows_scanned", 0) or 0),
            "extra_info": node.get("extra_info") or {},
        })
    for child in node.get("children", []) or []:
        _flatten_operators(child, depth + 1, out)
    return out


def _query_level_stats(prof: dict) -> dict:
    """Extract the top-level query metrics from the profile JSON."""
    return {
        "latency_s":                float(prof.get("latency", 0.0) or 0.0),
        "cpu_time_s":               float(prof.get("cpu_time", 0.0) or 0.0),
        "blocked_thread_time_s":    float(prof.get("blocked_thread_time", 0.0) or 0.0),
        "cumulative_cardinality":   int(prof.get("cumulative_cardinality", 0) or 0),
        "cumulative_rows_scanned":  int(prof.get("cumulative_rows_scanned", 0) or 0),
        "total_bytes_read":         int(prof.get("total_bytes_read", 0) or 0),
        "total_bytes_written":      int(prof.get("total_bytes_written", 0) or 0),
        "peak_buffer_memory":       int(prof.get("system_peak_buffer_memory", 0) or 0),
        "peak_temp_dir_size":       int(prof.get("system_peak_temp_dir_size", 0) or 0),
    }


def run_profiled(con, sql: str, profile_path: Path) -> Tuple[float, dict, List[dict]]:
    """Run the blackhole-wrapped `sql` and capture DuckDB's JSON profile.
    Returns (wall_ms, query_level_stats, operators)."""
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute("PRAGMA enable_profiling='json'")
    con.execute("PRAGMA profiling_mode='detailed'")
    con.execute(f"PRAGMA profiling_output='{profile_path}'")

    wrapped = _blackhole_wrap(sql)
    t0 = time.perf_counter()
    con.execute(wrapped).fetchall()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    con.execute("PRAGMA disable_profiling")

    with open(profile_path) as f:
        prof = json.load(f)
    return wall_ms, _query_level_stats(prof), _flatten_operators(prof)


def bench(con, label: str, sql: str, repeats: int, perf_dir: Path) -> dict:
    """Warmup + `repeats` measured runs. Profile is captured on the fastest run."""
    perf_dir.mkdir(parents=True, exist_ok=True)
    run_profiled(con, sql, perf_dir / f"{label}.warmup.json")

    samples = []
    for i in range(repeats):
        pp = perf_dir / f"{label}.run{i}.json"
        wall_ms, stats, ops = run_profiled(con, sql, pp)
        samples.append((wall_ms, stats, ops, pp))

    samples.sort(key=lambda s: s[0])
    _, best_stats, best_ops, best_path = samples[0]
    times = [s[0] for s in samples]
    return {
        "label":        label,
        "wall_min_ms":  min(times),
        "wall_med_ms":  statistics.median(times),
        "wall_max_ms":  max(times),
        "stats":        best_stats,
        "operators":    best_ops,
        "profile_path": str(best_path),
    }


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------

def _human_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:7.2f} {unit}"
        n /= 1024.0
    return f"{n:7.2f} PiB"


def _human_count(n: int) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n/div:7.2f}{unit}"
    return f"{n:>8d}"


def print_query_perf(result: dict, threads: int) -> None:
    s = result["stats"]
    lat_ms = s["latency_s"] * 1000.0
    cpu_ms = s["cpu_time_s"] * 1000.0
    blocked_ms = s["blocked_thread_time_s"] * 1000.0

    print(f"\n[perf ] {result['label']}")
    print(f"        wall:     min={result['wall_min_ms']:9.1f} ms   "
          f"median={result['wall_med_ms']:9.1f} ms   "
          f"max={result['wall_max_ms']:9.1f} ms")
    print(f"        latency:  {lat_ms:9.1f} ms   cpu_time: {cpu_ms:9.1f} ms   "
          f"blocked: {blocked_ms:9.1f} ms")
    if threads and lat_ms > 0:
        eff = (cpu_ms / threads) / lat_ms * 100.0
        print(f"        parallel_efficiency = cpu_time / (threads * latency) = {eff:5.1f}% "
              f"(threads={threads})")
    print(f"        peak_mem: {_human_bytes(s['peak_buffer_memory'])}   "
          f"peak_tempdir: {_human_bytes(s['peak_temp_dir_size'])}")
    print(f"        io_read:  {_human_bytes(s['total_bytes_read'])}   "
          f"io_write:     {_human_bytes(s['total_bytes_written'])}")
    print(f"        rows_scanned_cumul: {_human_count(s['cumulative_rows_scanned'])}   "
          f"cardinality_cumul: {_human_count(s['cumulative_cardinality'])}")

    ops = sorted(result["operators"], key=lambda o: -o["timing_s"])[:8]
    if ops:
        print(f"        {'top operators by time':<44}{'time(ms)':>11}{'card':>10}{'scan_rows':>12}")
        for op in ops:
            t_ms = op["timing_s"] * 1000.0
            if t_ms < 0.1:
                continue
            name = op["name"]
            if op["extra_info"]:
                hint_parts = []
                for key in ("Join Type", "Conditions", "Aggregates", "Function"):
                    v = op["extra_info"].get(key)
                    if v:
                        hint_parts.append(f"{key}={str(v)[:30]}")
                if hint_parts:
                    name = f"{name}[{', '.join(hint_parts)}]"
            print(f"          {name:<44}{t_ms:>9.1f}{_human_count(op['cardinality']):>12}"
                  f"{_human_count(op['rows_scanned']):>12}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--rows", type=int, default=1_000_000, help="rows per parquet file")
    p.add_argument("--overlap", type=float, default=0.5,
                   help="fraction of keys shared by all 5 files (0..1)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--threads", type=int, default=0, help="0 = duckdb default")
    p.add_argument("--memory-limit", default=None, help="e.g. '8GB' (DuckDB default if unset)")
    p.add_argument("--perf-dir", default=None,
                   help="directory to write per-run JSON profiles (default: <data_dir>/perf)")
    p.add_argument("--summary-json", default=None,
                   help="optional path to write a one-file JSON summary of all results")
    p.add_argument("--explain", action="store_true",
                   help="print EXPLAIN ANALYZE tree for the join")
    p.add_argument("--keep", action="store_true", help="do not delete the temp parquet dir")
    p.add_argument("--dir", default=None,
                   help="reuse / share an existing parquet dir (skips generation if "
                        "t0..t4.parquet already exist). Pair with `plan_bench --dir <same>` "
                        "for apples-to-apples comparison against the C++ bench.")
    args = p.parse_args()

    assert 0.0 <= args.overlap <= 1.0
    if args.dir:
        tmp = Path(args.dir)
        tmp.mkdir(parents=True, exist_ok=True)
        owns_dir = False
    else:
        tmp = Path(tempfile.mkdtemp(prefix="bench_fullouter5_"))
        owns_dir = True
    perf_dir = Path(args.perf_dir) if args.perf_dir else (tmp / "perf")
    perf_dir.mkdir(parents=True, exist_ok=True)

    try:
        con = duckdb.connect()
        _require_blackhole(con)
        if args.threads:
            con.execute(f"SET threads = {args.threads}")
        if args.memory_limit:
            con.execute(f"SET memory_limit = '{args.memory_limit}'")
        # At large scale the materialize-join COPY step blows up temp space when
        # it has to preserve insertion order. We don't care about row order here.
        con.execute("SET preserve_insertion_order = false")

        eff_threads = con.execute("SELECT current_setting('threads')::INTEGER").fetchone()[0]
        eff_memlim = con.execute("SELECT current_setting('memory_limit')").fetchone()[0]
        eff_tempdir = con.execute("SELECT current_setting('temp_directory')").fetchone()[0]

        print(f"[setup] rows/file={args.rows:,}  overlap={args.overlap}")
        print(f"        wrapper: SELECT blackhole(*COLUMNS(*)) FROM (...) _t")
        print(f"        threads={eff_threads}  memory_limit={eff_memlim}  "
              f"temp_dir={eff_tempdir or '<default>'}")
        print(f"        data_dir={tmp}")
        print(f"        perf_dir={perf_dir}")

        t0 = time.perf_counter()
        existing = sorted(tmp.glob("t*.parquet"))
        if len(existing) >= N_FILES:
            print(f"[setup] reusing {len(existing)} input parquet file(s) in {tmp}")
        else:
            gen_parquet(con, tmp, args.rows, args.overlap)
            print(f"[setup] generated inputs in {time.perf_counter() - t0:.2f}s")
        input_bytes = 0
        for pq in sorted(tmp.glob("t*.parquet")):
            sz = pq.stat().st_size
            input_bytes += sz
            print(f"        {pq.name:20s} {sz/1e6:8.2f} MB")
        print(f"        {'total inputs':20s} {input_bytes/1e6:8.2f} MB")

        if args.explain:
            print("\n[plan] EXPLAIN ANALYZE 5-way FULL OUTER JOIN:\n")
            print(con.execute("EXPLAIN ANALYZE " + _blackhole_wrap(join_query(tmp))).fetchone()[1])

        # Run join benchmark first so spill has maximum free disk.
        join_res = bench(con, "fullouter_join", join_query(tmp), args.repeats, perf_dir)
        print_query_perf(join_res, eff_threads)

        t1 = time.perf_counter()
        joined_path = tmp / "joined.parquet"
        if joined_path.exists():
            print(f"\n[setup] reusing joined.parquet in {tmp}")
        else:
            materialize_join(con, tmp)
            print(f"\n[setup] materialized joined.parquet in {time.perf_counter() - t1:.2f}s")
        joined_bytes = joined_path.stat().st_size
        print(f"        joined.parquet {joined_bytes/1e6:.2f} MB")

        scan_res = bench(con, "scan_materialized", scan_query(tmp), args.repeats, perf_dir)
        print_query_perf(scan_res, eff_threads)

        print(f"\n[timing] {'query':<32}{'wall_min_ms':>14}{'wall_med_ms':>14}")
        print(f"         {'5-way FULL OUTER JOIN':<32}"
              f"{join_res['wall_min_ms']:>14.1f}{join_res['wall_med_ms']:>14.1f}")
        print(f"         {'SCAN materialized parquet':<32}"
              f"{scan_res['wall_min_ms']:>14.1f}{scan_res['wall_med_ms']:>14.1f}")
        ratio = join_res['wall_min_ms'] / scan_res['wall_min_ms']
        print(f"[ratio ] join / scan = {ratio:.2f}x  (join overhead on top of scan)")

        if args.summary_json:
            summary = {
                "args": vars(args),
                "effective": {
                    "threads": eff_threads,
                    "memory_limit": eff_memlim,
                    "temp_directory": eff_tempdir,
                },
                "input_bytes": input_bytes,
                "joined_bytes": joined_bytes,
                "results": [join_res, scan_res],
                "ratio_join_over_scan": ratio,
            }
            Path(args.summary_json).write_text(json.dumps(summary, indent=2, default=str))
            print(f"[perf ] summary written to {args.summary_json}")
    finally:
        if not args.keep and owns_dir:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
