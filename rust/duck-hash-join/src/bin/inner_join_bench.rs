// Inner-join single-threaded baseline bench.
//
// Reads two parquet files from a kway_bench config dir (t0.parquet build side,
// t1.parquet probe side), joins on `k`, counts matches. Reports median wall
// time over N repeats. Compare to DuckDB's INNER JOIN on the same files for
// the first perf data point.

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use duck_hash_join::ht::JoinHashTable;
use duck_hash_join::io::{
    bench_layout, keys_only_layout, read_parquet_into_ht, read_parquet_into_ht_parallel,
    read_parquet_keys_for_probe, read_parquet_keys_into_ht,
    read_parquet_keys_into_ht_parallel,
};

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn main() {
    let mut args = env::args().skip(1);
    let dir = args
        .next()
        .expect("usage: inner_join_bench <config_dir> [repeats] [--keys-only] [--parallel] [--threads=N]");
    let repeats: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(3);
    // Remaining flags. --keys-only runs the apples-to-apples comparison with
    // DuckDB's `count(*)` query: only the key column is ingested, arena rows
    // are 16 bytes (header only, since key is i32 inside a i32-only row).
    let keys_only = env::args().any(|a| a == "--keys-only");
    let parallel = env::args().any(|a| a == "--parallel");
    let foj = env::args().any(|a| a == "--foj");
    let threads: Option<usize> = env::args()
        .find_map(|a| a.strip_prefix("--threads=").map(|s| s.parse().unwrap()));
    let dir = PathBuf::from(dir);

    if let Some(t) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .expect("build rayon pool");
    }

    let build_path = dir.join("t0.parquet");
    let probe_path = dir.join("t1.parquet");

    println!(
        "[inner_join_bench] build={:?} probe={:?} repeats={} keys_only={} parallel={} rayon_threads={}",
        build_path,
        probe_path,
        repeats,
        keys_only,
        parallel,
        rayon::current_num_threads()
    );

    let mut build_read_ms = Vec::with_capacity(repeats);
    let mut finalize_ms = Vec::with_capacity(repeats);
    let mut probe_read_ms = Vec::with_capacity(repeats);
    let mut probe_ms = Vec::with_capacity(repeats);
    let mut total_ms = Vec::with_capacity(repeats);
    let mut match_count_report: usize = 0;

    for rep in 0..=repeats {
        let t0 = Instant::now();

        // Build phase: read t0.parquet into the HT.
        let layout = if keys_only { keys_only_layout() } else { bench_layout() };
        let mut ht = JoinHashTable::new(layout);
        // Reserve an over-estimate; parquet row count is known from metadata
        // after reading, so over-reserve to avoid realloc invalidating ptrs.
        ht.reserve(1_500_000);
        let build_rows = if keys_only {
            if parallel {
                read_parquet_keys_into_ht_parallel(&build_path, &mut ht).unwrap()
            } else {
                read_parquet_keys_into_ht(&build_path, &mut ht).unwrap()
            }
        } else if parallel {
            read_parquet_into_ht_parallel(&build_path, &mut ht).unwrap()
        } else {
            read_parquet_into_ht(&build_path, &mut ht).unwrap()
        };
        let t_build_read = Instant::now();
        if parallel {
            ht.finalize_parallel();
        } else {
            ht.finalize();
        }
        // Sanity check: the parallel path must not lose rows. The probe
        // below uses the directory; no direct count, so we rely on matches.
        let t_finalize = Instant::now();

        // Probe phase: read t1.parquet keys, probe HT, count matches.
        let (probe_keys, probe_hashes) = read_parquet_keys_for_probe(&probe_path).unwrap();
        let t_probe_read = Instant::now();

        let match_count: usize = if foj {
            // FOJ row count = matches + probe_unmatched + build_unmatched.
            let (m, pu, bu) =
                ht.probe_i32_full_outer_count(&probe_hashes, &probe_keys, 16384);
            m + pu + bu
        } else if parallel {
            ht.probe_i32_inner_parallel(
                &probe_hashes,
                &probe_keys,
                16384,
                || 0usize,
                |s, _row, _idx| *s += 1,
                |a, b| a + b,
            )
        } else {
            let mut mc: usize = 0;
            ht.probe_i32_inner(&probe_hashes, &probe_keys, |_row, _idx| mc += 1);
            mc
        };
        let t_probe_done = Instant::now();

        let total = t_probe_done.duration_since(t0).as_secs_f64() * 1000.0;
        let br = t_build_read.duration_since(t0).as_secs_f64() * 1000.0;
        let fn_ = t_finalize.duration_since(t_build_read).as_secs_f64() * 1000.0;
        let pr = t_probe_read.duration_since(t_finalize).as_secs_f64() * 1000.0;
        let pb = t_probe_done.duration_since(t_probe_read).as_secs_f64() * 1000.0;

        if rep == 0 {
            println!(
                "[warmup] build={} probe={} matches={} total={:.1}ms",
                build_rows,
                probe_keys.len(),
                match_count,
                total
            );
            continue;
        }
        build_read_ms.push(br);
        finalize_ms.push(fn_);
        probe_read_ms.push(pr);
        probe_ms.push(pb);
        total_ms.push(total);
        match_count_report = match_count;
        println!(
            "[rep {}] build_read={:.1}  finalize={:.1}  probe_read={:.1}  probe={:.1}  total={:.1}  matches={}",
            rep, br, fn_, pr, pb, total, match_count
        );
    }

    println!();
    println!("[median over {} reps]", repeats);
    println!("  build-side parquet read : {:>7.1} ms", median(&mut build_read_ms));
    println!("  directory finalize      : {:>7.1} ms", median(&mut finalize_ms));
    println!("  probe-side parquet read : {:>7.1} ms", median(&mut probe_read_ms));
    println!("  probe + match emit      : {:>7.1} ms", median(&mut probe_ms));
    println!("  total                   : {:>7.1} ms", median(&mut total_ms));
    println!("  matches                 : {}", match_count_report);
}
