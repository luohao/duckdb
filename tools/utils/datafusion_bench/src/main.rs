// DataFusion FULL OUTER JOIN benchmark — mirror of kway_bench's hash_join
// variant, with the DuckDB plan_bench.cpp trick: build the physical plan once,
// re-execute it in a hot loop. That isolates hash-join execution time from
// SQL parsing / logical planning / optimization.
//
// The plan is constructed via the DataFrame API (not SQL) so we control
// column aliasing and avoid SQL-parser quirks that affect DataFusion's
// multi-way USING/NATURAL JOIN handling.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::physical_plan::display::DisplayableExecutionPlan;
use datafusion::physical_plan::execute_stream;
use datafusion::prelude::*;
use datafusion_functions::core::expr_fn::coalesce;
use futures::TryStreamExt;

#[derive(Parser, Debug)]
#[command(about = "DataFusion vs DuckDB FULL OUTER JOIN bench (plan-reuse variant)")]
struct Args {
    #[arg(long)]
    dir: String,
    #[arg(long, default_value = "2,5,6")]
    ks: String,
    #[arg(long, default_value = "0.0,0.5,1.0")]
    overlaps: String,
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    #[arg(long, default_value_t = 4)]
    threads: usize,
    #[arg(long, default_value_t = false)]
    print_plan: bool,
}

const DATA_COLS: &[&str] = &[
    "c_int", "c_bigint", "c_dbl", "c_flt", "c_short", "c_long", "c_date", "c_ts", "c_bool", "c_dec",
];

fn config_dir(base: &str, k: i32, overlap: f64) -> PathBuf {
    let pct = (overlap * 100.0).round() as i32;
    PathBuf::from(format!("{}/k{}_o{:03}", base, k, pct))
}

fn parse_csv<T: std::str::FromStr>(s: &str) -> Result<Vec<T>>
where
    T::Err: std::fmt::Display,
{
    s.split(',')
        .filter(|p| !p.is_empty())
        .map(|p| p.trim().parse::<T>().map_err(|e| anyhow!("parse '{}': {}", p, e)))
        .collect()
}

// Build a k-way FULL OUTER JOIN DataFrame. Caller clones + lowers to a
// physical plan per iteration (DataFusion's RepartitionExec is single-use,
// so a cached Arc<dyn ExecutionPlan> can't be re-executed). Physical planning
// is cheap compared to join execution — this still isolates SQL parse/bind
// overhead.
//
// Shape mirrors DuckDB's chained `USING(k)`:
// - t0 aliased as (k, c_int_0, c_bigint_0, ...)
// - t_i aliased as (k_i, c_int_i, ...) so the FOJ key names don't collide
// - After each FULL JOIN, a ProjectionExec introduces `k = COALESCE(k, k_i)`
//   and drops `k_i`. This gives us a single carrying key column for the next join.
async fn build_df(ctx: &SessionContext, dir: &Path, k: i32) -> Result<DataFrame> {
    // Load + alias t0.
    let t0_path = dir.join("t0.parquet");
    let df0 = ctx
        .read_parquet(t0_path.to_str().unwrap().to_string(), ParquetReadOptions::default())
        .await?;
    let mut cur = df0.select(
        std::iter::once(col("k"))
            .chain(DATA_COLS.iter().map(|c| col(*c).alias(format!("{}_0", c))))
            .collect::<Vec<_>>(),
    )?;

    for i in 1..k {
        let path = dir.join(format!("t{}.parquet", i));
        let dfi = ctx
            .read_parquet(path.to_str().unwrap().to_string(), ParquetReadOptions::default())
            .await?;
        // Alias right side so no column names collide with cur.
        let k_i = format!("k_{}", i);
        let dfi = dfi.select(
            std::iter::once(col("k").alias(&k_i))
                .chain(DATA_COLS.iter().map(|c| col(*c).alias(format!("{}_{}", c, i))))
                .collect::<Vec<_>>(),
        )?;
        // FULL OUTER JOIN on the carrying key `k` = renamed right key.
        cur = cur.join(dfi, JoinType::Full, &["k"], &[&k_i], None)?;
        // Project: collapse k and k_i to a single k via COALESCE; keep all data cols.
        let mut proj: Vec<Expr> =
            vec![coalesce(vec![col("k"), col(&k_i)]).alias("k")];
        for j in 0..=i {
            for c in DATA_COLS {
                proj.push(col(&format!("{}_{}", c, j)));
            }
        }
        cur = cur.select(proj)?;
    }

    Ok(cur)
}

async fn build_scan_df(ctx: &SessionContext, dir: &Path) -> Result<DataFrame> {
    let path = dir.join("joined.parquet");
    if !path.exists() {
        return Err(anyhow!("missing {}", path.display()));
    }
    Ok(ctx
        .read_parquet(path.to_str().unwrap().to_string(), ParquetReadOptions::default())
        .await?)
}

// Time a single execution. The `df.clone()` clones the LogicalPlan (cheap),
// then physical planning + execution happen inside the timed region. This is
// still cheaper than the full SQL pipeline (parser + logical optimizer).
async fn run_df_once(df: DataFrame) -> Result<(f64, usize)> {
    let task_ctx = Arc::new(df.task_ctx());
    let t0 = Instant::now();
    let plan = df.create_physical_plan().await?;
    let mut stream = execute_stream(plan, task_ctx)?;
    let mut rows = 0usize;
    while let Some(batch) = stream.try_next().await? {
        rows += batch.num_rows();
    }
    Ok((t0.elapsed().as_secs_f64() * 1000.0, rows))
}

fn mk_ctx(threads: usize) -> SessionContext {
    let rt = RuntimeEnvBuilder::new().build_arc().unwrap();
    let cfg = SessionConfig::new()
        .with_target_partitions(threads)
        .with_batch_size(8192);
    SessionContext::new_with_config_rt(cfg, rt)
}

async fn run_config(args: &Args, k: i32, overlap: f64) -> Result<(f64, f64, usize)> {
    let dir = config_dir(&args.dir, k, overlap);
    if !dir.exists() {
        return Err(anyhow!("missing {}", dir.display()));
    }

    // scan baseline
    let scan_min = {
        let ctx = mk_ctx(args.threads);
        let df = build_scan_df(&ctx, &dir).await?;
        let _ = run_df_once(df.clone()).await?; // warmup
        let mut samples = Vec::with_capacity(args.repeats);
        for _ in 0..args.repeats {
            let (ms, _) = run_df_once(df.clone()).await?;
            samples.push(ms);
        }
        samples.into_iter().fold(f64::INFINITY, f64::min)
    };

    // hash_join
    let (hash_min, rows_out) = {
        let ctx = mk_ctx(args.threads);
        let df = build_df(&ctx, &dir, k).await?;
        if args.print_plan {
            let plan = df.clone().create_physical_plan().await?;
            println!(
                "\n=== plan for k={} overlap={:.2} ===\n{}\n",
                k,
                overlap,
                DisplayableExecutionPlan::new(plan.as_ref()).indent(false)
            );
        }
        let (_, rows) = run_df_once(df.clone()).await?; // warmup
        let mut samples = Vec::with_capacity(args.repeats);
        for _ in 0..args.repeats {
            let (ms, _) = run_df_once(df.clone()).await?;
            samples.push(ms);
        }
        (samples.into_iter().fold(f64::INFINITY, f64::min), rows)
    };

    Ok((scan_min, hash_min, rows_out))
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();
    let ks: Vec<i32> = parse_csv(&args.ks)?;
    let overlaps: Vec<f64> = parse_csv(&args.overlaps)?;
    fs::metadata(&args.dir).with_context(|| format!("--dir {} not readable", args.dir))?;

    println!(
        "[datafusion_bench] plan-reuse mode, threads={} repeats={} dir={}",
        args.threads, args.repeats, args.dir
    );
    println!("                   ks={:?} overlaps={:?}\n", ks, overlaps);

    println!(
        "{:>4} {:>7} {:>10} {:>11} {:>11} {:>11}",
        "k", "overlap", "rows_out", "scan(ms)", "hash(ms)", "hash/scan"
    );
    println!("---- ------- ---------- ----------- ----------- -----------");

    for &k in &ks {
        for &o in &overlaps {
            let (scan_ms, hash_ms, rows) = run_config(&args, k, o).await?;
            let ratio = if scan_ms > 0.0 { hash_ms / scan_ms } else { 0.0 };
            println!(
                "{:>4} {:>7.2} {:>10} {:>9.1}ms {:>9.1}ms {:>10.2}x",
                k, o, rows, scan_ms, hash_ms, ratio
            );
        }
    }
    Ok(())
}
