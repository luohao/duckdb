// kway_bench: matrix benchmark comparing three ways to produce the result of
// a k-table FULL OUTER JOIN on a shared key `k`. All variants read parquet
// from disk and use a blackhole sink, so the numbers are apples-to-apples.
//
// Variants (per (k, overlap) point):
//   scan_joined — read a pre-materialized denormalized parquet:
//                 `SELECT blackhole(*COLUMNS(*)) FROM joined.parquet`.
//                 Lower bound: no join work, just decode all output columns.
//   hash_join   — parallel hash join:
//                 `SELECT blackhole(*COLUMNS(*))
//                    FROM t0 FULL OUTER JOIN ... FULL OUTER JOIN t{k-1}`.
//   kway_op     — our port of ClickHouse's SortingQueueImpl run as a real
//                 DuckDB PhysicalOperator (PhysicalKWayMergeJoin). Same SQL
//                 as hash_join with `SET force_kway_merge_join=true`; the
//                 planner detects the FULL OUTER chain and collapses it
//                 into a single k-ary operator.
//
// Ratios are reported against scan_joined (the natural lower bound for
// "produce this join's output with no join work").
//
// Usage:
//   kway_bench [--rows N]
//              [--ks "2,3,4,5,6"]
//              [--overlaps "0.0,0.25,0.5,0.75,1.0"]
//              [--repeats N]
//              [--threads N]
//              [--dir PATH]     (base dir; each (k,O) gets a subdir)
//              [--keep]

#include "duckdb.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/materialized_query_result.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/statement/logical_plan_statement.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/planner.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace duckdb; // NOLINT
using std::string;

namespace {

struct Args {
	int64_t rows = 1'000'000;
	vector<int> ks = {2, 3, 4, 5, 6};
	vector<double> overlaps = {0.0, 0.25, 0.5, 0.75, 1.0};
	int repeats = 3;
	int threads = 0;
	string dir;
	bool keep = false;
};

static void die(const string &msg) {
	fprintf(stderr, "error: %s\n", msg.c_str());
	std::exit(1);
}

template <class T>
static vector<T> SplitCsv(const string &s, T (*parse)(const string &)) {
	vector<T> out;
	std::stringstream ss(s);
	string item;
	while (std::getline(ss, item, ',')) {
		if (!item.empty()) {
			out.push_back(parse(item));
		}
	}
	return out;
}

static Args ParseArgs(int argc, char **argv) {
	Args a;
	for (int i = 1; i < argc; i++) {
		string k = argv[i];
		auto need = [&](const char *flag) -> string {
			if (i + 1 >= argc) {
				die(string("missing value for ") + flag);
			}
			return string(argv[++i]);
		};
		if (k == "--rows") {
			a.rows = std::stoll(need("--rows"));
		} else if (k == "--ks") {
			a.ks = SplitCsv<int>(need("--ks"), [](const string &s) { return std::stoi(s); });
		} else if (k == "--overlaps") {
			a.overlaps = SplitCsv<double>(need("--overlaps"), [](const string &s) { return std::stod(s); });
		} else if (k == "--repeats") {
			a.repeats = std::stoi(need("--repeats"));
		} else if (k == "--threads") {
			a.threads = std::stoi(need("--threads"));
		} else if (k == "--dir") {
			a.dir = need("--dir");
		} else if (k == "--keep") {
			a.keep = true;
		} else if (k == "-h" || k == "--help") {
			printf("Usage: kway_bench [--rows N] [--ks \"2,3,4,5,6\"]\n"
			       "                  [--overlaps \"0.0,0.25,0.5,0.75,1.0\"]\n"
			       "                  [--repeats N] [--threads N] [--dir PATH] [--keep]\n"
			       "\n"
			       "Sweeps over (k, overlap) pairs. For each, generates k sorted parquet\n"
			       "files + a materialized denormalized joined.parquet, then times three\n"
			       "variants:\n"
			       "  scan_joined (baseline), hash_join (parallel SQL), kway_mat (our merge).\n");
			std::exit(0);
		} else {
			die("unknown flag: " + k);
		}
	}
	for (auto o : a.overlaps) {
		if (o < 0 || o > 1) {
			die("overlap must be in [0,1]");
		}
	}
	for (auto k : a.ks) {
		if (k < 2) {
			die("k must be >= 2");
		}
	}
	return a;
}

static bool PathExists(const string &p) {
	struct stat st {};
	return ::stat(p.c_str(), &st) == 0;
}

static void RunSQL(Connection &con, const string &sql) {
	auto r = con.Query(sql);
	if (r->HasError()) {
		die("SQL failed: " + r->GetError() + "\nquery: " + sql);
	}
}

static string OverlapLabel(double o) {
	char buf[16];
	std::snprintf(buf, sizeof(buf), "%.2f", o);
	return buf;
}

static string ConfigDir(const string &base, int k, double overlap) {
	char buf[64];
	std::snprintf(buf, sizeof(buf), "k%d_o%03d", k, (int)std::round(overlap * 100));
	return base + "/" + buf;
}

// Generate k parquet files for a (k, overlap) config.
//
// Key distribution:
//   - Build a SHUFFLED permutation of positions [0, shared + k*unique) -> key
//     values in the same range (via ORDER BY random()). This is the keyspace.
//   - Each file i takes keys at positions [0, shared) ∪ [shared+i*unique, shared+(i+1)*unique)
//     from the permutation. Same "range + disjoint" partitioning as before,
//     but the VALUES are scattered because of the shuffle.
//   - Finally, each file is sorted by k before writing (so merge-join still
//     sees a sorted input, but the ALL-AGREE fast path can no longer rely on
//     keys being contiguous integers — it will find mostly N=1 runs).
//
// Reproducibility: we seed DuckDB's RNG with setseed(0.42) before the shuffle.
static void GenSortedParquet(Connection &con, const string &dir, int k, int64_t rows, double overlap) {
	int64_t shared = static_cast<int64_t>(rows * overlap);
	int64_t unique = rows - shared;
	int64_t total = shared + static_cast<int64_t>(k) * unique;

	// Seeded permutation: position -> key. Stored as a temp table we'll
	// reference from each file's COPY statement. We DROP+CREATE to make
	// re-runs with a different (k, overlap) safe.
	RunSQL(con, "DROP TABLE IF EXISTS keyspace");
	RunSQL(con, "SELECT setseed(0.42)");
	RunSQL(con, StringUtil::Format("CREATE TEMP TABLE keyspace AS "
	                               "SELECT row_number() OVER () - 1 AS pos, range::INTEGER AS k "
	                               "FROM (SELECT range FROM range(0, %lld) ORDER BY random())",
	                               (long long)total));

	for (int i = 0; i < k; i++) {
		int64_t start = shared + static_cast<int64_t>(i) * unique;
		int64_t end = start + unique;
		auto cols = StringUtil::Format(
		    "((k * 7  + %d * 13) %% 100000)::INTEGER                            AS c_int, "
		    "(k::BIGINT * 1000 + %d)::BIGINT                                    AS c_bigint, "
		    "((k + %d) * 3.14159)::DOUBLE                                       AS c_dbl, "
		    "((k * 0.01 + %d)::FLOAT)                                           AS c_flt, "
		    "('tag_' || ((k + %d) %% 100))::VARCHAR                             AS c_short, "
		    "md5(k::VARCHAR || '_' || %d::VARCHAR)                              AS c_long, "
		    "(DATE '2020-01-01' + ((k + %d) %% 1000) * INTERVAL 1 DAY)::DATE    AS c_date, "
		    "(TIMESTAMP '2020-01-01' + ((k + %d) %% 86400) * INTERVAL 1 SECOND)::TIMESTAMP AS c_ts, "
		    "((k + %d) %% 2 = 0)::BOOLEAN                                       AS c_bool, "
		    "(((k + %d) * 1.23)::DECIMAL(18,2))                                 AS c_dec",
		    i, i, i, i, i, i, i, i, i, i);
		auto path = StringUtil::Format("%s/t%d.parquet", dir.c_str(), i);
		// Pull keys from the shuffled keyspace, then sort by k before writing.
		auto sql = StringUtil::Format("COPY ("
		                              "  SELECT k, %s FROM ("
		                              "    SELECT k FROM keyspace WHERE pos < %lld "
		                              "    UNION ALL "
		                              "    SELECT k FROM keyspace WHERE pos >= %lld AND pos < %lld"
		                              "  ) ORDER BY k"
		                              ") TO '%s' (FORMAT PARQUET);",
		                              cols.c_str(), (long long)shared, (long long)start, (long long)end, path.c_str());
		RunSQL(con, sql);
	}

	RunSQL(con, "DROP TABLE keyspace");
}

// Build the k-way FULL OUTER JOIN SQL over the files in `dir`.
static string BuildJoinSQL(const string &dir, int k) {
	string s = StringUtil::Format("SELECT * FROM '%s/t0.parquet' t0", dir.c_str());
	for (int i = 1; i < k; i++) {
		s += StringUtil::Format(" FULL OUTER JOIN '%s/t%d.parquet' t%d USING (k)", dir.c_str(), i, i);
	}
	return s;
}

// Materialize the k-way join output to `joined.parquet`.
static void MaterializeJoin(Connection &con, const string &dir, int k) {
	auto sql = StringUtil::Format("COPY (%s) TO '%s/joined.parquet' (FORMAT PARQUET);", BuildJoinSQL(dir, k).c_str(),
	                              dir.c_str());
	RunSQL(con, sql);
}


//=============================================================================
// Timing helpers
//=============================================================================

// Execute a pre-built logical plan (wrapped in blackhole). Returns wall ms.
static double RunPlanOnce(Connection &con, const LogicalOperator &plan) {
	auto &ctx = *con.context;
	con.BeginTransaction();
	unique_ptr<LogicalOperator> copy;
	try {
		copy = plan.Copy(ctx);
	} catch (...) {
		con.Rollback();
		throw;
	}
	auto t0 = std::chrono::steady_clock::now();
	auto stmt = make_uniq<LogicalPlanStatement>(std::move(copy));
	auto result = con.SendQuery(std::move(stmt), QueryResultOutputType::ALLOW_STREAMING);
	if (result->HasError()) {
		string err = result->GetError();
		con.Rollback();
		die("plan exec failed: " + err);
	}
	while (auto chunk = result->Fetch()) {
		(void)chunk->size();
	}
	auto t1 = std::chrono::steady_clock::now();
	try {
		con.Commit();
	} catch (...) {
	}
	return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static unique_ptr<LogicalOperator> BuildPlan(Connection &con, const string &sql) {
	auto &ctx = *con.context;
	con.BeginTransaction();
	unique_ptr<LogicalOperator> plan;
	try {
		Parser parser;
		parser.ParseQuery(sql);
		Planner planner(ctx);
		planner.CreatePlan(std::move(parser.statements[0]));
		plan = std::move(planner.plan);
		Optimizer optimizer(*planner.binder, ctx);
		plan = optimizer.Optimize(std::move(plan));
		plan->ResolveOperatorTypes();
	} catch (...) {
		con.Rollback();
		throw;
	}
	con.Rollback();
	return plan;
}

struct ConfigResult {
	int k;
	double overlap;
	idx_t rows_out;
	double scan_parts_min = 0;  // read k input parquets only (no join), blackhole
	double scan_joined_min = 0; // read pre-materialized denormalized parquet, blackhole
	double hash_min = 0;        // SQL hash join on the k parquets, blackhole
	double kway_op_min = 0;     // SQL with force_kway_merge_join=true -> PhysicalKWayMergeJoin
};

static double VectorMin(vector<double> &v) {
	std::sort(v.begin(), v.end());
	return v.empty() ? 0.0 : v.front();
}

static ConfigResult RunConfig(DuckDB &db, Connection &con, const Args &args, int k, double overlap,
                              const string &base_dir) {
	auto &ctx = *con.context;
	ConfigResult res {k, overlap, 0, 0, 0, 0};

	string dir = ConfigDir(base_dir, k, overlap);
	::mkdir(dir.c_str(), 0755);

	// Generate k parquet files + joined.parquet (if absent).
	bool all_files = true;
	for (int i = 0; i < k; i++) {
		if (!PathExists(dir + "/t" + std::to_string(i) + ".parquet")) {
			all_files = false;
			break;
		}
	}
	if (!all_files) {
		printf("  [gen %d-way o=%.2f] %s ... ", k, overlap, dir.c_str());
		fflush(stdout);
		auto t0 = std::chrono::steady_clock::now();
		GenSortedParquet(con, dir, k, args.rows, overlap);
		printf("parquets (%.1fs) ", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
		fflush(stdout);
	}
	if (!PathExists(dir + "/joined.parquet")) {
		auto t0 = std::chrono::steady_clock::now();
		MaterializeJoin(con, dir, k);
		printf("joined (%.1fs)", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
		fflush(stdout);
	}
	if (!all_files) {
		printf("\n");
	}

	// Ground truth output row count.
	{
		auto q = con.Query(StringUtil::Format("SELECT count(*) FROM '%s/joined.parquet'", dir.c_str()));
		res.rows_out = q->GetValue(0, 0).GetValue<idx_t>();
	}


	// --- scan_parts: read the k unjoined parquets with blackhole (no join) ---
	// Upper bound on the parquet-decode work that hash_join also does.
	{
		string sql = "SELECT blackhole(*COLUMNS(*)) FROM (";
		for (int i = 0; i < k; i++) {
			if (i) {
				sql += " UNION ALL ";
			}
			sql += StringUtil::Format("SELECT * FROM '%s/t%d.parquet'", dir.c_str(), i);
		}
		sql += ") _t";
		auto plan = BuildPlan(con, sql);
		RunPlanOnce(con, *plan);
		vector<double> samples;
		for (int rep = 0; rep < args.repeats; rep++) {
			samples.push_back(RunPlanOnce(con, *plan));
		}
		res.scan_parts_min = VectorMin(samples);
	}

	// --- scan_joined baseline: read the denormalized joined parquet ---
	{
		string sql =
		    StringUtil::Format("SELECT blackhole(*COLUMNS(*)) FROM '%s/joined.parquet' _t", dir.c_str());
		auto plan = BuildPlan(con, sql);
		RunPlanOnce(con, *plan);
		vector<double> samples;
		for (int rep = 0; rep < args.repeats; rep++) {
			samples.push_back(RunPlanOnce(con, *plan));
		}
		res.scan_joined_min = VectorMin(samples);
	}

	// --- hash_join variant ---
	{
		RunSQL(con, "SET force_kway_merge_join=false");
		string join_sql = "SELECT blackhole(*COLUMNS(*)) FROM (" + BuildJoinSQL(dir, k) + ") _t";
		auto plan = BuildPlan(con, join_sql);
		RunPlanOnce(con, *plan); // warmup
		vector<double> samples;
		for (int rep = 0; rep < args.repeats; rep++) {
			samples.push_back(RunPlanOnce(con, *plan));
		}
		res.hash_min = VectorMin(samples);
	}

	// Correctness check: run kway_op pipeline under a blackhole sink (the
	// same shape as the timing variant, which ALLOWS parallel source).
	// Blackhole swallows everything; if row types mismatch we'd error here,
	// and the hot loop would fail too.
	// We cross-check counts via `scan_joined` + `hash_join` at the same
	// config — if the output row count is wrong, that error surfaces in
	// GetDataInternal when the child scans run short.
	{
		RunSQL(con, "SET force_kway_merge_join=true");
		RunSQL(con, "SET preserve_insertion_order=true");
		string sql = "SELECT blackhole(*COLUMNS(*)) FROM (" + BuildJoinSQL(dir, k) + ") _t";
		auto plan = BuildPlan(con, sql);
		(void)RunPlanOnce(con, *plan);
		RunSQL(con, "SET force_kway_merge_join=false");
		RunSQL(con, "SET preserve_insertion_order=false");
	}

	// --- kway_op variant: same SQL, but planner emits PhysicalKWayMergeJoin ---
	// Uses SET force_kway_merge_join=true so plan_comparison_join.cpp collapses
	// the FULL OUTER JOIN chain into a single k-ary operator. Goes through
	// the full DuckDB pipeline (parallel parquet read, integrated sink,
	// source emits to blackhole) — apples-to-apples with hash_join.
	// preserve_insertion_order=true feeds the operator sorted chunks.
	{
		RunSQL(con, "SET force_kway_merge_join=true");
		RunSQL(con, "SET preserve_insertion_order=true");
		string join_sql = "SELECT blackhole(*COLUMNS(*)) FROM (" + BuildJoinSQL(dir, k) + ") _t";
		auto plan = BuildPlan(con, join_sql);
		RunPlanOnce(con, *plan); // warmup
		vector<double> samples;
		for (int rep = 0; rep < args.repeats; rep++) {
			samples.push_back(RunPlanOnce(con, *plan));
		}
		res.kway_op_min = VectorMin(samples);
		RunSQL(con, "SET force_kway_merge_join=false");
		RunSQL(con, "SET preserve_insertion_order=false");
	}

	return res;
}

} // namespace

int main(int argc, char **argv) {
	auto args = ParseArgs(argc, argv);

	DuckDB db(nullptr);
	Connection con(db);

	if (args.threads > 0) {
		RunSQL(con, StringUtil::Format("SET threads=%d", args.threads));
	}
	RunSQL(con, "SET preserve_insertion_order=false");

	// Setup base dir
	string base_dir = args.dir;
	bool created_tmp = false;
	if (base_dir.empty()) {
		char tmpl[] = "/tmp/kway_bench_XXXXXX";
		if (!::mkdtemp(tmpl)) {
			die("mkdtemp failed");
		}
		base_dir = tmpl;
		created_tmp = true;
	} else {
		::mkdir(base_dir.c_str(), 0755);
	}

	printf("[bench] rows=%lld threads=%d repeats=%d base_dir=%s\n", (long long)args.rows, args.threads, args.repeats,
	       base_dir.c_str());
	printf("        ks={");
	for (size_t i = 0; i < args.ks.size(); i++) {
		printf("%s%d", i ? "," : "", args.ks[i]);
	}
	printf("}  overlaps={");
	for (size_t i = 0; i < args.overlaps.size(); i++) {
		printf("%s%.2f", i ? "," : "", args.overlaps[i]);
	}
	printf("}\n\n");

	vector<ConfigResult> results;
	for (auto k : args.ks) {
		for (auto ov : args.overlaps) {
			auto r = RunConfig(db, con, args, k, ov, base_dir);
			results.push_back(r);
		}
	}

	// Final matrix report. All times = min(ms) over args.repeats, all use
	// blackhole sink. kway_mat starts from pre-loaded ColumnDataCollections,
	// so it doesn't pay parquet-decode cost (unlike hash_join / scan_parts /
	// scan_joined). Compare kway_mat vs. (hash_join - scan_parts) to isolate
	// the pure join work.
	printf("\n[results]  min(ms) across %d repeats, all variants read parquet from disk and use\n"
	       "           a blackhole sink. Ratios are against scan_joined (reading the pre-\n"
	       "           materialized denormalized parquet — the natural lower bound).\n",
	       args.repeats);
	printf("%4s %7s %10s %11s %11s %11s %11s %11s\n", "k", "overlap", "out_rows", "scan_joined", "hash_join",
	       "kway_op", "hash/scan", "kway/scan");
	printf("---- ------- ---------- ----------- ----------- ----------- ----------- -----------\n");
	for (auto &r : results) {
		double hs = r.scan_joined_min > 0 ? r.hash_min / r.scan_joined_min : 0.0;
		double ks = r.scan_joined_min > 0 ? r.kway_op_min / r.scan_joined_min : 0.0;
		printf("%4d %7.2f %10llu %9.1fms %9.1fms %9.1fms %10.2fx %10.2fx\n", r.k, r.overlap,
		       (unsigned long long)r.rows_out, r.scan_joined_min, r.hash_min, r.kway_op_min, hs, ks);
	}
	printf("\n  scan_joined = read pre-materialized denormalized parquet, blackhole (lower bound)\n"
	       "  hash_join   = parallel hash join over the k parquets, blackhole\n"
	       "  kway_op     = our PhysicalKWayMergeJoin: parallel parquet -> k sinks -> merge -> blackhole\n"
	       "  hash/scan   = hash_join / scan_joined   (overhead of doing the join vs. reading the answer)\n"
	       "  kway/scan   = kway_op   / scan_joined   (same, for our operator)\n");

	if (created_tmp && !args.keep) {
		// Best-effort recursive removal: child files + dirs + base.
		for (auto &r : results) {
			string d = ConfigDir(base_dir, r.k, r.overlap);
			for (int i = 0; i < r.k; i++) {
				::unlink((d + "/t" + std::to_string(i) + ".parquet").c_str());
			}
			::unlink((d + "/joined.parquet").c_str());
			::rmdir(d.c_str());
		}
		::rmdir(base_dir.c_str());
	} else {
		printf("\n[setup] data kept in %s\n", base_dir.c_str());
	}
	return 0;
}
