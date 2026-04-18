// plan_bench: benchmark different logical plans directly, without going through
// SQL parsing/planning in the hot loop.
//
// Setup phase: generates 5 Parquet files matching bench_fullouter5.py.
// For each "variant", we:
//   1) parse + plan + (optionally) optimize ONCE via the planner/optimizer
//      (we do NOT run ColumnBindingResolver - that is the physical planner's job).
//   2) in the hot loop, Copy the logical plan and execute it through a
//      LogicalPlanStatement. No parser / binder runs per iteration.
//
// Every variant is wrapped with `SELECT blackhole(*COLUMNS(*)) FROM (...)`
// before planning, matching bench_fullouter5.py. The built-in `blackhole`
// aggregate forces every column to be evaluated (defeating the parquet
// COUNT(*) metadata fast path) but performs no per-row work, so we measure
// close to pure scan/join cost and a single BIGINT 0 row is produced.
//
// This intentionally mirrors tools/utils/plan_serializer.cpp, and uses the same
// plan->Copy(context) round-trip so bindings remain unresolved across runs.
//
// Usage:
//   plan_bench [--rows N] [--overlap F] [--repeats N] [--threads N]
//              [--dir PATH] [--keep] [--print-plan] [--variant NAME]
//
// If --dir is omitted we create a fresh tmp dir. If the directory already
// contains t0..t4.parquet we reuse them instead of regenerating.

#include "duckdb.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/client_context.hpp"
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
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace duckdb; // NOLINT
using std::string;
// Use duckdb::vector to avoid ambiguity with std::vector - duckdb pulls in its own
// vector type via its headers, and many duckdb APIs return duckdb::vector.

namespace {

struct Args {
	int64_t rows = 1'000'000;
	double overlap = 0.5;
	int repeats = 3;
	int threads = 0;
	string dir;
	bool keep = false;
	bool print_plan = false;
	string only_variant;
};

static void die(const string &msg) {
	fprintf(stderr, "error: %s\n", msg.c_str());
	std::exit(1);
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
		} else if (k == "--overlap") {
			a.overlap = std::stod(need("--overlap"));
		} else if (k == "--repeats") {
			a.repeats = std::stoi(need("--repeats"));
		} else if (k == "--threads") {
			a.threads = std::stoi(need("--threads"));
		} else if (k == "--dir") {
			a.dir = need("--dir");
		} else if (k == "--keep") {
			a.keep = true;
		} else if (k == "--print-plan") {
			a.print_plan = true;
		} else if (k == "--variant") {
			a.only_variant = need("--variant");
		} else if (k == "-h" || k == "--help") {
			printf("Usage: plan_bench [--rows N] [--overlap F] [--repeats N] [--threads N]\n"
			       "                  [--dir PATH] [--keep] [--print-plan] [--variant NAME]\n"
			       "\n"
			       "Every variant is wrapped with SELECT blackhole(*COLUMNS(*)) FROM (...)\n"
			       "to force column materialization with no per-row work, matching\n"
			       "bench_fullouter5.py.\n");
			std::exit(0);
		} else {
			die("unknown flag: " + k);
		}
	}
	if (a.overlap < 0 || a.overlap > 1) {
		die("--overlap must be in [0,1]");
	}
	return a;
}

static bool PathExists(const string &p) {
	struct stat st {};
	return ::stat(p.c_str(), &st) == 0;
}

static bool DirHasAllFiles(const string &dir) {
	for (int i = 0; i < 5; i++) {
		if (!PathExists(dir + "/t" + std::to_string(i) + ".parquet")) {
			return false;
		}
	}
	return true;
}

static void RunSQL(Connection &con, const string &sql) {
	auto r = con.Query(sql);
	if (r->HasError()) {
		die("SQL failed: " + r->GetError() + "\nquery: " + sql);
	}
}

static bool HasJoinedFile(const string &dir) {
	return PathExists(dir + "/joined.parquet");
}

static void MaterializeJoin(Connection &con, const string &dir) {
	auto sql = StringUtil::Format("COPY ("
	                              "  SELECT * FROM '%s/t0.parquet' t0"
	                              "  FULL OUTER JOIN '%s/t1.parquet' t1 USING (k)"
	                              "  FULL OUTER JOIN '%s/t2.parquet' t2 USING (k)"
	                              "  FULL OUTER JOIN '%s/t3.parquet' t3 USING (k)"
	                              "  FULL OUTER JOIN '%s/t4.parquet' t4 USING (k)"
	                              ") TO '%s/joined.parquet' (FORMAT PARQUET);",
	                              dir.c_str(), dir.c_str(), dir.c_str(), dir.c_str(), dir.c_str(), dir.c_str());
	RunSQL(con, sql);
}

static void GenParquet(Connection &con, const string &dir, int64_t rows, double overlap) {
	const int n_files = 5;
	int64_t shared = static_cast<int64_t>(rows * overlap);
	int64_t unique = rows - shared;
	for (int i = 0; i < n_files; i++) {
		int64_t start = shared + i * unique;
		int64_t end = start + unique;
		// Mirror bench_fullouter5.py's value columns so file sizes are comparable.
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
		auto sql = StringUtil::Format(
		    "COPY ("
		    "  SELECT k::INTEGER AS k, %s FROM ("
		    "    SELECT range AS k FROM range(0, %lld) "
		    "    UNION ALL "
		    "    SELECT range AS k FROM range(%lld, %lld)"
		    "  )"
		    ") TO '%s' (FORMAT PARQUET);",
		    cols.c_str(), (long long)shared, (long long)start, (long long)end, path.c_str());
		RunSQL(con, sql);
	}
}

// Build a logical plan without running ColumnBindingResolver, so the result is
// safe to hand to LogicalPlanStatement (physical planner will resolve bindings).
static unique_ptr<LogicalOperator> MakePlan(Connection &con, const string &sql, bool optimize) {
	auto &ctx = *con.context;
	con.BeginTransaction();
	unique_ptr<LogicalOperator> plan;
	try {
		Parser parser;
		parser.ParseQuery(sql);
		if (parser.statements.size() != 1) {
			throw std::runtime_error("expected exactly one SQL statement");
		}
		Planner planner(ctx);
		planner.CreatePlan(std::move(parser.statements[0]));
		plan = std::move(planner.plan);
		if (optimize) {
			Optimizer optimizer(*planner.binder, ctx);
			plan = optimizer.Optimize(std::move(plan));
		}
		plan->ResolveOperatorTypes();
	} catch (...) {
		con.Rollback();
		throw;
	}
	con.Rollback();
	return plan;
}

// Execute a pre-built logical plan; returns elapsed wall time in ms.
// The plan is cloned per call; the timer wraps only the query submission +
// streaming drain of results (mimicking EXPLAIN ANALYZE: full execution,
// no materialization to a ColumnDataCollection).
static double ExecPlanMs(Connection &con, const LogicalOperator &plan) {
	auto &ctx = *con.context;
	// Copy requires an active transaction (deserialization resolves catalog entries
	// like table function names). SendQuery will nest / reuse this transaction and
	// we Commit at the end so auto-commit semantics resume for the next call.
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
		die("plan execution failed: " + err);
	}
	idx_t total_rows = 0;
	while (auto chunk = result->Fetch()) {
		total_rows += chunk->size();
	}
	auto t1 = std::chrono::steady_clock::now();
	// The transaction may have been auto-committed by SendQuery. Commit() is a
	// no-op if no transaction is active; Rollback would error in that case.
	try {
		con.Commit();
	} catch (...) {
		// ignore - SendQuery already closed it
	}
	(void)total_rows;
	return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

struct Variant {
	string name;
	string sql;
	bool optimize; // run the optimizer when building the plan
};

static vector<Variant> DefineVariants(const string &dir) {
	auto t = [&](int i) { return StringUtil::Format("'%s/t%d.parquet'", dir.c_str(), i); };

	vector<Variant> v;

	// 1) Natural 5-way FULL OUTER JOIN, optimizer ON -> whatever DuckDB prefers.
	string natural = StringUtil::Format("SELECT * FROM %s t0 "
	                                    "FULL OUTER JOIN %s t1 USING (k) "
	                                    "FULL OUTER JOIN %s t2 USING (k) "
	                                    "FULL OUTER JOIN %s t3 USING (k) "
	                                    "FULL OUTER JOIN %s t4 USING (k)",
	                                    t(0).c_str(), t(1).c_str(), t(2).c_str(), t(3).c_str(), t(4).c_str());
	v.push_back({"fo5_optimized", natural, true});

	// 2) Same SQL, optimizer OFF -> left-deep, as-written.
	v.push_back({"fo5_leftdeep_raw", natural, false});

	// 3) Right-deep shape, optimizer OFF.
	string right_deep = StringUtil::Format("SELECT * FROM %s t0 FULL OUTER JOIN ("
	                                       "  SELECT * FROM %s t1 FULL OUTER JOIN ("
	                                       "    SELECT * FROM %s t2 FULL OUTER JOIN ("
	                                       "      SELECT * FROM %s t3 FULL OUTER JOIN %s t4 USING (k)"
	                                       "    ) USING (k)"
	                                       "  ) USING (k)"
	                                       ") USING (k)",
	                                       t(0).c_str(), t(1).c_str(), t(2).c_str(), t(3).c_str(), t(4).c_str());
	v.push_back({"fo5_rightdeep_raw", right_deep, false});

	// 4) Bushy-ish shape (two pairs joined then joined with the last), optimizer OFF.
	string bushy = StringUtil::Format("SELECT * FROM ("
	                                  "  SELECT * FROM %s t0 FULL OUTER JOIN %s t1 USING (k)"
	                                  ") l FULL OUTER JOIN ("
	                                  "  SELECT * FROM %s t2 FULL OUTER JOIN %s t3 USING (k)"
	                                  ") r USING (k) FULL OUTER JOIN %s t4 USING (k)",
	                                  t(0).c_str(), t(1).c_str(), t(2).c_str(), t(3).c_str(), t(4).c_str());
	v.push_back({"fo5_bushy_raw", bushy, false});

	// Reference: single-file scan that selects * - an upper bound on "just read a Parquet".
	v.push_back({"scan_t0", StringUtil::Format("SELECT * FROM %s", t(0).c_str()), false});

	// Reference: scan of the pre-materialized denormalized join output.
	// This matches bench_fullouter5.py's "scan" comparison and lets us report a
	// join/scan ratio directly comparable to the Python bench.
	v.push_back({"scan_joined", StringUtil::Format("SELECT * FROM '%s/joined.parquet'", dir.c_str()), false});

	return v;
}

} // namespace

int main(int argc, char **argv) {
	auto args = ParseArgs(argc, argv);

	DuckDB db(nullptr);
	Connection con(db);

	if (args.threads > 0) {
		RunSQL(con, StringUtil::Format("SET threads=%d", args.threads));
	}
	// Matches bench_fullouter5.py: avoid blowing up temp on big outputs.
	RunSQL(con, "SET preserve_insertion_order=false");

	string dir = args.dir;
	bool created_tmp = false;
	if (dir.empty()) {
		char tmpl[] = "/tmp/plan_bench_XXXXXX";
		if (!::mkdtemp(tmpl)) {
			die("mkdtemp failed");
		}
		dir = tmpl;
		created_tmp = true;
	} else {
		::mkdir(dir.c_str(), 0755); // ignore errors if it already exists
	}

	if (DirHasAllFiles(dir)) {
		printf("[setup] reusing data in %s\n", dir.c_str());
	} else {
		printf("[setup] generating data in %s  rows=%lld overlap=%.2f\n", dir.c_str(), (long long)args.rows,
		       args.overlap);
		auto t0 = std::chrono::steady_clock::now();
		GenParquet(con, dir, args.rows, args.overlap);
		double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		printf("[setup] done in %.2fs\n", secs);
	}
	if (!HasJoinedFile(dir)) {
		printf("[setup] materializing joined.parquet for scan reference...\n");
		auto t0 = std::chrono::steady_clock::now();
		MaterializeJoin(con, dir);
		double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		printf("[setup] joined.parquet materialized in %.2fs\n", secs);
	}

	auto variants = DefineVariants(dir);
	printf("\n[bench] repeats=%d  threads=%d  sink=blackhole\n", args.repeats, args.threads);
	printf("        %-28s %10s %12s %8s\n", "variant", "min(ms)", "median(ms)", "opt");
	printf("        ------------------------------------------------------------------\n");

	for (auto &v : variants) {
		if (!args.only_variant.empty() && v.name != args.only_variant) {
			continue;
		}
		// blackhole() is a built-in aggregate that evaluates its children but
		// does nothing with them. Bare `*` inside an aggregate call is stripped
		// by the Postgres parser (this is why COUNT(*) is rewritten to
		// count_star), so we use DuckDB's *COLUMNS(*) unpack to forward every
		// column as a separate argument.
		string sql = "SELECT blackhole(*COLUMNS(*)) FROM (" + v.sql + ") _t";
		unique_ptr<LogicalOperator> plan;
		try {
			plan = MakePlan(con, sql, v.optimize);
		} catch (std::exception &ex) {
			fprintf(stderr, "[%s] plan build failed: %s\n", v.name.c_str(), ex.what());
			continue;
		}
		if (args.print_plan) {
			printf("\n[%s] logical plan:\n%s\n", v.name.c_str(), plan->ToString().c_str());
		}
		// Warmup.
		try {
			ExecPlanMs(con, *plan);
		} catch (std::exception &ex) {
			fprintf(stderr, "[%s] warmup failed: %s\n", v.name.c_str(), ex.what());
			continue;
		}
		vector<double> samples;
		samples.reserve(args.repeats);
		for (int i = 0; i < args.repeats; i++) {
			samples.push_back(ExecPlanMs(con, *plan));
		}
		std::sort(samples.begin(), samples.end());
		double min_ms = samples.front();
		double med = samples[samples.size() / 2];
		printf("        %-28s %10.1f %12.1f %8s\n", v.name.c_str(), min_ms, med, v.optimize ? "on" : "off");
	}

	if (created_tmp && !args.keep) {
		// Best-effort cleanup only.
		for (int i = 0; i < 5; i++) {
			::unlink((dir + "/t" + std::to_string(i) + ".parquet").c_str());
		}
		::rmdir(dir.c_str());
	} else {
		printf("\n[setup] data kept in %s\n", dir.c_str());
	}
	return 0;
}
