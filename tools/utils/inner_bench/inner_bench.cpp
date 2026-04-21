// Time a single-threaded INNER JOIN over t0 + t1 via DuckDB, mirroring the
// shape of the Rust inner_join_bench in rust/duck-hash-join so we have a
// direct 1:1 perf comparison for the default parallel hash join.
//
// Usage: inner_bench <config_dir> [repeats] [threads] [--full-cols]
//
// --full-cols forces reading every numeric data column on both sides by
// selecting SUMs instead of count(*). Matches the work the Rust bench does
// when --keys-only is NOT set (6 columns per table).

#include "duckdb.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/statement/logical_plan_statement.hpp"
#include "duckdb/planner/planner.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using namespace duckdb; // NOLINT

int main(int argc, char **argv) {
	if (argc < 2) {
		fprintf(stderr, "usage: inner_bench <config_dir> [repeats] [threads] [--full-cols]\n");
		return 1;
	}
	std::string dir = argv[1];
	int repeats = 3, threads = 1;
	int kway = 2;
	bool full_cols = false;
	bool foj = false;
	bool scan_only = false;
	int pos = 0;
	for (int i = 2; i < argc; i++) {
		std::string a = argv[i];
		if (a == "--full-cols") {
			full_cols = true;
		} else if (a == "--foj") {
			foj = true;
		} else if (a == "--scan-only") {
			scan_only = true;
		} else if (a.rfind("-k", 0) == 0 && a.size() > 2) {
			kway = std::atoi(a.c_str() + 2);
		} else if (pos == 0) {
			repeats = std::atoi(argv[i]);
			pos++;
		} else if (pos == 1) {
			threads = std::atoi(argv[i]);
			pos++;
		}
	}
	if (kway < 2) {
		fprintf(stderr, "-k must be ≥ 2\n");
		return 1;
	}

	DuckDB db(nullptr);
	Connection con(db);
	con.Query(StringUtil::Format("SET threads=%d", threads));
	con.Query("SET preserve_insertion_order=false");

	// --full-cols: project all 6 numeric bench columns from both sides
	// (k, c_int, c_bigint, c_dbl, c_flt, c_bool) and wrap them with
	// `blackhole(*COLUMNS(*))` — the same sink kway_bench uses. This
	// forces reading every column from parquet and through the join
	// pipeline without adding per-row aggregate work, so the comparison
	// to the Rust bench (which reads all 6 cols into its arena) is fair.
	// Build k-way chain SQL. For --scan-only, just read joined.parquet.
	// Otherwise, chain (k-1) binary joins on key `k`. With --full-cols,
	// project the six numeric bench columns from every table and wrap
	// with blackhole so the parquet reader can't prune them away. Without
	// --full-cols, we use count(*) so projection-pushdown can strip every
	// data column (just `k` read from each).
	const char *join_kw = foj ? "FULL OUTER JOIN" : "INNER JOIN";
	const char *data_cols[] = {"c_int", "c_bigint", "c_dbl", "c_flt", "c_bool"};
	std::string sql;
	if (scan_only) {
		sql = StringUtil::Format(
		    "SELECT blackhole(*COLUMNS(*)) FROM '%s/joined.parquet'", dir.c_str());
	} else if (full_cols) {
		// SELECT list: t0.k, plus each data col from every table suffixed
		// with the table index.
		std::string select_list = "t0.k";
		for (int t = 0; t < kway; t++) {
			for (const char *col : data_cols) {
				select_list += StringUtil::Format(", t%d.%s AS %s_%d", t, col, col, t);
			}
		}
		// FROM chain: t0 JOIN t1 USING(k) JOIN t2 USING(k) ...
		std::string from_chain = StringUtil::Format("'%s/t0.parquet' t0", dir.c_str());
		for (int t = 1; t < kway; t++) {
			from_chain += StringUtil::Format(" %s '%s/t%d.parquet' t%d USING (k)", join_kw,
			                                 dir.c_str(), t, t);
		}
		sql = StringUtil::Format("SELECT blackhole(*COLUMNS(*)) FROM (SELECT %s FROM %s) _t",
		                         select_list.c_str(), from_chain.c_str());
	} else {
		std::string from_chain = StringUtil::Format("'%s/t0.parquet' t0", dir.c_str());
		for (int t = 1; t < kway; t++) {
			from_chain += StringUtil::Format(" %s '%s/t%d.parquet' t%d USING (k)", join_kw,
			                                 dir.c_str(), t, t);
		}
		sql = StringUtil::Format("SELECT count(*) FROM %s", from_chain.c_str());
	}

	printf("[inner_bench] dir=%s threads=%d repeats=%d\n", dir.c_str(), threads, repeats);
	printf("              sql: %s\n\n", sql.c_str());

	// Warmup + timed runs.
	std::vector<double> samples;
	samples.reserve(repeats);
	int64_t last_count = 0;
	for (int rep = 0; rep <= repeats; rep++) {
		auto t0 = std::chrono::steady_clock::now();
		auto result = con.Query(sql);
		if (result->HasError()) {
			fprintf(stderr, "error: %s\n", result->GetError().c_str());
			return 1;
		}
		// `blackhole` returns BIGINT 0; count(*) returns the count. Both
		// fit in int64_t.
		int64_t cnt = 0;
		try {
			cnt = result->GetValue(0, 0).GetValue<int64_t>();
		} catch (...) {
			cnt = -1;
		}
		auto t1 = std::chrono::steady_clock::now();
		double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
		if (rep == 0) {
			printf("[warmup] count=%lld  %.1fms\n", (long long)cnt, ms);
		} else {
			samples.push_back(ms);
			last_count = cnt;
			printf("[rep %d]  count=%lld  %.1fms\n", rep, (long long)cnt, ms);
		}
	}

	std::sort(samples.begin(), samples.end());
	double med = samples[samples.size() / 2];
	printf("\n[median over %zu reps] %.1fms  (count=%lld)\n", samples.size(), med,
	       (long long)last_count);
	return 0;
}
