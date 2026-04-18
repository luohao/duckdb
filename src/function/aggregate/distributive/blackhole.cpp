#include "duckdb/function/aggregate/distributive_functions.hpp"
#include "duckdb/function/aggregate/distributive_function_utils.hpp"

namespace duckdb {

namespace {

// The blackhole aggregate is a "do nothing" aggregate intended for benchmarking.
// It accepts any number of arguments of any type, forces them to be read from the
// underlying scan (because they appear as children of the aggregate expression, so
// projection pushdown keeps them), but performs no work on the values. The result
// is always BIGINT 0.
//
// This is useful to defeat metadata-only shortcuts like the one count_star(*) takes
// on parquet files while still avoiding the cost of computing an actual aggregate
// over the input values.

struct BlackholeState {
	uint8_t dummy;
};

struct BlackholeOp {
	template <class STATE>
	static void Initialize(STATE &state) {
		state.dummy = 0;
	}

	template <class STATE, class OP>
	static void Combine(const STATE &, STATE &, AggregateInputData &) {
	}

	template <class T, class STATE>
	static void Finalize(STATE &, T &target, AggregateFinalizeData &) {
		target = 0;
	}

	static bool IgnoreNull() {
		return false;
	}
};

void BlackholeUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, data_ptr_t, idx_t) {
	(void)inputs;
	(void)input_count;
}

void BlackholeScatter(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &states, idx_t count) {
	(void)inputs;
	(void)input_count;
	(void)states;
	(void)count;
}

AggregateFunction GetVariadicBlackhole() {
	AggregateFunction fun({LogicalType(LogicalTypeId::ANY)}, LogicalType::BIGINT,
	                      AggregateFunction::StateSize<BlackholeState>,
	                      AggregateFunction::StateInitialize<BlackholeState, BlackholeOp>, BlackholeScatter,
	                      AggregateFunction::StateCombine<BlackholeState, BlackholeOp>,
	                      AggregateFunction::StateFinalize<BlackholeState, int64_t, BlackholeOp>,
	                      FunctionNullHandling::SPECIAL_HANDLING, BlackholeUpdate);
	fun.name = "blackhole";
	fun.varargs = LogicalType::ANY;
	fun.SetOrderDependent(AggregateOrderDependent::NOT_ORDER_DEPENDENT);
	fun.SetDistinctDependent(AggregateDistinctDependent::NOT_DISTINCT_DEPENDENT);
	return fun;
}

AggregateFunction GetNullaryBlackhole() {
	AggregateFunction fun({}, LogicalType::BIGINT, AggregateFunction::StateSize<BlackholeState>,
	                      AggregateFunction::StateInitialize<BlackholeState, BlackholeOp>, BlackholeScatter,
	                      AggregateFunction::StateCombine<BlackholeState, BlackholeOp>,
	                      AggregateFunction::StateFinalize<BlackholeState, int64_t, BlackholeOp>,
	                      FunctionNullHandling::SPECIAL_HANDLING, BlackholeUpdate);
	fun.name = "blackhole";
	fun.SetOrderDependent(AggregateOrderDependent::NOT_ORDER_DEPENDENT);
	fun.SetDistinctDependent(AggregateDistinctDependent::NOT_DISTINCT_DEPENDENT);
	return fun;
}

} // namespace

AggregateFunctionSet BlackholeFun::GetFunctions() {
	AggregateFunctionSet set("blackhole");
	set.AddFunction(GetNullaryBlackhole());
	set.AddFunction(GetVariadicBlackhole());
	return set;
}

} // namespace duckdb
