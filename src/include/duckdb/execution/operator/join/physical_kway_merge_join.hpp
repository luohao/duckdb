//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/join/physical_kway_merge_join.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/common/reference_map.hpp"

namespace duckdb {

class ColumnDataCollection;

//! PhysicalKWayMergeJoin implements a FULL OUTER JOIN over k sorted input
//! tables using a k-way priority-queue merge (port of ClickHouse's
//! SortingQueueImpl, src/Core/SortCursor.h:378-621).
//!
//! All k children are sink-side: each child feeds chunks into our Sink via
//! its own MetaPipeline, chunks are appended to per-child ColumnDataCollections
//! (inputs are assumed pre-sorted on the join key, so no sort is performed).
//! After all k Sinks finish, the operator becomes a Source and runs the merge.
class PhysicalKWayMergeJoin : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::KWAY_MERGE_JOIN;

public:
	PhysicalKWayMergeJoin(PhysicalPlan &plan, vector<LogicalType> types, idx_t key_col_idx, idx_t estimated_cardinality);

public:
	//! Index (within each child's schema) of the join key column.
	//! For now, all k children share the same schema: [k, v1, v2, ...].
	idx_t key_col_idx;

	//! Map from sink-side pipeline pointer to the child index (0..k-1) whose
	//! chunks that pipeline carries. Populated in BuildPipelines; consulted
	//! by GetLocalSinkState to tag each local sink with the correct child.
	//! mutable: BuildPipelines is a non-const virtual but is conceptually
	//! "setup" — the map is immutable once execution starts.
	mutable reference_map_t<Pipeline, idx_t> pipeline_to_child;

public:
	// Sink interface — k children, one MetaPipeline per child.
	bool IsSink() const override {
		return true;
	}
	bool ParallelSink() const override {
		return false;
	}
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;
	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override;

public:
	// Source interface — runs the merge once all sinks have finalized.
	bool IsSource() const override {
		return true;
	}
	// Phase 2: ParallelSource via merge-path partitioning. Each local source
	// state handles one partition (key range), independently running the
	// k-way merge over its slice of each child's CDC.
	bool ParallelSource() const override {
		return true;
	}
	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
	unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context,
	                                                 GlobalSourceState &gstate) const override;

protected:
	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override;

public:

public:
	// Pipeline wiring — custom because we have k sink-side children.
	void BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) override;
	vector<const_reference<PhysicalOperator>> GetSources() const override;

public:
	string GetName() const override {
		return "KWAY_MERGE_JOIN";
	}

	// Each child must deliver its chunks in sorted order (the operator is a
	// k-way merge and assumes sorted inputs). Returning true here tells
	// DuckDB to preserve source-side ordering when feeding our Sink.
	bool SinkOrderDependent() const override {
		return true;
	}
};

} // namespace duckdb
