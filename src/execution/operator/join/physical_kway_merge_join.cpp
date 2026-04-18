#include "duckdb/execution/operator/join/physical_kway_merge_join.hpp"

#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"
#include "duckdb/parallel/pipeline.hpp"
#include "duckdb/parallel/task_scheduler.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>

namespace duckdb {

//===--------------------------------------------------------------------===//
// Constructor
//===--------------------------------------------------------------------===//
PhysicalKWayMergeJoin::PhysicalKWayMergeJoin(PhysicalPlan &plan, vector<LogicalType> types, idx_t key_col_idx_p,
                                             idx_t estimated_cardinality)
    : PhysicalOperator(plan, PhysicalOperatorType::KWAY_MERGE_JOIN, std::move(types), estimated_cardinality),
      key_col_idx(key_col_idx_p) {
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class KWayMergeJoinGlobalSinkState : public GlobalSinkState {
public:
	explicit KWayMergeJoinGlobalSinkState(const PhysicalKWayMergeJoin &op) {
		thread_tables.resize(op.children.size());
	}

	//! For each child i: the list of per-thread local ColumnDataCollections
	//! that contributed to it (moved in from LocalSinkState::local_table
	//! during Combine). Each individual CDC is internally sorted on key;
	//! different threads' CDCs for the same child may have overlapping key
	//! ranges (the parquet reader distributes row groups across threads).
	//! The Source merges all of them as independent cursors.
	vector<vector<unique_ptr<ColumnDataCollection>>> thread_tables;
	//! Guard concurrent Combine calls that push onto the vectors above.
	mutex combine_lock;
};

class KWayMergeJoinLocalSinkState : public LocalSinkState {
public:
	KWayMergeJoinLocalSinkState(ClientContext &context, const PhysicalKWayMergeJoin &op, idx_t child_idx_p)
	    : child_idx(child_idx_p) {
		const auto &child_types = op.children[child_idx].get().GetTypes();
		local_table = make_uniq<ColumnDataCollection>(context, child_types);
		local_table->InitializeAppend(append_state);
	}

	//! Which child this local sink belongs to — determined by the pipeline
	//! that requested this LocalSinkState (via op.pipeline_to_child).
	idx_t child_idx;
	unique_ptr<ColumnDataCollection> local_table;
	ColumnDataAppendState append_state;
};

unique_ptr<GlobalSinkState> PhysicalKWayMergeJoin::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<KWayMergeJoinGlobalSinkState>(*this);
}

unique_ptr<LocalSinkState> PhysicalKWayMergeJoin::GetLocalSinkState(ExecutionContext &context) const {
	// Each sink-side pipeline is keyed to exactly one child input. The
	// pipeline_to_child map was populated in BuildPipelines.
	auto pipeline_ptr = context.pipeline;
	if (!pipeline_ptr) {
		throw InternalException("PhysicalKWayMergeJoin::GetLocalSinkState: no pipeline in ExecutionContext");
	}
	auto it = pipeline_to_child.find(*pipeline_ptr);
	if (it == pipeline_to_child.end()) {
		throw InternalException("PhysicalKWayMergeJoin::GetLocalSinkState: pipeline not in pipeline_to_child map");
	}
	return make_uniq<KWayMergeJoinLocalSinkState>(context.client, *this, it->second);
}

SinkResultType PhysicalKWayMergeJoin::Sink(ExecutionContext &context, DataChunk &chunk,
                                            OperatorSinkInput &input) const {
	auto &lstate = input.local_state.Cast<KWayMergeJoinLocalSinkState>();
	lstate.local_table->Append(lstate.append_state, chunk);
	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalKWayMergeJoin::Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const {
	auto &gstate = input.global_state.Cast<KWayMergeJoinGlobalSinkState>();
	auto &lstate = input.local_state.Cast<KWayMergeJoinLocalSinkState>();
	if (lstate.local_table->Count() == 0) {
		return SinkCombineResultType::FINISHED;
	}
	lock_guard<mutex> guard(gstate.combine_lock);
	gstate.thread_tables[lstate.child_idx].push_back(std::move(lstate.local_table));
	return SinkCombineResultType::FINISHED;
}

SinkFinalizeType PhysicalKWayMergeJoin::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                  OperatorSinkFinalizeInput &input) const {
	// Nothing to do per-finalize — each child's per-thread sorted CDCs
	// are already accumulated. The Source will build (k · threads_per_child)
	// cursors and run the merge.
	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// K-Way merge algorithm — a local port of the same priority-queue logic
// used by tools/utils/kway_bench.cpp (which ports ClickHouse's
// SortingQueueImpl, src/Core/SortCursor.h:378-621).
//===--------------------------------------------------------------------===//
namespace {

// Cached per-chunk metadata: first key + cumulative row count BEFORE this chunk.
// Built once per CDC in the source's global state; lets cursors seek to a
// specific row/key in O(log n_chunks) instead of scanning from the start.
struct ChunkIndex {
	vector<int32_t> chunk_first_keys; // first key of each chunk (sorted ascending)
	vector<int32_t> chunk_last_keys;  // last key of each chunk (sorted ascending)
	vector<idx_t> chunk_start_rows;   // cumulative rows before chunk i
	idx_t total_rows = 0;
};

static ChunkIndex BuildChunkIndex(ClientContext &ctx, const ColumnDataCollection &cdc, idx_t key_col_idx) {
	ChunkIndex idx;
	idx.total_rows = cdc.Count();
	const idx_t n_chunks = cdc.ChunkCount();
	idx.chunk_first_keys.reserve(n_chunks);
	idx.chunk_last_keys.reserve(n_chunks);
	idx.chunk_start_rows.reserve(n_chunks);

	DataChunk chunk;
	chunk.Initialize(ctx, cdc.Types());
	idx_t cum = 0;
	for (idx_t i = 0; i < n_chunks; i++) {
		idx.chunk_start_rows.push_back(cum);
		chunk.Reset();
		cdc.FetchChunk(i, chunk);
		auto *keys = FlatVector::GetData<int32_t>(chunk.data[key_col_idx]);
		idx.chunk_first_keys.push_back(keys[0]);
		idx.chunk_last_keys.push_back(keys[chunk.size() - 1]);
		cum += chunk.size();
	}
	return idx;
}

// Find first row index in `cdc` with key >= target_key.
static idx_t LowerBoundRow(const ChunkIndex &idx, int32_t target_key) {
	if (idx.total_rows == 0) {
		return 0;
	}
	// Binary search over last_keys for the first chunk whose last key >= target.
	auto &last = idx.chunk_last_keys;
	auto it = std::lower_bound(last.begin(), last.end(), target_key);
	if (it == last.end()) {
		return idx.total_rows; // past end
	}
	// The chunk index and its start row.
	idx_t c = it - last.begin();
	return idx.chunk_start_rows[c]; // caller's cursor will land here and scan inside the chunk
}

struct KWayCursor {
	idx_t table_idx = 0;
	const ColumnDataCollection *data = nullptr;
	const ChunkIndex *cindex = nullptr;
	idx_t current_chunk_idx = 0;
	DataChunk chunk;
	idx_t chunk_pos = 0;
	idx_t key_col_idx = 0;
	const int32_t *key_ptr = nullptr;
	int32_t current_key = 0;
	bool exhausted = true;
	// For partition-bounded cursors (Phase 2 parallel source). If has_end_key,
	// the cursor reports exhausted once current_key >= end_key.
	bool has_end_key = false;
	int32_t end_key = 0;

	void LoadChunkAtIdx(idx_t cidx) {
		current_chunk_idx = cidx;
		chunk.Reset();
		if (cidx >= data->ChunkCount()) {
			exhausted = true;
			key_ptr = nullptr;
			return;
		}
		data->FetchChunk(cidx, chunk);
		if (chunk.size() == 0) {
			// Skip empty chunks
			LoadChunkAtIdx(cidx + 1);
			return;
		}
		chunk_pos = 0;
		key_ptr = FlatVector::GetData<int32_t>(chunk.data[key_col_idx]);
		current_key = key_ptr[0];
		if (has_end_key && current_key >= end_key) {
			exhausted = true;
		}
	}

	void Init(idx_t tbl_idx, const ColumnDataCollection &d, ClientContext &ctx, idx_t key_col_idx_p) {
		table_idx = tbl_idx;
		data = &d;
		cindex = nullptr;
		key_col_idx = key_col_idx_p;
		has_end_key = false;
		exhausted = false;
		chunk.Initialize(ctx, d.Types());
		LoadChunkAtIdx(0);
	}

	// Initialize cursor at the first row in `[start_row, total_rows)` whose
	// key is < end_key; if no such row exists, the cursor starts exhausted.
	void InitRange(idx_t tbl_idx, const ColumnDataCollection &d, ClientContext &ctx, idx_t key_col_idx_p,
	               const ChunkIndex &idx, idx_t start_row, int32_t end_key_exclusive) {
		table_idx = tbl_idx;
		data = &d;
		cindex = &idx;
		key_col_idx = key_col_idx_p;
		has_end_key = true;
		end_key = end_key_exclusive;
		if (start_row >= idx.total_rows) {
			exhausted = true;
			return;
		}
		exhausted = false;
		chunk.Initialize(ctx, d.Types());
		// Find chunk containing start_row
		auto it = std::upper_bound(idx.chunk_start_rows.begin(), idx.chunk_start_rows.end(), start_row);
		idx_t cidx = (it - idx.chunk_start_rows.begin()) - 1;
		LoadChunkAtIdx(cidx);
		if (exhausted) {
			return;
		}
		chunk_pos = start_row - idx.chunk_start_rows[cidx];
		current_key = key_ptr[chunk_pos];
		if (has_end_key && current_key >= end_key) {
			exhausted = true;
		}
	}

	void LoadNextChunk() {
		LoadChunkAtIdx(current_chunk_idx + 1);
	}

	void Advance() {
		++chunk_pos;
		if (chunk_pos >= chunk.size()) {
			LoadNextChunk();
			return;
		}
		current_key = key_ptr[chunk_pos];
		if (has_end_key && current_key >= end_key) {
			exhausted = true;
		}
	}
	void AdvanceBy(idx_t n) {
		chunk_pos += n;
		if (chunk_pos >= chunk.size()) {
			LoadNextChunk();
			return;
		}
		current_key = key_ptr[chunk_pos];
		if (has_end_key && current_key >= end_key) {
			exhausted = true;
		}
	}
	idx_t RowsInChunk() const {
		return chunk.size() - chunk_pos;
	}
	bool IsValid() const {
		return !exhausted;
	}
};

class KWayMergeQueue {
public:
	void Build(vector<KWayCursor *> &cursors) {
		heap.reserve(cursors.size());
		for (auto *c : cursors) {
			if (c->IsValid()) {
				heap.push_back(c);
			}
		}
		std::make_heap(heap.begin(), heap.end(), GreaterCmp());
	}
	void Rebuild(const vector<unique_ptr<KWayCursor>> &cursors) {
		heap.clear();
		for (auto &c : cursors) {
			if (c->IsValid()) {
				heap.push_back(c.get());
			}
		}
		std::make_heap(heap.begin(), heap.end(), GreaterCmp());
		next_child_idx = 0;
	}
	bool IsValid() const {
		return !heap.empty();
	}
	size_t Size() const {
		return heap.size();
	}
	KWayCursor *Top() {
		return heap.front();
	}
	int32_t NextBestKey() {
		return heap[NextChildIndex()]->current_key;
	}
	void Next() {
		auto *top = heap.front();
		top->Advance();
		if (!top->IsValid()) {
			RemoveTop();
		} else {
			UpdateTop(true);
		}
	}
	void FixupTop() {
		auto *top = heap.front();
		if (!top->IsValid()) {
			RemoveTop();
		} else {
			UpdateTop(true);
		}
	}

private:
	static bool Greater(const KWayCursor *a, const KWayCursor *b) {
		if (a->current_key != b->current_key) {
			return a->current_key > b->current_key;
		}
		return a->table_idx > b->table_idx;
	}
	struct GreaterCmp {
		bool operator()(const KWayCursor *a, const KWayCursor *b) const {
			return Greater(a, b);
		}
	};
	size_t NextChildIndex() {
		if (next_child_idx == 0) {
			next_child_idx = 1;
			if (heap.size() > 2 && Greater(heap[1], heap[2])) {
				next_child_idx = 2;
			}
		}
		return next_child_idx;
	}
	void RemoveTop() {
		std::pop_heap(heap.begin(), heap.end(), GreaterCmp());
		heap.pop_back();
		next_child_idx = 0;
	}
	void UpdateTop(bool check_in_order) {
		size_t size = heap.size();
		if (size < 2) {
			return;
		}
		size_t child_idx = NextChildIndex();
		if (check_in_order && Greater(heap[child_idx], heap[0])) {
			return;
		}
		next_child_idx = 0;
		size_t curr_idx = 0;
		KWayCursor *top = heap[0];
		do {
			heap[curr_idx] = heap[child_idx];
			curr_idx = child_idx;
			child_idx = 2 * child_idx + 1;
			if (child_idx >= size) {
				break;
			}
			if (child_idx + 1 < size && Greater(heap[child_idx], heap[child_idx + 1])) {
				++child_idx;
			}
		} while (!Greater(heap[child_idx], top));
		heap[curr_idx] = top;
	}

	vector<KWayCursor *> heap;
	size_t next_child_idx = 0;
};

} // namespace

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
// Per-child CDC + its chunk index (built once in the global source state).
struct ChildIndex {
	const ColumnDataCollection *cdc;
	ChunkIndex index;
};

class KWayMergeJoinGlobalSourceState : public GlobalSourceState {
public:
	KWayMergeJoinGlobalSourceState(ClientContext &ctx_p, const PhysicalKWayMergeJoin &op,
	                               KWayMergeJoinGlobalSinkState &sink)
	    : ctx(ctx_p), op(op), sink(sink) {
		// Output layout: concatenation of all children's full schemas.
		col_offsets.resize(op.children.size());
		n_cols_per_child.resize(op.children.size());
		idx_t offset = 0;
		for (idx_t i = 0; i < op.children.size(); i++) {
			col_offsets[i] = offset;
			const auto &types = op.children[i].get().GetTypes();
			n_cols_per_child[i] = types.size();
			offset += n_cols_per_child[i];
		}

		// Decide partition count first: we only build chunk indexes (which
		// costs O(chunks) FetchChunk calls per child) if partitioning is
		// actually going to help.
		auto &scheduler = TaskScheduler::GetScheduler(ctx);
		idx_t hw_threads = NumericCast<idx_t>(scheduler.NumberOfThreads());
		num_partitions = MaxValue<idx_t>(1, hw_threads);

		children_idx.resize(op.children.size());
		for (idx_t i = 0; i < op.children.size(); i++) {
			auto &per_thread = sink.thread_tables[i];
			if (per_thread.empty()) {
				children_idx[i].cdc = nullptr;
				continue;
			}
			// Serial sink: exactly one CDC per child.
			children_idx[i].cdc = per_thread[0].get();
		}

		// Single-threaded? Skip chunk-index + boundary_keys; partition 0's
		// InitializeLocalSourceForPartition will fall back to the simple
		// non-bounded cursor path.
		if (num_partitions <= 1) {
			return;
		}

		// Build chunk indexes only for the children we need to partition.
		for (idx_t i = 0; i < op.children.size(); i++) {
			if (children_idx[i].cdc) {
				children_idx[i].index = BuildChunkIndex(ctx, *children_idx[i].cdc, op.key_col_idx);
			}
		}

		// Cap partitions by cursor 0's chunk count.
		if (children_idx[0].cdc && children_idx[0].index.total_rows > 0) {
			auto &idx0 = children_idx[0].index;
			num_partitions = MinValue<idx_t>(num_partitions, idx0.chunk_last_keys.size());
			for (idx_t p = 1; p < num_partitions; p++) {
				idx_t target_row = (idx0.total_rows * p) / num_partitions;
				auto it = std::upper_bound(idx0.chunk_start_rows.begin(), idx0.chunk_start_rows.end(), target_row);
				idx_t cidx = (it - idx0.chunk_start_rows.begin()) - 1;
				DataChunk chunk;
				chunk.Initialize(ctx, children_idx[0].cdc->Types());
				children_idx[0].cdc->FetchChunk(cidx, chunk);
				auto *keys = FlatVector::GetData<int32_t>(chunk.data[op.key_col_idx]);
				idx_t within = target_row - idx0.chunk_start_rows[cidx];
				boundary_keys.push_back(keys[within]);
			}
		} else {
			num_partitions = 1;
		}
	}

	idx_t MaxThreads() override {
		return num_partitions;
	}

	// Atomic counter: each worker pulls the next partition to process.
	std::atomic<idx_t> next_partition {0};

	ClientContext &ctx;
	const PhysicalKWayMergeJoin &op;
	KWayMergeJoinGlobalSinkState &sink;
	vector<ChildIndex> children_idx;
	//! P-1 boundary keys. Partition p handles keys in [boundary_keys[p-1], boundary_keys[p]),
	//! with -INF used for boundary_keys[-1] (partition 0 start) and +INF for
	//! boundary_keys[P-1] (partition P-1 end).
	vector<int32_t> boundary_keys;
	idx_t num_partitions = 1;
	vector<idx_t> col_offsets;
	vector<idx_t> n_cols_per_child;
};

class KWayMergeJoinLocalSourceState : public LocalSourceState {
public:
	//! Which partition this local state is bound to. Assigned on creation
	//! via GlobalSourceState::next_partition.fetch_add(1). -1 = no more work.
	idx_t partition_idx = std::numeric_limits<idx_t>::max();
	vector<unique_ptr<KWayCursor>> cursors;
	KWayMergeQueue queue;
	bool initialized = false;
	bool done = false;
};

unique_ptr<GlobalSourceState> PhysicalKWayMergeJoin::GetGlobalSourceState(ClientContext &context) const {
	auto &sink = sink_state->Cast<KWayMergeJoinGlobalSinkState>();
	return make_uniq<KWayMergeJoinGlobalSourceState>(context, *this, sink);
}

unique_ptr<LocalSourceState> PhysicalKWayMergeJoin::GetLocalSourceState(ExecutionContext &context,
                                                                        GlobalSourceState &gstate) const {
	return make_uniq<KWayMergeJoinLocalSourceState>();
}

// Initialize the local state's cursors + heap for its assigned partition.
// Cursors are positioned at the first row >= start_key in each CDC and
// bounded by end_key.
static void InitializeLocalSourceForPartition(ExecutionContext &context,
                                               KWayMergeJoinGlobalSourceState &gstate,
                                               KWayMergeJoinLocalSourceState &lstate, idx_t partition_idx) {
	lstate.partition_idx = partition_idx;
	const bool is_last = partition_idx + 1 == gstate.num_partitions;
	// Determine [start_key, end_key) for this partition.
	int32_t start_key =
	    partition_idx == 0 ? std::numeric_limits<int32_t>::min() : gstate.boundary_keys[partition_idx - 1];
	int32_t end_key = is_last ? std::numeric_limits<int32_t>::max() : gstate.boundary_keys[partition_idx];

	lstate.cursors.clear();
	lstate.cursors.reserve(gstate.children_idx.size());
	for (idx_t i = 0; i < gstate.children_idx.size(); i++) {
		const auto &ci = gstate.children_idx[i];
		if (!ci.cdc || ci.cdc->Count() == 0) {
			continue;
		}
		auto c = make_uniq<KWayCursor>();
		if (gstate.num_partitions == 1) {
			// Single-partition fast path: no chunk index built, use simple Init.
			c->Init(i, *ci.cdc, context.client, gstate.op.key_col_idx);
		} else {
			// Multi-partition: start at the first row with key >= start_key,
			// and stop at key >= end_key. The LAST partition has end_key=MAX
			// so it effectively runs unbounded to the end.
			idx_t start_row = (partition_idx == 0) ? 0 : LowerBoundRow(ci.index, start_key);
			if (start_row >= ci.index.total_rows) {
				continue;
			}
			c->InitRange(i, *ci.cdc, context.client, gstate.op.key_col_idx, ci.index, start_row, end_key);
		}
		if (c->IsValid()) {
			lstate.cursors.push_back(std::move(c));
		}
	}
	vector<KWayCursor *> raw;
	raw.reserve(lstate.cursors.size());
	for (auto &c : lstate.cursors) {
		raw.push_back(c.get());
	}
	lstate.queue.Build(raw);
	lstate.initialized = true;
}

SourceResultType PhysicalKWayMergeJoin::GetDataInternal(ExecutionContext &context, DataChunk &output,
                                                         OperatorSourceInput &input) const {
	auto &gstate = input.global_state.Cast<KWayMergeJoinGlobalSourceState>();
	auto &lstate = input.local_state.Cast<KWayMergeJoinLocalSourceState>();

	// Lazy initialization: claim a partition the first time this local state
	// is asked for data. Each local state handles exactly one partition.
	if (!lstate.initialized) {
		idx_t pidx = gstate.next_partition.fetch_add(1);
		if (pidx >= gstate.num_partitions) {
			lstate.done = true;
			return SourceResultType::FINISHED;
		}
		InitializeLocalSourceForPartition(context, gstate, lstate, pidx);
	}
	if (lstate.done) {
		return SourceResultType::FINISHED;
	}

	auto &queue = lstate.queue;
	auto &cursors = lstate.cursors;
	if (!queue.IsValid()) {
		lstate.done = true;
		return SourceResultType::FINISHED;
	}

	auto &col_offsets = gstate.col_offsets;
	auto &n_cols_per_child = gstate.n_cols_per_child;

	output.Reset();
	idx_t out_row = 0;

	// Pre-mark ALL columns as invalid. When a cursor for child i contributes
	// to a row, its Copy calls will flip that child's slot back to valid.
	// Children that don't contribute stay NULL — which is exactly FULL OUTER
	// JOIN semantics.
	for (idx_t col = 0; col < output.data.size(); col++) {
		FlatVector::Validity(output.data[col]).SetAllInvalid(STANDARD_VECTOR_SIZE);
	}

	auto copy_child_batch = [&](const KWayCursor &c, idx_t src_row, idx_t n, idx_t out_start) {
		const idx_t t = c.table_idx;
		const idx_t start = col_offsets[t];
		const idx_t ncols = n_cols_per_child[t];
		auto &src = c.chunk;
		for (idx_t col = 0; col < ncols; col++) {
			VectorOperations::Copy(src.data[col], output.data[start + col], src_row + n, src_row, out_start);
		}
	};

	while (out_row < STANDARD_VECTOR_SIZE && queue.IsValid()) {
		auto *top = queue.Top();
		const int32_t key = top->current_key;
		const bool can_batch = queue.Size() == 1 || key < queue.NextBestKey();

		// BATCH PATH: unique-to-top run
		if (can_batch) {
			idx_t batch = 1;
			const idx_t chunk_remaining = top->RowsInChunk();
			const idx_t out_remaining = STANDARD_VECTOR_SIZE - out_row;
			const idx_t max_batch = MinValue<idx_t>(chunk_remaining, out_remaining);
			if (queue.Size() == 1) {
				batch = max_batch;
			} else {
				const int32_t next_key = queue.NextBestKey();
				const int32_t *k = top->key_ptr + top->chunk_pos;
				while (batch < max_batch && k[batch] < next_key) {
					++batch;
				}
			}
			copy_child_batch(*top, top->chunk_pos, batch, out_row);
			top->AdvanceBy(batch);
			queue.FixupTop();
			out_row += batch;
			continue;
		}

		// ALL-AGREE SUBPATH: every cursor is at `key` and has keys key..key+N-1.
		// This fast path only applies when there's exactly one cursor per
		// child (the single-threaded-sink case). With parallel sink we have
		// multiple cursors per child, so this check naturally gates us into
		// the generic GROUP path.
		bool all_at_key = cursors.size() == gstate.op.children.size() && queue.Size() == cursors.size();
		if (all_at_key) {
			for (idx_t i = 0; i < cursors.size(); i++) {
				if (cursors[i]->current_key != key) {
					all_at_key = false;
					break;
				}
			}
		}
		if (all_at_key) {
			idx_t max_run = STANDARD_VECTOR_SIZE - out_row;
			for (idx_t i = 0; i < cursors.size(); i++) {
				max_run = MinValue<idx_t>(max_run, cursors[i]->RowsInChunk());
			}
			idx_t N = 1;
			while (N < max_run) {
				bool ok = true;
				for (idx_t i = 0; i < cursors.size() && ok; i++) {
					if (cursors[i]->key_ptr[cursors[i]->chunk_pos + N] != key + static_cast<int32_t>(N)) {
						ok = false;
					}
				}
				if (!ok) {
					break;
				}
				++N;
			}
			for (idx_t i = 0; i < cursors.size(); i++) {
				auto *c = cursors[i].get();
				copy_child_batch(*c, c->chunk_pos, N, out_row);
				c->AdvanceBy(N);
			}
			queue.Rebuild(cursors);
			out_row += N;
			continue;
		}

		// GROUP PATH (generic): per-row copies for contributing tables
		while (queue.IsValid() && queue.Top()->current_key == key) {
			auto *c = queue.Top();
			copy_child_batch(*c, c->chunk_pos, 1, out_row);
			queue.Next();
		}
		++out_row;
	}

	output.SetCardinality(out_row);
	return out_row == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void PhysicalKWayMergeJoin::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	op_state.reset();
	sink_state.reset();
	pipeline_to_child.clear();

	// Adopt IEJoin's pattern: this operator is the SOURCE of `current`
	// (all downstream operators pull from us). All k children sink into us
	// via a single child MetaPipeline with k pipelines that share the same
	// sink (us). Each pipeline's Pipeline* is registered in
	// pipeline_to_child so GetLocalSinkState can tag each worker's local
	// state with the correct child index — avoiding races on a shared
	// "current_child" counter.
	meta_pipeline.GetState().SetPipelineSource(current, *this);

	auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);

	// Child 0 → base pipeline
	auto base_pipeline = child_meta_pipeline.GetBasePipeline();
	pipeline_to_child.emplace(*base_pipeline, 0);
	children[0].get().BuildPipelines(*base_pipeline, child_meta_pipeline);

	// Children 1..k-1 → new pipelines in the same child_meta_pipeline
	for (idx_t i = 1; i < children.size(); i++) {
		auto &pipe = child_meta_pipeline.CreatePipeline();
		pipeline_to_child.emplace(pipe, i);
		children[i].get().BuildPipelines(pipe, child_meta_pipeline);
		child_meta_pipeline.AddFinishEvent(pipe);
	}
}

vector<const_reference<PhysicalOperator>> PhysicalKWayMergeJoin::GetSources() const {
	return {*this};
}

} // namespace duckdb
