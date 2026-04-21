//! Row storage for the hash-join build side.
//!
//! Simplified port of DuckDB's `TupleDataCollection`
//! (`src/common/types/row/tuple_data_collection.cpp`) + `TupleDataLayout`
//! (`tuple_data_layout.hpp`). Scope trimmed to the bench workload:
//!
//! * Fixed-width columns only — no varlen strings, no nested types.
//! * No validity bitmap — columns are assumed non-null. All the parquet
//!   inputs in `kway_bench` are non-null.
//! * No buffer manager, no block index, no pinning. One contiguous `Vec<u8>`
//!   is enough while build-side data fits in memory. Out-of-core spilling
//!   is a post-MVP concern.
//!
//! Row layout (follows DuckDB's convention: header + fixed slots):
//! ```text
//!   offset  size  field
//!   0       8     next_ptr   u64   — links collision chain in JoinHashTable
//!   8       8     hash       u64   — cached hash; re-partitioning avoids recompute
//!   16      …     columns         — per-column slots at `RowLayout::column_offsets`
//! ```
//!
//! `next_ptr` at offset 0 is a DuckDB convention that lets the probe loop
//! chase chains with a single load; `hash` at offset 8 is adjacent so it's
//! on the same cache line. Keep this order when adding features.

pub const HEADER_NEXT_PTR: usize = 0;
pub const HEADER_HASH: usize = 8;
pub const HEADER_SIZE: usize = 16;

/// Physical column types the MVP supports — every column in the kway_bench
/// parquet except `c_short` and `c_long` (VARCHAR).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PhysType {
    I32,
    I64,
    F32,
    F64,
    Bool,
}

impl PhysType {
    #[inline]
    pub const fn size(self) -> usize {
        match self {
            PhysType::I32 | PhysType::F32 => 4,
            PhysType::I64 | PhysType::F64 => 8,
            PhysType::Bool => 1,
        }
    }

    /// Natural alignment; DuckDB aligns slots to the type's size.
    #[inline]
    pub const fn align(self) -> usize {
        self.size()
    }
}

/// Describes a row's fixed layout: header + per-column offsets.
#[derive(Clone, Debug)]
pub struct RowLayout {
    column_types: Vec<PhysType>,
    column_offsets: Vec<usize>,
    row_size: usize,
}

impl RowLayout {
    /// Build a layout for the given column types, packing columns in the
    /// order provided with natural alignment. The first column is placed
    /// right after the header (offset `HEADER_SIZE`).
    pub fn new(column_types: Vec<PhysType>) -> Self {
        let mut offset = HEADER_SIZE;
        let mut column_offsets = Vec::with_capacity(column_types.len());
        for ty in &column_types {
            offset = align_up(offset, ty.align());
            column_offsets.push(offset);
            offset += ty.size();
        }
        // Final row size aligned to 8 so the next row's header is aligned.
        let row_size = align_up(offset, 8);
        RowLayout {
            column_types,
            column_offsets,
            row_size,
        }
    }

    #[inline]
    pub fn row_size(&self) -> usize {
        self.row_size
    }

    #[inline]
    pub fn num_columns(&self) -> usize {
        self.column_types.len()
    }

    #[inline]
    pub fn column_offset(&self, idx: usize) -> usize {
        self.column_offsets[idx]
    }

    #[inline]
    pub fn column_type(&self, idx: usize) -> PhysType {
        self.column_types[idx]
    }
}

/// Arena that stores serialized rows. Monolithic contiguous buffer for MVP;
/// DuckDB uses 256KB blocks but a single Vec with enough up-front reserve
/// has the same cache-hit profile during build and keeps pointers stable
/// during probe.
pub struct RowArena {
    layout: RowLayout,
    /// Packed rows, `n_rows * layout.row_size()` bytes.
    data: Vec<u8>,
    n_rows: usize,
}

impl RowArena {
    pub fn new(layout: RowLayout) -> Self {
        RowArena {
            layout,
            data: Vec::new(),
            n_rows: 0,
        }
    }

    /// Pre-size the arena for an expected row count. After reserving, row
    /// pointers handed out by `append_blank` remain stable as long as
    /// `n_rows` stays ≤ `expected`.
    pub fn reserve(&mut self, expected_rows: usize) {
        let bytes = expected_rows.saturating_mul(self.layout.row_size);
        self.data.reserve(bytes);
    }

    #[inline]
    pub fn layout(&self) -> &RowLayout {
        &self.layout
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.n_rows
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_rows == 0
    }

    /// Append `n` blank rows (zeroed) and return a mutable slice spanning
    /// `n * row_size` bytes. Caller writes column values into the slice
    /// using `RowLayout::column_offset`.
    ///
    /// Panics if `n == 0`. Safe re: Vec realloc: if the arena wasn't
    /// pre-reserved with enough room, this may realloc and invalidate any
    /// previously-handed-out pointers (`row_ptr`). Always call `reserve`
    /// first when the final row count is known.
    pub fn append_blank(&mut self, n: usize) -> &mut [u8] {
        assert!(n > 0);
        let byte_count = n * self.layout.row_size;
        let start = self.data.len();
        self.data.resize(start + byte_count, 0);
        self.n_rows += n;
        &mut self.data[start..start + byte_count]
    }

    /// Pointer to row `i`. Stable only while the arena is not mutated.
    ///
    /// # Safety
    /// The caller must ensure `i < self.len()`. The returned pointer remains
    /// valid until the arena is dropped or reallocated (see `reserve`).
    #[inline]
    pub fn row_ptr(&self, i: usize) -> *const u8 {
        debug_assert!(i < self.n_rows);
        unsafe { self.data.as_ptr().add(i * self.layout.row_size) }
    }

    /// Mutable variant of `row_ptr`. Same safety notes apply.
    #[inline]
    pub fn row_ptr_mut(&mut self, i: usize) -> *mut u8 {
        debug_assert!(i < self.n_rows);
        unsafe { self.data.as_mut_ptr().add(i * self.layout.row_size) }
    }

    /// Pre-size the arena to exactly `n_rows` zeroed rows. After this,
    /// parallel workers can fill disjoint rows via `row_ptr_mut(i)` addresses
    /// computed up front. No realloc happens after this call, so pointers
    /// stay stable.
    pub fn set_len(&mut self, n_rows: usize) {
        let byte_count = n_rows.checked_mul(self.layout.row_size).expect("overflow");
        self.data.resize(byte_count, 0);
        self.n_rows = n_rows;
    }

    /// Base address for parallel writes. Caller turns this into a `*mut u8`
    /// per worker and indexes by `i * row_size`. Returns 0 if the arena is empty.
    #[inline]
    pub fn base_addr(&self) -> usize {
        if self.data.is_empty() {
            0
        } else {
            self.data.as_ptr() as usize
        }
    }
}

#[inline]
pub const fn align_up(x: usize, align: usize) -> usize {
    (x + align - 1) & !(align - 1)
}

/// Header accessors — unsafe because they bypass the slice API; always used
/// in hot paths so non-bounds-checked is intentional.
#[inline(always)]
pub unsafe fn row_hash(row: *const u8) -> u64 {
    std::ptr::read_unaligned(row.add(HEADER_HASH) as *const u64)
}

#[inline(always)]
pub unsafe fn set_row_hash(row: *mut u8, hash: u64) {
    std::ptr::write_unaligned(row.add(HEADER_HASH) as *mut u64, hash);
}

#[inline(always)]
pub unsafe fn row_next_ptr(row: *const u8) -> u64 {
    std::ptr::read_unaligned(row.add(HEADER_NEXT_PTR) as *const u64)
}

#[inline(always)]
pub unsafe fn set_row_next_ptr(row: *mut u8, next: u64) {
    std::ptr::write_unaligned(row.add(HEADER_NEXT_PTR) as *mut u64, next);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_packs_columns() {
        // k (i32), c_bigint (i64), c_flt (f32), c_bool (bool)
        let layout = RowLayout::new(vec![
            PhysType::I32,
            PhysType::I64,
            PhysType::F32,
            PhysType::Bool,
        ]);
        // header = 16
        // i32 at 16 (aligned to 4), size 4 → next at 20
        // i64 aligned to 8 → offset 24, size 8 → next at 32
        // f32 aligned to 4 → offset 32, size 4 → next at 36
        // bool aligned to 1 → offset 36, size 1 → next at 37
        // row padded to 8 → 40
        assert_eq!(layout.column_offset(0), 16);
        assert_eq!(layout.column_offset(1), 24);
        assert_eq!(layout.column_offset(2), 32);
        assert_eq!(layout.column_offset(3), 36);
        assert_eq!(layout.row_size(), 40);
    }

    #[test]
    fn arena_appends_and_reads() {
        let layout = RowLayout::new(vec![PhysType::I32]);
        let mut arena = RowArena::new(layout.clone());
        arena.reserve(4);
        let buf = arena.append_blank(4);
        // Write k = i for row i.
        for i in 0..4 {
            let off = i * layout.row_size() + layout.column_offset(0);
            buf[off..off + 4].copy_from_slice(&(i as i32).to_ne_bytes());
        }
        assert_eq!(arena.len(), 4);
        for i in 0..4 {
            unsafe {
                let p = arena.row_ptr(i).add(layout.column_offset(0)) as *const i32;
                assert_eq!(std::ptr::read_unaligned(p), i as i32);
            }
        }
    }

    #[test]
    fn header_round_trip() {
        let layout = RowLayout::new(vec![PhysType::I32]);
        let mut arena = RowArena::new(layout);
        arena.reserve(2);
        arena.append_blank(2);
        unsafe {
            let p = arena.row_ptr_mut(0);
            set_row_hash(p, 0xDEAD_BEEF_1234_5678);
            set_row_next_ptr(p, 0x0000_AAAA_BBBB_CCCC);
            assert_eq!(row_hash(p), 0xDEAD_BEEF_1234_5678);
            assert_eq!(row_next_ptr(p), 0x0000_AAAA_BBBB_CCCC);
        }
    }

    #[test]
    fn align_up_handles_edges() {
        assert_eq!(align_up(0, 8), 0);
        assert_eq!(align_up(1, 8), 8);
        assert_eq!(align_up(8, 8), 8);
        assert_eq!(align_up(9, 8), 16);
    }

    #[test]
    fn size_of_nextptr_matches_u64() {
        // Sanity: HEADER_SIZE must accommodate two u64 fields.
        assert!(HEADER_SIZE >= 2 * std::mem::size_of::<u64>());
    }
}
