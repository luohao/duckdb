#!/usr/bin/env bash
# Generate parquet files for the 5-way FOJ chain bench.
#
# Produces <DIR>/k{K}_o{OVERLAP*100}/{t0..t{K-1}.parquet,joined.parquet}
# for every combination of --ks × --overlaps.
#
# Uses the `kway_bench` C++ tool — the tool runs DuckDB's hash_join timing
# as a side effect, but it also writes the parquet files on disk which the
# Rust chain_bench consumes. --keep preserves them on exit.
#
# After running once, you can re-run all bench variants (DuckDB kway_bench,
# Rust chain_bench --lance / --operator / --streaming / --scan-lance) against
# the same files without regenerating.

set -euo pipefail

DIR="${DIR:-/tmp/kway_4t}"
KS="${KS:-2,5,6}"
OVERLAPS="${OVERLAPS:-0.0,0.5,1.0}"
ROWS="${ROWS:-1000000}"
THREADS="${THREADS:-4}"
REPEATS="${REPEATS:-3}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [ ! -x "build/tools/kway_bench" ]; then
    echo "Building kway_bench..."
    mkdir -p build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release .. >/dev/null
    cmake --build . --target kway_bench -j
    cd ..
fi

mkdir -p "$DIR"
echo "Generating parquet data to $DIR (ks=$KS overlaps=$OVERLAPS rows=$ROWS)"
./build/tools/kway_bench \
    --rows "$ROWS" \
    --ks "$KS" \
    --overlaps "$OVERLAPS" \
    --repeats "$REPEATS" \
    --threads "$THREADS" \
    --dir "$DIR" \
    --keep

echo
echo "Done. Parquet files at: $DIR"
echo "  Per config dir: <DIR>/k{K}_o{OVERLAP*100}/"
echo "    - t0.parquet .. t{K-1}.parquet   (input tables, 11 cols)"
echo "    - joined.parquet                  (pre-materialized FOJ output)"
echo
echo "Run Rust bench:"
echo "  cargo build --release --manifest-path rust/duck-hash-join/Cargo.toml"
echo "  ./rust/duck-hash-join/target/release/chain_bench \\"
echo "      $DIR/k5_o050 5 5 --threads=4 --lance"
