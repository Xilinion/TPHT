#!/usr/bin/env sh
set -eu

# Usage: benchmarks/run_latency.sh [min_log2] [max_log2] [value_bytes] [load_factor]
#
# Sweeps table sizes from 2^min_log2 to 2^max_log2 keys for both variants and
# both key widths, writing results/tpht_latency.csv and, when matplotlib is
# available, a vector PDF plot beside it.

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
CC=${CC:-cc}
BUILD_DIR=${BUILD_DIR:-"$ROOT/build-bench"}
RESULT_DIR=${RESULT_DIR:-"$ROOT/results"}
PYTHON=${PYTHON:-python3}

MIN_LOG2=${1:-10}
MAX_LOG2=${2:-24}
VALUE_BYTES=${3:-8}
LOAD_FACTOR=${4:-0.90}

mkdir -p "$BUILD_DIR" "$RESULT_DIR"

CFLAGS=${CFLAGS:-"-O3"}
echo "[tpht] building latency benchmark"
# shellcheck disable=SC2086
$CC -std=c11 $CFLAGS -Wall -Wextra -Werror -I"$ROOT" \
    "$ROOT/tpht.c" "$ROOT/benchmarks/tpht_latency.c" \
    -o "$BUILD_DIR/tpht_latency"

echo "[tpht] sweeping 2^$MIN_LOG2 .. 2^$MAX_LOG2 keys, ${VALUE_BYTES}-byte values, load $LOAD_FACTOR"
"$BUILD_DIR/tpht_latency" "$MIN_LOG2" "$MAX_LOG2" "$VALUE_BYTES" "$LOAD_FACTOR" \
    > "$RESULT_DIR/tpht_latency.csv"
echo "[tpht] wrote $RESULT_DIR/tpht_latency.csv"

if "$PYTHON" -c "import matplotlib" >/dev/null 2>&1; then
    "$PYTHON" "$ROOT/benchmarks/plot_latency.py" \
        "$RESULT_DIR/tpht_latency.csv" "$RESULT_DIR"
else
    echo "[tpht] skip plots: $PYTHON has no matplotlib"
fi
