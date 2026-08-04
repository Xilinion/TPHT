#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
CC=${CC:-cc}
BUILD_DIR=${BUILD_DIR:-"$ROOT/build-bench"}
RESULT_DIR=${RESULT_DIR:-"$ROOT/results"}
ENTRIES=${1:-200000}

mkdir -p "$BUILD_DIR" "$RESULT_DIR"

$CC -std=c11 -O2 -Wall -Wextra -Werror -I"$ROOT" \
    "$ROOT/tpht.c" "$ROOT/benchmarks/tpht_space_eff.c" \
    -o "$BUILD_DIR/tpht_space_eff"

"$BUILD_DIR/tpht_space_eff" "$ENTRIES" | tee "$RESULT_DIR/tpht_space_eff.csv"
