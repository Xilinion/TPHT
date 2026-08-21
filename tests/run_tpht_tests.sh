#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
CC=${CC:-cc}
BUILD_DIR=${BUILD_DIR:-"$ROOT/build-tests"}
COMMON_FLAGS="-std=c11 -Wall -Wextra -Werror -O2 -I$ROOT"
TEST_SRCS="$ROOT/tests/tpht_test.c $ROOT/tests/tpht_test_common.c $ROOT/tests/tpht_test_config.c $ROOT/tests/tpht_test_constructors.c $ROOT/tests/tpht_test_api_edges.c $ROOT/tests/tpht_test_deterministic.c $ROOT/tests/tpht_test_random_model.c $ROOT/tests/tpht_test_saturation.c $ROOT/tests/tpht_test_churn.c $ROOT/tests/tpht_test_threads.c"

mkdir -p "$BUILD_DIR"

run_case() {
    name=$1
    flags=$2
    libs=${3:-}
    out="$BUILD_DIR/$name"
    echo "[tpht] build $name"
    # shellcheck disable=SC2086
    $CC $COMMON_FLAGS $flags "$ROOT/tpht.c" $TEST_SRCS $libs -o "$out"
    echo "[tpht] run $name"
    "$out"
}

can_preprocess() {
    probe=$1
    printf '%s\n' "$probe" | $CC $COMMON_FLAGS -x c - -c -o "$BUILD_DIR/probe.o" >/dev/null 2>&1
}

run_case scalar_fallback "-DTPHT_ENABLE_SIMD=0"
run_case portable_simd "-DTPHT_ENABLE_SIMD=1"

if can_preprocess '#ifndef __SSE2__
#error no_sse2
#endif
int main(void){return 0;}'; then
    run_case forced_sse2 "-DTPHT_ENABLE_SIMD=1 -DTPHT_SIMD_MODE=2"
else
    echo "[tpht] skip forced_sse2: target does not define __SSE2__"
fi

if printf '%s\n' '#ifndef __AVX2__
#error no_avx2
#endif
int main(void){return 0;}' | $CC $COMMON_FLAGS -mavx2 -x c - -c -o "$BUILD_DIR/probe_avx2.o" >/dev/null 2>&1; then
    if $CC $COMMON_FLAGS -mavx2 -DTPHT_ENABLE_SIMD=1 -DTPHT_SIMD_MODE=3 \
        "$ROOT/tpht.c" $TEST_SRCS -o "$BUILD_DIR/forced_avx2" >/dev/null 2>&1; then
        :
    else
        echo "[tpht] skip forced_avx2: TPHT AVX2 build failed"
        forced_avx2_failed=1
    fi
    if [ "${forced_avx2_failed:-0}" = 0 ]; then
    if "$BUILD_DIR/forced_avx2" >/dev/null 2>&1; then
        echo "[tpht] run forced_avx2"
        "$BUILD_DIR/forced_avx2"
    else
        echo "[tpht] skip forced_avx2 run: binary built, CPU/runtime does not support AVX2"
    fi
    fi
else
    echo "[tpht] skip forced_avx2: compiler/target does not support AVX2"
fi

if echo 'int main(void){return 0;}' | $CC $COMMON_FLAGS -mavx512bw -mavx512vl -mbmi2 -x c - -c -o "$BUILD_DIR/probe_avx512.o" >/dev/null 2>&1; then
    if $CC $COMMON_FLAGS -mavx512bw -mavx512vl -mbmi2 -DTPHT_ENABLE_SIMD=1 -DTPHT_SIMD_MODE=5 \
        "$ROOT/tpht.c" $TEST_SRCS -o "$BUILD_DIR/forced_avx512" >/dev/null 2>&1; then
        if "$BUILD_DIR/forced_avx512" >/dev/null 2>&1; then
            echo "[tpht] run forced_avx512"
            "$BUILD_DIR/forced_avx512"
        else
            echo "[tpht] skip forced_avx512 run: binary built, CPU/runtime does not support AVX-512BW/VL"
        fi
    else
        echo "[tpht] skip forced_avx512: TPHT AVX-512 build failed"
    fi
else
    echo "[tpht] skip forced_avx512: compiler/target does not support AVX-512BW/VL"
fi

if can_preprocess '#ifndef __ARM_NEON
#error no_neon
#endif
int main(void){return 0;}'; then
    run_case forced_neon "-DTPHT_ENABLE_SIMD=1 -DTPHT_SIMD_MODE=4"
else
    echo "[tpht] skip forced_neon: target does not define __ARM_NEON"
fi

if $CC $COMMON_FLAGS -DTPHT_ENABLE_SIMD=1 -DTPHT_TEST_WITH_THREADS \
    "$ROOT/tpht.c" $TEST_SRCS -pthread -o "$BUILD_DIR/threaded" >/dev/null 2>&1; then
    echo "[tpht] run threaded"
    "$BUILD_DIR/threaded"
else
    echo "[tpht] skip threaded: compiler/libc does not accept -pthread"
fi

if $CC $COMMON_FLAGS -march=native -DTPHT_ENABLE_SIMD=1 \
    "$ROOT/tpht.c" $TEST_SRCS -o "$BUILD_DIR/native" >/dev/null 2>&1; then
    echo "[tpht] run native"
    "$BUILD_DIR/native"
    $CC $COMMON_FLAGS -march=native -DTPHT_ENABLE_SIMD=1 -c "$ROOT/tpht.c" -o "$BUILD_DIR/tripwire.o" >/dev/null 2>&1 \
        && "$ROOT/tools/check_config_branches.sh" "$BUILD_DIR/tripwire.o"
else
    echo "[tpht] skip native: compiler does not accept -march=native"
fi

echo "[tpht] all requested test modes completed"
