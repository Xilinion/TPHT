/*
 * Insert / lookup / remove latency for both TPHT variants across table sizes.
 *
 * Emits one CSV row per (variant, key width, size, phase).  Sizes sweep powers
 * of two so the cache hierarchy shows up: the point of the flattened variant is
 * that its cost per operation should stay flat once the table leaves cache,
 * while a chained table pays for every extra hop.
 */

#include "tpht.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PHASE_INSERT 0
#define PHASE_LOOKUP_HIT 1
#define PHASE_LOOKUP_MISS 2
#define PHASE_REMOVE 3
#define PHASE_COUNT 4

static const char *phase_name(int phase) {
    switch (phase) {
        case PHASE_INSERT: return "insert";
        case PHASE_LOOKUP_HIT: return "lookup_hit";
        case PHASE_LOOKUP_MISS: return "lookup_miss";
        default: return "remove";
    }
}

static double now_seconds(void) {
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#else
    return (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}

/* Bijective mixers, so a distinct index always yields a distinct key. */
static uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    x ^= x >> 31;
    return x;
}

static uint64_t make_key(uint64_t index, uint8_t key_size) {
    return key_size == 4u ? (uint64_t)mix32((uint32_t)index) : mix64(index);
}

static uint64_t rng_next(uint64_t *state) {
    *state = *state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    return *state;
}

static void shuffle(uint32_t *order, size_t n, uint64_t seed) {
    size_t i;
    for (i = 0; i < n; ++i) order[i] = (uint32_t)i;
    for (i = n; i > 1u; --i) {
        size_t j = (size_t)(rng_next(&seed) % i);
        uint32_t tmp = order[i - 1u];
        order[i - 1u] = order[j];
        order[j] = tmp;
    }
}

/*
 * The timed loops call the concrete functions directly - going through a
 * function pointer would defeat inlining and measure an indirect branch instead
 * of the table.  One loop body per type, expanded by macro.
 */

/* Repeat a read-only phase until it has done at least this many operations. */
#define MIN_TIMED_OPS 4000000u

#define DEFINE_RUN(NAME, P, K, KEY_BYTES, VARIANT_NAME, KEY_BITS)                             \
    static int run_##NAME(uint8_t value_size, size_t n, double load_factor, unsigned reps,    \
                          uint64_t *sink) {                                                   \
        size_t capacity = (size_t)((double)n / load_factor + 0.5);                            \
        double best[PHASE_COUNT];                                                             \
        uint32_t *order = (uint32_t *)malloc(n * sizeof(*order));                             \
        size_t memory_bytes = 0;                                                              \
        size_t grew = 0;                                                                      \
        unsigned rep;                                                                         \
        int phase;                                                                            \
                                                                                              \
        if (!order) return 1;                                                                 \
        for (phase = 0; phase < PHASE_COUNT; ++phase) best[phase] = 1e30;                     \
        shuffle(order, n, UINT64_C(0xc0ffee) ^ n);                                            \
                                                                                              \
        for (rep = 0; rep < reps; ++rep) {                                                    \
            tpht_options_t o = tpht_default_options();                                        \
            P##_t *t;                                                                         \
            size_t passes = (MIN_TIMED_OPS + n - 1u) / n;                                     \
            size_t i, pass, before;                                                           \
            double t0, dt;                                                                    \
            uint64_t acc = 0;                                                                 \
                                                                                              \
            o.resize_mode = TPHT_FIXED;                                                       \
            o.value_size = value_size;                                                         \
            o.hash_seed = UINT64_C(0x9ae16a3b2f90404f);                                       \
            t = P##_make(capacity, &o);                                                       \
            if (!t) { free(order); return 1; }                                                \
            before = P##_memory_bytes(t);                                                      \
                                                                                              \
            t0 = now_seconds();                                                               \
            for (i = 0; i < n; ++i) {                                                         \
                if (P##_insert(t, (K)make_key(i, KEY_BYTES), i) != TPHT_OK) {                 \
                    fprintf(stderr, "insert failed at %zu\n", i);                             \
                    P##_destroy(t); free(order); return 1;                                    \
                }                                                                             \
            }                                                                                 \
            dt = (now_seconds() - t0) / (double)n;                                            \
            if (dt < best[PHASE_INSERT]) best[PHASE_INSERT] = dt;                             \
            memory_bytes = P##_memory_bytes(t);                                                \
            if (memory_bytes != before) grew = 1;                                             \
                                                                                              \
            t0 = now_seconds();                                                               \
            for (pass = 0; pass < passes; ++pass)                                             \
                for (i = 0; i < n; ++i) {                                                     \
                    uint64_t value = 0;                                                       \
                    if (P##_get(t, (K)make_key(order[i], KEY_BYTES), &value) == TPHT_OK)      \
                        acc += value;                                                         \
                }                                                                             \
            dt = (now_seconds() - t0) / (double)(n * passes);                                 \
            if (dt < best[PHASE_LOOKUP_HIT]) best[PHASE_LOOKUP_HIT] = dt;                     \
                                                                                              \
            t0 = now_seconds();                                                               \
            for (pass = 0; pass < passes; ++pass)                                             \
                for (i = 0; i < n; ++i) {                                                     \
                    uint64_t value = 0;                                                       \
                    if (P##_get(t, (K)make_key(n + order[i], KEY_BYTES), &value) == TPHT_OK)  \
                        acc += value;                                                         \
                }                                                                             \
            dt = (now_seconds() - t0) / (double)(n * passes);                                 \
            if (dt < best[PHASE_LOOKUP_MISS]) best[PHASE_LOOKUP_MISS] = dt;                   \
                                                                                              \
            t0 = now_seconds();                                                               \
            for (i = 0; i < n; ++i)                                                           \
                acc += (uint64_t)P##_remove(t, (K)make_key(order[i], KEY_BYTES));             \
            dt = (now_seconds() - t0) / (double)n;                                            \
            if (dt < best[PHASE_REMOVE]) best[PHASE_REMOVE] = dt;                             \
                                                                                              \
            *sink += acc;                                                                     \
            P##_destroy(t);                                                                    \
        }                                                                                     \
                                                                                              \
        for (phase = 0; phase < PHASE_COUNT; ++phase) {                                       \
            printf("%s,%u,%u,%zu,%zu,%.4f,%s,%.2f,%.0f,%zu,%.2f,%zu\n", VARIANT_NAME,         \
                   KEY_BITS, (unsigned)value_size * 8u, n, capacity, load_factor,             \
                   phase_name(phase), best[phase] * 1e9, 1.0 / best[phase], memory_bytes,     \
                   (double)memory_bytes / (double)n, grew);                                   \
            fflush(stdout);                                                                   \
        }                                                                                     \
        free(order);                                                                          \
        return 0;                                                                             \
    }

static flatten_tpht32_t *flatten_tpht32_make(size_t c, const tpht_options_t *o) {
    return flatten_tpht32_create(c, o);
}
static flatten_tpht64_t *flatten_tpht64_make(size_t c, const tpht_options_t *o) {
    return flatten_tpht64_create(c, o);
}
static chained_tpht32_t *chained_tpht32_make(size_t c, const tpht_options_t *o) {
    return chained_tpht32_create(c, 0, o);
}
static chained_tpht64_t *chained_tpht64_make(size_t c, const tpht_options_t *o) {
    return chained_tpht64_create(c, 0, o);
}

DEFINE_RUN(chained32, chained_tpht32, uint32_t, 4u, "chained-tpht", 32u)
DEFINE_RUN(chained64, chained_tpht64, uint64_t, 8u, "chained-tpht", 64u)
DEFINE_RUN(flatten32, flatten_tpht32, uint32_t, 4u, "flatten-tpht", 32u)
DEFINE_RUN(flatten64, flatten_tpht64, uint64_t, 8u, "flatten-tpht", 64u)

typedef int (*run_fn_t)(uint8_t, size_t, double, unsigned, uint64_t *);

int main(int argc, char **argv) {
    static const double load_factors_unused[] = {0.0};
    static const run_fn_t runs[] = {run_chained32, run_chained64, run_flatten32, run_flatten64};
    unsigned min_log2 = 10u;
    unsigned max_log2 = 24u;
    uint8_t value_size = 8u;
    double load_factor = 0.90;
    uint64_t sink = 0;
    unsigned log2n;
    size_t vi;
    int rc = 0;

    (void)load_factors_unused;
    if (argc > 1) min_log2 = (unsigned)strtoul(argv[1], NULL, 10);
    if (argc > 2) max_log2 = (unsigned)strtoul(argv[2], NULL, 10);
    if (argc > 3) value_size = (uint8_t)strtoul(argv[3], NULL, 10);
    if (argc > 4) load_factor = strtod(argv[4], NULL);
    if (min_log2 < 6u) min_log2 = 6u;
    if (max_log2 > 28u) max_log2 = 28u;
    if (max_log2 < min_log2) max_log2 = min_log2;
    if (load_factor <= 0.0 || load_factor > 1.0) load_factor = 0.90;

    puts("variant,key_bits,value_bits,keys,capacity,load_factor,phase,ns_per_op,ops_per_sec,"
         "memory_bytes,bytes_per_key,grew");
    for (vi = 0; vi < sizeof(runs) / sizeof(runs[0]); ++vi) {
        for (log2n = min_log2; log2n <= max_log2; ++log2n) {
            size_t n = (size_t)1u << log2n;
            /*
             * Large tables are the noisy ones, not the small ones: a single
             * pass over a table that misses cache varies by tens of percent
             * run to run, so every size gets several repetitions and the best
             * is reported.
             */
            unsigned reps = n < (1u << 16) ? 5u : (n < (1u << 22) ? 4u : 3u);
            if (runs[vi](value_size, n, load_factor, reps, &sink) != 0) rc = 1;
        }
    }
    if (sink == UINT64_MAX) fputs("", stderr); /* keep the accumulator alive */
    return rc;
}
