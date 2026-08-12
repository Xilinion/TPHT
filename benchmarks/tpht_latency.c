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

static tpht_table_t *make_table(tpht_variant_t variant, size_t capacity, uint8_t key_size,
                                uint8_t value_size) {
    tpht_config_t c = tpht_default_config();
    c.variant = variant;
    c.threading = TPHT_SEQUENTIAL;
    c.resize_mode = TPHT_FIXED;
    c.initial_capacity = capacity;
    c.key_size = key_size;
    c.value_size = value_size;
    c.hash_seed = UINT64_C(0x9ae16a3b2f90404f);
    return tpht_create(&c);
}

/* Repeat a read-only phase until it has done at least this many operations. */
#define MIN_TIMED_OPS 4000000u

static int run_size(tpht_variant_t variant, uint8_t key_size, uint8_t value_size, size_t n,
                    double load_factor, unsigned reps, uint64_t *sink) {
    size_t capacity = (size_t)((double)n / load_factor + 0.5);
    double best[PHASE_COUNT];
    uint32_t *order = (uint32_t *)malloc(n * sizeof(*order));
    size_t memory_bytes = 0;
    size_t grew = 0;
    unsigned rep;
    int phase;

    if (!order) return 1;
    for (phase = 0; phase < PHASE_COUNT; ++phase) best[phase] = 1e30;
    shuffle(order, n, UINT64_C(0xc0ffee) ^ n);

    for (rep = 0; rep < reps; ++rep) {
        tpht_table_t *t = make_table(variant, capacity, key_size, value_size);
        size_t passes = (MIN_TIMED_OPS + n - 1u) / n;
        size_t i, pass;
        double t0, dt;
        uint64_t acc = 0;
        size_t before;

        if (!t) {
            free(order);
            return 1;
        }
        before = tpht_memory_bytes(t);

        t0 = now_seconds();
        for (i = 0; i < n; ++i) {
            if (tpht_insert(t, make_key(i, key_size), i) != TPHT_OK) {
                fprintf(stderr, "insert failed at %zu\n", i);
                tpht_destroy(t);
                free(order);
                return 1;
            }
        }
        dt = (now_seconds() - t0) / (double)n;
        if (dt < best[PHASE_INSERT]) best[PHASE_INSERT] = dt;

        memory_bytes = tpht_memory_bytes(t);
        if (memory_bytes != before) grew = 1;

        t0 = now_seconds();
        for (pass = 0; pass < passes; ++pass) {
            for (i = 0; i < n; ++i) {
                uint64_t value = 0;
                if (tpht_get(t, make_key(order[i], key_size), &value) == TPHT_OK) acc += value;
            }
        }
        dt = (now_seconds() - t0) / (double)(n * passes);
        if (dt < best[PHASE_LOOKUP_HIT]) best[PHASE_LOOKUP_HIT] = dt;

        t0 = now_seconds();
        for (pass = 0; pass < passes; ++pass) {
            for (i = 0; i < n; ++i) {
                uint64_t value = 0;
                if (tpht_get(t, make_key(n + order[i], key_size), &value) == TPHT_OK) acc += value;
            }
        }
        dt = (now_seconds() - t0) / (double)(n * passes);
        if (dt < best[PHASE_LOOKUP_MISS]) best[PHASE_LOOKUP_MISS] = dt;

        t0 = now_seconds();
        for (i = 0; i < n; ++i) acc += (uint64_t)tpht_remove(t, make_key(order[i], key_size));
        dt = (now_seconds() - t0) / (double)n;
        if (dt < best[PHASE_REMOVE]) best[PHASE_REMOVE] = dt;

        *sink += acc;
        tpht_destroy(t);
    }

    for (phase = 0; phase < PHASE_COUNT; ++phase) {
        printf("%s,%u,%u,%zu,%zu,%.4f,%s,%.2f,%.0f,%zu,%.2f,%zu\n",
               variant == TPHT_CHAINED ? "chained-tpht" : "flatten-tpht",
               (unsigned)key_size * 8u, (unsigned)value_size * 8u, n, capacity, load_factor,
               phase_name(phase), best[phase] * 1e9, 1.0 / best[phase], memory_bytes,
               (double)memory_bytes / (double)n, grew);
        fflush(stdout);
    }
    free(order);
    return 0;
}

int main(int argc, char **argv) {
    static const tpht_variant_t variants[] = {TPHT_CHAINED, TPHT_FLATTEN};
    static const uint8_t key_sizes[] = {4u, 8u};
    unsigned min_log2 = 10u;
    unsigned max_log2 = 24u;
    uint8_t value_size = 8u;
    double load_factor = 0.90;
    uint64_t sink = 0;
    unsigned log2n;
    size_t vi, ki;
    int rc = 0;

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
    for (vi = 0; vi < sizeof(variants) / sizeof(variants[0]); ++vi) {
        for (ki = 0; ki < sizeof(key_sizes) / sizeof(key_sizes[0]); ++ki) {
            for (log2n = min_log2; log2n <= max_log2; ++log2n) {
                size_t n = (size_t)1u << log2n;
                /* More repetitions for small tables, where noise dominates. */
                unsigned reps = n < (1u << 16) ? 5u : (n < (1u << 21) ? 3u : 1u);
                if (run_size(variants[vi], key_sizes[ki], value_size, n, load_factor, reps,
                             &sink) != 0) {
                    rc = 1;
                }
            }
        }
    }
    if (sink == UINT64_MAX) fputs("", stderr); /* keep the accumulator alive */
    return rc;
}
