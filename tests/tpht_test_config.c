#include "tpht_test_common.h"

#include <assert.h>

#define TEST_BIN_SIZE 127u

static size_t test_ceil_div(size_t x, size_t div) { return (x + div - 1u) / div; }

static size_t test_pow2_ceil(size_t x) {
    size_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

static size_t test_log2_pow2(size_t x) {
    size_t bits = 0;
    while (x > 1u) {
        x >>= 1u;
        ++bits;
    }
    return bits;
}

/* Mirrors the chained storage formulas in tpht.c. */
static size_t test_chained_storage_bytes(size_t capacity, uint8_t key_size, uint8_t value_size) {
    size_t base_count = test_pow2_ceil(capacity);
    size_t quotient_bytes = test_ceil_div(key_size * 8u - test_log2_pow2(base_count), 8u);
    size_t entry_size = 1u + quotient_bytes + value_size;
    size_t slots = test_ceil_div(capacity * 100u, 95u);
    size_t bins = test_ceil_div(slots, TEST_BIN_SIZE);
    return base_count + bins * TEST_BIN_SIZE * entry_size + bins * 2u;
}

static void test_chained_layout(void) {
    static const size_t capacities[] = {1024u, 8192u, 65536u};
    static const uint8_t key_sizes[] = {4u, 8u};
    static const uint8_t value_sizes[] = {1u, 3u, 4u, 8u};
    size_t ci, ki, vi;

    for (ci = 0; ci < sizeof(capacities) / sizeof(capacities[0]); ++ci) {
        for (ki = 0; ki < sizeof(key_sizes) / sizeof(key_sizes[0]); ++ki) {
            for (vi = 0; vi < sizeof(value_sizes) / sizeof(value_sizes[0]); ++vi) {
                size_t expected =
                    test_chained_storage_bytes(capacities[ci], key_sizes[ki], value_sizes[vi]);
                tpht_table_t *t =
                    key_sizes[ki] == 4u
                        ? chained_tpht32_fixed_create(capacities[ci], value_sizes[vi])
                        : chained_tpht64_fixed_create(capacities[ci], value_sizes[vi]);
                size_t bytes;
                assert(t != NULL);
                bytes = tpht_memory_bytes(t);
                /* memory_bytes also counts the table descriptor itself. */
                assert(bytes >= expected);
                assert(bytes - expected <= 1024u);
                tpht_destroy(t);
            }
        }
    }
}

/*
 * The flattened home array is one 64-byte block per few keys, so its footprint
 * per key must stay in a narrow band no matter what the key and value sizes
 * are.
 */
static void test_flat_footprint(void) {
    static const uint8_t key_sizes[] = {4u, 8u};
    size_t ki;
    uint8_t vs;

    for (ki = 0; ki < sizeof(key_sizes) / sizeof(key_sizes[0]); ++ki) {
        for (vs = 1u; vs <= 8u; ++vs) {
            const size_t capacity = 16384u;
            tpht_table_t *t = key_sizes[ki] == 4u ? flatten_tpht32_fixed_create(capacity, vs)
                                                  : flatten_tpht64_fixed_create(capacity, vs);
            double per_key;
            assert(t != NULL);
            per_key = (double)tpht_memory_bytes(t) / (double)capacity;
            /* At most one block per key, plus the dereference table. */
            assert(per_key > 4.0);
            assert(per_key < 40.0);
            tpht_destroy(t);
        }
    }
}

/*
 * Fill a fixed flattened table, drain it, and fill it again.  A leak in the
 * home-block bookkeeping or in the dereference table free lists shows up as a
 * short second pass.
 */
static void test_flat_fill_drain_refill(uint8_t key_size, uint8_t value_size) {
    const size_t capacity = 8192u;
    tpht_test_case_t tc;
    tpht_table_t *t = key_size == 4u ? flatten_tpht32_fixed_create(capacity, value_size)
                                     : flatten_tpht64_fixed_create(capacity, value_size);
    uint64_t i;

    tc.variant = TPHT_FLATTEN;
    tc.threading = TPHT_SEQUENTIAL;
    tc.resize_mode = TPHT_FIXED;
    tc.key_size = key_size;
    tc.value_size = value_size;
    assert(t != NULL);

    for (i = 0; i < capacity; ++i) {
        assert(tpht_insert(t, i * UINT64_C(2654435761), i) == TPHT_OK);
    }
    assert(tpht_size(t) == capacity);
    for (i = 0; i < capacity; ++i) tpht_test_assert_get(t, &tc, i * UINT64_C(2654435761), i);

    for (i = 0; i < capacity; ++i) {
        assert(tpht_remove(t, i * UINT64_C(2654435761)) == TPHT_OK);
    }
    assert(tpht_size(t) == 0u);
    for (i = 0; i < capacity; ++i) tpht_test_assert_missing(t, i * UINT64_C(2654435761));

    for (i = 0; i < capacity; ++i) {
        assert(tpht_insert(t, i * UINT64_C(40503) + 7u, i) == TPHT_OK);
    }
    assert(tpht_size(t) == capacity);
    for (i = 0; i < capacity; ++i) tpht_test_assert_get(t, &tc, i * UINT64_C(40503) + 7u, i);

    tpht_destroy(t);
}

/*
 * Interleaved removals and insertions in the same home blocks drive crystal
 * eviction and re-promotion, which is where the layout bookkeeping is easiest
 * to get wrong.
 */
static void test_flat_block_churn(uint8_t key_size, uint8_t value_size) {
    enum { KEYS = 3000 };
    const size_t capacity = 4096u;
    tpht_test_case_t tc;
    tpht_table_t *t = key_size == 4u ? flatten_tpht32_fixed_create(capacity, value_size)
                                     : flatten_tpht64_fixed_create(capacity, value_size);
    uint64_t rng = UINT64_C(0x5eed5eed5eed5eed) ^ ((uint64_t)key_size << 8u) ^ value_size;
    uint64_t keys[KEYS];
    int live[KEYS];
    int round;
    int i;

    tc.variant = TPHT_FLATTEN;
    tc.threading = TPHT_SEQUENTIAL;
    tc.resize_mode = TPHT_FIXED;
    tc.key_size = key_size;
    tc.value_size = value_size;
    assert(t != NULL);

    for (i = 0; i < KEYS; ++i) {
        keys[i] = tpht_test_trunc_to(tpht_test_next_rand(&rng), key_size);
        live[i] = 0;
    }

    for (round = 0; round < 4; ++round) {
        for (i = 0; i < KEYS; ++i) {
            if (((i + round) % 3) == 0) {
                if (live[i]) {
                    assert(tpht_remove(t, keys[i]) == TPHT_OK);
                    live[i] = 0;
                }
            } else if (!live[i]) {
                tpht_status_t st = tpht_insert(t, keys[i], keys[i] ^ 0x5a5au);
                assert(st == TPHT_OK || st == TPHT_EXISTS);
                live[i] = 1;
            }
        }
        for (i = 0; i < KEYS; ++i) {
            if (live[i]) {
                tpht_test_assert_get(t, &tc, keys[i], keys[i] ^ 0x5a5au);
            } else {
                tpht_test_assert_missing(t, keys[i]);
            }
        }
    }

    tpht_destroy(t);
}

/*
 * A dereference table with single-slot bins runs out constantly, which is a
 * hard overflow.  The table must absorb it by rebuilding itself with more home
 * blocks rather than reporting the write as failed - fixed capacity or not.
 */
static void test_flat_hard_overflow_growth(uint8_t key_size, uint8_t value_size,
                                           tpht_resize_mode_t resize_mode) {
    const size_t capacity = 8192u;
    tpht_test_case_t tc;
    tpht_config_t c = tpht_default_config();
    tpht_table_t *t;
    size_t bytes_before;
    uint64_t rng = UINT64_C(0x0dd1e5) ^ ((uint64_t)key_size << 8u) ^ value_size;
    uint64_t keys[8192];
    size_t i;

    tc.variant = TPHT_FLATTEN;
    tc.threading = TPHT_SEQUENTIAL;
    tc.resize_mode = resize_mode;
    tc.key_size = key_size;
    tc.value_size = value_size;

    c.variant = TPHT_FLATTEN;
    c.threading = TPHT_SEQUENTIAL;
    c.resize_mode = resize_mode;
    c.initial_capacity = capacity;
    c.key_size = key_size;
    c.value_size = value_size;
    c.bin_size = 1; /* one entry per dereference bin */
    t = tpht_create(&c);
    assert(t != NULL);
    bytes_before = tpht_memory_bytes(t);

    for (i = 0; i < capacity; ++i) {
        keys[i] = tpht_test_trunc_to(tpht_test_next_rand(&rng), key_size);
        assert(tpht_put(t, keys[i], i) == TPHT_OK);
    }
    for (i = 0; i < capacity; ++i) tpht_test_assert_get(t, &tc, keys[i], i);
    /* Absorbing the overflow means the table got bigger. */
    assert(tpht_memory_bytes(t) > bytes_before);
    if (resize_mode == TPHT_FIXED) assert(tpht_capacity(t) == capacity);
    tpht_destroy(t);
}

static void test_flat_behavior(void) {
    static const uint8_t key_sizes[] = {4u, 8u};
    static const uint8_t value_sizes[] = {1u, 3u, 4u, 8u};
    size_t ki, vi;
    for (ki = 0; ki < sizeof(key_sizes) / sizeof(key_sizes[0]); ++ki) {
        for (vi = 0; vi < sizeof(value_sizes) / sizeof(value_sizes[0]); ++vi) {
            test_flat_fill_drain_refill(key_sizes[ki], value_sizes[vi]);
            test_flat_block_churn(key_sizes[ki], value_sizes[vi]);
            test_flat_hard_overflow_growth(key_sizes[ki], value_sizes[vi], TPHT_FIXED);
            test_flat_hard_overflow_growth(key_sizes[ki], value_sizes[vi], TPHT_RESIZABLE);
        }
    }
}

void tpht_test_run_config_module(void) {
    tpht_config_t c = tpht_default_config();
    tpht_table_t *t;

    /* Keys are 4 or 8 bytes; values are 1 to 8 bytes. */
    c.key_size = 2;
    assert(tpht_create(&c) == NULL);
    c.key_size = 3;
    assert(tpht_create(&c) == NULL);
    c.key_size = 0;
    assert(tpht_create(&c) == NULL);
    c.key_size = 16;
    assert(tpht_create(&c) == NULL);

    c = tpht_default_config();
    c.value_size = 0;
    assert(tpht_create(&c) == NULL);
    c.value_size = 9;
    assert(tpht_create(&c) == NULL);

    c = tpht_default_config();
    c.bin_size = 128;
    assert(tpht_create(&c) == NULL);

    c = tpht_default_config();
    c.variant = (tpht_variant_t)0;
    assert(tpht_create(&c) == NULL);

    /* The flattened variant is sequential only. */
    c = tpht_default_config();
    c.variant = TPHT_FLATTEN;
    c.threading = TPHT_CONCURRENT;
    assert(tpht_create(&c) == NULL);
    c.threading = TPHT_SEQUENTIAL;
    c.resize_mode = TPHT_RESIZABLE;
    t = tpht_create(&c);
    assert(t != NULL);
    tpht_destroy(t);
    c.resize_mode = TPHT_FIXED;
    t = tpht_create(&c);
    assert(t != NULL);
    tpht_destroy(t);

    c = tpht_default_config();
    c.initial_capacity = 0;
    t = tpht_create(&c);
    assert(t != NULL);
    assert(tpht_capacity(t) >= 16u);
    tpht_destroy(t);

    test_chained_layout();
    test_flat_footprint();
    test_flat_behavior();
}
