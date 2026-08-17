#include "tpht_test_common.h"

#include <assert.h>

#define TEST_BIN_SIZE 127u
#define TEST_LINE_BYTES 64u

static size_t test_ceil_div(size_t x, size_t div) { return (x + div - 1u) / div; }

static size_t test_pow2_ceil(size_t x) {
    size_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

static size_t test_log2_pow2(size_t x) {
    size_t bits = 0;
    while (x > 1u) { x >>= 1u; ++bits; }
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
    static const uint8_t value_sizes[] = {1u, 3u, 4u, 8u};
    size_t ci, vi;
    int k32;

    for (ci = 0; ci < sizeof(capacities) / sizeof(capacities[0]); ++ci) {
        for (vi = 0; vi < sizeof(value_sizes) / sizeof(value_sizes[0]); ++vi) {
            for (k32 = 0; k32 < 2; ++k32) {
                uint8_t ks = k32 ? 4u : 8u;
                size_t expected = test_chained_storage_bytes(capacities[ci], ks, value_sizes[vi]);
                size_t bytes;
                if (k32) {
                    chained_tpht32_t *t = chained_tpht32_fixed_create(capacities[ci], value_sizes[vi]);
                    assert(t != NULL);
                    bytes = chained_tpht32_memory_bytes(t);
                    chained_tpht32_destroy(t);
                } else {
                    chained_tpht64_t *t = chained_tpht64_fixed_create(capacities[ci], value_sizes[vi]);
                    assert(t != NULL);
                    bytes = chained_tpht64_memory_bytes(t);
                    chained_tpht64_destroy(t);
                }
                /* memory_bytes also counts the table descriptor itself. */
                assert(bytes >= expected);
                assert(bytes - expected <= 1024u);
            }
        }
    }
}

/* The flattened home array is one 64-byte block per few keys. */
static void test_flat_footprint(void) {
    const size_t capacity = 16384u;
    uint8_t vs;
    int k32;
    for (k32 = 0; k32 < 2; ++k32) {
        for (vs = 1u; vs <= 8u; ++vs) {
            double per_key;
            if (k32) {
                flatten_tpht32_t *t = flatten_tpht32_fixed_create(capacity, vs);
                assert(t != NULL);
                per_key = (double)flatten_tpht32_memory_bytes(t) / (double)capacity;
                assert(flatten_tpht32_memory_bytes(t) >= TEST_LINE_BYTES);
                flatten_tpht32_destroy(t);
            } else {
                flatten_tpht64_t *t = flatten_tpht64_fixed_create(capacity, vs);
                assert(t != NULL);
                per_key = (double)flatten_tpht64_memory_bytes(t) / (double)capacity;
                flatten_tpht64_destroy(t);
            }
            assert(per_key > 4.0);
            assert(per_key < 40.0);
        }
    }
}

/* Fill, drain, refill: a leak in the block or free-list bookkeeping shows up as
 * a short second pass. */
static void test_flat_fill_drain_refill(tpht_test_kind_t kind, uint8_t value_size) {
    const size_t capacity = 8192u;
    tpht_test_case_t tc;
    tpht_test_table_t t;
    uint64_t i;

    tc.kind = kind; tc.concurrent = 0; tc.resize_mode = TPHT_FIXED; tc.value_size = value_size;
    t = tpht_test_make_table(&tc, capacity);
    assert(t.handle != NULL);

    for (i = 0; i < capacity; ++i) assert(t.insert(t.handle, i * UINT64_C(2654435761), i) == TPHT_OK);
    assert(t.size(t.handle) == capacity);
    for (i = 0; i < capacity; ++i) tpht_test_assert_get(&t, &tc, i * UINT64_C(2654435761), i);

    for (i = 0; i < capacity; ++i) assert(t.remove(t.handle, i * UINT64_C(2654435761)) == TPHT_OK);
    assert(t.size(t.handle) == 0u);
    for (i = 0; i < capacity; ++i) tpht_test_assert_missing(&t, i * UINT64_C(2654435761));

    for (i = 0; i < capacity; ++i) assert(t.insert(t.handle, i * UINT64_C(40503) + 7u, i) == TPHT_OK);
    assert(t.size(t.handle) == capacity);
    for (i = 0; i < capacity; ++i) tpht_test_assert_get(&t, &tc, i * UINT64_C(40503) + 7u, i);

    t.destroy(t.handle);
}

/* Interleaved removals and insertions drive eviction and re-promotion. */
static void test_flat_block_churn(tpht_test_kind_t kind, uint8_t value_size) {
    enum { KEYS = 3000 };
    tpht_test_case_t tc;
    tpht_test_table_t t;
    uint64_t rng = UINT64_C(0x5eed5eed5eed5eed) ^ ((uint64_t)kind << 8u) ^ value_size;
    uint64_t keys[KEYS];
    int live[KEYS];
    int round, i;

    tc.kind = kind; tc.concurrent = 0; tc.resize_mode = TPHT_FIXED; tc.value_size = value_size;
    t = tpht_test_make_table(&tc, 4096u);
    assert(t.handle != NULL);

    for (i = 0; i < KEYS; ++i) {
        keys[i] = tpht_test_trunc_to(tpht_test_next_rand(&rng), tpht_test_key_size(kind));
        live[i] = 0;
    }

    for (round = 0; round < 4; ++round) {
        for (i = 0; i < KEYS; ++i) {
            if (((i + round) % 3) == 0) {
                if (live[i]) { assert(t.remove(t.handle, keys[i]) == TPHT_OK); live[i] = 0; }
            } else if (!live[i]) {
                tpht_status_t st = t.insert(t.handle, keys[i], keys[i] ^ 0x5a5au);
                assert(st == TPHT_OK);
                live[i] = 1;
            }
        }
        for (i = 0; i < KEYS; ++i) {
            if (live[i]) tpht_test_assert_get(&t, &tc, keys[i], keys[i] ^ 0x5a5au);
            else tpht_test_assert_missing(&t, keys[i]);
        }
    }
    t.destroy(t.handle);
}

/*
 * Single-slot dereference bins run out constantly, which is a hard overflow.
 * A flattened table must absorb it by rebuilding with more blocks rather than
 * reporting the write as failed - fixed capacity or not.
 */
static void test_flat_hard_overflow_growth(tpht_test_kind_t kind, uint8_t value_size,
                                           tpht_resize_mode_t resize_mode) {
    const size_t capacity = 8192u;
    tpht_test_case_t tc;
    tpht_options_t o = tpht_default_options();
    tpht_test_table_t t;
    size_t bytes_before;
    uint64_t rng = UINT64_C(0x0dd1e5) ^ ((uint64_t)kind << 8u) ^ value_size;
    uint64_t keys[8192];
    size_t i;

    tc.kind = kind; tc.concurrent = 0; tc.resize_mode = resize_mode; tc.value_size = value_size;
    o.bin_size = 1; /* one entry per dereference bin */
    t = tpht_test_make_kind(kind, 0, resize_mode, capacity, value_size, &o);
    assert(t.handle != NULL);
    bytes_before = t.memory_bytes(t.handle);

    for (i = 0; i < capacity; ++i) {
        keys[i] = tpht_test_trunc_to(tpht_test_next_rand(&rng), tpht_test_key_size(kind));
        assert(t.put(t.handle, keys[i], i) == TPHT_OK);
    }
    for (i = 0; i < capacity; ++i) tpht_test_assert_get(&t, &tc, keys[i], i);
    assert(t.memory_bytes(t.handle) > bytes_before);
    if (resize_mode == TPHT_FIXED) assert(t.capacity(t.handle) == capacity);
    t.destroy(t.handle);
}

static void test_flat_behavior(void) {
    static const uint8_t value_sizes[] = {1u, 3u, 4u, 8u};
    tpht_test_kind_t kinds[2];
    size_t ki, vi;
    kinds[0] = TPHT_TEST_FLAT32;
    kinds[1] = TPHT_TEST_FLAT64;
    for (ki = 0; ki < 2u; ++ki) {
        for (vi = 0; vi < sizeof(value_sizes) / sizeof(value_sizes[0]); ++vi) {
            test_flat_fill_drain_refill(kinds[ki], value_sizes[vi]);
            test_flat_block_churn(kinds[ki], value_sizes[vi]);
            test_flat_hard_overflow_growth(kinds[ki], value_sizes[vi], TPHT_FIXED);
            test_flat_hard_overflow_growth(kinds[ki], value_sizes[vi], TPHT_RESIZABLE);
        }
    }
}

void tpht_test_run_config_module(void) {
    tpht_options_t o = tpht_default_options();
    flatten_tpht64_t *t;

    /* Values are 1 to 8 bytes; bins hold at most 127 entries. */
    o.value_size = 9;
    assert(flatten_tpht64_create(1024, &o) == NULL);
    o = tpht_default_options();
    o.bin_size = 128;
    assert(flatten_tpht64_create(1024, &o) == NULL);

    /* Zero capacity still yields a usable table. */
    o = tpht_default_options();
    t = flatten_tpht64_create(0, &o);
    assert(t != NULL);
    assert(flatten_tpht64_capacity(t) >= 16u);
    flatten_tpht64_destroy(t);

    test_chained_layout();
    test_flat_footprint();
    test_flat_behavior();
}
