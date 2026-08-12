#include "tpht_test_common.h"

#include <assert.h>

uint64_t tpht_test_mask_for_size(uint8_t size) {
    return size >= 8u ? UINT64_MAX : ((UINT64_C(1) << (8u * size)) - 1u);
}

uint64_t tpht_test_trunc_to(uint64_t x, uint8_t size) {
    return x & tpht_test_mask_for_size(size);
}

uint64_t tpht_test_next_rand(uint64_t *state) {
    *state = *state * UINT64_C(6364136223846793005) +
             UINT64_C(1442695040888963407);
    return *state;
}

size_t tpht_test_model_find(tpht_test_model_entry_t *model, size_t n, uint64_t key) {
    size_t i;
    for (i = 0; i < n; ++i) {
        if (model[i].live && model[i].key == key) return i;
    }
    return n;
}

int tpht_test_case_supported(const tpht_test_case_t *tc) {
    if (tc->key_size != 4u && tc->key_size != 8u) return 0;
    if (tc->value_size == 0u || tc->value_size > 8u) return 0;
    if (tc->variant != TPHT_FLATTEN) return 1;
    /* The flattened variant is sequential only. */
    return tc->threading == TPHT_SEQUENTIAL;
}

tpht_table_t *tpht_test_make_table(const tpht_test_case_t *tc, size_t capacity) {
    tpht_config_t c = tpht_default_config();
    c.variant = tc->variant;
    c.threading = tc->threading;
    c.resize_mode = tc->resize_mode;
    c.initial_capacity = capacity;
    c.key_size = tc->key_size;
    c.value_size = tc->value_size;
    c.max_load_factor = 0.70;
    c.hash_seed = UINT64_C(0x123456789abcdef0);
    return tpht_create(&c);
}

void tpht_test_assert_get(tpht_table_t *table, const tpht_test_case_t *tc, uint64_t key,
                          uint64_t expected) {
    uint64_t value = UINT64_C(0xa5a5a5a5a5a5a5a5);
    assert(tpht_get(table, key, &value) == TPHT_OK);
    assert(value == tpht_test_trunc_to(expected, tc->value_size));
}

void tpht_test_assert_missing(tpht_table_t *table, uint64_t key) {
    uint64_t value = 0;
    assert(tpht_get(table, key, &value) == TPHT_NOT_FOUND);
}

void tpht_test_for_each_case(void (*fn)(const tpht_test_case_t *tc)) {
    uint8_t key_sizes[] = {4u, 8u};
    uint8_t value_sizes[] = {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u};
    tpht_variant_t variants[] = {TPHT_CHAINED, TPHT_FLATTEN};
    tpht_threading_t threadings[] = {TPHT_SEQUENTIAL, TPHT_CONCURRENT};
    tpht_resize_mode_t resize_modes[] = {TPHT_FIXED, TPHT_RESIZABLE};
    size_t vi, ti, ri, ki, si;

    for (vi = 0; vi < 2u; ++vi) {
        for (ti = 0; ti < 2u; ++ti) {
            for (ri = 0; ri < 2u; ++ri) {
                for (ki = 0; ki < sizeof(key_sizes) / sizeof(key_sizes[0]); ++ki) {
                    for (si = 0; si < sizeof(value_sizes) / sizeof(value_sizes[0]); ++si) {
                        tpht_test_case_t tc;
                        tc.variant = variants[vi];
                        tc.threading = threadings[ti];
                        tc.resize_mode = resize_modes[ri];
                        tc.key_size = key_sizes[ki];
                        tc.value_size = value_sizes[si];
                        if (!tpht_test_case_supported(&tc)) continue;
                        fn(&tc);
                    }
                }
            }
        }
    }
}
