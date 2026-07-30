#include "tpht_test_common.h"

#include <assert.h>
#include <string.h>

uint64_t tpht_test_mask_for_size(uint8_t size) {
    return size == 8u ? UINT64_MAX : ((UINT64_C(1) << (8u * size)) - 1u);
}

uint64_t tpht_test_trunc_to(uint64_t x, uint8_t size) {
    return x & tpht_test_mask_for_size(size);
}

void tpht_test_put_bytes(uint8_t *dst, uint8_t size, uint64_t x) {
    uint8_t i;
    for (i = 0; i < size; ++i) dst[i] = (uint8_t)(x >> (8u * i));
}

uint64_t tpht_test_get_bytes(const uint8_t *src, uint8_t size) {
    uint64_t x = 0;
    uint8_t i;
    for (i = 0; i < size; ++i) x |= (uint64_t)src[i] << (8u * i);
    return x;
}

uint64_t tpht_test_next_rand(uint64_t *state) {
    *state = *state * UINT64_C(6364136223846793005) +
             UINT64_C(1442695040888963407);
    return *state;
}

size_t tpht_test_model_find(tpht_test_model_entry_t *model, size_t n,
                            uint64_t key) {
    size_t i;
    for (i = 0; i < n; ++i) {
        if (model[i].live && model[i].key == key) return i;
    }
    return n;
}

tpht_table_t *tpht_test_make_table(tpht_variant_t variant,
                                   tpht_threading_t threading,
                                   tpht_resize_mode_t resize_mode,
                                   uint8_t key_size, uint8_t value_size,
                                   size_t capacity) {
    tpht_config_t c = tpht_default_config();
    c.variant = variant;
    c.threading = threading;
    c.resize_mode = resize_mode;
    c.initial_capacity = capacity;
    c.key_size = key_size;
    c.value_size = value_size;
    c.max_load_factor = 0.70;
    c.hash_seed = UINT64_C(0x123456789abcdef0);
    return tpht_create(&c);
}

void tpht_test_assert_get(tpht_table_t *table, uint8_t key_size,
                          uint8_t value_size, uint64_t key,
                          uint64_t expected) {
    uint8_t kb[8], vb[8];
    memset(vb, 0xa5, sizeof(vb));
    tpht_test_put_bytes(kb, key_size, key);
    assert(tpht_get(table, kb, vb) == TPHT_OK);
    assert(tpht_test_get_bytes(vb, value_size) ==
           tpht_test_trunc_to(expected, value_size));
}

void tpht_test_assert_missing(tpht_table_t *table, uint8_t key_size,
                              uint64_t key) {
    uint8_t kb[8], vb[8];
    tpht_test_put_bytes(kb, key_size, key);
    assert(tpht_get(table, kb, vb) == TPHT_NOT_FOUND);
}

void tpht_test_for_each_case(void (*fn)(const tpht_test_case_t *tc)) {
    uint8_t sizes[] = {2u, 4u, 8u};
    tpht_variant_t variants[] = {TPHT_CHAINED, TPHT_FLATTEN};
    tpht_threading_t threadings[] = {TPHT_SEQUENTIAL, TPHT_CONCURRENT};
    tpht_resize_mode_t resize_modes[] = {TPHT_FIXED, TPHT_RESIZABLE};
    size_t vi, ti, ri, ki, vali;

    for (vi = 0; vi < 2u; ++vi) {
        for (ti = 0; ti < 2u; ++ti) {
            for (ri = 0; ri < 2u; ++ri) {
                for (ki = 0; ki < 3u; ++ki) {
                    for (vali = 0; vali < 3u; ++vali) {
                        tpht_test_case_t tc;
                        tc.variant = variants[vi];
                        tc.threading = threadings[ti];
                        tc.resize_mode = resize_modes[ri];
                        tc.key_size = sizes[ki];
                        tc.value_size = sizes[vali];
                        fn(&tc);
                    }
                }
            }
        }
    }
}
