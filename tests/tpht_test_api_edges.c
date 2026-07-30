#include "tpht_test_common.h"

#include <assert.h>

static void run_api_edges_case(const tpht_test_case_t *tc) {
    const size_t fixed_capacity = 16;
    tpht_table_t *t = tpht_test_make_table(tc->variant, tc->threading,
                                           tc->resize_mode, tc->key_size,
                                           tc->value_size, fixed_capacity);
    uint8_t k[8], v[8], out[8];
    uint64_t i;

    assert(t != NULL);
    assert(tpht_put(NULL, k, v) == TPHT_INVALID);
    assert(tpht_put(t, NULL, v) == TPHT_INVALID);
    assert(tpht_put(t, k, NULL) == TPHT_INVALID);
    assert(tpht_get(t, NULL, out) == TPHT_INVALID);
    assert(tpht_get(t, k, NULL) == TPHT_INVALID);
    assert(tpht_remove(t, NULL) == TPHT_INVALID);
    assert(tpht_update(t, NULL, v) == TPHT_INVALID);

    tpht_test_put_bytes(k, tc->key_size, 42u);
    tpht_test_put_bytes(v, tc->value_size, 100u);
    assert(tpht_update(t, k, v) == TPHT_NOT_FOUND);
    assert(tpht_remove(t, k) == TPHT_NOT_FOUND);
    assert(tpht_insert(t, k, v) == TPHT_OK);
    assert(tpht_insert(t, k, v) == TPHT_EXISTS);
    tpht_test_put_bytes(v, tc->value_size, 200u);
    assert(tpht_update(t, k, v) == TPHT_OK);
    tpht_test_assert_get(t, tc->key_size, tc->value_size, 42u, 200u);
    assert(tpht_remove(t, k) == TPHT_OK);
    tpht_test_assert_missing(t, tc->key_size, 42u);

    if (tc->resize_mode == TPHT_FIXED) {
        for (i = 0; i < (uint64_t)fixed_capacity; ++i) {
            tpht_test_put_bytes(k, tc->key_size, i + 1000u);
            tpht_test_put_bytes(v, tc->value_size, i + 9000u);
            assert(tpht_insert(t, k, v) == TPHT_OK);
        }
        tpht_test_put_bytes(k, tc->key_size, 999999u);
        tpht_test_put_bytes(v, tc->value_size, 1u);
        assert(tpht_insert(t, k, v) == TPHT_FULL);
        tpht_test_put_bytes(k, tc->key_size, 1000u);
        tpht_test_put_bytes(v, tc->value_size, 77u);
        assert(tpht_put(t, k, v) == TPHT_OK);
        tpht_test_assert_get(t, tc->key_size, tc->value_size, 1000u, 77u);
    }

    tpht_destroy(t);
}

void tpht_test_run_api_edges_module(void) {
    tpht_test_for_each_case(run_api_edges_case);
}
