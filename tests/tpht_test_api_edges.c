#include "tpht_test_common.h"

#include <assert.h>

static void run_api_edges_case(const tpht_test_case_t *tc) {
    const size_t fixed_capacity = 16;
    tpht_table_t *t = tpht_test_make_table(tc, fixed_capacity);
    uint64_t out;
    uint64_t i;

    assert(t != NULL);
    assert(tpht_put(NULL, 1, 1) == TPHT_INVALID);
    assert(tpht_get(NULL, 1, &out) == TPHT_INVALID);
    assert(tpht_get(t, 1, NULL) == TPHT_INVALID);
    assert(tpht_remove(NULL, 1) == TPHT_INVALID);
    assert(tpht_update(NULL, 1, 1) == TPHT_INVALID);

    assert(tpht_update(t, 42u, 100u) == TPHT_NOT_FOUND);
    assert(tpht_remove(t, 42u) == TPHT_NOT_FOUND);
    assert(tpht_insert(t, 42u, 100u) == TPHT_OK);
    assert(tpht_insert(t, 42u, 100u) == TPHT_EXISTS);
    assert(tpht_update(t, 42u, 200u) == TPHT_OK);
    tpht_test_assert_get(t, tc, 42u, 200u);
    assert(tpht_remove(t, 42u) == TPHT_OK);
    tpht_test_assert_missing(t, 42u);

    if (tc->resize_mode == TPHT_FIXED) {
        for (i = 0; i < (uint64_t)fixed_capacity; ++i) {
            assert(tpht_insert(t, i + 1000u, i + 9000u) == TPHT_OK);
        }
        assert(tpht_insert(t, 999999u, 1u) == TPHT_FULL);
        assert(tpht_put(t, 1000u, 77u) == TPHT_OK);
        tpht_test_assert_get(t, tc, 1000u, 77u);
    }

    tpht_destroy(t);
}

void tpht_test_run_api_edges_module(void) {
    tpht_test_for_each_case(run_api_edges_case);
}
