#include "tpht_test_common.h"

#include <assert.h>

static void run_api_edges_case(const tpht_test_case_t *tc) {
    const size_t fixed_capacity = 16;
    tpht_test_table_t t = tpht_test_make_table(tc, fixed_capacity);
    uint64_t out;
    uint64_t i;

    assert(t.handle != NULL);
    assert(t.get(t.handle, 1, NULL) == TPHT_INVALID);

    assert(t.update(t.handle, 42u, 100u) == TPHT_NOT_FOUND);
    assert(t.remove(t.handle, 42u) == TPHT_NOT_FOUND);
    assert(t.insert(t.handle, 42u, 100u) == TPHT_OK);
    assert(t.insert(t.handle, 42u, 100u) == TPHT_EXISTS);
    assert(t.update(t.handle, 42u, 200u) == TPHT_OK);
    tpht_test_assert_get(&t, tc, 42u, 200u);
    assert(t.remove(t.handle, 42u) == TPHT_OK);
    tpht_test_assert_missing(&t, 42u);

    if (tc->resize_mode == TPHT_FIXED) {
        for (i = 0; i < (uint64_t)fixed_capacity; ++i) {
            assert(t.insert(t.handle, i + 1000u, i + 9000u) == TPHT_OK);
        }
        assert(t.insert(t.handle, 999999u, 1u) == TPHT_FULL);
        /* An overwrite of a key already present still succeeds at capacity. */
        assert(t.put(t.handle, 1000u, 77u) == TPHT_OK);
        tpht_test_assert_get(&t, tc, 1000u, 77u);
        assert(t.insert(t.handle, 1000u, 5u) == TPHT_EXISTS);
    }
    (void)out;
    t.destroy(t.handle);
}

void tpht_test_run_api_edges_module(void) {
    tpht_test_for_each_case(run_api_edges_case);
}
