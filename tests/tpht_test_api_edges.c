#include "tpht_test_common.h"

#include <assert.h>

static void run_api_edges_case(const tpht_test_case_t *tc) {
    const size_t fixed_capacity = 16;
    tpht_test_table_t t = tpht_test_make_table(tc, fixed_capacity);
    uint64_t out;
    uint64_t i;

    assert(t.handle != NULL);

    assert(t.update(t.handle, 42u, 100u) == TPHT_NOT_FOUND);
    assert(t.remove(t.handle, 42u) == TPHT_NOT_FOUND);
    assert(t.insert(t.handle, 42u, 100u) == TPHT_OK);
    /* insert appends unconditionally (no existence probe), so a repeat insert
     * is an append; overwrite is put/update's job. */
    assert(t.insert(t.handle, 42u, 100u) == TPHT_OK);
    assert(t.update(t.handle, 42u, 200u) == TPHT_OK);
    tpht_test_assert_get(&t, tc, 42u, 200u);
    assert(t.remove(t.handle, 42u) == TPHT_OK);
    assert(t.remove(t.handle, 42u) == TPHT_OK); /* drop the appended duplicate */
    tpht_test_assert_missing(&t, 42u);

    if (tc->resize_mode == TPHT_FIXED) {
        for (i = 0; i < (uint64_t)fixed_capacity; ++i) {
            assert(t.insert(t.handle, i + 1000u, i + 9000u) == TPHT_OK);
        }
        /* A fixed table keeps no running size: capacity provisions its storage
         * but is not a hard cap, so an insert past it absorbs the overflow by
         * rebuilding with more blocks rather than reporting TPHT_OVERFLOW. */
        assert(t.insert(t.handle, 999999u, 1u) == TPHT_OK);
        tpht_test_assert_get(&t, tc, 999999u, 1u);
        /* An overwrite of a key already present still succeeds. */
        assert(t.put(t.handle, 1000u, 77u) == TPHT_OK);
        tpht_test_assert_get(&t, tc, 1000u, 77u);
        /* Append-only insert cannot acknowledge the existing key, so this
         * appends a duplicate rather than overwriting. */
        assert(t.insert(t.handle, 1000u, 5u) == TPHT_OK);
        /* Counted on demand for a fixed table: the loop's entries, the one that
         * forced the rebuild, and the duplicate append.  The put overwrote an
         * existing key, so it added nothing. */
        assert(t.size(t.handle) == (size_t)fixed_capacity + 2u);
    }
    (void)out;
    t.destroy(t.handle);
}

/* Tiny capacities, including 1: fill to the requested capacity, verify,
 * drain, refill.  Small tables exercise the minimum-geometry paths that big
 * ones never touch. */
static void run_tiny_capacity_case(const tpht_test_case_t *tc) {
    static const size_t caps[] = {1, 2, 3, 5, 8, 16};
    size_t ci;

    for (ci = 0; ci < sizeof(caps) / sizeof(caps[0]); ++ci) {
        tpht_test_table_t t = tpht_test_make_table(tc, caps[ci]);
        uint64_t k;

        assert(t.handle != NULL);
        for (k = 1; k <= (uint64_t)caps[ci]; ++k)
            assert(t.insert(t.handle, k, k * 3u) == TPHT_OK);
        for (k = 1; k <= (uint64_t)caps[ci]; ++k)
            tpht_test_assert_get(&t, tc, k, k * 3u);
        for (k = 1; k <= (uint64_t)caps[ci]; ++k)
            assert(t.remove(t.handle, k) == TPHT_OK);
        for (k = 1; k <= (uint64_t)caps[ci]; ++k)
            tpht_test_assert_missing(&t, k);
        for (k = 1; k <= (uint64_t)caps[ci]; ++k)
            assert(t.insert(t.handle, k, k * 3u) == TPHT_OK);
        for (k = 1; k <= (uint64_t)caps[ci]; ++k)
            tpht_test_assert_get(&t, tc, k, k * 3u);
        t.destroy(t.handle);
    }
}

void tpht_test_run_api_edges_module(void) {
    tpht_test_for_each_case(run_api_edges_case);
    tpht_test_for_each_case(run_tiny_capacity_case);
}
