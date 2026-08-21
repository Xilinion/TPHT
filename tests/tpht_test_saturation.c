#include "tpht_test_common.h"

#include <assert.h>

/*
 * Block-saturation regression tests.  A 64-byte home block can hold at most
 * TPHT_FLAT_MAX_TUPLES tuples (even all-spilled, x tuples cost x
 * fingerprint bytes plus x tiny-pointer bytes, and the count field's ceiling
 * of 31 needs 62 of the 61 usable bytes); the hard-overflow guard used to sit
 * at the ceiling itself, so the 31st tuple's layout ran one byte past the
 * line into the metadata and silently corrupted the block.  These cases force
 * every geometry into that regime - far more keys than blocks want to hold -
 * and verify each key still round-trips exactly.
 */
static void saturate_case(const tpht_test_case_t *tc) {
    /* Tiny capacity, many keys: overflow absorption must keep every key.
     * Sequential fixed tables absorb by rebuilding with more blocks; the
     * concurrent ones do the same through the shadow-migration machinery;
     * resizable ones double instead.  All must stay exact. */
    const size_t capacity = 16;
    const uint64_t n = 200;
    tpht_test_table_t t = tpht_test_make_table(tc, capacity);
    uint64_t i, out;

    assert(t.handle != NULL);
    for (i = 0; i < n; ++i) {
        assert(t.put(t.handle, i, i * 31u + 7u) == TPHT_OK);
        /* Verify the full prefix after every insert: the historical corruption
         * clobbered a *previous* tuple, invisible until the next lookup. */
        if ((i & 15u) == 15u || i + 1u == n) {
            uint64_t j;
            for (j = 0; j <= i; ++j) {
                assert(t.get(t.handle, j, &out) == TPHT_OK);
                assert(out == tpht_test_trunc_to(j * 31u + 7u, tc->value_size));
            }
        }
    }
    assert(t.size(t.handle) == (size_t)n);

    /* Drain half and refill with new values: saturation followed by churn. */
    for (i = 0; i < n; i += 2u) assert(t.remove(t.handle, i) == TPHT_OK);
    for (i = 0; i < n; i += 2u) assert(t.put(t.handle, i, i * 5u + 1u) == TPHT_OK);
    for (i = 0; i < n; ++i) {
        assert(t.get(t.handle, i, &out) == TPHT_OK);
        assert(out == tpht_test_trunc_to((i & 1u) ? i * 31u + 7u : i * 5u + 1u,
                                         tc->value_size));
    }
    t.destroy(t.handle);
}

/* Degenerate capacities must behave, not just not crash. */
static void degenerate_case(const tpht_test_case_t *tc) {
    static const size_t caps[] = {0u, 1u, 2u, 3u};
    size_t ci;
    for (ci = 0; ci < sizeof(caps) / sizeof(caps[0]); ++ci) {
        tpht_test_table_t t = tpht_test_make_table(tc, caps[ci]);
        uint64_t i, out;
        assert(t.handle != NULL);
        for (i = 0; i < 40u; ++i) assert(t.put(t.handle, i + 3u, i + 100u) == TPHT_OK);
        for (i = 0; i < 40u; ++i) {
            assert(t.get(t.handle, i + 3u, &out) == TPHT_OK);
            assert(out == tpht_test_trunc_to(i + 100u, tc->value_size));
        }
        for (i = 0; i < 40u; ++i) assert(t.remove(t.handle, i + 3u) == TPHT_OK);
        assert(t.size(t.handle) == 0u);
        tpht_test_assert_missing(&t, 3u);
        /* Key zero is a legal key like any other. */
        assert(t.put(t.handle, 0u, 12345u) == TPHT_OK);
        assert(t.get(t.handle, 0u, &out) == TPHT_OK);
        assert(out == tpht_test_trunc_to(12345u, tc->value_size));
        assert(t.update(t.handle, 0u, 54321u) == TPHT_OK);
        assert(t.get(t.handle, 0u, &out) == TPHT_OK);
        assert(out == tpht_test_trunc_to(54321u, tc->value_size));
        assert(t.remove(t.handle, 0u) == TPHT_OK);
        tpht_test_assert_missing(&t, 0u);
        /* So is the all-ones key: the top of the quotient range. */
        {
            uint64_t kmax = tpht_test_mask_for_size(tpht_test_key_size(tc->kind));
            assert(t.put(t.handle, kmax, 7u) == TPHT_OK);
            assert(t.get(t.handle, kmax, &out) == TPHT_OK);
            assert(out == tpht_test_trunc_to(7u, tc->value_size));
            assert(t.remove(t.handle, kmax) == TPHT_OK);
            tpht_test_assert_missing(&t, kmax);
        }
        t.destroy(t.handle);
    }
}

/*
 * insert() is append-only by contract: repeating a key stacks duplicates, the
 * size counts every copy, and each remove peels exactly one.  Stacking many
 * copies of one key concentrates them in a single home block, which is
 * another road into the saturation machinery.
 */
static void duplicate_case(const tpht_test_case_t *tc) {
    enum { DUPS = 120 };
    tpht_test_table_t t = tpht_test_make_table(tc, 16);
    uint64_t i, out, accepted = 0;
    int is_flat = tc->kind == TPHT_TEST_FLAT32 || tc->kind == TPHT_TEST_FLAT64;

    assert(t.handle != NULL);
    /*
     * Flattened tables bound duplicates of one key: every copy lives in the
     * same home block, and a block holds at most ~30 tuples - no amount of
     * growth splits them, so past the bound insert reports TPHT_OVERFLOW (see
     * tpht.h).  Chained tables stack all of them.
     */
    for (i = 0; i < DUPS; ++i) {
        tpht_status_t st = t.insert(t.handle, 77u, i);
        if (st != TPHT_OK) {
            assert(is_flat);
            assert(st == TPHT_OVERFLOW);
            break;
        }
        ++accepted;
    }
    if (is_flat) {
        assert(accepted >= 25u); /* near the block bound, never far below */
    } else {
        assert(accepted == (uint64_t)DUPS);
    }
    assert(t.size(t.handle) == (size_t)accepted);
    /* A lookup returns one of the stacked values; put overwrites one copy. */
    assert(t.get(t.handle, 77u, &out) == TPHT_OK);
    for (i = 0; i < accepted; ++i) assert(t.remove(t.handle, 77u) == TPHT_OK);
    assert(t.remove(t.handle, 77u) == TPHT_NOT_FOUND);
    assert(t.size(t.handle) == 0u);
    tpht_test_assert_missing(&t, 77u);
    /* The table remains fully usable afterwards. */
    for (i = 0; i < 50u; ++i) assert(t.put(t.handle, i + 1u, i) == TPHT_OK);
    for (i = 0; i < 50u; ++i) {
        assert(t.get(t.handle, i + 1u, &out) == TPHT_OK);
        assert(out == tpht_test_trunc_to(i, tc->value_size));
    }
    t.destroy(t.handle);
}

void tpht_test_run_saturation_module(void) {
    tpht_test_for_each_case(saturate_case);
    tpht_test_for_each_case(degenerate_case);
    tpht_test_for_each_case(duplicate_case);
}
