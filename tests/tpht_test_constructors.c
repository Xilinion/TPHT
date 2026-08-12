#include "tpht_test_common.h"

#include <assert.h>

static void check_table(tpht_table_t *t, uint8_t key_size, uint8_t value_size) {
    tpht_test_case_t tc;
    uint64_t out = 0;
    tc.variant = TPHT_CHAINED;
    tc.threading = TPHT_SEQUENTIAL;
    tc.resize_mode = TPHT_FIXED;
    tc.key_size = key_size;
    tc.value_size = value_size;
    assert(t != NULL);

    /* The width-specific operations only accept a table of that width. */
    if (key_size == 4u) {
        assert(tpht32_put(t, 7, 11) == TPHT_OK);
        assert(tpht32_get(t, 7, &out) == TPHT_OK);
        assert(tpht64_get(t, 7, &out) == TPHT_INVALID);
        assert(tpht64_put(t, 7, 11) == TPHT_INVALID);
        assert(tpht64_insert(t, 7, 11) == TPHT_INVALID);
        assert(tpht64_update(t, 7, 11) == TPHT_INVALID);
        assert(tpht64_remove(t, 7) == TPHT_INVALID);
    } else {
        assert(tpht64_put(t, 7, 11) == TPHT_OK);
        assert(tpht64_get(t, 7, &out) == TPHT_OK);
        assert(tpht32_get(t, 7, &out) == TPHT_INVALID);
        assert(tpht32_put(t, 7, 11) == TPHT_INVALID);
        assert(tpht32_insert(t, 7, 11) == TPHT_INVALID);
        assert(tpht32_update(t, 7, 11) == TPHT_INVALID);
        assert(tpht32_remove(t, 7) == TPHT_INVALID);
    }
    tpht_test_assert_get(t, &tc, 7, 11);
    tpht_destroy(t);
}

void tpht_test_run_constructor_module(void) {
    uint8_t vs;

    for (vs = 1u; vs <= 8u; ++vs) {
        check_table(chained_tpht32_fixed_create(32, vs), 4, vs);
        check_table(chained_tpht32_resizable_create(8, vs), 4, vs);
        check_table(chained_tpht32_concurrent_fixed_create(32, vs), 4, vs);
        check_table(chained_tpht32_concurrent_resizable_create(8, vs), 4, vs);
        check_table(flatten_tpht32_fixed_create(32, vs), 4, vs);
        check_table(flatten_tpht32_resizable_create(8, vs), 4, vs);

        check_table(chained_tpht64_fixed_create(32, vs), 8, vs);
        check_table(chained_tpht64_resizable_create(8, vs), 8, vs);
        check_table(chained_tpht64_concurrent_fixed_create(32, vs), 8, vs);
        check_table(chained_tpht64_concurrent_resizable_create(8, vs), 8, vs);
        check_table(flatten_tpht64_fixed_create(32, vs), 8, vs);
        check_table(flatten_tpht64_resizable_create(8, vs), 8, vs);

        /* Concurrency is not implemented for the flattened variant. */
        assert(flatten_tpht32_concurrent_fixed_create(32, vs) == NULL);
        assert(flatten_tpht32_concurrent_resizable_create(8, vs) == NULL);
        assert(flatten_tpht64_concurrent_fixed_create(32, vs) == NULL);
        assert(flatten_tpht64_concurrent_resizable_create(8, vs) == NULL);
    }

    /* Rejected value sizes. */
    assert(chained_tpht32_fixed_create(32, 0) == NULL);
    assert(chained_tpht64_fixed_create(32, 9) == NULL);
    assert(flatten_tpht32_fixed_create(32, 0) == NULL);
    assert(flatten_tpht64_fixed_create(32, 9) == NULL);

    /* Width-specific operations reject a NULL table. */
    assert(tpht32_put(NULL, 1, 1) == TPHT_INVALID);
    assert(tpht64_put(NULL, 1, 1) == TPHT_INVALID);
}
