#include "tpht_test_common.h"

#include <assert.h>

static void run_deterministic_case(const tpht_test_case_t *tc) {
    const size_t initial_capacity = tc->resize_mode == TPHT_RESIZABLE ? 8u : 96u;
    const uint64_t n = tc->resize_mode == TPHT_RESIZABLE ? 260u : 72u;
    tpht_table_t *t = tpht_test_make_table(tc->variant, tc->threading,
                                           tc->resize_mode, tc->key_size,
                                           tc->value_size, initial_capacity);
    uint64_t i, out;
    size_t start_capacity;

    assert(t != NULL);
    assert(tpht_get_variant(t) == tc->variant);
    assert(tpht_get_threading(t) == tc->threading);
    assert(tpht_get_resize_mode(t) == tc->resize_mode);
    start_capacity = tpht_capacity(t);

    for (i = 0; i < n; ++i) {
        assert(tpht_put_u64(t, i, i * 17u + 3u) == TPHT_OK);
    }
    assert(tpht_size(t) == (size_t)n);
    if (tc->resize_mode == TPHT_RESIZABLE) assert(tpht_capacity(t) > start_capacity);

    for (i = 0; i < n; ++i) {
        assert(tpht_get_u64(t, i, &out) == TPHT_OK);
        assert(out == tpht_test_trunc_to(i * 17u + 3u, tc->value_size));
    }

    for (i = 0; i < n; i += 5u) {
        uint8_t k[8], v[8];
        tpht_test_put_bytes(k, tc->key_size, i);
        tpht_test_put_bytes(v, tc->value_size, i * 19u + 5u);
        assert(tpht_insert(t, k, v) == TPHT_EXISTS);
        assert(tpht_update(t, k, v) == TPHT_OK);
    }
    for (i = 0; i < n; i += 5u) {
        tpht_test_assert_get(t, tc->key_size, tc->value_size, i, i * 19u + 5u);
    }

    for (i = 0; i < n; i += 2u) assert(tpht_remove_u64(t, i) == TPHT_OK);
    for (i = 0; i < n; ++i) {
        tpht_status_t st = tpht_get_u64(t, i, &out);
        assert(((i & 1u) == 0u && st == TPHT_NOT_FOUND) ||
               ((i & 1u) != 0u && st == TPHT_OK));
    }

    tpht_destroy(t);
}

void tpht_test_run_deterministic_module(void) {
    tpht_test_for_each_case(run_deterministic_case);
}
