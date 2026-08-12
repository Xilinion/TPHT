#include "tpht_test_common.h"

#include <assert.h>

static void run_deterministic_case(const tpht_test_case_t *tc) {
    const size_t initial_capacity = tc->resize_mode == TPHT_RESIZABLE ? 8u : 96u;
    const uint64_t n = tc->resize_mode == TPHT_RESIZABLE ? 260u : 72u;
    tpht_test_table_t t = tpht_test_make_table(tc, initial_capacity);
    uint64_t i, out;
    size_t start_capacity;

    assert(t.handle != NULL);
    start_capacity = t.capacity(t.handle);

    for (i = 0; i < n; ++i) assert(t.put(t.handle, i, i * 17u + 3u) == TPHT_OK);
    assert(t.size(t.handle) == (size_t)n);
    if (tc->resize_mode == TPHT_RESIZABLE) assert(t.capacity(t.handle) > start_capacity);

    for (i = 0; i < n; ++i) {
        assert(t.get(t.handle, i, &out) == TPHT_OK);
        assert(out == tpht_test_trunc_to(i * 17u + 3u, tc->value_size));
    }

    for (i = 0; i < n; i += 5u) {
        assert(t.insert(t.handle, i, i * 19u + 5u) == TPHT_EXISTS);
        assert(t.update(t.handle, i, i * 19u + 5u) == TPHT_OK);
    }
    for (i = 0; i < n; i += 5u) tpht_test_assert_get(&t, tc, i, i * 19u + 5u);

    for (i = 0; i < n; i += 2u) assert(t.remove(t.handle, i) == TPHT_OK);
    for (i = 0; i < n; ++i) {
        tpht_status_t st = t.get(t.handle, i, &out);
        assert(((i & 1u) == 0u && st == TPHT_NOT_FOUND) ||
               ((i & 1u) != 0u && st == TPHT_OK));
    }

    t.destroy(t.handle);
}

void tpht_test_run_deterministic_module(void) {
    tpht_test_for_each_case(run_deterministic_case);
}
