#include "tpht_test_common.h"

#include <assert.h>

/* Every named constructor produces a working table of its own type. */
void tpht_test_run_constructor_module(void) {
    uint8_t vs;
    uint64_t v;

    for (vs = 1u; vs <= 8u; ++vs) {
        uint64_t expect = tpht_test_trunc_to(11u, vs);

        flatten_tpht32_t *f32 = flatten_tpht32_fixed_create(32, vs);
        flatten_tpht32_t *f32r = flatten_tpht32_resizable_create(8, vs);
        flatten_tpht64_t *f64 = flatten_tpht64_fixed_create(32, vs);
        flatten_tpht64_t *f64r = flatten_tpht64_resizable_create(8, vs);
        chained_tpht32_t *c32 = chained_tpht32_fixed_create(32, vs);
        chained_tpht32_t *c32r = chained_tpht32_resizable_create(8, vs);
        chained_tpht32_t *c32c = chained_tpht32_concurrent_fixed_create(32, vs);
        chained_tpht32_t *c32cr = chained_tpht32_concurrent_resizable_create(8, vs);
        chained_tpht64_t *c64 = chained_tpht64_fixed_create(32, vs);
        chained_tpht64_t *c64r = chained_tpht64_resizable_create(8, vs);
        chained_tpht64_t *c64c = chained_tpht64_concurrent_fixed_create(32, vs);
        chained_tpht64_t *c64cr = chained_tpht64_concurrent_resizable_create(8, vs);

        assert(f32 && f32r && f64 && f64r);
        assert(c32 && c32r && c32c && c32cr && c64 && c64r && c64c && c64cr);

        assert(flatten_tpht32_put(f32, 7, 11) == TPHT_OK);
        assert(flatten_tpht32_get(f32, 7, &v) == TPHT_OK && v == expect);
        assert(flatten_tpht64_put(f64, 7, 11) == TPHT_OK);
        assert(flatten_tpht64_get(f64, 7, &v) == TPHT_OK && v == expect);
        assert(chained_tpht32_put(c32, 7, 11) == TPHT_OK);
        assert(chained_tpht32_get(c32, 7, &v) == TPHT_OK && v == expect);
        assert(chained_tpht64_put(c64c, 7, 11) == TPHT_OK);
        assert(chained_tpht64_get(c64c, 7, &v) == TPHT_OK && v == expect);

        flatten_tpht32_destroy(f32); flatten_tpht32_destroy(f32r);
        flatten_tpht64_destroy(f64); flatten_tpht64_destroy(f64r);
        chained_tpht32_destroy(c32); chained_tpht32_destroy(c32r);
        chained_tpht32_destroy(c32c); chained_tpht32_destroy(c32cr);
        chained_tpht64_destroy(c64); chained_tpht64_destroy(c64r);
        chained_tpht64_destroy(c64c); chained_tpht64_destroy(c64cr);
    }

    /* Rejected value sizes. */
    assert(flatten_tpht32_fixed_create(32, 9) == NULL);
    assert(flatten_tpht64_fixed_create(32, 9) == NULL);
    assert(chained_tpht32_fixed_create(32, 9) == NULL);
    assert(chained_tpht64_fixed_create(32, 9) == NULL);

    /* A NULL table is rejected, not dereferenced. */
    assert(flatten_tpht32_put(NULL, 1, 1) == TPHT_INVALID);
    assert(flatten_tpht64_get(NULL, 1, &v) == TPHT_INVALID);
    assert(chained_tpht32_remove(NULL, 1) == TPHT_INVALID);
    assert(chained_tpht64_update(NULL, 1, 1) == TPHT_INVALID);
    flatten_tpht32_destroy(NULL);
    chained_tpht64_destroy(NULL);
}
