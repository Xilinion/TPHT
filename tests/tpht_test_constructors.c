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
        /* The typed concurrent constructors. */
        flatten_conc_tpht32_t *fc32 = flatten_conc_tpht32_fixed_create(32, vs);
        flatten_conc_tpht32_t *fc32r = flatten_conc_tpht32_resizable_create(8, vs);
        flatten_conc_tpht64_t *fc64 = flatten_conc_tpht64_fixed_create(32, vs);
        flatten_conc_tpht64_t *fc64r = flatten_conc_tpht64_resizable_create(8, vs);
        chained_conc_tpht32_t *cc32 = chained_conc_tpht32_fixed_create(32, vs);
        chained_conc_tpht32_t *cc32r = chained_conc_tpht32_resizable_create(8, vs);
        chained_conc_tpht64_t *cc64 = chained_conc_tpht64_fixed_create(32, vs);
        chained_conc_tpht64_t *cc64r = chained_conc_tpht64_resizable_create(8, vs);

        assert(f32 && f32r && f64 && f64r);
        assert(c32 && c32r && c32c && c32cr && c64 && c64r && c64c && c64cr);
        assert(fc32 && fc32r && fc64 && fc64r && cc32 && cc32r && cc64 && cc64r);

        assert(flatten_tpht32_put(f32, 7, 11) == TPHT_OK);
        assert(flatten_tpht32_get(f32, 7, &v) == TPHT_OK && v == expect);
        assert(flatten_tpht64_put(f64, 7, 11) == TPHT_OK);
        assert(flatten_tpht64_get(f64, 7, &v) == TPHT_OK && v == expect);
        assert(chained_tpht32_put(c32, 7, 11) == TPHT_OK);
        assert(chained_tpht32_get(c32, 7, &v) == TPHT_OK && v == expect);
        assert(chained_tpht64_put(c64c, 7, 11) == TPHT_OK);
        assert(chained_tpht64_get(c64c, 7, &v) == TPHT_OK && v == expect);
        assert(flatten_conc_tpht64_put(fc64r, 7, 11) == TPHT_OK);
        assert(flatten_conc_tpht64_get(fc64r, 7, &v) == TPHT_OK && v == expect);
        assert(flatten_conc_tpht32_put(fc32, 7, 11) == TPHT_OK);
        assert(flatten_conc_tpht32_get(fc32, 7, &v) == TPHT_OK && v == expect);
        assert(chained_conc_tpht64_put(cc64r, 7, 11) == TPHT_OK);
        assert(chained_conc_tpht64_get(cc64r, 7, &v) == TPHT_OK && v == expect);
        assert(chained_conc_tpht32_put(cc32, 7, 11) == TPHT_OK);
        assert(chained_conc_tpht32_get(cc32, 7, &v) == TPHT_OK && v == expect);

        flatten_tpht32_destroy(f32); flatten_tpht32_destroy(f32r);
        flatten_tpht64_destroy(f64); flatten_tpht64_destroy(f64r);
        chained_tpht32_destroy(c32); chained_tpht32_destroy(c32r);
        chained_tpht32_destroy(c32c); chained_tpht32_destroy(c32cr);
        chained_tpht64_destroy(c64); chained_tpht64_destroy(c64r);
        chained_tpht64_destroy(c64c); chained_tpht64_destroy(c64cr);
        flatten_conc_tpht32_destroy(fc32); flatten_conc_tpht32_destroy(fc32r);
        flatten_conc_tpht64_destroy(fc64); flatten_conc_tpht64_destroy(fc64r);
        chained_conc_tpht32_destroy(cc32); chained_conc_tpht32_destroy(cc32r);
        chained_conc_tpht64_destroy(cc64); chained_conc_tpht64_destroy(cc64r);
    }

    /* Rejected value sizes. */
    assert(flatten_tpht32_fixed_create(32, 9) == NULL);
    assert(flatten_tpht64_fixed_create(32, 9) == NULL);
    assert(chained_tpht32_fixed_create(32, 9) == NULL);
    assert(chained_tpht64_fixed_create(32, 9) == NULL);

    /*
     * A null handle is a caller bug, not a supported input: the read and write
     * operations do not test for one.  Destroy is the exception and accepts
     * null, as free does.
     */
    flatten_tpht32_destroy(NULL);
    flatten_tpht64_destroy(NULL);
    chained_tpht32_destroy(NULL);
    chained_tpht64_destroy(NULL);
    (void)v;
}
