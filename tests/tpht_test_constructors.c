#include "tpht_test_common.h"

#include <assert.h>

void tpht_test_run_constructor_module(void) {
    tpht_table_t *tables[8];
    size_t i;

    tables[0] = chained_tpht_fixed_create(32, 8, 8);
    tables[1] = chained_tpht_resizable_create(8, 8, 8);
    tables[2] = chained_tpht_concurrent_fixed_create(32, 8, 8);
    tables[3] = chained_tpht_concurrent_resizable_create(8, 8, 8);
    tables[4] = flatten_tpht_fixed_create(32, 8, 8);
    tables[5] = flatten_tpht_resizable_create(8, 8, 8);
    tables[6] = flatten_tpht_concurrent_fixed_create(32, 8, 8);
    tables[7] = flatten_tpht_concurrent_resizable_create(8, 8, 8);

    for (i = 0; i < 8u; ++i) {
        assert(tables[i] != NULL);
        assert(tpht_put_u64(tables[i], 7, 11) == TPHT_OK);
        tpht_test_assert_get(tables[i], 8, 8, 7, 11);
        tpht_destroy(tables[i]);
    }
}
