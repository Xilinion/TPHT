#include "tpht_test_common.h"

#include <assert.h>

void tpht_test_run_config_module(void) {
    tpht_config_t c = tpht_default_config();
    tpht_table_t *t;

    c.key_size = 3;
    assert(tpht_create(&c) == NULL);

    c = tpht_default_config();
    c.value_size = 3;
    assert(tpht_create(&c) == NULL);

    c = tpht_default_config();
    c.bin_size = 128;
    assert(tpht_create(&c) == NULL);

    c = tpht_default_config();
    c.initial_capacity = 0;
    t = tpht_create(&c);
    assert(t != NULL);
    assert(tpht_capacity(t) >= 16u);
    tpht_destroy(t);
}
