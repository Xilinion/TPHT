#include "tpht_test_common.h"

#include <assert.h>

static size_t test_ceil_mul_div(size_t x, size_t mul, size_t div) {
    return (x * mul + div - 1u) / div;
}

static size_t test_chained_actual_slots(size_t capacity, size_t bin_size) {
    size_t requested = test_ceil_mul_div(capacity, 100u, 95u);
    return test_ceil_mul_div(requested, 1u, bin_size) * bin_size;
}

static size_t test_memory_for_chained(size_t capacity, uint8_t key_size,
                                      uint8_t value_size) {
    tpht_table_t *t = chained_tpht_fixed_create(capacity, key_size, value_size);
    size_t bytes;
    assert(t != NULL);
    bytes = tpht_memory_bytes(t);
    tpht_destroy(t);
    return bytes;
}

static void test_chained_compact_layout(void) {
    const size_t capacity = 1024u;
    size_t slots = test_chained_actual_slots(capacity, 127u);
    size_t k16v2 = test_memory_for_chained(capacity, 2, 2);
    size_t k16v4 = test_memory_for_chained(capacity, 2, 4);
    size_t k16v8 = test_memory_for_chained(capacity, 2, 8);
    size_t k32v2 = test_memory_for_chained(capacity, 4, 2);
    size_t k64v2 = test_memory_for_chained(capacity, 8, 2);
    size_t high_base_k16v2 = test_memory_for_chained(65536u, 2, 2);
    size_t high_base_k32v2 = test_memory_for_chained(65536u, 4, 2);

    assert(k16v4 - k16v2 == slots * 2u);
    assert(k16v8 - k16v4 == slots * 4u);
    assert(k32v2 - k16v2 == slots * 2u);
    assert(k64v2 - k32v2 == slots * 4u);

    slots = test_chained_actual_slots(65536u, 127u);
    assert(high_base_k32v2 - high_base_k16v2 == slots * 2u);
}

static void test_chained_zero_quotient_keys(void) {
    tpht_table_t *t = chained_tpht_fixed_create(65536u, 2, 2);
    uint8_t k1[2], k2[2], v1[2], v2[2], out[2];
    assert(t != NULL);

    tpht_test_put_bytes(k1, 2, 0x1234u);
    tpht_test_put_bytes(k2, 2, 0x5678u);
    tpht_test_put_bytes(v1, 2, 0xaaaau);
    tpht_test_put_bytes(v2, 2, 0xbbbbu);

    assert(tpht_put(t, k1, v1) == TPHT_OK);
    assert(tpht_put(t, k2, v2) == TPHT_OK);
    assert(tpht_get(t, k1, out) == TPHT_OK);
    assert(tpht_test_get_bytes(out, 2) == 0xaaaau);
    assert(tpht_get(t, k2, out) == TPHT_OK);
    assert(tpht_test_get_bytes(out, 2) == 0xbbbbu);

    tpht_destroy(t);
}

void tpht_test_run_config_module(void) {
    tpht_config_t c = tpht_default_config();
    tpht_table_t *t;

    c.key_size = 3;
    assert(tpht_create(&c) == NULL);

    c = tpht_default_config();
    c.value_size = 3;
    assert(tpht_create(&c) == NULL);
    c = tpht_default_config();
    c.value_size = 0;
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

    test_chained_compact_layout();
    test_chained_zero_quotient_keys();
}
