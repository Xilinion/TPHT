#ifndef TPHT_TEST_COMMON_H
#define TPHT_TEST_COMMON_H

#include "tpht.h"

#include <stddef.h>
#include <stdint.h>

typedef struct tpht_test_case {
    tpht_variant_t variant;
    tpht_threading_t threading;
    tpht_resize_mode_t resize_mode;
    uint8_t key_size;   /* 4 or 8 */
    uint8_t value_size; /* 1 to 8 */
} tpht_test_case_t;

typedef struct tpht_test_model_entry {
    uint64_t key;
    uint64_t value;
    int live;
} tpht_test_model_entry_t;

uint64_t tpht_test_mask_for_size(uint8_t size);
uint64_t tpht_test_trunc_to(uint64_t x, uint8_t size);
uint64_t tpht_test_next_rand(uint64_t *state);

size_t tpht_test_model_find(tpht_test_model_entry_t *model, size_t n, uint64_t key);

/* Returns NULL for combinations the library does not support. */
tpht_table_t *tpht_test_make_table(const tpht_test_case_t *tc, size_t capacity);

int tpht_test_case_supported(const tpht_test_case_t *tc);

void tpht_test_assert_get(tpht_table_t *table, const tpht_test_case_t *tc, uint64_t key,
                          uint64_t expected);
void tpht_test_assert_missing(tpht_table_t *table, uint64_t key);

/* Calls fn once per supported variant/threading/resize/key/value combination. */
void tpht_test_for_each_case(void (*fn)(const tpht_test_case_t *tc));

#endif /* TPHT_TEST_COMMON_H */
