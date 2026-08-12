#ifndef TPHT_TEST_COMMON_H
#define TPHT_TEST_COMMON_H

#include "tpht.h"

#include <stddef.h>
#include <stdint.h>

typedef enum tpht_test_kind {
    TPHT_TEST_FLAT32 = 0,
    TPHT_TEST_FLAT64,
    TPHT_TEST_CHAIN32,
    TPHT_TEST_CHAIN64,
    TPHT_TEST_KIND_COUNT
} tpht_test_kind_t;

typedef struct tpht_test_case {
    tpht_test_kind_t kind;
    int concurrent; /* chained only */
    tpht_resize_mode_t resize_mode;
    uint8_t value_size;
} tpht_test_case_t;

/*
 * The library exposes four concrete types with no common supertype, so the
 * tests drive them through a small vtable rather than duplicating every
 * scenario four times.
 */
typedef struct tpht_test_table {
    void *handle;
    tpht_status_t (*put)(void *, uint64_t, uint64_t);
    tpht_status_t (*insert)(void *, uint64_t, uint64_t);
    tpht_status_t (*update)(void *, uint64_t, uint64_t);
    tpht_status_t (*get)(void *, uint64_t, uint64_t *);
    tpht_status_t (*remove)(void *, uint64_t);
    size_t (*size)(const void *);
    size_t (*capacity)(const void *);
    size_t (*memory_bytes)(const void *);
    void (*destroy)(void *);
} tpht_test_table_t;

typedef struct tpht_test_model_entry {
    uint64_t key;
    uint64_t value;
    int live;
} tpht_test_model_entry_t;

uint64_t tpht_test_mask_for_size(uint8_t size);
uint64_t tpht_test_trunc_to(uint64_t x, uint8_t size);
uint64_t tpht_test_next_rand(uint64_t *state);
uint8_t tpht_test_key_size(tpht_test_kind_t kind);
const char *tpht_test_kind_name(tpht_test_kind_t kind);

size_t tpht_test_model_find(tpht_test_model_entry_t *model, size_t n, uint64_t key);

int tpht_test_case_supported(const tpht_test_case_t *tc);
/* handle is NULL when the combination is unsupported. */
tpht_test_table_t tpht_test_make_table(const tpht_test_case_t *tc, size_t capacity);
tpht_test_table_t tpht_test_make_kind(tpht_test_kind_t kind, int concurrent,
                                      tpht_resize_mode_t mode, size_t capacity,
                                      uint8_t value_size, const tpht_options_t *options);

void tpht_test_assert_get(const tpht_test_table_t *t, const tpht_test_case_t *tc, uint64_t key,
                          uint64_t expected);
void tpht_test_assert_missing(const tpht_test_table_t *t, uint64_t key);

void tpht_test_for_each_case(void (*fn)(const tpht_test_case_t *tc));

#endif /* TPHT_TEST_COMMON_H */
