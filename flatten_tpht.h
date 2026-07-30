#ifndef FLATTEN_TPHT_H
#define FLATTEN_TPHT_H

#include "tpht.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef tpht_table_t flatten_tpht_t;

static inline flatten_tpht_t *flatten_tpht_fixed_create(size_t capacity,
                                                        uint8_t key_size,
                                                        uint8_t value_size) {
    return tpht_flatten_fixed_create(capacity, key_size, value_size);
}

static inline flatten_tpht_t *flatten_tpht_resizable_create(size_t capacity,
                                                            uint8_t key_size,
                                                            uint8_t value_size) {
    return tpht_flatten_resizable_create(capacity, key_size, value_size);
}

static inline flatten_tpht_t *flatten_tpht_concurrent_fixed_create(size_t capacity,
                                                                   uint8_t key_size,
                                                                   uint8_t value_size) {
    return tpht_flatten_concurrent_fixed_create(capacity, key_size, value_size);
}

static inline flatten_tpht_t *flatten_tpht_concurrent_resizable_create(size_t capacity,
                                                                       uint8_t key_size,
                                                                       uint8_t value_size) {
    return tpht_flatten_concurrent_resizable_create(capacity, key_size, value_size);
}

static inline void flatten_tpht_destroy(flatten_tpht_t *table) { tpht_destroy(table); }
static inline tpht_status_t flatten_tpht_put(flatten_tpht_t *table, const void *key, const void *value) { return tpht_put(table, key, value); }
static inline tpht_status_t flatten_tpht_insert(flatten_tpht_t *table, const void *key, const void *value) { return tpht_insert(table, key, value); }
static inline tpht_status_t flatten_tpht_update(flatten_tpht_t *table, const void *key, const void *value) { return tpht_update(table, key, value); }
static inline tpht_status_t flatten_tpht_get(flatten_tpht_t *table, const void *key, void *value_out) { return tpht_get(table, key, value_out); }
static inline tpht_status_t flatten_tpht_remove(flatten_tpht_t *table, const void *key) { return tpht_remove(table, key); }
static inline size_t flatten_tpht_size(const flatten_tpht_t *table) { return tpht_size(table); }
static inline size_t flatten_tpht_capacity(const flatten_tpht_t *table) { return tpht_capacity(table); }

#ifdef __cplusplus
}
#endif

#endif /* FLATTEN_TPHT_H */
