#ifndef CHAINED_TPHT_H
#define CHAINED_TPHT_H

#include "tpht.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef tpht_table_t chained_tpht_t;

static inline chained_tpht_t *chained_tpht_fixed_create(size_t capacity,
                                                        uint8_t key_size,
                                                        uint8_t value_size) {
    return tpht_chained_fixed_create(capacity, key_size, value_size);
}

static inline chained_tpht_t *chained_tpht_resizable_create(size_t capacity,
                                                            uint8_t key_size,
                                                            uint8_t value_size) {
    return tpht_chained_resizable_create(capacity, key_size, value_size);
}

static inline chained_tpht_t *chained_tpht_concurrent_fixed_create(size_t capacity,
                                                                   uint8_t key_size,
                                                                   uint8_t value_size) {
    return tpht_chained_concurrent_fixed_create(capacity, key_size, value_size);
}

static inline chained_tpht_t *chained_tpht_concurrent_resizable_create(size_t capacity,
                                                                       uint8_t key_size,
                                                                       uint8_t value_size) {
    return tpht_chained_concurrent_resizable_create(capacity, key_size, value_size);
}

static inline void chained_tpht_destroy(chained_tpht_t *table) { tpht_destroy(table); }
static inline tpht_status_t chained_tpht_put(chained_tpht_t *table, const void *key, const void *value) { return tpht_put(table, key, value); }
static inline tpht_status_t chained_tpht_insert(chained_tpht_t *table, const void *key, const void *value) { return tpht_insert(table, key, value); }
static inline tpht_status_t chained_tpht_update(chained_tpht_t *table, const void *key, const void *value) { return tpht_update(table, key, value); }
static inline tpht_status_t chained_tpht_get(chained_tpht_t *table, const void *key, void *value_out) { return tpht_get(table, key, value_out); }
static inline tpht_status_t chained_tpht_remove(chained_tpht_t *table, const void *key) { return tpht_remove(table, key); }
static inline size_t chained_tpht_size(const chained_tpht_t *table) { return tpht_size(table); }
static inline size_t chained_tpht_capacity(const chained_tpht_t *table) { return tpht_capacity(table); }

#ifdef __cplusplus
}
#endif

#endif /* CHAINED_TPHT_H */
