#ifndef TPHT_H
#define TPHT_H

/*
 * TPHT - Tiny Pointer Hash Tables, industrial C edition.
 *
 * This public API intentionally keeps the library copy-pasteable: include
 * tpht.h and compile tpht.c as C11.  No SIMD, C++, pthreads, or platform
 * specific code is required.  Concurrent tables use C11 atomics.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum tpht_variant {
    TPHT_CHAINED = 1, /* TinyPtr byte_array_chained_ht family. */
    TPHT_FLATTEN = 2  /* TinyPtr blast_ht family: inline cloud + overflow. */
} tpht_variant_t;

typedef enum tpht_threading {
    TPHT_SEQUENTIAL = 0,
    TPHT_CONCURRENT = 1
} tpht_threading_t;

typedef enum tpht_resize_mode {
    TPHT_FIXED = 0,
    TPHT_RESIZABLE = 1
} tpht_resize_mode_t;

typedef enum tpht_status {
    TPHT_OK = 0,
    TPHT_NOT_FOUND = 1,
    TPHT_EXISTS = 2,
    TPHT_FULL = 3,
    TPHT_NO_MEMORY = 4,
    TPHT_INVALID = 5
} tpht_status_t;

typedef struct tpht_table tpht_table_t;

typedef struct tpht_config {
    tpht_variant_t variant;
    tpht_threading_t threading;
    tpht_resize_mode_t resize_mode;

    /* Number of key/value pairs expected before resizing or fixed-table full. */
    size_t initial_capacity;

    /* Supported values are 2, 4, and 8 bytes (int16/int32/int64 families). */
    uint8_t key_size;
    uint8_t value_size;

    /* Tiny pointers reserve one bit for two-choice bin identity, so <= 127. */
    uint8_t bin_size;

    /* Used by resizable tables.  0 means the default, 0.85. */
    double max_load_factor;

    /* 0 selects a stable built-in seed. */
    uint64_t hash_seed;

    /*
     * Cooperative resize work units for concurrent chained resizable tables.
     * 0 selects an automatic value. Larger values expose more resize work for
     * helper threads, but add slightly more scheduling overhead.
     */
    size_t resize_strides;
} tpht_config_t;

tpht_config_t tpht_default_config(void);

tpht_table_t *tpht_create(const tpht_config_t *config);
void tpht_destroy(tpht_table_t *table);

tpht_status_t tpht_put(tpht_table_t *table, const void *key, const void *value);
tpht_status_t tpht_insert(tpht_table_t *table, const void *key, const void *value);
tpht_status_t tpht_update(tpht_table_t *table, const void *key, const void *value);
tpht_status_t tpht_get(tpht_table_t *table, const void *key, void *value_out);
tpht_status_t tpht_remove(tpht_table_t *table, const void *key);

size_t tpht_size(const tpht_table_t *table);
size_t tpht_capacity(const tpht_table_t *table);
size_t tpht_memory_bytes(const tpht_table_t *table);
tpht_variant_t tpht_get_variant(const tpht_table_t *table);
tpht_threading_t tpht_get_threading(const tpht_table_t *table);
tpht_resize_mode_t tpht_get_resize_mode(const tpht_table_t *table);

/* Explicit constructors requested for the industrial variants. */
tpht_table_t *chained_tpht_fixed_create(size_t capacity, uint8_t key_size,
                                        uint8_t value_size);
tpht_table_t *chained_tpht_resizable_create(size_t capacity, uint8_t key_size,
                                            uint8_t value_size);
tpht_table_t *chained_tpht_concurrent_fixed_create(size_t capacity,
                                                   uint8_t key_size,
                                                   uint8_t value_size);
tpht_table_t *chained_tpht_concurrent_resizable_create(size_t capacity,
                                                       uint8_t key_size,
                                                       uint8_t value_size);

tpht_table_t *flatten_tpht_fixed_create(size_t capacity, uint8_t key_size,
                                        uint8_t value_size);
tpht_table_t *flatten_tpht_resizable_create(size_t capacity, uint8_t key_size,
                                            uint8_t value_size);
tpht_table_t *flatten_tpht_concurrent_fixed_create(size_t capacity,
                                                   uint8_t key_size,
                                                   uint8_t value_size);
tpht_table_t *flatten_tpht_concurrent_resizable_create(size_t capacity,
                                                       uint8_t key_size,
                                                       uint8_t value_size);

/* Convenience integer helpers.  Values are copied using the table's sizes. */
tpht_status_t tpht_put_u64(tpht_table_t *table, uint64_t key, uint64_t value);
tpht_status_t tpht_get_u64(tpht_table_t *table, uint64_t key,
                           uint64_t *value_out);
tpht_status_t tpht_remove_u64(tpht_table_t *table, uint64_t key);

#ifdef __cplusplus
}
#endif

#endif /* TPHT_H */
