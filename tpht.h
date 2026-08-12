/*
 * SPDX-FileCopyrightText: 2026 Xilin Tang and TPHT contributors
 *
 * TPHT is an independent industrial C implementation inspired by the TinyPtr
 * hash-table designs. See README.md for third-party acknowledgements.
 */

#ifndef TPHT_H
#define TPHT_H

/*
 * TPHT - Tiny Pointer Hash Tables, industrial C edition.
 *
 * This public API intentionally keeps the library copy-pasteable: include
 * tpht.h and compile tpht.c as C11.  No SIMD, C++, pthreads, or platform
 * specific code is required.  Concurrent tables use C11 atomics.
 *
 * Keys are 4 or 8 bytes and values are any size from 1 to 8 bytes.  Each key
 * width has its own constructors and operations - tpht32_* and tpht64_* - so a
 * table's key type is visible in the calls that use it.  The width-agnostic
 * tpht_* operations take a uint64_t key and work on either.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum tpht_variant {
    TPHT_CHAINED = 1, /* TinyPtr byte_array_chained_ht family. */
    TPHT_FLATTEN = 2  /* TinyPtr blast_ht family: 64B home blocks. */
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

    /*
     * Number of key/value pairs expected before resizing or fixed-table full.
     *
     * For TPHT_FLATTEN the home array holds one 64-byte block per few expected
     * keys - as many as one block can store inline, see README - rounded to the
     * nearest power of two.  Capacities near that multiple of a power of two
     * give the tightest layout.
     */
    size_t initial_capacity;

    /* Key size in bytes.  Supported values are 4 and 8. */
    uint8_t key_size;

    /* Value size in bytes.  Any size from 1 to 8 is supported. */
    uint8_t value_size;

    /* Tiny pointers reserve one bit for two-choice bin identity, so <= 127. */
    uint8_t bin_size;

    /* Used by resizable tables.  0 means the default, 0.85. */
    double max_load_factor;

    /* 0 selects a stable built-in seed. */
    uint64_t hash_seed;

    /*
     * Cooperative resize work units for concurrent chained resizable tables.
     * This is a target number of migration chunks, not a chunk size. 0 selects
     * an automatic value that targets about 64 old buckets per chunk. Larger
     * values expose more work units for helper threads, but add slightly more
     * scheduling overhead.
     */
    size_t resize_strides;
} tpht_config_t;

tpht_config_t tpht_default_config(void);

/*
 * TPHT_FLATTEN does not support TPHT_CONCURRENT yet; that combination returns
 * NULL.
 */
tpht_table_t *tpht_create(const tpht_config_t *config);
void tpht_destroy(tpht_table_t *table);

/*
 * Width-agnostic operations.  Keys are truncated to the table's key size and
 * values to its value size, so a table with a 3-byte value stores and returns
 * values modulo 2^24.
 */
tpht_status_t tpht_put(tpht_table_t *table, uint64_t key, uint64_t value);
tpht_status_t tpht_insert(tpht_table_t *table, uint64_t key, uint64_t value);
tpht_status_t tpht_update(tpht_table_t *table, uint64_t key, uint64_t value);
tpht_status_t tpht_get(tpht_table_t *table, uint64_t key, uint64_t *value_out);
tpht_status_t tpht_remove(tpht_table_t *table, uint64_t key);

size_t tpht_size(const tpht_table_t *table);
size_t tpht_capacity(const tpht_table_t *table);
size_t tpht_memory_bytes(const tpht_table_t *table);
tpht_variant_t tpht_get_variant(const tpht_table_t *table);
tpht_threading_t tpht_get_threading(const tpht_table_t *table);
tpht_resize_mode_t tpht_get_resize_mode(const tpht_table_t *table);

/*
 * 32-bit key tables.
 *
 * value_size is in bytes and may be anything from 1 to 8.  The tpht32_*
 * operations return TPHT_INVALID if handed a table whose keys are not 32-bit.
 */
tpht_table_t *chained_tpht32_fixed_create(size_t capacity, uint8_t value_size);
tpht_table_t *chained_tpht32_resizable_create(size_t capacity, uint8_t value_size);
tpht_table_t *chained_tpht32_concurrent_fixed_create(size_t capacity, uint8_t value_size);
tpht_table_t *chained_tpht32_concurrent_resizable_create(size_t capacity, uint8_t value_size);
tpht_table_t *flatten_tpht32_fixed_create(size_t capacity, uint8_t value_size);
tpht_table_t *flatten_tpht32_resizable_create(size_t capacity, uint8_t value_size);

tpht_status_t tpht32_put(tpht_table_t *table, uint32_t key, uint64_t value);
tpht_status_t tpht32_insert(tpht_table_t *table, uint32_t key, uint64_t value);
tpht_status_t tpht32_update(tpht_table_t *table, uint32_t key, uint64_t value);
tpht_status_t tpht32_get(tpht_table_t *table, uint32_t key, uint64_t *value_out);
tpht_status_t tpht32_remove(tpht_table_t *table, uint32_t key);

/*
 * 64-bit key tables.  The tpht64_* operations return TPHT_INVALID if handed a
 * table whose keys are not 64-bit.
 */
tpht_table_t *chained_tpht64_fixed_create(size_t capacity, uint8_t value_size);
tpht_table_t *chained_tpht64_resizable_create(size_t capacity, uint8_t value_size);
tpht_table_t *chained_tpht64_concurrent_fixed_create(size_t capacity, uint8_t value_size);
tpht_table_t *chained_tpht64_concurrent_resizable_create(size_t capacity, uint8_t value_size);
tpht_table_t *flatten_tpht64_fixed_create(size_t capacity, uint8_t value_size);
tpht_table_t *flatten_tpht64_resizable_create(size_t capacity, uint8_t value_size);

tpht_status_t tpht64_put(tpht_table_t *table, uint64_t key, uint64_t value);
tpht_status_t tpht64_insert(tpht_table_t *table, uint64_t key, uint64_t value);
tpht_status_t tpht64_update(tpht_table_t *table, uint64_t key, uint64_t value);
tpht_status_t tpht64_get(tpht_table_t *table, uint64_t key, uint64_t *value_out);
tpht_status_t tpht64_remove(tpht_table_t *table, uint64_t key);

/*
 * Concurrency is not implemented for the flattened variant yet.  These are kept
 * for source compatibility and always return NULL.
 */
tpht_table_t *flatten_tpht32_concurrent_fixed_create(size_t capacity, uint8_t value_size);
tpht_table_t *flatten_tpht32_concurrent_resizable_create(size_t capacity, uint8_t value_size);
tpht_table_t *flatten_tpht64_concurrent_fixed_create(size_t capacity, uint8_t value_size);
tpht_table_t *flatten_tpht64_concurrent_resizable_create(size_t capacity, uint8_t value_size);

#ifdef __cplusplus
}
#endif

#endif /* TPHT_H */
