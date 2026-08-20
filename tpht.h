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
 * Include tpht.h and compile tpht.c as C11.  No C++, no pthreads, no platform
 * specific code is required; concurrent tables use C11 atomics.
 *
 * There are four concrete table types, one per (variant, key width):
 *
 *     flatten_tpht32_t   flatten_tpht64_t     64-byte home blocks, no chaining
 *     chained_tpht32_t   chained_tpht64_t     tiny-pointer chains
 *
 * Each carries its own operations.  Nothing about a table is decided at run
 * time - the variant and the key width are fixed by the type, so an operation
 * never tests a configuration field before doing its work.
 *
 * Keys are 32 or 64 bits by type.  Values are any width from 1 to 8 bytes,
 * chosen per table, and cross the API as uint64_t truncated to that width.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum tpht_status {
    TPHT_OK = 0,
    TPHT_NOT_FOUND = 1,
    TPHT_EXISTS = 2,
    TPHT_FULL = 3,
    TPHT_NO_MEMORY = 4,
    /* Retained for API compatibility; no operation returns it. */
    TPHT_INVALID = 5
} tpht_status_t;

/*
 * Preconditions, not checked.  Every operation below requires a table pointer
 * returned by a create function and not yet destroyed, and every get requires
 * somewhere to put the value.  Passing null for either is a bug in the caller,
 * and these functions do not test for it: the check would cost a branch on
 * every operation to restate something the caller already knows.  A create
 * function returning null is the one case worth testing, and it is the caller's
 * job to test it.  Only *_destroy tolerates null, as free does.
 */

/*
 * The three write operations:
 *
 *   insert  append-only: it never probes for an existing key, so inserting a
 *           key that is already present appends a duplicate (matching TinyPtr's
 *           Insert).  Fastest write; callers that need uniqueness use put/update.
 *           Flattened tables bound the copies of any single key: they all live
 *           in one 64-byte home block whatever the table size, so past ~30
 *           duplicates insert reports TPHT_FULL.  Chained tables stack
 *           duplicates without that bound.
 *   put     upsert: overwrite the value if the key exists, else append it.
 *   update  overwrite-only: return TPHT_NOT_FOUND if the key is absent.
 *
 * TPHT_EXISTS is retained for API compatibility but insert no longer returns it.
 */

/*
 * How a table reacts to filling up.
 *
 *   TPHT_RESIZABLE  grows once its load passes max_load_factor, and keeps a
 *                   running entry count in order to know when that happens.
 *   TPHT_FIXED      never grows on load, and keeps no running count: capacity
 *                   provisions the storage rather than capping it, so a write
 *                   past capacity is not refused.  A hard overflow is still
 *                   absorbed by rebuilding larger, exactly as for a resizable
 *                   table, so writes do not report TPHT_FULL for fullness
 *                   alone - only TPHT_NO_MEMORY if that rebuild cannot be
 *                   allocated.  This is what makes a fixed table's writes the
 *                   cheaper of the two; the cost lands on *_size(), which
 *                   counts the table instead of reading a counter.
 */
typedef enum tpht_resize_mode {
    TPHT_FIXED = 0,
    TPHT_RESIZABLE = 1
} tpht_resize_mode_t;

/* Optional knobs; zero in any field selects the default. */
typedef struct tpht_options {
    tpht_resize_mode_t resize_mode;
    uint8_t value_size;      /* bytes, 1 to 8; 0 selects 8 */
    double max_load_factor;  /* resizable tables; 0 selects 0.85 */
    uint64_t hash_seed;      /* 0 selects a fresh random seed per table */
    size_t resize_strides;   /* concurrent resize migration chunks; 0 is automatic */
} tpht_options_t;

tpht_options_t tpht_default_options(void);

typedef struct flatten_tpht32 flatten_tpht32_t;
typedef struct flatten_tpht64 flatten_tpht64_t;
typedef struct chained_tpht32 chained_tpht32_t;
typedef struct chained_tpht64 chained_tpht64_t;
/*
 * Concurrent flattened tables.  Readers use the per-block seqlock and take no
 * locks at all; writers take the block they touch and, on overflow, the
 * dereference bins.  Both resize modes work: a resizable table grows past its
 * load factor and a fixed one absorbs hard overflows, in each case by
 * migrating into a shadow storage in parallel strides - writers pay a bounded
 * slice of migration work and keep writing (the old storage stays
 * authoritative and migrated blocks are mirrored), and readers are never
 * stalled by a resize at all.
 *
 * Concurrency model, both concurrent variants: every operation is atomic
 * with respect to every other (a get returns a value some put/update wrote in
 * full, never a torn one, and a put/insert/update/remove takes effect at one
 * point between its call and return).  Reads never block and never spin
 * longer than a concurrent writer holds the one block they need.  Progress:
 * writers to distinct blocks do not contend; a resize runs concurrently with
 * all reads and writes.  Tables may be shared freely across threads with no
 * external locking; create and destroy are not concurrent operations - no
 * other thread may touch the table during either.
 */
typedef struct flatten_conc_tpht32 flatten_conc_tpht32_t;
typedef struct flatten_conc_tpht64 flatten_conc_tpht64_t;
/*
 * Concurrent chained tables as their own type: every operation knows at
 * compile time that the table is concurrent, so nothing is re-decided per
 * call.  chained_tphtNN_create(cap, concurrent=1, opts) remains supported and
 * its handles keep working through the chained_tphtNN_* functions, at the cost
 * of one dispatch branch per call that this API does not pay.
 */
typedef struct chained_conc_tpht32 chained_conc_tpht32_t;
typedef struct chained_conc_tpht64 chained_conc_tpht64_t;

/*
 * Every *_size() below returns the number of entries held, duplicates from
 * repeated insert included.  It is O(1) on a resizable table, which maintains
 * a counter, and linear in the table on a fixed one, which does not - see
 * tpht_resize_mode.  *_capacity() reports the capacity the table was built
 * for; on a fixed table entries may exceed it.
 */

/* ------------------------------------------------------------------ flatten */
/*
 * The sequential flattened tables.  A hard overflow - a home block that cannot
 * address another tuple, or an exhausted dereference table - is absorbed by
 * rebuilding with more blocks.  Only an overflow that no growth can split
 * still reports TPHT_FULL: duplicates of one key share a home block whatever
 * the block count (see insert above), and a 32-bit-key table's block count is
 * pinned by the key width itself (blocks x fingerprints cannot outnumber the
 * keys), which caps a flatten_*32 table around the hundred-million-entry mark
 * however much it grows.  64-bit keys push that bound past any allocatable
 * table.  For concurrent use, see flatten_conc_tphtNN below.
 */
flatten_tpht32_t *flatten_tpht32_fixed_create(size_t capacity, uint8_t value_size);
flatten_tpht32_t *flatten_tpht32_resizable_create(size_t capacity, uint8_t value_size);
flatten_tpht32_t *flatten_tpht32_create(size_t capacity, const tpht_options_t *options);
void flatten_tpht32_destroy(flatten_tpht32_t *table);

tpht_status_t flatten_tpht32_put(flatten_tpht32_t *table, uint32_t key, uint64_t value);
tpht_status_t flatten_tpht32_insert(flatten_tpht32_t *table, uint32_t key, uint64_t value);
tpht_status_t flatten_tpht32_update(flatten_tpht32_t *table, uint32_t key, uint64_t value);
tpht_status_t flatten_tpht32_get(flatten_tpht32_t *table, uint32_t key, uint64_t *value_out);
tpht_status_t flatten_tpht32_remove(flatten_tpht32_t *table, uint32_t key);

size_t flatten_tpht32_size(const flatten_tpht32_t *table);
size_t flatten_tpht32_capacity(const flatten_tpht32_t *table);
size_t flatten_tpht32_memory_bytes(const flatten_tpht32_t *table);

flatten_tpht64_t *flatten_tpht64_fixed_create(size_t capacity, uint8_t value_size);
flatten_tpht64_t *flatten_tpht64_resizable_create(size_t capacity, uint8_t value_size);
flatten_tpht64_t *flatten_tpht64_create(size_t capacity, const tpht_options_t *options);
void flatten_tpht64_destroy(flatten_tpht64_t *table);

tpht_status_t flatten_tpht64_put(flatten_tpht64_t *table, uint64_t key, uint64_t value);
tpht_status_t flatten_tpht64_insert(flatten_tpht64_t *table, uint64_t key, uint64_t value);
tpht_status_t flatten_tpht64_update(flatten_tpht64_t *table, uint64_t key, uint64_t value);
tpht_status_t flatten_tpht64_get(flatten_tpht64_t *table, uint64_t key, uint64_t *value_out);
tpht_status_t flatten_tpht64_remove(flatten_tpht64_t *table, uint64_t key);

size_t flatten_tpht64_size(const flatten_tpht64_t *table);
size_t flatten_tpht64_capacity(const flatten_tpht64_t *table);
size_t flatten_tpht64_memory_bytes(const flatten_tpht64_t *table);

flatten_conc_tpht32_t *flatten_conc_tpht32_fixed_create(size_t capacity, uint8_t value_size);
flatten_conc_tpht32_t *flatten_conc_tpht32_resizable_create(size_t capacity, uint8_t value_size);
flatten_conc_tpht32_t *flatten_conc_tpht32_create(size_t capacity, const tpht_options_t *options);
void flatten_conc_tpht32_destroy(flatten_conc_tpht32_t *table);
tpht_status_t flatten_conc_tpht32_put(flatten_conc_tpht32_t *table, uint32_t key, uint64_t value);
tpht_status_t flatten_conc_tpht32_insert(flatten_conc_tpht32_t *table, uint32_t key, uint64_t value);
tpht_status_t flatten_conc_tpht32_update(flatten_conc_tpht32_t *table, uint32_t key, uint64_t value);
tpht_status_t flatten_conc_tpht32_get(flatten_conc_tpht32_t *table, uint32_t key, uint64_t *value_out);
tpht_status_t flatten_conc_tpht32_remove(flatten_conc_tpht32_t *table, uint32_t key);
size_t flatten_conc_tpht32_size(const flatten_conc_tpht32_t *table);
size_t flatten_conc_tpht32_capacity(const flatten_conc_tpht32_t *table);
size_t flatten_conc_tpht32_memory_bytes(const flatten_conc_tpht32_t *table);

flatten_conc_tpht64_t *flatten_conc_tpht64_fixed_create(size_t capacity, uint8_t value_size);
flatten_conc_tpht64_t *flatten_conc_tpht64_resizable_create(size_t capacity, uint8_t value_size);
flatten_conc_tpht64_t *flatten_conc_tpht64_create(size_t capacity, const tpht_options_t *options);
void flatten_conc_tpht64_destroy(flatten_conc_tpht64_t *table);
tpht_status_t flatten_conc_tpht64_put(flatten_conc_tpht64_t *table, uint64_t key, uint64_t value);
tpht_status_t flatten_conc_tpht64_insert(flatten_conc_tpht64_t *table, uint64_t key, uint64_t value);
tpht_status_t flatten_conc_tpht64_update(flatten_conc_tpht64_t *table, uint64_t key, uint64_t value);
tpht_status_t flatten_conc_tpht64_get(flatten_conc_tpht64_t *table, uint64_t key, uint64_t *value_out);
tpht_status_t flatten_conc_tpht64_remove(flatten_conc_tpht64_t *table, uint64_t key);
size_t flatten_conc_tpht64_size(const flatten_conc_tpht64_t *table);
size_t flatten_conc_tpht64_capacity(const flatten_conc_tpht64_t *table);
size_t flatten_conc_tpht64_memory_bytes(const flatten_conc_tpht64_t *table);

chained_conc_tpht32_t *chained_conc_tpht32_fixed_create(size_t capacity, uint8_t value_size);
chained_conc_tpht32_t *chained_conc_tpht32_resizable_create(size_t capacity, uint8_t value_size);
chained_conc_tpht32_t *chained_conc_tpht32_create(size_t capacity, const tpht_options_t *options);
void chained_conc_tpht32_destroy(chained_conc_tpht32_t *table);
tpht_status_t chained_conc_tpht32_put(chained_conc_tpht32_t *table, uint32_t key, uint64_t value);
tpht_status_t chained_conc_tpht32_insert(chained_conc_tpht32_t *table, uint32_t key, uint64_t value);
tpht_status_t chained_conc_tpht32_update(chained_conc_tpht32_t *table, uint32_t key, uint64_t value);
tpht_status_t chained_conc_tpht32_get(chained_conc_tpht32_t *table, uint32_t key, uint64_t *value_out);
tpht_status_t chained_conc_tpht32_remove(chained_conc_tpht32_t *table, uint32_t key);
size_t chained_conc_tpht32_size(const chained_conc_tpht32_t *table);
size_t chained_conc_tpht32_capacity(const chained_conc_tpht32_t *table);
size_t chained_conc_tpht32_memory_bytes(const chained_conc_tpht32_t *table);

chained_conc_tpht64_t *chained_conc_tpht64_fixed_create(size_t capacity, uint8_t value_size);
chained_conc_tpht64_t *chained_conc_tpht64_resizable_create(size_t capacity, uint8_t value_size);
chained_conc_tpht64_t *chained_conc_tpht64_create(size_t capacity, const tpht_options_t *options);
void chained_conc_tpht64_destroy(chained_conc_tpht64_t *table);
tpht_status_t chained_conc_tpht64_put(chained_conc_tpht64_t *table, uint64_t key, uint64_t value);
tpht_status_t chained_conc_tpht64_insert(chained_conc_tpht64_t *table, uint64_t key, uint64_t value);
tpht_status_t chained_conc_tpht64_update(chained_conc_tpht64_t *table, uint64_t key, uint64_t value);
tpht_status_t chained_conc_tpht64_get(chained_conc_tpht64_t *table, uint64_t key, uint64_t *value_out);
tpht_status_t chained_conc_tpht64_remove(chained_conc_tpht64_t *table, uint64_t key);
size_t chained_conc_tpht64_size(const chained_conc_tpht64_t *table);
size_t chained_conc_tpht64_capacity(const chained_conc_tpht64_t *table);
size_t chained_conc_tpht64_memory_bytes(const chained_conc_tpht64_t *table);

/* ------------------------------------------------------------------ chained */
chained_tpht32_t *chained_tpht32_fixed_create(size_t capacity, uint8_t value_size);
chained_tpht32_t *chained_tpht32_resizable_create(size_t capacity, uint8_t value_size);
chained_tpht32_t *chained_tpht32_concurrent_fixed_create(size_t capacity, uint8_t value_size);
chained_tpht32_t *chained_tpht32_concurrent_resizable_create(size_t capacity, uint8_t value_size);
chained_tpht32_t *chained_tpht32_create(size_t capacity, int concurrent,
                                        const tpht_options_t *options);
void chained_tpht32_destroy(chained_tpht32_t *table);

tpht_status_t chained_tpht32_put(chained_tpht32_t *table, uint32_t key, uint64_t value);
tpht_status_t chained_tpht32_insert(chained_tpht32_t *table, uint32_t key, uint64_t value);
tpht_status_t chained_tpht32_update(chained_tpht32_t *table, uint32_t key, uint64_t value);
tpht_status_t chained_tpht32_get(chained_tpht32_t *table, uint32_t key, uint64_t *value_out);
tpht_status_t chained_tpht32_remove(chained_tpht32_t *table, uint32_t key);

size_t chained_tpht32_size(const chained_tpht32_t *table);
size_t chained_tpht32_capacity(const chained_tpht32_t *table);
size_t chained_tpht32_memory_bytes(const chained_tpht32_t *table);

chained_tpht64_t *chained_tpht64_fixed_create(size_t capacity, uint8_t value_size);
chained_tpht64_t *chained_tpht64_resizable_create(size_t capacity, uint8_t value_size);
chained_tpht64_t *chained_tpht64_concurrent_fixed_create(size_t capacity, uint8_t value_size);
chained_tpht64_t *chained_tpht64_concurrent_resizable_create(size_t capacity, uint8_t value_size);
chained_tpht64_t *chained_tpht64_create(size_t capacity, int concurrent,
                                        const tpht_options_t *options);
void chained_tpht64_destroy(chained_tpht64_t *table);

tpht_status_t chained_tpht64_put(chained_tpht64_t *table, uint64_t key, uint64_t value);
tpht_status_t chained_tpht64_insert(chained_tpht64_t *table, uint64_t key, uint64_t value);
tpht_status_t chained_tpht64_update(chained_tpht64_t *table, uint64_t key, uint64_t value);
tpht_status_t chained_tpht64_get(chained_tpht64_t *table, uint64_t key, uint64_t *value_out);
tpht_status_t chained_tpht64_remove(chained_tpht64_t *table, uint64_t key);

size_t chained_tpht64_size(const chained_tpht64_t *table);
size_t chained_tpht64_capacity(const chained_tpht64_t *table);
size_t chained_tpht64_memory_bytes(const chained_tpht64_t *table);

#ifdef __cplusplus
}
#endif

#endif /* TPHT_H */
