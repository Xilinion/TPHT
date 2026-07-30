#include "tpht.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#define TPHT_DEFAULT_BIN_SIZE 127u
#define TPHT_DEFAULT_LOAD_FACTOR 0.85
#define TPHT_MIN_CAPACITY 16u
#define TPHT_FLAT_CLOUD_BYTES 64u
#define TPHT_FLAT_META_BYTES 8u

typedef struct tpht_pool {
    uint8_t *entries;
    uint8_t *cnt_head; /* [count, freelist-head] per bin. */
    size_t bin_count;
    uint8_t bin_size;
    size_t entry_size; /* next-byte + key + value. */
} tpht_pool_t;

struct tpht_table {
    tpht_config_t cfg;
    size_t size;
    size_t capacity;

    size_t key_size;
    size_t value_size;
    size_t inline_entry_size; /* key + value. */
    size_t pool_entry_size;   /* next + key + value. */

    size_t base_count; /* chained base buckets or flatten clouds. */
    uint8_t *heads;    /* chained heads, or flatten overflow heads. */

    uint8_t *flat_count;
    uint8_t *flat_fp;
    uint8_t *flat_entries;
    uint8_t flat_inline_cap;

    tpht_pool_t pool;
    atomic_flag lock;
};

static int tpht_valid_size(uint8_t n) { return n == 2u || n == 4u || n == 8u; }

static size_t tpht_max_size(size_t a, size_t b) { return a > b ? a : b; }

static size_t tpht_pow2_ceil(size_t x) {
    size_t p = 1;
    if (x <= 1) return 1;
    while (p < x && p <= ((size_t)-1 / 2)) p <<= 1;
    return p < x ? x : p;
}

static uint64_t tpht_read_le(const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p;
    uint64_t x = 0;
    size_t i;
    for (i = 0; i < n; ++i) x |= ((uint64_t)b[i]) << (8u * i);
    return x;
}

static void tpht_write_le(void *p, size_t n, uint64_t x) {
    uint8_t *b = (uint8_t *)p;
    size_t i;
    for (i = 0; i < n; ++i) b[i] = (uint8_t)(x >> (8u * i));
}

static uint64_t tpht_mix64(uint64_t x) {
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    x ^= x >> 31;
    return x;
}

static uint64_t tpht_hash_bytes(const void *data, size_t len, uint64_t seed) {
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = seed ^ UINT64_C(0x9e3779b97f4a7c15) ^ (uint64_t)len;
    size_t i;
    for (i = 0; i < len; ++i) {
        h ^= (uint64_t)p[i] + UINT64_C(0x9e3779b97f4a7c15) + (h << 6) + (h >> 2);
    }
    return tpht_mix64(h);
}

static uint64_t tpht_hash_word(uint64_t x, uint64_t seed) {
    return tpht_mix64(x ^ seed ^ UINT64_C(0x517cc1b727220a95));
}

static void tpht_lock(tpht_table_t *t) {
    if (t->cfg.threading == TPHT_CONCURRENT) {
        while (atomic_flag_test_and_set_explicit(&t->lock, memory_order_acquire)) {
        }
    }
}

static void tpht_unlock(tpht_table_t *t) {
    if (t->cfg.threading == TPHT_CONCURRENT) {
        atomic_flag_clear_explicit(&t->lock, memory_order_release);
    }
}

static uint8_t *tpht_pool_entry(tpht_pool_t *p, size_t bin, uint8_t pos) {
    return p->entries + ((bin * (size_t)p->bin_size + (size_t)pos) * p->entry_size);
}

static uint8_t tpht_pool_count(const tpht_pool_t *p, size_t bin) {
    return p->cnt_head[bin << 1u];
}

static uint8_t *tpht_pool_count_ptr(tpht_pool_t *p, size_t bin) {
    return &p->cnt_head[bin << 1u];
}

static uint8_t *tpht_pool_head_ptr(tpht_pool_t *p, size_t bin) {
    return &p->cnt_head[(bin << 1u) | 1u];
}

static uint8_t *tpht_pool_deref(tpht_table_t *t, uint64_t deref_key, uint8_t ptr) {
    uint8_t flag = (uint8_t)(ptr >> 7u);
    uint8_t pos = (uint8_t)((ptr & 0x7fu) - 1u);
    uint64_t seed = t->cfg.hash_seed;
    size_t bin = (size_t)(tpht_hash_word(deref_key, seed + (flag ? 0x200 : 0x100)) % t->pool.bin_count);
    return tpht_pool_entry(&t->pool, bin, pos);
}

static uint8_t *tpht_pool_alloc(tpht_table_t *t, uint64_t deref_key, uint8_t *encoded_out) {
    size_t bin1 = (size_t)(tpht_hash_word(deref_key, t->cfg.hash_seed + 0x100) % t->pool.bin_count);
    size_t bin2 = (size_t)(tpht_hash_word(deref_key, t->cfg.hash_seed + 0x200) % t->pool.bin_count);
    uint8_t flag = 0;
    size_t bin = bin1;
    uint8_t *cnt;
    uint8_t *head;
    uint8_t pos;
    uint8_t *entry;

    if (bin1 != bin2 && tpht_pool_count(&t->pool, bin1) > tpht_pool_count(&t->pool, bin2)) {
        bin = bin2;
        flag = 1;
    }

    cnt = tpht_pool_count_ptr(&t->pool, bin);
    head = tpht_pool_head_ptr(&t->pool, bin);
    if (*head >= t->pool.bin_size) return NULL;

    pos = *head;
    entry = tpht_pool_entry(&t->pool, bin, pos);
    *head = (uint8_t)(pos + 1u + entry[0]);
    if (*head > t->pool.bin_size) *head = (uint8_t)(*head - (t->pool.bin_size + 1u));
    *cnt = (uint8_t)(*cnt + 1u);
    *encoded_out = (uint8_t)((pos + 1u) | (flag << 7u));
    return entry;
}

static void tpht_pool_free(tpht_table_t *t, uint8_t encoded_ptr, uint8_t *entry) {
    size_t ordinal = (size_t)((entry - t->pool.entries) / (ptrdiff_t)t->pool.entry_size);
    size_t bin = ordinal / t->pool.bin_size;
    uint8_t pos = (uint8_t)(ordinal % t->pool.bin_size);
    uint8_t *cnt = tpht_pool_count_ptr(&t->pool, bin);
    uint8_t *head = tpht_pool_head_ptr(&t->pool, bin);
    (void)encoded_ptr;
    entry[0] = (uint8_t)(*head + t->pool.bin_size - pos);
    if (entry[0] > t->pool.bin_size) entry[0] = (uint8_t)(entry[0] - (t->pool.bin_size + 1u));
    *head = pos;
    *cnt = (uint8_t)(*cnt - 1u);
}

static void tpht_free_storage(tpht_table_t *t) {
    free(t->heads);
    free(t->flat_count);
    free(t->flat_fp);
    free(t->flat_entries);
    free(t->pool.entries);
    free(t->pool.cnt_head);
    t->heads = NULL;
    t->flat_count = NULL;
    t->flat_fp = NULL;
    t->flat_entries = NULL;
    t->pool.entries = NULL;
    t->pool.cnt_head = NULL;
}

static int tpht_alloc_storage(tpht_table_t *t, size_t capacity) {
    size_t inline_bytes;
    size_t overflow_slots;
    size_t cloud_target;

    t->capacity = tpht_max_size(capacity, TPHT_MIN_CAPACITY);
    t->key_size = t->cfg.key_size;
    t->value_size = t->cfg.value_size;
    t->inline_entry_size = t->key_size + t->value_size;
    t->pool_entry_size = 1u + t->inline_entry_size;
    t->pool.bin_size = t->cfg.bin_size ? t->cfg.bin_size : TPHT_DEFAULT_BIN_SIZE;
    if (t->pool.bin_size == 0 || t->pool.bin_size > 127u) return 0;
    t->pool.entry_size = t->pool_entry_size;

    if (t->cfg.variant == TPHT_CHAINED) {
        t->base_count = tpht_pow2_ceil(t->capacity);
        t->heads = (uint8_t *)calloc(t->base_count, 1);
        overflow_slots = t->capacity + t->capacity / 4u + (size_t)t->pool.bin_size;
    } else {
        inline_bytes = TPHT_FLAT_CLOUD_BYTES - TPHT_FLAT_META_BYTES;
        t->flat_inline_cap = (uint8_t)(inline_bytes / (1u + t->inline_entry_size));
        if (t->flat_inline_cap == 0) t->flat_inline_cap = 1;
        if (t->flat_inline_cap > 7u) t->flat_inline_cap = 7u;

        cloud_target = (t->capacity + t->flat_inline_cap - 1u) / t->flat_inline_cap;
        t->base_count = tpht_pow2_ceil(tpht_max_size(cloud_target, 1));
        t->heads = (uint8_t *)calloc(t->base_count, 1);
        t->flat_count = (uint8_t *)calloc(t->base_count, 1);
        t->flat_fp = (uint8_t *)calloc(t->base_count * (size_t)t->flat_inline_cap, 1);
        t->flat_entries = (uint8_t *)calloc(t->base_count * (size_t)t->flat_inline_cap, t->inline_entry_size);
        overflow_slots = t->capacity / 2u + (size_t)t->pool.bin_size;
    }

    t->pool.bin_count = (overflow_slots + t->pool.bin_size - 1u) / t->pool.bin_size;
    t->pool.bin_count = tpht_max_size(t->pool.bin_count, 1);
    t->pool.entries = (uint8_t *)calloc(t->pool.bin_count * (size_t)t->pool.bin_size, t->pool.entry_size);
    t->pool.cnt_head = (uint8_t *)calloc(t->pool.bin_count * 2u, 1);

    if (!t->heads || !t->pool.entries || !t->pool.cnt_head ||
        (t->cfg.variant == TPHT_FLATTEN && (!t->flat_count || !t->flat_fp || !t->flat_entries))) {
        tpht_free_storage(t);
        return 0;
    }
    return 1;
}

static uint8_t *tpht_inline_entry(tpht_table_t *t, size_t cloud, size_t pos) {
    return t->flat_entries + ((cloud * (size_t)t->flat_inline_cap + pos) * t->inline_entry_size);
}

static int tpht_key_equal(tpht_table_t *t, const uint8_t *a, const void *key) {
    return memcmp(a, key, t->key_size) == 0;
}

static size_t tpht_chained_base(tpht_table_t *t, const void *key) {
    return (size_t)(tpht_hash_bytes(key, t->key_size, t->cfg.hash_seed) & (uint64_t)(t->base_count - 1u));
}

static size_t tpht_flat_cloud(tpht_table_t *t, const void *key) {
    return (size_t)(tpht_hash_bytes(key, t->key_size, t->cfg.hash_seed + 0x8000u) & (uint64_t)(t->base_count - 1u));
}

static uint8_t tpht_fingerprint(tpht_table_t *t, const void *key) {
    uint8_t fp = (uint8_t)tpht_hash_bytes(key, t->key_size, t->cfg.hash_seed + 0x55u);
    return fp ? fp : 1u;
}

static tpht_status_t tpht_chained_get_raw(tpht_table_t *t, const void *key, void *value_out) {
    size_t base = tpht_chained_base(t, key);
    uint8_t *prev = &t->heads[base];
    while (*prev) {
        uint8_t *entry = tpht_pool_deref(t, (uint64_t)(uintptr_t)prev, *prev);
        if (tpht_key_equal(t, entry + 1u, key)) {
            if (value_out) memcpy(value_out, entry + 1u + t->key_size, t->value_size);
            return TPHT_OK;
        }
        prev = entry;
    }
    return TPHT_NOT_FOUND;
}

static tpht_status_t tpht_chained_insert_raw(tpht_table_t *t, const void *key,
                                             const void *value, int replace) {
    size_t base = tpht_chained_base(t, key);
    uint8_t *prev = &t->heads[base];
    uint8_t encoded;
    uint8_t *entry;
    while (*prev) {
        entry = tpht_pool_deref(t, (uint64_t)(uintptr_t)prev, *prev);
        if (tpht_key_equal(t, entry + 1u, key)) {
            if (!replace) return TPHT_EXISTS;
            memcpy(entry + 1u + t->key_size, value, t->value_size);
            return TPHT_OK;
        }
        prev = entry;
    }

    entry = tpht_pool_alloc(t, (uint64_t)(uintptr_t)prev, &encoded);
    if (!entry) return TPHT_FULL;
    *prev = encoded;
    entry[0] = 0;
    memcpy(entry + 1u, key, t->key_size);
    memcpy(entry + 1u + t->key_size, value, t->value_size);
    t->size++;
    return TPHT_OK;
}

static tpht_status_t tpht_chained_remove_raw(tpht_table_t *t, const void *key) {
    size_t base = tpht_chained_base(t, key);
    uint8_t *prev = &t->heads[base];
    uint8_t *target = NULL;
    uint8_t *last_prev = NULL;
    uint8_t *last_entry = NULL;
    uint8_t last_encoded = 0;
    while (*prev) {
        uint8_t encoded = *prev;
        uint8_t *entry = tpht_pool_deref(t, (uint64_t)(uintptr_t)prev, encoded);
        if (tpht_key_equal(t, entry + 1u, key)) {
            target = entry;
        }
        last_prev = prev;
        last_entry = entry;
        last_encoded = encoded;
        prev = entry;
    }
    if (!target) return TPHT_NOT_FOUND;

    if (target != last_entry) {
        uint8_t target_next = target[0];
        memcpy(target, last_entry, t->pool.entry_size);
        target[0] = target_next;
    }
    *last_prev = 0;
    tpht_pool_free(t, last_encoded, last_entry);
    t->size--;
    return TPHT_OK;
}

static tpht_status_t tpht_flat_get_raw(tpht_table_t *t, const void *key, void *value_out) {
    size_t cloud = tpht_flat_cloud(t, key);
    uint8_t fp = tpht_fingerprint(t, key);
    uint8_t count = t->flat_count[cloud];
    uint8_t *fps = t->flat_fp + cloud * (size_t)t->flat_inline_cap;
    uint8_t *prev;
    uint8_t i;

    for (i = 0; i < count; ++i) {
        uint8_t *slot = tpht_inline_entry(t, cloud, i);
        if (fps[i] == fp && tpht_key_equal(t, slot, key)) {
            if (value_out) memcpy(value_out, slot + t->key_size, t->value_size);
            return TPHT_OK;
        }
    }

    prev = &t->heads[cloud];
    while (*prev) {
        uint8_t *entry = tpht_pool_deref(t, (uint64_t)cloud, *prev);
        if (tpht_key_equal(t, entry + 1u, key)) {
            if (value_out) memcpy(value_out, entry + 1u + t->key_size, t->value_size);
            return TPHT_OK;
        }
        prev = entry;
    }
    return TPHT_NOT_FOUND;
}

static tpht_status_t tpht_flat_insert_raw(tpht_table_t *t, const void *key,
                                          const void *value, int replace) {
    size_t cloud = tpht_flat_cloud(t, key);
    uint8_t fp = tpht_fingerprint(t, key);
    uint8_t count = t->flat_count[cloud];
    uint8_t *fps = t->flat_fp + cloud * (size_t)t->flat_inline_cap;
    uint8_t *prev;
    uint8_t encoded;
    uint8_t i;
    uint8_t *entry;

    for (i = 0; i < count; ++i) {
        uint8_t *slot = tpht_inline_entry(t, cloud, i);
        if (fps[i] == fp && tpht_key_equal(t, slot, key)) {
            if (!replace) return TPHT_EXISTS;
            memcpy(slot + t->key_size, value, t->value_size);
            return TPHT_OK;
        }
    }
    prev = &t->heads[cloud];
    while (*prev) {
        entry = tpht_pool_deref(t, (uint64_t)cloud, *prev);
        if (tpht_key_equal(t, entry + 1u, key)) {
            if (!replace) return TPHT_EXISTS;
            memcpy(entry + 1u + t->key_size, value, t->value_size);
            return TPHT_OK;
        }
        prev = entry;
    }

    if (count < t->flat_inline_cap) {
        uint8_t *slot = tpht_inline_entry(t, cloud, count);
        fps[count] = fp;
        memcpy(slot, key, t->key_size);
        memcpy(slot + t->key_size, value, t->value_size);
        t->flat_count[cloud] = (uint8_t)(count + 1u);
        t->size++;
        return TPHT_OK;
    }

    entry = tpht_pool_alloc(t, (uint64_t)cloud, &encoded);
    if (!entry) return TPHT_FULL;
    entry[0] = t->heads[cloud];
    t->heads[cloud] = encoded;
    memcpy(entry + 1u, key, t->key_size);
    memcpy(entry + 1u + t->key_size, value, t->value_size);
    t->size++;
    return TPHT_OK;
}

static tpht_status_t tpht_flat_remove_raw(tpht_table_t *t, const void *key) {
    size_t cloud = tpht_flat_cloud(t, key);
    uint8_t fp = tpht_fingerprint(t, key);
    uint8_t count = t->flat_count[cloud];
    uint8_t *fps = t->flat_fp + cloud * (size_t)t->flat_inline_cap;
    uint8_t i;
    uint8_t *prev;

    for (i = 0; i < count; ++i) {
        uint8_t *slot = tpht_inline_entry(t, cloud, i);
        if (fps[i] == fp && tpht_key_equal(t, slot, key)) {
            if (t->heads[cloud]) {
                uint8_t encoded = t->heads[cloud];
                uint8_t *entry = tpht_pool_deref(t, (uint64_t)cloud, encoded);
                t->heads[cloud] = entry[0];
                fps[i] = tpht_fingerprint(t, entry + 1u);
                memcpy(slot, entry + 1u, t->inline_entry_size);
                tpht_pool_free(t, encoded, entry);
            } else {
                uint8_t last = (uint8_t)(count - 1u);
                if (i != last) {
                    memcpy(slot, tpht_inline_entry(t, cloud, last), t->inline_entry_size);
                    fps[i] = fps[last];
                }
                t->flat_count[cloud] = last;
            }
            t->size--;
            return TPHT_OK;
        }
    }

    prev = &t->heads[cloud];
    while (*prev) {
        uint8_t encoded = *prev;
        uint8_t *entry = tpht_pool_deref(t, (uint64_t)cloud, encoded);
        if (tpht_key_equal(t, entry + 1u, key)) {
            *prev = entry[0];
            tpht_pool_free(t, encoded, entry);
            t->size--;
            return TPHT_OK;
        }
        prev = entry;
    }
    return TPHT_NOT_FOUND;
}

static tpht_status_t tpht_raw_insert(tpht_table_t *t, const void *key, const void *value, int replace) {
    return t->cfg.variant == TPHT_CHAINED ? tpht_chained_insert_raw(t, key, value, replace)
                                          : tpht_flat_insert_raw(t, key, value, replace);
}

static tpht_status_t tpht_raw_get(tpht_table_t *t, const void *key, void *value_out) {
    return t->cfg.variant == TPHT_CHAINED ? tpht_chained_get_raw(t, key, value_out)
                                          : tpht_flat_get_raw(t, key, value_out);
}

static tpht_status_t tpht_raw_remove(tpht_table_t *t, const void *key) {
    return t->cfg.variant == TPHT_CHAINED ? tpht_chained_remove_raw(t, key)
                                          : tpht_flat_remove_raw(t, key);
}

static tpht_status_t tpht_resize_locked(tpht_table_t *t, size_t new_capacity) {
    tpht_table_t nt;
    size_t i;
    memset(&nt, 0, sizeof(nt));
    nt.cfg = t->cfg;
    atomic_flag_clear(&nt.lock);
    if (!tpht_alloc_storage(&nt, new_capacity)) return TPHT_NO_MEMORY;

    if (t->cfg.variant == TPHT_CHAINED) {
        for (i = 0; i < t->base_count; ++i) {
            uint8_t *prev = &t->heads[i];
            while (*prev) {
                uint8_t *entry = tpht_pool_deref(t, (uint64_t)(uintptr_t)prev, *prev);
                tpht_status_t st = tpht_raw_insert(&nt, entry + 1u, entry + 1u + t->key_size, 0);
                if (st != TPHT_OK) {
                    tpht_free_storage(&nt);
                    return st == TPHT_FULL ? TPHT_NO_MEMORY : st;
                }
                prev = entry;
            }
        }
    } else {
        for (i = 0; i < t->base_count; ++i) {
            uint8_t j;
            uint8_t *prev;
            for (j = 0; j < t->flat_count[i]; ++j) {
                uint8_t *slot = tpht_inline_entry(t, i, j);
                tpht_status_t st = tpht_raw_insert(&nt, slot, slot + t->key_size, 0);
                if (st != TPHT_OK) {
                    tpht_free_storage(&nt);
                    return st == TPHT_FULL ? TPHT_NO_MEMORY : st;
                }
            }
            prev = &t->heads[i];
            while (*prev) {
                uint8_t *entry = tpht_pool_deref(t, (uint64_t)i, *prev);
                tpht_status_t st = tpht_raw_insert(&nt, entry + 1u, entry + 1u + t->key_size, 0);
                if (st != TPHT_OK) {
                    tpht_free_storage(&nt);
                    return st == TPHT_FULL ? TPHT_NO_MEMORY : st;
                }
                prev = entry;
            }
        }
    }

    tpht_free_storage(t);
    t->capacity = nt.capacity;
    t->size = nt.size;
    t->base_count = nt.base_count;
    t->heads = nt.heads;
    t->flat_count = nt.flat_count;
    t->flat_fp = nt.flat_fp;
    t->flat_entries = nt.flat_entries;
    t->flat_inline_cap = nt.flat_inline_cap;
    t->pool = nt.pool;
    nt.heads = NULL;
    nt.flat_count = NULL;
    nt.flat_fp = NULL;
    nt.flat_entries = NULL;
    nt.pool.entries = NULL;
    nt.pool.cnt_head = NULL;
    return TPHT_OK;
}

tpht_config_t tpht_default_config(void) {
    tpht_config_t c;
    c.variant = TPHT_CHAINED;
    c.threading = TPHT_SEQUENTIAL;
    c.resize_mode = TPHT_RESIZABLE;
    c.initial_capacity = 1024;
    c.key_size = 8;
    c.value_size = 8;
    c.bin_size = TPHT_DEFAULT_BIN_SIZE;
    c.max_load_factor = TPHT_DEFAULT_LOAD_FACTOR;
    c.hash_seed = UINT64_C(0x243f6a8885a308d3);
    return c;
}

tpht_table_t *tpht_create(const tpht_config_t *config) {
    tpht_table_t *t;
    tpht_config_t c = config ? *config : tpht_default_config();
    if (c.variant != TPHT_CHAINED && c.variant != TPHT_FLATTEN) return NULL;
    if (!tpht_valid_size(c.key_size) || !tpht_valid_size(c.value_size)) return NULL;
    if (c.bin_size == 0) c.bin_size = TPHT_DEFAULT_BIN_SIZE;
    if (c.bin_size > 127u) return NULL;
    if (c.initial_capacity == 0) c.initial_capacity = TPHT_MIN_CAPACITY;
    if (c.max_load_factor <= 0.0 || c.max_load_factor > 1.0) c.max_load_factor = TPHT_DEFAULT_LOAD_FACTOR;
    if (c.hash_seed == 0) c.hash_seed = UINT64_C(0x243f6a8885a308d3);

    t = (tpht_table_t *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->cfg = c;
    atomic_flag_clear(&t->lock);
    if (!tpht_alloc_storage(t, c.initial_capacity)) {
        free(t);
        return NULL;
    }
    return t;
}

void tpht_destroy(tpht_table_t *table) {
    if (!table) return;
    tpht_free_storage(table);
    free(table);
}

tpht_status_t tpht_put(tpht_table_t *table, const void *key, const void *value) {
    tpht_status_t st;
    if (!table || !key || !value) return TPHT_INVALID;
    tpht_lock(table);
    if (table->cfg.resize_mode == TPHT_FIXED && table->size >= table->capacity &&
        tpht_raw_get(table, key, NULL) != TPHT_OK) {
        tpht_unlock(table);
        return TPHT_FULL;
    }
    if (table->cfg.resize_mode == TPHT_RESIZABLE &&
        (double)(table->size + 1u) > (double)table->capacity * table->cfg.max_load_factor) {
        st = tpht_resize_locked(table, table->capacity * 2u);
        if (st != TPHT_OK) {
            tpht_unlock(table);
            return st;
        }
    }
    st = tpht_raw_insert(table, key, value, 1);
    if (st == TPHT_FULL && table->cfg.resize_mode == TPHT_RESIZABLE) {
        st = tpht_resize_locked(table, table->capacity * 2u);
        if (st == TPHT_OK) st = tpht_raw_insert(table, key, value, 1);
    }
    tpht_unlock(table);
    return st;
}

tpht_status_t tpht_insert(tpht_table_t *table, const void *key, const void *value) {
    tpht_status_t st;
    if (!table || !key || !value) return TPHT_INVALID;
    tpht_lock(table);
    if (table->cfg.resize_mode == TPHT_FIXED && table->size >= table->capacity) {
        st = tpht_raw_get(table, key, NULL);
        tpht_unlock(table);
        return st == TPHT_OK ? TPHT_EXISTS : TPHT_FULL;
    }
    if (table->cfg.resize_mode == TPHT_RESIZABLE &&
        (double)(table->size + 1u) > (double)table->capacity * table->cfg.max_load_factor) {
        st = tpht_resize_locked(table, table->capacity * 2u);
        if (st != TPHT_OK) {
            tpht_unlock(table);
            return st;
        }
    }
    st = tpht_raw_insert(table, key, value, 0);
    if (st == TPHT_FULL && table->cfg.resize_mode == TPHT_RESIZABLE) {
        st = tpht_resize_locked(table, table->capacity * 2u);
        if (st == TPHT_OK) st = tpht_raw_insert(table, key, value, 0);
    }
    tpht_unlock(table);
    return st;
}

tpht_status_t tpht_update(tpht_table_t *table, const void *key, const void *value) {
    tpht_status_t st;
    uint8_t old_value[8];
    if (!table || !key || !value) return TPHT_INVALID;
    tpht_lock(table);
    st = tpht_raw_get(table, key, old_value);
    if (st == TPHT_OK) st = tpht_raw_insert(table, key, value, 1);
    tpht_unlock(table);
    return st;
}

tpht_status_t tpht_get(tpht_table_t *table, const void *key, void *value_out) {
    tpht_status_t st;
    if (!table || !key || !value_out) return TPHT_INVALID;
    tpht_lock(table);
    st = tpht_raw_get(table, key, value_out);
    tpht_unlock(table);
    return st;
}

tpht_status_t tpht_remove(tpht_table_t *table, const void *key) {
    tpht_status_t st;
    if (!table || !key) return TPHT_INVALID;
    tpht_lock(table);
    st = tpht_raw_remove(table, key);
    tpht_unlock(table);
    return st;
}

size_t tpht_size(const tpht_table_t *table) { return table ? table->size : 0; }
size_t tpht_capacity(const tpht_table_t *table) { return table ? table->capacity : 0; }
tpht_variant_t tpht_get_variant(const tpht_table_t *table) { return table ? table->cfg.variant : 0; }
tpht_threading_t tpht_get_threading(const tpht_table_t *table) { return table ? table->cfg.threading : 0; }
tpht_resize_mode_t tpht_get_resize_mode(const tpht_table_t *table) { return table ? table->cfg.resize_mode : 0; }

static tpht_table_t *tpht_make(size_t capacity, uint8_t key_size, uint8_t value_size,
                               tpht_variant_t variant, tpht_threading_t threading,
                               tpht_resize_mode_t resize_mode) {
    tpht_config_t c = tpht_default_config();
    c.variant = variant;
    c.threading = threading;
    c.resize_mode = resize_mode;
    c.initial_capacity = capacity;
    c.key_size = key_size;
    c.value_size = value_size;
    return tpht_create(&c);
}

tpht_table_t *tpht_chained_fixed_create(size_t c, uint8_t k, uint8_t v) { return tpht_make(c, k, v, TPHT_CHAINED, TPHT_SEQUENTIAL, TPHT_FIXED); }
tpht_table_t *tpht_chained_resizable_create(size_t c, uint8_t k, uint8_t v) { return tpht_make(c, k, v, TPHT_CHAINED, TPHT_SEQUENTIAL, TPHT_RESIZABLE); }
tpht_table_t *tpht_chained_concurrent_fixed_create(size_t c, uint8_t k, uint8_t v) { return tpht_make(c, k, v, TPHT_CHAINED, TPHT_CONCURRENT, TPHT_FIXED); }
tpht_table_t *tpht_chained_concurrent_resizable_create(size_t c, uint8_t k, uint8_t v) { return tpht_make(c, k, v, TPHT_CHAINED, TPHT_CONCURRENT, TPHT_RESIZABLE); }
tpht_table_t *tpht_flatten_fixed_create(size_t c, uint8_t k, uint8_t v) { return tpht_make(c, k, v, TPHT_FLATTEN, TPHT_SEQUENTIAL, TPHT_FIXED); }
tpht_table_t *tpht_flatten_resizable_create(size_t c, uint8_t k, uint8_t v) { return tpht_make(c, k, v, TPHT_FLATTEN, TPHT_SEQUENTIAL, TPHT_RESIZABLE); }
tpht_table_t *tpht_flatten_concurrent_fixed_create(size_t c, uint8_t k, uint8_t v) { return tpht_make(c, k, v, TPHT_FLATTEN, TPHT_CONCURRENT, TPHT_FIXED); }
tpht_table_t *tpht_flatten_concurrent_resizable_create(size_t c, uint8_t k, uint8_t v) { return tpht_make(c, k, v, TPHT_FLATTEN, TPHT_CONCURRENT, TPHT_RESIZABLE); }

tpht_status_t tpht_put_u64(tpht_table_t *table, uint64_t key, uint64_t value) {
    uint8_t k[8], v[8];
    if (!table) return TPHT_INVALID;
    tpht_write_le(k, table->key_size, key);
    tpht_write_le(v, table->value_size, value);
    return tpht_put(table, k, v);
}

tpht_status_t tpht_get_u64(tpht_table_t *table, uint64_t key, uint64_t *value_out) {
    uint8_t k[8], v[8];
    tpht_status_t st;
    if (!table || !value_out) return TPHT_INVALID;
    tpht_write_le(k, table->key_size, key);
    st = tpht_get(table, k, v);
    if (st == TPHT_OK) *value_out = tpht_read_le(v, table->value_size);
    return st;
}

tpht_status_t tpht_remove_u64(tpht_table_t *table, uint64_t key) {
    uint8_t k[8];
    if (!table) return TPHT_INVALID;
    tpht_write_le(k, table->key_size, key);
    return tpht_remove(table, k);
}
