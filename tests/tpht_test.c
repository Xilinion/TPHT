#include "chained_tpht.h"
#include "flatten_tpht.h"
#include "tpht.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef TPHT_TEST_WITH_THREADS
#include <pthread.h>
#endif

typedef struct model_entry {
    uint64_t key;
    uint64_t value;
    int live;
} model_entry_t;

static uint64_t mask_for_size(uint8_t size) {
    return size == 8u ? UINT64_MAX : ((UINT64_C(1) << (8u * size)) - 1u);
}

static uint64_t trunc_to(uint64_t x, uint8_t size) { return x & mask_for_size(size); }

static void put_bytes(uint8_t *dst, uint8_t size, uint64_t x) {
    uint8_t i;
    for (i = 0; i < size; ++i) dst[i] = (uint8_t)(x >> (8u * i));
}

static uint64_t get_bytes(const uint8_t *src, uint8_t size) {
    uint64_t x = 0;
    uint8_t i;
    for (i = 0; i < size; ++i) x |= (uint64_t)src[i] << (8u * i);
    return x;
}

static size_t model_find(model_entry_t *m, size_t n, uint64_t key) {
    size_t i;
    for (i = 0; i < n; ++i) {
        if (m[i].live && m[i].key == key) return i;
    }
    return n;
}

static void assert_get(tpht_table_t *t, uint8_t key_size, uint8_t value_size,
                       uint64_t key, uint64_t expected) {
    uint8_t kb[8], vb[8];
    memset(vb, 0xa5, sizeof(vb));
    put_bytes(kb, key_size, key);
    assert(tpht_get(t, kb, vb) == TPHT_OK);
    assert(get_bytes(vb, value_size) == trunc_to(expected, value_size));
}

static void assert_missing(tpht_table_t *t, uint8_t key_size, uint64_t key) {
    uint8_t kb[8], vb[8];
    put_bytes(kb, key_size, key);
    assert(tpht_get(t, kb, vb) == TPHT_NOT_FOUND);
}

static tpht_table_t *make_table(tpht_variant_t variant, tpht_threading_t threading,
                                tpht_resize_mode_t resize_mode, uint8_t key_size,
                                uint8_t value_size, size_t capacity) {
    tpht_config_t c = tpht_default_config();
    c.variant = variant;
    c.threading = threading;
    c.resize_mode = resize_mode;
    c.initial_capacity = capacity;
    c.key_size = key_size;
    c.value_size = value_size;
    c.max_load_factor = 0.70;
    c.hash_seed = UINT64_C(0x123456789abcdef0);
    return tpht_create(&c);
}

static void exercise_api_edges(tpht_table_t *t, uint8_t key_size, uint8_t value_size,
                               size_t fixed_capacity) {
    uint8_t k[8], v[8], out[8];
    uint64_t i;

    assert(tpht_put(NULL, k, v) == TPHT_INVALID);
    assert(tpht_put(t, NULL, v) == TPHT_INVALID);
    assert(tpht_put(t, k, NULL) == TPHT_INVALID);
    assert(tpht_get(t, NULL, out) == TPHT_INVALID);
    assert(tpht_get(t, k, NULL) == TPHT_INVALID);
    assert(tpht_remove(t, NULL) == TPHT_INVALID);
    assert(tpht_update(t, NULL, v) == TPHT_INVALID);

    put_bytes(k, key_size, 42u);
    put_bytes(v, value_size, 100u);
    assert(tpht_update(t, k, v) == TPHT_NOT_FOUND);
    assert(tpht_remove(t, k) == TPHT_NOT_FOUND);
    assert(tpht_insert(t, k, v) == TPHT_OK);
    assert(tpht_insert(t, k, v) == TPHT_EXISTS);
    put_bytes(v, value_size, 200u);
    assert(tpht_update(t, k, v) == TPHT_OK);
    assert_get(t, key_size, value_size, 42u, 200u);
    assert(tpht_remove(t, k) == TPHT_OK);
    assert_missing(t, key_size, 42u);

    if (tpht_get_resize_mode(t) == TPHT_FIXED) {
        for (i = 0; i < (uint64_t)fixed_capacity; ++i) {
            put_bytes(k, key_size, i + 1000u);
            put_bytes(v, value_size, i + 9000u);
            assert(tpht_insert(t, k, v) == TPHT_OK);
        }
        put_bytes(k, key_size, 999999u);
        put_bytes(v, value_size, 1u);
        assert(tpht_insert(t, k, v) == TPHT_FULL);
        put_bytes(k, key_size, 1000u);
        put_bytes(v, value_size, 77u);
        assert(tpht_put(t, k, v) == TPHT_OK);
        assert_get(t, key_size, value_size, 1000u, 77u);
    }
}

static void exercise_deterministic(tpht_variant_t variant, tpht_threading_t threading,
                                   tpht_resize_mode_t resize_mode, uint8_t key_size,
                                   uint8_t value_size) {
    const size_t initial_capacity = resize_mode == TPHT_RESIZABLE ? 8u : 96u;
    const uint64_t n = resize_mode == TPHT_RESIZABLE ? 260u : 72u;
    tpht_table_t *t = make_table(variant, threading, resize_mode, key_size, value_size,
                                 initial_capacity);
    uint64_t i, out;
    size_t start_capacity;
    assert(t != NULL);
    assert(tpht_get_variant(t) == variant);
    assert(tpht_get_threading(t) == threading);
    assert(tpht_get_resize_mode(t) == resize_mode);
    start_capacity = tpht_capacity(t);

    for (i = 0; i < n; ++i) {
        assert(tpht_put_u64(t, i, i * 17u + 3u) == TPHT_OK);
    }
    assert(tpht_size(t) == (size_t)n);
    if (resize_mode == TPHT_RESIZABLE) assert(tpht_capacity(t) > start_capacity);

    for (i = 0; i < n; ++i) {
        assert(tpht_get_u64(t, i, &out) == TPHT_OK);
        assert(out == trunc_to(i * 17u + 3u, value_size));
    }

    for (i = 0; i < n; i += 5u) {
        uint8_t k[8], v[8];
        put_bytes(k, key_size, i);
        put_bytes(v, value_size, i * 19u + 5u);
        assert(tpht_insert(t, k, v) == TPHT_EXISTS);
        assert(tpht_update(t, k, v) == TPHT_OK);
    }
    for (i = 0; i < n; i += 5u) assert_get(t, key_size, value_size, i, i * 19u + 5u);

    for (i = 0; i < n; i += 2u) assert(tpht_remove_u64(t, i) == TPHT_OK);
    for (i = 0; i < n; ++i) {
        tpht_status_t st = tpht_get_u64(t, i, &out);
        if ((i & 1u) == 0u) {
            assert(st == TPHT_NOT_FOUND);
        } else {
            assert(st == TPHT_OK);
        }
    }

    tpht_destroy(t);
}

static uint64_t next_rand(uint64_t *state) {
    *state = *state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    return *state;
}

static void exercise_random_model(tpht_variant_t variant, tpht_threading_t threading,
                                  tpht_resize_mode_t resize_mode, uint8_t key_size,
                                  uint8_t value_size) {
    enum { MODEL_CAP = 4096, OPS = 1800 };
    tpht_table_t *t = make_table(variant, threading, resize_mode, key_size, value_size,
                                 resize_mode == TPHT_RESIZABLE ? 8u : MODEL_CAP);
    model_entry_t model[MODEL_CAP];
    uint64_t rng = UINT64_C(0xf00d123456789abc) ^ ((uint64_t)variant << 8u) ^
                   ((uint64_t)threading << 16u) ^ ((uint64_t)resize_mode << 24u) ^
                   ((uint64_t)key_size << 32u) ^ ((uint64_t)value_size << 40u);
    int op;
    memset(model, 0, sizeof(model));
    assert(t != NULL);

    for (op = 0; op < OPS; ++op) {
        uint64_t key = trunc_to(next_rand(&rng) % 4096u, key_size);
        uint64_t val = trunc_to(next_rand(&rng), value_size);
        uint8_t kb[8], vb[8], outb[8];
        size_t idx = model_find(model, MODEL_CAP, key);
        int action = (int)(next_rand(&rng) % 5u);
        put_bytes(kb, key_size, key);
        put_bytes(vb, value_size, val);

        if (action <= 1) {
            tpht_status_t st = tpht_put(t, kb, vb);
            assert(st == TPHT_OK || (resize_mode == TPHT_FIXED && st == TPHT_FULL));
            if (st == TPHT_OK) {
                if (idx == MODEL_CAP) {
                    size_t i;
                    for (i = 0; i < MODEL_CAP; ++i) {
                        if (!model[i].live) {
                            idx = i;
                            break;
                        }
                    }
                }
                assert(idx != MODEL_CAP);
                model[idx].key = key;
                model[idx].value = val;
                model[idx].live = 1;
            }
        } else if (action == 2) {
            tpht_status_t st = tpht_update(t, kb, vb);
            assert(st == (idx == MODEL_CAP ? TPHT_NOT_FOUND : TPHT_OK));
            if (st == TPHT_OK) model[idx].value = val;
        } else if (action == 3) {
            tpht_status_t st = tpht_remove(t, kb);
            assert(st == (idx == MODEL_CAP ? TPHT_NOT_FOUND : TPHT_OK));
            if (st == TPHT_OK) model[idx].live = 0;
        } else {
            tpht_status_t st = tpht_get(t, kb, outb);
            assert(st == (idx == MODEL_CAP ? TPHT_NOT_FOUND : TPHT_OK));
            if (st == TPHT_OK) assert(get_bytes(outb, value_size) == model[idx].value);
        }
    }

    {
        size_t live = 0, i;
        for (i = 0; i < MODEL_CAP; ++i) {
            if (model[i].live) {
                live++;
                assert_get(t, key_size, value_size, model[i].key, model[i].value);
            }
        }
        assert(tpht_size(t) == live);
    }
    tpht_destroy(t);
}

static void exercise_config_validation(void) {
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

static void exercise_explicit_constructors(void) {
    tpht_table_t *tables[8];
    chained_tpht_t *chained_wrapper;
    flatten_tpht_t *flatten_wrapper;
    size_t i;
    tables[0] = tpht_chained_fixed_create(32, 8, 8);
    tables[1] = tpht_chained_resizable_create(8, 8, 8);
    tables[2] = tpht_chained_concurrent_fixed_create(32, 8, 8);
    tables[3] = tpht_chained_concurrent_resizable_create(8, 8, 8);
    tables[4] = tpht_flatten_fixed_create(32, 8, 8);
    tables[5] = tpht_flatten_resizable_create(8, 8, 8);
    tables[6] = tpht_flatten_concurrent_fixed_create(32, 8, 8);
    tables[7] = tpht_flatten_concurrent_resizable_create(8, 8, 8);
    for (i = 0; i < 8u; ++i) {
        assert(tables[i] != NULL);
        assert(tpht_put_u64(tables[i], 7, 11) == TPHT_OK);
        assert_get(tables[i], 8, 8, 7, 11);
        tpht_destroy(tables[i]);
    }

    chained_wrapper = chained_tpht_resizable_create(8, 8, 8);
    assert(chained_wrapper != NULL);
    assert(chained_tpht_put(chained_wrapper, &(uint64_t){1}, &(uint64_t){2}) == TPHT_OK);
    assert(chained_tpht_size(chained_wrapper) == 1u);
    chained_tpht_destroy(chained_wrapper);

    flatten_wrapper = flatten_tpht_resizable_create(8, 8, 8);
    assert(flatten_wrapper != NULL);
    assert(flatten_tpht_put(flatten_wrapper, &(uint64_t){1}, &(uint64_t){2}) == TPHT_OK);
    assert(flatten_tpht_size(flatten_wrapper) == 1u);
    flatten_tpht_destroy(flatten_wrapper);
}

#ifdef TPHT_TEST_WITH_THREADS
typedef struct thread_arg {
    tpht_table_t *table;
    uint64_t base;
    uint64_t count;
} thread_arg_t;

static void *thread_insert(void *arg) {
    thread_arg_t *a = (thread_arg_t *)arg;
    uint64_t i;
    for (i = 0; i < a->count; ++i) {
        uint64_t key = a->base + i;
        assert(tpht_put_u64(a->table, key, key * 13u + 1u) == TPHT_OK);
    }
    return NULL;
}

static void exercise_real_threads(tpht_variant_t variant) {
    enum { THREADS = 4, PER_THREAD = 500 };
    pthread_t threads[THREADS];
    thread_arg_t args[THREADS];
    tpht_table_t *t = make_table(variant, TPHT_CONCURRENT, TPHT_RESIZABLE, 8, 8, 16);
    int i;
    uint64_t j, out;
    assert(t != NULL);
    for (i = 0; i < THREADS; ++i) {
        args[i].table = t;
        args[i].base = (uint64_t)i * 100000u;
        args[i].count = PER_THREAD;
        assert(pthread_create(&threads[i], NULL, thread_insert, &args[i]) == 0);
    }
    for (i = 0; i < THREADS; ++i) assert(pthread_join(threads[i], NULL) == 0);
    assert(tpht_size(t) == (size_t)THREADS * (size_t)PER_THREAD);
    for (i = 0; i < THREADS; ++i) {
        for (j = 0; j < PER_THREAD; ++j) {
            uint64_t key = args[i].base + j;
            assert(tpht_get_u64(t, key, &out) == TPHT_OK);
            assert(out == key * 13u + 1u);
        }
    }
    tpht_destroy(t);
}
#endif

int main(void) {
    uint8_t sizes[] = {2u, 4u, 8u};
    tpht_variant_t variants[] = {TPHT_CHAINED, TPHT_FLATTEN};
    tpht_threading_t threadings[] = {TPHT_SEQUENTIAL, TPHT_CONCURRENT};
    tpht_resize_mode_t resize_modes[] = {TPHT_FIXED, TPHT_RESIZABLE};
    size_t vi, ti, ri, ki, vali;

    exercise_config_validation();
    exercise_explicit_constructors();

    for (vi = 0; vi < 2u; ++vi) {
        for (ti = 0; ti < 2u; ++ti) {
            for (ri = 0; ri < 2u; ++ri) {
                for (ki = 0; ki < 3u; ++ki) {
                    for (vali = 0; vali < 3u; ++vali) {
                        tpht_table_t *edge = make_table(variants[vi], threadings[ti], resize_modes[ri],
                                                        sizes[ki], sizes[vali], 16);
                        assert(edge != NULL);
                        exercise_api_edges(edge, sizes[ki], sizes[vali], 16);
                        tpht_destroy(edge);
                        exercise_deterministic(variants[vi], threadings[ti], resize_modes[ri],
                                               sizes[ki], sizes[vali]);
                        exercise_random_model(variants[vi], threadings[ti], resize_modes[ri],
                                              sizes[ki], sizes[vali]);
                    }
                }
            }
        }
    }

#ifdef TPHT_TEST_WITH_THREADS
    exercise_real_threads(TPHT_CHAINED);
    exercise_real_threads(TPHT_FLATTEN);
#endif

    puts("tpht exhaustive tests passed");
    return 0;
}
