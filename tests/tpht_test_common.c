#include "tpht_test_common.h"

#include <assert.h>
#include <string.h>

/* One adapter set per concrete table type. */

static tpht_status_t flatten_tpht32_a_put(void *h, uint64_t k, uint64_t v) { return flatten_tpht32_put((flatten_tpht32_t *)h, (uint32_t)k, v); }
static tpht_status_t flatten_tpht32_a_insert(void *h, uint64_t k, uint64_t v) { return flatten_tpht32_insert((flatten_tpht32_t *)h, (uint32_t)k, v); }
static tpht_status_t flatten_tpht32_a_update(void *h, uint64_t k, uint64_t v) { return flatten_tpht32_update((flatten_tpht32_t *)h, (uint32_t)k, v); }
static tpht_status_t flatten_tpht32_a_get(void *h, uint64_t k, uint64_t *v) { return flatten_tpht32_get((flatten_tpht32_t *)h, (uint32_t)k, v); }
static tpht_status_t flatten_tpht32_a_remove(void *h, uint64_t k) { return flatten_tpht32_remove((flatten_tpht32_t *)h, (uint32_t)k); }
static size_t flatten_tpht32_a_size(const void *h) { return flatten_tpht32_size((const flatten_tpht32_t *)h); }
static size_t flatten_tpht32_a_capacity(const void *h) { return flatten_tpht32_capacity((const flatten_tpht32_t *)h); }
static size_t flatten_tpht32_a_memory(const void *h) { return flatten_tpht32_memory_bytes((const flatten_tpht32_t *)h); }
static void flatten_tpht32_a_destroy(void *h) { flatten_tpht32_destroy((flatten_tpht32_t *)h); }

static tpht_status_t flatten_tpht64_a_put(void *h, uint64_t k, uint64_t v) { return flatten_tpht64_put((flatten_tpht64_t *)h, (uint64_t)k, v); }
static tpht_status_t flatten_tpht64_a_insert(void *h, uint64_t k, uint64_t v) { return flatten_tpht64_insert((flatten_tpht64_t *)h, (uint64_t)k, v); }
static tpht_status_t flatten_tpht64_a_update(void *h, uint64_t k, uint64_t v) { return flatten_tpht64_update((flatten_tpht64_t *)h, (uint64_t)k, v); }
static tpht_status_t flatten_tpht64_a_get(void *h, uint64_t k, uint64_t *v) { return flatten_tpht64_get((flatten_tpht64_t *)h, (uint64_t)k, v); }
static tpht_status_t flatten_tpht64_a_remove(void *h, uint64_t k) { return flatten_tpht64_remove((flatten_tpht64_t *)h, (uint64_t)k); }
static size_t flatten_tpht64_a_size(const void *h) { return flatten_tpht64_size((const flatten_tpht64_t *)h); }
static size_t flatten_tpht64_a_capacity(const void *h) { return flatten_tpht64_capacity((const flatten_tpht64_t *)h); }
static size_t flatten_tpht64_a_memory(const void *h) { return flatten_tpht64_memory_bytes((const flatten_tpht64_t *)h); }
static void flatten_tpht64_a_destroy(void *h) { flatten_tpht64_destroy((flatten_tpht64_t *)h); }

static tpht_status_t chained_tpht32_a_put(void *h, uint64_t k, uint64_t v) { return chained_tpht32_put((chained_tpht32_t *)h, (uint32_t)k, v); }
static tpht_status_t chained_tpht32_a_insert(void *h, uint64_t k, uint64_t v) { return chained_tpht32_insert((chained_tpht32_t *)h, (uint32_t)k, v); }
static tpht_status_t chained_tpht32_a_update(void *h, uint64_t k, uint64_t v) { return chained_tpht32_update((chained_tpht32_t *)h, (uint32_t)k, v); }
static tpht_status_t chained_tpht32_a_get(void *h, uint64_t k, uint64_t *v) { return chained_tpht32_get((chained_tpht32_t *)h, (uint32_t)k, v); }
static tpht_status_t chained_tpht32_a_remove(void *h, uint64_t k) { return chained_tpht32_remove((chained_tpht32_t *)h, (uint32_t)k); }
static size_t chained_tpht32_a_size(const void *h) { return chained_tpht32_size((const chained_tpht32_t *)h); }
static size_t chained_tpht32_a_capacity(const void *h) { return chained_tpht32_capacity((const chained_tpht32_t *)h); }
static size_t chained_tpht32_a_memory(const void *h) { return chained_tpht32_memory_bytes((const chained_tpht32_t *)h); }
static void chained_tpht32_a_destroy(void *h) { chained_tpht32_destroy((chained_tpht32_t *)h); }

static tpht_status_t chained_tpht64_a_put(void *h, uint64_t k, uint64_t v) { return chained_tpht64_put((chained_tpht64_t *)h, (uint64_t)k, v); }
static tpht_status_t chained_tpht64_a_insert(void *h, uint64_t k, uint64_t v) { return chained_tpht64_insert((chained_tpht64_t *)h, (uint64_t)k, v); }
static tpht_status_t chained_tpht64_a_update(void *h, uint64_t k, uint64_t v) { return chained_tpht64_update((chained_tpht64_t *)h, (uint64_t)k, v); }
static tpht_status_t chained_tpht64_a_get(void *h, uint64_t k, uint64_t *v) { return chained_tpht64_get((chained_tpht64_t *)h, (uint64_t)k, v); }
static tpht_status_t chained_tpht64_a_remove(void *h, uint64_t k) { return chained_tpht64_remove((chained_tpht64_t *)h, (uint64_t)k); }
static size_t chained_tpht64_a_size(const void *h) { return chained_tpht64_size((const chained_tpht64_t *)h); }
static size_t chained_tpht64_a_capacity(const void *h) { return chained_tpht64_capacity((const chained_tpht64_t *)h); }
static size_t chained_tpht64_a_memory(const void *h) { return chained_tpht64_memory_bytes((const chained_tpht64_t *)h); }
static void chained_tpht64_a_destroy(void *h) { chained_tpht64_destroy((chained_tpht64_t *)h); }

static void bind(tpht_test_table_t *t, tpht_test_kind_t kind) {
    switch (kind) {
        case TPHT_TEST_FLAT32:
            t->put = flatten_tpht32_a_put; t->insert = flatten_tpht32_a_insert; t->update = flatten_tpht32_a_update;
            t->get = flatten_tpht32_a_get; t->remove = flatten_tpht32_a_remove; t->size = flatten_tpht32_a_size;
            t->capacity = flatten_tpht32_a_capacity; t->memory_bytes = flatten_tpht32_a_memory;
            t->destroy = flatten_tpht32_a_destroy;
            return;
        case TPHT_TEST_FLAT64:
            t->put = flatten_tpht64_a_put; t->insert = flatten_tpht64_a_insert; t->update = flatten_tpht64_a_update;
            t->get = flatten_tpht64_a_get; t->remove = flatten_tpht64_a_remove; t->size = flatten_tpht64_a_size;
            t->capacity = flatten_tpht64_a_capacity; t->memory_bytes = flatten_tpht64_a_memory;
            t->destroy = flatten_tpht64_a_destroy;
            return;
        case TPHT_TEST_CHAIN32:
            t->put = chained_tpht32_a_put; t->insert = chained_tpht32_a_insert; t->update = chained_tpht32_a_update;
            t->get = chained_tpht32_a_get; t->remove = chained_tpht32_a_remove; t->size = chained_tpht32_a_size;
            t->capacity = chained_tpht32_a_capacity; t->memory_bytes = chained_tpht32_a_memory;
            t->destroy = chained_tpht32_a_destroy;
            return;
        case TPHT_TEST_CHAIN64:
            t->put = chained_tpht64_a_put; t->insert = chained_tpht64_a_insert; t->update = chained_tpht64_a_update;
            t->get = chained_tpht64_a_get; t->remove = chained_tpht64_a_remove; t->size = chained_tpht64_a_size;
            t->capacity = chained_tpht64_a_capacity; t->memory_bytes = chained_tpht64_a_memory;
            t->destroy = chained_tpht64_a_destroy;
            return;
        default: return;
    }
}

uint8_t tpht_test_key_size(tpht_test_kind_t kind) {
    return (kind == TPHT_TEST_FLAT32 || kind == TPHT_TEST_CHAIN32) ? 4u : 8u;
}

const char *tpht_test_kind_name(tpht_test_kind_t kind) {
    switch (kind) {
        case TPHT_TEST_FLAT32: return "flatten_tpht32";
        case TPHT_TEST_FLAT64: return "flatten_tpht64";
        case TPHT_TEST_CHAIN32: return "chained_tpht32";
        default: return "chained_tpht64";
    }
}

static int is_flatten(tpht_test_kind_t kind) {
    return kind == TPHT_TEST_FLAT32 || kind == TPHT_TEST_FLAT64;
}

uint64_t tpht_test_mask_for_size(uint8_t size) {
    return size >= 8u ? UINT64_MAX : ((UINT64_C(1) << (8u * size)) - 1u);
}

uint64_t tpht_test_trunc_to(uint64_t x, uint8_t size) {
    return x & tpht_test_mask_for_size(size);
}

uint64_t tpht_test_next_rand(uint64_t *state) {
    *state = *state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    return *state;
}

size_t tpht_test_model_find(tpht_test_model_entry_t *model, size_t n, uint64_t key) {
    size_t i;
    for (i = 0; i < n; ++i) {
        if (model[i].live && model[i].key == key) return i;
    }
    return n;
}

int tpht_test_case_supported(const tpht_test_case_t *tc) {
    if (tc->value_size == 0u || tc->value_size > 8u) return 0;
    /* The flattened variant is sequential only. */
    return !(is_flatten(tc->kind) && tc->concurrent);
}

tpht_test_table_t tpht_test_make_kind(tpht_test_kind_t kind, int concurrent,
                                      tpht_resize_mode_t mode, size_t capacity,
                                      uint8_t value_size, const tpht_options_t *options) {
    tpht_test_table_t t;
    tpht_options_t o = options ? *options : tpht_default_options();
    memset(&t, 0, sizeof(t));
    o.resize_mode = mode;
    o.value_size = value_size;
    bind(&t, kind);
    switch (kind) {
        case TPHT_TEST_FLAT32: t.handle = flatten_tpht32_create(capacity, &o); break;
        case TPHT_TEST_FLAT64: t.handle = flatten_tpht64_create(capacity, &o); break;
        case TPHT_TEST_CHAIN32: t.handle = chained_tpht32_create(capacity, concurrent, &o); break;
        default: t.handle = chained_tpht64_create(capacity, concurrent, &o); break;
    }
    return t;
}

tpht_test_table_t tpht_test_make_table(const tpht_test_case_t *tc, size_t capacity) {
    tpht_options_t o = tpht_default_options();
    o.max_load_factor = 0.70;
    o.hash_seed = UINT64_C(0x123456789abcdef0);
    return tpht_test_make_kind(tc->kind, tc->concurrent, tc->resize_mode, capacity,
                               tc->value_size, &o);
}

void tpht_test_assert_get(const tpht_test_table_t *t, const tpht_test_case_t *tc, uint64_t key,
                          uint64_t expected) {
    uint64_t value = UINT64_C(0xa5a5a5a5a5a5a5a5);
    assert(t->get(t->handle, key, &value) == TPHT_OK);
    assert(value == tpht_test_trunc_to(expected, tc->value_size));
}

void tpht_test_assert_missing(const tpht_test_table_t *t, uint64_t key) {
    uint64_t value = 0;
    assert(t->get(t->handle, key, &value) == TPHT_NOT_FOUND);
}

void tpht_test_for_each_case(void (*fn)(const tpht_test_case_t *tc)) {
    tpht_resize_mode_t modes[] = {TPHT_FIXED, TPHT_RESIZABLE};
    int ki, ci, mi;
    uint8_t vs;
    for (ki = 0; ki < (int)TPHT_TEST_KIND_COUNT; ++ki) {
        for (ci = 0; ci < 2; ++ci) {
            for (mi = 0; mi < 2; ++mi) {
                for (vs = 1u; vs <= 8u; ++vs) {
                    tpht_test_case_t tc;
                    tc.kind = (tpht_test_kind_t)ki;
                    tc.concurrent = ci;
                    tc.resize_mode = modes[mi];
                    tc.value_size = vs;
                    if (!tpht_test_case_supported(&tc)) continue;
                    fn(&tc);
                }
            }
        }
    }
}
