#include "tpht_test_common.h"

#include <assert.h>

#ifdef TPHT_TEST_WITH_THREADS
#include <pthread.h>

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

static void exercise_real_threads(tpht_variant_t variant, tpht_resize_mode_t resize_mode,
                                  size_t capacity) {
    enum { THREADS = 4, PER_THREAD = 500 };
    pthread_t threads[THREADS];
    thread_arg_t args[THREADS];
    tpht_table_t *t = tpht_test_make_table(variant, TPHT_CONCURRENT,
                                           resize_mode, 8, 8, capacity);
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

static void exercise_chained_custom_resize_strides(void) {
    enum { THREADS = 4, PER_THREAD = 500 };
    pthread_t threads[THREADS];
    thread_arg_t args[THREADS];
    tpht_config_t c = tpht_default_config();
    tpht_table_t *t;
    int i;

    c.variant = TPHT_CHAINED;
    c.threading = TPHT_CONCURRENT;
    c.resize_mode = TPHT_RESIZABLE;
    c.initial_capacity = 16;
    c.key_size = 8;
    c.value_size = 8;
    c.max_load_factor = 0.70;
    c.resize_strides = 17;
    t = tpht_create(&c);
    assert(t != NULL);

    for (i = 0; i < THREADS; ++i) {
        args[i].table = t;
        args[i].base = (uint64_t)i * 100000u;
        args[i].count = PER_THREAD;
        assert(pthread_create(&threads[i], NULL, thread_insert, &args[i]) == 0);
    }
    for (i = 0; i < THREADS; ++i) assert(pthread_join(threads[i], NULL) == 0);
    assert(tpht_size(t) == (size_t)THREADS * (size_t)PER_THREAD);
    tpht_destroy(t);
}
#endif

void tpht_test_run_thread_module(void) {
#ifdef TPHT_TEST_WITH_THREADS
    exercise_real_threads(TPHT_CHAINED, TPHT_FIXED, 4096);
    exercise_real_threads(TPHT_CHAINED, TPHT_RESIZABLE, 16);
    exercise_chained_custom_resize_strides();
    exercise_real_threads(TPHT_FLATTEN, TPHT_RESIZABLE, 16);
#endif
}
