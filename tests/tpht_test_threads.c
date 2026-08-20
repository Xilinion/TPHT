#include "tpht_test_common.h"

#include <assert.h>

#ifdef TPHT_TEST_WITH_THREADS
#include <pthread.h>

typedef struct thread_arg {
    tpht_test_table_t *table;
    uint64_t base;
    uint64_t count;
} thread_arg_t;

static void *thread_insert(void *arg) {
    thread_arg_t *a = (thread_arg_t *)arg;
    uint64_t i;
    for (i = 0; i < a->count; ++i) {
        uint64_t key = a->base + i;
        assert(a->table->put(a->table->handle, key, key * 13u + 1u) == TPHT_OK);
    }
    return NULL;
}

static void exercise_real_threads(tpht_test_kind_t kind, tpht_resize_mode_t resize_mode,
                                  size_t capacity, uint8_t value_size, size_t resize_strides) {
    enum { THREADS = 4, PER_THREAD = 500 };
    pthread_t threads[THREADS];
    thread_arg_t args[THREADS];
    tpht_options_t o = tpht_default_options();
    tpht_test_table_t t;
    int i;
    uint64_t j, out;

    o.max_load_factor = 0.70;
    o.resize_strides = resize_strides;
    t = tpht_test_make_kind(kind, 1, resize_mode, capacity, value_size, &o);
    assert(t.handle != NULL);

    for (i = 0; i < THREADS; ++i) {
        args[i].table = &t;
        args[i].base = (uint64_t)i * 100000u;
        args[i].count = PER_THREAD;
        assert(pthread_create(&threads[i], NULL, thread_insert, &args[i]) == 0);
    }
    for (i = 0; i < THREADS; ++i) assert(pthread_join(threads[i], NULL) == 0);
    assert(t.size(t.handle) == (size_t)THREADS * (size_t)PER_THREAD);
    for (i = 0; i < THREADS; ++i) {
        for (j = 0; j < PER_THREAD; ++j) {
            uint64_t key = args[i].base + j;
            assert(t.get(t.handle, key, &out) == TPHT_OK);
            assert(out == tpht_test_trunc_to(key * 13u + 1u, value_size));
        }
    }
    t.destroy(t.handle);
}
#endif

#ifdef TPHT_TEST_WITH_THREADS
/*
 * Insert, remove, update and verify concurrently while the table grows from a
 * tiny capacity, so the resize machinery runs underneath every operation
 * kind.  Threads own disjoint key ranges; each verifies its own range, so the
 * expected state is exact despite the shared storage.
 */
typedef struct churn_arg {
    tpht_test_table_t *table;
    uint8_t value_size;
    uint64_t base;
    uint64_t count;
} churn_arg_t;

static void *thread_churn(void *arg) {
    churn_arg_t *a = (churn_arg_t *)arg;
    uint64_t i, out;
    int round;
    for (round = 0; round < 2; ++round) {
        for (i = 0; i < a->count; ++i)
            assert(a->table->put(a->table->handle, a->base + i, i * 7u + 1u) == TPHT_OK);
        for (i = 1; i < a->count; i += 2u)
            assert(a->table->remove(a->table->handle, a->base + i) == TPHT_OK);
        for (i = 0; i < a->count; i += 2u)
            assert(a->table->update(a->table->handle, a->base + i, i * 9u + 2u) == TPHT_OK);
        for (i = 0; i < a->count; ++i) {
            tpht_status_t st = a->table->get(a->table->handle, a->base + i, &out);
            if ((i & 1u) == 0u) {
                assert(st == TPHT_OK);
                assert(out == tpht_test_trunc_to(i * 9u + 2u, a->value_size));
            } else {
                assert(st == TPHT_NOT_FOUND);
            }
        }
        for (i = 1; i < a->count; i += 2u)
            assert(a->table->put(a->table->handle, a->base + i, i * 7u + 1u) == TPHT_OK);
    }
    return NULL;
}

static void exercise_threaded_churn(tpht_test_kind_t kind, size_t capacity,
                                    uint8_t value_size) {
    enum { THREADS = 4, PER_THREAD = 600 };
    pthread_t threads[THREADS];
    churn_arg_t args[THREADS];
    tpht_options_t o = tpht_default_options();
    tpht_test_table_t t;
    int i;

    o.max_load_factor = 0.70;
    t = tpht_test_make_kind(kind, 1, TPHT_RESIZABLE, capacity, value_size, &o);
    assert(t.handle != NULL);
    for (i = 0; i < THREADS; ++i) {
        args[i].table = &t;
        args[i].value_size = value_size;
        args[i].base = (uint64_t)i * 100000u + 1u;
        args[i].count = PER_THREAD;
        assert(pthread_create(&threads[i], NULL, thread_churn, &args[i]) == 0);
    }
    for (i = 0; i < THREADS; ++i) assert(pthread_join(threads[i], NULL) == 0);
    assert(t.size(t.handle) == (size_t)THREADS * (size_t)PER_THREAD);
    t.destroy(t.handle);
}
#endif

void tpht_test_run_thread_module(void) {
#ifdef TPHT_TEST_WITH_THREADS
    exercise_real_threads(TPHT_TEST_CHAIN64, TPHT_FIXED, 4096, 8, 0);
    exercise_real_threads(TPHT_TEST_CHAIN32, TPHT_FIXED, 4096, 3, 0);
    exercise_real_threads(TPHT_TEST_CHAIN64, TPHT_RESIZABLE, 16, 8, 0);
    exercise_real_threads(TPHT_TEST_CHAIN32, TPHT_RESIZABLE, 16, 1, 0);
    exercise_real_threads(TPHT_TEST_CHAIN64, TPHT_RESIZABLE, 16, 8, 17);
    /* The flattened concurrent tables, through the same drills. */
    exercise_real_threads(TPHT_TEST_FLAT64, TPHT_FIXED, 4096, 8, 0);
    exercise_real_threads(TPHT_TEST_FLAT32, TPHT_FIXED, 4096, 3, 0);
    exercise_real_threads(TPHT_TEST_FLAT64, TPHT_RESIZABLE, 16, 8, 0);
    exercise_real_threads(TPHT_TEST_FLAT32, TPHT_RESIZABLE, 16, 1, 0);
    /* Every operation kind racing a growing table. */
    exercise_threaded_churn(TPHT_TEST_FLAT64, 16, 8);
    exercise_threaded_churn(TPHT_TEST_FLAT32, 16, 3);
    exercise_threaded_churn(TPHT_TEST_CHAIN64, 16, 8);
#endif
}
