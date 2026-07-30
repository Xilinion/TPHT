#include "tpht.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

static void exercise_table(tpht_table_t *t, uint64_t n) {
    uint64_t i;
    uint64_t out;
    assert(t != 0);

    for (i = 0; i < n; ++i) {
        assert(tpht_put_u64(t, i, i * 17u + 3u) == TPHT_OK);
    }
    assert(tpht_size(t) == (size_t)n);

    for (i = 0; i < n; ++i) {
        out = 0;
        assert(tpht_get_u64(t, i, &out) == TPHT_OK);
        assert(out == i * 17u + 3u);
    }

    for (i = 0; i < n; i += 3u) {
        assert(tpht_put_u64(t, i, i * 19u + 5u) == TPHT_OK);
    }
    for (i = 0; i < n; i += 3u) {
        out = 0;
        assert(tpht_get_u64(t, i, &out) == TPHT_OK);
        assert(out == i * 19u + 5u);
    }

    for (i = 0; i < n; i += 2u) {
        tpht_status_t st = tpht_remove_u64(t, i);
        if (st != TPHT_OK) {
            printf("remove failed at key=%llu status=%d size=%zu capacity=%zu variant=%d\n",
                   (unsigned long long)i, (int)st, tpht_size(t), tpht_capacity(t),
                   (int)tpht_get_variant(t));
        }
        assert(st == TPHT_OK);
    }
    for (i = 0; i < n; ++i) {
        tpht_status_t st = tpht_get_u64(t, i, &out);
        if ((i & 1u) == 0) {
            assert(st == TPHT_NOT_FOUND);
        } else {
            assert(st == TPHT_OK);
        }
    }
}

static void exercise_bytes(tpht_variant_t variant, tpht_threading_t threading,
                           tpht_resize_mode_t resize_mode, uint8_t key_size,
                           uint8_t value_size) {
    tpht_config_t c = tpht_default_config();
    tpht_table_t *t;
    uint64_t out;
    c.variant = variant;
    c.threading = threading;
    c.resize_mode = resize_mode;
    c.initial_capacity = resize_mode == TPHT_RESIZABLE ? 8u : 256u;
    c.key_size = key_size;
    c.value_size = value_size;
    t = tpht_create(&c);
    exercise_table(t, resize_mode == TPHT_RESIZABLE ? 200u : 100u);
    assert(tpht_get_u64(t, 7777777u, &out) == TPHT_NOT_FOUND);
    tpht_destroy(t);
}

int main(void) {
    uint8_t sizes[] = {2u, 4u, 8u};
    size_t i, j;

    exercise_bytes(TPHT_CHAINED, TPHT_SEQUENTIAL, TPHT_FIXED, 8u, 8u);
    exercise_bytes(TPHT_CHAINED, TPHT_SEQUENTIAL, TPHT_RESIZABLE, 8u, 8u);
    exercise_bytes(TPHT_CHAINED, TPHT_CONCURRENT, TPHT_FIXED, 8u, 8u);
    exercise_bytes(TPHT_CHAINED, TPHT_CONCURRENT, TPHT_RESIZABLE, 8u, 8u);

    exercise_bytes(TPHT_FLATTEN, TPHT_SEQUENTIAL, TPHT_FIXED, 8u, 8u);
    exercise_bytes(TPHT_FLATTEN, TPHT_SEQUENTIAL, TPHT_RESIZABLE, 8u, 8u);
    exercise_bytes(TPHT_FLATTEN, TPHT_CONCURRENT, TPHT_FIXED, 8u, 8u);
    exercise_bytes(TPHT_FLATTEN, TPHT_CONCURRENT, TPHT_RESIZABLE, 8u, 8u);

    for (i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
        for (j = 0; j < sizeof(sizes) / sizeof(sizes[0]); ++j) {
            exercise_bytes(TPHT_CHAINED, TPHT_SEQUENTIAL, TPHT_RESIZABLE, sizes[i], sizes[j]);
            exercise_bytes(TPHT_FLATTEN, TPHT_SEQUENTIAL, TPHT_RESIZABLE, sizes[i], sizes[j]);
        }
    }

    puts("tpht tests passed");
    return 0;
}
