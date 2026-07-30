#include "tpht_test_common.h"

#include <assert.h>
#include <string.h>

static void run_random_model_case(const tpht_test_case_t *tc) {
    enum { MODEL_CAP = 4096, OPS = 1800 };
    tpht_table_t *t = tpht_test_make_table(tc->variant, tc->threading,
                                           tc->resize_mode, tc->key_size,
                                           tc->value_size,
                                           tc->resize_mode == TPHT_RESIZABLE ? 8u : MODEL_CAP);
    tpht_test_model_entry_t model[MODEL_CAP];
    uint64_t rng = UINT64_C(0xf00d123456789abc) ^ ((uint64_t)tc->variant << 8u) ^
                   ((uint64_t)tc->threading << 16u) ^ ((uint64_t)tc->resize_mode << 24u) ^
                   ((uint64_t)tc->key_size << 32u) ^ ((uint64_t)tc->value_size << 40u);
    int op;

    memset(model, 0, sizeof(model));
    assert(t != NULL);

    for (op = 0; op < OPS; ++op) {
        uint64_t key = tpht_test_trunc_to(tpht_test_next_rand(&rng) % 4096u, tc->key_size);
        uint64_t val = tpht_test_trunc_to(tpht_test_next_rand(&rng), tc->value_size);
        uint8_t kb[8], vb[8], outb[8];
        size_t idx = tpht_test_model_find(model, MODEL_CAP, key);
        int action = (int)(tpht_test_next_rand(&rng) % 5u);
        tpht_test_put_bytes(kb, tc->key_size, key);
        tpht_test_put_bytes(vb, tc->value_size, val);

        if (action <= 1) {
            tpht_status_t st = tpht_put(t, kb, vb);
            assert(st == TPHT_OK || (tc->resize_mode == TPHT_FIXED && st == TPHT_FULL));
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
            if (st == TPHT_OK) assert(tpht_test_get_bytes(outb, tc->value_size) == model[idx].value);
        }
    }

    {
        size_t live = 0, i;
        for (i = 0; i < MODEL_CAP; ++i) {
            if (model[i].live) {
                live++;
                tpht_test_assert_get(t, tc->key_size, tc->value_size, model[i].key, model[i].value);
            }
        }
        assert(tpht_size(t) == live);
    }
    tpht_destroy(t);
}

void tpht_test_run_random_model_module(void) {
    tpht_test_for_each_case(run_random_model_case);
}
