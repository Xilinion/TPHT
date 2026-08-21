#include "tpht_test_common.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

/*
 * Long-run churn, drain cycles, size swings and overflow-budget checks.
 *
 * The random-model module covers short mixed workloads on a few thousand
 * keys; this module covers the lifetimes that break tables in production:
 * a large steady-state population whose membership slowly turns over
 * (insert n, then round after round delete k and insert k different keys),
 * repeated full drains and refills of the same table, populations that grow
 * and shrink by orders of magnitude, and - separately - the statistical
 * claim behind the dereference-pool sizing: that a fixed table never needs
 * its hard-overflow absorb machinery below its design load.
 *
 * Every scenario runs over the full case matrix (four table kinds, both
 * threadings, both resize modes, value sizes 1..8), so value-width and
 * 32-bit-key masking edges ride along everywhere.
 */

/* ------------------------------------------------------------------ model
 * Open-addressing reference model with tombstones.  The linear-scan model in
 * tpht_test_common is O(n) per lookup, unusable at churn sizes; this one
 * keeps the same "the test must not trust the table" property at O(1).
 */
typedef struct {
    uint64_t *keys;
    uint64_t *vals;
    uint8_t *state; /* 0 empty, 1 live, 2 tombstone */
    size_t mask;    /* capacity - 1, capacity a power of two */
    size_t live;
    size_t used; /* live + tombstones; rehash before it saturates */
} churn_model_t;

static void model_init(churn_model_t *m, size_t min_cap) {
    size_t cap = 16;
    while (cap < min_cap * 2u) cap <<= 1;
    m->keys = (uint64_t *)malloc(cap * sizeof(uint64_t));
    m->vals = (uint64_t *)malloc(cap * sizeof(uint64_t));
    m->state = (uint8_t *)calloc(cap, 1);
    m->mask = cap - 1u;
    m->live = 0;
    m->used = 0;
    assert(m->keys && m->vals && m->state);
}

static void model_free(churn_model_t *m) {
    free(m->keys);
    free(m->vals);
    free(m->state);
}

static size_t model_slot(const churn_model_t *m, uint64_t key, int for_insert) {
    size_t i = (size_t)(key * UINT64_C(0x9e3779b97f4a7c15)) & m->mask;
    size_t first_tomb = (size_t)-1;
    for (;;) {
        if (m->state[i] == 0)
            return for_insert && first_tomb != (size_t)-1 ? first_tomb : i;
        if (m->state[i] == 2) {
            if (first_tomb == (size_t)-1) first_tomb = i;
        } else if (m->keys[i] == key) {
            return i;
        }
        i = (i + 1u) & m->mask;
    }
}

/* Tombstones accumulate forever under churn; without this the array
 * eventually holds no empty slot and a miss probe never terminates. */
static void model_rehash(churn_model_t *m) {
    churn_model_t n;
    size_t i;
    model_init(&n, m->live + 8u);
    for (i = 0; i <= m->mask; ++i) {
        if (m->state[i] != 1) continue;
        {
            size_t j = model_slot(&n, m->keys[i], 1);
            n.keys[j] = m->keys[i];
            n.vals[j] = m->vals[i];
            n.state[j] = 1;
            n.live++;
            n.used++;
        }
    }
    free(m->keys);
    free(m->vals);
    free(m->state);
    *m = n;
}

static void model_put(churn_model_t *m, uint64_t key, uint64_t val) {
    size_t i;
    if (m->used * 10u >= (m->mask + 1u) * 7u) model_rehash(m);
    i = model_slot(m, key, 1);
    if (m->state[i] != 1) {
        m->live++;
        if (m->state[i] == 0) m->used++;
    }
    m->keys[i] = key;
    m->vals[i] = val;
    m->state[i] = 1;
}

static int model_remove(churn_model_t *m, uint64_t key) {
    size_t i = model_slot(m, key, 0);
    if (m->state[i] != 1) return 0;
    m->state[i] = 2;
    m->live--;
    return 1;
}

/* ------------------------------------------------------------- key streams
 * Distinct keys forever: an odd-multiplier permutation of the counter for
 * 32-bit keys (a bijection on 2^32, so no accidental collisions), a mix
 * bijection for 64-bit.
 */
static uint64_t churn_key(uint64_t counter, uint8_t key_size) {
    if (key_size == 4u) return (counter * UINT64_C(2654435761)) & UINT32_MAX;
    counter ^= counter >> 33;
    counter *= UINT64_C(0xff51afd7ed558ccd);
    counter ^= counter >> 33;
    counter *= UINT64_C(0xc4ceb9fe1a85ec53);
    counter ^= counter >> 33;
    return counter;
}

/* Every live key present with its exact value, a sample of dead keys absent,
 * and the size counter exact. */
static void churn_verify_full(const tpht_test_table_t *t, const churn_model_t *m,
                              uint64_t next_key, uint8_t key_size) {
    size_t i;
    uint64_t k, out;
    for (i = 0; i <= m->mask; ++i) {
        if (m->state[i] != 1) continue;
        assert(t->get(t->handle, m->keys[i], &out) == TPHT_OK);
        assert(out == m->vals[i]);
    }
    for (k = 0; k < 64u; ++k) {
        uint64_t dead = churn_key(next_key + 1000000u + k, key_size);
        assert(t->get(t->handle, dead, &out) == TPHT_NOT_FOUND);
    }
    assert(t->size(t->handle) == m->live);
}

/* -------------------------------------------------------- steady churn
 * Fill to n, then rounds of: delete k members, insert k keys never seen
 * before.  The population size holds while its membership turns over -
 * exactly the "insert n, delete n/1000, insert n/1000 different, repeat"
 * lifetime.  Runs big on a subset of value sizes for time, and small (a
 * 64-slot table) for all of them, where per-block bounds bite hardest.
 */
static void churn_steady(const tpht_test_case_t *tc, uint64_t n, uint64_t k,
                         int rounds, size_t fixed_cap) {
    uint8_t key_size = tpht_test_key_size(tc->kind);
    tpht_test_table_t t = tpht_test_make_table(
        tc, tc->resize_mode == TPHT_RESIZABLE ? 8u : fixed_cap);
    churn_model_t m;
    uint64_t next_key = 0, victim_cursor = 0, rng = 0x1234u ^ n ^ ((uint64_t)tc->kind << 8);
    int r;
    uint64_t i;

    assert(t.handle != NULL);
    model_init(&m, (size_t)n + (size_t)k);

    for (i = 0; i < n; ++i) {
        uint64_t key = churn_key(next_key++, key_size);
        uint64_t val = tpht_test_trunc_to(tpht_test_next_rand(&rng), tc->value_size);
        assert(t.put(t.handle, key, val) == TPHT_OK);
        model_put(&m, key, val);
    }
    churn_verify_full(&t, &m, next_key, key_size);

    for (r = 0; r < rounds; ++r) {
        /* Delete k of the oldest surviving keys: age-ordered victims keep the
         * turnover marching through every home block over time. */
        uint64_t deleted = 0;
        while (deleted < k) {
            uint64_t key = churn_key(victim_cursor++, key_size);
            assert(victim_cursor <= next_key);
            if (!model_remove(&m, key)) continue;
            assert(t.remove(t.handle, key) == TPHT_OK);
            ++deleted;
        }
        for (i = 0; i < k; ++i) {
            uint64_t key = churn_key(next_key++, key_size);
            uint64_t val = tpht_test_trunc_to(tpht_test_next_rand(&rng), tc->value_size);
            assert(t.put(t.handle, key, val) == TPHT_OK);
            model_put(&m, key, val);
        }
        assert(t.size(t.handle) == m.live);
        if ((r & 15) == 15) churn_verify_full(&t, &m, next_key, key_size);
    }
    churn_verify_full(&t, &m, next_key, key_size);
    model_free(&m);
    t.destroy(t.handle);
}

static void churn_steady_case(const tpht_test_case_t *tc) {
    /* Small table, every value size: 48 keys churning inside a 64-capacity
     * table, long enough for the whole population to turn over many times. */
    churn_steady(tc, 48u, 6u, 64, 64u);
    /* Large table, edge value sizes only (1 byte, an odd width, the full
     * word) to keep the suite fast; the small run covers the rest. */
    if (tc->value_size == 1u || tc->value_size == 5u || tc->value_size == 8u)
        churn_steady(tc, 20000u, 20u, 48, 25000u);
}

/* ------------------------------------------------------ drain and refill
 * Insert tons, delete every one of them, insert again - whole-table cycles.
 * A leak anywhere in the dereference pool's freelists shows up as FULL on a
 * later cycle; stale tiny pointers show up as wrong lookups.
 */
static void churn_cycles_case(const tpht_test_case_t *tc) {
    enum { N = 6000, CYCLES = 4 };
    uint8_t key_size = tpht_test_key_size(tc->kind);
    tpht_test_table_t t = tpht_test_make_table(
        tc, tc->resize_mode == TPHT_RESIZABLE ? 8u : (size_t)N + N / 4u);
    int c;
    uint64_t i, out;

    assert(t.handle != NULL);
    for (c = 0; c < CYCLES; ++c) {
        /* Different key population every cycle. */
        uint64_t base = (uint64_t)c * 10u * N;
        for (i = 0; i < N; ++i)
            assert(t.put(t.handle, churn_key(base + i, key_size), i ^ base) == TPHT_OK);
        assert(t.size(t.handle) == (size_t)N);
        for (i = 0; i < N; ++i) {
            assert(t.get(t.handle, churn_key(base + i, key_size), &out) == TPHT_OK);
            assert(out == tpht_test_trunc_to(i ^ base, tc->value_size));
        }
        /* Previous cycle's keys must all be gone. */
        if (c > 0) {
            uint64_t prev = (uint64_t)(c - 1) * 10u * N;
            for (i = 0; i < 64u; ++i)
                assert(t.get(t.handle, churn_key(prev + i, key_size), &out) ==
                       TPHT_NOT_FOUND);
        }
        for (i = 0; i < N; ++i)
            assert(t.remove(t.handle, churn_key(base + i, key_size)) == TPHT_OK);
        assert(t.size(t.handle) == 0u);
    }
    t.destroy(t.handle);
}

/* ------------------------------------------------------------ size swings
 * Populations that grow and shrink by orders of magnitude, including churn
 * where inserts outpace deletes 10:1 (the "delete k, insert 10k" shape) and
 * the reverse.  Resizable tables ride their growth machinery up; capacity
 * never shrinks, so the drained table must stay exact at a tiny population
 * inside a huge structure.
 */
static void churn_swing_case(const tpht_test_case_t *tc) {
    enum { PEAK = 12000, TROUGH = 12 };
    uint8_t key_size = tpht_test_key_size(tc->kind);
    tpht_test_table_t t = tpht_test_make_table(
        tc, tc->resize_mode == TPHT_RESIZABLE ? 8u : (size_t)PEAK + PEAK / 4u);
    churn_model_t m;
    uint64_t next_key = 0, victim = 0, rng = 99u;
    int swing;
    uint64_t out;

    assert(t.handle != NULL);
    model_init(&m, PEAK);
    for (swing = 0; swing < 2; ++swing) {
        /* Grow-churn up to PEAK: each round inserts ten fresh keys and
         * deletes one old one. */
        while (m.live < PEAK) {
            int j;
            for (j = 0; j < 10; ++j) {
                uint64_t key = churn_key(next_key++, key_size);
                uint64_t val = tpht_test_trunc_to(tpht_test_next_rand(&rng), tc->value_size);
                assert(t.put(t.handle, key, val) == TPHT_OK);
                model_put(&m, key, val);
            }
            for (;;) {
                uint64_t key = churn_key(victim++, key_size);
                assert(victim <= next_key);
                if (model_remove(&m, key)) {
                    assert(t.remove(t.handle, key) == TPHT_OK);
                    break;
                }
            }
            assert(t.size(t.handle) == m.live);
        }
        churn_verify_full(&t, &m, next_key, key_size);
        /* Shrink-churn down to TROUGH: ten deletes per fresh insert. */
        while (m.live > TROUGH) {
            int j;
            uint64_t key = churn_key(next_key++, key_size);
            uint64_t val = tpht_test_trunc_to(tpht_test_next_rand(&rng), tc->value_size);
            assert(t.put(t.handle, key, val) == TPHT_OK);
            model_put(&m, key, val);
            for (j = 0; j < 10 && m.live > TROUGH; ++j) {
                for (;;) {
                    uint64_t k2 = churn_key(victim++, key_size);
                    assert(victim <= next_key);
                    if (model_remove(&m, k2)) {
                        assert(t.remove(t.handle, k2) == TPHT_OK);
                        break;
                    }
                }
            }
        }
        churn_verify_full(&t, &m, next_key, key_size);
        /* The near-empty table still answers dead keys correctly. */
        assert(t.get(t.handle, churn_key(victim ? victim - 1u : 0u, key_size), &out) ==
               TPHT_NOT_FOUND);
    }
    model_free(&m);
    t.destroy(t.handle);
}

/* ----------------------------------------------------- overflow budget
 * The parameter-sizing claim, tested rather than assumed.  A fixed table's
 * dereference pool is provisioned so that filling to the design load never
 * trips the hard-overflow absorb machinery; an absorb rebuild reallocates
 * storage, so memory_bytes moving below the design load would betray it.
 * Above the design load, absorb may fire, but every write must still land
 * and every key must still round-trip - across several hash seeds, so the
 * property is about the sizing, not one lucky key layout.
 */
static void overflow_budget_one(tpht_test_kind_t kind, int concurrent, size_t cap,
                                uint64_t seed) {
    tpht_options_t o = tpht_default_options();
    tpht_test_table_t t;
    uint8_t key_size = tpht_test_key_size(kind);
    size_t design = (size_t)((double)cap * 0.85);
    size_t mem_fresh;
    uint64_t i, out;

    o.hash_seed = seed;
    t = tpht_test_make_kind(kind, concurrent, TPHT_FIXED, cap, 8u, &o);
    if (!t.handle) return; /* combination not supported */
    mem_fresh = t.memory_bytes(t.handle);

    for (i = 0; i < (uint64_t)design; ++i)
        assert(t.put(t.handle, churn_key(i ^ (seed << 20), key_size), i) == TPHT_OK);
    /* No hard overflow below the design load: storage untouched. */
    assert(t.memory_bytes(t.handle) == mem_fresh);

    /* On to completely full: absorb may rebuild, correctness may not bend. */
    for (; i < (uint64_t)cap; ++i)
        assert(t.put(t.handle, churn_key(i ^ (seed << 20), key_size), i) == TPHT_OK);
    assert(t.size(t.handle) == cap);
    for (i = 0; i < (uint64_t)cap; ++i) {
        assert(t.get(t.handle, churn_key(i ^ (seed << 20), key_size), &out) == TPHT_OK);
        assert(out == i);
    }
    t.destroy(t.handle);
}

/* Resizable growth must be driven by the load factor alone: the capacity a
 * fill ends at is exactly the doubling chain the load factor predicts, never
 * more.  A hard overflow sneaking in below the load line would add doublings
 * (or growth escalations) and land the capacity somewhere else. */
static void growth_chain_one(tpht_test_kind_t kind, uint64_t n, uint64_t seed) {
    tpht_options_t o = tpht_default_options();
    tpht_test_table_t t;
    uint8_t key_size = tpht_test_key_size(kind);
    size_t cap, expected;
    uint64_t i;

    o.hash_seed = seed;
    o.max_load_factor = 0.85;
    t = tpht_test_make_kind(kind, 0, TPHT_RESIZABLE, 8u, 8u, &o);
    if (!t.handle) return;
    expected = t.capacity(t.handle);
    /* Mirror tpht_refresh_write_limit: insert #i sees size i-1 before it
     * lands, and the table doubles when that size has reached the write
     * limit floor(cap * lf - 1) + 1. */
    for (i = 1; i <= n; ++i) {
        double limit = (double)expected * 0.85 - 1.0;
        size_t whole = limit <= 0.0 ? 0u : (size_t)limit;
        if (i - 1u >= whole + 1u) expected *= 2u;
    }
    for (i = 0; i < n; ++i)
        assert(t.put(t.handle, churn_key(i + seed * 1000000u, key_size), i) == TPHT_OK);
    cap = t.capacity(t.handle);
    assert(cap == expected);
    t.destroy(t.handle);
}

static void overflow_budget_module(void) {
    static const size_t caps[] = {4096u, 65536u};
    static const tpht_test_kind_t kinds[] = {TPHT_TEST_FLAT32, TPHT_TEST_FLAT64,
                                             TPHT_TEST_CHAIN32, TPHT_TEST_CHAIN64};
    size_t ci, ki;
    uint64_t seed;
    for (ki = 0; ki < 4; ++ki) {
        for (ci = 0; ci < 2; ++ci) {
            uint64_t seeds = ci == 0 ? 5u : 2u; /* more seeds at the cheap size */
            for (seed = 1; seed <= seeds; ++seed) {
                overflow_budget_one(kinds[ki], 0, caps[ci], seed);
                if (seed == 1u) overflow_budget_one(kinds[ki], 1, caps[ci], seed);
            }
        }
        for (seed = 1; seed <= 3u; ++seed) growth_chain_one(kinds[ki], 30000u, seed);
    }
}

void tpht_test_run_churn_module(void) {
    tpht_test_for_each_case(churn_steady_case);
    tpht_test_for_each_case(churn_cycles_case);
    tpht_test_for_each_case(churn_swing_case);
    overflow_budget_module();
}
