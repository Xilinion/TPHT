#include "tpht.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

typedef enum bench_kind {
    BENCH_CHAINED32, BENCH_CHAINED64, BENCH_FLATTEN32, BENCH_FLATTEN64
} bench_kind_t;

static const char *kind_variant(bench_kind_t k) {
    return (k == BENCH_CHAINED32 || k == BENCH_CHAINED64) ? "chained-tpht" : "flatten-tpht";
}

static unsigned kind_key_bits(bench_kind_t k) {
    return (k == BENCH_CHAINED32 || k == BENCH_FLATTEN32) ? 32u : 64u;
}

static int run_one(bench_kind_t kind, uint8_t value_size, double load_factor,
                   size_t target_entries) {
    size_t capacity = (size_t)((double)target_entries / load_factor + 0.999999);
    tpht_options_t o = tpht_default_options();
    void *table;
    size_t inserted = 0, memory_bytes, payload_bytes, i;
    double actual_load, space_efficiency;

    o.resize_mode = TPHT_FIXED;
    o.value_size = value_size;
    o.hash_seed = UINT64_C(0x9ae16a3b2f90404f);

    switch (kind) {
        case BENCH_CHAINED32: table = chained_tpht32_create(capacity, 0, &o); break;
        case BENCH_CHAINED64: table = chained_tpht64_create(capacity, 0, &o); break;
        case BENCH_FLATTEN32: table = flatten_tpht32_create(capacity, &o); break;
        default: table = flatten_tpht64_create(capacity, &o); break;
    }
    if (!table) return 1;

    for (i = 0; i < target_entries; ++i) {
        uint64_t key = (uint64_t)(i + 1u);
        uint64_t value = (uint64_t)(i + 1u) * UINT64_C(2654435761);
        tpht_status_t st;
        switch (kind) {
            case BENCH_CHAINED32: st = chained_tpht32_put(table, (uint32_t)key, value); break;
            case BENCH_CHAINED64: st = chained_tpht64_put(table, key, value); break;
            case BENCH_FLATTEN32: st = flatten_tpht32_put(table, (uint32_t)key, value); break;
            default: st = flatten_tpht64_put(table, key, value); break;
        }
        if (st != TPHT_OK) break;
        inserted++;
    }

    switch (kind) {
        case BENCH_CHAINED32:
            memory_bytes = chained_tpht32_memory_bytes(table);
            capacity = chained_tpht32_capacity(table);
            break;
        case BENCH_CHAINED64:
            memory_bytes = chained_tpht64_memory_bytes(table);
            capacity = chained_tpht64_capacity(table);
            break;
        case BENCH_FLATTEN32:
            memory_bytes = flatten_tpht32_memory_bytes(table);
            capacity = flatten_tpht32_capacity(table);
            break;
        default:
            memory_bytes = flatten_tpht64_memory_bytes(table);
            capacity = flatten_tpht64_capacity(table);
            break;
    }

    payload_bytes = inserted * ((size_t)(kind_key_bits(kind) / 8u) + (size_t)value_size);
    actual_load = (double)inserted / (double)capacity;
    space_efficiency = memory_bytes ? (double)payload_bytes / (double)memory_bytes : 0.0;

    printf("%s,%u,%u,%.2f,%zu,%zu,%zu,%zu,%.6f,%.6f,%.2f\n",
           kind_variant(kind), kind_key_bits(kind), (unsigned)value_size * 8u, load_factor,
           capacity, inserted, payload_bytes, memory_bytes, actual_load, space_efficiency,
           memory_bytes / (1024.0 * 1024.0));

    switch (kind) {
        case BENCH_CHAINED32: chained_tpht32_destroy(table); break;
        case BENCH_CHAINED64: chained_tpht64_destroy(table); break;
        case BENCH_FLATTEN32: flatten_tpht32_destroy(table); break;
        default: flatten_tpht64_destroy(table); break;
    }
    return inserted == target_entries ? 0 : 2;
}

int main(int argc, char **argv) {
    static const double load_factors[] = {
        0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99};
    static const bench_kind_t kinds[] = {BENCH_CHAINED32, BENCH_CHAINED64, BENCH_FLATTEN32,
                                        BENCH_FLATTEN64};
    static const uint8_t value_sizes[] = {1u, 2u, 4u, 8u};
    size_t target_entries = 200000;
    size_t ki, si, li;
    int rc = 0;

    if (argc > 1) {
        target_entries = (size_t)strtoull(argv[1], NULL, 10);
        if (target_entries == 0) target_entries = 200000;
    }

    puts("variant,key_bits,value_bits,target_load,capacity,inserted,payload_bytes,memory_bytes,"
         "actual_load,space_efficiency,memory_mb");
    for (ki = 0; ki < sizeof(kinds) / sizeof(kinds[0]); ++ki) {
        for (si = 0; si < sizeof(value_sizes) / sizeof(value_sizes[0]); ++si) {
            for (li = 0; li < sizeof(load_factors) / sizeof(load_factors[0]); ++li) {
                int one = run_one(kinds[ki], value_sizes[si], load_factors[li], target_entries);
                if (one != 0) rc = one;
            }
        }
    }
    return rc;
}
