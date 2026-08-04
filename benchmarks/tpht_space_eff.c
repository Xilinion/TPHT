#include "tpht.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

static void put_le(uint8_t *dst, uint8_t size, uint64_t x) {
    uint8_t i;
    for (i = 0; i < size; ++i) dst[i] = (uint8_t)(x >> (8u * i));
}

static const char *variant_name(tpht_variant_t variant) {
    return variant == TPHT_CHAINED ? "chained-tpht" : "flatten-tpht";
}

static tpht_table_t *make_table(tpht_variant_t variant, size_t capacity,
                                uint8_t key_size, uint8_t value_size) {
    tpht_config_t c = tpht_default_config();
    c.variant = variant;
    c.threading = TPHT_SEQUENTIAL;
    c.resize_mode = TPHT_FIXED;
    c.initial_capacity = capacity;
    c.key_size = key_size;
    c.value_size = value_size;
    c.hash_seed = UINT64_C(0x9ae16a3b2f90404f);
    return tpht_create(&c);
}

static int run_one(tpht_variant_t variant, uint8_t key_size, uint8_t value_size,
                   double load_factor, size_t target_entries) {
    size_t capacity = (size_t)((double)target_entries / load_factor + 0.999999);
    tpht_table_t *table = make_table(variant, capacity, key_size, value_size);
    uint8_t key[8], value[8];
    size_t inserted = 0;
    size_t memory_bytes;
    size_t payload_bytes;
    double actual_load;
    double space_efficiency;
    size_t i;

    if (!table) return 1;

    for (i = 0; i < target_entries; ++i) {
        put_le(key, key_size, i + 1u);
        put_le(value, value_size, (i + 1u) * UINT64_C(11400714819323198485));
        if (tpht_put(table, key, value) != TPHT_OK) break;
        inserted++;
    }

    memory_bytes = tpht_memory_bytes(table);
    payload_bytes = inserted * ((size_t)key_size + (size_t)value_size);
    actual_load = (double)inserted / (double)tpht_capacity(table);
    space_efficiency = memory_bytes ? (double)payload_bytes / (double)memory_bytes : 0.0;

    printf("%s,%u,%u,%.2f,%zu,%zu,%zu,%zu,%.6f,%.6f,%.2f\n",
           variant_name(variant), (unsigned)key_size * 8u, (unsigned)value_size * 8u,
           load_factor, tpht_capacity(table), inserted, payload_bytes, memory_bytes,
           actual_load, space_efficiency, memory_bytes / (1024.0 * 1024.0));

    tpht_destroy(table);
    return inserted == target_entries ? 0 : 2;
}

int main(int argc, char **argv) {
    static const double load_factors[] = {
        0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99};
    static const uint8_t key_sizes[] = {4u, 8u};
    static const uint8_t value_sizes[] = {4u, 8u};
    static const tpht_variant_t variants[] = {TPHT_CHAINED, TPHT_FLATTEN};
    size_t target_entries = 200000;
    size_t vi, si, li;
    int rc = 0;

    if (argc > 1) {
        target_entries = (size_t)strtoull(argv[1], NULL, 10);
        if (target_entries == 0) target_entries = 200000;
    }

    puts("variant,key_bits,value_bits,target_load,capacity,inserted,payload_bytes,memory_bytes,actual_load,space_efficiency,memory_mb");
    for (vi = 0; vi < sizeof(variants) / sizeof(variants[0]); ++vi) {
        for (si = 0; si < sizeof(key_sizes) / sizeof(key_sizes[0]); ++si) {
            size_t vsi;
            for (vsi = 0; vsi < sizeof(value_sizes) / sizeof(value_sizes[0]); ++vsi) {
            for (li = 0; li < sizeof(load_factors) / sizeof(load_factors[0]); ++li) {
                int one = run_one(variants[vi], key_sizes[si], value_sizes[vsi],
                                  load_factors[li], target_entries);
                if (one != 0) rc = one;
            }
            }
        }
    }
    return rc;
}
