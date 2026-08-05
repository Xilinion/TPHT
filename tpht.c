/*
 * SPDX-FileCopyrightText: 2026 Xilin Tang and TPHT contributors
 *
 * TPHT is an independent industrial C implementation inspired by the TinyPtr
 * hash-table designs. This file also embeds a compact XXH64 implementation;
 * see the local xxHash attribution comment and README.md for details.
 */

#include "tpht.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#ifndef TPHT_ENABLE_SIMD
#define TPHT_ENABLE_SIMD 1
#endif

#define TPHT_SIMD_AUTO 0
#define TPHT_SIMD_SCALAR 1
#define TPHT_SIMD_SSE2 2
#define TPHT_SIMD_AVX2 3
#define TPHT_SIMD_NEON 4

#ifndef TPHT_SIMD_MODE
#define TPHT_SIMD_MODE TPHT_SIMD_AUTO
#endif

#if !TPHT_ENABLE_SIMD
#undef TPHT_SIMD_MODE
#define TPHT_SIMD_MODE TPHT_SIMD_SCALAR
#endif

#if TPHT_ENABLE_SIMD && (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
#include <immintrin.h>
#define TPHT_X86_SIMD 1
#endif

#if TPHT_ENABLE_SIMD && defined(__ARM_NEON)
#include <arm_neon.h>
#define TPHT_NEON_SIMD 1
#endif

#if defined(__GNUC__) || defined(__clang__)
#define TPHT_UNUSED __attribute__((unused))
#else
#define TPHT_UNUSED
#endif

#define TPHT_DEFAULT_BIN_SIZE 127u
#define TPHT_DEFAULT_LOAD_FACTOR 0.85
#define TPHT_MIN_CAPACITY 16u
#define TPHT_FLAT_CLOUD_BYTES 64u
#define TPHT_FLAT_META_BYTES 8u
#define TPHT_FLAT_FP_GROUP 32u
#define TPHT_CHAINED_DEREF_LOAD_NUM 95u
#define TPHT_CHAINED_DEREF_LOAD_DEN 100u
#define TPHT_DEFAULT_RESIZE_STRIDE_BUCKETS 64u

typedef struct tpht_pool {
    uint8_t *entries;
    uint8_t *cnt_head; /* [count, freelist-head] per bin. */
    size_t bin_count;
    uint8_t bin_size;
    size_t entry_size; /* next-byte + key + value. */
} tpht_pool_t;

typedef struct tpht_retired_storage {
    uint8_t *heads;
    uint8_t *flat_count;
    uint8_t *flat_fp;
    uint8_t *flat_entries;
    uint8_t *pool_entries;
    uint8_t *pool_cnt_head;
    atomic_flag *chain_locks;
    atomic_flag *pool_locks;
    uint8_t *resize_migrated;
    tpht_table_t *resize_descriptor;
    struct tpht_retired_storage *next;
} tpht_retired_storage_t;

struct tpht_table {
    tpht_config_t cfg;
    atomic_size_t size;
    size_t capacity;

    size_t key_size;
    size_t value_size;
    size_t key_quotient_size;
    size_t inline_entry_size; /* key + value. */
    size_t pool_entry_size;   /* next + key + value. */
    uint8_t key_bits;
    uint8_t base_bits;
    uint64_t base_mask;

    size_t base_count; /* chained base buckets or flatten clouds. */
    uint8_t *heads;    /* chained heads, or flatten overflow heads. */

    uint8_t *flat_count;
    uint8_t *flat_fp;
    uint8_t *flat_entries;
    uint8_t flat_inline_cap;

    tpht_pool_t pool;
    atomic_flag lock;
    atomic_flag *chain_locks;
    atomic_flag *pool_locks;
    size_t chain_lock_count;
    size_t pool_lock_count;

    atomic_bool resize_active;
    atomic_flag resize_start_lock;
    atomic_flag resize_commit_lock;
    atomic_size_t resize_next_stride;
    atomic_size_t resize_done_strides;
    atomic_size_t active_ops;
    size_t resize_stride_count;
    size_t resize_stride_size;
    tpht_table_t *resize_target;
    uint8_t *resize_migrated;
    tpht_retired_storage_t *retired;
};

static int tpht_valid_key_size(uint8_t n) { return n == 2u || n == 4u || n == 8u; }
static int tpht_valid_value_size(uint8_t n) { return n == 2u || n == 4u || n == 8u; }

static size_t tpht_max_size(size_t a, size_t b) { return a > b ? a : b; }

static size_t tpht_size_load(const tpht_table_t *t) {
    return atomic_load_explicit((atomic_size_t *)&t->size, memory_order_acquire);
}

static void tpht_size_store(tpht_table_t *t, size_t size) {
    atomic_store_explicit(&t->size, size, memory_order_release);
}

static void tpht_size_inc(tpht_table_t *t) {
    atomic_fetch_add_explicit(&t->size, 1u, memory_order_acq_rel);
}

static void tpht_size_dec(tpht_table_t *t) {
    atomic_fetch_sub_explicit(&t->size, 1u, memory_order_acq_rel);
}

static int tpht_size_try_reserve(tpht_table_t *t) {
    size_t cur = tpht_size_load(t);
    while (cur < t->capacity) {
        if (atomic_compare_exchange_weak_explicit(&t->size, &cur, cur + 1u,
                                                  memory_order_acq_rel,
                                                  memory_order_acquire)) {
            return 1;
        }
    }
    return 0;
}

static size_t tpht_ceil_mul_div(size_t x, size_t mul, size_t div) {
    size_t q = x / div;
    size_t r = x % div;
    if (q > ((size_t)-1) / mul) return (size_t)-1;
    q *= mul;
    if (r) {
        size_t add = (r * mul + div - 1u) / div;
        if (q > (size_t)-1 - add) return (size_t)-1;
        q += add;
    }
    return q;
}

static size_t tpht_pow2_ceil(size_t x) {
    size_t p = 1;
    if (x <= 1) return 1;
    while (p < x && p <= ((size_t)-1 / 2)) p <<= 1;
    return p < x ? x : p;
}

static uint8_t tpht_log2_pow2(size_t x) {
    uint8_t bits = 0;
    while (x > 1u) {
        x >>= 1u;
        ++bits;
    }
    return bits;
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

/*
 * Embedded XXH64 implementation for copy-paste portability.
 * Algorithm: xxHash / XXH64 by Yann Collet, BSD 2-Clause licensed.
 * Project: https://github.com/Cyan4973/xxHash
 * This file keeps a small local implementation instead of depending on
 * libxxhash so TPHT remains a two-file C library (tpht.h + tpht.c).
 */
static uint32_t tpht_read32_le(const void *p) {
    const uint8_t *b = (const uint8_t *)p;
    return ((uint32_t)b[0]) | ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

static uint64_t tpht_read64_le(const void *p) {
    const uint8_t *b = (const uint8_t *)p;
    return ((uint64_t)b[0]) | ((uint64_t)b[1] << 8) |
           ((uint64_t)b[2] << 16) | ((uint64_t)b[3] << 24) |
           ((uint64_t)b[4] << 32) | ((uint64_t)b[5] << 40) |
           ((uint64_t)b[6] << 48) | ((uint64_t)b[7] << 56);
}

static uint64_t tpht_xxh64_rotl(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64u - r));
}

static uint64_t tpht_xxh64_round(uint64_t acc, uint64_t input) {
    acc += input * UINT64_C(0xc2b2ae3d27d4eb4f);
    acc = tpht_xxh64_rotl(acc, 31);
    acc *= UINT64_C(0x9e3779b185ebca87);
    return acc;
}

static uint64_t tpht_xxh64_merge_round(uint64_t acc, uint64_t val) {
    val = tpht_xxh64_round(0, val);
    acc ^= val;
    acc = acc * UINT64_C(0x9e3779b185ebca87) + UINT64_C(0x85ebca77c2b2ae63);
    return acc;
}

static uint64_t tpht_xxh64_avalanche(uint64_t h) {
    h ^= h >> 33;
    h *= UINT64_C(0xc2b2ae3d27d4eb4f);
    h ^= h >> 29;
    h *= UINT64_C(0x165667b19e3779f9);
    h ^= h >> 32;
    return h;
}

static uint64_t tpht_hash_bytes(const void *data, size_t len, uint64_t seed) {
    const uint8_t *p = (const uint8_t *)data;
    const uint8_t *end = p + len;
    uint64_t h;

    if (len >= 32u) {
        const uint8_t *limit = end - 32u;
        uint64_t v1 = seed + UINT64_C(0x9e3779b185ebca87) + UINT64_C(0xc2b2ae3d27d4eb4f);
        uint64_t v2 = seed + UINT64_C(0xc2b2ae3d27d4eb4f);
        uint64_t v3 = seed;
        uint64_t v4 = seed - UINT64_C(0x9e3779b185ebca87);

        do {
            v1 = tpht_xxh64_round(v1, tpht_read64_le(p));
            p += 8;
            v2 = tpht_xxh64_round(v2, tpht_read64_le(p));
            p += 8;
            v3 = tpht_xxh64_round(v3, tpht_read64_le(p));
            p += 8;
            v4 = tpht_xxh64_round(v4, tpht_read64_le(p));
            p += 8;
        } while (p <= limit);

        h = tpht_xxh64_rotl(v1, 1) + tpht_xxh64_rotl(v2, 7) +
            tpht_xxh64_rotl(v3, 12) + tpht_xxh64_rotl(v4, 18);
        h = tpht_xxh64_merge_round(h, v1);
        h = tpht_xxh64_merge_round(h, v2);
        h = tpht_xxh64_merge_round(h, v3);
        h = tpht_xxh64_merge_round(h, v4);
    } else {
        h = seed + UINT64_C(0x27d4eb2f165667c5);
    }

    h += (uint64_t)len;

    while (p + 8u <= end) {
        uint64_t k1 = tpht_xxh64_round(0, tpht_read64_le(p));
        h ^= k1;
        h = tpht_xxh64_rotl(h, 27) * UINT64_C(0x9e3779b185ebca87) +
            UINT64_C(0x85ebca77c2b2ae63);
        p += 8;
    }

    if (p + 4u <= end) {
        h ^= (uint64_t)tpht_read32_le(p) * UINT64_C(0x9e3779b185ebca87);
        h = tpht_xxh64_rotl(h, 23) * UINT64_C(0xc2b2ae3d27d4eb4f) +
            UINT64_C(0x165667b19e3779f9);
        p += 4;
    }

    while (p < end) {
        h ^= (uint64_t)(*p) * UINT64_C(0x27d4eb2f165667c5);
        h = tpht_xxh64_rotl(h, 11) * UINT64_C(0x9e3779b185ebca87);
        ++p;
    }

    return tpht_xxh64_avalanche(h);
}

static uint64_t tpht_hash_word(uint64_t x, uint64_t seed) {
    uint8_t b[8];
    tpht_write_le(b, sizeof(b), x);
    return tpht_hash_bytes(b, sizeof(b), seed);
}

static uint64_t tpht_key_word(const tpht_table_t *t, const void *key) {
    return tpht_read_le(key, t->key_size) &
           (t->key_bits == 64u ? UINT64_MAX : ((UINT64_C(1) << t->key_bits) - 1u));
}

static uint64_t tpht_key_quotient(const tpht_table_t *t, const void *key) {
    return tpht_key_word(t, key) >> t->base_bits;
}

static size_t tpht_base_from_word(const tpht_table_t *t, uint64_t key_word) {
    uint64_t quotient = key_word >> t->base_bits;
    return (size_t)((tpht_hash_word(quotient, t->cfg.hash_seed) ^ key_word) & t->base_mask);
}

static size_t tpht_base_from_key(const tpht_table_t *t, const void *key) {
    return tpht_base_from_word(t, tpht_key_word(t, key));
}

static void tpht_write_quotient(const tpht_table_t *t, uint8_t *dst, const void *key) {
    tpht_write_le(dst, t->key_quotient_size, tpht_key_quotient(t, key));
}

static uint64_t tpht_read_quotient(const tpht_table_t *t, const uint8_t *stored_key) {
    return tpht_read_le(stored_key, t->key_quotient_size);
}

static void tpht_rebuild_key(const tpht_table_t *t, size_t base, const uint8_t *stored_key,
                             uint8_t *key_out) {
    uint64_t quotient = tpht_read_quotient(t, stored_key);
    uint64_t low = (tpht_hash_word(quotient, t->cfg.hash_seed) ^ (uint64_t)base) & t->base_mask;
    uint64_t key_word = (quotient << t->base_bits) | low;
    tpht_write_le(key_out, t->key_size, key_word);
}

static TPHT_UNUSED uint32_t tpht_low_mask(uint8_t count) {
    return count >= 32u ? UINT32_MAX : ((UINT32_C(1) << count) - 1u);
}

static TPHT_UNUSED uint32_t tpht_fp_match_mask_scalar(const uint8_t *fps, uint8_t count, uint8_t fp) {
    uint32_t mask = 0;
    uint8_t i;
    for (i = 0; i < count; ++i) {
        if (fps[i] == fp) mask |= UINT32_C(1) << i;
    }
    return mask & tpht_low_mask(count);
}

#if defined(TPHT_X86_SIMD) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
static TPHT_UNUSED uint32_t tpht_fp_match_mask_avx2(const uint8_t *fps, uint8_t count, uint8_t fp) {
    __m256i needle = _mm256_set1_epi8((char)fp);
    __m256i hay = _mm256_loadu_si256((const __m256i *)(const void *)fps);
    __m256i eq = _mm256_cmpeq_epi8(hay, needle);
    return (uint32_t)_mm256_movemask_epi8(eq) & tpht_low_mask(count);
}

static TPHT_UNUSED int tpht_cpu_has_avx2(void) {
#if defined(__GNUC__) || defined(__clang__)
    static int cached = -1;
    if (cached < 0) cached = __builtin_cpu_supports("avx2") ? 1 : 0;
    return cached;
#else
    return 0;
#endif
}
#endif

#if defined(TPHT_X86_SIMD) && (defined(__SSE2__) || defined(_M_X64) || defined(_M_IX86_FP))
static TPHT_UNUSED uint32_t tpht_fp_match_mask_sse2(const uint8_t *fps, uint8_t count, uint8_t fp) {
    __m128i needle = _mm_set1_epi8((char)fp);
    __m128i hay0 = _mm_loadu_si128((const __m128i *)(const void *)fps);
    uint32_t mask = (uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi8(hay0, needle));
    if (count > 16u) {
        __m128i hay1 = _mm_loadu_si128((const __m128i *)(const void *)(fps + 16u));
        mask |= (uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi8(hay1, needle)) << 16u;
    }
    return mask & tpht_low_mask(count);
}
#endif

#if defined(TPHT_NEON_SIMD)
static TPHT_UNUSED uint32_t tpht_fp_match_mask_neon(const uint8_t *fps, uint8_t count, uint8_t fp) {
    uint8x16_t needle = vdupq_n_u8(fp);
    uint32_t mask = 0;
    uint8_t tmp[16];
    uint8_t i;
    vst1q_u8(tmp, vceqq_u8(vld1q_u8(fps), needle));
    for (i = 0; i < 16u && i < count; ++i) {
        if (tmp[i]) mask |= UINT32_C(1) << i;
    }
    if (count > 16u) {
        vst1q_u8(tmp, vceqq_u8(vld1q_u8(fps + 16u), needle));
        for (i = 0; i < count - 16u; ++i) {
            if (tmp[i]) mask |= UINT32_C(1) << (i + 16u);
        }
    }
    return mask;
}
#endif

static uint32_t tpht_fp_match_mask(const uint8_t *fps, uint8_t count, uint8_t fp) {
#if TPHT_SIMD_MODE == TPHT_SIMD_SCALAR
    return tpht_fp_match_mask_scalar(fps, count, fp);
#elif TPHT_SIMD_MODE == TPHT_SIMD_AVX2
#if defined(TPHT_X86_SIMD) && (defined(__GNUC__) || defined(__clang__))
    return tpht_fp_match_mask_avx2(fps, count, fp);
#else
    return tpht_fp_match_mask_scalar(fps, count, fp);
#endif
#elif TPHT_SIMD_MODE == TPHT_SIMD_SSE2
#if defined(TPHT_X86_SIMD) && (defined(__SSE2__) || defined(_M_X64) || defined(_M_IX86_FP))
    return tpht_fp_match_mask_sse2(fps, count, fp);
#else
    return tpht_fp_match_mask_scalar(fps, count, fp);
#endif
#elif TPHT_SIMD_MODE == TPHT_SIMD_NEON
#if defined(TPHT_NEON_SIMD)
    return tpht_fp_match_mask_neon(fps, count, fp);
#else
    return tpht_fp_match_mask_scalar(fps, count, fp);
#endif
#else
#if defined(TPHT_X86_SIMD) && (defined(__GNUC__) || defined(__clang__))
    if (count > 16u && tpht_cpu_has_avx2()) return tpht_fp_match_mask_avx2(fps, count, fp);
#endif
#if defined(TPHT_X86_SIMD) && (defined(__SSE2__) || defined(_M_X64) || defined(_M_IX86_FP))
    return tpht_fp_match_mask_sse2(fps, count, fp);
#elif defined(TPHT_NEON_SIMD)
    return tpht_fp_match_mask_neon(fps, count, fp);
#else
    return tpht_fp_match_mask_scalar(fps, count, fp);
#endif
#endif
}

static uint8_t tpht_ctz32(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return (uint8_t)__builtin_ctz(x);
#else
    uint8_t n = 0;
    while (((x >> n) & 1u) == 0u) ++n;
    return n;
#endif
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

static void tpht_flag_lock(atomic_flag *lock) {
    while (atomic_flag_test_and_set_explicit(lock, memory_order_acquire)) {
    }
}

static void tpht_flag_unlock(atomic_flag *lock) {
    atomic_flag_clear_explicit(lock, memory_order_release);
}

static int tpht_chained_fine_grained(const tpht_table_t *t) {
    return t->cfg.variant == TPHT_CHAINED && t->cfg.threading == TPHT_CONCURRENT &&
           t->chain_locks && t->pool_locks;
}

static void tpht_chain_lock_base(tpht_table_t *t, size_t base) {
    if (tpht_chained_fine_grained(t)) tpht_flag_lock(&t->chain_locks[base % t->chain_lock_count]);
}

static void tpht_chain_unlock_base(tpht_table_t *t, size_t base) {
    if (tpht_chained_fine_grained(t)) tpht_flag_unlock(&t->chain_locks[base % t->chain_lock_count]);
}

static void tpht_pool_lock_bin(tpht_table_t *t, size_t bin) {
    if (t->pool_locks) tpht_flag_lock(&t->pool_locks[bin]);
}

static void tpht_pool_unlock_bin(tpht_table_t *t, size_t bin) {
    if (t->pool_locks) tpht_flag_unlock(&t->pool_locks[bin]);
}

static void tpht_pool_lock_pair(tpht_table_t *t, size_t a, size_t b) {
    if (!t->pool_locks) return;
    if (a == b) {
        tpht_pool_lock_bin(t, a);
    } else if (a < b) {
        tpht_pool_lock_bin(t, a);
        tpht_pool_lock_bin(t, b);
    } else {
        tpht_pool_lock_bin(t, b);
        tpht_pool_lock_bin(t, a);
    }
}

static void tpht_pool_unlock_pair(tpht_table_t *t, size_t a, size_t b) {
    if (!t->pool_locks) return;
    if (a == b) {
        tpht_pool_unlock_bin(t, a);
    } else if (a < b) {
        tpht_pool_unlock_bin(t, b);
        tpht_pool_unlock_bin(t, a);
    } else {
        tpht_pool_unlock_bin(t, a);
        tpht_pool_unlock_bin(t, b);
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

    tpht_pool_lock_pair(t, bin1, bin2);

    if (bin1 != bin2 && tpht_pool_count(&t->pool, bin1) > tpht_pool_count(&t->pool, bin2)) {
        bin = bin2;
        flag = 1;
    }

    cnt = tpht_pool_count_ptr(&t->pool, bin);
    head = tpht_pool_head_ptr(&t->pool, bin);
    if (*head >= t->pool.bin_size) {
        tpht_pool_unlock_pair(t, bin1, bin2);
        return NULL;
    }

    pos = *head;
    entry = tpht_pool_entry(&t->pool, bin, pos);
    *head = (uint8_t)(pos + 1u + entry[0]);
    if (*head > t->pool.bin_size) *head = (uint8_t)(*head - (t->pool.bin_size + 1u));
    *cnt = (uint8_t)(*cnt + 1u);
    *encoded_out = (uint8_t)((pos + 1u) | (flag << 7u));
    tpht_pool_unlock_pair(t, bin1, bin2);
    return entry;
}

static void tpht_pool_free(tpht_table_t *t, uint8_t encoded_ptr, uint8_t *entry) {
    size_t ordinal = (size_t)((entry - t->pool.entries) / (ptrdiff_t)t->pool.entry_size);
    size_t bin = ordinal / t->pool.bin_size;
    uint8_t pos = (uint8_t)(ordinal % t->pool.bin_size);
    uint8_t *cnt = tpht_pool_count_ptr(&t->pool, bin);
    uint8_t *head = tpht_pool_head_ptr(&t->pool, bin);
    (void)encoded_ptr;
    tpht_pool_lock_bin(t, bin);
    entry[0] = (uint8_t)(*head + t->pool.bin_size - pos);
    if (entry[0] > t->pool.bin_size) entry[0] = (uint8_t)(entry[0] - (t->pool.bin_size + 1u));
    *head = pos;
    *cnt = (uint8_t)(*cnt - 1u);
    tpht_pool_unlock_bin(t, bin);
}

static void tpht_free_storage(tpht_table_t *t) {
    tpht_retired_storage_t *r = t->retired;
    while (r) {
        tpht_retired_storage_t *next = r->next;
        free(r->heads);
        free(r->flat_count);
        free(r->flat_fp);
        free(r->flat_entries);
        free(r->pool_entries);
        free(r->pool_cnt_head);
        free(r->chain_locks);
        free(r->pool_locks);
        free(r->resize_migrated);
        free(r->resize_descriptor);
        free(r);
        r = next;
    }
    t->retired = NULL;
    free(t->heads);
    free(t->flat_count);
    free(t->flat_fp);
    free(t->flat_entries);
    free(t->pool.entries);
    free(t->pool.cnt_head);
    free(t->chain_locks);
    free(t->pool_locks);
    if (t->resize_target) {
        tpht_free_storage(t->resize_target);
        free(t->resize_target);
    }
    free(t->resize_migrated);
    t->heads = NULL;
    t->flat_count = NULL;
    t->flat_fp = NULL;
    t->flat_entries = NULL;
    t->pool.entries = NULL;
    t->pool.cnt_head = NULL;
    t->chain_locks = NULL;
    t->pool_locks = NULL;
    t->chain_lock_count = 0;
    t->pool_lock_count = 0;
    t->resize_target = NULL;
    t->resize_migrated = NULL;
    t->resize_stride_count = 0;
    t->resize_stride_size = 0;
}

static int tpht_alloc_storage(tpht_table_t *t, size_t capacity) {
    size_t overflow_slots;
    size_t cloud_target;

    t->capacity = tpht_max_size(capacity, TPHT_MIN_CAPACITY);
    t->key_size = t->cfg.key_size;
    t->value_size = t->cfg.value_size;
    t->pool.bin_size = t->cfg.bin_size ? t->cfg.bin_size : TPHT_DEFAULT_BIN_SIZE;
    if (t->pool.bin_size == 0 || t->pool.bin_size > 127u) return 0;

    if (t->cfg.variant == TPHT_CHAINED) {
        t->base_count = tpht_pow2_ceil(t->capacity);
        t->key_bits = (uint8_t)(t->key_size * 8u);
        t->base_bits = tpht_log2_pow2(t->base_count);
        if (t->base_bits > t->key_bits) t->base_bits = t->key_bits;
        t->base_mask = t->base_bits == 64u ? UINT64_MAX : ((UINT64_C(1) << t->base_bits) - 1u);
        t->key_quotient_size = (size_t)((t->key_bits - t->base_bits + 7u) / 8u);
        t->inline_entry_size = t->key_quotient_size + t->value_size;
        t->pool_entry_size = 1u + t->inline_entry_size;
        t->pool.entry_size = t->pool_entry_size;
        t->heads = (uint8_t *)calloc(t->base_count, 1);
        overflow_slots = tpht_ceil_mul_div(t->capacity, TPHT_CHAINED_DEREF_LOAD_DEN,
                                           TPHT_CHAINED_DEREF_LOAD_NUM);
    } else {
        t->flat_inline_cap = TPHT_FLAT_FP_GROUP;

        cloud_target = (t->capacity + t->flat_inline_cap - 1u) / t->flat_inline_cap;
        t->base_count = tpht_pow2_ceil(tpht_max_size(cloud_target, 1));
        t->key_bits = (uint8_t)(t->key_size * 8u);
        t->base_bits = tpht_log2_pow2(t->base_count);
        if (t->base_bits > t->key_bits) t->base_bits = t->key_bits;
        t->base_mask = t->base_bits == 64u ? UINT64_MAX : ((UINT64_C(1) << t->base_bits) - 1u);
        t->key_quotient_size = (size_t)((t->key_bits - t->base_bits + 7u) / 8u);
        t->inline_entry_size = t->key_quotient_size + t->value_size;
        t->pool_entry_size = 1u + t->inline_entry_size;
        t->pool.entry_size = t->pool_entry_size;
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

    if (t->cfg.variant == TPHT_CHAINED && t->cfg.threading == TPHT_CONCURRENT) {
        size_t i;
        t->chain_lock_count = t->base_count;
        t->pool_lock_count = t->pool.bin_count;
        t->chain_locks = (atomic_flag *)malloc(t->chain_lock_count * sizeof(*t->chain_locks));
        t->pool_locks = (atomic_flag *)malloc(t->pool_lock_count * sizeof(*t->pool_locks));
        if (t->chain_locks && t->pool_locks) {
            for (i = 0; i < t->chain_lock_count; ++i) atomic_flag_clear(&t->chain_locks[i]);
            for (i = 0; i < t->pool_lock_count; ++i) atomic_flag_clear(&t->pool_locks[i]);
        }
    }

    if (!t->heads || !t->pool.entries || !t->pool.cnt_head ||
        (t->cfg.variant == TPHT_CHAINED && t->cfg.threading == TPHT_CONCURRENT &&
         (!t->chain_locks || !t->pool_locks)) ||
        (t->cfg.variant == TPHT_FLATTEN && (!t->flat_count || !t->flat_fp || !t->flat_entries))) {
        tpht_free_storage(t);
        return 0;
    }
    return 1;
}

static uint8_t *tpht_inline_entry(tpht_table_t *t, size_t cloud, size_t pos) {
    return t->flat_entries + ((cloud * (size_t)t->flat_inline_cap + pos) * t->inline_entry_size);
}

static int tpht_stored_key_equal(tpht_table_t *t, const uint8_t *stored_key, const void *key) {
    return tpht_read_quotient(t, stored_key) == tpht_key_quotient(t, key);
}

static size_t tpht_chained_base(tpht_table_t *t, const void *key) {
    return tpht_base_from_key(t, key);
}

static size_t tpht_flat_cloud(tpht_table_t *t, const void *key) {
    return tpht_base_from_key(t, key);
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
        if (tpht_stored_key_equal(t, entry + 1u, key)) {
            if (value_out && t->value_size) memcpy(value_out, entry + 1u + t->key_quotient_size, t->value_size);
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
        if (tpht_stored_key_equal(t, entry + 1u, key)) {
            if (!replace) return TPHT_EXISTS;
            if (t->value_size) memcpy(entry + 1u + t->key_quotient_size, value, t->value_size);
            return TPHT_OK;
        }
        prev = entry;
    }

    entry = tpht_pool_alloc(t, (uint64_t)(uintptr_t)prev, &encoded);
    if (!entry) return TPHT_FULL;
    *prev = encoded;
    entry[0] = 0;
    tpht_write_quotient(t, entry + 1u, key);
    if (t->value_size) memcpy(entry + 1u + t->key_quotient_size, value, t->value_size);
    tpht_size_inc(t);
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
        if (tpht_stored_key_equal(t, entry + 1u, key)) {
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
    tpht_size_dec(t);
    return TPHT_OK;
}

static tpht_status_t tpht_flat_get_raw(tpht_table_t *t, const void *key, void *value_out) {
    size_t cloud = tpht_flat_cloud(t, key);
    uint8_t fp = tpht_fingerprint(t, key);
    uint8_t count = t->flat_count[cloud];
    uint8_t *fps = t->flat_fp + cloud * (size_t)t->flat_inline_cap;
    uint8_t *prev;
    uint32_t mask = tpht_fp_match_mask(fps, count, fp);

    while (mask) {
        uint8_t i = tpht_ctz32(mask);
        uint8_t *slot = tpht_inline_entry(t, cloud, i);
        mask &= mask - 1u;
        if (tpht_stored_key_equal(t, slot, key)) {
            if (value_out && t->value_size) memcpy(value_out, slot + t->key_quotient_size, t->value_size);
            return TPHT_OK;
        }
    }

    prev = &t->heads[cloud];
    while (*prev) {
        uint8_t *entry = tpht_pool_deref(t, (uint64_t)cloud, *prev);
        if (tpht_stored_key_equal(t, entry + 1u, key)) {
            if (value_out && t->value_size) memcpy(value_out, entry + 1u + t->key_quotient_size, t->value_size);
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
    uint32_t mask = tpht_fp_match_mask(fps, count, fp);
    uint8_t *entry;

    while (mask) {
        uint8_t i = tpht_ctz32(mask);
        uint8_t *slot = tpht_inline_entry(t, cloud, i);
        mask &= mask - 1u;
        if (tpht_stored_key_equal(t, slot, key)) {
            if (!replace) return TPHT_EXISTS;
            if (t->value_size) memcpy(slot + t->key_quotient_size, value, t->value_size);
            return TPHT_OK;
        }
    }
    prev = &t->heads[cloud];
    while (*prev) {
        entry = tpht_pool_deref(t, (uint64_t)cloud, *prev);
        if (tpht_stored_key_equal(t, entry + 1u, key)) {
            if (!replace) return TPHT_EXISTS;
            if (t->value_size) memcpy(entry + 1u + t->key_quotient_size, value, t->value_size);
            return TPHT_OK;
        }
        prev = entry;
    }

    if (count < t->flat_inline_cap) {
        uint8_t *slot = tpht_inline_entry(t, cloud, count);
        fps[count] = fp;
        tpht_write_quotient(t, slot, key);
        if (t->value_size) memcpy(slot + t->key_quotient_size, value, t->value_size);
        t->flat_count[cloud] = (uint8_t)(count + 1u);
        tpht_size_inc(t);
        return TPHT_OK;
    }

    entry = tpht_pool_alloc(t, (uint64_t)cloud, &encoded);
    if (!entry) return TPHT_FULL;
    entry[0] = t->heads[cloud];
    t->heads[cloud] = encoded;
    tpht_write_quotient(t, entry + 1u, key);
    if (t->value_size) memcpy(entry + 1u + t->key_quotient_size, value, t->value_size);
    tpht_size_inc(t);
    return TPHT_OK;
}

static tpht_status_t tpht_flat_remove_raw(tpht_table_t *t, const void *key) {
    size_t cloud = tpht_flat_cloud(t, key);
    uint8_t fp = tpht_fingerprint(t, key);
    uint8_t count = t->flat_count[cloud];
    uint8_t *fps = t->flat_fp + cloud * (size_t)t->flat_inline_cap;
    uint32_t mask = tpht_fp_match_mask(fps, count, fp);
    uint8_t *prev;

    while (mask) {
        uint8_t i = tpht_ctz32(mask);
        uint8_t *slot = tpht_inline_entry(t, cloud, i);
        mask &= mask - 1u;
        if (tpht_stored_key_equal(t, slot, key)) {
            if (t->heads[cloud]) {
                uint8_t encoded = t->heads[cloud];
                uint8_t *entry = tpht_pool_deref(t, (uint64_t)cloud, encoded);
                uint8_t moved_key[8];
                t->heads[cloud] = entry[0];
                tpht_rebuild_key(t, cloud, entry + 1u, moved_key);
                fps[i] = tpht_fingerprint(t, moved_key);
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
            tpht_size_dec(t);
            return TPHT_OK;
        }
    }

    prev = &t->heads[cloud];
    while (*prev) {
        uint8_t encoded = *prev;
        uint8_t *entry = tpht_pool_deref(t, (uint64_t)cloud, encoded);
        if (tpht_stored_key_equal(t, entry + 1u, key)) {
            *prev = entry[0];
            tpht_pool_free(t, encoded, entry);
            tpht_size_dec(t);
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

static tpht_status_t tpht_chained_write_fine(tpht_table_t *t, const void *key,
                                             const void *value, int replace);
static tpht_status_t tpht_chained_get_fine(tpht_table_t *t, const void *key,
                                           void *value_out);
static void tpht_chained_resize_quiesce_and_commit(tpht_table_t *t);

static int tpht_chained_resize_active(tpht_table_t *t) {
    return atomic_load_explicit(&t->resize_active, memory_order_acquire) != 0;
}

static void tpht_op_enter(tpht_table_t *t) {
    tpht_flag_lock(&t->resize_start_lock);
    atomic_fetch_add_explicit(&t->active_ops, 1u, memory_order_acq_rel);
    tpht_flag_unlock(&t->resize_start_lock);
}

static void tpht_op_exit(tpht_table_t *t) {
    atomic_fetch_sub_explicit(&t->active_ops, 1u, memory_order_acq_rel);
}

static int tpht_chained_resize_start(tpht_table_t *t, size_t new_capacity) {
    tpht_table_t *nt;
    size_t requested_strides;
    if (tpht_chained_resize_active(t)) return 1;

    tpht_flag_lock(&t->resize_start_lock);
    if (tpht_chained_resize_active(t)) {
        tpht_flag_unlock(&t->resize_start_lock);
        return 1;
    }

    nt = (tpht_table_t *)calloc(1, sizeof(*nt));
    if (!nt) {
        tpht_flag_unlock(&t->resize_start_lock);
        return 0;
    }
    nt->cfg = t->cfg;
    nt->cfg.resize_mode = TPHT_FIXED;
    nt->capacity = 0;
    atomic_init(&nt->size, 0);
    atomic_init(&nt->resize_active, 0);
    atomic_init(&nt->resize_next_stride, 0);
    atomic_init(&nt->resize_done_strides, 0);
    atomic_init(&nt->active_ops, 0);
    atomic_flag_clear(&nt->lock);
    atomic_flag_clear(&nt->resize_start_lock);
    atomic_flag_clear(&nt->resize_commit_lock);
    if (!tpht_alloc_storage(nt, new_capacity)) {
        free(nt);
        tpht_flag_unlock(&t->resize_start_lock);
        return 0;
    }

    t->resize_migrated = (uint8_t *)calloc(t->base_count, 1);
    if (!t->resize_migrated) {
        tpht_free_storage(nt);
        free(nt);
        tpht_flag_unlock(&t->resize_start_lock);
        return 0;
    }

    t->resize_target = nt;
    requested_strides = t->cfg.resize_strides
                            ? t->cfg.resize_strides
                            : ((t->base_count + TPHT_DEFAULT_RESIZE_STRIDE_BUCKETS - 1u) /
                               TPHT_DEFAULT_RESIZE_STRIDE_BUCKETS);
    if (requested_strides == 0) requested_strides = 1;
    if (requested_strides > t->base_count) requested_strides = t->base_count;
    t->resize_stride_size = (t->base_count + requested_strides - 1u) / requested_strides;
    t->resize_stride_count = (t->base_count + t->resize_stride_size - 1u) / t->resize_stride_size;
    atomic_store_explicit(&t->resize_next_stride, 0, memory_order_release);
    atomic_store_explicit(&t->resize_done_strides, 0, memory_order_release);
    atomic_flag_clear(&t->resize_commit_lock);
    atomic_store_explicit(&t->resize_active, 1, memory_order_release);
    tpht_flag_unlock(&t->resize_start_lock);
    return 1;
}

static void tpht_chained_resize_commit(tpht_table_t *t) {
    tpht_table_t *nt;
    uint8_t *old_heads;
    uint8_t *old_flat_count;
    uint8_t *old_flat_fp;
    uint8_t *old_flat_entries;
    uint8_t *old_pool_entries;
    uint8_t *old_pool_cnt_head;
    atomic_flag *old_chain_locks;
    atomic_flag *old_pool_locks;
    uint8_t *old_migrated;
    tpht_retired_storage_t *retired;

    if (atomic_load_explicit(&t->resize_done_strides, memory_order_acquire) < t->resize_stride_count) return;
    if (atomic_flag_test_and_set_explicit(&t->resize_commit_lock, memory_order_acquire)) return;
    tpht_flag_lock(&t->resize_start_lock);
    if (atomic_load_explicit(&t->active_ops, memory_order_acquire) > 1u) {
        tpht_flag_unlock(&t->resize_start_lock);
        atomic_flag_clear_explicit(&t->resize_commit_lock, memory_order_release);
        return;
    }

    nt = t->resize_target;
    if (!nt) {
        atomic_store_explicit(&t->resize_active, 0, memory_order_release);
        tpht_flag_unlock(&t->resize_start_lock);
        atomic_flag_clear_explicit(&t->resize_commit_lock, memory_order_release);
        return;
    }

    old_heads = t->heads;
    old_flat_count = t->flat_count;
    old_flat_fp = t->flat_fp;
    old_flat_entries = t->flat_entries;
    old_pool_entries = t->pool.entries;
    old_pool_cnt_head = t->pool.cnt_head;
    old_chain_locks = t->chain_locks;
    old_pool_locks = t->pool_locks;
    old_migrated = t->resize_migrated;
    retired = (tpht_retired_storage_t *)calloc(1, sizeof(*retired));

    t->capacity = nt->capacity;
    tpht_size_store(t, tpht_size_load(nt));
    t->key_size = nt->key_size;
    t->value_size = nt->value_size;
    t->key_quotient_size = nt->key_quotient_size;
    t->inline_entry_size = nt->inline_entry_size;
    t->pool_entry_size = nt->pool_entry_size;
    t->key_bits = nt->key_bits;
    t->base_bits = nt->base_bits;
    t->base_mask = nt->base_mask;
    t->base_count = nt->base_count;
    t->heads = nt->heads;
    t->flat_count = nt->flat_count;
    t->flat_fp = nt->flat_fp;
    t->flat_entries = nt->flat_entries;
    t->flat_inline_cap = nt->flat_inline_cap;
    t->pool = nt->pool;
    t->chain_locks = nt->chain_locks;
    t->pool_locks = nt->pool_locks;
    t->chain_lock_count = nt->chain_lock_count;
    t->pool_lock_count = nt->pool_lock_count;
    t->resize_target = NULL;
    t->resize_migrated = NULL;
    t->resize_stride_count = 0;
    t->resize_stride_size = 0;

    /*
     * Keep the small resize-target descriptor alive after publishing. A helper
     * thread may have taken a local copy of resize_target just before commit;
     * the storage now belongs to t, but the descriptor must remain readable
     * until that operation returns. Old backing arrays are retired below.
     */

    if (retired) {
        retired->heads = old_heads;
        retired->flat_count = old_flat_count;
        retired->flat_fp = old_flat_fp;
        retired->flat_entries = old_flat_entries;
        retired->pool_entries = old_pool_entries;
        retired->pool_cnt_head = old_pool_cnt_head;
        retired->chain_locks = old_chain_locks;
        retired->pool_locks = old_pool_locks;
        retired->resize_migrated = old_migrated;
        retired->resize_descriptor = nt;
        retired->next = t->retired;
        t->retired = retired;
    }

    atomic_store_explicit(&t->resize_active, 0, memory_order_release);
    tpht_flag_unlock(&t->resize_start_lock);
    atomic_flag_clear_explicit(&t->resize_commit_lock, memory_order_release);
}

static void tpht_chained_resize_migrate_bucket(tpht_table_t *t, size_t base) {
    tpht_table_t *nt;
    if (!tpht_chained_resize_active(t)) return;
    nt = t->resize_target;
    if (!nt) return;

    tpht_chain_lock_base(t, base);
    if (!t->resize_migrated[base]) {
        uint8_t *prev = &t->heads[base];
        while (*prev) {
            uint8_t rebuilt_key[8];
            uint8_t *entry = tpht_pool_deref(t, (uint64_t)(uintptr_t)prev, *prev);
            tpht_rebuild_key(t, base, entry + 1u, rebuilt_key);
            (void)tpht_chained_write_fine(nt, rebuilt_key, entry + 1u + t->key_quotient_size, 1);
            prev = entry;
        }
        t->resize_migrated[base] = 1u;
    }
    tpht_chain_unlock_base(t, base);
}

static void tpht_chained_resize_migrate_stride(tpht_table_t *t, size_t stride) {
    size_t begin = stride * t->resize_stride_size;
    size_t end = begin + t->resize_stride_size;
    size_t base;
    if (begin >= t->base_count) return;
    if (end > t->base_count) end = t->base_count;
    for (base = begin; base < end && tpht_chained_resize_active(t); ++base) {
        tpht_chained_resize_migrate_bucket(t, base);
    }
    atomic_fetch_add_explicit(&t->resize_done_strides, 1u, memory_order_acq_rel);
    tpht_chained_resize_commit(t);
}

static void tpht_chained_resize_finish_all(tpht_table_t *t) {
    size_t stride;
    while (tpht_chained_resize_active(t) &&
           (stride = atomic_fetch_add_explicit(&t->resize_next_stride, 1u,
                                               memory_order_acq_rel)) < t->resize_stride_count) {
        tpht_chained_resize_migrate_stride(t, stride);
    }
    if (tpht_chained_resize_active(t) &&
        atomic_load_explicit(&t->resize_done_strides, memory_order_acquire) >= t->resize_stride_count) {
        tpht_chained_resize_quiesce_and_commit(t);
    }
}

static int tpht_chained_resize_needed(tpht_table_t *t) {
    return (double)(tpht_size_load(t) + 1u) > (double)t->capacity * t->cfg.max_load_factor;
}

static void tpht_chained_resize_quiesce_and_commit(tpht_table_t *t) {
    tpht_op_exit(t);
    for (;;) {
        tpht_chained_resize_commit(t);
        if (!tpht_chained_resize_active(t)) break;
    }
    tpht_op_enter(t);
}

static tpht_status_t tpht_chained_ensure_resize(tpht_table_t *t) {
    if (!tpht_chained_resize_start(t, t->capacity * 2u)) return TPHT_NO_MEMORY;
    return TPHT_OK;
}

static tpht_status_t tpht_chained_resizable_write_fine(tpht_table_t *t, const void *key,
                                                       const void *value, int replace) {
    tpht_status_t st;
    for (;;) {
        if (tpht_chained_resize_active(t)) {
            tpht_chained_resize_finish_all(t);
            continue;
        }
        if (tpht_chained_resize_needed(t)) {
            st = tpht_chained_ensure_resize(t);
            if (st != TPHT_OK) return st;
            continue;
        }
        st = tpht_chained_write_fine(t, key, value, replace);
        if (st == TPHT_FULL) {
            st = tpht_chained_ensure_resize(t);
            if (st != TPHT_OK) return st;
            continue;
        }
        return st;
    }
}

static tpht_status_t tpht_chained_get_fine(tpht_table_t *t, const void *key, void *value_out) {
    size_t base = tpht_chained_base(t, key);
    tpht_status_t st;
retry:
    if (tpht_chained_resize_active(t)) {
        tpht_chained_resize_finish_all(t);
        goto retry;
    }
    base = tpht_chained_base(t, key);
    tpht_chain_lock_base(t, base);
    if (tpht_chained_resize_active(t)) {
        tpht_chain_unlock_base(t, base);
        goto retry;
    }
    st = tpht_chained_get_raw(t, key, value_out);
    tpht_chain_unlock_base(t, base);
    return st;
}

static tpht_status_t tpht_chained_write_fine(tpht_table_t *t, const void *key,
                                             const void *value, int replace) {
    size_t base = tpht_chained_base(t, key);
    uint8_t *prev = &t->heads[base];
    uint8_t encoded;
    uint8_t *entry;

retry:
    if (tpht_chained_resize_active(t)) {
        tpht_chained_resize_finish_all(t);
        goto retry;
    }
    base = tpht_chained_base(t, key);
    prev = &t->heads[base];
    tpht_chain_lock_base(t, base);
    if (tpht_chained_resize_active(t)) {
        tpht_chain_unlock_base(t, base);
        goto retry;
    }
    while (*prev) {
        entry = tpht_pool_deref(t, (uint64_t)(uintptr_t)prev, *prev);
        if (tpht_stored_key_equal(t, entry + 1u, key)) {
            if (!replace) {
                tpht_chain_unlock_base(t, base);
                return TPHT_EXISTS;
            }
            memcpy(entry + 1u + t->key_quotient_size, value, t->value_size);
            tpht_chain_unlock_base(t, base);
            return TPHT_OK;
        }
        prev = entry;
    }

    if (!tpht_size_try_reserve(t)) {
        tpht_chain_unlock_base(t, base);
        return TPHT_FULL;
    }

    entry = tpht_pool_alloc(t, (uint64_t)(uintptr_t)prev, &encoded);
    if (!entry) {
        tpht_size_dec(t);
        tpht_chain_unlock_base(t, base);
        return TPHT_FULL;
    }

    *prev = encoded;
    entry[0] = 0;
    tpht_write_quotient(t, entry + 1u, key);
    memcpy(entry + 1u + t->key_quotient_size, value, t->value_size);
    tpht_chain_unlock_base(t, base);
    return TPHT_OK;
}

static tpht_status_t tpht_chained_update_fine(tpht_table_t *t, const void *key,
                                              const void *value) {
    size_t base = tpht_chained_base(t, key);
    uint8_t *prev = &t->heads[base];
retry:
    if (tpht_chained_resize_active(t)) {
        tpht_chained_resize_finish_all(t);
        goto retry;
    }
    base = tpht_chained_base(t, key);
    prev = &t->heads[base];
    tpht_chain_lock_base(t, base);
    if (tpht_chained_resize_active(t)) {
        tpht_chain_unlock_base(t, base);
        goto retry;
    }
    while (*prev) {
        uint8_t *entry = tpht_pool_deref(t, (uint64_t)(uintptr_t)prev, *prev);
        if (tpht_stored_key_equal(t, entry + 1u, key)) {
            memcpy(entry + 1u + t->key_quotient_size, value, t->value_size);
            tpht_chain_unlock_base(t, base);
            return TPHT_OK;
        }
        prev = entry;
    }
    tpht_chain_unlock_base(t, base);
    return TPHT_NOT_FOUND;
}

static tpht_status_t tpht_chained_remove_fine(tpht_table_t *t, const void *key) {
    size_t base = tpht_chained_base(t, key);
    uint8_t *prev = &t->heads[base];
    uint8_t *target = NULL;
    uint8_t *last_prev = NULL;
    uint8_t *last_entry = NULL;
    uint8_t last_encoded = 0;

retry:
    if (tpht_chained_resize_active(t)) {
        tpht_chained_resize_finish_all(t);
        goto retry;
    }
    base = tpht_chained_base(t, key);
    prev = &t->heads[base];
    target = NULL;
    last_prev = NULL;
    last_entry = NULL;
    last_encoded = 0;
    tpht_chain_lock_base(t, base);
    if (tpht_chained_resize_active(t)) {
        tpht_chain_unlock_base(t, base);
        goto retry;
    }
    while (*prev) {
        uint8_t encoded = *prev;
        uint8_t *entry = tpht_pool_deref(t, (uint64_t)(uintptr_t)prev, encoded);
        if (tpht_stored_key_equal(t, entry + 1u, key)) target = entry;
        last_prev = prev;
        last_entry = entry;
        last_encoded = encoded;
        prev = entry;
    }
    if (!target) {
        tpht_chain_unlock_base(t, base);
        return TPHT_NOT_FOUND;
    }

    if (target != last_entry) {
        uint8_t target_next = target[0];
        memcpy(target, last_entry, t->pool.entry_size);
        target[0] = target_next;
    }
    *last_prev = 0;
    tpht_pool_free(t, last_encoded, last_entry);
    tpht_size_dec(t);
    tpht_chain_unlock_base(t, base);
    return TPHT_OK;
}

static tpht_status_t tpht_resize_locked(tpht_table_t *t, size_t new_capacity) {
    tpht_table_t nt;
    size_t i;
    memset(&nt, 0, sizeof(nt));
    nt.cfg = t->cfg;
    atomic_init(&nt.size, 0);
    atomic_init(&nt.resize_active, 0);
    atomic_init(&nt.resize_next_stride, 0);
    atomic_init(&nt.resize_done_strides, 0);
    atomic_init(&nt.active_ops, 0);
    atomic_flag_clear(&nt.lock);
    atomic_flag_clear(&nt.resize_start_lock);
    atomic_flag_clear(&nt.resize_commit_lock);
    if (!tpht_alloc_storage(&nt, new_capacity)) return TPHT_NO_MEMORY;

    if (t->cfg.variant == TPHT_CHAINED) {
        for (i = 0; i < t->base_count; ++i) {
            uint8_t *prev = &t->heads[i];
            while (*prev) {
                uint8_t *entry = tpht_pool_deref(t, (uint64_t)(uintptr_t)prev, *prev);
                uint8_t rebuilt_key[8];
                tpht_status_t st;
                tpht_rebuild_key(t, i, entry + 1u, rebuilt_key);
                st = tpht_raw_insert(&nt, rebuilt_key, entry + 1u + t->key_quotient_size, 0);
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
                uint8_t rebuilt_key[8];
                tpht_status_t st;
                tpht_rebuild_key(t, i, slot, rebuilt_key);
                st = tpht_raw_insert(&nt, rebuilt_key, slot + t->key_quotient_size, 0);
                if (st != TPHT_OK) {
                    tpht_free_storage(&nt);
                    return st == TPHT_FULL ? TPHT_NO_MEMORY : st;
                }
            }
            prev = &t->heads[i];
            while (*prev) {
                uint8_t *entry = tpht_pool_deref(t, (uint64_t)i, *prev);
                uint8_t rebuilt_key[8];
                tpht_status_t st;
                tpht_rebuild_key(t, i, entry + 1u, rebuilt_key);
                st = tpht_raw_insert(&nt, rebuilt_key, entry + 1u + t->key_quotient_size, 0);
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
    tpht_size_store(t, tpht_size_load(&nt));
    t->key_size = nt.key_size;
    t->value_size = nt.value_size;
    t->key_quotient_size = nt.key_quotient_size;
    t->inline_entry_size = nt.inline_entry_size;
    t->pool_entry_size = nt.pool_entry_size;
    t->key_bits = nt.key_bits;
    t->base_bits = nt.base_bits;
    t->base_mask = nt.base_mask;
    t->base_count = nt.base_count;
    t->heads = nt.heads;
    t->flat_count = nt.flat_count;
    t->flat_fp = nt.flat_fp;
    t->flat_entries = nt.flat_entries;
    t->flat_inline_cap = nt.flat_inline_cap;
    t->pool = nt.pool;
    t->chain_locks = nt.chain_locks;
    t->pool_locks = nt.pool_locks;
    t->chain_lock_count = nt.chain_lock_count;
    t->pool_lock_count = nt.pool_lock_count;
    nt.heads = NULL;
    nt.flat_count = NULL;
    nt.flat_fp = NULL;
    nt.flat_entries = NULL;
    nt.pool.entries = NULL;
    nt.pool.cnt_head = NULL;
    nt.chain_locks = NULL;
    nt.pool_locks = NULL;
    nt.chain_lock_count = 0;
    nt.pool_lock_count = 0;
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
    c.resize_strides = 0;
    return c;
}

tpht_table_t *tpht_create(const tpht_config_t *config) {
    tpht_table_t *t;
    tpht_config_t c = config ? *config : tpht_default_config();
    if (c.variant != TPHT_CHAINED && c.variant != TPHT_FLATTEN) return NULL;
    if (!tpht_valid_key_size(c.key_size) || !tpht_valid_value_size(c.value_size)) return NULL;
    if (c.bin_size == 0) c.bin_size = TPHT_DEFAULT_BIN_SIZE;
    if (c.bin_size > 127u) return NULL;
    if (c.initial_capacity == 0) c.initial_capacity = TPHT_MIN_CAPACITY;
    if (c.max_load_factor <= 0.0 || c.max_load_factor > 1.0) c.max_load_factor = TPHT_DEFAULT_LOAD_FACTOR;
    if (c.hash_seed == 0) c.hash_seed = UINT64_C(0x243f6a8885a308d3);

    t = (tpht_table_t *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->cfg = c;
    atomic_init(&t->size, 0);
    atomic_init(&t->resize_active, 0);
    atomic_init(&t->resize_next_stride, 0);
    atomic_init(&t->resize_done_strides, 0);
    atomic_init(&t->active_ops, 0);
    atomic_flag_clear(&t->lock);
    atomic_flag_clear(&t->resize_start_lock);
    atomic_flag_clear(&t->resize_commit_lock);
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
    if (tpht_chained_fine_grained(table)) {
        tpht_op_enter(table);
        st = table->cfg.resize_mode == TPHT_RESIZABLE
                 ? tpht_chained_resizable_write_fine(table, key, value, 1)
                 : tpht_chained_write_fine(table, key, value, 1);
        tpht_op_exit(table);
        return st;
    }
    tpht_lock(table);
    if (table->cfg.resize_mode == TPHT_FIXED && tpht_size_load(table) >= table->capacity &&
        tpht_raw_get(table, key, NULL) != TPHT_OK) {
        tpht_unlock(table);
        return TPHT_FULL;
    }
    if (table->cfg.resize_mode == TPHT_RESIZABLE &&
        (double)(tpht_size_load(table) + 1u) > (double)table->capacity * table->cfg.max_load_factor) {
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
    if (tpht_chained_fine_grained(table)) {
        tpht_op_enter(table);
        st = table->cfg.resize_mode == TPHT_RESIZABLE
                 ? tpht_chained_resizable_write_fine(table, key, value, 0)
                 : tpht_chained_write_fine(table, key, value, 0);
        tpht_op_exit(table);
        return st;
    }
    tpht_lock(table);
    if (table->cfg.resize_mode == TPHT_FIXED && tpht_size_load(table) >= table->capacity) {
        st = tpht_raw_get(table, key, NULL);
        tpht_unlock(table);
        return st == TPHT_OK ? TPHT_EXISTS : TPHT_FULL;
    }
    if (table->cfg.resize_mode == TPHT_RESIZABLE &&
        (double)(tpht_size_load(table) + 1u) > (double)table->capacity * table->cfg.max_load_factor) {
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
    if (tpht_chained_fine_grained(table)) {
        tpht_op_enter(table);
        st = tpht_chained_update_fine(table, key, value);
        tpht_op_exit(table);
        return st;
    }
    tpht_lock(table);
    st = tpht_raw_get(table, key, old_value);
    if (st == TPHT_OK) st = tpht_raw_insert(table, key, value, 1);
    tpht_unlock(table);
    return st;
}

tpht_status_t tpht_get(tpht_table_t *table, const void *key, void *value_out) {
    tpht_status_t st;
    if (!table || !key || !value_out) return TPHT_INVALID;
    if (tpht_chained_fine_grained(table)) {
        tpht_op_enter(table);
        st = tpht_chained_get_fine(table, key, value_out);
        tpht_op_exit(table);
        return st;
    }
    tpht_lock(table);
    st = tpht_raw_get(table, key, value_out);
    tpht_unlock(table);
    return st;
}

tpht_status_t tpht_remove(tpht_table_t *table, const void *key) {
    tpht_status_t st;
    if (!table || !key) return TPHT_INVALID;
    if (tpht_chained_fine_grained(table)) {
        tpht_op_enter(table);
        st = tpht_chained_remove_fine(table, key);
        tpht_op_exit(table);
        return st;
    }
    tpht_lock(table);
    st = tpht_raw_remove(table, key);
    tpht_unlock(table);
    return st;
}

size_t tpht_size(const tpht_table_t *table) { return table ? tpht_size_load(table) : 0; }
size_t tpht_capacity(const tpht_table_t *table) { return table ? table->capacity : 0; }
size_t tpht_memory_bytes(const tpht_table_t *table) {
    size_t bytes;
    if (!table) return 0;
    bytes = sizeof(*table);
    bytes += table->base_count; /* heads */
    bytes += table->pool.bin_count * (size_t)table->pool.bin_size * table->pool.entry_size;
    bytes += table->pool.bin_count * 2u; /* pool cnt/head */
    bytes += table->chain_lock_count * sizeof(*table->chain_locks);
    bytes += table->pool_lock_count * sizeof(*table->pool_locks);
    if (table->cfg.variant == TPHT_FLATTEN) {
        bytes += table->base_count; /* flat_count */
        bytes += table->base_count * (size_t)table->flat_inline_cap; /* fingerprints */
        bytes += table->base_count * (size_t)table->flat_inline_cap * table->inline_entry_size;
    }
    return bytes;
}
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

tpht_table_t *chained_tpht_fixed_create(size_t c, uint8_t k, uint8_t v) { return tpht_make(c, k, v, TPHT_CHAINED, TPHT_SEQUENTIAL, TPHT_FIXED); }
tpht_table_t *chained_tpht_resizable_create(size_t c, uint8_t k, uint8_t v) { return tpht_make(c, k, v, TPHT_CHAINED, TPHT_SEQUENTIAL, TPHT_RESIZABLE); }
tpht_table_t *chained_tpht_concurrent_fixed_create(size_t c, uint8_t k, uint8_t v) { return tpht_make(c, k, v, TPHT_CHAINED, TPHT_CONCURRENT, TPHT_FIXED); }
tpht_table_t *chained_tpht_concurrent_resizable_create(size_t c, uint8_t k, uint8_t v) { return tpht_make(c, k, v, TPHT_CHAINED, TPHT_CONCURRENT, TPHT_RESIZABLE); }
tpht_table_t *flatten_tpht_fixed_create(size_t c, uint8_t k, uint8_t v) { return tpht_make(c, k, v, TPHT_FLATTEN, TPHT_SEQUENTIAL, TPHT_FIXED); }
tpht_table_t *flatten_tpht_resizable_create(size_t c, uint8_t k, uint8_t v) { return tpht_make(c, k, v, TPHT_FLATTEN, TPHT_SEQUENTIAL, TPHT_RESIZABLE); }
tpht_table_t *flatten_tpht_concurrent_fixed_create(size_t c, uint8_t k, uint8_t v) { return tpht_make(c, k, v, TPHT_FLATTEN, TPHT_CONCURRENT, TPHT_FIXED); }
tpht_table_t *flatten_tpht_concurrent_resizable_create(size_t c, uint8_t k, uint8_t v) { return tpht_make(c, k, v, TPHT_FLATTEN, TPHT_CONCURRENT, TPHT_RESIZABLE); }

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
