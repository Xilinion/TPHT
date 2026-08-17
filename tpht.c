/*
 * SPDX-FileCopyrightText: 2026 Xilin Tang and TPHT contributors
 *
 * TPHT is an independent industrial C implementation inspired by the TinyPtr
 * hash-table designs. This file also embeds a compact XXH64 implementation;
 * see the local xxHash attribution comment and README.md for details.
 */

#include "tpht.h"

/*
 * Configuration is internal now: every public entry point knows its variant and
 * key width at compile time, so nothing here is ever consulted on a hot path.
 */
typedef enum tpht_variant { TPHT_CHAINED = 1, TPHT_FLATTEN = 2 } tpht_variant_t;
typedef enum tpht_threading { TPHT_SEQUENTIAL = 0, TPHT_CONCURRENT = 1 } tpht_threading_t;

typedef struct tpht_config {
    tpht_variant_t variant;
    tpht_threading_t threading;
    tpht_resize_mode_t resize_mode;
    size_t initial_capacity;
    uint8_t key_size;
    uint8_t value_size;
    uint8_t bin_size;
    double max_load_factor;
    uint64_t hash_seed;
    size_t resize_strides;
} tpht_config_t;

typedef struct tpht_table tpht_table_t;

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
/*
 * The lookup path takes the key width as a parameter and every entry point
 * passes a literal, which only pays off if the compiler specialises per call
 * site.  Left to its own heuristics GCC clones the insert path but not the
 * bigger lookup one, so the width stays a run-time argument and the hash
 * branch survives.  Forcing the inline makes each entry point fold it away.
 */
#define TPHT_HOT static inline __attribute__((always_inline))
/* Cold-path bodies whose stack frames must stay off the hot path. */
#define TPHT_NOINLINE __attribute__((noinline))
#else
#define TPHT_UNUSED
#define TPHT_HOT static inline
#define TPHT_NOINLINE
#endif

#define TPHT_DEFAULT_BIN_SIZE 127u
#define TPHT_DEFAULT_LOAD_FACTOR 0.85
#define TPHT_MIN_CAPACITY 16u
#define TPHT_CHAINED_DEREF_LOAD_NUM 95u
#define TPHT_CHAINED_DEREF_LOAD_DEN 100u
#define TPHT_DEFAULT_RESIZE_STRIDE_BUCKETS 64u

/* Keys are 4 or 8 bytes; values are any size from 1 to 8 bytes. */
#define TPHT_MAX_KEY_BYTES 8u
#define TPHT_MAX_VALUE_BYTES 8u

/*
 * Flattened variant ("blast") geometry.
 *
 * One 64-byte cache line per quotient group, called a home block:
 *
 *   byte 0 ....................................................... byte 63
 *   [ fingerprints -> ] [ free ] [ <- tiny ptrs ][ <- crystals ][ctl][ver]
 *
 *   - fingerprints grow up from offset 0, one byte per tuple stored in or via
 *     this block, home tuples ("crystals") first and overflow tuples after.
 *   - crystals are whole quotiented key/value entries and grow down from the
 *     control byte.
 *   - tiny pointers into the dereference table are one byte each and grow down
 *     from the end of the crystal region.
 *   - control byte: number of tuples in the block, 0 to 31, in its low 5 bits.
 *     How many of them are crystals follows from that count and the entry size
 *     (see tpht_flat_crystals_for), because a block always keeps as many tuples
 *     inline as its remaining space allows.  TinyPtr instead splits this byte
 *     into a 3-bit crystal count and a 5-bit tiny pointer count, which caps
 *     inline tuples at 7 even when many more would fit.
 *   - version byte: seqlock counter, odd while a writer is inside the block.
 *     Kept as in the TinyPtr layout even though this variant is sequential.
 */
#define TPHT_FLAT_LINE_BYTES 64u
#define TPHT_FLAT_LINE_SHIFT 6u
#define TPHT_FLAT_VERSION_OFF 63u
#define TPHT_FLAT_CONTROL_OFF 62u
#define TPHT_FLAT_COUNT_MASK 0x1fu
#define TPHT_FLAT_MAX_TUPLES 31u
/* Fingerprints are one quotiented byte, so the home array needs 8 spare bits. */
#define TPHT_FLAT_FP_BITS 8u
/* Usable bytes in a block once the control and version bytes are removed. */
#define TPHT_FLAT_USABLE_BYTES (TPHT_FLAT_CONTROL_OFF)
/* Extra dereference table capacity over the expected overflow, in percent. */
#define TPHT_FLAT_DEREF_HEADROOM 50u

typedef struct tpht_pool {
    uint8_t *entries;
    uint8_t *cnt_head; /* [count, freelist-head] per bin. */
    size_t bin_count;
    uint8_t bin_size;
    size_t entry_size; /* next-byte + key + value. */
} tpht_pool_t;

typedef struct tpht_retired_storage {
    uint8_t *heads;
    void *flat_lines_raw;
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

    size_t key_size;   /* 4 or 8. */
    size_t value_size; /* 1 to 8. */
    uint64_t key_mask;
    /* XXH3 "bitflip" value for the main seed, precomputed so the lookup hash
     * does not re-derive it (seed ^ bswap32(seed) << 32) on every call. */
    uint64_t hash_bitflip;
    /* Bitflips for the two chained dereference seeds (seed + 0x100 / 0x200),
     * precomputed so the per-step chain walk does not re-derive them either. */
    uint64_t hash_bitflip_100;
    uint64_t hash_bitflip_200;
    size_t key_quotient_size;
    uint64_t quotient_mask; /* low (key_quotient_size*8) bits set. */
    size_t inline_entry_size; /* stored key + value. */
    size_t pool_entry_size;   /* dereference table entry. */
    uint8_t key_bits;
    uint8_t base_bits;
    uint64_t base_mask;

    size_t base_count; /* chained base buckets or flatten home blocks. */
    uint8_t *heads;    /* chained heads only. */

    /* Flattened variant: array of 64-byte home blocks. */
    uint8_t *flat_lines;   /* 64-byte aligned view of flat_lines_raw. */
    void *flat_lines_raw;
    uint8_t flat_entry_size; /* quotiented key bytes + value bytes. */
    uint8_t flat_qkey_bytes; /* bytes of key remainder stored per entry. */
    uint8_t flat_cloud_bits; /* bits of the quotient used as block index. */
    uint8_t flat_quot_bits;  /* flat_cloud_bits + 8 fingerprint bits. */
    uint64_t flat_cloud_mask;
    uint64_t flat_quot_mask;
    uint64_t flat_rem_mask;   /* covers flat_qkey_bytes */
    uint64_t flat_value_mask; /* covers value_size */
    /* Crystal count for each possible tuple count; see tpht_flat_crystals_for. */
    uint8_t flat_crystals[TPHT_FLAT_MAX_TUPLES + 1u];
    /* Byte offset of each crystal inside a block, indexed by crystal index. */
    uint8_t flat_crystal_off[TPHT_FLAT_MAX_TUPLES + 1u];
    /* Extra block-count doublings applied after hard overflows. */
    uint8_t flat_growth;
    /* Set when the last hard overflow came from the dereference table. */
    uint8_t flat_deref_pressure;
    /* Lower bound on dereference table slots, raised when it runs out. */
    size_t flat_deref_floor;

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

static size_t tpht_max_size(size_t a, size_t b) { return a > b ? a : b; }

static size_t tpht_size_load(const tpht_table_t *t) {
    return atomic_load_explicit((atomic_size_t *)&t->size, memory_order_acquire);
}

static void tpht_size_store(tpht_table_t *t, size_t size) {
    atomic_store_explicit(&t->size, size, memory_order_release);
}

/*
 * A read-modify-write on the size counter compiles to a lock-prefixed
 * instruction, which costs more than the rest of an insert put together.  Only
 * a table that actually has concurrent writers needs one; a sequential table
 * gets a plain load, add and store through the same atomic object.
 */
static void tpht_size_inc(tpht_table_t *t) {
    if (t->cfg.threading == TPHT_SEQUENTIAL) {
        atomic_store_explicit(
            &t->size, atomic_load_explicit(&t->size, memory_order_relaxed) + 1u,
            memory_order_relaxed);
        return;
    }
    atomic_fetch_add_explicit(&t->size, 1u, memory_order_acq_rel);
}

static void tpht_size_dec(tpht_table_t *t) {
    if (t->cfg.threading == TPHT_SEQUENTIAL) {
        atomic_store_explicit(
            &t->size, atomic_load_explicit(&t->size, memory_order_relaxed) - 1u,
            memory_order_relaxed);
        return;
    }
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

/*
 * Entry fields are read with one unaligned 64-bit load and a mask instead of a
 * byte loop.  Every array these run over carries TPHT_LOAD_SLACK spare bytes so
 * the load can always take a full word.
 */
#define TPHT_LOAD_SLACK 8u

#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
#define TPHT_LITTLE_ENDIAN (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#elif defined(_MSC_VER) || defined(_M_X64) || defined(_M_IX86) || defined(_M_ARM64)
#define TPHT_LITTLE_ENDIAN 1
#else
#define TPHT_LITTLE_ENDIAN 0
#endif

static TPHT_UNUSED uint64_t tpht_load_le(const uint8_t *p, size_t n) {
#if TPHT_LITTLE_ENDIAN
    uint64_t v;
    memcpy(&v, p, 8);
    return n >= 8u ? v : (v & ((UINT64_C(1) << (8u * n)) - 1u));
#else
    uint64_t v = 0;
    size_t i;
    for (i = 0; i < n; ++i) v |= ((uint64_t)p[i]) << (8u * i);
    return v;
#endif
}

/* Stores must not touch neighbouring bytes, so they stay exactly n wide. */
static void tpht_store_le(uint8_t *p, size_t n, uint64_t x) {
#if TPHT_LITTLE_ENDIAN
    switch (n) {
        case 8u: memcpy(p, &x, 8); return;
        case 4u: { uint32_t v = (uint32_t)x; memcpy(p, &v, 4); return; }
        case 2u: { uint16_t v = (uint16_t)x; memcpy(p, &v, 2); return; }
        case 1u: p[0] = (uint8_t)x; return;
        default: break;
    }
#endif
    {
        size_t i;
        if (n > 8u) n = 8u;
        for (i = 0; i < n; ++i) p[i] = (uint8_t)(x >> (8u * i));
    }
}

static uint64_t tpht_read_le(const void *p, size_t n) {
    return tpht_load_le((const uint8_t *)p, n);
}

static void tpht_write_le(void *p, size_t n, uint64_t x) {
    tpht_store_le((uint8_t *)p, n, x);
}

/*
 * Hashing.
 *
 * The library only ever hashes a single machine word, and the two supported key
 * widths get their own function so a 32-bit table never pays for 64-bit work.
 * Both default to XXH3, the fastest member of the xxHash family for short
 * inputs, specialised to its 4-to-8-byte path so no secret table or streaming
 * state has to be embedded.  Results are bit-identical to
 * XXH3_64bits_withSeed() over the key's little-endian bytes.
 *
 * Override either macro to swap the hash without touching anything else:
 *
 *   cc -DTPHT_HASH32(w,s)=my_hash32 -DTPHT_HASH64(w,s)=my_hash64 ...
 *
 * Algorithm: xxHash / XXH3 by Yann Collet, BSD 2-Clause licensed.
 * Project: https://github.com/Cyan4973/xxHash
 */
#define TPHT_XXH3_SECRET_08 UINT64_C(0x1cad21f72c81017c) /* kSecret bytes 8..15  */
#define TPHT_XXH3_SECRET_16 UINT64_C(0xdb979083e96dd4de) /* kSecret bytes 16..23 */
#define TPHT_XXH3_PRIME_MX2 UINT64_C(0x9fb21c651e98df25)

static TPHT_UNUSED uint64_t tpht_rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64u - r));
}

static TPHT_UNUSED uint32_t tpht_bswap32(uint32_t x) {
    return ((x << 24) & 0xff000000u) | ((x << 8) & 0x00ff0000u) |
           ((x >> 8) & 0x0000ff00u) | ((x >> 24) & 0x000000ffu);
}

/* XXH3's 4-to-8-byte path; word holds the key's little-endian bytes. */
static TPHT_UNUSED uint64_t tpht_xxh3_word(uint64_t word, uint32_t len, uint64_t seed) {
    uint64_t s = seed ^ ((uint64_t)tpht_bswap32((uint32_t)seed) << 32);
    uint64_t bitflip = (TPHT_XXH3_SECRET_08 ^ TPHT_XXH3_SECRET_16) - s;
    uint32_t input1 = (uint32_t)word;
    uint32_t input2 = len == 8u ? (uint32_t)(word >> 32) : (uint32_t)word;
    uint64_t h = ((uint64_t)input2 + ((uint64_t)input1 << 32)) ^ bitflip;

    h ^= tpht_rotl64(h, 49) ^ tpht_rotl64(h, 24);
    h *= TPHT_XXH3_PRIME_MX2;
    h ^= (h >> 35) + len;
    h *= TPHT_XXH3_PRIME_MX2;
    return h ^ (h >> 28);
}

/*
 * Same 4-to-8-byte path, but the caller supplies the bitflip value directly.
 * bitflip depends only on the seed, so it is computed once at table creation
 * (see hash_bitflip) and stored; the per-call seed shuffling disappears.
 */
TPHT_HOT uint64_t tpht_xxh3_word_bitflip(uint64_t word, uint32_t len, uint64_t bitflip) {
    uint32_t input1 = (uint32_t)word;
    uint32_t input2 = len == 8u ? (uint32_t)(word >> 32) : (uint32_t)word;
    uint64_t h = ((uint64_t)input2 + ((uint64_t)input1 << 32)) ^ bitflip;

    h ^= tpht_rotl64(h, 49) ^ tpht_rotl64(h, 24);
    h *= TPHT_XXH3_PRIME_MX2;
    h ^= (h >> 35) + len;
    h *= TPHT_XXH3_PRIME_MX2;
    return h ^ (h >> 28);
}

#ifndef TPHT_HASH32
#define TPHT_HASH32(word, seed) tpht_xxh3_word((uint64_t)(uint32_t)(word), 4u, (seed))
#endif
#ifndef TPHT_HASH64
#define TPHT_HASH64(word, seed) tpht_xxh3_word((word), 8u, (seed))
#endif

/*
 * Hash one word with the function for a given key width.  Callers on a hot path
 * pass a literal, so the branch folds away and only one hash is compiled in.
 */
TPHT_HOT uint64_t tpht_hash_w(uint64_t word, uint64_t seed, unsigned key_bytes) {
    return key_bytes == 4u ? TPHT_HASH32(word, seed) : TPHT_HASH64(word, seed);
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
    return (size_t)((tpht_xxh3_word_bitflip(quotient, t->key_size, t->hash_bitflip) ^ key_word) & t->base_mask);
}

static size_t tpht_base_from_key(const tpht_table_t *t, const void *key) {
    return tpht_base_from_word(t, tpht_key_word(t, key));
}

static void tpht_write_quotient(const tpht_table_t *t, uint8_t *dst, const void *key) {
    tpht_write_le(dst, t->key_quotient_size, tpht_key_quotient(t, key));
}

static uint64_t tpht_read_quotient(const tpht_table_t *t, const uint8_t *stored_key) {
#if TPHT_LITTLE_ENDIAN
    uint64_t v;
    memcpy(&v, stored_key, 8);
    return v & t->quotient_mask;
#else
    return tpht_read_le(stored_key, t->key_quotient_size);
#endif
}

static void tpht_rebuild_key(const tpht_table_t *t, size_t base, const uint8_t *stored_key,
                             uint8_t *key_out) {
    uint64_t quotient = tpht_read_quotient(t, stored_key);
    uint64_t low = (tpht_xxh3_word_bitflip(quotient, t->key_size, t->hash_bitflip) ^ (uint64_t)base) & t->base_mask;
    uint64_t key_word = (quotient << t->base_bits) | low;
    tpht_write_le(key_out, t->key_size, key_word);
}

static TPHT_UNUSED uint32_t tpht_low_mask(uint8_t count) {
    return count >= 32u ? UINT32_MAX : ((UINT32_C(1) << count) - 1u);
}

TPHT_HOT uint32_t tpht_fp_match_mask_scalar(const uint8_t *fps, uint8_t count, uint8_t fp) {
    uint32_t mask = 0;
    uint8_t i;
    for (i = 0; i < count; ++i) {
        if (fps[i] == fp) mask |= UINT32_C(1) << i;
    }
    return mask & tpht_low_mask(count);
}

#if defined(TPHT_X86_SIMD) && (defined(__GNUC__) || defined(__clang__))
#if defined(__AVX2__)
TPHT_HOT uint32_t tpht_fp_match_mask_avx2(const uint8_t *fps, uint8_t count, uint8_t fp) {
#else
__attribute__((target("avx2")))
static TPHT_UNUSED uint32_t tpht_fp_match_mask_avx2(const uint8_t *fps, uint8_t count, uint8_t fp) {
#endif
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
TPHT_HOT uint32_t tpht_fp_match_mask_sse2(const uint8_t *fps, uint8_t count, uint8_t fp) {
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
TPHT_HOT uint32_t tpht_fp_match_mask_neon(const uint8_t *fps, uint8_t count, uint8_t fp) {
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

TPHT_HOT uint32_t tpht_fp_match_mask(const uint8_t *fps, uint8_t count, uint8_t fp) {
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
#if defined(__AVX2__)
    /* Compiled for AVX2 already: a 256-bit compare handles every block size,
     * so there is no count>16 branch and no per-lookup CPU feature test. */
    return tpht_fp_match_mask_avx2(fps, count, fp);
#else
    if (count > 16u && tpht_cpu_has_avx2()) return tpht_fp_match_mask_avx2(fps, count, fp);
#endif
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

/*
 * Map a hash onto [0, n) without a divide.  A 64-bit modulo by a runtime
 * divisor costs about 3.8ns here; the multiply-high form costs 0.8ns and needs
 * no precomputation.  It consumes the high bits, which is where a finalised
 * XXH64 value is strongest.
 */
static size_t tpht_bin_of(uint64_t hash, size_t n) {
#if defined(__SIZEOF_INT128__)
    __extension__ typedef unsigned __int128 tpht_u128_t;
    return (size_t)(((tpht_u128_t)hash * (tpht_u128_t)n) >> 64);
#else
    return (size_t)(hash % n);
#endif
}

TPHT_HOT uint8_t *tpht_pool_deref(tpht_table_t *t, uint64_t deref_key, uint8_t ptr,
                                unsigned key_bytes) {
    uint8_t flag = (uint8_t)(ptr >> 7u);
    uint8_t pos = (uint8_t)((ptr & 0x7fu) - 1u);
    uint64_t bitflip = flag ? t->hash_bitflip_200 : t->hash_bitflip_100;
    size_t bin = tpht_bin_of(tpht_xxh3_word_bitflip(deref_key, key_bytes, bitflip),
                             t->pool.bin_count);
    return tpht_pool_entry(&t->pool, bin, pos);
}

static uint8_t *tpht_pool_alloc(tpht_table_t *t, uint64_t deref_key, uint8_t *encoded_out,
                                unsigned key_bytes) {
    size_t bin1 = tpht_bin_of(tpht_xxh3_word_bitflip(deref_key, key_bytes, t->hash_bitflip_100), t->pool.bin_count);
    size_t bin2 = tpht_bin_of(tpht_xxh3_word_bitflip(deref_key, key_bytes, t->hash_bitflip_200), t->pool.bin_count);
    uint8_t flag = 0;
    size_t bin = bin1;
    uint8_t *cnt;
    uint8_t *head;
    uint8_t pos;
    uint8_t *entry;

    tpht_pool_lock_pair(t, bin1, bin2);

    /*
     * Branchless two-choice: the comparison is a coin flip on real data, so a
     * branch here mispredicts about half the time.  bin1 == bin2 needs no
     * special case - flag 1 decodes through the second hash to the same bin.
     */
    flag = (uint8_t)(tpht_pool_count(&t->pool, bin1) > tpht_pool_count(&t->pool, bin2));
    bin = flag ? bin2 : bin1;

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
        free(r->flat_lines_raw);
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
    free(t->flat_lines_raw);
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
    t->flat_lines = NULL;
    t->flat_lines_raw = NULL;
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

/*
 * How many of a block's x tuples are stored inline.
 *
 * A block holding x tuples spends one fingerprint byte on each, entry_size
 * bytes on every inline tuple (a crystal) and one byte on every tiny pointer,
 * so it needs crystals * (entry_size - 1) + 2 * x usable bytes.  A block always
 * keeps as many tuples inline as that leaves room for, which makes the crystal
 * count a function of the tuple count alone - nothing else has to be stored.
 * The remaining x - crystals(x) tuples are the ones migrated to the
 * dereference table, which is a soft overflow.
 */
static uint8_t tpht_flat_crystals_for(uint8_t entry_size, uint32_t x) {
    /* An inline entry costs entry_size bytes but saves the tiny pointer byte. */
    uint32_t cost = entry_size > 1u ? (uint32_t)entry_size - 1u : 1u;
    uint32_t room;
    uint32_t crystals;
    if (2u * x >= TPHT_FLAT_USABLE_BYTES) return 0;
    room = TPHT_FLAT_USABLE_BYTES - 2u * x;
    crystals = room / cost;
    if (crystals > x) crystals = x;
    return (uint8_t)crystals;
}

/*
 * Average tuples per home block to aim for: as many as fit inline when nothing
 * has spilled yet.  With 8-byte keys and values this is the paper's 4 tuples
 * per 64-byte block; smaller entries pack more, up to the crystal counter's 7.
 */
static size_t tpht_flat_target_load(uint8_t entry_size) {
    size_t load = 1;
    while (load < TPHT_FLAT_MAX_TUPLES &&
           tpht_flat_crystals_for(entry_size, (uint32_t)load + 1u) == load + 1u) {
        ++load;
    }
    return load;
}

/*
 * The block index is masked out of the quotient, so the block count has to be a
 * power of two.  Pick the nearer of the two candidates rather than always
 * rounding up: rounding up alone leaves a home array up to twice as large as
 * the target, whereas the block below it only trades a little more overflow.
 */
static size_t tpht_flat_block_count(size_t wanted) {
    size_t blocks = tpht_pow2_ceil(tpht_max_size(wanted, 1));
    if (blocks > 1u && wanted * 10u < blocks * 7u) blocks >>= 1u;
    return blocks;
}

/* exp(-x) for 0 <= x <= 64, without pulling in libm. */
static double tpht_flat_exp_neg(double x) {
    double term = 1.0;
    double sum = 1.0;
    int i;
    for (i = 1; i < 128; ++i) {
        term *= x / (double)i;
        sum += term;
        if (term < sum * 1e-17) break;
    }
    return 1.0 / sum;
}

/*
 * Dereference table sizing.  Block loads are Poisson(lambda) with lambda the
 * average tuples per block, so the expected share of tuples that overflow is
 * sum_x P(x) * (x - crystals(x)) / lambda.  For 8-byte keys and values this
 * reproduces the paper's ~25%; sparser tables need far less.  Headroom covers
 * the deviation bounded by Azuma-Hoeffding plus dereference-table slack.
 */
static size_t tpht_flat_deref_slots(uint8_t entry_size, size_t capacity, size_t blocks) {
    double lambda = (double)capacity / (double)blocks;
    double p = tpht_flat_exp_neg(lambda);
    double overflow = 0.0;
    uint32_t x;
    size_t slots;

    for (x = 0; x <= TPHT_FLAT_MAX_TUPLES; ++x) {
        if (x) p *= lambda / (double)x;
        overflow += p * (double)(x - tpht_flat_crystals_for(entry_size, x));
    }
    /* overflow is per block; scale to the whole table. */
    slots = (size_t)(overflow * (double)blocks) + 1u;
    slots = tpht_ceil_mul_div(slots, 100u + TPHT_FLAT_DEREF_HEADROOM, 100u);
    return tpht_max_size(slots, 1u);
}

/*
 * Touch one byte per page so the kernel maps every page at create time.  A
 * calloc of many megabytes is lazy zero pages, and without this each first
 * write during an insert pays a page fault - a latency spike in the middle of
 * the workload, and a systematic handicap in any insert benchmark against an
 * implementation that populates its arena up front.
 */
static void tpht_prefault(uint8_t *p, size_t bytes) {
    size_t off;
    if (!p) return;
    for (off = 0; off < bytes; off += 4096u) ((volatile uint8_t *)p)[off] = p[off];
    if (bytes) ((volatile uint8_t *)p)[bytes - 1u] = p[bytes - 1u];
}

static int tpht_alloc_flat_lines(tpht_table_t *t) {
    size_t bytes = t->base_count * (size_t)TPHT_FLAT_LINE_BYTES;
    uint8_t *raw;
    uintptr_t addr;

    if (bytes / TPHT_FLAT_LINE_BYTES != t->base_count) return 0;
    raw = (uint8_t *)calloc(bytes + (TPHT_FLAT_LINE_BYTES - 1u) + TPHT_LOAD_SLACK, 1);
    if (!raw) return 0;
    t->flat_lines_raw = raw;
    addr = (uintptr_t)raw;
    addr = (addr + (TPHT_FLAT_LINE_BYTES - 1u)) & ~(uintptr_t)(TPHT_FLAT_LINE_BYTES - 1u);
    t->flat_lines = (uint8_t *)addr;
    return 1;
}

static int tpht_alloc_storage(tpht_table_t *t, size_t capacity) {
    size_t overflow_slots;

    t->capacity = tpht_max_size(capacity, TPHT_MIN_CAPACITY);
    t->key_size = t->cfg.key_size;
    t->value_size = t->cfg.value_size;
    t->key_bits = (uint8_t)(t->key_size * 8u);
    t->key_mask = t->key_bits >= 64u ? UINT64_MAX : ((UINT64_C(1) << t->key_bits) - 1u);
    t->hash_bitflip =
        (TPHT_XXH3_SECRET_08 ^ TPHT_XXH3_SECRET_16) -
        (t->cfg.hash_seed ^ ((uint64_t)tpht_bswap32((uint32_t)t->cfg.hash_seed) << 32));
    t->hash_bitflip_100 =
        (TPHT_XXH3_SECRET_08 ^ TPHT_XXH3_SECRET_16) -
        ((t->cfg.hash_seed + UINT64_C(0x100)) ^
         ((uint64_t)tpht_bswap32((uint32_t)(t->cfg.hash_seed + UINT64_C(0x100))) << 32));
    t->hash_bitflip_200 =
        (TPHT_XXH3_SECRET_08 ^ TPHT_XXH3_SECRET_16) -
        ((t->cfg.hash_seed + UINT64_C(0x200)) ^
         ((uint64_t)tpht_bswap32((uint32_t)(t->cfg.hash_seed + UINT64_C(0x200))) << 32));
    t->pool.bin_size = t->cfg.bin_size ? t->cfg.bin_size : TPHT_DEFAULT_BIN_SIZE;
    if (t->pool.bin_size == 0 || t->pool.bin_size > 127u) return 0;

    if (t->cfg.variant == TPHT_CHAINED) {
        t->base_count = tpht_pow2_ceil(t->capacity);
        t->base_bits = tpht_log2_pow2(t->base_count);
        if (t->base_bits > t->key_bits) t->base_bits = t->key_bits;
        t->base_mask = ((UINT64_C(1) << t->base_bits) - 1u);
        t->key_quotient_size = (size_t)((t->key_bits - t->base_bits + 7u) / 8u);
        t->quotient_mask = t->key_quotient_size >= 8u
                               ? UINT64_MAX
                               : ((UINT64_C(1) << (8u * t->key_quotient_size)) - 1u);
        t->inline_entry_size = t->key_quotient_size + t->value_size;
        /* Chained entries carry the next tiny pointer in their first byte. */
        t->pool_entry_size = 1u + t->inline_entry_size;
        t->pool.entry_size = t->pool_entry_size;
        t->heads = (uint8_t *)calloc(t->base_count, 1);
        overflow_slots = tpht_ceil_mul_div(t->capacity, TPHT_CHAINED_DEREF_LOAD_DEN,
                                           TPHT_CHAINED_DEREF_LOAD_NUM);
        if (!t->heads) {
            tpht_free_storage(t);
            return 0;
        }
    } else {
        /*
         * The entry size sets how many tuples a block should hold on average,
         * and the resulting block count sets how much of the key is quotiented
         * away, which in turn sets the entry size.  Settle that in a few
         * rounds; any fixed point is workable because the dereference table is
         * sized from the block count actually chosen.
         */
        uint8_t entry_size = (uint8_t)(t->key_size + t->value_size);
        uint32_t x;
        int round;
        for (round = 0; round < 4; ++round) {
            size_t target = tpht_flat_target_load(entry_size);
            size_t blocks = tpht_flat_block_count((t->capacity + target - 1u) / target);
            uint8_t cloud_bits;
            uint8_t quot_bits;
            uint8_t next;
            /* Hard overflows are absorbed by giving the table more blocks. */
            if (t->flat_growth) {
                if (blocks > ((size_t)-1) >> t->flat_growth) blocks = ((size_t)-1) >> 1u;
                blocks <<= t->flat_growth;
            }
            cloud_bits = tpht_log2_pow2(blocks);
            /* The fingerprint needs a whole quotiented byte of its own. */
            if (cloud_bits + TPHT_FLAT_FP_BITS > t->key_bits) {
                cloud_bits = (uint8_t)(t->key_bits - TPHT_FLAT_FP_BITS);
                blocks = (size_t)1u << cloud_bits;
            }
            quot_bits = (uint8_t)(cloud_bits + TPHT_FLAT_FP_BITS);
            next = (uint8_t)(((t->key_bits - quot_bits + 7u) / 8u) + t->value_size);
            t->base_count = blocks;
            t->flat_cloud_bits = cloud_bits;
            t->flat_quot_bits = quot_bits;
            if (next == entry_size) break;
            entry_size = next;
        }
        t->flat_cloud_mask = ((UINT64_C(1) << t->flat_cloud_bits) - 1u);
        t->flat_quot_mask = t->flat_quot_bits >= 64u
                                ? UINT64_MAX
                                : ((UINT64_C(1) << t->flat_quot_bits) - 1u);
        t->flat_qkey_bytes = (uint8_t)((t->key_bits - t->flat_quot_bits + 7u) / 8u);
        t->flat_entry_size = (uint8_t)(t->flat_qkey_bytes + t->value_size);
        t->flat_rem_mask = t->flat_qkey_bytes >= 8u
                               ? UINT64_MAX
                               : ((UINT64_C(1) << (8u * t->flat_qkey_bytes)) - 1u);
        t->flat_value_mask = t->value_size >= 8u
                                 ? UINT64_MAX
                                 : ((UINT64_C(1) << (8u * t->value_size)) - 1u);
        /* Kept in sync so shared reporting helpers stay meaningful. */
        t->base_bits = t->flat_cloud_bits;
        t->base_mask = t->flat_cloud_mask;
        t->key_quotient_size = t->flat_qkey_bytes;
        t->quotient_mask = t->flat_rem_mask;
        t->inline_entry_size = t->flat_entry_size;
        /*
         * Overflow entries are reached by a tiny pointer held in the home
         * block, so unlike the chained variant they need no next-pointer byte.
         */
        t->pool_entry_size = t->flat_entry_size;
        t->pool.entry_size = t->pool_entry_size;
        if (!tpht_alloc_flat_lines(t)) {
            tpht_free_storage(t);
            return 0;
        }
        for (x = 0; x <= TPHT_FLAT_MAX_TUPLES; ++x) {
            uint32_t off = (x + 1u) * t->flat_entry_size;
            t->flat_crystals[x] = tpht_flat_crystals_for(t->flat_entry_size, x);
            t->flat_crystal_off[x] =
                off <= TPHT_FLAT_CONTROL_OFF ? (uint8_t)(TPHT_FLAT_CONTROL_OFF - off) : 0u;
        }
        overflow_slots = tpht_flat_deref_slots(t->flat_entry_size, t->capacity, t->base_count);
        overflow_slots = tpht_max_size(overflow_slots, t->flat_deref_floor);
    }

    t->pool.bin_count = (overflow_slots + t->pool.bin_size - 1u) / t->pool.bin_size;
    t->pool.bin_count = tpht_max_size(t->pool.bin_count, 1);
    t->pool.entries = (uint8_t *)calloc(
        t->pool.bin_count * (size_t)t->pool.bin_size * t->pool.entry_size + TPHT_LOAD_SLACK, 1);
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

    if (!t->pool.entries || !t->pool.cnt_head ||
        (t->cfg.variant == TPHT_CHAINED && t->cfg.threading == TPHT_CONCURRENT &&
         (!t->chain_locks || !t->pool_locks))) {
        tpht_free_storage(t);
        return 0;
    }

    tpht_prefault(t->heads, t->base_count);
    tpht_prefault(t->flat_lines_raw,
                  t->cfg.variant == TPHT_FLATTEN
                      ? t->base_count * (size_t)TPHT_FLAT_LINE_BYTES + TPHT_FLAT_LINE_BYTES - 1u
                      : 0u);
    tpht_prefault(t->pool.entries,
                  t->pool.bin_count * (size_t)t->pool.bin_size * t->pool.entry_size);
    tpht_prefault(t->pool.cnt_head, t->pool.bin_count * 2u);
    return 1;
}

static int tpht_stored_key_equal(tpht_table_t *t, const uint8_t *stored_key, const void *key) {
    return tpht_read_quotient(t, stored_key) == tpht_key_quotient(t, key);
}

static size_t tpht_chained_base(tpht_table_t *t, const void *key) {
    return tpht_base_from_key(t, key);
}

static tpht_status_t tpht_chained_get_raw(tpht_table_t *t, const void *key, void *value_out) {
    uint64_t key_word = tpht_key_word(t, key);
    size_t base = tpht_base_from_word(t, key_word);
    uint64_t key_quot = key_word >> t->base_bits;
    uint8_t *prev = &t->heads[base];
    while (*prev) {
        uint8_t *entry = tpht_pool_deref(t, (uint64_t)(uintptr_t)prev, *prev, t->key_size);
        if (tpht_read_quotient(t, entry + 1u) == key_quot) {
            if (value_out && t->value_size)
                tpht_store_le(value_out, t->value_size,
                              tpht_read_le(entry + 1u + t->key_quotient_size, t->value_size));
            return TPHT_OK;
        }
        prev = entry;
    }
    return TPHT_NOT_FOUND;
}

static tpht_status_t tpht_chained_insert_raw(tpht_table_t *t, const void *key,
                                             const void *value, int replace) {
    uint64_t key_word = tpht_key_word(t, key);
    size_t base = tpht_base_from_word(t, key_word);
    uint64_t key_quot = key_word >> t->base_bits;
    uint8_t *prev = &t->heads[base];
    uint8_t encoded;
    uint8_t *entry;
    if (replace) {
        while (*prev) {
            entry = tpht_pool_deref(t, (uint64_t)(uintptr_t)prev, *prev, t->key_size);
            if (tpht_read_quotient(t, entry + 1u) == key_quot) {
                if (t->value_size)
                    tpht_store_le(entry + 1u + t->key_quotient_size, t->value_size,
                                  tpht_read_le(value, t->value_size));
                return TPHT_OK;
            }
            prev = entry;
        }
    } else {
        /* append only: no existence probe, just walk to the tail */
        while (*prev) {
            entry = tpht_pool_deref(t, (uint64_t)(uintptr_t)prev, *prev, t->key_size);
            prev = entry;
        }
    }

    entry = tpht_pool_alloc(t, (uint64_t)(uintptr_t)prev, &encoded, t->key_size);
    if (!entry) return TPHT_FULL;
    *prev = encoded;
    entry[0] = 0;
#if TPHT_LITTLE_ENDIAN && defined(__SIZEOF_INT128__)
    /*
     * [quotient][value] written as two overlapping 8-byte stores instead of
     * two byte loops over runtime sizes; the second store ends exactly at the
     * entry's last byte.  Same trick as tpht_flat_write_payload.
     */
    if (t->key_quotient_size + t->value_size >= 8u) {
        __extension__ typedef unsigned __int128 tpht_u128_t;
        unsigned qv = (unsigned)(t->key_quotient_size + t->value_size);
        tpht_u128_t v = ((tpht_u128_t)tpht_read_le(value, t->value_size)
                         << (8u * t->key_quotient_size)) |
                        key_quot;
        uint64_t lo = (uint64_t)v;
        uint64_t hi = (uint64_t)(v >> (8u * (qv - 8u)));
        memcpy(entry + 1u, &lo, 8);
        memcpy(entry + 1u + qv - 8u, &hi, 8);
    } else
#endif
    {
        tpht_write_quotient(t, entry + 1u, key);
        if (t->value_size)
            tpht_store_le(entry + 1u + t->key_quotient_size, t->value_size,
                          tpht_read_le(value, t->value_size));
    }
    tpht_size_inc(t);
    return TPHT_OK;
}

static tpht_status_t tpht_chained_remove_raw(tpht_table_t *t, const void *key) {
    uint64_t key_word = tpht_key_word(t, key);
    size_t base = tpht_base_from_word(t, key_word);
    uint64_t key_quot = key_word >> t->base_bits;
    uint8_t *prev = &t->heads[base];
    uint8_t *target = NULL;
    uint8_t *last_prev = NULL;
    uint8_t *last_entry = NULL;
    uint8_t last_encoded = 0;
    while (*prev) {
        uint8_t encoded = *prev;
        uint8_t *entry = tpht_pool_deref(t, (uint64_t)(uintptr_t)prev, encoded, t->key_size);
        if (tpht_read_quotient(t, entry + 1u) == key_quot) {
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

/* ---------------------------------------------------------------------------
 * Flattened variant.
 *
 * Every quotient group owns one 64-byte home block.  A lookup reads that single
 * cache line, matches the query fingerprint against the whole fingerprint array
 * with one SIMD compare, and only then touches memory again - at most once, for
 * an overflow entry addressed by a tiny pointer that also lives in the line.
 * There is no chaining anywhere in this variant.
 *
 * A block stores only its tuple count; how many of those tuples are inline
 * follows from the count and the entry size, because the block always keeps as
 * many inline as its space allows.  Insertion therefore reduces to comparing
 * crystals(x) with crystals(x + 1):
 *
 *   crystals(x+1) == crystals(x) + 1   the new tuple fits inline
 *   crystals(x+1) <= crystals(x)       the new tuple, and any inline tuple the
 *                                      block can no longer afford, migrate to
 *                                      the dereference table - a soft overflow
 *
 * A hard overflow is either a block that cannot address another tuple at all
 * (31 of them, the counter's limit) or a dereference table that cannot hand out
 * an entry.  Both are handled by rebuilding the table with more blocks; they
 * are never reported to the caller.
 * ------------------------------------------------------------------------- */

typedef struct tpht_flat_loc {
    size_t block;  /* home block index, the low part of the quotient. */
    uint64_t rem;  /* key remainder that an entry actually stores. */
    uint8_t fp;    /* fingerprint, the high byte of the quotient. */
} tpht_flat_loc_t;

typedef struct tpht_flat_slot {
    uint8_t *payload; /* [remainder][value] */
    uint8_t fp_index; /* index into the block's fingerprint array. */
    int is_crystal;   /* stored in the line, as opposed to behind a tiny ptr. */
    uint8_t tiny_ptr; /* encoded tiny pointer when !is_crystal. */
} tpht_flat_slot_t;

/*
 * One-round Feistel quotient (see the paper's Section "Quotienting"): the low
 * flat_quot_bits of h(remainder) ^ key form the quotient, whose low bits index
 * the home block and whose high byte is the fingerprint.  The transform is
 * invertible, so those bits never have to be stored.
 */
TPHT_HOT void tpht_flat_locate(const tpht_table_t *t, uint64_t key, tpht_flat_loc_t *out,
                             unsigned key_bytes) {
    uint64_t rem = t->flat_quot_bits >= 64u ? 0u : (key >> t->flat_quot_bits);
    uint64_t quot = (tpht_xxh3_word_bitflip(rem, key_bytes, t->hash_bitflip) ^ key) & t->flat_quot_mask;
    out->rem = rem;
    out->fp = (uint8_t)(quot >> t->flat_cloud_bits);
    out->block = (size_t)(quot & t->flat_cloud_mask);
}

/* Inverse of tpht_flat_locate, used when the table is rebuilt. */
static uint64_t tpht_flat_rebuild_key(const tpht_table_t *t, size_t block, uint8_t fp,
                                      uint64_t rem, unsigned key_bytes) {
    uint64_t quot = ((uint64_t)fp << t->flat_cloud_bits) | (uint64_t)block;
    uint64_t low = (tpht_xxh3_word_bitflip(rem, key_bytes, t->hash_bitflip) ^ quot) & t->flat_quot_mask;
    return t->flat_quot_bits >= 64u ? low : (low | (rem << t->flat_quot_bits));
}

static uint64_t tpht_flat_deref_key(size_t block, uint8_t fp) {
    return ((uint64_t)block << 8u) | (uint64_t)fp;
}

TPHT_HOT uint8_t *tpht_flat_line(tpht_table_t *t, size_t block) {
    return t->flat_lines + (block << TPHT_FLAT_LINE_SHIFT);
}

TPHT_HOT uint8_t tpht_flat_count(const uint8_t *line) {
    return (uint8_t)(line[TPHT_FLAT_CONTROL_OFF] & TPHT_FLAT_COUNT_MASK);
}

static void tpht_flat_set_count(uint8_t *line, uint8_t count) {
    line[TPHT_FLAT_CONTROL_OFF] = (uint8_t)(count & TPHT_FLAT_COUNT_MASK);
}

/*
 * Crystal count for a block's current tuple count.  It is re-derived from the
 * count via the descriptor's small flat_crystals table; that table is a hot L1
 * line, so the load resolves while the home block is still on its way from L2,
 * unlike a count cached inside the block itself, which would have to wait on it.
 */
TPHT_HOT uint8_t tpht_flat_crystals(const tpht_table_t *t, const uint8_t *line) {
    return t->flat_crystals[tpht_flat_count(line)];
}

/*
 * No writer fence is needed: this variant is sequential only, so the seqlock
 * version byte (TPHT_FLAT_VERSION_OFF) is left unused.
 */
static void tpht_flat_write_begin(uint8_t *line) { (void)line; }
static void tpht_flat_write_end(uint8_t *line) { (void)line; }

/* First byte past the crystal region, and the anchor of the tiny pointers. */
TPHT_HOT uint8_t tpht_flat_crystal_end(const tpht_table_t *t, uint8_t crystals) {
    return crystals ? t->flat_crystal_off[crystals - 1u] : (uint8_t)TPHT_FLAT_CONTROL_OFF;
}

TPHT_HOT uint8_t *tpht_flat_crystal(const tpht_table_t *t, uint8_t *line, uint8_t i) {
    return line + t->flat_crystal_off[i];
}

static uint8_t *tpht_flat_tp_slot(uint8_t *line, uint8_t crystal_end, uint8_t j) {
    return line + crystal_end - j - 1u;
}

TPHT_HOT uint64_t tpht_flat_read_rem(const tpht_table_t *t, const uint8_t *payload) {
#if TPHT_LITTLE_ENDIAN
    uint64_t v;
    memcpy(&v, payload, 8);
    return v & t->flat_rem_mask;
#else
    return tpht_load_le(payload, t->flat_qkey_bytes);
#endif
}

TPHT_HOT uint64_t tpht_flat_read_value(const tpht_table_t *t, const uint8_t *payload) {
#if TPHT_LITTLE_ENDIAN
    uint64_t v;
    memcpy(&v, payload + t->flat_qkey_bytes, 8);
    return v & t->flat_value_mask;
#else
    return tpht_load_le(payload + t->flat_qkey_bytes, t->value_size);
#endif
}

TPHT_HOT void tpht_flat_write_value(const tpht_table_t *t, uint8_t *payload, uint64_t value) {
#if TPHT_LITTLE_ENDIAN
    /* Merge the value into the entry's last 8 bytes with one load and one
     * store; a byte loop over a runtime value_size costs more than the whole
     * write.  Needs the entry to span at least 8 bytes. */
    if (t->value_size && t->flat_entry_size >= 8u) {
        unsigned shift = 8u * (8u - t->value_size);
        uint64_t w;
        memcpy(&w, payload + t->flat_entry_size - 8u, 8);
        w = (w & ~(t->flat_value_mask << shift)) | (value << shift);
        memcpy(payload + t->flat_entry_size - 8u, &w, 8);
        return;
    }
#endif
    tpht_store_le(payload + t->flat_qkey_bytes, t->value_size, value);
}

TPHT_HOT void tpht_flat_write_payload(const tpht_table_t *t, uint8_t *payload, uint64_t rem,
                                    uint64_t value) {
#if TPHT_LITTLE_ENDIAN && defined(__SIZEOF_INT128__)
    /*
     * [remainder][value] written as two overlapping 8-byte stores instead of
     * two byte loops over runtime sizes.  The second store ends exactly at the
     * entry's last byte, so nothing beyond the entry is touched.
     */
    if (t->flat_entry_size >= 8u) {
        __extension__ typedef unsigned __int128 tpht_u128_t;
        tpht_u128_t v = ((tpht_u128_t)value << (8u * t->flat_qkey_bytes)) | rem;
        uint64_t lo = (uint64_t)v;
        uint64_t hi = (uint64_t)(v >> (8u * (t->flat_entry_size - 8u)));
        memcpy(payload, &lo, 8);
        memcpy(payload + t->flat_entry_size - 8u, &hi, 8);
        return;
    }
#endif
    tpht_store_le(payload, t->flat_qkey_bytes, rem);
    tpht_store_le(payload + t->flat_qkey_bytes, t->value_size, value);
}

/*
 * Fingerprint array order is crystals first, then overflow tuples, so a
 * fingerprint index below the crystal count names a crystal and any higher
 * index names tiny pointer (index - crystals).
 */
static int tpht_flat_find(tpht_table_t *t, uint8_t *line, const tpht_flat_loc_t *loc,
                          tpht_flat_slot_t *out, unsigned key_bytes) {
    uint8_t count = tpht_flat_count(line);
    uint8_t crystals;
    uint8_t crystal_end;
    uint32_t mask;

    crystals = tpht_flat_crystals(t, line);
    crystal_end = tpht_flat_crystal_end(t, crystals);
    mask = tpht_fp_match_mask(line, count, loc->fp);

    while (mask) {
        uint8_t i = tpht_ctz32(mask);
        mask &= mask - 1u;
        if (i < crystals) {
            uint8_t *payload = tpht_flat_crystal(t, line, i);
            if (tpht_flat_read_rem(t, payload) == loc->rem) {
                out->payload = payload;
                out->fp_index = i;
                out->is_crystal = 1;
                out->tiny_ptr = 0;
                return 1;
            }
        } else {
            uint8_t encoded = *tpht_flat_tp_slot(line, crystal_end, (uint8_t)(i - crystals));
            uint8_t *payload =
                tpht_pool_deref(t, tpht_flat_deref_key(loc->block, loc->fp), encoded, key_bytes);
            if (tpht_flat_read_rem(t, payload) == loc->rem) {
                out->payload = payload;
                out->fp_index = i;
                out->is_crystal = 0;
                out->tiny_ptr = encoded;
                return 1;
            }
        }
    }
    return 0;
}

/*
 * Pull overflow tuples back into the line until the layout matches the crystal
 * count the new tuple count calls for.  Fingerprint indices are unaffected: the
 * promoted tuple already owns the first overflow fingerprint, which is exactly
 * the slot the new crystal takes.
 */
static void tpht_flat_promote(tpht_table_t *t, uint8_t *line, size_t block, uint8_t from,
                              uint8_t to, unsigned key_bytes) {
    uint8_t crystals = from;
    uint8_t count = tpht_flat_count(line);
    while (crystals < to) {
        uint8_t tps = (uint8_t)(count - crystals);
        uint8_t crystal_end = tpht_flat_crystal_end(t, crystals);
        uint8_t new_end = (uint8_t)(crystal_end - t->flat_entry_size);
        uint8_t encoded = *tpht_flat_tp_slot(line, crystal_end, 0u);
        uint8_t *payload =
            tpht_pool_deref(t, tpht_flat_deref_key(block, line[crystals]), encoded, key_bytes);
        uint8_t buffer[TPHT_FLAT_LINE_BYTES];

        memcpy(buffer, payload, t->flat_entry_size);
        tpht_pool_free(t, encoded, payload);
        /* Remaining tiny pointers keep their order below the new crystal. */
        if (tps > 1u) {
            memmove(line + new_end - (tps - 1u), line + crystal_end - tps, (size_t)(tps - 1u));
        }
        memcpy(line + new_end, buffer, t->flat_entry_size);
        ++crystals;
    }
}

/*
 * Lookups run their own scan rather than tpht_flat_find: they need the value,
 * not the slot descriptor that insert and remove use to edit the block, and
 * building that descriptor costs more than the comparison itself.
 */
/* value_out must not be NULL; internal probes pass a scratch slot. */
TPHT_HOT tpht_status_t tpht_flat_get_raw(tpht_table_t *t, uint64_t key, uint64_t *value_out,
                                       unsigned key_bytes) {
    tpht_flat_loc_t loc;
    uint8_t *line;
    uint8_t count;
    uint8_t crystals;
    uint8_t crystal_end;
    uint32_t mask;
    uint32_t tp_mask;

    tpht_flat_locate(t, key, &loc, key_bytes);
    line = tpht_flat_line(t, loc.block);
    count = tpht_flat_count(line);
    crystals = tpht_flat_crystals(t, line);
    mask = tpht_fp_match_mask(line, count, loc.fp);
    crystal_end = tpht_flat_crystal_end(t, crystals);

    /*
     * Split the match mask once instead of asking "is this one inline?" per
     * candidate.  Fingerprints are ordered crystals first, so the low
     * `crystals` bits are the inline matches and the rest are overflow ones.
     * That keeps the common loop free of a data-dependent branch and keeps the
     * dereference out of it entirely.
     */
    tp_mask = mask >> crystals;
    mask &= (UINT32_C(1) << crystals) - 1u;

    while (mask) {
        uint8_t i = tpht_ctz32(mask);
        const uint8_t *payload = tpht_flat_crystal(t, line, i);
        mask &= mask - 1u;
        if (tpht_flat_read_rem(t, payload) == loc.rem) {
            *value_out = tpht_flat_read_value(t, payload);
            return TPHT_OK;
        }
    }

    while (tp_mask) {
        uint8_t j = tpht_ctz32(tp_mask);
        uint8_t encoded = *tpht_flat_tp_slot(line, crystal_end, j);
        const uint8_t *payload =
            tpht_pool_deref(t, tpht_flat_deref_key(loc.block, loc.fp), encoded, key_bytes);
        tp_mask &= tp_mask - 1u;
        if (tpht_flat_read_rem(t, payload) == loc.rem) {
            *value_out = tpht_flat_read_value(t, payload);
            return TPHT_OK;
        }
    }
    return TPHT_NOT_FOUND;
}

/*
 * Soft overflow, deliberately out of line: it runs for a minority of inserts,
 * but its eviction buffers would otherwise bloat every insert's stack frame
 * and register saves - a measurable cost on the majority path.
 */
TPHT_NOINLINE static tpht_status_t tpht_flat_insert_soft(tpht_table_t *t, uint8_t *line,
                                                         const tpht_flat_loc_t *locp,
                                                         uint64_t value, uint8_t count,
                                                         uint8_t crystals, uint8_t next_crystals,
                                                         uint8_t crystal_end, unsigned key_bytes) {
    tpht_flat_loc_t loc = *locp;
    {
        /*
         * The new tuple goes to the dereference table, and so
         * does every inline tuple the block can no longer afford.  All the
         * dereference entries are reserved before the line is touched, so a
         * hard overflow there leaves the block exactly as it was.
         */
        uint8_t evictions = (uint8_t)(crystals - next_crystals);
        uint8_t new_end = tpht_flat_crystal_end(t, next_crystals);
        uint8_t tps = (uint8_t)(count - crystals);
        uint8_t encoded[TPHT_FLAT_MAX_TUPLES + 1u];
        uint8_t *entries[TPHT_FLAT_MAX_TUPLES + 1u];
        uint8_t buffer[TPHT_FLAT_LINE_BYTES];
        uint8_t i;

        for (i = 0; i <= evictions; ++i) {
            /* Evicted crystal next_crystals + i keeps its fingerprint slot; the
             * new tuple appends one. */
            uint8_t fp = i < evictions ? line[next_crystals + i] : loc.fp;
            entries[i] = tpht_pool_alloc(t, tpht_flat_deref_key(loc.block, fp), &encoded[i], key_bytes);
            if (!entries[i]) {
                while (i-- > 0u) tpht_pool_free(t, encoded[i], entries[i]);
                /* Hard overflow: the dereference table is exhausted. */
                t->flat_deref_pressure = 1u;
                return TPHT_FULL;
            }
        }

        tpht_flat_write_begin(line);
        /* Copy the evicted pairs out before their bytes are reused. */
        for (i = 0; i < evictions; ++i) {
            uint8_t *victim = tpht_flat_crystal(t, line, (uint8_t)(next_crystals + i));
            memcpy(buffer, victim, t->flat_entry_size);
            memcpy(entries[i], buffer, t->flat_entry_size);
        }
        tpht_flat_write_payload(t, entries[evictions], loc.rem, value);

        /* Existing tiny pointers shift up to sit under the smaller crystal
         * region; evicted crystals take the slots they vacate. */
        if (tps) memmove(line + new_end - evictions - tps, line + crystal_end - tps, tps);
        for (i = 0; i < evictions; ++i) *tpht_flat_tp_slot(line, new_end, i) = encoded[i];
        *tpht_flat_tp_slot(line, new_end, (uint8_t)(count - next_crystals)) = encoded[evictions];
        line[count] = loc.fp;
        tpht_flat_set_count(line, (uint8_t)(count + 1u));
        tpht_flat_write_end(line);
        tpht_size_inc(t);
        return TPHT_OK;
    }
}

/*
 * Force-inlined so a constant `replace` from the caller folds the existence
 * probe (and its SIMD fingerprint match setup) out of the append path.
 */
TPHT_HOT tpht_status_t tpht_flat_insert_raw(tpht_table_t *t, uint64_t key, uint64_t value,
                                            int replace, unsigned key_bytes) {
    tpht_flat_loc_t loc;
    tpht_flat_slot_t slot;
    uint8_t *line;
    uint8_t count;
    uint8_t crystals;
    uint8_t next_crystals;
    uint8_t crystal_end;

    tpht_flat_locate(t, key, &loc, key_bytes);
    line = tpht_flat_line(t, loc.block);

    /*
     * insert (replace == 0) appends unconditionally, so no existence probe is
     * needed; only an overwrite (replace != 0) must locate the existing slot.
     */
    if (replace && tpht_flat_find(t, line, &loc, &slot, key_bytes)) {
        tpht_flat_write_begin(line);
        tpht_flat_write_value(t, slot.payload, value);
        tpht_flat_write_end(line);
        return TPHT_OK;
    }

    count = tpht_flat_count(line);
    /* Hard overflow: the block cannot address another tuple. */
    if (count >= TPHT_FLAT_MAX_TUPLES) {
        t->flat_deref_pressure = 0u;
        return TPHT_FULL;
    }

    crystals = t->flat_crystals[count];
    next_crystals = t->flat_crystals[count + 1u];
    crystal_end = tpht_flat_crystal_end(t, crystals);

    if (next_crystals > crystals) {
        /* The pair itself still fits in the line. */
        uint8_t tps = (uint8_t)(count - crystals);
        uint8_t new_end = (uint8_t)(crystal_end - t->flat_entry_size);
        tpht_flat_write_begin(line);
        if (tps) {
            /* Tiny pointers stay anchored to the crystal region, and the new
             * crystal's fingerprint takes index `crystals`. */
            memmove(line + new_end - tps, line + crystal_end - tps, tps);
            memmove(line + crystals + 1u, line + crystals, tps);
        }
        line[crystals] = loc.fp;
        tpht_flat_write_payload(t, line + new_end, loc.rem, value);
        tpht_flat_set_count(line, (uint8_t)(count + 1u));
        tpht_flat_write_end(line);
        tpht_size_inc(t);
        return TPHT_OK;
    }

    return tpht_flat_insert_soft(t, line, &loc, value, count, crystals, next_crystals,
                                 crystal_end, key_bytes);
}

static tpht_status_t tpht_flat_remove_raw(tpht_table_t *t, uint64_t key, unsigned key_bytes) {
    tpht_flat_loc_t loc;
    tpht_flat_slot_t slot;
    uint8_t *line;
    uint8_t count;
    uint8_t crystals;
    uint8_t tps;
    uint8_t crystal_end;
    uint8_t left;

    tpht_flat_locate(t, key, &loc, key_bytes);
    line = tpht_flat_line(t, loc.block);
    if (!tpht_flat_find(t, line, &loc, &slot, key_bytes)) return TPHT_NOT_FOUND;

    count = tpht_flat_count(line);
    crystals = t->flat_crystals[count];
    tps = (uint8_t)(count - crystals);
    crystal_end = tpht_flat_crystal_end(t, crystals);
    left = crystals;

    tpht_flat_write_begin(line);
    if (slot.is_crystal && tps > 0u) {
        /* Refill the freed crystal with the last overflow tuple. */
        uint8_t last = (uint8_t)(count - 1u);
        uint8_t encoded = *tpht_flat_tp_slot(line, crystal_end, (uint8_t)(tps - 1u));
        uint8_t *payload =
            tpht_pool_deref(t, tpht_flat_deref_key(loc.block, line[last]), encoded, key_bytes);
        memcpy(slot.payload, payload, t->flat_entry_size);
        line[slot.fp_index] = line[last];
        tpht_pool_free(t, encoded, payload);
    } else if (slot.is_crystal) {
        /* No overflow tuples: the last crystal fills the hole. */
        uint8_t last = (uint8_t)(crystals - 1u);
        if (slot.fp_index != last) {
            memcpy(slot.payload, tpht_flat_crystal(t, line, last), t->flat_entry_size);
            line[slot.fp_index] = line[last];
        }
        left = last;
    } else {
        /* Move the last tiny pointer into the freed slot. */
        uint8_t j = (uint8_t)(slot.fp_index - crystals);
        uint8_t last = (uint8_t)(tps - 1u);
        tpht_pool_free(t, slot.tiny_ptr, slot.payload);
        if (j != last) {
            *tpht_flat_tp_slot(line, crystal_end, j) =
                *tpht_flat_tp_slot(line, crystal_end, last);
            line[slot.fp_index] = line[count - 1u];
        }
    }
    tpht_flat_set_count(line, (uint8_t)(count - 1u));
    tpht_flat_promote(t, line, loc.block, left, t->flat_crystals[count - 1u], key_bytes);
    tpht_flat_write_end(line);
    tpht_size_dec(t);
    return TPHT_OK;
}

/* Reinsert every tuple of src into dst, rebuilding the original keys. */
static tpht_status_t tpht_flat_reinsert_all(tpht_table_t *dst, tpht_table_t *src,
                                            unsigned key_bytes) {
    size_t block;
    for (block = 0; block < src->base_count; ++block) {
        uint8_t *line = tpht_flat_line(src, block);
        uint8_t count = tpht_flat_count(line);
        uint8_t crystals = src->flat_crystals[count];
        uint8_t crystal_end = tpht_flat_crystal_end(src, crystals);
        uint8_t i;
        for (i = 0; i < count; ++i) {
            uint8_t fp = line[i];
            uint8_t *payload;
            uint64_t key;
            tpht_status_t st;
            if (i < crystals) {
                payload = tpht_flat_crystal(src, line, i);
            } else {
                uint8_t encoded = *tpht_flat_tp_slot(line, crystal_end, (uint8_t)(i - crystals));
                payload = tpht_pool_deref(src, tpht_flat_deref_key(block, fp), encoded, key_bytes);
            }
            key = tpht_flat_rebuild_key(src, block, fp, tpht_flat_read_rem(src, payload), key_bytes);
            st = tpht_flat_insert_raw(dst, key, tpht_flat_read_value(src, payload), 0, key_bytes);
            if (st != TPHT_OK) return st;
        }
    }
    return TPHT_OK;
}

static void tpht_flat_adopt(tpht_table_t *t, tpht_table_t *nt) {
    tpht_free_storage(t);
    t->capacity = nt->capacity;
    tpht_size_store(t, tpht_size_load(nt));
    t->hash_bitflip = nt->hash_bitflip;
    t->hash_bitflip_100 = nt->hash_bitflip_100;
    t->hash_bitflip_200 = nt->hash_bitflip_200;
    t->key_quotient_size = nt->key_quotient_size;
    t->quotient_mask = nt->quotient_mask;
    t->inline_entry_size = nt->inline_entry_size;
    t->pool_entry_size = nt->pool_entry_size;
    t->base_bits = nt->base_bits;
    t->base_mask = nt->base_mask;
    t->base_count = nt->base_count;
    t->flat_lines = nt->flat_lines;
    t->flat_lines_raw = nt->flat_lines_raw;
    t->flat_entry_size = nt->flat_entry_size;
    t->flat_qkey_bytes = nt->flat_qkey_bytes;
    t->flat_cloud_bits = nt->flat_cloud_bits;
    t->flat_quot_bits = nt->flat_quot_bits;
    t->flat_cloud_mask = nt->flat_cloud_mask;
    t->flat_quot_mask = nt->flat_quot_mask;
    t->flat_rem_mask = nt->flat_rem_mask;
    t->flat_value_mask = nt->flat_value_mask;
    t->flat_growth = nt->flat_growth;
    t->flat_deref_floor = nt->flat_deref_floor;
    memcpy(t->flat_crystals, nt->flat_crystals, sizeof(t->flat_crystals));
    memcpy(t->flat_crystal_off, nt->flat_crystal_off, sizeof(t->flat_crystal_off));
    t->pool = nt->pool;
    nt->flat_lines = NULL;
    nt->flat_lines_raw = NULL;
    nt->pool.entries = NULL;
    nt->pool.cnt_head = NULL;
}

static void tpht_flat_init_shadow(tpht_table_t *nt, const tpht_table_t *t) {
    memset(nt, 0, sizeof(*nt));
    nt->cfg = t->cfg;
    atomic_init(&nt->size, 0);
    atomic_init(&nt->resize_active, 0);
    atomic_init(&nt->resize_next_stride, 0);
    atomic_init(&nt->resize_done_strides, 0);
    atomic_init(&nt->active_ops, 0);
    atomic_flag_clear(&nt->lock);
    atomic_flag_clear(&nt->resize_start_lock);
    atomic_flag_clear(&nt->resize_commit_lock);
}

/*
 * Rebuild the whole table into a fresh geometry and adopt it.  Each retry adds
 * another doubling of the block count, because the rebuild can itself hit a
 * hard overflow.
 */
static tpht_status_t tpht_flat_rebuild(tpht_table_t *t, size_t new_capacity,
                                       uint32_t base_growth, size_t deref_floor,
                                       unsigned key_bytes) {
    size_t prev_blocks = 0;
    size_t prev_bins = 0;
    uint32_t step;

    for (step = 0u; step <= 8u; ++step) {
        tpht_table_t nt;
        tpht_status_t st;
        if (base_growth + step > 255u) break;
        tpht_flat_init_shadow(&nt, t);
        nt.flat_growth = (uint8_t)(base_growth + step);
        nt.flat_deref_floor = deref_floor;
        if (!tpht_alloc_storage(&nt, new_capacity)) return TPHT_NO_MEMORY;
        if (step && nt.base_count == prev_blocks && nt.pool.bin_count == prev_bins) {
            /* Already at the largest geometry this key width allows. */
            tpht_free_storage(&nt);
            break;
        }
        prev_blocks = nt.base_count;
        prev_bins = nt.pool.bin_count;
        st = tpht_flat_reinsert_all(&nt, t, key_bytes);
        if (st == TPHT_OK) {
            tpht_flat_adopt(t, &nt);
            return TPHT_OK;
        }
        tpht_free_storage(&nt);
        if (st != TPHT_FULL) return st;
    }
    return TPHT_FULL;
}

/*
 * Absorb a hard overflow: rebuild with more home blocks, keeping the capacity
 * the caller asked for.  When the dereference table was the part that ran out,
 * make sure the rebuild does not hand it fewer slots than it had.
 */
static tpht_status_t tpht_flat_grow(tpht_table_t *t, unsigned key_bytes) {
    size_t slots = t->pool.bin_count * (size_t)t->pool.bin_size;
    size_t floor = t->flat_deref_pressure ? slots * 2u : slots;
    return tpht_flat_rebuild(t, t->capacity, (uint32_t)t->flat_growth + 1u, floor, key_bytes);
}

/* Rebuild at a new capacity, for resizable tables. */
static tpht_status_t tpht_flat_resize(tpht_table_t *t, size_t new_capacity, unsigned key_bytes) {
    return tpht_flat_rebuild(t, new_capacity, 0u, 0u, key_bytes);
}

static tpht_status_t tpht_resize_locked(tpht_table_t *t, size_t new_capacity);

/* ------------------------------------------------------- flattened operations
 * These never test a variant, a threading mode or a key width: the entry point
 * that calls them already knows all three.
 */
TPHT_HOT tpht_status_t tpht_flat_write(tpht_table_t *t, uint64_t key, uint64_t value, int replace,
                                       unsigned key_bytes) {
    tpht_status_t st;

    if (t->cfg.resize_mode == TPHT_FIXED) {
        if (tpht_size_load(t) >= t->capacity) {
            /*
             * insert is append-only: at capacity there is no room for another
             * entry, so it reports TPHT_FULL without probing.  Only an overwrite
             * (replace != 0) may still look the key up and touch it in place.
             */
            if (!replace) return TPHT_FULL;
            uint64_t scratch;
            st = tpht_flat_get_raw(t, key, &scratch, key_bytes);
            if (st != TPHT_OK) return TPHT_FULL;
        }
    } else if ((double)(tpht_size_load(t) + 1u) >
               (double)t->capacity * t->cfg.max_load_factor) {
        st = tpht_flat_resize(t, t->capacity * 2u, key_bytes);
        if (st != TPHT_OK) return st;
    }

    st = tpht_flat_insert_raw(t, key, value, replace, key_bytes);
    if (st == TPHT_FULL) {
        /* Hard overflow: rebuild with more blocks and retry, always. */
        st = tpht_flat_grow(t, key_bytes);
        if (st != TPHT_OK) return st;
        st = tpht_flat_insert_raw(t, key, value, replace, key_bytes);
    }
    return st;
}

static tpht_status_t tpht_flat_update_op(tpht_table_t *t, uint64_t key, uint64_t value,
                                         unsigned key_bytes) {
    uint64_t scratch;
    tpht_status_t st = tpht_flat_get_raw(t, key, &scratch, key_bytes);
    if (st != TPHT_OK) return st;
    return tpht_flat_insert_raw(t, key, value, 1, key_bytes);
}

/* --------------------------------------------------------- chained operations
 * The chained variant still branches on threading, because a chained table can
 * be sequential or concurrent; the variant and key width are compile time.
 */
static tpht_status_t tpht_chained_raw_insert(tpht_table_t *t, uint64_t key, uint64_t value,
                                             int replace) {
    uint8_t kb[TPHT_MAX_KEY_BYTES];
    uint8_t vb[TPHT_MAX_VALUE_BYTES];
    tpht_write_le(kb, t->key_size, key);
    tpht_write_le(vb, t->value_size, value);
    return tpht_chained_insert_raw(t, kb, vb, replace);
}

static tpht_status_t tpht_chained_raw_get(tpht_table_t *t, uint64_t key, uint64_t *value_out) {
    uint8_t kb[TPHT_MAX_KEY_BYTES];
    uint8_t vb[TPHT_MAX_VALUE_BYTES];
    tpht_status_t st;
    tpht_write_le(kb, t->key_size, key);
    st = tpht_chained_get_raw(t, kb, vb);
    if (st == TPHT_OK && value_out) *value_out = tpht_read_le(vb, t->value_size);
    return st;
}

static tpht_status_t tpht_chained_raw_remove(tpht_table_t *t, uint64_t key) {
    uint8_t kb[TPHT_MAX_KEY_BYTES];
    tpht_write_le(kb, t->key_size, key);
    return tpht_chained_remove_raw(t, kb);
}

static tpht_status_t tpht_chained_write_locked(tpht_table_t *t, uint64_t key, uint64_t value,
                                               int replace) {
    tpht_status_t st = tpht_chained_raw_insert(t, key, value, replace);
    if (st == TPHT_FULL && t->cfg.resize_mode == TPHT_RESIZABLE) {
        st = tpht_resize_locked(t, t->capacity * 2u);
        if (st == TPHT_OK) st = tpht_chained_raw_insert(t, key, value, replace);
    }
    return st;
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
    void *old_flat_lines_raw;
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
    old_flat_lines_raw = t->flat_lines_raw;
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
    t->quotient_mask = nt->quotient_mask;
    t->inline_entry_size = nt->inline_entry_size;
    t->pool_entry_size = nt->pool_entry_size;
    t->key_bits = nt->key_bits;
    t->base_bits = nt->base_bits;
    t->base_mask = nt->base_mask;
    t->base_count = nt->base_count;
    t->heads = nt->heads;
    t->flat_lines = nt->flat_lines;
    t->flat_lines_raw = nt->flat_lines_raw;
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
        retired->flat_lines_raw = old_flat_lines_raw;
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
            uint8_t *entry = tpht_pool_deref(t, (uint64_t)(uintptr_t)prev, *prev, t->key_size);
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
        entry = tpht_pool_deref(t, (uint64_t)(uintptr_t)prev, *prev, t->key_size);
        if (replace && tpht_stored_key_equal(t, entry + 1u, key)) {
            tpht_store_le(entry + 1u + t->key_quotient_size, t->value_size,
                          tpht_read_le(value, t->value_size));
            tpht_chain_unlock_base(t, base);
            return TPHT_OK;
        }
        prev = entry;
    }

    if (!tpht_size_try_reserve(t)) {
        tpht_chain_unlock_base(t, base);
        return TPHT_FULL;
    }

    entry = tpht_pool_alloc(t, (uint64_t)(uintptr_t)prev, &encoded, t->key_size);
    if (!entry) {
        tpht_size_dec(t);
        tpht_chain_unlock_base(t, base);
        return TPHT_FULL;
    }

    *prev = encoded;
    entry[0] = 0;
    tpht_write_quotient(t, entry + 1u, key);
    tpht_store_le(entry + 1u + t->key_quotient_size, t->value_size,
                  tpht_read_le(value, t->value_size));
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
        uint8_t *entry = tpht_pool_deref(t, (uint64_t)(uintptr_t)prev, *prev, t->key_size);
        if (tpht_stored_key_equal(t, entry + 1u, key)) {
            tpht_store_le(entry + 1u + t->key_quotient_size, t->value_size,
                          tpht_read_le(value, t->value_size));
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
        uint8_t *entry = tpht_pool_deref(t, (uint64_t)(uintptr_t)prev, encoded, t->key_size);
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

    /* Only the chained variant is resizable. */
    for (i = 0; i < t->base_count; ++i) {
        uint8_t *prev = &t->heads[i];
        while (*prev) {
            uint8_t *entry = tpht_pool_deref(t, (uint64_t)(uintptr_t)prev, *prev, t->key_size);
            uint8_t rebuilt_key[8];
            tpht_status_t st;
            tpht_rebuild_key(t, i, entry + 1u, rebuilt_key);
            st = tpht_chained_insert_raw(&nt, rebuilt_key, entry + 1u + t->key_quotient_size, 0);
            if (st != TPHT_OK) {
                tpht_free_storage(&nt);
                return st == TPHT_FULL ? TPHT_NO_MEMORY : st;
            }
            prev = entry;
        }
    }

    tpht_free_storage(t);
    t->capacity = nt.capacity;
    tpht_size_store(t, tpht_size_load(&nt));
    t->key_size = nt.key_size;
    t->value_size = nt.value_size;
    t->hash_bitflip = nt.hash_bitflip;
    t->hash_bitflip_100 = nt.hash_bitflip_100;
    t->hash_bitflip_200 = nt.hash_bitflip_200;
    t->key_quotient_size = nt.key_quotient_size;
    t->quotient_mask = nt.quotient_mask;
    t->inline_entry_size = nt.inline_entry_size;
    t->pool_entry_size = nt.pool_entry_size;
    t->key_bits = nt.key_bits;
    t->base_bits = nt.base_bits;
    t->base_mask = nt.base_mask;
    t->base_count = nt.base_count;
    t->heads = nt.heads;
    t->flat_lines = nt.flat_lines;
    t->flat_lines_raw = nt.flat_lines_raw;
    t->pool = nt.pool;
    t->chain_locks = nt.chain_locks;
    t->pool_locks = nt.pool_locks;
    t->chain_lock_count = nt.chain_lock_count;
    t->pool_lock_count = nt.pool_lock_count;
    nt.heads = NULL;
    nt.flat_lines = NULL;
    nt.flat_lines_raw = NULL;
    nt.pool.entries = NULL;
    nt.pool.cnt_head = NULL;
    nt.chain_locks = NULL;
    nt.pool_locks = NULL;
    nt.chain_lock_count = 0;
    nt.pool_lock_count = 0;
    return TPHT_OK;
}


static size_t tpht_size_of(const tpht_table_t *t) { return t ? tpht_size_load(t) : 0; }
static size_t tpht_capacity_of(const tpht_table_t *t) { return t ? t->capacity : 0; }

static size_t tpht_memory_of(const tpht_table_t *t) {
    size_t bytes;
    if (!t) return 0;
    bytes = sizeof(*t);
    bytes += t->pool.bin_count * (size_t)t->pool.bin_size * t->pool.entry_size;
    bytes += t->pool.bin_count * 2u; /* pool cnt/head */
    bytes += t->chain_lock_count * sizeof(*t->chain_locks);
    bytes += t->pool_lock_count * sizeof(*t->pool_locks);
    if (t->cfg.variant == TPHT_FLATTEN) {
        bytes += t->base_count * (size_t)TPHT_FLAT_LINE_BYTES; /* home array */
    } else {
        bytes += t->base_count; /* chained heads */
    }
    return bytes;
}

tpht_options_t tpht_default_options(void) {
    tpht_options_t o;
    o.resize_mode = TPHT_FIXED;
    o.value_size = 8;
    o.bin_size = TPHT_DEFAULT_BIN_SIZE;
    o.max_load_factor = TPHT_DEFAULT_LOAD_FACTOR;
    o.hash_seed = UINT64_C(0x243f6a8885a308d3);
    o.resize_strides = 0;
    return o;
}

static tpht_table_t *tpht_create_internal(tpht_variant_t variant, tpht_threading_t threading,
                                          uint8_t key_size, size_t capacity,
                                          const tpht_options_t *options) {
    tpht_options_t o = options ? *options : tpht_default_options();
    tpht_config_t c;
    tpht_table_t *t;

    if (o.value_size == 0u) o.value_size = 8u;
    if (o.value_size > TPHT_MAX_VALUE_BYTES) return NULL;
    if (o.bin_size == 0u) o.bin_size = TPHT_DEFAULT_BIN_SIZE;
    if (o.bin_size > 127u) return NULL;
    if (o.max_load_factor <= 0.0 || o.max_load_factor > 1.0) {
        o.max_load_factor = TPHT_DEFAULT_LOAD_FACTOR;
    }
    if (o.hash_seed == 0u) o.hash_seed = UINT64_C(0x243f6a8885a308d3);
    if (capacity == 0u) capacity = TPHT_MIN_CAPACITY;
    /* Concurrency is not implemented for the flattened variant. */
    if (variant == TPHT_FLATTEN && threading != TPHT_SEQUENTIAL) return NULL;

    c.variant = variant;
    c.threading = threading;
    c.resize_mode = o.resize_mode;
    c.initial_capacity = capacity;
    c.key_size = key_size;
    c.value_size = o.value_size;
    c.bin_size = o.bin_size;
    c.max_load_factor = o.max_load_factor;
    c.hash_seed = o.hash_seed;
    c.resize_strides = o.resize_strides;

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
    if (!tpht_alloc_storage(t, capacity)) {
        free(t);
        return NULL;
    }
    return t;
}

static tpht_table_t *tpht_named_create(tpht_variant_t variant, tpht_threading_t threading,
                                       uint8_t key_size, tpht_resize_mode_t mode,
                                       size_t capacity, uint8_t value_size) {
    tpht_options_t o = tpht_default_options();
    o.resize_mode = mode;
    o.value_size = value_size;
    return tpht_create_internal(variant, threading, key_size, capacity, &o);
}

static void tpht_destroy_internal(tpht_table_t *t) {
    if (!t) return;
    tpht_free_storage(t);
    free(t);
}


/* Chained entry points share these; threading is still a run-time property. */
static tpht_status_t tpht_chained_op_write(tpht_table_t *t, uint64_t key, uint64_t value,
                                           int replace) {
    tpht_status_t st;
    key &= t->key_mask;
    if (tpht_chained_fine_grained(t)) {
        uint8_t kb[TPHT_MAX_KEY_BYTES];
        uint8_t vb[TPHT_MAX_VALUE_BYTES];
        tpht_write_le(kb, t->key_size, key);
        tpht_write_le(vb, t->value_size, value);
        tpht_op_enter(t);
        st = t->cfg.resize_mode == TPHT_RESIZABLE
                 ? tpht_chained_resizable_write_fine(t, kb, vb, replace)
                 : tpht_chained_write_fine(t, kb, vb, replace);
        tpht_op_exit(t);
        return st;
    }
    tpht_lock(t);
    if (t->cfg.resize_mode == TPHT_FIXED && tpht_size_load(t) >= t->capacity) {
        /*
         * insert is append-only: at capacity report TPHT_FULL without probing;
         * only an overwrite (replace != 0) may look the key up.
         */
        if (!replace) {
            tpht_unlock(t);
            return TPHT_FULL;
        }
        st = tpht_chained_raw_get(t, key, NULL);
        if (st != TPHT_OK) {
            tpht_unlock(t);
            return TPHT_FULL;
        }
    }
    if (t->cfg.resize_mode == TPHT_RESIZABLE &&
        (double)(tpht_size_load(t) + 1u) > (double)t->capacity * t->cfg.max_load_factor) {
        st = tpht_resize_locked(t, t->capacity * 2u);
        if (st != TPHT_OK) {
            tpht_unlock(t);
            return st;
        }
    }
    st = tpht_chained_write_locked(t, key, value, replace);
    tpht_unlock(t);
    return st;
}

static tpht_status_t tpht_chained_op_update(tpht_table_t *t, uint64_t key, uint64_t value) {
    tpht_status_t st;
    key &= t->key_mask;
    if (tpht_chained_fine_grained(t)) {
        uint8_t kb[TPHT_MAX_KEY_BYTES];
        uint8_t vb[TPHT_MAX_VALUE_BYTES];
        tpht_write_le(kb, t->key_size, key);
        tpht_write_le(vb, t->value_size, value);
        tpht_op_enter(t);
        st = tpht_chained_update_fine(t, kb, vb);
        tpht_op_exit(t);
        return st;
    }
    tpht_lock(t);
    st = tpht_chained_raw_get(t, key, NULL);
    if (st == TPHT_OK) st = tpht_chained_raw_insert(t, key, value, 1);
    tpht_unlock(t);
    return st;
}

static tpht_status_t tpht_chained_op_get(tpht_table_t *t, uint64_t key, uint64_t *value_out) {
    tpht_status_t st;
    key &= t->key_mask;
    if (tpht_chained_fine_grained(t)) {
        uint8_t kb[TPHT_MAX_KEY_BYTES];
        uint8_t vb[TPHT_MAX_VALUE_BYTES];
        tpht_write_le(kb, t->key_size, key);
        tpht_op_enter(t);
        st = tpht_chained_get_fine(t, kb, vb);
        tpht_op_exit(t);
        if (st == TPHT_OK) *value_out = tpht_read_le(vb, t->value_size);
        return st;
    }
    tpht_lock(t);
    st = tpht_chained_raw_get(t, key, value_out);
    tpht_unlock(t);
    return st;
}

static tpht_status_t tpht_chained_op_remove(tpht_table_t *t, uint64_t key) {
    tpht_status_t st;
    key &= t->key_mask;
    if (tpht_chained_fine_grained(t)) {
        uint8_t kb[TPHT_MAX_KEY_BYTES];
        tpht_write_le(kb, t->key_size, key);
        tpht_op_enter(t);
        st = tpht_chained_remove_fine(t, kb);
        tpht_op_exit(t);
        return st;
    }
    tpht_lock(t);
    st = tpht_chained_raw_remove(t, key);
    tpht_unlock(t);
    return st;
}


/* ------------------------------------------------------------- flatten_tpht32 */

flatten_tpht32_t *flatten_tpht32_create(size_t capacity, const tpht_options_t *options) {
    return (flatten_tpht32_t *)tpht_create_internal(TPHT_FLATTEN, TPHT_SEQUENTIAL, 4, capacity, options);
}

flatten_tpht32_t *flatten_tpht32_fixed_create(size_t capacity, uint8_t value_size) {
    return (flatten_tpht32_t *)tpht_named_create(TPHT_FLATTEN, TPHT_SEQUENTIAL, 4, TPHT_FIXED, capacity,
                                    value_size);
}

flatten_tpht32_t *flatten_tpht32_resizable_create(size_t capacity, uint8_t value_size) {
    return (flatten_tpht32_t *)tpht_named_create(TPHT_FLATTEN, TPHT_SEQUENTIAL, 4, TPHT_RESIZABLE, capacity,
                                    value_size);
}

void flatten_tpht32_destroy(flatten_tpht32_t *table) { tpht_destroy_internal((tpht_table_t *)table); }

tpht_status_t flatten_tpht32_put(flatten_tpht32_t *table, uint32_t key, uint64_t value) {
    if (!table) return TPHT_INVALID;
    return tpht_flat_write((tpht_table_t *)table, key, value, 1, 4);
}

tpht_status_t flatten_tpht32_insert(flatten_tpht32_t *table, uint32_t key, uint64_t value) {
    if (!table) return TPHT_INVALID;
    return tpht_flat_write((tpht_table_t *)table, key, value, 0, 4);
}

tpht_status_t flatten_tpht32_update(flatten_tpht32_t *table, uint32_t key, uint64_t value) {
    if (!table) return TPHT_INVALID;
    return tpht_flat_update_op((tpht_table_t *)table, key, value, 4);
}

tpht_status_t flatten_tpht32_get(flatten_tpht32_t *table, uint32_t key, uint64_t *value_out) {
    if (!table || !value_out) return TPHT_INVALID;
    return tpht_flat_get_raw((tpht_table_t *)table, key, value_out, 4);
}

tpht_status_t flatten_tpht32_remove(flatten_tpht32_t *table, uint32_t key) {
    if (!table) return TPHT_INVALID;
    return tpht_flat_remove_raw((tpht_table_t *)table, key, 4);
}

size_t flatten_tpht32_size(const flatten_tpht32_t *table) { return tpht_size_of((const tpht_table_t *)table); }
size_t flatten_tpht32_capacity(const flatten_tpht32_t *table) { return tpht_capacity_of((const tpht_table_t *)table); }
size_t flatten_tpht32_memory_bytes(const flatten_tpht32_t *table) { return tpht_memory_of((const tpht_table_t *)table); }

/* ------------------------------------------------------------- flatten_tpht64 */

flatten_tpht64_t *flatten_tpht64_create(size_t capacity, const tpht_options_t *options) {
    return (flatten_tpht64_t *)tpht_create_internal(TPHT_FLATTEN, TPHT_SEQUENTIAL, 8, capacity, options);
}

flatten_tpht64_t *flatten_tpht64_fixed_create(size_t capacity, uint8_t value_size) {
    return (flatten_tpht64_t *)tpht_named_create(TPHT_FLATTEN, TPHT_SEQUENTIAL, 8, TPHT_FIXED, capacity,
                                    value_size);
}

flatten_tpht64_t *flatten_tpht64_resizable_create(size_t capacity, uint8_t value_size) {
    return (flatten_tpht64_t *)tpht_named_create(TPHT_FLATTEN, TPHT_SEQUENTIAL, 8, TPHT_RESIZABLE, capacity,
                                    value_size);
}

void flatten_tpht64_destroy(flatten_tpht64_t *table) { tpht_destroy_internal((tpht_table_t *)table); }

tpht_status_t flatten_tpht64_put(flatten_tpht64_t *table, uint64_t key, uint64_t value) {
    if (!table) return TPHT_INVALID;
    return tpht_flat_write((tpht_table_t *)table, key, value, 1, 8);
}

tpht_status_t flatten_tpht64_insert(flatten_tpht64_t *table, uint64_t key, uint64_t value) {
    if (!table) return TPHT_INVALID;
    return tpht_flat_write((tpht_table_t *)table, key, value, 0, 8);
}

tpht_status_t flatten_tpht64_update(flatten_tpht64_t *table, uint64_t key, uint64_t value) {
    if (!table) return TPHT_INVALID;
    return tpht_flat_update_op((tpht_table_t *)table, key, value, 8);
}

tpht_status_t flatten_tpht64_get(flatten_tpht64_t *table, uint64_t key, uint64_t *value_out) {
    if (!table || !value_out) return TPHT_INVALID;
    return tpht_flat_get_raw((tpht_table_t *)table, key, value_out, 8);
}

tpht_status_t flatten_tpht64_remove(flatten_tpht64_t *table, uint64_t key) {
    if (!table) return TPHT_INVALID;
    return tpht_flat_remove_raw((tpht_table_t *)table, key, 8);
}

size_t flatten_tpht64_size(const flatten_tpht64_t *table) { return tpht_size_of((const tpht_table_t *)table); }
size_t flatten_tpht64_capacity(const flatten_tpht64_t *table) { return tpht_capacity_of((const tpht_table_t *)table); }
size_t flatten_tpht64_memory_bytes(const flatten_tpht64_t *table) { return tpht_memory_of((const tpht_table_t *)table); }

/* ------------------------------------------------------------- chained_tpht32 */

chained_tpht32_t *chained_tpht32_create(size_t capacity, int concurrent, const tpht_options_t *options) {
    return (chained_tpht32_t *)tpht_create_internal(TPHT_CHAINED,
                                       concurrent ? TPHT_CONCURRENT : TPHT_SEQUENTIAL, 4,
                                       capacity, options);
}

chained_tpht32_t *chained_tpht32_fixed_create(size_t capacity, uint8_t value_size) {
    return (chained_tpht32_t *)tpht_named_create(TPHT_CHAINED, TPHT_SEQUENTIAL, 4, TPHT_FIXED, capacity,
                                    value_size);
}

chained_tpht32_t *chained_tpht32_resizable_create(size_t capacity, uint8_t value_size) {
    return (chained_tpht32_t *)tpht_named_create(TPHT_CHAINED, TPHT_SEQUENTIAL, 4, TPHT_RESIZABLE, capacity,
                                    value_size);
}

chained_tpht32_t *chained_tpht32_concurrent_fixed_create(size_t capacity, uint8_t value_size) {
    return (chained_tpht32_t *)tpht_named_create(TPHT_CHAINED, TPHT_CONCURRENT, 4, TPHT_FIXED, capacity,
                                    value_size);
}

chained_tpht32_t *chained_tpht32_concurrent_resizable_create(size_t capacity, uint8_t value_size) {
    return (chained_tpht32_t *)tpht_named_create(TPHT_CHAINED, TPHT_CONCURRENT, 4, TPHT_RESIZABLE, capacity,
                                    value_size);
}

void chained_tpht32_destroy(chained_tpht32_t *table) { tpht_destroy_internal((tpht_table_t *)table); }

tpht_status_t chained_tpht32_put(chained_tpht32_t *table, uint32_t key, uint64_t value) {
    if (!table) return TPHT_INVALID;
    return tpht_chained_op_write((tpht_table_t *)table, key, value, 1);
}

tpht_status_t chained_tpht32_insert(chained_tpht32_t *table, uint32_t key, uint64_t value) {
    if (!table) return TPHT_INVALID;
    return tpht_chained_op_write((tpht_table_t *)table, key, value, 0);
}

tpht_status_t chained_tpht32_update(chained_tpht32_t *table, uint32_t key, uint64_t value) {
    if (!table) return TPHT_INVALID;
    return tpht_chained_op_update((tpht_table_t *)table, key, value);
}

tpht_status_t chained_tpht32_get(chained_tpht32_t *table, uint32_t key, uint64_t *value_out) {
    if (!table || !value_out) return TPHT_INVALID;
    return tpht_chained_op_get((tpht_table_t *)table, key, value_out);
}

tpht_status_t chained_tpht32_remove(chained_tpht32_t *table, uint32_t key) {
    if (!table) return TPHT_INVALID;
    return tpht_chained_op_remove((tpht_table_t *)table, key);
}

size_t chained_tpht32_size(const chained_tpht32_t *table) { return tpht_size_of((const tpht_table_t *)table); }
size_t chained_tpht32_capacity(const chained_tpht32_t *table) { return tpht_capacity_of((const tpht_table_t *)table); }
size_t chained_tpht32_memory_bytes(const chained_tpht32_t *table) { return tpht_memory_of((const tpht_table_t *)table); }

/* ------------------------------------------------------------- chained_tpht64 */

chained_tpht64_t *chained_tpht64_create(size_t capacity, int concurrent, const tpht_options_t *options) {
    return (chained_tpht64_t *)tpht_create_internal(TPHT_CHAINED,
                                       concurrent ? TPHT_CONCURRENT : TPHT_SEQUENTIAL, 8,
                                       capacity, options);
}

chained_tpht64_t *chained_tpht64_fixed_create(size_t capacity, uint8_t value_size) {
    return (chained_tpht64_t *)tpht_named_create(TPHT_CHAINED, TPHT_SEQUENTIAL, 8, TPHT_FIXED, capacity,
                                    value_size);
}

chained_tpht64_t *chained_tpht64_resizable_create(size_t capacity, uint8_t value_size) {
    return (chained_tpht64_t *)tpht_named_create(TPHT_CHAINED, TPHT_SEQUENTIAL, 8, TPHT_RESIZABLE, capacity,
                                    value_size);
}

chained_tpht64_t *chained_tpht64_concurrent_fixed_create(size_t capacity, uint8_t value_size) {
    return (chained_tpht64_t *)tpht_named_create(TPHT_CHAINED, TPHT_CONCURRENT, 8, TPHT_FIXED, capacity,
                                    value_size);
}

chained_tpht64_t *chained_tpht64_concurrent_resizable_create(size_t capacity, uint8_t value_size) {
    return (chained_tpht64_t *)tpht_named_create(TPHT_CHAINED, TPHT_CONCURRENT, 8, TPHT_RESIZABLE, capacity,
                                    value_size);
}

void chained_tpht64_destroy(chained_tpht64_t *table) { tpht_destroy_internal((tpht_table_t *)table); }

tpht_status_t chained_tpht64_put(chained_tpht64_t *table, uint64_t key, uint64_t value) {
    if (!table) return TPHT_INVALID;
    return tpht_chained_op_write((tpht_table_t *)table, key, value, 1);
}

tpht_status_t chained_tpht64_insert(chained_tpht64_t *table, uint64_t key, uint64_t value) {
    if (!table) return TPHT_INVALID;
    return tpht_chained_op_write((tpht_table_t *)table, key, value, 0);
}

tpht_status_t chained_tpht64_update(chained_tpht64_t *table, uint64_t key, uint64_t value) {
    if (!table) return TPHT_INVALID;
    return tpht_chained_op_update((tpht_table_t *)table, key, value);
}

tpht_status_t chained_tpht64_get(chained_tpht64_t *table, uint64_t key, uint64_t *value_out) {
    if (!table || !value_out) return TPHT_INVALID;
    return tpht_chained_op_get((tpht_table_t *)table, key, value_out);
}

tpht_status_t chained_tpht64_remove(chained_tpht64_t *table, uint64_t key) {
    if (!table) return TPHT_INVALID;
    return tpht_chained_op_remove((tpht_table_t *)table, key);
}

size_t chained_tpht64_size(const chained_tpht64_t *table) { return tpht_size_of((const tpht_table_t *)table); }
size_t chained_tpht64_capacity(const chained_tpht64_t *table) { return tpht_capacity_of((const tpht_table_t *)table); }
size_t chained_tpht64_memory_bytes(const chained_tpht64_t *table) { return tpht_memory_of((const tpht_table_t *)table); }
