/*
 * SPDX-FileCopyrightText: 2026 Xilin Tang and TPHT contributors
 *
 * TPHT is an independent industrial C implementation inspired by the TinyPtr
 * hash-table designs. This file also embeds a compact XXH64 implementation;
 * see the local xxHash attribution comment and README.md for details.
 */

/*
 * MADV_HUGEPAGE and MADV_POPULATE_WRITE hide behind _DEFAULT_SOURCE; under a
 * strict -std=c11 build glibc leaves them undefined and the huge-page advice
 * in tpht_advise_hugepages would silently compile away.  Declared before any
 * header so it also covers whatever tpht.h pulls in.
 */
#if defined(__linux__) && !defined(_DEFAULT_SOURCE)
#define _DEFAULT_SOURCE 1
#endif

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
    double max_load_factor;
    uint64_t hash_seed;
    size_t resize_strides;
} tpht_config_t;

typedef struct tpht_table tpht_table_t;

#include <stdatomic.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#if defined(__linux__)
#include <sys/mman.h>
#endif

#ifndef TPHT_ENABLE_SIMD
#define TPHT_ENABLE_SIMD 1
#endif

/*
 * SIMD levels, ranked.  AUTO picks the best level the compilation target
 * supports (AVX-512 > AVX2 > SSE2 > NEON > scalar); any level can be forced
 * with -DTPHT_SIMD_MODE=TPHT_SIMD_<LEVEL> for testing, and a forced level the
 * target cannot execute falls back to the best one it can.
 */
#define TPHT_SIMD_AUTO 0
#define TPHT_SIMD_SCALAR 1
#define TPHT_SIMD_SSE2 2
#define TPHT_SIMD_AVX2 3
#define TPHT_SIMD_NEON 4
#define TPHT_SIMD_AVX512 5

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

/*
 * The level actually compiled in, with AUTO resolved against what the target
 * supports.  Code that wants to know which instructions it may emit - as
 * opposed to which fingerprint matcher to call - tests this.  A level forced
 * with -DTPHT_SIMD_MODE that the target cannot execute degrades here rather
 * than producing instructions the CPU would fault on.
 */
#if TPHT_SIMD_MODE == TPHT_SIMD_AUTO
#if defined(TPHT_X86_SIMD) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__BMI2__)
#define TPHT_SIMD_MODE_EFFECTIVE TPHT_SIMD_AVX512
#elif defined(TPHT_X86_SIMD) && defined(__AVX2__)
#define TPHT_SIMD_MODE_EFFECTIVE TPHT_SIMD_AVX2
#elif defined(TPHT_X86_SIMD) && (defined(__SSE2__) || defined(_M_X64) || defined(_M_IX86_FP))
#define TPHT_SIMD_MODE_EFFECTIVE TPHT_SIMD_SSE2
#elif defined(TPHT_NEON_SIMD)
#define TPHT_SIMD_MODE_EFFECTIVE TPHT_SIMD_NEON
#else
#define TPHT_SIMD_MODE_EFFECTIVE TPHT_SIMD_SCALAR
#endif
#elif TPHT_SIMD_MODE == TPHT_SIMD_AVX512 && \
    !(defined(TPHT_X86_SIMD) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__BMI2__))
/* Asked for AVX-512 without the flags to emit it: use the next level down. */
#if defined(TPHT_X86_SIMD) && defined(__AVX2__)
#define TPHT_SIMD_MODE_EFFECTIVE TPHT_SIMD_AVX2
#elif defined(TPHT_X86_SIMD) && (defined(__SSE2__) || defined(_M_X64) || defined(_M_IX86_FP))
#define TPHT_SIMD_MODE_EFFECTIVE TPHT_SIMD_SSE2
#else
#define TPHT_SIMD_MODE_EFFECTIVE TPHT_SIMD_SCALAR
#endif
#else
#define TPHT_SIMD_MODE_EFFECTIVE TPHT_SIMD_MODE
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
#define TPHT_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define TPHT_LIKELY(x) __builtin_expect(!!(x), 1)
#else
#define TPHT_UNUSED
#define TPHT_HOT static inline
#define TPHT_NOINLINE
#define TPHT_UNLIKELY(x) (x)
#define TPHT_LIKELY(x) (x)
#endif

/*
 * Slots per dereference bin.  Fixed, not configurable: a tiny pointer is one
 * byte - bit 7 selects which of the two hash choices produced it, bits 0..6
 * the slot within the bin - so 127 is the largest a bin can be, and it is also
 * the best: bigger bins spill less.  Holding it constant lets the compiler
 * strength-reduce the bin arithmetic, and turns the divide and modulo that
 * locate a freed entry into multiplies.
 */
#define TPHT_BIN_SIZE 127u
#define TPHT_DEFAULT_LOAD_FACTOR 0.85
#define TPHT_MIN_CAPACITY 16u

/*
 * Internal only - never crosses the public API.  A concurrent attempt that
 * raced a resize commit and saw a stale storage pointer reports this to its
 * entry point, which simply re-reads everything and tries again.  It lives
 * outside the public enum so the public surface stays the four codes that
 * mean something to a caller.
 */
#define TPHT_RETRY ((tpht_status_t)-1)
#define TPHT_CHAINED_DEREF_LOAD_NUM 95u
#define TPHT_CHAINED_DEREF_LOAD_DEN 100u
/*
 * Buckets per migration stride.  Bigger strides mean fewer claims on the
 * shared stride counter and better locality per helper; measured on a
 * 256-thread box, 256 edged out 64 by a few percent at 32 threads and cost
 * nothing at 8.  Overridable for experiments.
 */
#ifndef TPHT_DEFAULT_RESIZE_STRIDE_BUCKETS
#define TPHT_DEFAULT_RESIZE_STRIDE_BUCKETS 256u
#endif
/*
 * Chain version slots for a concurrent chained table.  One version byte per
 * base is needless: what matters is that the threads actually running rarely
 * share a slot by accident, and 2^16 slots keep that chance below a percent
 * even at hundreds of threads while fitting the whole table in 64KB - a
 * per-base array on a large table is megabytes of cold memory.  A power of
 * two, so the slot index is a mask, never a divide.  Small tables use one
 * slot per base, which is both smaller and collision-free.
 */
#define TPHT_CHAIN_VERSION_SLOTS 65536u
/*
 * Byte stride (log2) between chain seqlock slots.  0 packs them - best cache
 * behaviour for lock-free readers, which only load versions; larger strides
 * spread writers' RMWs over more lines at the readers' expense.  Kept
 * overridable for measurement.
 */
#ifndef TPHT_CHAIN_LOCK_SHIFT
#define TPHT_CHAIN_LOCK_SHIFT 0u
#endif
#define TPHT_CHAIN_SLOT(x) ((size_t)(x) << TPHT_CHAIN_LOCK_SHIFT)

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
/*
 * Byte 63 is the seqlock version a concurrent flattened variant needs.  It is
 * reserved unconditionally - never read, never written - so that variant can be
 * added without changing the shape of an existing block.
 */
#define TPHT_FLAT_VERSION_OFF 63u
/* Tuple count and crystal count, adjacent so both move in one 16-bit access. */
#define TPHT_FLAT_CONTROL_OFF 61u
#define TPHT_FLAT_CRYSTALS_OFF 62u
#define TPHT_FLAT_COUNT_MASK 0x1fu
/*
 * Most tuples a block may hold: 30.  The 5-bit count field could encode 31,
 * but 31 tuples never fit - even all-spilled they need 31 fingerprints plus
 * 31 tiny pointers = 62 bytes, one past the 61 usable.  30 of each is 60.
 * This is the capacity itself, so the insert guard compares against it
 * directly; arrays indexed by a count are sized +1 as usual.
 */
#define TPHT_FLAT_MAX_TUPLES 30u


/* Fingerprints are one quotiented byte, so the home array needs 8 spare bits. */
#define TPHT_FLAT_FP_BITS 8u
/* Usable bytes in a block once the control and version bytes are removed. */
#define TPHT_FLAT_USABLE_BYTES (TPHT_FLAT_CONTROL_OFF)
/* Both counts up by one, applied to the 16-bit metadata field. */
#define TPHT_FLAT_META_CRYSTAL 0x0101u
/* Extra dereference table capacity over the expected overflow, in percent. */
#define TPHT_FLAT_DEREF_HEADROOM 50u

/*
 * Per-bin metadata stride, in log2 - a per-table value now, because the two
 * threadings want opposite layouts.  Concurrent: count, freelist head and the
 * bin lock used to live in tightly packed arrays - 32 bins' counts or 64
 * bins' locks per cache line - so two threads touching unrelated bins fought
 * over the same line; the lock's xchg alone was half of a chained insert's
 * profile.  One 64-byte record per bin gives every bin its own line at a cost
 * of well under a byte per stored key.  Sequential: there are no locks and no
 * sharing, and the same 64-byte stride just turned the two-choice comparison
 * into two scattered cache lines over a megabyte-scale array; two packed
 * bytes per bin keep the whole metadata cache-resident, which is worth a few
 * ns on every overflow insert.
 */
#define TPHT_POOL_META_SHIFT_CONC 6u
#define TPHT_POOL_META_SHIFT_SEQ 1u
#define TPHT_POOL_META_COUNT 0u
#define TPHT_POOL_META_HEAD 1u
#define TPHT_POOL_META_LOCK 2u /* concurrent records only; unused at the packed stride */

typedef struct tpht_pool {
    uint8_t *entries;
    uint8_t *cnt_head; /* per-bin record: [count, head] packed, or [count, head,
                          lock, pad] on its own line - see meta_shift. */
    size_t bin_count;
    size_t entry_size;  /* next-byte + key + value. */
    uint8_t locked;     /* concurrent table: bins carry a live lock byte. */
    uint8_t meta_shift; /* log2 record stride: _SEQ packed or _CONC padded. */
} tpht_pool_t;

/*
 * One shard of the running entry count, alone on its cache line.  Power-of-two
 * shard count so the thread salt maps with a mask.
 */
#define TPHT_SIZE_SHARDS 8u

#if defined(_MSC_VER)
#define TPHT_THREAD_LOCAL __declspec(thread)
#else
#define TPHT_THREAD_LOCAL _Thread_local
#endif

typedef struct {
    atomic_size_t v;
    char pad[64 - sizeof(atomic_size_t)];
} tpht_size_shard_t;

typedef struct tpht_retired_storage {
    uint8_t *heads;
    void *flat_lines_raw;
    uint8_t *pool_entries;
    uint8_t *pool_cnt_head;
    atomic_uchar *chain_locks;
    tpht_table_t *resize_descriptor;
    struct tpht_retired_storage *next;
} tpht_retired_storage_t;

/*
 * One in-flight resize, heap-allocated so that every helper works from the same
 * immutable snapshot.  The previous design kept this state in the table itself
 * and cleared it at commit, which raced with helpers that had already checked
 * resize_active: a straggler could read a NULLed migrated array (crash), or -
 * worse - a helper left over from resize N-1 could bump the done-strides
 * counter of resize N and let its commit fire before every bucket was migrated.
 * With the state boxed per-resize, a stale helper only ever touches its own
 * finished descriptor, where every bucket is already marked migrated and every
 * step degrades to a no-op.  Descriptors are retired at commit, never freed
 * until the table is destroyed, so a straggler can always still read one.
 */
typedef struct tpht_resize_op {
    tpht_table_t *target;          /* the shadow table being filled */
    uint64_t old_pack;             /* flat_lines_pack current when this began */
    tpht_table_t *old_geo;         /* geo_snap current when this began */
    uint8_t *migrated;             /* one byte per old block/bucket group */
    uint8_t *old_lines;            /* flatten: the old block array */
    atomic_uchar *old_chain_locks; /* chained: the old seqlock version array */
    size_t old_chain_version_mask;
    size_t old_base_count;
    size_t stride_size;
    size_t stride_count;
    atomic_size_t next_stride;
    atomic_size_t done_strides;
    atomic_uchar failed;   /* flatten: the shadow overflowed; abort, do not lose keys */
    atomic_flag commit_lock; /* one committer per resize, taken once, never cleared */
    struct tpht_resize_op *next; /* retired-descriptor chain, freed at destroy */
} tpht_resize_op_t;

struct tpht_table {
    tpht_config_t cfg;
    /*
     * The fields every concurrent operation reads once per call, grouped so
     * they share one cache line instead of five scattered ones: the resize
     * flag, the size-tracking gate and its check period, the packed line
     * word, and the geometry snapshot pointer.  All are read-mostly; the
     * resize paths that write them already hold the necessary exclusion.
     * Their design comments live at their original sites below.
     */
    atomic_bool resize_active;
    uint8_t tracks_size;
    _Atomic unsigned size_check_period;
    _Atomic uint64_t flat_lines_pack; /* 64-byte aligned view of flat_lines_raw */
    struct tpht_table *_Atomic geo_snap;
    struct tpht_table *initial_geo; /* create-time snapshot copy, freed at destroy */
    size_t capacity;
    /*
     * Size at which a write must leave the hot path: the load-factor threshold
     * for a resizable table, unreachable for a fixed one, which has no size to
     * test.  Precomputed so the hot path needs neither a resize-mode branch nor
     * a per-insert product.
     */
    size_t write_limit;
    /*
     * How many writes a thread may run between looks at write_limit.  Checking
     * costs a sweep over every size shard - cross-core traffic when writers
     * are hot - so large tables amortise it; a small table cannot afford the
     * overshoot (64 skipped checks per thread can be several times its whole
     * limit) and checks every write.
     */
    /* (size_check_period declared in the hot group at the top) */
    /*
     * Only a resizable table has to know how many entries it holds, to decide
     * when to grow.  A fixed table never grows on load, absorbs a hard overflow
     * by rebuilding with more blocks, and reports its size by counting on
     * demand - so it does not pay for the counter on every write.
     */

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
    uint8_t *flat_lines;
    /*
     * flat_lines and the block-count log2 in one atomic word (the array is
     * 64-byte aligned, so the low six bits are free).  A concurrent writer
     * derives its line from this single load: the block index is clamped by
     * the packed bit count, so whatever torn mask values the locate read, the
     * pointer stays inside the packed array - memory safety by construction.
     * The post-lock re-read of the same word then says whether a commit
     * interleaved (storages are never address-reused, so equality is exact),
     * and with the line lock held that verdict is final: a commit takes every
     * old line lock before it stores a single field, so a stable pack means
     * every geometry field this writer read was stable too.  Readers cannot
     * lean on a lock and use the full geo_snap snapshot instead.
     */
    /* (declaration moved to the hot group at the top of the struct) */
    void *flat_lines_raw;
    uint8_t flat_entry_size; /* quotiented key bytes + value bytes. */
    uint8_t flat_cost;       /* inline cost per tuple: max(entry_size - 1, 1). */
    /*
     * Bit x is set when a block already holding x tuples can take one more
     * inline, i.e. crystals(x + 1) == crystals(x) + 1.  The crystal count is a
     * function of the tuple count alone, so this whole decision collapses to a
     * shift and a test of a word the insert path already has in a register.
     */
    uint32_t flat_inline_ok;
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
    /* Skip populating the arrays at allocation.  Almost never right: an
     * unpopulated shadow pays a CoW break with a full TLB-shootdown IPI per
     * first-written page mid-migration (see the concurrent resize starts),
     * which measures far worse than the starter's one batched populate. */
    uint8_t no_prefault;
    /* Set when the last hard overflow came from the dereference table. */
    _Atomic uint8_t flat_deref_pressure; /* relaxed: heuristic read lock-free */
    /* Lower bound on dereference table slots, raised when it runs out. */
    size_t flat_deref_floor;

    tpht_pool_t pool;
    atomic_flag lock;
    atomic_uchar *chain_locks;
    size_t chain_lock_count;
    size_t chain_version_mask; /* chain_lock_count - 1; slot = base & mask. */

    /*
     * The published geometry snapshot.  A resize commit is a multi-word swap
     * of the storage pointers and every derived mask, and a thread that holds
     * no lock (a reader's whole walk; a writer up to its block lock) can
     * observe that swap half-done: base mask from one storage, line array
     * from another, and the composed line pointer lands outside either
     * allocation - a wild read for a lookup, a wild lock RMW for a write.  So
     * the lock-free phases never read those fields from this struct at all:
     * they load geo_snap once and take everything from the immutable table
     * image it points to - the shadow descriptor of the resize that built the
     * storage (or the create-time copy, for the first one), alive until the
     * table dies.  One pointer load is one atomic word, so the view is
     * consistent by construction, and "did a commit interleave?" is one
     * pointer comparison instead of the storage-generation arithmetic this
     * replaces.  Sequential tables point it at the table itself: their
     * geometry only changes single-threaded, and the compiler folds the
     * indirection away.
     */
    /* (geo_snap / initial_geo declared in the hot group at the top) */
    atomic_flag resize_start_lock;
    /*
     * The running entry count, split across per-thread shards so concurrent
     * writers do not fight over one cache line.  A single atomic counter beside
     * the geometry that every operation reads was measured at several times the
     * cost of the insert itself on an 8-writer growing table: every add took
     * the line exclusive and forced every other thread to refetch it.  Each
     * shard sits on its own line (padding rather than _Alignas, because the
     * tables come from calloc, which does not honour an over-aligned type);
     * a thread adds to the shard its thread-local salt names, and the total is
     * the wrapping sum of the shards - exact whenever writers are excluded,
     * which is the only time an exact answer is needed.  A sequential table
     * only ever touches shard 0, so its hot path reads one line as before.
     */
    char size_pad_before[64];
    tpht_size_shard_t size_shard[TPHT_SIZE_SHARDS];
    tpht_resize_op_t *_Atomic resize_op; /* the in-flight resize, or NULL */
    tpht_resize_op_t *retired_ops;       /* finished descriptors, freed at destroy */
    tpht_retired_storage_t *retired;
};

/*
 * splitmix64: a full avalanche on a counter, used to separate the hash streams
 * and to spread the entropy behind a generated seed.
 */
static uint64_t tpht_splitmix64(uint64_t x) {
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static size_t tpht_max_size(size_t a, size_t b) { return a > b ? a : b; }

/*
 * Which shard this thread's adds land in.  Each thread draws one salt for its
 * lifetime; with the shard count a power of two, distinct salts spread threads
 * round-robin, and up to TPHT_SIZE_SHARDS writers add without sharing a line.
 */
static atomic_uint tpht_shard_salt_seq;
static TPHT_THREAD_LOCAL unsigned tpht_tls_shard_salt;

/*
 * How many writes a thread performs between looks at the size limit.  Summing
 * the shards reads a line per shard, most of them freshly written by other
 * threads; doing that per insert would put the cross-core traffic right back.
 * Checking every 64th write bounds the overshoot past the limit to
 * 64 * threads entries, absorbed by the load-factor slack below the storage's
 * hard bound - and a hard overflow in the gap still trips the TPHT_OVERFLOW arm,
 * which resizes regardless of what the counter says.
 */
#define TPHT_SIZE_CHECK_PERIOD 64u
static TPHT_THREAD_LOCAL unsigned tpht_tls_limit_tick;

static unsigned tpht_size_shard_pick(void) {
    unsigned salt = tpht_tls_shard_salt;
    if (TPHT_UNLIKELY(salt == 0)) {
        salt = atomic_fetch_add_explicit(&tpht_shard_salt_seq, 1u, memory_order_relaxed) + 1u;
        tpht_tls_shard_salt = salt;
    }
    return salt & (TPHT_SIZE_SHARDS - 1u);
}

/*
 * The wrapping sum of the shards.  Exact whenever writers are excluded (a
 * commit holds every lock; a sequential table has no other threads); at any
 * other moment it is the same order-free snapshot a single atomic counter
 * would have given.  Wraparound in individual shards - a remove's decrement
 * landing in a different shard than the insert's increment - cancels in the
 * modular sum.
 */
static size_t tpht_size_load(const tpht_table_t *t) {
    size_t sum = 0;
    unsigned i;
    for (i = 0; i < TPHT_SIZE_SHARDS; ++i)
        sum += atomic_load_explicit((atomic_size_t *)&t->size_shard[i].v, memory_order_acquire);
    return sum;
}

/*
 * A sequential table only ever moves shard 0, so its hot-path limit check
 * stays one load instead of a sweep over every shard line.
 */
TPHT_HOT size_t tpht_size_load_seq(const tpht_table_t *t) {
    return atomic_load_explicit((atomic_size_t *)&t->size_shard[0].v, memory_order_relaxed);
}

static void tpht_size_store(tpht_table_t *t, size_t size) {
    unsigned i;
    atomic_store_explicit(&t->size_shard[0].v, size, memory_order_release);
    for (i = 1; i < TPHT_SIZE_SHARDS; ++i)
        atomic_store_explicit(&t->size_shard[i].v, 0, memory_order_release);
}

/*
 * A read-modify-write on a shared counter compiles to a lock-prefixed
 * instruction whose line bounces between writers - measured at several times
 * the cost of the insert it was counting.  A concurrent table adds to this
 * thread's own shard; a sequential one gets a plain load, add and store.
 */
static void tpht_size_inc(tpht_table_t *t) {
    if (t->cfg.threading == TPHT_SEQUENTIAL) {
        atomic_store_explicit(
            &t->size_shard[0].v,
            atomic_load_explicit(&t->size_shard[0].v, memory_order_relaxed) + 1u,
            memory_order_relaxed);
        return;
    }
    atomic_fetch_add_explicit(&t->size_shard[tpht_size_shard_pick()].v, 1u,
                              memory_order_relaxed);
}

/*
 * Same, for the paths whose concurrency is a call-site literal: testing the
 * threading mode there would be a per-insert branch on a known value.
 */
TPHT_HOT void tpht_size_inc_seq(tpht_table_t *t) {
    atomic_store_explicit(&t->size_shard[0].v,
                          atomic_load_explicit(&t->size_shard[0].v, memory_order_relaxed) + 1u,
                          memory_order_relaxed);
}

TPHT_HOT void tpht_size_dec_seq(tpht_table_t *t) {
    atomic_store_explicit(&t->size_shard[0].v,
                          atomic_load_explicit(&t->size_shard[0].v, memory_order_relaxed) - 1u,
                          memory_order_relaxed);
}

TPHT_HOT void tpht_size_inc_conc(tpht_table_t *t) {
    atomic_fetch_add_explicit(&t->size_shard[tpht_size_shard_pick()].v, 1u,
                              memory_order_relaxed);
}

TPHT_HOT void tpht_size_dec_conc(tpht_table_t *t) {
    atomic_fetch_sub_explicit(&t->size_shard[tpht_size_shard_pick()].v, 1u,
                              memory_order_relaxed);
}

static void tpht_size_dec(tpht_table_t *t) {
    if (t->cfg.threading == TPHT_SEQUENTIAL) {
        atomic_store_explicit(
            &t->size_shard[0].v,
            atomic_load_explicit(&t->size_shard[0].v, memory_order_relaxed) - 1u,
            memory_order_relaxed);
        return;
    }
    atomic_fetch_sub_explicit(&t->size_shard[tpht_size_shard_pick()].v, 1u,
                              memory_order_relaxed);
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
/* The precomputed XXH3 bitflip for one of a table's hash streams. */
static uint64_t tpht_stream_bitflip(uint64_t seed, unsigned stream) {
    uint64_t s = tpht_splitmix64(seed ^ (UINT64_C(0x9e3779b97f4a7c15) * (stream + 1u)));
    return (TPHT_XXH3_SECRET_08 ^ TPHT_XXH3_SECRET_16) -
           (s ^ ((uint64_t)tpht_bswap32((uint32_t)s) << 32));
}

/*
 * A seed for a table whose caller did not choose one.  Two tables must never
 * share a hash function by accident, so a process-wide counter is mixed in:
 * that alone guarantees distinct seeds even where no entropy source exists.
 * The entropy makes them unpredictable as well, which is what keeps a table
 * from being driven into its worst case by chosen keys.
 */
static uint64_t tpht_random_seed(void) {
    static atomic_uint_least64_t tpht_seed_counter;
    static atomic_uint_least64_t tpht_seed_entropy;
    uint64_t entropy = atomic_load_explicit(&tpht_seed_entropy, memory_order_acquire);
    uint64_t ordinal = (uint64_t)atomic_fetch_add_explicit(&tpht_seed_counter, 1u,
                                                           memory_order_relaxed);
    if (entropy == 0u) {
        uint64_t bits = 0;
#if defined(__unix__) || defined(__APPLE__) || defined(__linux__)
        FILE *f = fopen("/dev/urandom", "rb");
        if (f) {
            if (fread(&bits, 1, sizeof(bits), f) != sizeof(bits)) bits = 0;
            fclose(f);
        }
#endif
        if (bits == 0u) {
            /* No entropy source: fall back to whatever varies per run - the
             * clock, and addresses the loader placed. */
            uintptr_t here = (uintptr_t)(const void *)&tpht_seed_counter;
            uintptr_t stack = (uintptr_t)(const void *)&bits;
            bits = tpht_splitmix64((uint64_t)time(NULL)) ^
                   tpht_splitmix64((uint64_t)clock()) ^ ((uint64_t)here << 16) ^
                   (uint64_t)stack;
        }
        if (bits == 0u) bits = UINT64_C(0x243f6a8885a308d3);
        atomic_store_explicit(&tpht_seed_entropy, bits, memory_order_release);
        entropy = bits;
    }
    return tpht_splitmix64(entropy ^ tpht_splitmix64(ordinal));
}

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


static uint64_t tpht_key_word(const tpht_table_t *t, const void *key) {
    return tpht_read_le(key, t->key_size) &
           (t->key_bits == 64u ? UINT64_MAX : ((UINT64_C(1) << t->key_bits) - 1u));
}

static uint64_t tpht_key_quotient(const tpht_table_t *t, const void *key) {
    return tpht_key_word(t, key) >> t->base_bits;
}

/* Force-inlined: as a shared out-of-line helper it showed up as its own hot
 * profile entry on the chained insert path - call overhead plus a reloaded
 * hash constant on every chain operation. */
TPHT_HOT size_t tpht_base_from_word(const tpht_table_t *t, uint64_t key_word) {
    uint64_t quotient = key_word >> t->base_bits;
    return (size_t)((tpht_xxh3_word_bitflip(quotient, t->key_size, t->hash_bitflip) ^ key_word) & t->base_mask);
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
#if defined(TPHT_X86_SIMD) && defined(__BMI2__)
    /* bzhi keeps all bits when the index is >= 32, exactly the wide case. */
    return _bzhi_u32(UINT32_MAX, count);
#else
    return count >= 32u ? UINT32_MAX : ((UINT32_C(1) << count) - 1u);
#endif
}

/* Referenced only by the SIMD modes that fall back to it. */
TPHT_UNUSED TPHT_HOT uint32_t tpht_fp_match_mask_scalar(const uint8_t *fps, uint8_t count, uint8_t fp) {
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

#if defined(TPHT_X86_SIMD) && (defined(__GNUC__) || defined(__clang__))
#if defined(__AVX512BW__) && defined(__AVX512VL__)
TPHT_HOT uint32_t tpht_fp_match_mask_avx512(const uint8_t *fps, uint8_t count, uint8_t fp) {
#else
__attribute__((target("avx512bw,avx512vl,bmi2")))
static TPHT_UNUSED uint32_t tpht_fp_match_mask_avx512(const uint8_t *fps, uint8_t count, uint8_t fp) {
#endif
    /*
     * One 256-bit compare straight into a mask register: unlike the AVX2 form
     * there is no separate movemask, and the low-mask trim rides in the
     * compare's result via bzhi.  Covers every possible fingerprint count in
     * a 64-byte block (at most 31).
     */
    __m256i needle = _mm256_set1_epi8((char)fp);
    __m256i hay = _mm256_loadu_si256((const __m256i *)(const void *)fps);
    return (uint32_t)_mm256_cmpeq_epi8_mask(hay, needle) & _bzhi_u32(UINT32_MAX, count);
}

static TPHT_UNUSED int tpht_cpu_has_avx512bw_vl(void) {
#if defined(__GNUC__) || defined(__clang__)
    static int cached = -1;
    if (cached < 0)
        cached = (__builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl")) ? 1 : 0;
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
#elif TPHT_SIMD_MODE == TPHT_SIMD_AVX512
#if defined(TPHT_X86_SIMD) && (defined(__GNUC__) || defined(__clang__))
    return tpht_fp_match_mask_avx512(fps, count, fp);
#else
    return tpht_fp_match_mask_scalar(fps, count, fp);
#endif
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
#if defined(__AVX512BW__) && defined(__AVX512VL__)
    /* Best ranked level the target supports: one compare into a mask register,
     * no movemask, no count>16 branch, no per-lookup CPU feature test. */
    return tpht_fp_match_mask_avx512(fps, count, fp);
#elif defined(__AVX2__)
    /* Compiled for AVX2 already: a 256-bit compare handles every block size,
     * so there is no count>16 branch and no per-lookup CPU feature test. */
    return tpht_fp_match_mask_avx2(fps, count, fp);
#else
    if (count > 16u && tpht_cpu_has_avx512bw_vl()) return tpht_fp_match_mask_avx512(fps, count, fp);
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

/*
 * Set the low bit and report what it was.  On x86 this expands to a load and a
 * compare-exchange loop: there is no byte-wide OR that returns its old value,
 * and `bts` - which would do it in one locked instruction - has no byte form,
 * so a one-byte seqlock cannot be taken in a single access.  Widening the
 * version to 32 bits would allow it, but the flattened block has no spare byte
 * and the chained lock array would grow fourfold, which is not worth it while
 * both already match or beat the reference.
 */
TPHT_HOT int tpht_bit_test_and_set(atomic_uchar *v) {
    return atomic_fetch_or_explicit(v, 1u, memory_order_acq_rel) & 1u;
}

/* A hint that this core is spinning, so it yields pipeline resources. */
TPHT_HOT void tpht_cpu_relax(void) {
#if defined(TPHT_X86_SIMD)
    _mm_pause();
#elif defined(TPHT_NEON_SIMD) && (defined(__GNUC__) || defined(__clang__))
    __asm__ __volatile__("yield" ::: "memory");
#endif
}

static void tpht_flag_lock(atomic_flag *lock) {
    while (atomic_flag_test_and_set_explicit(lock, memory_order_acquire)) {
    }
}

/* One shot, no spin: the caller has useful work to do if someone else holds it. */
static int tpht_flag_trylock(atomic_flag *lock) {
    return !atomic_flag_test_and_set_explicit(lock, memory_order_acquire);
}

static void tpht_flag_unlock(atomic_flag *lock) {
    atomic_flag_clear_explicit(lock, memory_order_release);
}

static int tpht_chained_fine_grained(const tpht_table_t *t) {
    /* Decided from the immutable configuration alone: a concurrent chained
     * table always carries its lock arrays, and reading those pointers here
     * raced the resize commit's swap of them. */
    return t->cfg.variant == TPHT_CHAINED && t->cfg.threading == TPHT_CONCURRENT;
}

/*
 * Lock the chain for a base.  The index needs no reduction: a base comes from
 * tpht_base_from_word, which masks it below base_count, and there is one lock
 * per base - the modulo that used to be here was a hardware division on every
 * lock and unlock, twice per operation, for a value already in range.
 *
 * The _fine forms skip the configuration test as well: their callers reached
 * them through it, so re-deriving it from four table fields per lock is work
 * the call site has already done.
 */
/*
 * A chain's lock is a seqlock version rather than a mutex.  A writer takes the
 * chain by moving the version from even to odd and releases it by moving on to
 * the next even value.  A reader never writes it at all: it takes the version,
 * walks, and re-reads - so readers do not serialise against each other and pay
 * no atomic read-modify-write, which is what a lookup on a shared table used to
 * cost.  A reader can observe a chain mid-edit; the re-read discards it.
 */
TPHT_HOT void tpht_chain_lock_base_fine(tpht_table_t *t, size_t base) {
    atomic_uchar *v = &t->chain_locks[TPHT_CHAIN_SLOT(base & t->chain_version_mask)];
    for (;;) {
        if (!tpht_bit_test_and_set(v)) return;
        do {
            tpht_cpu_relax();
        } while (atomic_load_explicit(v, memory_order_relaxed) & 1u);
    }
}


/* Reader side: an even version, waiting out any writer currently in the chain. */
TPHT_HOT unsigned char tpht_chain_read_begin(tpht_table_t *t, size_t base) {
    atomic_uchar *v = &t->chain_locks[TPHT_CHAIN_SLOT(base & t->chain_version_mask)];
    for (;;) {
        unsigned char cur = atomic_load_explicit(v, memory_order_acquire);
        if (!(cur & 1u)) return cur;
        tpht_cpu_relax();
    }
}

TPHT_HOT int tpht_chain_read_valid(tpht_table_t *t, size_t base, unsigned char snapshot) {
    return atomic_load_explicit(&t->chain_locks[TPHT_CHAIN_SLOT(base & t->chain_version_mask)],
                                memory_order_acquire) == snapshot;
}

/*
 * Pointer-returning form: the caller must release exactly the lock it took,
 * and a resize commit can swap t->chain_locks between the acquire and the
 * release - recomputing the slot from the table at unlock time would then
 * unlock a different, live lock.  The storage-generation check right after the
 * acquire is what detects that swap and sends the caller back around.
 */
static atomic_uchar *tpht_chain_lock_take(tpht_table_t *t, size_t base) {
    atomic_uchar *v = &t->chain_locks[TPHT_CHAIN_SLOT(base & t->chain_version_mask)];
    for (;;) {
        if (!tpht_bit_test_and_set(v)) return v;
        do {
            tpht_cpu_relax();
        } while (atomic_load_explicit(v, memory_order_relaxed) & 1u);
    }
}

static void tpht_chain_lock_release(atomic_uchar *v) {
    atomic_store_explicit(v, (unsigned char)(atomic_load_explicit(v, memory_order_relaxed) + 1u),
                          memory_order_release);
}

static void tpht_pool_lock_bin(tpht_table_t *t, size_t bin) {
    if (t->pool.locked)
        tpht_flag_lock((atomic_flag *)&t->pool.cnt_head[(bin << t->pool.meta_shift) |
                                                        TPHT_POOL_META_LOCK]);
}

static void tpht_pool_unlock_bin(tpht_table_t *t, size_t bin) {
    if (t->pool.locked)
        tpht_flag_unlock((atomic_flag *)&t->pool.cnt_head[(bin << t->pool.meta_shift) |
                                                          TPHT_POOL_META_LOCK]);
}

static uint8_t *tpht_pool_entry(tpht_pool_t *p, size_t bin, uint8_t pos) {
    return p->entries + ((bin * (size_t)TPHT_BIN_SIZE + (size_t)pos) * p->entry_size);
}

/*
 * The count byte is read both under the bin lock and without it (the
 * two-choice comparison, and size() on a fixed table), so every access is a
 * relaxed atomic: same machine code as a plain byte access, and no
 * mixed-atomicity race.  Mutations happen only under the bin lock, so a plain
 * load-add-store through the atomic object is exact.
 */
static uint8_t tpht_pool_count(const tpht_pool_t *p, size_t bin) {
    return atomic_load_explicit(
        (const _Atomic uint8_t *)&p->cnt_head[bin << p->meta_shift],
        memory_order_relaxed);
}

static void tpht_pool_count_add(tpht_pool_t *p, size_t bin, int delta) {
    _Atomic uint8_t *c = (_Atomic uint8_t *)&p->cnt_head[bin << p->meta_shift];
    atomic_store_explicit(
        c, (uint8_t)(atomic_load_explicit(c, memory_order_relaxed) + delta),
        memory_order_relaxed);
}

static uint8_t *tpht_pool_head_ptr(tpht_pool_t *p, size_t bin) {
    return &p->cnt_head[(bin << p->meta_shift) | TPHT_POOL_META_HEAD];
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

TPHT_HOT uint8_t *tpht_pool_alloc(tpht_table_t *t, uint64_t deref_key, uint8_t *encoded_out,
                                  unsigned key_bytes) {
    size_t bin1 = tpht_bin_of(tpht_xxh3_word_bitflip(deref_key, key_bytes, t->hash_bitflip_100), t->pool.bin_count);
    size_t bin2 = tpht_bin_of(tpht_xxh3_word_bitflip(deref_key, key_bytes, t->hash_bitflip_200), t->pool.bin_count);
    uint8_t flag;
    size_t bin;
    uint8_t *head;
    uint8_t pos;
    uint8_t *entry;

    /*
     * Optimistic two-choice: compare the fill levels before taking any lock
     * and lock only the chosen bin.  Locking both bins doubled the lock
     * traffic of every overflow insert to buy an exact comparison that the
     * scheme never needed - two-choice is a balance heuristic, and a stale
     * count just places one entry slightly less evenly.  If the chosen bin
     * turns out full under its lock (a stale read, or a race), the other bin
     * gets one try before the caller sees a hard overflow; a full chosen bin
     * with an exact comparison would have meant both were full anyway.
     */
    flag = (uint8_t)(tpht_pool_count(&t->pool, bin1) > tpht_pool_count(&t->pool, bin2));
    bin = flag ? bin2 : bin1;
    tpht_pool_lock_bin(t, bin);
    head = tpht_pool_head_ptr(&t->pool, bin);
    if (TPHT_UNLIKELY(*head >= TPHT_BIN_SIZE)) {
        tpht_pool_unlock_bin(t, bin);
        flag ^= 1u;
        bin = flag ? bin2 : bin1;
        tpht_pool_lock_bin(t, bin);
        head = tpht_pool_head_ptr(&t->pool, bin);
        if (*head >= TPHT_BIN_SIZE) {
            tpht_pool_unlock_bin(t, bin);
            return NULL;
        }
    }

    pos = *head;
    entry = tpht_pool_entry(&t->pool, bin, pos);
    *head = (uint8_t)(pos + 1u + entry[0]);
    if (*head > TPHT_BIN_SIZE) *head = (uint8_t)(*head - (TPHT_BIN_SIZE + 1u));
    tpht_pool_count_add(&t->pool, bin, 1);
    *encoded_out = (uint8_t)((pos + 1u) | (flag << 7u));
    tpht_pool_unlock_bin(t, bin);
    return entry;
}

static void tpht_pool_free(tpht_table_t *t, uint8_t encoded_ptr, uint8_t *entry) {
    size_t ordinal = (size_t)((entry - t->pool.entries) / (ptrdiff_t)t->pool.entry_size);
    size_t bin = ordinal / TPHT_BIN_SIZE;
    uint8_t pos = (uint8_t)(ordinal % TPHT_BIN_SIZE);
    uint8_t *head = tpht_pool_head_ptr(&t->pool, bin);
    (void)encoded_ptr;
    tpht_pool_lock_bin(t, bin);
    entry[0] = (uint8_t)(*head + TPHT_BIN_SIZE - pos);
    if (entry[0] > TPHT_BIN_SIZE) entry[0] = (uint8_t)(entry[0] - (TPHT_BIN_SIZE + 1u));
    *head = pos;
    tpht_pool_count_add(&t->pool, bin, -1);
    tpht_pool_unlock_bin(t, bin);
}

static void tpht_free_storage(tpht_table_t *t) {
    tpht_retired_storage_t *r = t->retired;
    tpht_resize_op_t *op;
    while (r) {
        tpht_retired_storage_t *next = r->next;
        free(r->heads);
        free(r->flat_lines_raw);
        free(r->pool_entries);
        free(r->pool_cnt_head);
        free(r->chain_locks);
        free(r->resize_descriptor);
        free(r);
        r = next;
    }
    t->retired = NULL;
    op = t->retired_ops;
    while (op) {
        tpht_resize_op_t *next = op->next;
        free(op->migrated);
        free(op);
        op = next;
    }
    t->retired_ops = NULL;
    free(t->heads);
    free(t->flat_lines_raw);
    free(t->pool.entries);
    free(t->pool.cnt_head);
    free(t->chain_locks);
    op = atomic_load_explicit(&t->resize_op, memory_order_acquire);
    if (op) {
        /* Destroying mid-resize: the shadow was never published, free it too. */
        if (op->target) {
            tpht_free_storage(op->target);
            free(op->target);
        }
        free(op->migrated);
        free(op);
        atomic_store_explicit(&t->resize_op, NULL, memory_order_release);
    }
    t->heads = NULL;
    t->flat_lines = NULL;
    t->flat_lines_raw = NULL;
    t->pool.entries = NULL;
    t->pool.cnt_head = NULL;
    t->chain_locks = NULL;
    t->chain_lock_count = 0;
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
 * power of two.  Round up: a sparser home array keeps blocks below their inline
 * capacity, so the fast no-overflow insert and the one-line lookup stay the
 * common case.  Build with -DTPHT_FLAT_DENSE=1 to pick the nearer power of two
 * instead, which halves the home array (down to ~60% of the memory) whenever
 * the target lands below 70% of the rounded-up size, at the price of far more
 * dereference-table traffic.
 */
#ifndef TPHT_FLAT_DENSE
#define TPHT_FLAT_DENSE 0
#endif
static size_t tpht_flat_block_count(size_t wanted) {
    size_t blocks = tpht_pow2_ceil(tpht_max_size(wanted, 1));
#if TPHT_FLAT_DENSE
    if (blocks > 1u && wanted * 10u < blocks * 7u) blocks >>= 1u;
#endif
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

/*
 * Ask for transparent huge pages on a table-sized array.  A block table at 4K
 * pages needs a dTLB entry per 64 lines, so a random-access workload larger
 * than the TLB's reach pays a page-walk per operation; 2MB pages put the whole
 * array under a handful of entries.  On madvise-mode kernels THP is granted
 * only where asked, and the advice must precede the first touch to matter, so
 * this runs right after the calloc and before any prefault.  Only the 2MB-
 * aligned interior is eligible - the ragged edges stay 4K - and small arrays
 * are left alone so a modest table does not round up to huge-page granularity.
 */
static void tpht_advise_hugepages(void *p, size_t bytes) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    const uintptr_t huge = (uintptr_t)2u << 20;
    uintptr_t lo, hi;
    if (!p || bytes < ((size_t)8u << 20)) return;
    lo = ((uintptr_t)p + huge - 1u) & ~(huge - 1u);
    hi = ((uintptr_t)p + bytes) & ~(huge - 1u);
    if (hi > lo) (void)madvise((void *)lo, hi - lo, MADV_HUGEPAGE);
#else
    (void)p;
    (void)bytes;
#endif
}

/*
 * Fault in one slice of an array, for the resize strides: slice idx of count,
 * page-aligned so neighbouring slices never touch the same page twice.  The
 * batched madvise populates without a trap per page - with huge pages, without
 * a trap per 4K of a 2MB fault - and doing it here, before the migration takes
 * any block seqlock, keeps fault stalls out of the sections writers wait on.
 * The advice is best-effort: where it is unsupported the first-touch faults
 * simply happen where they always did.
 */
static void tpht_populate_slice(uint8_t *p, size_t bytes, size_t idx, size_t count) {
#if defined(__linux__) && defined(MADV_POPULATE_WRITE)
    uintptr_t begin, end;
    if (!p || !bytes || count == 0 || idx >= count) return;
    /* Both bounds round down to a page, so the slices partition the pages
     * exactly; a boundary page belongs to the slice that starts inside it.
     * The edges may fall in pages the allocation only partly owns - those
     * pages are still mapped (they hold the allocation's own bytes), so
     * populating them is safe. */
    begin = ((uintptr_t)p + bytes * idx / count) & ~(uintptr_t)4095u;
    end = idx + 1u == count ? (uintptr_t)p + bytes
                            : ((uintptr_t)p + bytes * (idx + 1u) / count) & ~(uintptr_t)4095u;
    if (end > begin) (void)madvise((void *)begin, end - begin, MADV_POPULATE_WRITE);
#else
    (void)p;
    (void)bytes;
    (void)idx;
    (void)count;
#endif
}

/*
 * Populate an array the huge-page-friendly way, then finish with the manual
 * touch.  A trap-per-4K prefault into a MADV_HUGEPAGE region is granted huge
 * pages only haphazardly - each touch is its own racy shot at an order-9
 * allocation - where one MADV_POPULATE_WRITE of the range is granted them
 * reliably, and maps the whole array in one syscall besides.  The touch after
 * it is then a cheap sweep over already-present pages, and the whole prefault
 * where the populate call does not exist.
 */
static void tpht_prefault_arr(uint8_t *p, size_t bytes) {
    if (!p || !bytes) return;
    tpht_populate_slice(p, bytes, 0, 1);
    tpht_prefault(p, bytes);
}

static int tpht_alloc_flat_lines(tpht_table_t *t) {
    size_t bytes = t->base_count * (size_t)TPHT_FLAT_LINE_BYTES;
    uint8_t *raw;
    uintptr_t addr;

    if (bytes / TPHT_FLAT_LINE_BYTES != t->base_count) return 0;
    raw = (uint8_t *)calloc(bytes + (TPHT_FLAT_LINE_BYTES - 1u) + TPHT_LOAD_SLACK, 1);
    if (!raw) return 0;
    tpht_advise_hugepages(raw, bytes + (TPHT_FLAT_LINE_BYTES - 1u) + TPHT_LOAD_SLACK);
    t->flat_lines_raw = raw;
    addr = (uintptr_t)raw;
    addr = (addr + (TPHT_FLAT_LINE_BYTES - 1u)) & ~(uintptr_t)(TPHT_FLAT_LINE_BYTES - 1u);
    t->flat_lines = (uint8_t *)addr;
    return 1;
}

static uint64_t tpht_flat_pack_of(const tpht_table_t *t) {
    return (uint64_t)(uintptr_t)t->flat_lines | (uint64_t)t->flat_cloud_bits;
}

/*
 * A resizable table leaves the hot path one insert before it would exceed its
 * load factor; a fixed one when it is exactly full.  Recomputed wherever the
 * capacity changes, so the hot path only ever compares against this.
 */
/*
 * Commit-time form: recompute only what the capacity changes.  tracks_size is
 * an invariant of the table's resize mode, written once at creation; the
 * commit re-storing it (same value) raced the hot path's unsynchronized read
 * of it for no benefit.
 */
static void tpht_refresh_write_limit(tpht_table_t *t) {
    if (t->cfg.resize_mode == TPHT_FIXED) {
        t->write_limit = (size_t)-1;
        return;
    }
    {
        /*
         * Grow one insert before the load factor would be exceeded: the test
         * this replaces was size + 1 > capacity * load, so the smallest size
         * that trips it is floor(capacity * load - 1) + 1.
         */
        double limit = (double)t->capacity * t->cfg.max_load_factor - 1.0;
        size_t whole = limit <= 0.0 ? 0u : (size_t)limit;
        t->write_limit = whole + 1u;
    }
    {
        size_t period = t->write_limit / (8u * TPHT_SIZE_CHECK_PERIOD);
        if (period > TPHT_SIZE_CHECK_PERIOD) period = TPHT_SIZE_CHECK_PERIOD;
        if (period == 0) period = 1;
        atomic_store_explicit(&t->size_check_period, (unsigned)period, memory_order_relaxed);
    }
}

/* Creation-time form: also settles the size-tracking invariant. */
static void tpht_set_write_limit(tpht_table_t *t) {
    t->tracks_size = t->cfg.resize_mode == TPHT_FIXED ? 0u : 1u;
    tpht_refresh_write_limit(t);
}


static int tpht_alloc_storage(tpht_table_t *t, size_t capacity) {
    size_t overflow_slots;

    t->capacity = tpht_max_size(capacity, TPHT_MIN_CAPACITY);
    tpht_set_write_limit(t);
    t->key_size = t->cfg.key_size;
    t->value_size = t->cfg.value_size;
    t->key_bits = (uint8_t)(t->key_size * 8u);
    t->key_mask = t->key_bits >= 64u ? UINT64_MAX : ((UINT64_C(1) << t->key_bits) - 1u);
    /*
     * Three hash streams: one for quotienting, two for the dereference table's
     * two-choice allocation.  They are derived by mixing the table seed with a
     * stream index rather than by adding a small constant to it: adjacent seeds
     * produce bitflips that differ in only a few bits, and the two-choice
     * placement then measurably correlates, which is exactly the independence
     * the scheme depends on.
     */
    t->hash_bitflip = tpht_stream_bitflip(t->cfg.hash_seed, 0u);
    t->hash_bitflip_100 = tpht_stream_bitflip(t->cfg.hash_seed, 1u);
    t->hash_bitflip_200 = tpht_stream_bitflip(t->cfg.hash_seed, 2u);

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
        tpht_advise_hugepages(t->heads, t->base_count);
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
            /*
             * Keep the quotient inside 63 bits so the lookup path can shift by
             * it unconditionally.  The bound is 2^55 blocks - exabytes of home
             * array - so it never binds on a table that can be allocated.
             */
            if (cloud_bits + TPHT_FLAT_FP_BITS > 63u) {
                cloud_bits = (uint8_t)(63u - TPHT_FLAT_FP_BITS);
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
        atomic_store_explicit(&t->flat_lines_pack, tpht_flat_pack_of(t), memory_order_release);
        t->flat_cost = t->flat_entry_size > 1u ? (uint8_t)(t->flat_entry_size - 1u) : 1u;
        t->flat_inline_ok = 0u;
        for (x = 0; x < TPHT_FLAT_MAX_TUPLES; ++x) {
            if (tpht_flat_crystals_for(t->flat_entry_size, x + 1u) ==
                (uint32_t)tpht_flat_crystals_for(t->flat_entry_size, x) + 1u)
                t->flat_inline_ok |= UINT32_C(1) << x;
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

    t->pool.bin_count = (overflow_slots + TPHT_BIN_SIZE - 1u) / TPHT_BIN_SIZE;
    t->pool.bin_count = tpht_max_size(t->pool.bin_count, 1);
    t->pool.meta_shift = t->cfg.threading == TPHT_CONCURRENT ? TPHT_POOL_META_SHIFT_CONC
                                                             : TPHT_POOL_META_SHIFT_SEQ;
    t->pool.entries = (uint8_t *)calloc(
        t->pool.bin_count * (size_t)TPHT_BIN_SIZE * t->pool.entry_size + TPHT_LOAD_SLACK, 1);
    tpht_advise_hugepages(t->pool.entries,
                          t->pool.bin_count * (size_t)TPHT_BIN_SIZE * t->pool.entry_size +
                              TPHT_LOAD_SLACK);
    t->pool.cnt_head =
        (uint8_t *)calloc(t->pool.bin_count << t->pool.meta_shift, 1);
    tpht_advise_hugepages(t->pool.cnt_head, t->pool.bin_count << t->pool.meta_shift);
    t->pool.locked = t->cfg.threading == TPHT_CONCURRENT ? 1u : 0u;

    if (t->cfg.threading == TPHT_CONCURRENT) {
        size_t i;
        /* The flattened variant serialises a block with its own seqlock and
         * only needs the dereference bins locked; the chained one locks the
         * base chains as well. */
        t->chain_lock_count = t->cfg.variant == TPHT_CHAINED
                                  ? (t->base_count < TPHT_CHAIN_VERSION_SLOTS
                                         ? t->base_count
                                         : TPHT_CHAIN_VERSION_SLOTS)
                                  : 1u;
        t->chain_version_mask = t->chain_lock_count - 1u;
        t->chain_locks = (atomic_uchar *)calloc(TPHT_CHAIN_SLOT(t->chain_lock_count),
                                                sizeof(*t->chain_locks));
        if (t->chain_locks) {
            for (i = 0; i < t->chain_lock_count; ++i)
                atomic_init(&t->chain_locks[TPHT_CHAIN_SLOT(i)], 0u);
        }
        if (t->pool.cnt_head) {
            for (i = 0; i < t->pool.bin_count; ++i)
                atomic_flag_clear(
                    (atomic_flag *)&t->pool.cnt_head[(i << t->pool.meta_shift) |
                                                     TPHT_POOL_META_LOCK]);
        }
    }

    if (!t->pool.entries || !t->pool.cnt_head ||
        (t->cfg.threading == TPHT_CONCURRENT && !t->chain_locks)) {
        tpht_free_storage(t);
        return 0;
    }

    if (!t->no_prefault) {
        tpht_prefault_arr(t->heads, t->base_count);
        tpht_prefault_arr(t->flat_lines_raw,
                          t->cfg.variant == TPHT_FLATTEN
                              ? t->base_count * (size_t)TPHT_FLAT_LINE_BYTES +
                                    TPHT_FLAT_LINE_BYTES - 1u
                              : 0u);
        tpht_prefault_arr(t->pool.entries,
                          t->pool.bin_count * (size_t)TPHT_BIN_SIZE * t->pool.entry_size);
        tpht_prefault_arr(t->pool.cnt_head, t->pool.bin_count << t->pool.meta_shift);
    }
    return 1;
}




TPHT_HOT tpht_status_t tpht_chained_insert_raw(tpht_table_t *t, const void *key,
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
    if (!entry) return TPHT_OVERFLOW;
    *prev = encoded;
    entry[0] = 0;
#if TPHT_LITTLE_ENDIAN
    /*
     * Whole-value entries need no masking: the quotient goes down as a plain
     * 8-byte word and the value's own 8-byte store overwrites the bytes above
     * it, ending exactly at the entry's last byte.  Same trick as
     * tpht_flat_write_payload.
     */
    if (t->value_size == 8u && t->key_quotient_size + t->value_size >= 8u) {
        uint64_t v8 = tpht_read_le(value, 8u);
        memcpy(entry + 1u, &key_quot, 8);
        memcpy(entry + 1u + t->key_quotient_size, &v8, 8);
    } else
#endif
#if TPHT_LITTLE_ENDIAN && defined(__SIZEOF_INT128__)
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
        tpht_store_le(entry + 1u + t->key_quotient_size, t->value_size,
                      tpht_read_le(value, t->value_size));
    }
    if (t->tracks_size) tpht_size_inc(t);
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
    if (t->tracks_size) tpht_size_dec(t);
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
 * Block layout, 64 bytes:
 *
 *   [0 .. count)          fingerprint array, one byte per tuple, at most 31
 *   ...                   free space
 *   [end - tps .. end)    tiny pointers, one byte each, growing down from end
 *   [end .. 61)           inline tuples ("crystals"), entry_size bytes each,
 *                         also growing down, so end = 61 - crystals*entry_size
 *   byte 61               tuple count, in the low 5 bits
 *   byte 62               crystal count
 *   byte 63               reserved: the seqlock version a concurrent variant
 *                         will need.  Never read, never written.
 *
 * Bytes 61 and 62 are adjacent and are read and written as one 16-bit field,
 * so both counts arrive with the block's own cache miss and the ordinary
 * insert bumps them with a single add.  Holding the crystal count outright
 * rather than deriving it from the tuple count costs a byte of payload space
 * and buys back the dependent table load that derivation would put after the
 * miss on every insert, lookup and remove.  Insertion reduces to comparing
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
 *
 * TinyPtr fits both counts into a single control byte - a 3-bit crystal count
 * and a 5-bit tiny pointer count - and reserves one byte for concurrency, so
 * it keeps 62 bytes of payload to TPHT's 61.  Three bits suffice there only
 * because its values are always 8 bytes, which caps inline tuples at 7; TPHT
 * supports 1-byte values, where a block can hold about twenty, and capping
 * those at 7 was measured to roughly double the memory a small-value table
 * needs.  The byte of payload is the cheaper thing to spend.
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
    /* flat_quot_bits is held below 64 at construction, so no guard is needed
     * on the shift that starts every lookup and every insert. */
    uint64_t rem = key >> t->flat_quot_bits;
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

/*
 * Tuple count and crystal count sit in the two adjacent bytes 62 and 63, which
 * this variant reads and writes as one 16-bit field: everything a block
 * operation needs arrives with the line's own cache miss, and the ordinary
 * inline insert - both counts up by one - is a single `add word [line+62],
 * 0x0101`, the same one instruction TinyPtr spends on its packed control byte.
 * Byte 63 is left alone: it is the seqlock version a concurrent flattened
 * variant will need, and spending it on metadata would make that impossible to
 * add later without changing the on-disk shape of every block.
 */

TPHT_HOT uint8_t tpht_flat_count(const uint8_t *line) {
    return (uint8_t)(line[TPHT_FLAT_CONTROL_OFF] & TPHT_FLAT_COUNT_MASK);
}

TPHT_HOT uint8_t tpht_flat_crystals(const tpht_table_t *t, const uint8_t *line) {
    (void)t;
    return line[TPHT_FLAT_CRYSTALS_OFF];
}

/*
 * Both counts as one value: tuple count in the low byte, crystal count in the
 * high.  They live in adjacent bytes, so this is a single 16-bit load that
 * arrives with the block's own cache miss - nothing downstream waits on a
 * second access, and no table has to be consulted to learn how many tuples the
 * block holds inline.
 */
TPHT_HOT uint16_t tpht_flat_meta(const uint8_t *line) {
#if TPHT_LITTLE_ENDIAN
    uint16_t m;
    memcpy(&m, line + TPHT_FLAT_CONTROL_OFF, 2);
    return m;
#else
    return (uint16_t)(line[TPHT_FLAT_CONTROL_OFF] |
                      ((uint16_t)line[TPHT_FLAT_CRYSTALS_OFF] << 8u));
#endif
}

TPHT_HOT uint8_t tpht_flat_meta_count(uint16_t meta) {
    return (uint8_t)(meta & TPHT_FLAT_COUNT_MASK);
}

TPHT_HOT uint8_t tpht_flat_meta_crystals(uint16_t meta) { return (uint8_t)(meta >> 8u); }

/* Append one inline tuple: one 16-bit add, both counts at once. */
TPHT_HOT void tpht_flat_add_crystal(uint8_t *line) {
#if TPHT_LITTLE_ENDIAN
    uint16_t m;
    memcpy(&m, line + TPHT_FLAT_CONTROL_OFF, 2);
    m = (uint16_t)(m + TPHT_FLAT_META_CRYSTAL);
    memcpy(line + TPHT_FLAT_CONTROL_OFF, &m, 2);
#else
    line[TPHT_FLAT_CONTROL_OFF] = (uint8_t)(line[TPHT_FLAT_CONTROL_OFF] + 1u);
    line[TPHT_FLAT_CRYSTALS_OFF] = (uint8_t)(line[TPHT_FLAT_CRYSTALS_OFF] + 1u);
#endif
}

TPHT_HOT void tpht_flat_set_meta(uint8_t *line, uint8_t count, uint8_t crystals) {
#if TPHT_LITTLE_ENDIAN
    uint16_t m = (uint16_t)((count & TPHT_FLAT_COUNT_MASK) | ((uint16_t)crystals << 8u));
    memcpy(line + TPHT_FLAT_CONTROL_OFF, &m, 2);
#else
    line[TPHT_FLAT_CONTROL_OFF] = (uint8_t)(count & TPHT_FLAT_COUNT_MASK);
    line[TPHT_FLAT_CRYSTALS_OFF] = crystals;
#endif
}

/*
 * Per-block seqlock, in the byte reserved for it.  Even means the block is
 * stable; odd means a writer is inside it.  A writer takes the block by moving
 * the version from even to odd and releases it by moving it on to the next
 * even value, so a reader that sees the same even version before and after its
 * read knows nothing changed underneath it.
 *
 * `concurrent` is a literal at every call site, so a sequential table compiles
 * all of this away and pays nothing - the same way the key width and the
 * overwrite flag fold.
 */
TPHT_HOT atomic_uchar *tpht_flat_version(uint8_t *line) {
    return (atomic_uchar *)(void *)(line + TPHT_FLAT_VERSION_OFF);
}

TPHT_HOT void tpht_flat_write_begin(uint8_t *line, int concurrent) {
    if (!concurrent) {
        (void)line;
        return;
    }
    {
        /*
         * Taking the block is one read-modify-write, not a load followed by a
         * compare-exchange.  Setting the low bit is idempotent, so the returned
         * value alone says whether we took it: even means it was ours, odd
         * means someone else already held it and nothing was disturbed.  The
         * load-then-CAS form fetched the line shared and then had to upgrade it
         * to exclusive - two coherence round trips on the block's first touch,
         * which measured as two thirds of an insert.
         */
        atomic_uchar *v = tpht_flat_version(line);
        for (;;) {
            if (!tpht_bit_test_and_set(v)) return;
            /* Spin read-only so a waiter does not keep stealing the line. */
            do {
                tpht_cpu_relax();
            } while (atomic_load_explicit(v, memory_order_relaxed) & 1u);
        }
    }
}

TPHT_HOT void tpht_flat_write_end(uint8_t *line, int concurrent) {
    if (!concurrent) {
        (void)line;
        return;
    }
    {
        /*
         * The release is deliberately a plain store, not a locked increment.
         * Measured on Xeon 6980P at 32-64 threads, a locked release is 15-24%
         * slower - a second locked RMW on a contended line - and a concurrent
         * table is built for the contended case.  (Single-thread comparisons
         * between the two proved too noise-prone to support any claim; a
         * single-threaded caller has the sequential variant, which beats both.)
         * Do not "fix" this to fetch_add without re-measuring under threads.
         */
        atomic_uchar *v = tpht_flat_version(line);
        atomic_store_explicit(v, (unsigned char)(atomic_load_explicit(v, memory_order_relaxed) + 1u),
                              memory_order_release);
    }
}

/* Snapshot for a reader: an even version, waiting out any writer in the block. */
TPHT_HOT unsigned char tpht_flat_read_begin(const uint8_t *line, int concurrent) {
    if (!concurrent) {
        (void)line;
        return 0u;
    }
    {
        atomic_uchar *v = tpht_flat_version((uint8_t *)(void *)(uintptr_t)line);
        for (;;) {
            unsigned char cur = atomic_load_explicit(v, memory_order_acquire);
            if (!(cur & 1u)) return cur;
            tpht_cpu_relax();
        }
    }
}

/* True when the block did not change while the reader was inside it. */
TPHT_HOT int tpht_flat_read_valid(const uint8_t *line, unsigned char snapshot, int concurrent) {
    if (!concurrent) {
        (void)line;
        (void)snapshot;
        return 1;
    }
    return atomic_load_explicit(tpht_flat_version((uint8_t *)(void *)(uintptr_t)line),
                                memory_order_acquire) == snapshot;
}

/* First byte past the crystal region, and the anchor of the tiny pointers. */
TPHT_HOT uint8_t tpht_flat_crystal_end(const tpht_table_t *t, uint8_t crystals) {
    return (uint8_t)(TPHT_FLAT_CONTROL_OFF - (unsigned)crystals * t->flat_entry_size);
}

TPHT_HOT uint8_t *tpht_flat_crystal(const tpht_table_t *t, uint8_t *line, uint8_t i) {
    return line + TPHT_FLAT_CONTROL_OFF - ((unsigned)i + 1u) * t->flat_entry_size;
}

static uint8_t *tpht_flat_tp_slot(uint8_t *line, uint8_t crystal_end, uint8_t j) {
    return line + crystal_end - j - 1u;
}

/*
 * Move n <= 63 bytes inside one home block, where the ranges may overlap.
 * memmove would do this correctly, but its length is a run-time value, so the
 * compiler emits a call into libc on the hottest path in the table.  Every form
 * below reads all of the source before writing any of the destination, so
 * overlap is safe, and none of them touches a byte outside [dst, dst + n) -
 * the neighbouring bytes are live block data.
 */
TPHT_HOT void tpht_flat_move(uint8_t *dst, const uint8_t *src, uint8_t n) {
    /*
     * A masked AVX-512 load/store pair expresses this in two instructions, but
     * measures slower than the scalar form below: the store's mask is a late
     * dependency and the masked store forwards poorly to the loads that follow
     * it in the next insert.  The plain integer moves win, so every level uses
     * them.
     */
    {
    /*
     * Two overlapping power-of-two moves cover exactly n bytes for any n in
     * [k, 2k): the pair meets in the middle and writes the seam twice.
     */
    if (n >= 16u) {
        /* Scalars, not a buffer: a local array here would put a stack
         * protector canary on every insert this is inlined into. */
        uint64_t a0, a1, b0, b1;
        memcpy(&a0, src, 8);
        memcpy(&a1, src + 8, 8);
        memcpy(&b0, src + n - 16u, 8);
        memcpy(&b1, src + n - 8u, 8);
        memcpy(dst, &a0, 8);
        memcpy(dst + 8, &a1, 8);
        memcpy(dst + n - 16u, &b0, 8);
        memcpy(dst + n - 8u, &b1, 8);
    } else if (n >= 8u) {
        uint64_t a, b;
        memcpy(&a, src, 8);
        memcpy(&b, src + n - 8u, 8);
        memcpy(dst, &a, 8);
        memcpy(dst + n - 8u, &b, 8);
    } else if (n >= 4u) {
        uint32_t a, b;
        memcpy(&a, src, 4);
        memcpy(&b, src + n - 4u, 4);
        memcpy(dst, &a, 4);
        memcpy(dst + n - 4u, &b, 4);
    } else if (n >= 2u) {
        uint16_t a, b;
        memcpy(&a, src, 2);
        memcpy(&b, src + n - 2u, 2);
        memcpy(dst, &a, 2);
        memcpy(dst + n - 2u, &b, 2);
    } else if (n) {
        dst[0] = src[0];
    }
    }
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
#if TPHT_LITTLE_ENDIAN
    /*
     * Whole-value entries need no masking at all: store the remainder as a
     * plain 8-byte word, then let the value's own 8-byte store overwrite the
     * bytes above it.  The two stores overlap exactly where the remainder ends,
     * so the entry ends up correct without a single shift or mask.  The value
     * store ends at payload + flat_entry_size, still inside the entry.
     */
    if (t->value_size == 8u && t->flat_entry_size >= 8u) {
        memcpy(payload, &rem, 8);
        memcpy(payload + t->flat_qkey_bytes, &value, 8);
        return;
    }
#endif
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
static int tpht_flat_resize_active(const tpht_table_t *t);
static int tpht_flat_conc_resize_start(tpht_table_t *t, size_t new_capacity, uint32_t growth,
                                        size_t deref_floor, int block);
static void tpht_flat_resize_finish_all(tpht_table_t *t);
static void tpht_flat_resize_help_one(tpht_table_t *t);
static tpht_resize_op_t *tpht_resize_op_snapshot(tpht_table_t *t);
static void tpht_flat_shadow_write(tpht_resize_op_t *op, uint64_t key, uint64_t value,
                                   int replace, unsigned key_bytes);
static void tpht_flat_shadow_remove(tpht_resize_op_t *op, uint64_t key, unsigned key_bytes);

TPHT_HOT tpht_status_t tpht_flat_get_raw(tpht_table_t *t, uint64_t key, uint64_t *value_out,
                                       unsigned key_bytes, int concurrent) {
    tpht_flat_loc_t loc;
    uint8_t *line;
    tpht_table_t *g = t;

    /*
     * The whole walk runs off one immutable geometry snapshot (see geo_snap):
     * a reader holds no lock, so the table's own fields can be mid-swap under
     * a resize commit, and any pointer composed from a torn pair of them can
     * point outside every allocation.  The snapshot cannot tear.  For a
     * sequential table it is the table itself and `g` folds back to `t`.
     */
    if (concurrent) g = atomic_load_explicit(&t->geo_snap, memory_order_acquire);
    tpht_flat_locate(g, key, &loc, key_bytes);
    line = tpht_flat_line(g, loc.block);

    /*
     * A concurrent reader takes the block's version, reads without any lock,
     * and only trusts what it found if the version is still the one it started
     * from; a writer that touched the block in between forces another attempt.
     * A resize never stalls a reader: the old storage stays complete for the
     * whole migration (every write during one lands in the old block first),
     * so the walk below is valid whether a resize is running or not, and the
     * snapshot re-check catches a commit that swapped the storage mid-walk.
     * For a sequential table all of this folds away and the loop runs exactly
     * once, leaving the code below identical to what it was.
     */
    for (;;) {
        unsigned char snapshot;
        snapshot = tpht_flat_read_begin(line, concurrent);
        uint8_t count = tpht_flat_count(line);
        uint8_t crystals = tpht_flat_crystals(g, line);
        uint32_t mask = tpht_fp_match_mask(line, count, loc.fp);
        uint8_t crystal_end = tpht_flat_crystal_end(g, crystals);
        uint32_t tp_mask;
        int found = 0;
        uint64_t value = 0;

        /*
         * Split the match mask once instead of asking "is this one inline?" per
         * candidate.  Fingerprints are ordered crystals first, so the low
         * `crystals` bits are the inline matches and the rest are overflow ones.
         * That keeps the common loop free of a data-dependent branch and keeps
         * the dereference out of it entirely.
         */
        tp_mask = crystals >= 32u ? 0u : (mask >> crystals);
        mask &= crystals >= 32u ? UINT32_MAX : ((UINT32_C(1) << crystals) - 1u);

        while (mask) {
            uint8_t i = tpht_ctz32(mask);
            const uint8_t *payload = tpht_flat_crystal(g, line, i);
            mask &= mask - 1u;
            if (tpht_flat_read_rem(g, payload) == loc.rem) {
                /* A sequential table cannot see a torn block, so it answers
                 * straight from the loop as it always did; only the concurrent
                 * form has to carry the result out to the version check. */
                if (!concurrent) {
                    *value_out = tpht_flat_read_value(g, payload);
                    return TPHT_OK;
                }
                value = tpht_flat_read_value(g, payload);
                found = 1;
                break;
            }
        }

        while (!found && tp_mask) {
            uint8_t j = tpht_ctz32(tp_mask);
            uint8_t encoded = *tpht_flat_tp_slot(line, crystal_end, j);
            const uint8_t *payload;
            /*
             * A tiny pointer's low seven bits are a slot number plus one, so a
             * zero there is not a slot at all - and the subtraction would wrap
             * to 255, addressing far past the bin.  A lock-free reader can see
             * that while a writer is rearranging the block, so it stops here
             * and lets the version check send it round again.  A sequential
             * table never observes a half-written block, and the test folds.
             */
            if (concurrent && (encoded & 0x7fu) == 0u) break;
            payload = tpht_pool_deref(g, tpht_flat_deref_key(loc.block, loc.fp), encoded,
                                      key_bytes);
            tp_mask &= tp_mask - 1u;
            if (tpht_flat_read_rem(g, payload) == loc.rem) {
                if (!concurrent) {
                    *value_out = tpht_flat_read_value(g, payload);
                    return TPHT_OK;
                }
                value = tpht_flat_read_value(g, payload);
                found = 1;
            }
        }

        /* Everything read above is only real if the block held still... */
        if (!tpht_flat_read_valid(line, snapshot, concurrent)) continue;
        /* ...and only if the storage itself was not swapped mid-walk. */
        if (concurrent &&
            TPHT_UNLIKELY(atomic_load_explicit(&t->geo_snap, memory_order_acquire) != g)) {
            g = atomic_load_explicit(&t->geo_snap, memory_order_acquire);
            tpht_flat_locate(g, key, &loc, key_bytes);
            line = tpht_flat_line(g, loc.block);
            continue;
        }
        if (!found) return TPHT_NOT_FOUND;
        *value_out = value;
        return TPHT_OK;
    }
}

/*
 * Soft overflow, deliberately out of line: it runs for a minority of inserts,
 * but its eviction buffers would otherwise bloat every insert's stack frame
 * and register saves - a measurable cost on the majority path.
 */
TPHT_NOINLINE static tpht_status_t tpht_flat_insert_soft(tpht_table_t *t, uint8_t *line,
                                                         tpht_flat_loc_t loc,
                                                         uint64_t value, uint8_t count,
                                                         uint8_t crystals, uint8_t crystal_end,
                                                         unsigned key_bytes, int concurrent) {
    uint8_t next_crystals = t->flat_crystals[count + 1u];
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
                atomic_store_explicit(&t->flat_deref_pressure, 1u, memory_order_relaxed);
                return TPHT_OVERFLOW;
            }
        }

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
        tpht_flat_set_meta(line, (uint8_t)(count + 1u), next_crystals);
        /* The caller still holds this block, so a concurrent table's counter
         * moves inside the critical section: the resize commit holds every
         * block seqlock while it overwrites size with the shadow's count, and
         * an increment left pending outside the lock would land after that
         * store and count its key twice - or be overwritten and count it not
         * at all.  Inside the lock, commit and counter cannot interleave. */
        if (t->tracks_size) {
            if (concurrent)
                tpht_size_inc_conc(t);
            else
                tpht_size_inc_seq(t);
        }
        return TPHT_OK;
    }
}

/*
 * Force-inlined so a constant `replace` from the caller folds the existence
 * probe (and its SIMD fingerprint match setup) out of the append path.
 */
TPHT_HOT tpht_status_t tpht_flat_insert_raw(tpht_table_t *t, uint64_t key, uint64_t value,
                                            int replace, unsigned key_bytes, int concurrent) {
    tpht_flat_loc_t loc;
    tpht_flat_slot_t slot;
    uint8_t *line;
    uint16_t meta;
    uint8_t count;
    uint8_t crystals;
    uint8_t crystal_end;
    tpht_status_t st;
    tpht_resize_op_t *shadow_op = NULL;
    uint64_t pack = 0;

    /*
     * The line pointer must be memory-safe even against a half-observed
     * resize commit, and a writer holds no lock yet.  flat_lines_pack carries
     * the line array and its block-count log2 in one atomic word: clamping
     * the located block by the packed bit count keeps the pointer inside the
     * packed array whatever torn mask values the locate read.  Torn values
     * can still pick a wrong (in-bounds) block - the pack re-read under the
     * lock catches exactly those interleavings and retries.
     */
    if (concurrent) pack = atomic_load_explicit(&t->flat_lines_pack, memory_order_acquire);
    tpht_flat_locate(t, key, &loc, key_bytes);
    if (concurrent) {
        /*
         * The range check, not a clamp: a clamp sat on the address dependency
         * chain of the block lock and measurably slowed every insert, while
         * this branch runs beside it and predicts perfectly - it only fires
         * when the locate's mask raced a resize commit, in which case the
         * caller retries with fresh values.  In-bounds-but-wrong blocks from
         * the same race are caught by the pack re-read under the lock.
         */
        if (TPHT_UNLIKELY(loc.block >> (pack & 63u))) return TPHT_RETRY;
        line = (uint8_t *)(uintptr_t)(pack & ~UINT64_C(63)) +
               ((size_t)loc.block << TPHT_FLAT_LINE_SHIFT);
    } else {
        line = tpht_flat_line(t, loc.block);
    }

    /*
     * The block is taken before anything is read from it, not just before the
     * write: deciding where a tuple goes reads the counts, and two writers that
     * read the same counts would both write the same slot.  The whole
     * read-decide-write sequence has to be the critical section.
     */
    tpht_flat_write_begin(line, concurrent);
    if (concurrent) {
        /*
         * The lock means nothing if it was taken in a storage that a resize
         * commit has replaced: the pack re-read under the lock says whether
         * one landed in between - one word comparison, exact because storages
         * are never address-reused while the table lives.  And with the lock
         * held the verdict is final: a commit takes every old line lock
         * before storing a single field, so a stable pack means every
         * geometry field the locate read was stable too.
         *
         * During an active resize the old storage stays authoritative - every
         * write lands here first - so writers never stall for the migration.
         * If this block was already copied to the shadow, the write is applied
         * there as well before the block is released, keeping the shadow an
         * exact mirror of every migrated block; the commit then publishes a
         * storage that already contains this write.  The mirror is gated on
         * the descriptor's own starting pack: the flag alone can name a
         * finished resize during its commit's tail, whose shadow is gone.
         */
        if (TPHT_UNLIKELY(atomic_load_explicit(&t->flat_lines_pack, memory_order_acquire) !=
                          pack)) {
            tpht_flat_write_end(line, concurrent);
            return TPHT_RETRY;
        }
        if (TPHT_UNLIKELY(tpht_flat_resize_active(t))) {
            tpht_resize_op_t *op = tpht_resize_op_snapshot(t);
            if (TPHT_UNLIKELY(!op || op->old_pack != pack)) {
                tpht_flat_write_end(line, concurrent);
                return TPHT_RETRY;
            }
            if (op->migrated[loc.block]) shadow_op = op;
        }
    }

    /*
     * `replace` selects the write's semantics, and is a literal at every call
     * site so the arms that do not apply disappear:
     *   0  append unconditionally - no existence probe at all
     *   1  overwrite if present, else append
     *   2  overwrite only, and report a missing key rather than adding it
     *
     * Mode 2 exists so an update is a single critical section.  It used to be a
     * lookup followed by a separate overwrite, and a key removed between the
     * two would fall through to the append path - an update on an absent key
     * would create it, which is exactly what update promises not to do.
     */
    if (replace && tpht_flat_find(t, line, &loc, &slot, key_bytes)) {
        tpht_flat_write_value(t, slot.payload, value);
        if (concurrent && TPHT_UNLIKELY(shadow_op != NULL))
            tpht_flat_shadow_write(shadow_op, key, value, 1, key_bytes);
        tpht_flat_write_end(line, concurrent);
        return TPHT_OK;
    }
    if (replace == 2) {
        tpht_flat_write_end(line, concurrent);
        return TPHT_NOT_FOUND;
    }

    /* Both counts arrive in one 16-bit load, with the line's own cache miss. */
    meta = tpht_flat_meta(line);
    count = tpht_flat_meta_count(meta);
    /* Hard overflow: the block is at capacity (see TPHT_FLAT_MAX_TUPLES). */
    if (TPHT_UNLIKELY(count >= TPHT_FLAT_MAX_TUPLES)) {
        atomic_store_explicit(&t->flat_deref_pressure, 0u, memory_order_relaxed);
        tpht_flat_write_end(line, concurrent);
        return TPHT_OVERFLOW;
    }

    crystals = tpht_flat_meta_crystals(meta);
    crystal_end = tpht_flat_crystal_end(t, crystals);

    /*
     * One more tuple fits inline iff crystals_for(count + 1) == crystals + 1,
     * which unfolds (see tpht_flat_crystals_for) to the byte-budget check
     * below - arithmetic on values already in registers, in place of the
     * side-table load the cold path still uses.
     */
    if ((t->flat_inline_ok >> count) & 1u) {
        /* The pair itself still fits in the line. */
        uint8_t tps = (uint8_t)(count - crystals);
        uint8_t new_end = (uint8_t)(crystal_end - t->flat_entry_size);
        if (tps) {
            /* Tiny pointers stay anchored to the crystal region, and the new
             * crystal's fingerprint takes index `crystals`. */
            tpht_flat_move(line + new_end - tps, line + crystal_end - tps, tps);
            tpht_flat_move(line + crystals + 1u, line + crystals, tps);
        }
        line[crystals] = loc.fp;
        tpht_flat_write_payload(t, line + new_end, loc.rem, value);
        /* Both counts up by one: one add on the 16-bit meta field. */
        tpht_flat_add_crystal(line);
        /* Inside the critical section, for the same reason as in
         * tpht_flat_insert_soft: the resize commit overwrites size under all
         * block seqlocks, and a pending increment outside them is either
         * double-counted or lost against that store. */
        if (t->tracks_size) {
            if (concurrent)
                tpht_size_inc_conc(t);
            else
                tpht_size_inc_seq(t);
        }
        /* Mirror into the shadow while this block is still held: the resize
         * commit and abort both take every old block lock first, so the shadow
         * cannot be published or freed under this call. */
        if (concurrent && TPHT_UNLIKELY(shadow_op != NULL))
            tpht_flat_shadow_write(shadow_op, key, value, 0, key_bytes);
        tpht_flat_write_end(line, concurrent);
        return TPHT_OK;
    }

    /* Sequential form keeps its tail call; only the concurrent one has to come
     * back here to release the block. */
    if (!concurrent)
        return tpht_flat_insert_soft(t, line, loc, value, count, crystals, crystal_end,
                                     key_bytes, concurrent);
    st = tpht_flat_insert_soft(t, line, loc, value, count, crystals, crystal_end, key_bytes,
                               concurrent);
    if (TPHT_UNLIKELY(shadow_op != NULL) && st == TPHT_OK)
        tpht_flat_shadow_write(shadow_op, key, value, 0, key_bytes);
    tpht_flat_write_end(line, concurrent);
    return st;
}

static tpht_status_t tpht_flat_remove_raw(tpht_table_t *t, uint64_t key, unsigned key_bytes,
                                          int concurrent) {
    tpht_flat_loc_t loc;
    tpht_flat_slot_t slot;
    uint8_t *line;
    uint8_t count;
    uint8_t crystals;
    uint8_t tps;
    uint8_t crystal_end;
    uint8_t left;
    tpht_resize_op_t *shadow_op;
    uint64_t pack;

retry_conc:
    shadow_op = NULL;
    pack = 0;
    if (concurrent) pack = atomic_load_explicit(&t->flat_lines_pack, memory_order_acquire);
    tpht_flat_locate(t, key, &loc, key_bytes);
    if (concurrent) {
        /* Range check off the address chain, as in insert. */
        if (TPHT_UNLIKELY(loc.block >> (pack & 63u))) goto retry_conc;
        line = (uint8_t *)(uintptr_t)(pack & ~UINT64_C(63)) +
               ((size_t)loc.block << TPHT_FLAT_LINE_SHIFT);
    } else {
        line = tpht_flat_line(t, loc.block);
    }
    /* As in insert: locating the victim reads the block, so the block is taken
     * first and the search happens inside the critical section; the packed
     * check and the mirror-into-the-shadow rule are the same. */
    tpht_flat_write_begin(line, concurrent);
    if (concurrent) {
        if (TPHT_UNLIKELY(atomic_load_explicit(&t->flat_lines_pack, memory_order_acquire) !=
                          pack)) {
            tpht_flat_write_end(line, concurrent);
            goto retry_conc;
        }
        if (TPHT_UNLIKELY(tpht_flat_resize_active(t))) {
            /* As in insert: only the descriptor whose starting pack is ours
             * names a live resize. */
            tpht_resize_op_t *op = tpht_resize_op_snapshot(t);
            if (TPHT_UNLIKELY(!op || op->old_pack != pack)) {
                tpht_flat_write_end(line, concurrent);
                goto retry_conc;
            }
            if (op->migrated[loc.block]) shadow_op = op;
        }
    }
    if (!tpht_flat_find(t, line, &loc, &slot, key_bytes)) {
        tpht_flat_write_end(line, concurrent);
        return TPHT_NOT_FOUND;
    }

    count = tpht_flat_count(line);
    crystals = tpht_flat_crystals(t, line);
    tps = (uint8_t)(count - crystals);
    crystal_end = tpht_flat_crystal_end(t, crystals);
    left = crystals;
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
    /* The crystal count is a function of the tuple count alone; promote pulls
     * dereference entries back inline until the line matches it again. */
    tpht_flat_set_meta(line, (uint8_t)(count - 1u), t->flat_crystals[count - 1u]);
    tpht_flat_promote(t, line, loc.block, left, t->flat_crystals[count - 1u], key_bytes);
    /* Atomic and inside the critical section for a concurrent table, as in the
     * insert paths: the counter must not race other writers or the commit. */
    if (t->tracks_size) {
        if (concurrent)
            tpht_size_dec_conc(t);
        else
            tpht_size_dec_seq(t);
    }
    if (concurrent && TPHT_UNLIKELY(shadow_op != NULL))
        tpht_flat_shadow_remove(shadow_op, key, key_bytes);
    tpht_flat_write_end(line, concurrent);
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
            /* dst is this thread's private shadow: no seqlock needed. */
            st = tpht_flat_insert_raw(dst, key, tpht_flat_read_value(src, payload), 0, key_bytes, 0);
            if (st != TPHT_OK) return st;
        }
    }
    return TPHT_OK;
}

static void tpht_flat_adopt(tpht_table_t *t, tpht_table_t *nt) {
    tpht_free_storage(t);
    t->capacity = nt->capacity;
    tpht_set_write_limit(t);
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
    t->flat_cost = nt->flat_cost;
    t->flat_inline_ok = nt->flat_inline_ok;
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
    atomic_store_explicit(&t->flat_lines_pack, tpht_flat_pack_of(t), memory_order_release);
    nt->flat_lines = NULL;
    nt->flat_lines_raw = NULL;
    nt->pool.entries = NULL;
    nt->pool.cnt_head = NULL;
}

static void tpht_flat_init_shadow(tpht_table_t *nt, const tpht_table_t *t) {
    memset(nt, 0, sizeof(*nt));
    nt->cfg = t->cfg;
    { unsigned si; for (si = 0; si < TPHT_SIZE_SHARDS; ++si) atomic_init(&nt->size_shard[si].v, 0); }
    atomic_init(&nt->geo_snap, nt);
    nt->initial_geo = NULL;
    atomic_init(&nt->resize_active, 0);
    atomic_init(&nt->resize_op, NULL);
    atomic_flag_clear(&nt->lock);
    atomic_flag_clear(&nt->resize_start_lock);
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
        if (st != TPHT_OVERFLOW) return st;
    }
    return TPHT_OVERFLOW;
}

/*
 * Absorb a hard overflow: rebuild with more home blocks, keeping the capacity
 * the caller asked for.  When the dereference table was the part that ran out,
 * make sure the rebuild does not hand it fewer slots than it had.
 */
static tpht_status_t tpht_flat_grow(tpht_table_t *t, unsigned key_bytes) {
    size_t slots = t->pool.bin_count * (size_t)TPHT_BIN_SIZE;
    size_t floor = atomic_load_explicit(&t->flat_deref_pressure, memory_order_relaxed)
                       ? slots * 2u
                       : slots;
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
/*
 * Everything that reshapes the table before or after an insert attempt lives
 * out of line: tpht_flat_insert_raw is force-inlined, so leaving these blocks
 * in the hot function would duplicate its whole body per retry site and spill
 * hot registers on every ordinary insert.
 */
static tpht_status_t tpht_flat_write_grow(tpht_table_t *t, uint64_t key, uint64_t value,
                                          int replace, unsigned key_bytes, int concurrent);

TPHT_NOINLINE static tpht_status_t tpht_flat_write_slow(tpht_table_t *t, uint64_t key,
                                                        uint64_t value, int replace,
                                                        unsigned key_bytes, int at_capacity,
                                                        int concurrent) {
    tpht_status_t st;
    if (at_capacity) {
        /*
         * insert is append-only: at capacity there is no room for another
         * entry, so it reports TPHT_OVERFLOW without probing.  Only an overwrite
         * (replace != 0) may still look the key up and touch it in place.
         */
        if (!replace) return TPHT_OVERFLOW;
        uint64_t scratch;
        st = tpht_flat_get_raw(t, key, &scratch, key_bytes, concurrent);
        if (st != TPHT_OK) return TPHT_OVERFLOW;
    } else {
        st = tpht_flat_resize(t, t->capacity * 2u, key_bytes);
        if (st != TPHT_OK) return st;
    }
    st = tpht_flat_insert_raw(t, key, value, replace, key_bytes, concurrent);
    if (st == TPHT_OVERFLOW)
        st = tpht_flat_write_grow(t, key, value, replace, key_bytes, concurrent);
    return st;
}

TPHT_NOINLINE static tpht_status_t tpht_flat_write_grow(tpht_table_t *t, uint64_t key,
                                                        uint64_t value, int replace,
                                                        unsigned key_bytes, int concurrent) {
    /*
     * Hard overflow: rebuild with more blocks and retry - in a bounded loop,
     * because a successful rebuild can still leave THIS key's new home block
     * saturated (dense tables re-deal every block's load), and a single
     * retry then reported a transient TPHT_OVERFLOW that one more grow would
     * have absorbed.  The bound matters: a block saturated by duplicates of
     * one key never splits however many blocks a rebuild adds, so without it
     * every round would succeed, double the block count, and fail the insert
     * again until the growth counter or memory ran out.
     */
    unsigned rounds;
    for (rounds = 0; rounds < 4u; ++rounds) {
        tpht_status_t st = tpht_flat_grow(t, key_bytes);
        if (st != TPHT_OK) return st;
        st = tpht_flat_insert_raw(t, key, value, replace, key_bytes, concurrent);
        if (st != TPHT_OVERFLOW) return st;
    }
    return TPHT_OVERFLOW;
}

TPHT_HOT tpht_status_t tpht_flat_write(tpht_table_t *t, uint64_t key, uint64_t value, int replace,
                                       unsigned key_bytes, int concurrent) {
    tpht_status_t st;

    /*
     * Fixed and resizable differ only in where the size trips a slow path, and
     * that point is settled when the table is created (see tpht_set_write_limit).
     * The hot path therefore tests one precomputed limit instead of branching on
     * the resize mode and recomputing a load-factor product per insert.
     */
    if (concurrent) {
        /*
         * Same growth policy as the chained variant, but a write never stalls
         * for a migration: it pays one stride of migration work as its help
         * fee when a resize is active, then applies itself to the old storage
         * (and to the shadow, if its block is already migrated - see
         * tpht_flat_insert_raw).  Only a hard overflow waits: it needs the
         * in-flight resize's larger geometry before it can succeed.
         */
        unsigned full_rounds = 0;
        for (;;) {
            if (TPHT_UNLIKELY(tpht_flat_resize_active(t))) tpht_flat_resize_help_one(t);
            if (t->tracks_size &&
                TPHT_UNLIKELY(++tpht_tls_limit_tick >=
                              atomic_load_explicit(&t->size_check_period, memory_order_relaxed))) {
                /* Trigger fields through the snapshot: no lock is held here,
                 * and a commit may be swapping the table's own copies.  Stale
                 * values at worst start a same-size resize, which the start
                 * path treats as a no-op cycle. */
                tpht_table_t *gl = atomic_load_explicit(&t->geo_snap, memory_order_acquire);
                tpht_tls_limit_tick = 0;
                if (tpht_size_load(t) >= gl->write_limit &&
                    !tpht_flat_resize_active(t)) {
                    /* Floor 0: a capacity doubling re-derives its pool from
                     * the new geometry, as the sequential resize does. */
                    if (!tpht_flat_conc_resize_start(t, gl->capacity * 2u, gl->flat_growth,
                                                     0u, 0))
                        return TPHT_NO_MEMORY;
                    continue;
                }
            }
            st = tpht_flat_insert_raw(t, key, value, replace, key_bytes, concurrent);
            if (TPHT_UNLIKELY(st == TPHT_RETRY)) continue; /* raced a commit */
            if (TPHT_UNLIKELY(st == TPHT_OVERFLOW)) {
                if (tpht_flat_resize_active(t)) {
                    /* The block cannot take another tuple until the running
                     * resize lands; finishing it is the fastest way there. */
                    tpht_flat_resize_finish_all(t);
                    continue;
                }
                /* Hard overflow: rebuild with more blocks at the same
                 * capacity, the concurrent way.  As in the sequential grow,
                 * the dereference table must not come back smaller - and when
                 * it was the part that ran out, it comes back doubled;
                 * without that, an overloaded fixed table can abort its
                 * absorb resizes forever (the pool sizing shrinks as blocks
                 * grow, while the keys it must hold do not).
                 *
                 * The escalation is bounded: unlike the sequential rebuild,
                 * whose failed attempts leave the table untouched, every
                 * concurrent escalation commits and doubles the block count
                 * for good.  Sixteen doublings past the base geometry is
                 * 65536x the blocks - beyond any overload growth can absorb.
                 * What remains full past that is a block that growth cannot
                 * split, i.e. TPHT_FLAT_MAX_TUPLES-bounded duplicates of one
                 * key (see tpht.h), and the honest answer is TPHT_OVERFLOW, not
                 * an allocation march into TPHT_NO_MEMORY. */
                {
                    tpht_table_t *gl = atomic_load_explicit(&t->geo_snap, memory_order_acquire);
                    size_t slots = gl->pool.bin_count * (size_t)TPHT_BIN_SIZE;
                    size_t fl = atomic_load_explicit(&t->flat_deref_pressure,
                                                     memory_order_relaxed)
                                    ? slots * 2u
                                    : slots;
                    /* An abort's escalation lives in the table's floor until a
                     * successful resize consumes it. */
                    fl = tpht_max_size(fl, t->flat_deref_floor);
                    /*
                     * When the key width already pins the block count (the
                     * geometry clamp: cloud bits + fingerprint bits = key
                     * bits), growing cannot add a single block - the resize
                     * would migrate the whole table and change nothing.  The
                     * per-block tuple ceiling is then structural, exactly as
                     * the sequential rebuild's same-geometry check concludes,
                     * and the honest answer is TPHT_OVERFLOW now, not after
                     * sixteen futile full-table migrations.
                     */
                    if (TPHT_UNLIKELY((unsigned)gl->flat_cloud_bits + TPHT_FLAT_FP_BITS >=
                                      t->key_bits))
                        return TPHT_OVERFLOW;
                    if (TPHT_UNLIKELY(gl->flat_growth >= 16u || ++full_rounds > 64u))
                        return TPHT_OVERFLOW;
                    if (!tpht_flat_conc_resize_start(t, gl->capacity,
                                                     (uint32_t)gl->flat_growth + 1u, fl, 1))
                        return TPHT_NO_MEMORY;
                }
                continue;
            }
            return st;
        }
    }

    if (TPHT_UNLIKELY(tpht_size_load_seq(t) >= t->write_limit))
        return tpht_flat_write_slow(t, key, value, replace, key_bytes,
                                    t->cfg.resize_mode == TPHT_FIXED, concurrent);

    st = tpht_flat_insert_raw(t, key, value, replace, key_bytes, concurrent);
    if (TPHT_UNLIKELY(st == TPHT_OVERFLOW)) {
        /*
         * A rebuild replaces every block and the whole dereference table, which
         * cannot be done underneath readers that hold no lock.  A concurrent
         * table therefore reports the overflow instead of absorbing it, as its
         * documentation says; only a sequential one grows here.  `concurrent`
         * is a literal at every call site, so one arm or the other disappears.
         */
        if (concurrent) return TPHT_OVERFLOW;
        return tpht_flat_write_grow(t, key, value, replace, key_bytes, concurrent);
    }
    return st;
}

static tpht_status_t tpht_flat_update_op(tpht_table_t *t, uint64_t key, uint64_t value,
                                         unsigned key_bytes, int concurrent) {
    /* One critical section: find and overwrite, or report absent.  On a
     * concurrent table TPHT_RETRY means the attempt raced a resize commit
     * and saw a stale storage pointer; retrying re-reads everything. */
    for (;;) {
        tpht_status_t st = tpht_flat_insert_raw(t, key, value, 2, key_bytes, concurrent);
        if (!concurrent || !TPHT_UNLIKELY(st == TPHT_RETRY)) return st;
    }
}

/* --------------------------------------------------------- chained operations
 * The chained variant still branches on threading, because a chained table can
 * be sequential or concurrent; the variant and key width are compile time.
 */
/*
 * Word-native chain operations.  The byte-buffer forms above exist because a
 * chained table stores keys and values at their configured widths, but a caller
 * that already holds both in registers should not have to spill them to a stack
 * buffer and read them back - which also puts a stack-protector canary on the
 * whole insert.  These two do the same work directly on the words.
 */
TPHT_HOT tpht_status_t tpht_chained_insert_word(tpht_table_t *t, uint64_t key, uint64_t value,
                                                int replace) {
    uint64_t key_word = key & t->key_mask;
    size_t base = tpht_base_from_word(t, key_word);
    uint64_t key_quot = key_word >> t->base_bits;
    uint8_t *prev = &t->heads[base];
    uint8_t encoded;
    uint8_t *entry;
    size_t value_off = 1u + t->key_quotient_size;

    if (replace) {
        while (*prev) {
            entry = tpht_pool_deref(t, (uint64_t)(uintptr_t)prev, *prev, t->key_size);
            if (tpht_read_quotient(t, entry + 1u) == key_quot) {
                tpht_store_le(entry + value_off, t->value_size, value);
                return TPHT_OK;
            }
            prev = entry;
        }
    } else {
        /* append only: no existence probe, just walk to the tail */
        while (*prev) prev = tpht_pool_deref(t, (uint64_t)(uintptr_t)prev, *prev, t->key_size);
    }

    entry = tpht_pool_alloc(t, (uint64_t)(uintptr_t)prev, &encoded, t->key_size);
    if (!entry) return TPHT_OVERFLOW;
    *prev = encoded;
    entry[0] = 0;
#if TPHT_LITTLE_ENDIAN
    /* Overlapping whole-word stores, as in the flattened variant. */
    if (t->value_size == 8u && value_off + 8u >= 9u) {
        memcpy(entry + 1u, &key_quot, 8);
        memcpy(entry + value_off, &value, 8);
    } else
#endif
    {
        tpht_store_le(entry + 1u, t->key_quotient_size, key_quot);
        tpht_store_le(entry + value_off, t->value_size, value);
    }
    if (t->tracks_size) tpht_size_inc(t);
    return TPHT_OK;
}

/*
 * Walk a chain whose base the caller has already located.  The concurrent path
 * has to hash the key to choose which base lock to take, and hashing it again
 * here doubled the XXH3 work on every lookup.
 */
/*
 * Bounded form for the optimistic reader: `budget` caps how many links are
 * followed, so a chain observed part-way through an edit cannot spin forever.
 * Running out of budget returns NOT_FOUND, which the caller discards along with
 * everything else when the version check fails.
 */
TPHT_HOT tpht_status_t tpht_chained_get_at_bounded(tpht_table_t *t, uint64_t key_word, size_t base,
                                                   uint64_t *value_out, size_t budget) {
    uint64_t key_quot = key_word >> t->base_bits;
    uint8_t *prev = &t->heads[base];
    size_t value_off = 1u + t->key_quotient_size;
    while (*prev && budget--) {
        uint8_t *entry;
        /* Same guard as the flattened reader: a slot number of zero is not a
         * slot, and would wrap past the end of the bin. */
        if ((*prev & 0x7fu) == 0u) return TPHT_NOT_FOUND;
        entry = tpht_pool_deref(t, (uint64_t)(uintptr_t)prev, *prev, t->key_size);
        if (tpht_read_quotient(t, entry + 1u) == key_quot) {
            /* value_size >= 1 (enforced at create) and value_out is the
             * caller's contract: no guards, just the read. */
            *value_out = tpht_read_le(entry + value_off, t->value_size);
            return TPHT_OK;
        }
        prev = entry;
    }
    return TPHT_NOT_FOUND;
}

TPHT_HOT tpht_status_t tpht_chained_get_at(tpht_table_t *t, uint64_t key_word, size_t base,
                                           uint64_t *value_out) {
    uint64_t key_quot = key_word >> t->base_bits;
    uint8_t *prev = &t->heads[base];
    size_t value_off = 1u + t->key_quotient_size;
    while (*prev) {
        uint8_t *entry = tpht_pool_deref(t, (uint64_t)(uintptr_t)prev, *prev, t->key_size);
        if (tpht_read_quotient(t, entry + 1u) == key_quot) {
            /* value_size >= 1 (enforced at create) and value_out is the
             * caller's contract: no guards, just the read. */
            *value_out = tpht_read_le(entry + value_off, t->value_size);
            return TPHT_OK;
        }
        prev = entry;
    }
    return TPHT_NOT_FOUND;
}

TPHT_HOT tpht_status_t tpht_chained_get_word(tpht_table_t *t, uint64_t key, uint64_t *value_out) {
    uint64_t key_word = key & t->key_mask;
    return tpht_chained_get_at(t, key_word, tpht_base_from_word(t, key_word), value_out);
}

static tpht_status_t tpht_chained_raw_insert(tpht_table_t *t, uint64_t key, uint64_t value,
                                             int replace) {
    return tpht_chained_insert_word(t, key, value, replace);
}

static tpht_status_t tpht_chained_raw_remove(tpht_table_t *t, uint64_t key) {
    uint8_t kb[TPHT_MAX_KEY_BYTES];
    tpht_write_le(kb, t->key_size, key);
    return tpht_chained_remove_raw(t, kb);
}

static tpht_status_t tpht_chained_write_locked(tpht_table_t *t, uint64_t key, uint64_t value,
                                               int replace) {
    tpht_status_t st = tpht_chained_raw_insert(t, key, value, replace);
    /*
     * A hard overflow is absorbed the same way whatever the resize mode: the
     * dereference table cannot hand out an entry, so the table is rebuilt
     * larger and the insert retried.  Only growth *on load* is a resizable
     * table's privilege, and that is decided by write_limit.  A loop, not a
     * single retry: a table overloaded far past its provisioned capacity may
     * need several doublings before the dereference table can take one more
     * entry, and reporting TPHT_OVERFLOW in between would break the contract that
     * fullness alone is never reported.  Each round doubles the capacity, so
     * it terminates - in the worst case at TPHT_NO_MEMORY.
     */
    while (TPHT_UNLIKELY(st == TPHT_OVERFLOW)) {
        st = tpht_resize_locked(t, t->capacity * 2u);
        if (st != TPHT_OK) return st;
        st = tpht_chained_raw_insert(t, key, value, replace);
    }
    return st;
}

static tpht_status_t tpht_chained_write_fine(tpht_table_t *t, uint64_t key_word,
                                             uint64_t value_word, int replace);

static int tpht_chained_resize_active(tpht_table_t *t) {
    return atomic_load_explicit(&t->resize_active, memory_order_acquire) != 0;
}

/* --------------------------------------------- concurrent flatten resize
 * Same growth policy as the chained variant, block for bucket: a shadow table
 * is allocated and blocks are migrated in strides by whichever writers arrive
 * while the resize is active - each pays at most one stride as its help fee.
 * Writers never stall: the old storage stays authoritative (every write lands
 * there first), and a write to an already-migrated block is mirrored into the
 * shadow under the same block lock, so the commit publishes a storage that
 * already contains it.  The commit swaps storage under writer exclusion
 * (every old block seqlock held); lock-free readers never help and never
 * stall - they walk the immutable geometry snapshot and re-check it at the
 * end (see geo_snap).
 */
static int tpht_flat_resize_active(const tpht_table_t *t) {
    return atomic_load_explicit(&((tpht_table_t *)(uintptr_t)t)->resize_active,
                                memory_order_acquire) != 0;
}

/* The descriptor a helper snapshots once and works from until it returns. */
static tpht_resize_op_t *tpht_resize_op_snapshot(tpht_table_t *t) {
    return atomic_load_explicit(&t->resize_op, memory_order_acquire);
}

/* Whether the snapshot still names the in-flight resize (helpers use this only
 * to stop early; correctness never depends on the answer being fresh). */
static int tpht_resize_op_current(tpht_table_t *t, const tpht_resize_op_t *op) {
    return atomic_load_explicit(&t->resize_op, memory_order_acquire) == op;
}

static tpht_resize_op_t *tpht_resize_op_new(tpht_table_t *t, tpht_table_t *nt,
                                            size_t requested_strides) {
    tpht_resize_op_t *op = (tpht_resize_op_t *)calloc(1, sizeof(*op));
    if (!op) return NULL;
    op->migrated = (uint8_t *)calloc(t->base_count, 1);
    if (!op->migrated) {
        free(op);
        return NULL;
    }
    op->target = nt;
    op->old_pack = atomic_load_explicit(&t->flat_lines_pack, memory_order_acquire);
    op->old_geo = atomic_load_explicit(&t->geo_snap, memory_order_acquire);
    op->old_lines = t->flat_lines;
    op->old_chain_locks = t->chain_locks;
    op->old_chain_version_mask = t->chain_version_mask;
    op->old_base_count = t->base_count;
    if (requested_strides == 0) requested_strides = 1;
    if (requested_strides > t->base_count) requested_strides = t->base_count;
    op->stride_size = (t->base_count + requested_strides - 1u) / requested_strides;
    op->stride_count = (t->base_count + op->stride_size - 1u) / op->stride_size;
    atomic_init(&op->next_stride, 0);
    atomic_init(&op->done_strides, 0);
    atomic_init(&op->failed, 0);
    atomic_flag_clear(&op->commit_lock);
    op->next = NULL;
    return op;
}

static int tpht_flat_conc_resize_start(tpht_table_t *t, size_t new_capacity, uint32_t growth,
                                        size_t deref_floor, int block) {
    tpht_table_t *nt;
    tpht_resize_op_t *op;
    if (tpht_flat_resize_active(t)) return 1;

    /*
     * Whoever holds the lock is either allocating the shadow (several
     * milliseconds for a large table) or committing.  A load-factor trigger
     * must not block here: until resize_active flips, the old storage keeps
     * absorbing inserts, so that loser reports success and goes back to
     * inserting.  A hard-overflow caller has no such luxury - its block
     * cannot take the tuple until a resize lands - so it waits its turn
     * instead of spinning through futile retries.
     */
    if (block) {
        tpht_flag_lock(&t->resize_start_lock);
    } else if (!tpht_flag_trylock(&t->resize_start_lock)) {
        return 1;
    }
    if (tpht_flat_resize_active(t)) {
        tpht_flag_unlock(&t->resize_start_lock);
        return 1;
    }
    nt = (tpht_table_t *)calloc(1, sizeof(*nt));
    if (!nt) {
        tpht_flag_unlock(&t->resize_start_lock);
        return 0;
    }
    nt->cfg = t->cfg;
    nt->cfg.resize_mode = TPHT_FIXED; /* the shadow itself must not resize */
    { unsigned si; for (si = 0; si < TPHT_SIZE_SHARDS; ++si) atomic_init(&nt->size_shard[si].v, 0); }
    atomic_init(&nt->geo_snap, nt);
    nt->initial_geo = NULL;
    atomic_init(&nt->resize_active, 0);
    atomic_init(&nt->resize_op, NULL);
    atomic_flag_clear(&nt->resize_start_lock);
    nt->flat_growth = (uint8_t)growth;
    /*
     * The caller's floor alone, not max-ed with the table's: the sequential
     * rebuild resets the dereference floor on every capacity doubling (only
     * same-capacity grows and abort escalations raise it), and carrying the
     * table's floor into every shadow let one transient hard-overflow
     * escalation compound across all later doublings - measured as the pool
     * ratcheting to ~1KB per stored key on a 200M-key growth run.
     */
    nt->flat_deref_floor = deref_floor;
    /*
     * The shadow is populated here, before resize_active flips, not lazily by
     * the migration.  Lazy first touches looked cheaper but measured far
     * worse: an insert's first access to a shadow line is a read of its count
     * byte, which maps the shared zero page, and the write that follows
     * breaks CoW with a TLB shootdown IPI to every core - a resize window's
     * dominant kernel cost.  Populating up front is one batched, huge-page
     * backed sweep (see tpht_prefault_arr), and only this thread waits:
     * writers keep inserting into the old storage until the flip.
     */
    nt->no_prefault = 0u;
    if (!tpht_alloc_storage(nt, new_capacity)) {
        free(nt);
        tpht_flag_unlock(&t->resize_start_lock);
        return 0;
    }
    nt->tracks_size = t->tracks_size;

    op = tpht_resize_op_new(t, nt,
                            t->cfg.resize_strides
                                ? t->cfg.resize_strides
                                : (t->base_count + TPHT_DEFAULT_RESIZE_STRIDE_BUCKETS - 1u) /
                                      TPHT_DEFAULT_RESIZE_STRIDE_BUCKETS);
    if (!op) {
        tpht_free_storage(nt);
        free(nt);
        tpht_flag_unlock(&t->resize_start_lock);
        return 0;
    }
#ifdef TPHT_DEBUG_RESIZE
    {
        struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
        fprintf(stderr, "[rs] start cap=%zu->%zu strides=%zu t=%.6f\n",
                t->capacity, new_capacity, op->stride_count,
                (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec);
    }
#endif
    atomic_store_explicit(&t->resize_op, op, memory_order_release);
    atomic_store_explicit(&t->resize_active, 1, memory_order_release);
    tpht_flag_unlock(&t->resize_start_lock);
    return 1;
}

static void tpht_flat_conc_resize_commit(tpht_table_t *t, tpht_resize_op_t *op);

static void tpht_flat_resize_migrate_block(tpht_table_t *t, tpht_resize_op_t *op, size_t block) {
    /*
     * Only the descriptor's own pointers are touched before the migrated check.
     * A straggler that arrives after the commit finds every block marked, so it
     * releases the (retired, still allocated) old line and leaves; the table's
     * live fields are read only inside the unmigrated branch, where holding the
     * old line's seqlock blocks the commit sweep and keeps them the old ones.
     */
    uint8_t *line = op->old_lines + (block << TPHT_FLAT_LINE_SHIFT);
    tpht_flat_write_begin(line, 1);
    if (!op->migrated[block]) {
        tpht_table_t *nt = op->target;
        uint8_t count = tpht_flat_count(line);
        uint8_t crystals = tpht_flat_crystals(t, line);
        uint8_t crystal_end = tpht_flat_crystal_end(t, crystals);
        uint8_t i;
        for (i = 0; i < count; ++i) {
            uint8_t fp = line[i];
            const uint8_t *payload;
            uint64_t key;
            if (i < crystals) {
                payload = tpht_flat_crystal(t, line, i);
            } else {
                uint8_t encoded = *tpht_flat_tp_slot(line, crystal_end, (uint8_t)(i - crystals));
                payload = tpht_pool_deref(t, tpht_flat_deref_key(block, fp), encoded,
                                          t->key_size);
            }
            key = tpht_flat_rebuild_key(t, block, fp, tpht_flat_read_rem(t, payload),
                                        t->key_size);
            /* Append into the shadow through its own seqlocks - no existence
             * probe, because the migrated[] flag under this block's lock makes
             * each block's keys enter exactly once, and writers cannot add to
             * the shadow while the resize is active.  A FULL here means the
             * grown geometry was still too small - record it and let the
             * commit abort the resize rather than lose the key. */
            if (tpht_flat_insert_raw(nt, key, tpht_flat_read_value(t, payload), 0,
                                     t->key_size, 1) != TPHT_OK)
                atomic_store_explicit(&op->failed, 1u, memory_order_release);
        }
        op->migrated[block] = 1u;
    }
    tpht_flat_write_end(line, 1);
}

static void tpht_flat_resize_migrate_stride(tpht_table_t *t, tpht_resize_op_t *op,
                                            size_t stride) {
    size_t begin = stride * op->stride_size;
    size_t end = begin + op->stride_size;
    size_t block;
    if (end > op->old_base_count) end = op->old_base_count;
    for (block = begin; block < end; ++block)
        tpht_flat_resize_migrate_block(t, op, block);
    atomic_fetch_add_explicit(&op->done_strides, 1u, memory_order_acq_rel);
    tpht_flat_conc_resize_commit(t, op);
}

static void tpht_flat_resize_finish_all(tpht_table_t *t) {
    tpht_resize_op_t *op = tpht_resize_op_snapshot(t);
    size_t stride;
    if (!op) return;
    while ((stride = atomic_fetch_add_explicit(&op->next_stride, 1u, memory_order_acq_rel)) <
           op->stride_count) {
        tpht_flat_resize_migrate_stride(t, op, stride);
    }
    while (tpht_resize_op_current(t, op) &&
           atomic_load_explicit(&op->done_strides, memory_order_acquire) < op->stride_count) {
        tpht_cpu_relax();
    }
    tpht_flat_conc_resize_commit(t, op);
}

/*
 * A writer's help fee: migrate at most one stride, then get back to its own
 * work.  No stride left means the migration tail is in other threads' hands;
 * the writer does not wait for them - the old storage is still authoritative,
 * so it can just proceed.  The thread that takes the last stride commits from
 * inside tpht_flat_resize_migrate_stride.
 */
static void tpht_flat_resize_help_one(tpht_table_t *t) {
    tpht_resize_op_t *op = tpht_resize_op_snapshot(t);
    size_t stride;
    if (!op) return;
    /*
     * Once every stride is claimed, each in-flight insert still lands here
     * until the last migrating thread commits - and a fetch_add per insert on
     * one shared counter is a cache-line all writers serialize through.  The
     * relaxed read costs a shared load instead; it may lag and skip a claim
     * it could have made, which only leaves that stride to another helper.
     */
    if (atomic_load_explicit(&op->next_stride, memory_order_relaxed) >= op->stride_count) {
        if (atomic_load_explicit(&op->done_strides, memory_order_acquire) >= op->stride_count)
            tpht_flat_conc_resize_commit(t, op);
        return;
    }
    stride = atomic_fetch_add_explicit(&op->next_stride, 1u, memory_order_acq_rel);
    if (stride < op->stride_count)
        tpht_flat_resize_migrate_stride(t, op, stride);
    else if (atomic_load_explicit(&op->done_strides, memory_order_acquire) >= op->stride_count)
        tpht_flat_conc_resize_commit(t, op);
}

/*
 * Mirror one write into the shadow.  Called with the key's old block seqlock
 * held and that block marked migrated, which is what makes it safe: the
 * commit and the abort both take every old block lock before touching the
 * shadow, so it cannot be published or freed while this runs, and the
 * migration cannot copy this block again, so nothing lands twice.  A shadow
 * that cannot absorb the write marks the resize failed; the old storage
 * already holds the write, so aborting loses nothing.
 */
TPHT_NOINLINE static void tpht_flat_shadow_write(tpht_resize_op_t *op, uint64_t key,
                                                 uint64_t value, int replace,
                                                 unsigned key_bytes) {
    if (tpht_flat_insert_raw(op->target, key, value, replace, key_bytes, 1) != TPHT_OK)
        atomic_store_explicit(&op->failed, 1u, memory_order_release);
}

TPHT_NOINLINE static void tpht_flat_shadow_remove(tpht_resize_op_t *op, uint64_t key,
                                                  unsigned key_bytes) {
    /* The mirror invariant says the key is there; a miss means the shadow has
     * diverged, and the only safe answer is to abandon it. */
    if (tpht_flat_remove_raw(op->target, key, key_bytes, 1) != TPHT_OK)
        atomic_store_explicit(&op->failed, 1u, memory_order_release);
}

static void tpht_flat_conc_resize_commit(tpht_table_t *t, tpht_resize_op_t *op) {
    tpht_table_t *nt;
    uint8_t *old_lines;
    void *old_lines_raw;
    uint8_t *old_heads;
    uint8_t *old_pool_entries;
    uint8_t *old_pool_cnt_head;
    atomic_uchar *old_chain_locks;
    size_t old_blocks;
    size_t li;
    tpht_retired_storage_t *retired;

    if (atomic_load_explicit(&op->done_strides, memory_order_acquire) < op->stride_count)
        return;
    /* One committer per resize, forever: the flag is never cleared, so a
     * straggler that reaches here after the swap just bounces off. */
    if (atomic_flag_test_and_set_explicit(&op->commit_lock, memory_order_acquire)) return;
    tpht_flag_lock(&t->resize_start_lock);
    nt = op->target;

    /*
     * A migration insert that failed set the descriptor's marker: the shadow
     * could not hold everything, so the resize is abandoned rather than
     * committed short of keys.  The old storage is untouched and still
     * authoritative; the next overflow starts a bigger attempt.
     */
    if (atomic_load_explicit(&op->failed, memory_order_acquire)) {
        /*
         * Writer exclusion before the shadow is freed: a writer holding an old
         * block lock may be mid-mirror into the shadow (tpht_flat_shadow_write),
         * so every old block lock is taken - and with it the guarantee that no
         * such mirror is in flight - before the shadow's storage goes away.
         */
        old_blocks = op->old_base_count;
        old_lines = op->old_lines;
        for (li = 0; li < old_blocks; ++li)
            tpht_flat_write_begin(old_lines + (li << TPHT_FLAT_LINE_SHIFT), 1);
        atomic_store_explicit(&t->resize_op, NULL, memory_order_release);
        op->target = NULL;
        op->next = t->retired_ops;
        t->retired_ops = op;
        /* The failed geometry must not be retried verbatim: raise the block
         * growth so the next attempt allocates more blocks per capacity, and
         * raise the dereference floor past the pool that just proved too
         * small - the migration's FULL was almost certainly there. */
        if (nt->flat_growth >= t->flat_growth)
            t->flat_growth = (uint8_t)(nt->flat_growth + 1u);
        t->flat_deref_floor = tpht_max_size(
            t->flat_deref_floor, nt->pool.bin_count * (size_t)TPHT_BIN_SIZE * 2u);
#ifdef TPHT_DEBUG_RESIZE
        fprintf(stderr, "[rs] ABORT growth->%u\n", (unsigned)t->flat_growth);
#endif
        tpht_free_storage(nt);
        free(nt);
        for (li = 0; li < old_blocks; ++li)
            tpht_flat_write_end(old_lines + (li << TPHT_FLAT_LINE_SHIFT), 1);
        atomic_store_explicit(&t->resize_active, 0, memory_order_release);
        tpht_flag_unlock(&t->resize_start_lock);
        return;
    }

    /*
     * Writer exclusion, as in the chained commit: every write to a block holds
     * its seqlock, so holding all of them means no writer is inside the old
     * storage.  Readers take no locks; they revalidate the snapshot instead.
     */
    old_blocks = op->old_base_count;
    old_lines = op->old_lines;
    for (li = 0; li < old_blocks; ++li)
        tpht_flat_write_begin(old_lines + (li << TPHT_FLAT_LINE_SHIFT), 1);

    /*
     * The failed flag must be read again now that every writer is excluded: a
     * mirror into the shadow can fail after the first check above, and only
     * with all block locks held is the flag's value final.  Publishing a
     * shadow that a mirror could not write to would lose that writer's key.
     */
    if (TPHT_UNLIKELY(atomic_load_explicit(&op->failed, memory_order_acquire) != 0u)) {
        atomic_store_explicit(&t->resize_op, NULL, memory_order_release);
        op->target = NULL;
        op->next = t->retired_ops;
        t->retired_ops = op;
        if (nt->flat_growth >= t->flat_growth)
            t->flat_growth = (uint8_t)(nt->flat_growth + 1u);
        tpht_free_storage(nt);
        free(nt);
        for (li = 0; li < old_blocks; ++li)
            tpht_flat_write_end(old_lines + (li << TPHT_FLAT_LINE_SHIFT), 1);
        atomic_store_explicit(&t->resize_active, 0, memory_order_release);
        tpht_flag_unlock(&t->resize_start_lock);
        return;
    }

    old_lines_raw = t->flat_lines_raw;
    old_heads = t->heads;
    old_pool_entries = t->pool.entries;
    old_pool_cnt_head = t->pool.cnt_head;
    old_chain_locks = t->chain_locks;
    retired = (tpht_retired_storage_t *)calloc(1, sizeof(*retired));

    t->capacity = nt->capacity;
    tpht_refresh_write_limit(t);
    /* The shadow was built FIXED so it would not resize while being filled,
     * which left its write_limit unreachable; as the published snapshot it
     * must carry the live table's trigger value instead. */
    nt->write_limit = t->write_limit;
    tpht_size_store(t, tpht_size_load(nt));
    t->key_quotient_size = nt->key_quotient_size;
    t->quotient_mask = nt->quotient_mask;
    t->inline_entry_size = nt->inline_entry_size;
    t->pool_entry_size = nt->pool_entry_size;
    t->base_bits = nt->base_bits;
    t->base_mask = nt->base_mask;
    t->base_count = nt->base_count;
    t->heads = nt->heads;
    t->flat_lines = nt->flat_lines;
    t->flat_lines_raw = nt->flat_lines_raw;
    t->flat_entry_size = nt->flat_entry_size;
    t->flat_cost = nt->flat_cost;
    t->flat_inline_ok = nt->flat_inline_ok;
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
    t->pool = nt->pool;
    t->chain_locks = nt->chain_locks;
    t->chain_lock_count = nt->chain_lock_count;
    t->chain_version_mask = nt->chain_version_mask;
    /* Under every old line lock, like the rest of the swap: a writer that
     * still holds an old line has not validated yet, and one that validates
     * later re-reads this word. */
    atomic_store_explicit(&t->flat_lines_pack, tpht_flat_pack_of(t), memory_order_release);

    /*
     * The shadow descriptor keeps its pointers: it is the next published
     * geometry snapshot, and readers will take the new storage through it.
     * Nothing is ever freed through a descriptor - the retired-storage node
     * below owns the arrays - so no pointer here can double-free.
     */

    /* Retire the descriptor before reopening the table: any straggler still
     * holding it finds every block migrated and does nothing. */
    atomic_store_explicit(&t->resize_op, NULL, memory_order_release);
    op->target = NULL;
    op->next = t->retired_ops;
    t->retired_ops = op;

    if (retired) {
        retired->heads = old_heads;
        retired->flat_lines_raw = old_lines_raw;
        retired->pool_entries = old_pool_entries;
        retired->pool_cnt_head = old_pool_cnt_head;
        retired->chain_locks = old_chain_locks;
        retired->resize_descriptor = nt;
        retired->next = t->retired;
        t->retired = retired;
    }

    /*
     * The snapshot must move while every old block lock is still held: the
     * lock release below is what a later writer's acquire synchronizes with,
     * so only stores sequenced before it are guaranteed visible to that
     * writer.  Published after the release, a writer could take a freshly
     * released old line, still see the old snapshot, pass its check and write
     * a key into retired storage - counted by the live table, findable
     * nowhere.  The shadow descriptor becomes the snapshot: its geometry is
     * exactly the storage just installed, and it is retired, never freed, so
     * a straggling reader can hold it for as long as it likes.
     */
    atomic_store_explicit(&t->geo_snap, nt, memory_order_release);

    /* Release every old block so spinning readers move on and revalidate. */
    for (li = 0; li < old_blocks; ++li)
        tpht_flat_write_end(old_lines + (li << TPHT_FLAT_LINE_SHIFT), 1);

    atomic_store_explicit(&t->resize_active, 0, memory_order_release);
    tpht_flag_unlock(&t->resize_start_lock);
#ifdef TPHT_DEBUG_RESIZE
    {
        struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
        fprintf(stderr, "[rs] commit cap=%zu t=%.6f\n", t->capacity,
                (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec);
    }
#endif
}

static int tpht_chained_resize_start(tpht_table_t *t, size_t new_capacity, int block) {
    tpht_table_t *nt;
    tpht_resize_op_t *op;
    size_t requested_strides;
    if (tpht_chained_resize_active(t)) return 1;

    /* As in the flattened start: load-factor losers keep inserting instead of
     * queueing behind a multi-millisecond shadow allocation; a hard-overflow
     * caller cannot proceed anyway and waits its turn. */
    if (block) {
        tpht_flag_lock(&t->resize_start_lock);
    } else if (!tpht_flag_trylock(&t->resize_start_lock)) {
        return 1;
    }
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
    { unsigned si; for (si = 0; si < TPHT_SIZE_SHARDS; ++si) atomic_init(&nt->size_shard[si].v, 0); }
    atomic_init(&nt->geo_snap, nt);
    nt->initial_geo = NULL;
    atomic_init(&nt->resize_active, 0);
    atomic_init(&nt->resize_op, NULL);
    atomic_flag_clear(&nt->lock);
    atomic_flag_clear(&nt->resize_start_lock);
    /* Populated up front for the same reason as the flattened start: lazy
     * first touches turn into CoW TLB-shootdown IPIs mid-migration. */
    nt->no_prefault = 0u;
    if (!tpht_alloc_storage(nt, new_capacity)) {
        free(nt);
        tpht_flag_unlock(&t->resize_start_lock);
        return 0;
    }
    /*
     * The shadow is marked fixed only so it does not try to resize while it is
     * being filled; its write_limit stays unreachable.  It must still count
     * what it receives, because the commit hands that count back to the live
     * table - "does not grow" and "keeps no size" are separate properties.
     */
    nt->tracks_size = t->tracks_size;

    requested_strides = t->cfg.resize_strides
                            ? t->cfg.resize_strides
                            : ((t->base_count + TPHT_DEFAULT_RESIZE_STRIDE_BUCKETS - 1u) /
                               TPHT_DEFAULT_RESIZE_STRIDE_BUCKETS);
    op = tpht_resize_op_new(t, nt, requested_strides);
    if (!op) {
        tpht_free_storage(nt);
        free(nt);
        tpht_flag_unlock(&t->resize_start_lock);
        return 0;
    }
    atomic_store_explicit(&t->resize_op, op, memory_order_release);
    atomic_store_explicit(&t->resize_active, 1, memory_order_release);
    tpht_flag_unlock(&t->resize_start_lock);
    return 1;
}

static void tpht_chained_resize_commit(tpht_table_t *t, tpht_resize_op_t *op) {
    tpht_table_t *nt;
    uint8_t *old_heads;
    void *old_flat_lines_raw;
    uint8_t *old_pool_entries;
    uint8_t *old_pool_cnt_head;
    atomic_uchar *old_chain_locks;
    size_t old_chain_lock_count;
    tpht_retired_storage_t *retired;

    if (atomic_load_explicit(&op->done_strides, memory_order_acquire) < op->stride_count) return;
    /* One committer per resize, forever: a straggler bounces off the flag. */
    if (atomic_flag_test_and_set_explicit(&op->commit_lock, memory_order_acquire)) return;
    tpht_flag_lock(&t->resize_start_lock);
    nt = op->target;

    /*
     * Quiescence, without any per-operation bookkeeping.  Every operation on a
     * chained table holds the lock for the base it touches, so holding all of
     * them means no operation is inside the storage this commit is about to
     * replace.  An operation that computed its base from the old geometry and
     * is waiting for one of these locks re-reads resize_active once it gets in
     * and starts over.  The cost is paid once per resize instead of by every
     * operation for the table's whole life.
     */
    {
        size_t li;
        for (li = 0; li < t->chain_lock_count; ++li) tpht_chain_lock_base_fine(t, li);
    }

    /*
     * With every old chain lock held no mirror can be mid-flight, so the
     * failed flag is final.  A failed shadow (a migration or mirror insert
     * its dereference bins could not absorb) is abandoned rather than
     * committed short of keys: the old storage is complete and stays
     * authoritative, and the next trigger starts a fresh attempt whose pool
     * layout differs (bin placement keys off entry addresses).
     */
    if (TPHT_UNLIKELY(atomic_load_explicit(&op->failed, memory_order_acquire) != 0u)) {
        size_t li;
        atomic_store_explicit(&t->resize_op, NULL, memory_order_release);
        op->target = NULL;
        op->next = t->retired_ops;
        t->retired_ops = op;
        tpht_free_storage(nt);
        free(nt);
        for (li = 0; li < t->chain_lock_count; ++li)
            atomic_store_explicit(&t->chain_locks[TPHT_CHAIN_SLOT(li)],
                                  (unsigned char)(atomic_load_explicit(
                                                      &t->chain_locks[TPHT_CHAIN_SLOT(li)],
                                                      memory_order_relaxed) + 1u),
                                  memory_order_release);
        atomic_store_explicit(&t->resize_active, 0, memory_order_release);
        tpht_flag_unlock(&t->resize_start_lock);
        return;
    }

    old_heads = t->heads;
    old_flat_lines_raw = t->flat_lines_raw;
    old_pool_entries = t->pool.entries;
    old_pool_cnt_head = t->pool.cnt_head;
    old_chain_locks = t->chain_locks;
    old_chain_lock_count = t->chain_lock_count;
    retired = (tpht_retired_storage_t *)calloc(1, sizeof(*retired));

    t->capacity = nt->capacity;
    tpht_refresh_write_limit(t);
    tpht_size_store(t, tpht_size_load(nt));
    /* key_size and value_size are invariants of the table; re-storing them
     * here raced the marshalling reads in the write entry points for no
     * benefit, so they are simply not written. */
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
    t->chain_lock_count = nt->chain_lock_count;
    t->chain_version_mask = nt->chain_version_mask;

    /* Retire the descriptor before reopening the table: a straggler that still
     * holds it finds every bucket migrated and does nothing.  The shadow's
     * small descriptor struct is kept alive the same way, below. */
    atomic_store_explicit(&t->resize_op, NULL, memory_order_release);
    op->target = NULL;
    op->next = t->retired_ops;
    t->retired_ops = op;

    if (retired) {
        retired->heads = old_heads;
        retired->flat_lines_raw = old_flat_lines_raw;
        retired->pool_entries = old_pool_entries;
        retired->pool_cnt_head = old_pool_cnt_head;
        retired->chain_locks = old_chain_locks;
        retired->resize_descriptor = nt;
        retired->next = t->retired;
        t->retired = retired;
    }

    /* Before the lock release, for the same reason as in the flattened
     * commit: the release is the synchronization edge that publishes it.  The
     * shadow descriptor keeps its pointers and becomes the geometry snapshot,
     * exactly as in the flattened commit. */
    atomic_store_explicit(&t->geo_snap, nt, memory_order_release);

    {
        size_t li;
        for (li = 0; li < old_chain_lock_count; ++li)
            atomic_store_explicit(&old_chain_locks[TPHT_CHAIN_SLOT(li)],
                                  (unsigned char)(atomic_load_explicit(
                                                      &old_chain_locks[TPHT_CHAIN_SLOT(li)],
                                                      memory_order_relaxed) + 1u),
                                  memory_order_release);
    }

    atomic_store_explicit(&t->resize_active, 0, memory_order_release);
    tpht_flag_unlock(&t->resize_start_lock);
}

/* Lock a bucket's seqlock through the descriptor's own array, so a straggler
 * locks the retired old lock instead of an unrelated live one. */
static void tpht_resize_op_lock_base(tpht_resize_op_t *op, size_t base) {
    atomic_uchar *v = &op->old_chain_locks[TPHT_CHAIN_SLOT(base & op->old_chain_version_mask)];
    for (;;) {
        if (!tpht_bit_test_and_set(v)) return;
        do {
            tpht_cpu_relax();
        } while (atomic_load_explicit(v, memory_order_relaxed) & 1u);
    }
}

static void tpht_resize_op_unlock_base(tpht_resize_op_t *op, size_t base) {
    atomic_uchar *v = &op->old_chain_locks[TPHT_CHAIN_SLOT(base & op->old_chain_version_mask)];
    atomic_store_explicit(v, (unsigned char)(atomic_load_explicit(v, memory_order_relaxed) + 1u),
                          memory_order_release);
}

static void tpht_chained_resize_migrate_bucket(tpht_table_t *t, tpht_resize_op_t *op,
                                               size_t base) {
    /* As in the flatten variant: only descriptor state is touched before the
     * migrated check.  Holding the old bucket's seqlock blocks the commit's
     * all-locks sweep, so inside the unmigrated branch the table's live fields
     * are still the old ones; after a commit every bucket is marked and this
     * degrades to a lock/unlock of a retired seqlock. */
    tpht_resize_op_lock_base(op, base);
    if (!op->migrated[base]) {
        tpht_table_t *nt = op->target;
        uint8_t *prev = &t->heads[base];
        while (*prev) {
            uint8_t rebuilt_key[8];
            uint8_t *entry = tpht_pool_deref(t, (uint64_t)(uintptr_t)prev, *prev, t->key_size);
            tpht_rebuild_key(t, base, entry + 1u, rebuilt_key);
            /* Append, no existence probe: migrated[] under this bucket's lock
             * makes each bucket's keys enter the shadow exactly once.  A
             * failure here (an improbable but possible dereference-bin
             * overflow in the shadow) marks the resize failed; committing
             * anyway would silently drop this key. */
            if (tpht_chained_write_fine(nt, tpht_key_word(nt, rebuilt_key),
                                        tpht_read_le(entry + 1u + t->key_quotient_size,
                                                     t->value_size),
                                        0) != TPHT_OK)
                atomic_store_explicit(&op->failed, 1u, memory_order_release);
            prev = entry;
        }
        op->migrated[base] = 1u;
    }
    tpht_resize_op_unlock_base(op, base);
}

static void tpht_chained_resize_migrate_stride(tpht_table_t *t, tpht_resize_op_t *op,
                                               size_t stride) {
    size_t begin = stride * op->stride_size;
    size_t end = begin + op->stride_size;
    size_t base;
    if (end > op->old_base_count) end = op->old_base_count;
    /* A claimed stride is always finished: done_strides must never count a
     * stride whose buckets were not all migrated. */
    for (base = begin; base < end; ++base) {
        tpht_chained_resize_migrate_bucket(t, op, base);
    }
    atomic_fetch_add_explicit(&op->done_strides, 1u, memory_order_acq_rel);
    tpht_chained_resize_commit(t, op);
}

static void tpht_chained_resize_finish_all(tpht_table_t *t) {
    tpht_resize_op_t *op = tpht_resize_op_snapshot(t);
    size_t stride;
    if (!op) return;
    while ((stride = atomic_fetch_add_explicit(&op->next_stride, 1u, memory_order_acq_rel)) <
           op->stride_count) {
        tpht_chained_resize_migrate_stride(t, op, stride);
    }
    while (tpht_resize_op_current(t, op) &&
           atomic_load_explicit(&op->done_strides, memory_order_acquire) < op->stride_count) {
        tpht_cpu_relax();
    }
    tpht_chained_resize_commit(t, op);
}

static int tpht_chained_resize_needed(tpht_table_t *t) {
    /* Capacity through the snapshot: this runs before any lock is held, and a
     * commit may be swapping the table's own field mid-read.  A stale value
     * only delays the trigger by one check period. */
    tpht_table_t *g = atomic_load_explicit(&t->geo_snap, memory_order_acquire);
    return (double)(tpht_size_load(t) + 1u) > (double)g->capacity * t->cfg.max_load_factor;
}

static tpht_status_t tpht_chained_ensure_resize(tpht_table_t *t, int block) {
    /* Capacity through the snapshot: no lock is held here.  Stale at worst
     * requests a same-size resize, which the start path shrugs off. */
    tpht_table_t *g = atomic_load_explicit(&t->geo_snap, memory_order_acquire);
    if (!tpht_chained_resize_start(t, g->capacity * 2u, block)) return TPHT_NO_MEMORY;
    return TPHT_OK;
}

static tpht_status_t tpht_chained_resizable_write_fine(tpht_table_t *t, uint64_t key_word,
                                                       uint64_t value_word, int replace) {
    tpht_status_t st;
    for (;;) {
        /* tpht_chained_write_fine waits out an active resize itself (helping
         * finish it), so this loop only owns the growth trigger and the
         * hard-overflow arm. */
        if (TPHT_UNLIKELY(++tpht_tls_limit_tick >=
                          atomic_load_explicit(&t->size_check_period, memory_order_relaxed))) {
            tpht_tls_limit_tick = 0;
            if (!tpht_chained_resize_active(t) && tpht_chained_resize_needed(t)) {
                st = tpht_chained_ensure_resize(t, 0);
                if (st != TPHT_OK) return st;
                continue;
            }
        }
        st = tpht_chained_write_fine(t, key_word, value_word, replace);
        if (TPHT_UNLIKELY(st == TPHT_OVERFLOW)) {
            if (tpht_chained_resize_active(t)) {
                tpht_chained_resize_finish_all(t);
                continue;
            }
            st = tpht_chained_ensure_resize(t, 1);
            if (st != TPHT_OK) return st;
            continue;
        }
        return st;
    }
}


/*
 * Word-native, like the sequential path: routing a key through a stack buffer
 * and reading it back costs a copy each way and puts a stack-protector canary
 * on the whole lookup.
 */
static tpht_status_t tpht_chained_get_fine_word(tpht_table_t *t, uint64_t key,
                                                uint64_t *value_out) {
    size_t base;
    uint64_t key_word;
    tpht_table_t *g;
    tpht_status_t st;
retry:
    /*
     * The whole walk runs off one immutable geometry snapshot (see geo_snap),
     * for the same reason as the flattened reader: a commit is a multi-word
     * swap, and a base composed from one storage's mask with another's arrays
     * would index outside both.  A resize never stalls this reader: writers
     * stall while a chained resize is active, so the old storage is frozen
     * and complete for the whole migration, and a walk over it is a valid
     * answer as of the moment the resize began.
     */
    g = atomic_load_explicit(&t->geo_snap, memory_order_acquire);
    key_word = key & g->key_mask;
    base = tpht_base_from_word(g, key_word);
    {
        /*
         * Optimistic read: take the chain's version, walk it without writing
         * anything, and keep the answer only if the version is unchanged.  A
         * writer that edited the chain in the meantime forces another attempt.
         * The walk is bounded because a chain seen mid-edit can appear to loop;
         * every tiny pointer still addresses a slot inside the pool, so a stale
         * one is harmless to follow, just not to believe.
         */
        unsigned char snapshot = tpht_chain_read_begin(g, base);
        /* A chain cannot hold more entries than the pool has slots; bounding
         * by bin_count alone truncated real chains on small tables. */
        st = tpht_chained_get_at_bounded(g, key_word, base, value_out,
                                         g->pool.bin_count * (size_t)TPHT_BIN_SIZE);
        if (!tpht_chain_read_valid(g, base, snapshot)) goto retry;
        /* The storage itself must also be the one this walk started on. */
        if (atomic_load_explicit(&t->geo_snap, memory_order_acquire) != g) goto retry;
    }
    return st;
}

/*
 * Word-native, like the read path: the key and value cross as words, sparing
 * the stack-buffer marshalling (a copy each way plus a stack-protector canary
 * on every write) that the byte form paid.
 */
static tpht_status_t tpht_chained_write_fine(tpht_table_t *t, uint64_t key_word,
                                             uint64_t value_word, int replace) {
    size_t base;
    uint64_t key_quot;
    uint8_t *prev;
    uint8_t encoded;
    uint8_t *entry;
    atomic_uchar *lk;
    tpht_table_t *g;

retry:
    /*
     * Chained writers wait out an active resize by helping finish it - the
     * write-through mirror the flattened variant uses was measured slower
     * here: a chained operation is a chain walk plus a pool allocation, and
     * doubling that per write costs more than the bulk many-hands migration
     * saves.  The failed/abort machinery still guards the migration itself.
     */
    if (TPHT_UNLIKELY(tpht_chained_resize_active(t))) {
        tpht_chained_resize_finish_all(t);
        goto retry;
    }
    /*
     * The base, the heads array and the lock array must all come from the same
     * storage, and no field of `t` can promise that mid-commit; the immutable
     * snapshot can (see geo_snap).  The re-read under the lock then says in
     * one pointer comparison whether a commit interleaved; if the snapshot is
     * still current, the lock just taken is one the next commit must sweep,
     * so the table's own fields are stable for the rest of the operation.
     */
    g = atomic_load_explicit(&t->geo_snap, memory_order_acquire);
    base = tpht_base_from_word(g, key_word);
    prev = &g->heads[base];
    lk = tpht_chain_lock_take(g, base);
    if (TPHT_UNLIKELY(atomic_load_explicit(&t->geo_snap, memory_order_acquire) != g) ||
        tpht_chained_resize_active(t)) {
        tpht_chain_lock_release(lk);
        goto retry;
    }
    key_quot = key_word >> t->base_bits;
    while (*prev) {
        entry = tpht_pool_deref(t, (uint64_t)(uintptr_t)prev, *prev, t->key_size);
        if (replace && tpht_read_quotient(t, entry + 1u) == key_quot) {
            tpht_store_le(entry + 1u + t->key_quotient_size, t->value_size, value_word);
            tpht_chain_lock_release(lk);
            return TPHT_OK;
        }
        prev = entry;
    }

    /*
     * A resizable table counts the entry before allocating it; the hard bound
     * is the pool itself, whose failed allocation below reports TPHT_OVERFLOW and
     * gives the count back.  The old compare-and-swap reservation against
     * capacity added a contended read-modify-write per insert to enforce a
     * bound the storage already enforces.
     */
    if (t->tracks_size) tpht_size_inc(t);

    entry = tpht_pool_alloc(t, (uint64_t)(uintptr_t)prev, &encoded, t->key_size);
    if (!entry) {
        if (t->tracks_size) tpht_size_dec(t); /* give the reservation back. */
        tpht_chain_lock_release(lk);
        return TPHT_OVERFLOW;
    }

    *prev = encoded;
    entry[0] = 0;
    tpht_store_le(entry + 1u, t->key_quotient_size, key_quot);
    tpht_store_le(entry + 1u + t->key_quotient_size, t->value_size, value_word);
    tpht_chain_lock_release(lk);
    return TPHT_OK;
}

static tpht_status_t tpht_chained_update_fine(tpht_table_t *t, uint64_t key_word,
                                              uint64_t value_word) {
    size_t base;
    uint8_t *prev;
    atomic_uchar *lk;
    tpht_table_t *g;
retry:
    if (TPHT_UNLIKELY(tpht_chained_resize_active(t))) {
        tpht_chained_resize_finish_all(t);
        goto retry;
    }
    /* Same snapshot discipline as tpht_chained_write_fine. */
    g = atomic_load_explicit(&t->geo_snap, memory_order_acquire);
    base = tpht_base_from_word(g, key_word);
    prev = &g->heads[base];
    lk = tpht_chain_lock_take(g, base);
    if (TPHT_UNLIKELY(atomic_load_explicit(&t->geo_snap, memory_order_acquire) != g) ||
        tpht_chained_resize_active(t)) {
        tpht_chain_lock_release(lk);
        goto retry;
    }
    {
        uint64_t key_quot = key_word >> t->base_bits;
        while (*prev) {
            uint8_t *entry = tpht_pool_deref(t, (uint64_t)(uintptr_t)prev, *prev, t->key_size);
            if (tpht_read_quotient(t, entry + 1u) == key_quot) {
                tpht_store_le(entry + 1u + t->key_quotient_size, t->value_size, value_word);
                tpht_chain_lock_release(lk);
                return TPHT_OK;
            }
            prev = entry;
        }
    }
    tpht_chain_lock_release(lk);
    return TPHT_NOT_FOUND;
}

static tpht_status_t tpht_chained_remove_fine(tpht_table_t *t, uint64_t key_word) {
    size_t base;
    uint8_t *prev;
    uint8_t *target;
    uint8_t *last_prev;
    uint8_t *last_entry;
    uint8_t last_encoded;
    atomic_uchar *lk;
    tpht_table_t *g;

retry:
    if (TPHT_UNLIKELY(tpht_chained_resize_active(t))) {
        tpht_chained_resize_finish_all(t);
        goto retry;
    }
    /* Same snapshot discipline as tpht_chained_write_fine. */
    g = atomic_load_explicit(&t->geo_snap, memory_order_acquire);
    base = tpht_base_from_word(g, key_word);
    prev = &g->heads[base];
    target = NULL;
    last_prev = NULL;
    last_entry = NULL;
    last_encoded = 0;
    lk = tpht_chain_lock_take(g, base);
    if (TPHT_UNLIKELY(atomic_load_explicit(&t->geo_snap, memory_order_acquire) != g) ||
        tpht_chained_resize_active(t)) {
        tpht_chain_lock_release(lk);
        goto retry;
    }
    {
        uint64_t key_quot = key_word >> t->base_bits;
        while (*prev) {
            uint8_t encoded = *prev;
            uint8_t *entry = tpht_pool_deref(t, (uint64_t)(uintptr_t)prev, encoded, t->key_size);
            if (tpht_read_quotient(t, entry + 1u) == key_quot) target = entry;
            last_prev = prev;
            last_entry = entry;
            last_encoded = encoded;
            prev = entry;
        }
    }
    if (!target) {
        tpht_chain_lock_release(lk);
        return TPHT_NOT_FOUND;
    }

    if (target != last_entry) {
        uint8_t target_next = target[0];
        memcpy(target, last_entry, t->pool.entry_size);
        target[0] = target_next;
    }
    *last_prev = 0;
    tpht_pool_free(t, last_encoded, last_entry);
    if (t->tracks_size) tpht_size_dec(t);
    tpht_chain_lock_release(lk);
    return TPHT_OK;
}

static tpht_status_t tpht_resize_locked(tpht_table_t *t, size_t new_capacity) {
    tpht_table_t nt;
    size_t i;
    memset(&nt, 0, sizeof(nt));
    nt.cfg = t->cfg;
    { unsigned si; for (si = 0; si < TPHT_SIZE_SHARDS; ++si) atomic_init(&nt.size_shard[si].v, 0); }
    atomic_init(&nt.geo_snap, &nt);
    nt.initial_geo = NULL;
    atomic_init(&nt.resize_active, 0);
    atomic_init(&nt.resize_op, NULL);
    atomic_flag_clear(&nt.lock);
    atomic_flag_clear(&nt.resize_start_lock);
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
                return st == TPHT_OVERFLOW ? TPHT_NO_MEMORY : st;
            }
            prev = entry;
        }
    }

    tpht_free_storage(t);
    t->capacity = nt.capacity;
    tpht_set_write_limit(t);
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
    t->chain_lock_count = nt.chain_lock_count;
    t->chain_version_mask = nt.chain_version_mask;
    nt.heads = NULL;
    nt.flat_lines = NULL;
    nt.flat_lines_raw = NULL;
    nt.pool.entries = NULL;
    nt.pool.cnt_head = NULL;
    nt.chain_locks = NULL;
    nt.chain_lock_count = 0;
    return TPHT_OK;
}


/*
 * A fixed table keeps no running size, so its entry count is summed on demand:
 * over the home blocks for the flattened variant, over the dereference bins for
 * the chained one, where every live entry occupies exactly one bin slot.  The
 * walk is linear in the table, which is why only this accessor pays for it -
 * writes stay free of the counter.
 */
static size_t tpht_size_count(const tpht_table_t *t) {
    size_t total = 0;
    size_t i;
    if (t->cfg.variant == TPHT_FLATTEN) {
        for (i = 0; i < t->base_count; ++i)
            total += tpht_flat_count(t->flat_lines + (i << TPHT_FLAT_LINE_SHIFT));
        return total;
    }
    for (i = 0; i < t->pool.bin_count; ++i) total += tpht_pool_count(&t->pool, i);
    return total;
}

static size_t tpht_size_of(const tpht_table_t *t) {
    if (!t) return 0;
    return t->tracks_size ? tpht_size_load(t) : tpht_size_count(t);
}
static size_t tpht_capacity_of(const tpht_table_t *t) { return t ? t->capacity : 0; }

static size_t tpht_memory_of(const tpht_table_t *t) {
    size_t bytes;
    if (!t) return 0;
    bytes = sizeof(*t);
    bytes += t->pool.bin_count * (size_t)TPHT_BIN_SIZE * t->pool.entry_size;
    bytes += t->pool.bin_count << t->pool.meta_shift; /* pool bin metadata */
    bytes += TPHT_CHAIN_SLOT(t->chain_lock_count) * sizeof(*t->chain_locks);
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
    o.max_load_factor = TPHT_DEFAULT_LOAD_FACTOR;
    o.hash_seed = 0u; /* zero means: give each table its own random seed */
    o.resize_strides = 0;
    return o;
}

/*
 * A concurrent flattened table cannot grow: readers walk its blocks without a
 * lock, so replacing the storage underneath them is not something the seqlock
 * can cover.  Such a table is refused rather than handed back to fail later.
 */
static int tpht_flat_concurrent_ok(tpht_variant_t variant, tpht_threading_t threading,
                                   tpht_resize_mode_t mode) {
    (void)variant; (void)threading; (void)mode;
    return 1; /* concurrent flattened tables resize with the chained policy now */
}

static tpht_table_t *tpht_create_internal(tpht_variant_t variant, tpht_threading_t threading,
                                          uint8_t key_size, size_t capacity,
                                          const tpht_options_t *options) {
    tpht_options_t o = options ? *options : tpht_default_options();
    tpht_config_t c;
    tpht_table_t *t;

    if (o.value_size == 0u) o.value_size = 8u;
    if (o.value_size > TPHT_MAX_VALUE_BYTES) return NULL;
    if (o.max_load_factor <= 0.0 || o.max_load_factor > 1.0) {
        o.max_load_factor = TPHT_DEFAULT_LOAD_FACTOR;
    }
    /*
     * A caller that supplies a seed gets exactly that seed, so a table can be
     * made reproducible on purpose.  Everyone else gets a fresh one per table:
     * a fixed default would give every table in every process the same hash
     * function, so one set of colliding keys would degrade all of them.
     */
    if (o.hash_seed == 0u) o.hash_seed = tpht_random_seed();
    if (capacity == 0u) capacity = TPHT_MIN_CAPACITY;
    if (!tpht_flat_concurrent_ok(variant, threading, o.resize_mode)) return NULL;

    c.variant = variant;
    c.threading = threading;
    c.resize_mode = o.resize_mode;
    c.initial_capacity = capacity;
    c.key_size = key_size;
    c.value_size = o.value_size;
    c.max_load_factor = o.max_load_factor;
    c.hash_seed = o.hash_seed;
    c.resize_strides = o.resize_strides;

    t = (tpht_table_t *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->cfg = c;
    { unsigned si; for (si = 0; si < TPHT_SIZE_SHARDS; ++si) atomic_init(&t->size_shard[si].v, 0); }
    atomic_init(&t->geo_snap, t);
    atomic_init(&t->resize_active, 0);
    atomic_init(&t->resize_op, NULL);
    atomic_flag_clear(&t->lock);
    atomic_flag_clear(&t->resize_start_lock);
    if (!tpht_alloc_storage(t, capacity)) {
        free(t);
        return NULL;
    }
    /*
     * A concurrent flattened table's first commit will overwrite this struct's
     * geometry in place, so the published snapshot cannot be the table itself:
     * a reader still holding it would watch the fields move.  Publish an
     * immutable copy instead; later snapshots are the resize shadow
     * descriptors, which are retired rather than freed for the same reason.
     */
    if (c.threading == TPHT_CONCURRENT) {
        tpht_table_t *g0 = (tpht_table_t *)malloc(sizeof(*g0));
        if (!g0) {
            tpht_free_storage(t);
            free(t);
            return NULL;
        }
        memcpy(g0, t, sizeof(*g0));
        t->initial_geo = g0;
        atomic_store_explicit(&t->geo_snap, g0, memory_order_release);
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
    free(t->initial_geo);
    free(t);
}


/* Chained entry points share these; threading is still a run-time property. */
/*
 * Split entry paths (see the branch plan): a sequential chained table and a
 * concurrent one are separate public types, so neither pays a per-call test of
 * what it is.  The old chained_tphtNN_* functions remain as the compatibility
 * surface for handles created with the run-time `concurrent` flag and keep the
 * single dispatch branch; these cores are what the split entry points call.
 */
TPHT_HOT tpht_status_t tpht_chained_seq_write(tpht_table_t *t, uint64_t key, uint64_t value,
                                              int replace) {
    tpht_status_t st;
    key &= t->key_mask;
    if (TPHT_UNLIKELY(tpht_size_load_seq(t) >= t->write_limit)) {
        st = tpht_resize_locked(t, t->capacity * 2u);
        if (st != TPHT_OK) return st;
    }
    return tpht_chained_write_locked(t, key, value, replace);
}

/*
 * A fixed concurrent table still absorbs hard overflows, as the resize-mode
 * contract requires: the pool ran out, so grow the storage through the same
 * concurrent resize machinery a resizable table uses, and retry.  This is the
 * cold arm of an insert; the loop costs nothing until the pool is exhausted.
 */
TPHT_NOINLINE static tpht_status_t tpht_chained_fixed_write_fine(tpht_table_t *t,
                                                                 uint64_t key_word,
                                                                 uint64_t value_word,
                                                                 int replace) {
    tpht_status_t st;
    for (;;) {
        st = tpht_chained_write_fine(t, key_word, value_word, replace);
        if (TPHT_LIKELY(st != TPHT_OVERFLOW)) return st;
        if (tpht_chained_resize_active(t)) {
            tpht_chained_resize_finish_all(t);
            continue;
        }
        st = tpht_chained_ensure_resize(t, 1);
        if (st != TPHT_OK) return st;
    }
}

TPHT_HOT tpht_status_t tpht_chained_conc_write(tpht_table_t *t, uint64_t key, uint64_t value,
                                               int replace, int resizable) {
    tpht_status_t st;
    key &= t->key_mask;
    if (resizable) return tpht_chained_resizable_write_fine(t, key, value, replace);
    st = tpht_chained_write_fine(t, key, value, replace);
    if (TPHT_UNLIKELY(st == TPHT_OVERFLOW))
        st = tpht_chained_fixed_write_fine(t, key, value, replace);
    return st;
}

static tpht_status_t tpht_chained_op_write(tpht_table_t *t, uint64_t key, uint64_t value,
                                           int replace) {
    if (tpht_chained_fine_grained(t))
        return tpht_chained_conc_write(t, key, value, replace,
                                       t->cfg.resize_mode == TPHT_RESIZABLE);
    return tpht_chained_seq_write(t, key, value, replace);
}

TPHT_HOT tpht_status_t tpht_chained_conc_update(tpht_table_t *t, uint64_t key, uint64_t value) {
    return tpht_chained_update_fine(t, key & t->key_mask, value);
}

static tpht_status_t tpht_chained_op_update(tpht_table_t *t, uint64_t key, uint64_t value) {
    key &= t->key_mask;
    if (tpht_chained_fine_grained(t)) return tpht_chained_conc_update(t, key, value);
    {
        uint64_t probe_scratch;
        tpht_status_t st = tpht_chained_get_word(t, key, &probe_scratch);
        if (st != TPHT_OK) return st;
    }
    return tpht_chained_insert_word(t, key, value, 1);
}

static tpht_status_t tpht_chained_op_get(tpht_table_t *t, uint64_t key, uint64_t *value_out) {
    key &= t->key_mask;
    if (tpht_chained_fine_grained(t)) return tpht_chained_get_fine_word(t, key, value_out);
    return tpht_chained_get_word(t, key, value_out);
}

TPHT_HOT tpht_status_t tpht_chained_conc_remove(tpht_table_t *t, uint64_t key) {
    return tpht_chained_remove_fine(t, key & t->key_mask);
}

static tpht_status_t tpht_chained_op_remove(tpht_table_t *t, uint64_t key) {
    key &= t->key_mask;
    if (tpht_chained_fine_grained(t)) return tpht_chained_conc_remove(t, key);
    return tpht_chained_raw_remove(t, key);
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
    return tpht_flat_write((tpht_table_t *)table, key, value, 1, 4, 0);
}

tpht_status_t flatten_tpht32_insert(flatten_tpht32_t *table, uint32_t key, uint64_t value) {
    return tpht_flat_write((tpht_table_t *)table, key, value, 0, 4, 0);
}

tpht_status_t flatten_tpht32_update(flatten_tpht32_t *table, uint32_t key, uint64_t value) {
    return tpht_flat_update_op((tpht_table_t *)table, key, value, 4, 0);
}

tpht_status_t flatten_tpht32_get(flatten_tpht32_t *table, uint32_t key, uint64_t *value_out) {
    return tpht_flat_get_raw((tpht_table_t *)table, key, value_out, 4, 0);
}

tpht_status_t flatten_tpht32_remove(flatten_tpht32_t *table, uint32_t key) {
    return tpht_flat_remove_raw((tpht_table_t *)table, key, 4, 0);
}

size_t flatten_tpht32_size(const flatten_tpht32_t *table) { return tpht_size_of((const tpht_table_t *)table); }
size_t flatten_tpht32_capacity(const flatten_tpht32_t *table) { return tpht_capacity_of((const tpht_table_t *)table); }
size_t flatten_tpht32_memory_bytes(const flatten_tpht32_t *table) { return tpht_memory_of((const tpht_table_t *)table); }

/* --------------------------------------------------- concurrent flatten API
 * Same table, same layout; these entry points differ only in passing the
 * concurrent flag, so each block is taken through its seqlock and the
 * dereference bins through their locks.  A concurrent flattened table has a
 * fixed geometry: it cannot rebuild itself underneath live readers, so a hard
 * overflow is reported as TPHT_OVERFLOW instead of being absorbed.
 */
flatten_conc_tpht32_t *flatten_conc_tpht32_create(size_t capacity, const tpht_options_t *options) {
    return (flatten_conc_tpht32_t *)tpht_create_internal(TPHT_FLATTEN, TPHT_CONCURRENT, 4, capacity,
                                                         options);
}

flatten_conc_tpht32_t *flatten_conc_tpht32_fixed_create(size_t capacity, uint8_t value_size) {
    return (flatten_conc_tpht32_t *)tpht_named_create(TPHT_FLATTEN, TPHT_CONCURRENT, 4, TPHT_FIXED,
                                                      capacity, value_size);
}

flatten_conc_tpht32_t *flatten_conc_tpht32_resizable_create(size_t capacity, uint8_t value_size) {
    return (flatten_conc_tpht32_t *)tpht_named_create(TPHT_FLATTEN, TPHT_CONCURRENT, 4,
                                                      TPHT_RESIZABLE, capacity, value_size);
}

void flatten_conc_tpht32_destroy(flatten_conc_tpht32_t *table) {
    tpht_destroy_internal((tpht_table_t *)table);
}

tpht_status_t flatten_conc_tpht32_put(flatten_conc_tpht32_t *table, uint32_t key, uint64_t value) {
    return tpht_flat_write((tpht_table_t *)table, key, value, 1, 4, 1);
}
tpht_status_t flatten_conc_tpht32_insert(flatten_conc_tpht32_t *table, uint32_t key, uint64_t value) {
    return tpht_flat_write((tpht_table_t *)table, key, value, 0, 4, 1);
}
tpht_status_t flatten_conc_tpht32_update(flatten_conc_tpht32_t *table, uint32_t key, uint64_t value) {
    return tpht_flat_update_op((tpht_table_t *)table, key, value, 4, 1);
}
tpht_status_t flatten_conc_tpht32_get(flatten_conc_tpht32_t *table, uint32_t key, uint64_t *value_out) {
    return tpht_flat_get_raw((tpht_table_t *)table, key, value_out, 4, 1);
}
tpht_status_t flatten_conc_tpht32_remove(flatten_conc_tpht32_t *table, uint32_t key) {
    return tpht_flat_remove_raw((tpht_table_t *)table, key, 4, 1);
}
size_t flatten_conc_tpht32_size(const flatten_conc_tpht32_t *t) { return tpht_size_of((const tpht_table_t *)t); }
size_t flatten_conc_tpht32_capacity(const flatten_conc_tpht32_t *t) { return tpht_capacity_of((const tpht_table_t *)t); }
size_t flatten_conc_tpht32_memory_bytes(const flatten_conc_tpht32_t *t) { return tpht_memory_of((const tpht_table_t *)t); }

flatten_conc_tpht64_t *flatten_conc_tpht64_create(size_t capacity, const tpht_options_t *options) {
    return (flatten_conc_tpht64_t *)tpht_create_internal(TPHT_FLATTEN, TPHT_CONCURRENT, 8, capacity,
                                                         options);
}

flatten_conc_tpht64_t *flatten_conc_tpht64_fixed_create(size_t capacity, uint8_t value_size) {
    return (flatten_conc_tpht64_t *)tpht_named_create(TPHT_FLATTEN, TPHT_CONCURRENT, 8, TPHT_FIXED,
                                                      capacity, value_size);
}

flatten_conc_tpht64_t *flatten_conc_tpht64_resizable_create(size_t capacity, uint8_t value_size) {
    return (flatten_conc_tpht64_t *)tpht_named_create(TPHT_FLATTEN, TPHT_CONCURRENT, 8,
                                                      TPHT_RESIZABLE, capacity, value_size);
}

void flatten_conc_tpht64_destroy(flatten_conc_tpht64_t *table) {
    tpht_destroy_internal((tpht_table_t *)table);
}

tpht_status_t flatten_conc_tpht64_put(flatten_conc_tpht64_t *table, uint64_t key, uint64_t value) {
    return tpht_flat_write((tpht_table_t *)table, key, value, 1, 8, 1);
}
tpht_status_t flatten_conc_tpht64_insert(flatten_conc_tpht64_t *table, uint64_t key, uint64_t value) {
    return tpht_flat_write((tpht_table_t *)table, key, value, 0, 8, 1);
}
tpht_status_t flatten_conc_tpht64_update(flatten_conc_tpht64_t *table, uint64_t key, uint64_t value) {
    return tpht_flat_update_op((tpht_table_t *)table, key, value, 8, 1);
}
tpht_status_t flatten_conc_tpht64_get(flatten_conc_tpht64_t *table, uint64_t key, uint64_t *value_out) {
    return tpht_flat_get_raw((tpht_table_t *)table, key, value_out, 8, 1);
}
tpht_status_t flatten_conc_tpht64_remove(flatten_conc_tpht64_t *table, uint64_t key) {
    return tpht_flat_remove_raw((tpht_table_t *)table, key, 8, 1);
}
size_t flatten_conc_tpht64_size(const flatten_conc_tpht64_t *t) { return tpht_size_of((const tpht_table_t *)t); }
size_t flatten_conc_tpht64_capacity(const flatten_conc_tpht64_t *t) { return tpht_capacity_of((const tpht_table_t *)t); }
size_t flatten_conc_tpht64_memory_bytes(const flatten_conc_tpht64_t *t) { return tpht_memory_of((const tpht_table_t *)t); }

/* ------------------------------------------------- concurrent chained API
 * Same table, separate type: the entry points know the table is concurrent, so
 * no operation re-decides it at run time.  Created tables are identical to
 * chained_tphtNN_create(cap, 1, opts) ones; only the dispatch is pre-resolved.
 */
chained_conc_tpht32_t *chained_conc_tpht32_fixed_create(size_t capacity, uint8_t value_size) {
    return (chained_conc_tpht32_t *)tpht_named_create(TPHT_CHAINED, TPHT_CONCURRENT, 4, TPHT_FIXED,
                                                      capacity, value_size);
}
chained_conc_tpht32_t *chained_conc_tpht32_resizable_create(size_t capacity, uint8_t value_size) {
    return (chained_conc_tpht32_t *)tpht_named_create(TPHT_CHAINED, TPHT_CONCURRENT, 4,
                                                      TPHT_RESIZABLE, capacity, value_size);
}
chained_conc_tpht32_t *chained_conc_tpht32_create(size_t capacity, const tpht_options_t *options) {
    return (chained_conc_tpht32_t *)tpht_create_internal(TPHT_CHAINED, TPHT_CONCURRENT, 4, capacity,
                                                         options);
}
void chained_conc_tpht32_destroy(chained_conc_tpht32_t *t) { tpht_destroy_internal((tpht_table_t *)t); }
tpht_status_t chained_conc_tpht32_put(chained_conc_tpht32_t *t, uint32_t key, uint64_t value) {
    tpht_table_t *tt = (tpht_table_t *)t;
    return tpht_chained_conc_write(tt, key, value, 1, tt->cfg.resize_mode == TPHT_RESIZABLE);
}
tpht_status_t chained_conc_tpht32_insert(chained_conc_tpht32_t *t, uint32_t key, uint64_t value) {
    tpht_table_t *tt = (tpht_table_t *)t;
    return tpht_chained_conc_write(tt, key, value, 0, tt->cfg.resize_mode == TPHT_RESIZABLE);
}
tpht_status_t chained_conc_tpht32_update(chained_conc_tpht32_t *t, uint32_t key, uint64_t value) {
    return tpht_chained_conc_update((tpht_table_t *)t, key, value);
}
tpht_status_t chained_conc_tpht32_get(chained_conc_tpht32_t *t, uint32_t key, uint64_t *value_out) {
    tpht_table_t *tt = (tpht_table_t *)t;
    return tpht_chained_get_fine_word(tt, key & tt->key_mask, value_out);
}
tpht_status_t chained_conc_tpht32_remove(chained_conc_tpht32_t *t, uint32_t key) {
    tpht_table_t *tt = (tpht_table_t *)t;
    return tpht_chained_conc_remove(tt, key & tt->key_mask);
}
size_t chained_conc_tpht32_size(const chained_conc_tpht32_t *t) { return tpht_size_of((const tpht_table_t *)t); }
size_t chained_conc_tpht32_capacity(const chained_conc_tpht32_t *t) { return tpht_capacity_of((const tpht_table_t *)t); }
size_t chained_conc_tpht32_memory_bytes(const chained_conc_tpht32_t *t) { return tpht_memory_of((const tpht_table_t *)t); }

chained_conc_tpht64_t *chained_conc_tpht64_fixed_create(size_t capacity, uint8_t value_size) {
    return (chained_conc_tpht64_t *)tpht_named_create(TPHT_CHAINED, TPHT_CONCURRENT, 8, TPHT_FIXED,
                                                      capacity, value_size);
}
chained_conc_tpht64_t *chained_conc_tpht64_resizable_create(size_t capacity, uint8_t value_size) {
    return (chained_conc_tpht64_t *)tpht_named_create(TPHT_CHAINED, TPHT_CONCURRENT, 8,
                                                      TPHT_RESIZABLE, capacity, value_size);
}
chained_conc_tpht64_t *chained_conc_tpht64_create(size_t capacity, const tpht_options_t *options) {
    return (chained_conc_tpht64_t *)tpht_create_internal(TPHT_CHAINED, TPHT_CONCURRENT, 8, capacity,
                                                         options);
}
void chained_conc_tpht64_destroy(chained_conc_tpht64_t *t) { tpht_destroy_internal((tpht_table_t *)t); }
tpht_status_t chained_conc_tpht64_put(chained_conc_tpht64_t *t, uint64_t key, uint64_t value) {
    tpht_table_t *tt = (tpht_table_t *)t;
    return tpht_chained_conc_write(tt, key, value, 1, tt->cfg.resize_mode == TPHT_RESIZABLE);
}
tpht_status_t chained_conc_tpht64_insert(chained_conc_tpht64_t *t, uint64_t key, uint64_t value) {
    tpht_table_t *tt = (tpht_table_t *)t;
    return tpht_chained_conc_write(tt, key, value, 0, tt->cfg.resize_mode == TPHT_RESIZABLE);
}
tpht_status_t chained_conc_tpht64_update(chained_conc_tpht64_t *t, uint64_t key, uint64_t value) {
    return tpht_chained_conc_update((tpht_table_t *)t, key, value);
}
tpht_status_t chained_conc_tpht64_get(chained_conc_tpht64_t *t, uint64_t key, uint64_t *value_out) {
    tpht_table_t *tt = (tpht_table_t *)t;
    return tpht_chained_get_fine_word(tt, key & tt->key_mask, value_out);
}
tpht_status_t chained_conc_tpht64_remove(chained_conc_tpht64_t *t, uint64_t key) {
    tpht_table_t *tt = (tpht_table_t *)t;
    return tpht_chained_conc_remove(tt, key & tt->key_mask);
}
size_t chained_conc_tpht64_size(const chained_conc_tpht64_t *t) { return tpht_size_of((const tpht_table_t *)t); }
size_t chained_conc_tpht64_capacity(const chained_conc_tpht64_t *t) { return tpht_capacity_of((const tpht_table_t *)t); }
size_t chained_conc_tpht64_memory_bytes(const chained_conc_tpht64_t *t) { return tpht_memory_of((const tpht_table_t *)t); }

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
    return tpht_flat_write((tpht_table_t *)table, key, value, 1, 8, 0);
}

tpht_status_t flatten_tpht64_insert(flatten_tpht64_t *table, uint64_t key, uint64_t value) {
    return tpht_flat_write((tpht_table_t *)table, key, value, 0, 8, 0);
}

tpht_status_t flatten_tpht64_update(flatten_tpht64_t *table, uint64_t key, uint64_t value) {
    return tpht_flat_update_op((tpht_table_t *)table, key, value, 8, 0);
}

tpht_status_t flatten_tpht64_get(flatten_tpht64_t *table, uint64_t key, uint64_t *value_out) {
    return tpht_flat_get_raw((tpht_table_t *)table, key, value_out, 8, 0);
}

tpht_status_t flatten_tpht64_remove(flatten_tpht64_t *table, uint64_t key) {
    return tpht_flat_remove_raw((tpht_table_t *)table, key, 8, 0);
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
    return tpht_chained_op_write((tpht_table_t *)table, key, value, 1);
}

tpht_status_t chained_tpht32_insert(chained_tpht32_t *table, uint32_t key, uint64_t value) {
    return tpht_chained_op_write((tpht_table_t *)table, key, value, 0);
}

tpht_status_t chained_tpht32_update(chained_tpht32_t *table, uint32_t key, uint64_t value) {
    return tpht_chained_op_update((tpht_table_t *)table, key, value);
}

tpht_status_t chained_tpht32_get(chained_tpht32_t *table, uint32_t key, uint64_t *value_out) {
    return tpht_chained_op_get((tpht_table_t *)table, key, value_out);
}

tpht_status_t chained_tpht32_remove(chained_tpht32_t *table, uint32_t key) {
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
    return tpht_chained_op_write((tpht_table_t *)table, key, value, 1);
}

tpht_status_t chained_tpht64_insert(chained_tpht64_t *table, uint64_t key, uint64_t value) {
    return tpht_chained_op_write((tpht_table_t *)table, key, value, 0);
}

tpht_status_t chained_tpht64_update(chained_tpht64_t *table, uint64_t key, uint64_t value) {
    return tpht_chained_op_update((tpht_table_t *)table, key, value);
}

tpht_status_t chained_tpht64_get(chained_tpht64_t *table, uint64_t key, uint64_t *value_out) {
    return tpht_chained_op_get((tpht_table_t *)table, key, value_out);
}

tpht_status_t chained_tpht64_remove(chained_tpht64_t *table, uint64_t key) {
    return tpht_chained_op_remove((tpht_table_t *)table, key);
}

size_t chained_tpht64_size(const chained_tpht64_t *table) { return tpht_size_of((const tpht_table_t *)table); }
size_t chained_tpht64_capacity(const chained_tpht64_t *table) { return tpht_capacity_of((const tpht_table_t *)table); }
size_t chained_tpht64_memory_bytes(const chained_tpht64_t *table) { return tpht_memory_of((const tpht_table_t *)table); }
