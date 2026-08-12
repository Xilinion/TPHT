# TPHT

> **Development status: experimental / under active development.**
>
> TPHT is currently being rewritten and validated. APIs, internal layouts,
> concurrency behavior, and benchmark results may still change. At this stage,
> correctness is **not guaranteed**, and performance is even less so. Do **not**
> treat this repository as production-stable yet. The current target for a
> stabilized implementation is around late August.

Industrial C implementation of Tiny Pointer Hash Tables.  This repository keeps only the two intended variants from the academic TinyPtr codebase:

- `chained-tpht`: derived from the `byte_array_chained_ht` idea.
- `flatten-tpht`: derived from the `blast_ht` idea, with 64-byte home blocks, inline fingerprints, and tiny pointers to overflow entries.

The code is designed to be copied directly into another project: copy `tpht.h` and `tpht.c`.

## Features

- C11 implementation, no C++ dependency.
- **Keys are 4 or 8 bytes, with a separate API for each width.  Values are any
  size from 1 to 8 bytes.**
- Chained variant: sequential and concurrent, fixed-capacity and resizable.
- Flattened variant: sequential only for now, fixed-capacity or resizable.
- Resizing uses normal capacity doubling.
- XXH64 hashing, embedded directly for copy-paste portability.
- SIMD optimized fingerprint matching for `flatten-tpht` when available.
- Safe scalar fallback when SIMD is disabled or unavailable.

## Quick start

```c
#include "tpht.h"

int main(void) {
    /* 64-bit keys, 3-byte values, room for 1024 pairs */
    tpht_table_t *t = flatten_tpht64_fixed_create(1024, 3);
    uint64_t value;

    tpht64_put(t, 42, 9001);
    if (tpht64_get(t, 42, &value) == TPHT_OK) {
        /* value == 9001 */
    }

    tpht_destroy(t);
    return 0;
}
```

Compile:

```sh
cc -std=c11 -O2 -I. tpht.c your_file.c -o your_program
```

## Constructors

Each key width has its own family.  `value_size` is in bytes and may be anything
from 1 to 8.

```c
/* 32-bit keys */
chained_tpht32_fixed_create(capacity, value_size);
chained_tpht32_resizable_create(capacity, value_size);
chained_tpht32_concurrent_fixed_create(capacity, value_size);
chained_tpht32_concurrent_resizable_create(capacity, value_size);
flatten_tpht32_fixed_create(capacity, value_size);
flatten_tpht32_resizable_create(capacity, value_size);

/* 64-bit keys: the same six with tpht64 */
```

Concurrency is not implemented for the flattened variant yet;
`flatten_tpht32_concurrent_*` and `flatten_tpht64_concurrent_*` are declared for
source compatibility and always return `NULL`.

## Operations

```c
tpht32_put(t, key, value);          /* uint32_t key, uint64_t value */
tpht32_insert(t, key, value);
tpht32_update(t, key, value);
tpht32_get(t, key, &value);
tpht32_remove(t, key);

/* and the same five with tpht64, taking a uint64_t key */
```

A `tpht32_*` call on a 64-bit-key table returns `TPHT_INVALID`, and vice versa.
The width-agnostic `tpht_put`, `tpht_insert`, `tpht_update`, `tpht_get` and
`tpht_remove` take a `uint64_t` key and work on either.

Values cross the API as `uint64_t` and are truncated to the table's value size,
so a table with a 3-byte value stores and returns values modulo 2^24.

## Generic configuration

Use `tpht_create` when you want to configure everything explicitly:

```c
tpht_config_t cfg = tpht_default_config();
cfg.variant = TPHT_FLATTEN;
cfg.threading = TPHT_SEQUENTIAL;
cfg.resize_mode = TPHT_FIXED;
cfg.initial_capacity = 1 << 20;
cfg.key_size = 8;
cfg.value_size = 5;

tpht_table_t *t = tpht_create(&cfg);
```

`tpht_create` returns `NULL` for `TPHT_FLATTEN` combined with `TPHT_CONCURRENT`.

## Flattened layout

Each quotient group owns exactly one 64-byte home block, so a lookup reads one
cache line and, at most, one more for an overflow entry.  There is no chaining.

```
byte 0 ........................................................ byte 63
[ fingerprints -> ] [ free ] [ <- tiny pointers ][ <- crystals ][ctl][ver]
```

- **Fingerprints** grow up from offset 0, one byte per tuple stored in *or via*
  this block, home tuples first and overflow tuples after.  One SIMD compare
  screens all of them at once.
- **Crystals** are whole quotiented key/value entries stored in the line itself
  and grow down from the control byte.
- **Tiny pointers** are one byte each, address the dereference table, and grow
  down from the bottom of the crystal region.
- **Control byte** holds the number of tuples in the block, 0 to 31, in its low
  5 bits.  How many of them are crystals follows from that count and the entry
  size, because a block always keeps as many tuples inline as its remaining
  space allows, so it does not have to be stored.  TinyPtr instead splits this
  byte into a 3-bit crystal count and a 5-bit tiny pointer count, which caps
  inline tuples at 7 even when many more would fit - with 4-byte keys and 1-byte
  values a block has room for 20.
- **Version byte** is a seqlock counter, kept from the TinyPtr layout so the
  block is ready for concurrent readers; the sequential code bumps it around
  every mutation.

Keys are quotiented with a one-round Feistel permutation over
`block_bits + 8` bits: the low bits index the home block and the extra byte
*is* the fingerprint, so neither has to be stored.  Only the remaining key
remainder and the value live in an entry, which is therefore
`ceil((key_bits - block_bits - 8) / 8) + value_size` bytes.

Because the crystal count is a function of the tuple count, insertion reduces to
comparing `crystals(x)` with `crystals(x + 1)`:

- `crystals(x+1) == crystals(x) + 1` - the pair itself fits in the line.
- otherwise - a **soft overflow**: the new tuple, along with any inline tuple the
  block can no longer afford, migrates to the dereference table and is addressed
  by a tiny pointer.  This is the paper's eviction case, generalised to evict as
  many tuples as the arithmetic calls for.

Deletion is the inverse and promotes overflow tuples back into the line whenever
whole entries fit again.

### Overflow handling

A **hard overflow** is a structural failure rather than a full table: either a
home block cannot address another tuple at all (31 of them, the counter's
limit), or the dereference table cannot hand out an entry.  Both are absorbed
automatically - the table is rebuilt with twice the home blocks and the write is
retried - so neither is ever reported to the caller.  This happens whether or
not the table was created resizable; a fixed table keeps the capacity it was
given and only its internal geometry grows.  All the dereference entries a write
needs are reserved before the home block is touched, so a hard overflow leaves
the block exactly as it was.

**Sizing.** The target number of keys per block is however many entries fit in
a line before anything spills: 4 for 8-byte keys and values, which is the
paper's figure, and up to 20 for the smallest entries.  The
block index is masked out of the quotient, so the block count is rounded to the
*nearer* power of two - rounding up alone can leave a home array twice as large
as intended.  The dereference table is then sized from the Poisson expectation
of the overflow at the average block load that results, plus headroom.  A
capacity near the target load times a power of two gives the tightest layout.

## SIMD and portability

By default, `tpht.c` enables SIMD-aware fingerprint scanning for `flatten-tpht`:

- x86/x86_64: AVX2 is selected at runtime when the CPU supports it; otherwise SSE2/scalar is used.
- ARM with NEON: NEON is used when available at compile time.
- Other CPUs: scalar fallback is used.

You can force scalar-only compilation:

```sh
cc -std=c11 -O2 -DTPHT_ENABLE_SIMD=0 -I. tpht.c your_file.c -o your_program
```

For instruction-set testing, `TPHT_SIMD_MODE` supports:

```c
TPHT_SIMD_AUTO   /* default */
TPHT_SIMD_SCALAR
TPHT_SIMD_SSE2
TPHT_SIMD_AVX2
TPHT_SIMD_NEON
```

Only run forced instruction-set binaries on machines that support that instruction set.

## Tests

Run the exhaustive test script:

```sh
./tests/run_tpht_tests.sh
```

The script covers:

- scalar fallback build
- default portable SIMD build
- forced SSE2 build when supported
- forced AVX2 build when supported by compiler and runtime
- forced NEON build when supported
- pthread concurrent stress build when `-pthread` is available
- `-march=native` build when supported

The test program covers all supported variant/threading/resize combinations for
every key size (4, 8) and value size (1 through 8), plus duplicate insertion, updates, removals, fixed-table full behavior, doubling
resize behavior, invalid configs, storage-layout checks, randomized model
checking, flattened fill/drain/refill and block-churn checks, and real threaded
insertion stress.

The tests are split into controlled-granularity modules:

- `tests/tpht_test.c`: single aggregate entrypoint.
- `tests/tpht_test_common.[ch]`: shared helpers and case enumeration.
- `tests/tpht_test_config.c`: invalid configuration, storage layout, and flattened block-churn tests.
- `tests/tpht_test_constructors.c`: generic and variant-specific constructor tests.
- `tests/tpht_test_api_edges.c`: API edge/error/full-table behavior.
- `tests/tpht_test_deterministic.c`: deterministic insert/get/update/remove/resize checks.
- `tests/tpht_test_random_model.c`: randomized model-check tests.
- `tests/tpht_test_threads.c`: pthread-backed concurrent stress tests when enabled.

## Acknowledgements

- Tiny Pointer Hash Tables / TinyPtr: source design inspiration for the chained and flatten variants.
- xxHash / XXH64 by Yann Collet: TPHT embeds a small dependency-free XXH64 implementation so users can still copy only `tpht.h` and `tpht.c`. xxHash is BSD 2-Clause licensed: https://github.com/Cyan4973/xxHash
