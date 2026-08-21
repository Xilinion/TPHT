# TPHT internals

Design notes for readers of the code.  The user-facing contract is in the
[README](../README.md); everything here is implementation.

TPHT is an independent implementation of the paper's designs, not a wrapper
around the [artifact evaluation code base](https://github.com/Xilinion/TinyPtr):
it neither includes nor links it.  The library needs only a C11 compiler and
the C standard library; its XXH3 word-hashing path is embedded in `tpht.c`,
and the benchmark runners compile directly from this repository.  Python and
Matplotlib are optional, used only to turn latency CSVs into a PDF.

## The type split

One concrete type per (variant, key width, threading) means the hash, the key
mask, the threading discipline and the code path are all fixed before the
call: there is no `if (table->key_size == 4)` — and no
`if (table->concurrent)` — on any hot path.  Internally the entry points pass
compile-time constants (`key_bytes`, `concurrent`, the insert/put/update
replace mode) into force-inlined shared logic, so each public function
compiles to its own specialized body inside the one translation unit.  The
test suite enforces this with a disassembly tripwire
(`tools/check_config_branches.sh`): it fails the run if any hot entry point
compares a table-configuration field.

The older `chained_tphtNN_concurrent_*_create` constructors, which return a
`chained_tphtNN_t` that dispatches on a runtime flag, remain supported for
compatibility; the `chained_conc` types exist to skip that branch.

## Status codes

```c
TPHT_OK          TPHT_NOT_FOUND          TPHT_OVERFLOW          TPHT_NO_MEMORY
```

`TPHT_NOT_FOUND` is a get/update/remove miss.  `TPHT_OVERFLOW` is *structural*
overflow only — no amount of growth could absorb the write (see the ceilings
under Overflow handling); ordinary fullness grows the table internally, in
fixed and resizable modes alike, and is never reported.  `TPHT_NO_MEMORY`
means a needed allocation failed; the table is untouched and keeps serving at
its current size.  Most callers only ever branch on `TPHT_OK` versus
`TPHT_NOT_FOUND`.

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
- **Control byte** holds the number of tuples in the block in its low 5 bits.
  A block holds at most **30** tuples: the field could encode 31, but 31 never
  fit — even all-spilled they need 31 fingerprints plus 31 tiny pointers, one
  byte more than the 61 usable.  How many tuples are crystals follows from the
  count and the entry size, because a block always keeps as many tuples inline
  as its remaining space allows, so it does not have to be stored.  TinyPtr
  instead splits this byte into a 3-bit crystal count and a 5-bit tiny pointer
  count, which caps inline tuples at 7 even when many more would fit — with
  4-byte keys and 1-byte values a block has room for 20.
- **Version byte** is a seqlock counter: even means stable, odd means a writer
  is inside.  Concurrent readers retry on a torn read; the sequential code
  bumps it around every mutation so a table's blocks are always
  reader-compatible.

Keys are quotiented with a one-round Feistel permutation over
`block_bits + 8` bits: the low bits index the home block and the extra byte
*is* the fingerprint, so neither has to be stored.  Only the remaining key
remainder and the value live in an entry, which is therefore
`ceil((key_bits - block_bits - 8) / 8) + value_size` bytes.

Because the crystal count is a function of the tuple count, insertion reduces
to comparing `crystals(x)` with `crystals(x + 1)`:

- `crystals(x+1) == crystals(x) + 1` — the pair itself fits in the line.
- otherwise — a **soft overflow**: the new tuple, along with any inline tuple
  the block can no longer afford, migrates to the dereference table and is
  addressed by a tiny pointer.  This is the paper's eviction case, generalised
  to evict as many tuples as the arithmetic calls for.

Deletion is the inverse and promotes overflow tuples back into the line
whenever whole entries fit again.

## Overflow handling

A **hard overflow** is a structural failure rather than a full table: either a
home block cannot hold another tuple (30 of them, its capacity), or the
dereference table cannot hand out an entry.  Both are absorbed automatically —
the table is rebuilt with twice the home blocks and the write is retried — so
neither is ever reported to the caller.  This happens whether or not the table
was created resizable; a fixed table keeps the capacity it was given and only
its internal geometry grows.  All the dereference entries a write needs are
reserved before the home block is touched, so a hard overflow leaves the block
exactly as it was.

Two structural ceilings exist that growth cannot absorb, and only they produce
`TPHT_OVERFLOW`:

- All duplicates of one key share one home block forever, so about 30 copies
  of a *single* key is the flattened variant's limit.  The chained variant has
  no such bound.
- The 32-bit flattened types run out of key bits.  Quotienting spends the
  key's own bits on the block index and the 8-bit fingerprint; at 2^24 blocks
  a 32-bit key is fully consumed (24 + 8 = 32, the stored remainder is
  already zero bytes), so growth cannot add blocks — the same bound the
  paper's scheme implies.  2^24 blocks × 30 tuples would allow ~500M entries
  if keys spread evenly, but they arrive Poisson: around 190M some block
  exceeds what it and the dereference table can absorb, and since growth
  cannot spread keys further the write honestly reports `TPHT_OVERFLOW`
  instead of migrating the whole table for nothing.  Key sets past that scale
  belong in the 64-bit types (a `uint32_t` key fits unchanged), which clamp
  at 2^55 blocks and never bind in practice.

**Sizing.**  The target number of keys per block is however many entries fit
in a line before anything spills: 4 for 8-byte keys and values, which is the
paper's figure, and up to 20 for the smallest entries.  The block index is
masked out of the quotient, so the block count is rounded to the *nearer*
power of two — rounding up alone can leave a home array twice as large as
intended.  The dereference table is then sized from the Poisson expectation of
the overflow at the average block load that results, plus headroom.  A
capacity near the target load times a power of two gives the tightest layout.

## Concurrency

Concurrent tables are safe for any mix of operations from any number of
threads, and they resize online.  The design keeps readers out of every
resize:

- **Readers are lock-free.**  A lookup walks an immutable snapshot of the
  table's geometry reached through one atomic pointer, screened by per-block
  seqlocks, and never blocks — not even mid-resize.  Retired storages are kept
  until destroy and never address-reused, so validation is pointer identity.
- **Writers lock one 64-byte block** (a one-byte seqlock), never the table.
- **Resize never has a stop-the-world phase.**  The old storage stays
  authoritative until the whole migration is done: every write lands there
  first, so queries always answer from one complete table.  Blocks migrate to
  the shadow in parallel strides that writers absorb in bounded slices; a
  write to an already-migrated block is also mirrored into the shadow under
  the same block lock, so at commit the shadow is already exact.  The commit
  is a pointer swap taken under the old block locks.
- **A resize that fails is abandoned, not committed.**  If the shadow cannot
  absorb an entry, the old table — which already holds every write — simply
  keeps serving, and the next attempt escalates its sizing.  Termination is
  honest: repeated failure ends in `TPHT_NO_MEMORY` or, when geometry is
  pinned, `TPHT_OVERFLOW`.

## Hashing

The library only ever hashes a single machine word, so it embeds just XXH3's
4-to-8-byte path rather than the whole algorithm — no secret table, no
streaming state.  Results are bit-identical to `XXH3_64bits_withSeed()` over
the key's little-endian bytes, verified against upstream xxHash for both
widths.

Each key width has its own hash, so a 32-bit table never pays for 64-bit work.
Either can be replaced without touching anything else:

```sh
cc -DTPHT_HASH32'(w,s)'=my_hash32 -DTPHT_HASH64'(w,s)'=my_hash64 -c tpht.c
```

Both macros take a word and a seed and return `uint64_t`.

## SIMD and portability

By default, `tpht.c` enables SIMD-aware fingerprint scanning for
`flatten-tpht`:

- x86/x86_64: AVX2 is selected at runtime when the CPU supports it; otherwise
  SSE2/scalar is used.
- ARM with NEON: NEON is used when available at compile time.
- Other CPUs: scalar fallback is used.

Scalar-only compilation: `-DTPHT_ENABLE_SIMD=0`.  For instruction-set testing,
`TPHT_SIMD_MODE` accepts `TPHT_SIMD_AUTO` (default), `TPHT_SIMD_SCALAR`,
`TPHT_SIMD_SSE2`, `TPHT_SIMD_AVX2`, or `TPHT_SIMD_NEON`.  Only run forced
instruction-set binaries on machines that support that instruction set.

## Tests

`./tests/run_tpht_tests.sh` builds and runs the suite at every applicable
level: scalar fallback, default portable SIMD, forced SSE2/AVX2/NEON where
supported, a pthread concurrent stress build, and a `-march=native` build that
also runs the config-branch tripwire.

The test program covers all variant/threading/resize combinations for every
key size (4, 8) and value size (1 through 8), plus duplicate insertion,
updates, removals, doubling resize behavior, invalid configs, storage-layout
checks, randomized model checking, flattened fill/drain/refill and block-churn
checks, block and pool saturation (hard overflow, duplicate ceilings), a
model-checked churn workload across every configuration, and real threaded
insertion stress.  Modules:

- `tests/tpht_test.c`: single aggregate entrypoint.
- `tests/tpht_test_common.[ch]`: shared helpers and case enumeration.
- `tests/tpht_test_config.c`: invalid configuration, storage layout, and flattened block-churn tests.
- `tests/tpht_test_constructors.c`: generic and variant-specific constructor tests.
- `tests/tpht_test_api_edges.c`: API edge/error behavior.
- `tests/tpht_test_deterministic.c`: deterministic insert/get/update/remove/resize checks.
- `tests/tpht_test_random_model.c`: randomized model-check tests.
- `tests/tpht_test_saturation.c`: hard-overflow and duplicate-ceiling saturation tests.
- `tests/tpht_test_churn.c`: model-checked insert/remove/update churn across all configurations.
- `tests/tpht_test_threads.c`: pthread-backed concurrent stress tests when enabled.

For ThreadSanitizer runs, `tools/tsan.supp` documents the two by-design
seqlock races and nothing else; the file's header has the exact invocation.

## Benchmarks

`run_latency.sh` sweeps table sizes from `2^min_log2` to `2^max_log2` keys
(default 2^10 to 2^24) for both variants and both key widths, timing four
phases per size: insert, lookup of a present key, lookup of an absent key, and
remove.  Tables are sequential and fixed-capacity so no resize work lands in
the measurement; lookups and removes run in a shuffled order.  Each figure is
the best of several repetitions, and small tables get more repetitions than
large ones.  Keys are produced inside the timed loop by a bijective mixer, so
the same few nanoseconds of key derivation are included in every number and
the memory traffic being measured is the table's own.

If `matplotlib` is present the script also renders `results/tpht_latency.pdf`
(vector, TrueType-embedded, so it drops straight into a paper); otherwise it
leaves the CSV and says so.  `benchmarks/plot_latency.py` can be re-run on the
CSV by hand.
