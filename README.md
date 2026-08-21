# TPHT

Industrial C implementation of **Tiny Pointer Hash Tables**, from the paper:

> Xilin Tang, Yuqi Mai, William Kuszmaul, Alex Conway.
> **Succinct and Fast Tiny Pointer Hash Tables.**
> Proceedings of the VLDB Endowment (PVLDB) 19(9), 2026.
> [doi:10.14778/3819518.3819542](https://dl.acm.org/doi/10.14778/3819518.3819542) ·
> [arXiv:2607.28892](https://arxiv.org/abs/2607.28892)

Two variants: `chained-tpht` (tiny-pointer chains, tuned for space) and
`flatten-tpht` (64-byte home blocks, tuned for latency).  To use it, copy
`tpht.h` and `tpht.c` into your project — C11 and the standard library are the
only requirements.

> **Status:** well-tested pre-release, under active development; the API may
> still move.

## Quick start

```c
#include "tpht.h"

int main(void) {
    /* 64-bit keys, 3-byte values, room for 1024 pairs */
    flatten_tpht64_t *t = flatten_tpht64_fixed_create(1024, 3);
    uint64_t value;

    flatten_tpht64_put(t, 42, 9001);
    if (flatten_tpht64_get(t, 42, &value) == TPHT_OK) {
        /* value == 9001 */
    }

    flatten_tpht64_destroy(t);
    return 0;
}
```

```sh
cc -std=c11 -O2 -I. tpht.c your_file.c -o your_program
```

## The eight types

```
flatten_tpht32_t        flatten_tpht64_t         sequential flattened
flatten_conc_tpht32_t   flatten_conc_tpht64_t    concurrent flattened
chained_tpht32_t        chained_tpht64_t         sequential chained
chained_conc_tpht32_t   chained_conc_tpht64_t    concurrent chained
```

One type per (variant, key width, threading), so every code path is fixed at
compile time — no configuration test on any hot path.  All eight come from the
one `tpht.c` and can be mixed freely in one program.  Each type carries its own
constructors and operations:

```c
flatten_tpht64_fixed_create(capacity, value_size);
flatten_tpht64_resizable_create(capacity, value_size);
flatten_tpht64_put / _insert / _update / _get / _remove / _destroy
flatten_tpht64_size / _capacity / _memory_bytes
```

Keys are `uint32_t` or `uint64_t` by type.  Values cross the API as `uint64_t`
and are truncated to the table's value size (1 to 8 bytes, chosen per table).
The remaining knobs live in an options struct, where zero means default:

```c
tpht_options_t o = tpht_default_options();
o.resize_mode = TPHT_RESIZABLE;      /* also: value_size, max_load_factor, hash_seed */
flatten_conc_tpht64_t *t = flatten_conc_tpht64_create(1 << 20, &o);
```

## What to expect

- **A table never fills up on you.**  Both fixed and resizable tables absorb
  overflow by growing internally; a write fails only on allocation failure or
  a structural ceiling (status-code details in
  [docs/INTERNALS.md](docs/INTERNALS.md)).
- **`insert` appends duplicates**; `put`/`update` overwrite.  In the flattened
  variant ~30 duplicates of one single key is a structural ceiling, and the
  32-bit flattened types top out near 190M entries because quotienting
  consumes all 32 key bits — use the 64-bit types beyond that.
- **Concurrent tables** are safe for any mix of operations from any number of
  threads and resize online: readers never block, writers lock one 64-byte
  block and pay at most a bounded slice of migration work.

## SIMD

Fingerprint screening is the flattened variant's hot loop, and SIMD matters a
lot there: with vectorized screening (picked automatically — AVX2, SSE2, or
NEON, at runtime where possible) lookups run about **2× faster** than the
scalar fallback.  Measured with the stock latency benchmark, 4M keys at load
0.85, medians of alternating runs on a pinned core:

| flatten op   | scalar | SIMD (AVX2) | speedup |
|--------------|-------:|------------:|--------:|
| lookup hit   | 57.9 ns |    25.4 ns | 2.3× |
| lookup miss  | 42.9 ns |    21.7 ns | 2.0× |
| remove       | 97.9 ns |    87.7 ns | 1.1× |
| insert       | 25.6 ns |    25.7 ns | 1.0× |

Nothing to configure: the default build does the right thing.  If your
toolchain or target cannot use SIMD, `-DTPHT_ENABLE_SIMD=0` forces the scalar
fallback — correct, just slower to look up.  Forced instruction-set builds for
testing are described in [docs/INTERNALS.md](docs/INTERNALS.md).

## Tests and benchmarks

```sh
./tests/run_tpht_tests.sh          # exhaustive suite across all SIMD levels
./benchmarks/run_latency.sh        # latency sweep -> results/*.csv (+ PDF if matplotlib)
./benchmarks/run_space_eff.sh      # space efficiency -> results/*.csv
./benchmarks/plot_tradeoff.py      # pair the two CSVs into a space-vs-throughput figure
```

## More

Design and internals — block layout, quotienting, overflow handling, the
concurrent resize, hashing, SIMD modes, test architecture — are in
[docs/INTERNALS.md](docs/INTERNALS.md).

## References

- Xilin Tang, Yuqi Mai, William Kuszmaul, Alex Conway.
  *Succinct and Fast Tiny Pointer Hash Tables.* PVLDB 19(9), 2026.
  [doi:10.14778/3819518.3819542](https://dl.acm.org/doi/10.14778/3819518.3819542),
  [arXiv:2607.28892](https://arxiv.org/abs/2607.28892).
- [TinyPtr](https://github.com/Xilinion/TinyPtr): the paper's artifact
  evaluation code base.
- xxHash / XXH3 by Yann Collet: TPHT embeds a small dependency-free subset of
  XXH3 (BSD 2-Clause): https://github.com/Cyan4973/xxHash
