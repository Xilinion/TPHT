# TPHT

Industrial C implementation of Tiny Pointer Hash Tables.  This repository keeps only the two intended variants from the academic TinyPtr codebase:

- `chained-tpht`: derived from the `byte_array_chained_ht` idea.
- `flatten-tpht`: derived from the `blast_ht` idea, with inline fingerprint groups and overflow tiny pointers.

The code is designed to be copied directly into another project: copy `tpht.h` and `tpht.c`.  If you prefer variant-specific names, also copy `chained_tpht.h` and/or `flatten_tpht.h`.

## Features

- C11 implementation, no C++ dependency.
- Sequential and concurrent modes.
- Fixed-capacity and resizable modes.
- Resizing uses normal capacity doubling.
- Key and value sizes can independently be 2, 4, or 8 bytes.
- SIMD optimized fingerprint matching for `flatten-tpht` when available.
- Safe scalar fallback when SIMD is disabled or unavailable.

## Quick start

```c
#include "tpht.h"

int main(void) {
    tpht_table_t *t = tpht_flatten_resizable_create(1024, 8, 8);
    unsigned long long value;

    tpht_put_u64(t, 42, 9001);
    if (tpht_get_u64(t, 42, &value) == TPHT_OK) {
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

## Variant-specific headers

For explicit chained usage:

```c
#include "chained_tpht.h"

chained_tpht_t *t = chained_tpht_resizable_create(1024, 8, 8);
```

For explicit flattened usage:

```c
#include "flatten_tpht.h"

flatten_tpht_t *t = flatten_tpht_concurrent_resizable_create(1024, 8, 8);
```

Both headers are thin wrappers over `tpht.h`; `tpht.c` is still the only source file needed.

## Constructors

Chained variant:

```c
tpht_chained_fixed_create(capacity, key_size, value_size);
tpht_chained_resizable_create(capacity, key_size, value_size);
tpht_chained_concurrent_fixed_create(capacity, key_size, value_size);
tpht_chained_concurrent_resizable_create(capacity, key_size, value_size);
```

Flattened variant:

```c
tpht_flatten_fixed_create(capacity, key_size, value_size);
tpht_flatten_resizable_create(capacity, key_size, value_size);
tpht_flatten_concurrent_fixed_create(capacity, key_size, value_size);
tpht_flatten_concurrent_resizable_create(capacity, key_size, value_size);
```

`key_size` and `value_size` must be one of `2`, `4`, or `8`.

## Generic configuration

Use `tpht_create` when you want to configure everything explicitly:

```c
tpht_config_t cfg = tpht_default_config();
cfg.variant = TPHT_FLATTEN;
cfg.threading = TPHT_CONCURRENT;
cfg.resize_mode = TPHT_RESIZABLE;
cfg.initial_capacity = 1 << 20;
cfg.key_size = 8;
cfg.value_size = 4;
cfg.max_load_factor = 0.80;

tpht_table_t *t = tpht_create(&cfg);
```

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

Example forced AVX2 test build:

```sh
cc -std=c11 -O2 -mavx2 -DTPHT_SIMD_MODE=3 -I. tpht.c tests/tpht_test.c -o tpht_avx2_test
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

The test program covers all variants, threading modes, resize modes, and all 2/4/8-byte key-value combinations, plus duplicate insertion, updates, removals, fixed-table full behavior, doubling resize behavior, invalid configs, randomized model checking, and real threaded insertion stress.
