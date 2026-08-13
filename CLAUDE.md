# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

sqlite-ndvss is a dependency-free SQLite extension (loadable shared library) that adds vector
similarity search SQL functions — cosine, euclidean distance (and squared), and dot product —
over BLOBs of packed `float` or `double` arrays. It also provides helpers to convert a text/JSON
array into such a BLOB. The similarity kernels are naive (no ANN indexing); the only optimization
is dispatching to CPU-specific SIMD intrinsics at runtime.

There is no build system for tests, no package manager, and no CI config in this repo — compiling
the `.c` file into a shared library *is* the primary workflow.

## Build

The extension needs SQLite's amalgamation files (`sqlite3.c`, `sqlite3.h`, `sqlite3ext.h`) present
in the repo root to compile; they are gitignored and must be downloaded from
https://sqlite.org/download.html and copied in first.

**Quick local build** (matches `compile.sh`):
```sh
gcc -g -fPIC -shared sqlite-ndvss.c -o ndvss.so -O3 -ffast-math
```

**Cross-compilation via `make`** (requires `zig`) builds per-target shared libraries into
subdirectories:
```sh
make x86          # x86_64 Linux -> x86/ndvss.so
make arm64         # AArch64 Linux (+SVE2 cpu features) -> arm64/ndvss.so
make riscv         # RISC-V RV64GCV -> riscv/ndvss.so
make win64         # Windows x64 -> win64/ndvss.dll
make win-arm64     # Windows ARM64 -> win-arm64/ndvss.dll
make macos-arm64   # macOS Apple Silicon -> macos-arm64/ndvss.dylib
make macos-x64     # macOS Intel -> macos-x64/ndvss.dylib
make all           # builds every target above
make clean         # removes all target output directories
```

**Manual gcc build per platform**, with a `-mavx` fallback for pre-2013 x86_64 CPUs without AVX2
(see README.md "Compilation" section for the exact flag matrix):
- Windows: `gcc -g -shared sqlite-ndvss.c -o ndvss.dll -mavx2 -mfma -Ofast -ffast-math`
- Linux: `gcc -g -fPIC -shared sqlite-ndvss.c -o ndvss.so -mavx2 -mfma -Ofast -ffast-math`
- Mac: `gcc -g -fPIC -dynamiclib sqlite-ndvss.c -o ndvss.dylib -mavx2 -mfma -Ofast -ffast-math`

`-ffast-math` trades numeric accuracy for speed; omit it when accuracy matters more than raw
throughput. Loading the built extension in the `sqlite3` CLI: `.load ./ndvss`.

There is no automated test suite in this repo. Verification is done manually by loading the
extension in a SQLite shell and running example queries (see `examples/examples.md`), and by
checking `SELECT ndvss_instruction_set();` to confirm the expected SIMD path was selected at
runtime.

## Architecture

### Runtime CPU dispatch, not compile-time-only

`sqlite-ndvss.c` defines global function pointers (`cosine_func_f/d`, `euclidean_func_f/d`,
`dot_product_func_f/d`) that default to the `_basic` (scalar) implementations. In the extension's
entrypoint `sqlite3_ndvss_init`, CPU features are probed at *runtime* and the pointers are
repointed to the fastest available SIMD implementation:

- x86_64: `__get_cpuid`/`__get_cpuid_count` checks for SSE4.1 → AVX → AVX2 → AVX512F, in that
  priority order (higher wins).
- AArch64 Linux: reads `AT_HWCAP2` via `getauxval` to detect SVE2, else falls back to NEON.
- AArch64 macOS (Apple Silicon): always NEON (no SVE2 path on Apple Silicon).
- AArch64 Windows: `IsProcessorFeaturePresent(PF_ARM_SVE_INSTRUCTIONS_AVAILABLE)` when
  `__ARM_FEATURE_SVE` is compiled in, else NEON.
- RISC-V with the vector extension (`__riscv_vector`): RVV.
- RISC-V without vector extension: falls through to `_basic`.

The `LOAD_SIMILARITY_FUNCTIONS(SUFFIX)` macro (near the top of `sqlite-ndvss.c`) wires all six
function pointers to the `_<SUFFIX>` variants in one shot and records the chosen instruction set
name into `g_instruction_set`, exposed via SQL as `ndvss_instruction_set()`.

### One header per instruction set, one naming convention

Each `similarity_functions_<isa>.h` (`basic`, `sse41`, `avx`, `avx2`, `avx512f`, `neon`, `sve2`,
`rvv`) implements the same six kernels with an identical naming scheme:

```
cosine_similarity_f_<isa>              cosine_similarity_d_<isa>
euclidean_distance_similarity_f_<isa>  euclidean_distance_similarity_d_<isa>
dot_product_similarity_f_<isa>         dot_product_similarity_d_<isa>
```

(`_f` = float32 variant, `_d` = float64 variant.) `similarity_functions.h` conditionally includes
only the header(s) relevant to the target architecture (AVX family on x86_64; NEON+SVE2 on
AArch64; RVV on RISC-V-with-V), based on preprocessor checks (`__aarch64__`, `__riscv_vector`,
etc.), so `_basic` is always available as the guaranteed fallback and other headers are opt-in
per-arch.

When adding a new ISA backend: create `similarity_functions_<isa>.h` implementing all six
functions under that naming convention, `#include` it from `similarity_functions.h` behind the
correct arch guard, add a runtime feature-detection branch in `sqlite3_ndvss_init` in
`sqlite-ndvss.c` that calls `LOAD_SIMILARITY_FUNCTIONS(<isa>)`, and add a `make <target>` rule if
it corresponds to a new cross-compilation target.

### Cosine similarity's two-pass shape

Unlike euclidean/dot-product, cosine similarity kernels take two extra output pointers
(`divider_a`, `divider_b`) and return the raw dot product; they do **not** divide internally. The
SQL wrapper functions (`ndvss_cosine_similarity_f/d` in `sqlite-ndvss.c`) compute
`sqrt(divider_a * divider_b)` and divide after the call, and short-circuit to `0.0` if either
divider is `0.0` (avoiding a division by zero for zero-vectors). Keep this split when touching
cosine kernels — the normalization step is deliberately kept out of the SIMD kernels.

### SQL-function argument conventions

All `ndvss_*` SQL functions registered in `sqlite3_ndvss_init` follow the same argument shape:
`(searched_vector BLOB, compared_vector/column BLOB, [num_dimensions INT])`. The dimension count
is optional — when omitted or `NULL`, it's inferred from `sqlite3_value_bytes(argv[0]) /
sizeof(float|double)`. All BLOB-arg functions validate that both BLOBs are non-NULL and have equal
byte length before computing. `ndvss_dot_product_similarity_str` is the exception: it takes TEXT
arrays instead of BLOBs, requires the dimension count explicitly, and caches the parsed first
argument via `sqlite3_get_auxdata`/`sqlite3_set_auxdata` — it exists mainly for testing/convenience
and is documented as inefficient relative to the BLOB-based functions.

### Registering a new SQL function

Follow the existing repeated pattern at the bottom of `sqlite3_ndvss_init`: call
`sqlite3_create_function` with `SQLITE_UTF8|SQLITE_INNOCUOUS|SQLITE_DETERMINISTIC`, check `rc !=
SQLITE_OK` and propagate `sqlite3_errmsg(db)` via `*pzErrMsg` on failure.
