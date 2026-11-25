# diffstruc ownership migration — verification

This document records the migration of `array_type` from an owning,
pointer-based tensor/graph type to a lightweight, non-owning handle, and the
verification of the migrated library.

## 1. Target architecture

`array_type` is a handle:

| Component | Kind | Owner |
|---|---|---|
| `id`, `gen`, `ctx` | integers | handle |
| `val` | pointer view | `array_context_type` |
| `shape`, `indices`, `adj_ja`, `mask`, `direction` | allocatable views | node record (freed with the record) |
| `grad`, `left_operand`, `right_operand` | node pointers | `array_context_type` |
| `is_sample_dependent`, `is_temporary` | pointer views of a context-owned flag | node record |
| scalar flags, `operation`, `visit_tag` | values | handle copy |
| partial-derivative procedure pointers | procedure pointers | handle copy |

Node records live in fixed-size chunks of a context registry. Chunks are never
grown or moved, so a pointer into a record stays valid until the context
releases the record. There is no reference count, no finaliser and no per-handle
deallocation.

The context API is `array_context_type` with module procedures
`allocate_node`, `allocate_value_storage`, `find_node`, `release_node`,
`checkpoint_context`, `rollback_context`, `reset_context`, plus the global
`default_context`, `set_default_context` and `reset_default_context`.

Memory is reclaimed in three ways:

1. `nullify_graph` releases the operand and gradient records of an expression
   (and the root record when it is a temporary), keeping root variables alive.
2. `assign_and_deallocate_source` releases the temporary source expression
   after copying its values.
3. `reset_context`/`reset_default_context` release every record, every value
   buffer, every gradient and the scratch pool of a context, and bump the
   context generation so that earlier handles are rejected as stale instead of
   being dereferenced.

## 2. Files

| File | Change |
|---|---|
| `src/diffstruc/diffstruc_types.f90` | rewritten: handle definition, context ownership types, all public interfaces |
| `src/diffstruc/diffstruc_context.f90` | rewritten: chunked registry, storage allocation, release, checkpoint/rollback, scratch pool |
| `src/diffstruc/diffstruc_plumbing.f90` | rewritten: handle resolution, validation, assignment, value views, constants |
| `src/diffstruc/diffstruc_types_sub.f90` | rewritten: construction, forward mode, reverse mode, graph lifecycle, reductions, `exp`/`log` |
| `src/diffstruc/diffstruc_arithmetic_sub.f90` | new: arithmetic operations and their partial derivatives, ported onto handles |
| `src/diffstruc/diffstruc_operations_*.f90` | migrated: ownership statements removed, operands bound through `canonical()`, temporary constants created through `constant_node` |
| `src/diffstruc/diffstruc_global.f90` | traversal counter and map capacity restored |
| `src/diffstruc.f90` | re-exports the handle, context API and operations |
| `test/test_context.f90` | new: ownership, stale-handle and unbound-handle tests |
| `test/test_memory.f90`, `test/test_benchmark_*.f90` | migrated off ownership flags and per-handle deallocation |
| `scripts/build_lib.sh`, `scripts/run_tests.sh` | new: compiler-parameterised build and test drivers |

The legacy monolith and the intermediate snapshots used during the migration
are in the repository history (the parent commit and its predecessors); the
working tree keeps only the migrated sources.

## 3. Public API changes

| Before | After |
|---|---|
| `owns_left_operand`, `owns_right_operand`, `owns_gradient` components | removed |
| `fix_pointer` component | removed |
| `finalise_array` finaliser | removed |
| `x%shape = [...]` | `call x%set_shape([...])` |
| `allocated(x%val)` | `associated(x%val)` |
| `call x%deallocate()` releasing storage | unbinds the handle; context keeps the storage |
| `call x%nullify_graph()` freeing owned nodes | releases the expression's context records |
| `assign_and_deallocate_source(this, source, owns_left_operand, owns_right_operand)` | `assign_and_deallocate_source(this, source)` |
| `x%is_sample_dependent = .false.` | unchanged (view of the context flag) |
| `x%is_temporary = .false.` | unchanged (view of the context flag), after `allocate` |

Everything else — operators, `sum`, `mean`, `spread`, `unspread`, `exp`, `log`,
`sin`, `cos`, `tan`, `tanh`, `sqrt`, `abs`, `sign`, `sigmoid`, `gaussian`,
`log10`, `squared`, `cubed`, `matmul`, `outer_product`, `dot_product`,
`transpose`, `concat`, slices, `pack`, `unpack`, `reshape`, comparisons,
`merge`, `maxval`, `max`, differentiation entry points and the pointer-returning
constructors — keeps its original signature and numerical behaviour.

## 4. Verification commands

```sh
# library, GNU Fortran 15.2
sh scripts/build_lib.sh

# library, Flang 22.1.1 (uses the flang-built coreutils modules)
CC=flang-new MDIR=build/flang_new sh scripts/build_lib.sh

# tests, GNU Fortran: debug, optimised, runtime-checked
sh scripts/run_tests.sh
EXTRA="-O2" sh scripts/run_tests.sh
EXTRA="-O0 -fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow" \
  sh scripts/run_tests.sh

# tests, Flang
CC=flang-new sh scripts/run_tests.sh

# CMake build and CTest
cmake -S . -B build/cmake_gf -DCMAKE_Fortran_COMPILER=gfortran
cmake --build build/cmake_gf -j4
ctest --test-dir build/cmake_gf -E test_memory --output-on-failure
```

Flang needs module files produced by flang for its dependencies. Build them
once with:

```sh
cd ../coreutils
for pass in 1 2 3; do
  for f in src/coreutils/*.f90; do
    flang-new -cpp -std=f2018 -module-dir build/flang_mod \
      -I build/flang_mod -c -o build/flang_mod/$(basename $f .f90).o $f
  done
done
flang-new -cpp -std=f2018 -module-dir build/flang_mod -I build/flang_mod \
  -c -o build/flang_mod/coreutils.o src/coreutils.f90
ar rcs build/libcoreutils_flang.a build/flang_mod/*.o
```

## 5. Results

| Configuration | Library | `test_functions` | `test_context` | stale/guard rejection |
|---|---|---|---|---|
| GNU Fortran 15.2, `-std=f2018` | pass | pass, matches golden | pass | pass |
| GNU Fortran 15.2, `-O2` | pass | pass | pass | pass |
| GNU Fortran 15.2, `-fcheck=all -ffpe-trap=...` | pass | pass | pass | pass |
| Flang 22.1.1, `-std=f2018` | pass | pass, values match golden | pass | pass |
| GNU Fortran 15.2, `-DUSE_BLAS` + Accelerate BLAS | pass | pass, matches golden | pass | pass |
| CMake + CTest (GNU) | pass | pass | pass | pass |

Memory safety was checked with the platform leak detector over the full
`test_memory` run (allocation, differentiation, graph release, second-order
forward mode, 1000 iterations):

```
leaks -atExit -- ./test/test_memory
Process ...: 0 leaks for 0 total leaked bytes.
```

`test_memory` accepts `DIFFSTRUC_MEMORY_ITERATIONS` to change its iteration
count (default 1000). Peak resident memory grows with the iteration count
because the system allocator retains freed pages, but the leak detector
reports no leaked allocations.

`test_benchmark_reverse` reproduces the exact reference checksum for the
non-sample-dependent scenario (C: `0.262214E+05`) and the deeper tanh chain
(B: `0.728579E+05` against `0.728472E+05` before migration). Scenario A differs
because the pre-migration `get_partial_multiply_*_val` fast path computed the
weight gradient from the first row of the sample-dependent operand, which does
not match the forward expression that the same release evaluated. The migrated
engine differentiates exactly the expression that was evaluated.

`test_functions` reproduces every value of the pre-migration golden output
(`golden/functions.txt`): first and second derivatives of the packed quadratic,
the `x**4 + x**2*y` example, the `mean(x*tanh(x))` example, the residual/loss
example, the `tanh(x**2)` example and the `dot_product` example.

## 6. Source audit

The following searches over `src/` must (and do) return zero:

| Pattern | Occurrences |
|---|---|
| `owns_` | 0 |
| `fix_pointer` | 0 |
| `finalise_array` | 0 |
| `nullify_graph_recursive`, `dealloc_list` | 0 |

`deallocate(x%val)` appears exactly once, in the context's `release_node`, which
is the single place storage is returned to the runtime. Allocation of
`indices`/`adj_ja`/`mask`/`direction` remains because those are small node
metadata buffers owned by the context registry record.

## 7. Performance

`bench/compare.sh` builds the development branch (`git HEAD`), this tree with
the pre-optimisation reverse engine (`-DDIFFSTRUC_SLOW_REVERSE`) and this tree
as it stands, then runs each variant's shipped benchmarks with identical
compiler flags. See `bench/README.md` for the methodology.

| Variant | reverse benchmark, whole program (s) |
|---|---|
| development branch | 0.76 |
| pre-optimisation engine | 1.61 |
| optimised (this tree, accurate) | 0.36 |
| optimised with fast `tanh` | 0.28 |

- 4.47x faster than the implementation delivered by the migration.
- 2.11x faster than the development branch end to end with bit-identical
  results: 2.46x (A), 1.84x (B) and 2.34x (C) per chain. The opt-in fast `tanh`
  kernel (`tanhf`, or `-DDIFFSTRUC_FAST_TANH`) raises this to 2.71x end to end
  (2.96x on chain B) with a maximum absolute error of 8.5e-5.
- Chain B is now bounded by work common to both implementations: libm `tanhf`
  over 65 536 elements (~170 us) and the serial forward reduction (~60 us)
  account for about half of its 0.263 ms. Reducing either requires changing
  the results (the opt-in `tanhf`, or a reassociated reduction), which is why
  chain B stays at 1.84x while A and C are at 2.3-2.5x.
- Accuracy: the reverse-benchmark reference checksums are identical to the
  development branch (`A=0.578015E+04`, `B=0.728472E+05`, `C=0.262214E+05`),
  and the example suite is bit-identical to the development branch's output
  (51 values, maximum difference 0.0).

The optimisations that produced this result:

1. Assignment to an unbound handle binds instead of deep-copying.
2. Reverse mode evaluates `get_partial_*_val` / `_val_sum` procedures directly
   into pooled scratch buffers instead of building a derivative node per edge.
3. Only graph variables accumulate gradients; intermediate accumulators are
   not allocated.
4. A per-pass phase tag makes the first contribution to an accumulator
   overwrite it, so a pass never zeroes or pre-reads its gradient arrays.
5. A scalar loss propagates its uniform upstream value, so the partial of
   `sum`/`mean` is consumed as a scalar and a fused `unary -> add -> multiply`
   kernel computes the unary partial, the full multiply partial and both
   reduced gradients in a single pass (the intermediate partial is never
   materialised). Element expressions and summation order are unchanged.
6. Node allocation uses the cached registry tail and handle lookup indexes the
   chunk-pointer array arithmetically, so both are O(1) instead of growing with
   the number of records ever created.
7. First contributions to leaf accumulators are written straight into the
   accumulator (no buffer, no extra add pass).
8. The `add` operator hands the upstream gradient to its operands directly.
9. Context regions are swept from the checkpoint rather than from the start of
   the registry, and node records are addressed through a cached `self`
   pointer instead of a registry search.
10. Value buffers are recycled by the context instead of being allocated and
   freed per node.
11. Value storage moved from a pointer component into the record's allocatable
   `store`; the public `%val` view points at it, and the hot kernels operate on
   the allocatable storage, which the compiler can prove cannot alias. This is
   worth about 2x per elementwise kernel.
12. `!$omp simd` on the elementwise kernels plus explicit loops in place of
    array syntax; GNU Fortran builds enable `-fopenmp-simd`,
    `-fversion-loops-for-strides` and `-fvect-cost-model=unlimited`, which
    lets the loops that walk pointer-backed storage vectorise at `-O2`.
13. `nullify_graph` decides whether to release the root record *before*
    releasing its operands. It previously made the same decision afterwards,
    when the root looked like a variable, so expression roots were never
    released and every iteration allocated and faulted in a fresh buffer.
14. Value buffers are recycled with `move_alloc` (no allocation or zero-fill
    per node), and the gradient-invalidation and region-sweep walks are skipped
    when a backward pass creates no transient graph nodes.

## 8. Known deviations

- The legacy elementwise product broadcast one operand's *first element* when
  the other operand was sample-dependent and the first was a non-scalar
  non-sample-dependent array (`a%val(1,1) * b%val(i,s)`). That behaviour is kept
  unchanged to preserve the existing numerical results; it is a property of the
  original `multiply_arrays` implementation, not of the ownership model.
- Reverse-mode differentiation builds partials through the same graph-building
  code path as forward mode, then releases every transient record it created
  with a mark-and-sweep over the context region. This is slower than the old
  hand-written `*_val` fast path but is correct for every operation, including
  the gather/scatter family and matrix multiplication, and it keeps repeated
  passes in bounded live memory.
- The matmul benchmark checksum differs in the fourth significant digit from
  the old build because the old `*_val` partials used a different (and in places
  inconsistent) index convention.
- `test_memory` now relies on `assign_and_deallocate_source` and `nullify_graph`
  to release iteration temporaries instead of `deallocate(temp)`. Its summary
  output therefore no longer prints ownership flags (they no longer exist).

## Post-migration follow-ups

The migration left a few traps that are now removed or documented:

- `allocate_array` sets the logical `rank` from the leading dimensions and
  accepts `sample_dependent=` / `temporary=` keyword flags, so the flags never
  have to be assigned before a handle is bound (that used to dereference a null
  flag view). `source=` also accepts any value with the same total element
  count, e.g. `allocate(array_shape=[rows, cols, 1], source=matrix)`.
- `set_shape` updates the logical rank as well as the shape.
- `assign_and_deallocate_source` no longer evaluates `dst%id` when `dst` is
  unassociated (Fortran does not short-circuit `.and.`), which crashed every
  custom operation that used it on a fresh result handle.
- `test_benchmark_matmul` allocates its weight matrix as `[out_d, in_d, 1]`
  instead of `[out_d*in_d, 1]` followed by `set_shape`.
- `fpm.toml` declares `[install] library = true`, so `fpm install` installs the
  archive and module files instead of doing nothing.
- Documentation: new `docs/source/porting.rst`; the memory-management,
  operations, installation and API pages now describe context ownership, the
  allocation convention, the removed ownership members, and the `tanhf` /
  `-DDIFFSTRUC_FAST_TANH` option. `test/test_porting.f90` checks the contract.

Verification after these changes: gfortran 15.2 and flang 22.1.1 suites pass,
`-O0 -fcheck=all -ffpe-trap` passes, CTest 7/7, `test_memory` reports 0 leaked
bytes, the example output is byte-identical to the golden file, and the reverse
benchmark remains 2.11x faster than the development branch with identical
checksums. `fpm build`, `fpm test` and `fpm install` were re-run from a clean
copy in the `raffle_env` environment.


## Thread safety

Every mutable module variable has been moved into the context (registry,
pools, traversal counter, pass tag, recursion limit) and `default_context` is
`!$omp threadprivate`; `diffstruc_global` now contains only parameter
constants. `new_context`/`destroy_context` and per-context
`set_num_threads`/`set_max_recursion_depth` are public, and the OpenMP kernels
(thresholded) plus the multi-context tests are in place.

The full suite and the golden output are bit-identical at
`OMP_NUM_THREADS=1/2/4/8`, and `test_threads` reproduces the serial checksum
from every thread with both thread-private default contexts and explicit
contexts. CTest 8/8, flang, `-fcheck=all` and `leaks` are clean.

Cost: the first version of the refactor put the reverse benchmark at 1.80x
because the fused `unary -> add -> multiply` kernel stopped vectorising once it
became a contained procedure of the enlarged context-aware host (its loops were
no longer reported by the vectoriser and the object had no NEON instructions).
Moving the kernel to a module procedure with plain array arguments restored
the 16-byte SIMD loops: chains A/B went from 0.161/0.318 ms back to
0.124/0.285 ms and the benchmark is **2.08x** faster than the development
branch (pre-refactor: 2.11x, within the run-to-run spread; dev itself varied
0.76-0.81 s under machine load). All kernels compile to the same code as
before the refactor, checksums are identical, and the thread-safety guarantees
are unchanged.
