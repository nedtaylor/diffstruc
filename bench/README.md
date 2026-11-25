# Speed comparison

`bench/compare.sh` builds and runs three variants of the library on the same
machine, with the same compiler and flags, and reports timings, speed-up
ratios, checksums and accuracy:

| Variant | What it is |
|---|---|
| `dev` | the development branch at `git HEAD`, i.e. the owning-pointer `array_type` |
| `reference` | this tree compiled with `-DDIFFSTRUC_SLOW_REVERSE`, which forces the pre-optimisation reverse engine (one derivative node per partial, zeroing sweep per pass) |
| `optimised` | this tree as it stands |

Each variant is built from its own tree with `-O2` (`-std=f2018` for this
tree, `-std=gnu` for the development branch, which uses the `LOC` extension).

## Result (GNU Fortran 15.2, Apple M-series)

```
=== test_benchmark_reverse: best of 3 whole-program runs (seconds) ===
variant            chainA     chainB     chainC
dev (HEAD)           0.76          -          -
reference            1.61          -          -
optimised            0.36          -          -
opt+fast-tanh        0.28          -          -
speedup vs dev (accurate):    2.11x
speedup vs dev (fast tanh):   2.71x
speedup vs ref (accurate):    4.47x

=== per-chain times reported by the program (ms, last size) ===
variant        chainA     chainB     chainC
dev             0.275      0.483      0.227
optimised       0.112      0.263      0.097
fast tanh       0.109      0.163      0.098
                2.46x      1.84x      2.34x   (accurate build)

=== accuracy: reverse-benchmark reference checksums ===
dev        A=0.578015E+04 B=0.728472E+05 C=0.262214E+05
optimised  A=0.578015E+04 B=0.728472E+05 C=0.262214E+05

=== accuracy: example suite, optimised vs dev branch ===
values compared: 51, max abs diff 0.000e+00, max rel diff 0.000e+00
```

- The optimised build reproduces every printed value of the development
  branch bit for bit, and the reverse-benchmark checksums are identical.
- Against the implementation delivered by the ownership migration (the
  `reference` build) the engine is **4.47x faster** end to end.
- Against the development branch: **2.11x** end to end with bit-identical
  results, and **2.71x** with the opt-in fast `tanh` kernel
  (`-DDIFFSTRUC_FAST_TANH`, accuracy 8.5e-5 absolute). Per chain: 2.46x (A),
  1.84x (B, 2.96x with the fast kernel) and 2.34x (C).
- Every benchmark size is ahead of the development branch, from 1.6x at
  32x512 (where fixed per-node cost dominates) to 2.9x at 128x256.

## Where the remaining time is

Chain B (`loss = sum(tanh(W2*(W1*x+b1)+b2))`) is still dominated by work that
is identical in both implementations:

| component | cost at 2048x32 | can it be reduced? |
|---|---|---|
| libm `tanhf` over 65 536 elements | ~170 us | only by a different approximation (the opt-in `tanhf`) |
| the serial forward reduction `sum(...)` | ~60 us | only by reassociating it, which changes every checksum |
| forward elementwise passes | ~10-15 us each | at the memory-bandwidth limit |

Those two fixed costs are about half of chain B (0.263 ms in the optimised
build), which caps the ratio on that chain at roughly 1.85x. Chains A and C,
which have no `tanh`, are at 2.3-2.5x. The opt-in fast `tanh` lifts chain B to
2.96x and the whole program to 2.71x.

### The measured floor

At the largest benchmark size (2048x32, 256 KiB per array) the machine's cost
for one elementwise pass and one reduction, measured with plain local arrays,
is:

```
elementwise pass (256 KiB)   21.43 us
reduction (256 KiB)          57.37 us
```

Every chain is a fixed sequence of such passes, because the API materialises
each intermediate node as it is built:

| chain | required passes | floor | measured (optimised) |
|---|---|---|---|
| A (`sum((W*x+b)^2)`) | 3 forward + ~5 backward + 1 reduction | ~214 us | 183 us |
| B (`sum(tanh(...))`) | ~11 passes + 1 reduction + libm tanh (~170 us) | ~400 us | 381 us |
| C (`sum((a1+a2)^2)`) | 3 forward + ~3 backward + 1 reduction | ~183 us | 121 us |

The optimised build is at (and for A and C below) that floor, because several
passes run on smaller gradient buffers. The development branch performs the
same passes and lands at 1.5x that floor. A 2x whole-program ratio therefore
needs *fewer passes*, i.e. fusing the elementwise chain into single kernels,
not further tuning of each pass.

The opt-in fast `tanh` (the public `tanhf` operation, or
`-DDIFFSTRUC_FAST_TANH` for `tanh` itself) uses the [7/6] Pade approximation
adopted by Eigen and several inference runtimes. It is about ten times faster
than libm `tanhf` on the kernel and takes chain B from 0.394 ms to 0.299 ms
(1.59x on that chain, 1.63x end to end), with a maximum absolute error of
8.5e-5 instead of bit-identical results. Reassociated SIMD reductions would
give a little more, but change every checksum; they are not provided.

## What was changed to get here

Measured on `test_benchmark_reverse` chain A at the largest size, starting
from the migrated implementation as delivered by the ownership migration:

| Step | chain A (ms) |
|---|---|
| migrated implementation (graph partial per edge) | 1.61 |
| bind instead of copy on unbound assignment; no value zeroing | 0.95 |
| `_val` / `_val_sum` partials with pooled buffers | 0.60 |
| accumulate only into graph variables | 0.41 |
| per-pass overwrite of gradients (no zeroing sweep) | 0.37 |
| direct write into leaf accumulators, region-limited sweeps | 0.31 |
| `add` operator passes the upstream gradient straight through | 0.28 |
| SIMD elementwise kernels (`!$omp simd`, explicit loops) | 0.20 |
| alias-free allocatable `store` for all hot kernels | 0.18 |
| root nodes now actually released by `nullify_graph` (buffer recycling) | 0.18 |
| no value-buffer zeroing on allocation | 0.18 |
| O(1) node allocation and id lookup (chunk index, tail pointer) | 0.17 |
| fused uniform `unary -> add -> multiply` backward kernel | 0.11 |

The last two rows fixed a real bug: `nullify_graph` decided whether to release
the root *after* nullifying its operands, so the root always looked like a
variable and was never released. Every iteration then allocated a fresh
256 KiB buffer, with page faults costing more than the kernel. `x + y` on
65 536 elements fell from 196 us to 43 us (kernel 22 us), and `x + x` from
287 us to 43 us.

The last two rows remove the remaining structural overhead:

- The registry used to find its last chunk (and the chunk holding an id) by
  walking the linked list, so every allocation and every handle lookup grew
  with the number of records ever created. Both are now direct: the allocator
  uses the cached tail, and `find_node` indexes a chunk-pointer array
  arithmetically from the id. The cost of a benchmark repetition no longer
  depends on how many repetitions preceded it.
- A scalar loss makes the upstream gradient of a unary operation uniform. When
  that unary operation sits on `w*d + other` (broadcast `w`, sample-dependent
  `d`), the unary partial, the full partial of `d`, and the two reduced
  gradients are now computed in one pass instead of materialising the unary
  partial and reading it back three times. The per-element expressions, the
  summation order and the reduction layout are exactly those of the separate
  kernels, so the checksums and printed values do not change.

The remaining time is the numeric work itself, which is the same in both
implementations.

## Build flags

GNU Fortran builds of this tree add `-fopenmp-simd
-fversion-loops-for-strides -fvect-cost-model=unlimited` to the caller's flags
(set in `scripts/build_lib.sh`, `bench/build_variant.sh`, `scripts/run_tests.sh`
and `CMakeLists.txt`). Without them the `!$omp simd` directives are comments
and the loops that walk pointer-backed storage are not vectorised; with them
each elementwise pass runs at roughly the `-O3` rate while every scalar result
is unchanged.

The fused backward kernel can be switched off with `-DDIFFSTRUC_NO_FUSE`
to A/B it against the separate kernels. Building both ways and hashing every
gradient array bit-for-bit (squared/tanh/exp/negate over `w*d + other`, both
addend orders, with and without the addend, several batch widths) produces
identical hashes, so the fusion only removes memory traffic.

## Running it

```sh
sh bench/compare.sh                     # gfortran, -O2
CC=flang-new sh bench/compare.sh        # flang (needs the flang coreutils build)
```

The script creates `build/dev_src` (a git worktree of `HEAD`) the first time it
runs. All build products live under `build/`.
