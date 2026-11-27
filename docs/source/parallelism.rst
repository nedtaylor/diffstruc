Parallelism and thread safety
=============================

The library is safe to use from one thread, from several OpenMP threads, and
with several independent graphs running at the same time. It contains no
mutable module variables: every registry, pool, counter, pass tag and setting
lives in a context (`array_context_type`). The only module variable in the
whole library is `default_context`, which is `threadprivate` (see below).

The three supported usage patterns
----------------------------------

**1. Single thread (the default).** Build and use the library exactly as
before. No OpenMP flags are needed; every `!$omp` directive is a comment and
every kernel is serial (with the SIMD vectorisation it has always had). The
numeric results are unchanged.

**2. OpenMP inside one graph.** Build with OpenMP and each kernel may split its
work across a team. The split is always over independent elements or over
independent accumulators, never over the sample-reduction order, so the results
are bit-for-bit the same as the serial build:

.. code-block:: bash

   fpm build --profile release --flag "-fopenmp"
   # or: OPENMP=1 sh scripts/build_lib.sh

   cmake -S . -B build -DUSE_OPENMP=ON

Per-context control:

.. code-block:: fortran

   call new_context(ctx, n_threads=8)      ! explicit context with 8 threads
   call set_num_threads(ctx, 4)            ! 0 = runtime default, 1 = serial
   print *, get_num_threads(ctx)

   x%ctx = ctx                             ! bind a handle to that context
   call x%allocate(array_shape=[d, batch], sample_dependent=.true.)

A kernel only asks for a team when its work (elements x samples) clears the
serial floor, and the team is sized by the work rather than by the machine:
`nth = clamp(work / diffstruc__parallel_elems_per_thread, 1, available)`
(262 144 elements per thread). A 1 MiB node therefore stays serial, 4 MiB uses
four threads and 8 MiB uses eight. `set_num_threads(ctx, 1)` forces a context
to be serial even in an OpenMP build, and a kernel entered from inside an
existing team runs serial (`in_parallel`).

The elementwise work is done by `diffstruc__kernels`: module procedures with
plain array arguments and explicit `intent`. This is deliberate -- the pointer
components of `array_type` prevent the compiler from proving that the output
does not alias the inputs, so loops written directly against them lose SIMD
once they are outlined for OpenMP. Kernels take `%store` views, keep their
per-element expressions identical to the serial code, and are wrapped as
`!$omp parallel do if(nth > 1)` over samples with `!$omp simd` inside.

**3. Several independent graphs on several threads.** Every thread must use its
own context. Two ways:

.. code-block:: fortran

   ! a) rely on the thread-private default (allocated on first use)
   !$omp parallel
      call run_one_model()          ! uses this thread's default context

   ! b) create and destroy one explicitly
   !$omp parallel
      type(array_context_type) :: ctx
      call new_context(ctx)
      ... allocate, differentiate, read gradients ...
      call destroy_context(ctx)

Never pass a context (or a handle belonging to it) between threads, and never
mutate a shared `array_type` from two threads. Handles are cheap views; give
each thread its own roots, or copy values into each thread's own arrays.

Why this is safe
----------------

What used to be module `save` state is now per context:

| Former global | Now |
|---|---|
| `diffstruc__visit_counter` | `context_state%visit_counter` |
| `diffstruc__max_recursion_depth` | `context_state%max_recursion_depth` (settable per context) |
| forward/reverse pass tag | `context_state%accumulate_phase` |
| transient-node flag | `context_state%scratch_nodes_created` |
| registry and pools | `context_state` (one heap allocation per context) |
| default context | `default_context`, **threadprivate**, allocated on first use |

`default_context` is the single ambient exception: a default has to come from
somewhere. Each thread gets its own copy (OpenMP `threadprivate`), so two
threads that both rely on the default still own separate registries and pools.
Without OpenMP the directive is ignored and there is one default, which is what
a serial program wants.

Consequences to keep in mind:

* Allocators (`allocate_node`, `take_value`, `acquire_scratch`) and graph
  bookkeeping (`mark_graph`, `sweep_region`) are never called inside a parallel
  region. Kernels only touch disjoint elements of already-allocated arrays.
* Gradient accumulation is parallelised over the element index, so each
  accumulator element is written by exactly one thread; the reduction over
  samples keeps its serial order.
* `sum`/`mean` reductions over all elements keep their strict order and stay
  serial: a parallel reduction would reassociate the additions and change every
  result. Parallelise around them, not through them.
* `matmul` keeps using BLAS; the library does not add a team around it (BLAS
  has its own thread control).
* Splitting a kernel over samples means a single-sample array with more
  elements than the threshold does not benefit; use several graphs on threads
  for that shape.

Measured character (Apple M2 Pro)
---------------------------------

Per-kernel ceilings measured with the same loop shapes:

| Work | 1 thread | 8 threads |
|---|---|---|
| `tanh` over 1 MiB | 2.5 ms | 0.38 ms (6.6x) |
| `tanh` over 64 MiB | 76 ms | 11 ms (6.9x) |
| elementwise `a+a`, 64 MiB | 2.8 ms | 0.86 ms (3.3x, memory-limited) |
| elementwise `a+a`, 256 KiB | 6 us | 46 us (region overhead dominates) |

`bench/threads.sh` measures chain A and chain B with `system_clock` (wall
clock; `cpu_time` returns process CPU time, the sum over threads) for
`OMP_NUM_THREADS=1,2,4,8`, prints the checksums, and reports the speed-up
against the first thread count:

.. code-block:: bash

   sh bench/threads.sh
   THREADS="1 4 8" sh bench/threads.sh

Measured at 2048x2048 (4.2 M) and 2048x4096 (8.4 M), best of three, GNU
Fortran 15.2, M2 Pro (8P+4E):

| elements | chain | 1 thread | 4 threads | 8 threads |
|---|---|---|---|---|
| 4.2 M | A | 8.25 ms | 7.36 (1.12x) | 7.04 (1.17x) |
| 4.2 M | B | 16.44 ms | 9.92 (1.66x) | 8.44 (1.95x) |
| 8.4 M | A | 15.49 ms | 13.32 (1.16x) | 13.14 (1.18x) |
| 8.4 M | B | 32.86 ms | 18.90 (1.74x) | 17.20 (1.91x) |

The whole run is 1.47x at four threads and 1.59x at eight. Chain A is capped
by the ordered `sum`: it stays serial to keep the results bit-identical and is
about 45% of that chain at these sizes. Elementwise kernels are bandwidth-bound
at 8 MiB and scale about 2x; `tanh` scales up to 4.8x. Run the sweep on an idle
machine -- under load the parallel and serial runs both degrade.

Opt-in reassociated reduction
-----------------------------

Compiling with `-DDIFFSTRUC_FAST_REDUCTION` replaces the ordered `sum`/`mean`
reduction with a deterministic two-level one: each sample column is summed in
parallel by a vectorised kernel, then the column sums are added serially. This
removes the serial bottleneck at the cost of a last-bit change in the result
(the gradients for `sum` are unchanged because the upstream value is one).
Measured with the flag at 8 threads: 2.24x on chain A (4.2 M), 1.95x (8.4 M),
and 3.1x on chain B (4.2 M). It is opt-in so the default stays bit-identical.

Future work
-----------

The kernels are written as pure elementwise loops over plain array arguments, so
they can be expressed with `do concurrent` (with `local`/`reduce` locality
specs) instead of OpenMP in a future revision. The threshold and per-context
thread count would then select `do concurrent` versus a serial loop in the same
way.
