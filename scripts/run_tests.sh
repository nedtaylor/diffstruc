#!/bin/sh
# Builds the library and every test with one compiler and runs them.
#
#   sh scripts/run_tests.sh                # gfortran, debug checks
#   CC=flang-new sh scripts/run_tests.sh   # flang
#   EXTRA="-O2 -fcheck=all" sh scripts/run_tests.sh
set -e
cd "$(dirname "$0")/.."
CC=${CC:-gfortran}
EXTRA=${EXTRA:-}
if [ "$CC" = "flang-new" ] || [ "$CC" = "flang" ]; then
  COREUTILS_MOD=../coreutils/build/flang_mod
  MODFLAG="-module-dir"
else
  COREUTILS_MOD=../coreutils/build/mod
  MODFLAG="-J"
fi
LIBDIR=build/testlib
BINDIR=build/testbin
rm -rf "$LIBDIR" "$BINDIR"
mkdir -p "$LIBDIR" "$BINDIR"

sh scripts/build_lib.sh >/dev/null

SIMDFLAGS=""
if [ "$CC" != "flang-new" ] && [ "$CC" != "flang" ]; then
  SIMDFLAGS="-fopenmp-simd -fversion-loops-for-strides -fvect-cost-model=unlimited"
fi
LINKFLAGS=""
if [ -n "$OPENMP" ]; then
  SIMDFLAGS="$SIMDFLAGS -fopenmp"
  LINKFLAGS="-fopenmp"
fi
FLAGS="-cpp -std=f2018 -g $SIMDFLAGS -I $COREUTILS_MOD -I $LIBDIR $EXTRA"
MOD="$MODFLAG $LIBDIR"

# Rebuild the library objects into the test module dir so tests see the same
# module files.
for f in src/diffstruc/diffstruc_global.f90 src/diffstruc/diffstruc_parallel.f90 \
         src/diffstruc/diffstruc_kernels.f90 src/diffstruc/diffstruc_types.f90 \
         src/diffstruc/diffstruc_context.f90 src/diffstruc/diffstruc_plumbing.f90 \
         src/diffstruc/diffstruc_types_sub.f90 src/diffstruc/diffstruc_arithmetic_sub.f90 \
         src/diffstruc/diffstruc_operations_maths.f90 \
         src/diffstruc/diffstruc_operations_maths_sub.f90 \
         src/diffstruc/diffstruc_operations_trig.f90 \
         src/diffstruc/diffstruc_operations_hyp.f90 \
         src/diffstruc/diffstruc_operations_linalg.f90 \
         src/diffstruc/diffstruc_operations_linalg_sub.f90 \
         src/diffstruc/diffstruc_operations_broadcast.f90 \
         src/diffstruc/diffstruc_operations_broadcast_sub.f90 \
         src/diffstruc/diffstruc_operations_comparison.f90 \
         src/diffstruc/diffstruc_operations_reduction.f90 src/diffstruc.f90; do
  name=$(basename "$f" .f90)
  $CC $FLAGS $MOD -c -o "$LIBDIR/$name.o" "$f"
done

LIBS="$LIBDIR"/*.o
if [ "$CC" = "flang-new" ] || [ "$CC" = "flang" ]; then
  COREUTILS_LIB=../coreutils/build/libcoreutils_flang.a
else
  COREUTILS_LIB=../coreutils/build/libcoreutils.a
fi

for t in test_functions test_context test_memory test_tanh test_porting \
         test_threads test_benchmark_matmul test_benchmark_reverse \
         test_benchmark_threads; do
  $CC $FLAGS $MOD -c -o "$BINDIR/$t.o" "test/$t.f90"
  $CC $LINKFLAGS -o "$BINDIR/$t" "$BINDIR/$t.o" $LIBS $COREUTILS_LIB
  echo "built $t"
done

echo "=== test_functions ==="
"$BINDIR/test_functions"
echo "=== test_context ==="
"$BINDIR/test_context"
echo "=== test_tanh ==="
"$BINDIR/test_tanh"
echo "=== test_porting ==="
"$BINDIR/test_porting"
echo "=== test_threads ==="
"$BINDIR/test_threads"
echo "=== test_context stale (expected to fail) ==="
if "$BINDIR/test_context" stale; then
  echo "ERROR: stale handle was accepted"
  exit 1
else
  echo "rejected as expected"
fi
echo "=== test_context guard (expected to fail) ==="
if "$BINDIR/test_context" guard; then
  echo "ERROR: unbound handle was accepted"
  exit 1
else
  echo "rejected as expected"
fi
