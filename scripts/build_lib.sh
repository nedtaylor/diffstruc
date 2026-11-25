#!/bin/sh
# Full incremental build of the library for the migration.
# Module files are compiler-specific, so the module directory is wiped on
# every run: a stale .mod produces phantom "not a member" errors.
set -e
cd "$(dirname "$0")/.."
CC=${CC:-gfortran}
M=${MDIR:-build/lib}
rm -rf "$M"
mkdir -p "$M"
if [ "$CC" = "flang-new" ] || [ "$CC" = "flang" ]; then
  COREUTILS_MOD=../coreutils/build/flang_mod
else
  COREUTILS_MOD=../coreutils/build/mod
fi
FLAGS="-cpp -std=f2018 -I $COREUTILS_MOD"
# The elementwise kernels carry !$omp simd directives; honour them without
# enabling threading.
if [ "$CC" != "flang-new" ] && [ "$CC" != "flang" ]; then
  FLAGS="$FLAGS -fopenmp-simd"
  # Version loops on their runtime stride so unit-stride copies vectorise.
  FLAGS="$FLAGS -fversion-loops-for-strides -fvect-cost-model=unlimited"
fi
if [ -n "$USE_BLAS" ]; then
  FLAGS="$FLAGS -DUSE_BLAS"
fi
if [ "$CC" = "flang-new" ] || [ "$CC" = "flang" ]; then
  MOD="-module-dir $M"
else
  MOD="-J $M"
fi
SRC=src/diffstruc
build() {
  $CC $FLAGS -c -o "$M/$2.o" $MOD -I "$M" "$1"
}
build $SRC/diffstruc_global.f90 global
build $SRC/diffstruc_types.f90 types
build $SRC/diffstruc_context.f90 context
build $SRC/diffstruc_plumbing.f90 plumbing
build $SRC/diffstruc_types_sub.f90 types_sub
build $SRC/diffstruc_arithmetic_sub.f90 arithmetic
build $SRC/diffstruc_operations_maths.f90 operations_maths
build $SRC/diffstruc_operations_maths_sub.f90 operations_maths_sub
build $SRC/diffstruc_operations_trig.f90 operations_trig
build $SRC/diffstruc_operations_hyp.f90 operations_hyp
build $SRC/diffstruc_operations_linalg.f90 operations_linalg
build $SRC/diffstruc_operations_linalg_sub.f90 operations_linalg_sub
build $SRC/diffstruc_operations_broadcast.f90 operations_broadcast
build $SRC/diffstruc_operations_broadcast_sub.f90 operations_broadcast_sub
build $SRC/diffstruc_operations_comparison.f90 operations_comparison
build $SRC/diffstruc_operations_reduction.f90 operations_reduction
build src/diffstruc.f90 diffstruc
echo LIB_OK
