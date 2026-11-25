#!/bin/sh
# Builds a library tree (legacy or migrated) with one compiler, optimised,
# and its benchmark programs, for speed comparisons.
#
#   sh bench/build_variant.sh /tmp/ds_legacy build/legacy
#   sh bench/build_variant.sh . build/current
set -e
ROOT=$1
OUT=$2
CC=${CC:-gfortran}
FLAGS=${FLAGS:--O2}
MOD=$OUT/mod
OBJ=$OUT/obj
BIN=$OUT/bin
rm -rf "$OUT"
mkdir -p "$MOD" "$OBJ" "$BIN"

if [ "$CC" = "flang-new" ] || [ "$CC" = "flang" ]; then
  COREUTILS_MOD=/Users/nedtaylor/DCoding/DGit/coreutils/build/flang_mod
  COREUTILS_LIB=/Users/nedtaylor/DCoding/DGit/coreutils/build/libcoreutils_flang.a
  MODFLAG="-module-dir"
else
  COREUTILS_MOD=/Users/nedtaylor/DCoding/DGit/coreutils/build/mod
  COREUTILS_LIB=/Users/nedtaylor/DCoding/DGit/coreutils/build/libcoreutils.a
  MODFLAG="-J"
  FLAGS="$FLAGS -fopenmp-simd"
  # Version loops on their runtime stride so unit-stride copies vectorise.
  FLAGS="$FLAGS -fversion-loops-for-strides -fvect-cost-model=unlimited"
fi

cd "$ROOT"
SRC=src/diffstruc
ORDER="diffstruc_global diffstruc_parallel diffstruc_types diffstruc_context diffstruc_plumbing \
diffstruc_types_sub diffstruc_arithmetic_sub \
diffstruc_operations_maths diffstruc_operations_maths_sub \
diffstruc_operations_trig diffstruc_operations_hyp \
diffstruc_operations_linalg diffstruc_operations_linalg_sub \
diffstruc_operations_broadcast diffstruc_operations_broadcast_sub \
diffstruc_operations_comparison diffstruc_operations_reduction"

for name in $ORDER; do
  if [ -f "$SRC/$name.f90" ]; then
    $CC -cpp -std=f2018 $FLAGS -I "$COREUTILS_MOD" $MODFLAG "$MOD" \
      -c -o "$OBJ/$name.o" "$SRC/$name.f90"
  fi
done
$CC -cpp -std=f2018 $FLAGS -I "$COREUTILS_MOD" $MODFLAG "$MOD" \
  -c -o "$OBJ/diffstruc.o" src/diffstruc.f90

for t in test_benchmark_reverse test_benchmark_matmul test_functions; do
  $CC -cpp -std=f2018 $FLAGS -I "$COREUTILS_MOD" $MODFLAG "$MOD" \
    -c -o "$OBJ/$t.o" "test/$t.f90"
  $CC $FLAGS -o "$BIN/$t" "$OBJ/$t.o" "$OBJ"/diffstruc*.o "$COREUTILS_LIB"
done
echo "built $OUT"
