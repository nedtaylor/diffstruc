#!/bin/sh
# Thread-scaling benchmark for the OpenMP kernels.
#
#   sh bench/threads.sh            # gfortran, sizes in test_benchmark_threads
#   THREADS="1 2 4 8" sh bench/threads.sh
#
# Builds the library and the thread benchmark with -fopenmp, then runs the
# benchmark once per thread count. Checksums must be identical across runs;
# the machine should be idle for the timings to be meaningful.
set -e
cd "$(dirname "$0")/.."
CC=${CC:-gfortran}
THREADS=${THREADS:-"1 2 4 8"}
OUT=build/threads
rm -rf "$OUT"
mkdir -p "$OUT"
COREUTILS_MOD=/Users/nedtaylor/DCoding/DGit/coreutils/build/mod
COREUTILS_LIB=/Users/nedtaylor/DCoding/DGit/coreutils/build/libcoreutils.a
EXTRA=${EXTRA:-}
FLAGS="-cpp -std=f2018 -O2 -fopenmp -fopenmp-simd -fversion-loops-for-strides -fvect-cost-model=unlimited $EXTRA"

SRC=src/diffstruc
for name in diffstruc_global diffstruc_parallel diffstruc_kernels \
            diffstruc_types \
            diffstruc_context diffstruc_plumbing diffstruc_types_sub \
            diffstruc_arithmetic_sub diffstruc_operations_maths \
            diffstruc_operations_maths_sub diffstruc_operations_trig \
            diffstruc_operations_hyp diffstruc_operations_linalg \
            diffstruc_operations_linalg_sub diffstruc_operations_broadcast \
            diffstruc_operations_broadcast_sub diffstruc_operations_comparison \
            diffstruc_operations_reduction; do
  $CC $FLAGS -I "$COREUTILS_MOD" -J "$OUT" -c -o "$OUT/$name.o" "$SRC/$name.f90"
done
$CC $FLAGS -I "$COREUTILS_MOD" -J "$OUT" -c -o "$OUT/diffstruc.o" src/diffstruc.f90
$CC $FLAGS -I "$COREUTILS_MOD" -J "$OUT" -c -o "$OUT/bench.o" test/test_benchmark_threads.f90
$CC -fopenmp -O2 -o "$OUT/bench" "$OUT"/*.o "$COREUTILS_LIB"

echo "compiler: $($CC --version | head -1)"
for t in $THREADS; do
  echo
  echo "=== OMP_NUM_THREADS=$t ==="
  OMP_NUM_THREADS=$t "$OUT/bench" | tee "$OUT/run_$t.txt"
done

first=$(echo $THREADS | cut -d' ' -f1)
echo
echo "=== speed-up relative to OMP_NUM_THREADS=$first ==="
printf "%-8s %-6s %-6s %10s %8s\n" dim chain threads "ms" "speedup"
for t in $THREADS; do
  awk -v t="$t" '
    FNR == NR { if ($4 == "A" || $4 == "B") base[$1 " " $2 " " $4] = $5; next }
    ($4 == "A" || $4 == "B") {
       key = $1 " " $2 " " $4
       printf "%-8s %-6s %-6s %10.3f %7.2fx\n", $1, $4, t, $5, base[key] / $5
    }
  ' "$OUT/run_$first.txt" "$OUT/run_$t.txt"
done
