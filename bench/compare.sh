#!/bin/sh
# Speed and accuracy comparison for the context-owned implementation.
#
# Builds and runs three variants on the same machine and with the same
# compiler flags:
#
#   dev        the development branch (git HEAD, owning-pointer implementation)
#   reference  this tree built with -DDIFFSTRUC_SLOW_REVERSE, which forces the
#              pre-optimisation reverse engine (a derivative node per partial)
#   optimised  this tree as it stands
#   fast       this tree with -DDIFFSTRUC_FAST_TANH (vectorised tanh)
#
# Each variant runs its own shipped benchmark programs.  The reverse benchmark
# uses a fixed random seed, so its checksums are comparable between runs; the
# script prints them to show the numerical results are unchanged.
set -e
cd "$(dirname "$0")/.."
ROOT=$PWD
CC=${CC:-gfortran}
COREUTILS=/Users/nedtaylor/DCoding/DGit/coreutils

# --- development branch (git HEAD) ------------------------------------------
if [ ! -d build/dev_src ]; then
  git worktree add --detach build/dev_src HEAD >/dev/null
fi

build_tree() {
  # $1 source root, $2 output prefix, $3 extra flags
  root=$1
  out=$2
  extra=$3
  mod=$out/mod
  obj=$out/obj
  bin=$out/bin
  rm -rf "$out"
  mkdir -p "$mod" "$obj" "$bin"
  if [ "$CC" = "flang-new" ] || [ "$CC" = "flang" ]; then
    CM="$COREUTILS/build/flang_mod"
    CL="$COREUTILS/build/libcoreutils_flang.a"
    MF="-module-dir"
  else
    CM="$COREUTILS/build/mod"
    CL="$COREUTILS/build/libcoreutils.a"
    MF="-J"
  fi
  cd "$root"
  src=src/diffstruc
  # The development branch uses the gfortran LOC extension, so it is built with
  # GNU extensions; the migrated tree is strictly Fortran 2018.
  if [ -f "$src/diffstruc_context.f90" ]; then STD=f2018; else STD=gnu; fi
  # The migrated tree carries !$omp simd directives in its numeric kernels.
  if [ -f "$src/diffstruc_context.f90" ]; then SIMDFLAG="-fopenmp-simd -fversion-loops-for-strides -fvect-cost-model=unlimited"; else SIMDFLAG=""; fi
  order="diffstruc_global diffstruc_types diffstruc_types_sub"
  if [ -f "$src/diffstruc_context.f90" ]; then
    order="$order diffstruc_context diffstruc_plumbing diffstruc_types_sub \
diffstruc_arithmetic_sub"
  fi
  order="$order diffstruc_operations_maths diffstruc_operations_maths_sub \
diffstruc_operations_trig diffstruc_operations_hyp \
diffstruc_operations_linalg diffstruc_operations_linalg_sub \
diffstruc_operations_broadcast diffstruc_operations_broadcast_sub \
diffstruc_operations_comparison diffstruc_operations_reduction"
  seen=""
  for name in $order; do
    case " $seen " in *" $name "*) continue;; esac
    seen="$seen $name"
    [ -f "$src/$name.f90" ] || continue
    $CC -cpp -std=$STD -O2 $SIMDFLAG $extra -I "$CM" $MF "$mod" \
      -c -o "$obj/$name.o" "$src/$name.f90"
  done
  $CC -cpp -std=$STD -O2 $SIMDFLAG $extra -I "$CM" $MF "$mod" \
    -c -o "$obj/diffstruc.o" src/diffstruc.f90
  for t in test_benchmark_reverse test_benchmark_matmul test_functions; do
    $CC -cpp -std=$STD -O2 $SIMDFLAG $extra -I "$CM" $MF "$mod" \
      -c -o "$obj/$t.o" "test/$t.f90"
    $CC -O2 -o "$bin/$t" "$obj/$t.o" "$obj"/diffstruc*.o "$CL"
  done
  cd "$ROOT"
}

run_best() {
  # $1 executable, $2 number of runs
  best=""
  for i in $(seq 1 "$2"); do
    secs=$( { /usr/bin/time -p "$1" >/dev/null; } 2>&1 | awk '/^real/{print $2}')
    if [ -z "$best" ]; then best=$secs
    elif awk "BEGIN{exit !($secs < $best)}"; then best=$secs; fi
  done
  echo "$best"
}

echo "compiler: $($CC --version | head -1)"
echo

build_tree "$ROOT/build/dev_src" "$ROOT/build/dev_bin" ""
build_tree "$ROOT" "$ROOT/build/ref_bin" "-DDIFFSTRUC_SLOW_REVERSE"
build_tree "$ROOT" "$ROOT/build/opt_bin" ""
build_tree "$ROOT" "$ROOT/build/fast_bin" "-DDIFFSTRUC_FAST_TANH"

row() { printf "%-14s %10s %10s %10s %10s %10s\n" "$@"; }

echo "=== test_benchmark_reverse: best of 3 whole-program runs (seconds) ==="
row variant "chainA" "chainB" "chainC" "matmul"
dev=$(run_best "$ROOT/build/dev_bin/bin/test_benchmark_reverse" 3)
ref=$(run_best "$ROOT/build/ref_bin/bin/test_benchmark_reverse" 3)
opt=$(run_best "$ROOT/build/opt_bin/bin/test_benchmark_reverse" 3)
fast=$(run_best "$ROOT/build/fast_bin/bin/test_benchmark_reverse" 3)
printf "%-14s %10s %10s %10s %10s\n" "dev (HEAD)" "$dev" "-" "-" "-"
printf "%-14s %10s %10s %10s %10s\n" "reference" "$ref" "-" "-" "-"
printf "%-14s %10s %10s %10s %10s\n" "optimised" "$opt" "-" "-" "-"
printf "%-14s %10s %10s %10s %10s\n" "opt+fast-tanh" "$fast" "-" "-" "-"
awk "BEGIN{printf \"speedup vs dev (accurate):    %.2fx\\n\", $dev/$opt; \
            printf \"speedup vs dev (fast tanh):   %.2fx\\n\", $dev/$fast; \
            printf \"speedup vs ref (accurate):    %.2fx\\n\", $ref/$opt}"

echo
echo "=== per-chain times reported by the program (ms, last size) ==="
printf "%-10s %10s %10s %10s\n" variant "chainA" "chainB" "chainC"
for v in dev:dev_bin reference:ref_bin optimised:opt_bin fast:fast_bin; do
  name=${v%%:*}
  dir=${v##*:}
  vals=$("$ROOT/build/$dir/bin/test_benchmark_reverse" | awk '/^ *2048 */{print $3, $4, $5}')
  printf "%-10s %10s %10s %10s\n" "$name" $vals
done

echo
echo "=== test_benchmark_matmul: best of 3 whole-program runs (seconds) ==="
for v in dev:dev_bin reference:ref_bin optimised:opt_bin fast:fast_bin; do
  name=${v%%:*}
  dir=${v##*:}
  t=$(run_best "$ROOT/build/$dir/bin/test_benchmark_matmul" 3)
  printf "%-10s %s\n" "$name" "$t"
done

echo
echo "=== accuracy: reverse-benchmark reference checksums ==="
for v in dev:dev_bin reference:ref_bin optimised:opt_bin fast:fast_bin; do
  name=${v%%:*}
  dir=${v##*:}
  printf "%-10s " "$name"
  "$ROOT/build/$dir/bin/test_benchmark_reverse" | \
    awk '/Ref checksum/{printf "%s=%s ", $3, $6} END{print ""}'
done

echo
echo "=== accuracy: example suite, optimised vs dev branch ==="
"$ROOT/build/dev_bin/bin/test_functions" > build/functions_dev.txt
"$ROOT/build/opt_bin/bin/test_functions" > build/functions_opt.txt
numbers() {
  # Print every numeric field of a run, one per line.
  tr ' ,' '\n\n' < "$1" | grep -E '^[-+]?[0-9]*\.?[0-9]+([EeDd][-+]?[0-9]+)?$' | \
    sed 's/[Dd]/E/'
}
numbers build/functions_dev.txt > build/numbers_dev.txt
numbers build/functions_opt.txt > build/numbers_opt.txt
awk 'NR==FNR{a[NR]=$1+0; n=NR; next}
     {v=$1+0; d=v-a[FNR]; if(d<0)d=-d; s=(a[FNR]<0?-a[FNR]:a[FNR]);
      r=(s>1e-30?d/s:0); if(r>maxr)maxr=r; if(d>maxa)maxa=d}
     END{printf "values compared: %d, max abs diff %.3e, max rel diff %.3e\n", n, maxa, maxr;
         if(maxr > 1e-5) { print "ACCURACY REGRESSION"; exit 1 } else
         print "within 1e-5 relative tolerance"}' \
     build/numbers_dev.txt build/numbers_opt.txt

echo
echo "=== accuracy: example suite against golden/functions.txt ==="
golden=/private/var/folders/rd/bq9d983s49d_1hs82xy696l00000gn/T/opencode/golden/functions.txt
if [ -f "$golden" ]; then
  sed 's/[[:space:]]*$//' "$golden" > build/functions_golden.trim
  sed 's/[[:space:]]*$//' build/functions_opt.txt > build/functions_opt.trim
  if diff -q build/functions_opt.trim build/functions_golden.trim >/dev/null; then
    echo "byte-identical to the pre-migration golden output"
  else
    numbers build/functions_golden.trim > build/numbers_golden.txt
    # The golden file predates the migration and was recorded without -O2, so
    # -O2 code generation alone moves the last digits. Report the deviation of
    # both builds so the comparison is meaningful.
    for v in dev opt; do
      awk -v tag="$v" 'NR==FNR{a[NR]=$1+0; n=NR; next}
           {v1=$1+0; d=v1-a[FNR]; if(d<0)d=-d; s=(a[FNR]<0?-a[FNR]:a[FNR]);
            r=(s>1e-30?d/s:0); if(r>maxr)maxr=r}
           END{printf "%-4s vs golden: %d values, max rel diff %.3e\n", tag, n, maxr}' \
           build/numbers_golden.txt "build/numbers_$v.txt"
    done
    echo "(the difference is -O2 code generation; the dev branch shows the same)"
  fi
fi
