module diffstruc
  !! This is the top-level module for the diffstruc Fortran library.
  use coreutils, only: real32
  use diffstruc__global, only: diffstruc__max_recursion_depth, diffstruc__init_map_cap
  use diffstruc__types, only: &
       array_type, get_partial, &
       operator(+), operator(-), operator(*), operator(/), operator(**), &
       sum, mean, spread, unspread, exp, log
  use diffstruc__operations_trig, only: sin, cos, tan
  use diffstruc__operations_hyp, only: tanh
  use diffstruc__operations_linalg, only: &
       matmul, operator(.mmul.), &
       outer_product, operator(.outer.), &
       dot_product, operator(.dot.), &
       transpose
  use diffstruc__operations_broadcast, only: &
       concat, slice_left, slice_right, ltrim, rtrim, &
       operator(.index.), reverse_index, &
       pack, unpack
  use diffstruc__operations_comparison, only: &
       operator(.lt.), operator(.gt.), operator(.le.), operator(.ge.), merge
  use diffstruc__operations_reduction, only: maxval, max
  use diffstruc__operations_maths, only: sqrt, sign, sigmoid, gaussian, abs
  implicit none

  private

  public :: real32
  public :: diffstruc__max_recursion_depth, diffstruc__init_map_cap
  public :: array_type, get_partial

  public :: operator(+), operator(-), operator(*), operator(/), operator(**)
  public :: sum, mean, spread, unspread, exp, log
  public :: sin, cos, tan
  public :: tanh
  public :: matmul, operator(.mmul.), &
       outer_product, operator(.outer.), &
       dot_product, operator(.dot.), &
       transpose
  public :: concat, slice_left, slice_right, ltrim, rtrim, &
       operator(.index.), reverse_index, &
       pack, unpack
  public :: operator(.lt.), operator(.gt.), operator(.le.), operator(.ge.), merge
  public :: maxval, max
  public :: sqrt, sign, sigmoid, gaussian, abs

end module diffstruc
