module diffstruc
  !! This is the top-level module for the diffstruc Fortran library.
  use coreutils, only: real32
  use diffstruc__global, only: diffstruc__init_map_cap, &
       diffstruc__default_max_recursion_depth, diffstruc__parallel_min_elems
  use diffstruc__types, only: &
       array_type, array_context_type, default_context, &
       set_default_context, reset_default_context, get_partial, &
       new_context, destroy_context, set_num_threads, get_num_threads, &
       set_max_recursion_depth, team_size_for, context_of, canonical, live, &
       discard_sources, operator(+), operator(-), operator(*), operator(/), operator(**), &
       sum, mean, spread, unspread, exp, log
  use diffstruc__operations_trig, only: sin, cos, tan
  use diffstruc__operations_hyp, only: tanh, tanhf
  use diffstruc__operations_linalg, only: &
       matmul, operator(.mmul.), &
       outer_product, operator(.outer.), &
       dot_product, operator(.dot.), &
       transpose
  use diffstruc__operations_broadcast, only: &
       concat, slice_left, slice_right, ltrim, rtrim, &
       operator(.index.), reverse_index, &
       pack, unpack, reshape
  use diffstruc__operations_comparison, only: &
       operator(.lt.), operator(.gt.), operator(.le.), operator(.ge.), merge
  use diffstruc__operations_reduction, only: maxval, max
  use diffstruc__operations_maths, only: sqrt, sign, sigmoid, gaussian, abs, &
       log10, squared, cubed
  implicit none

  private

  public :: real32
  public :: diffstruc__init_map_cap, diffstruc__default_max_recursion_depth
  public :: diffstruc__parallel_min_elems
  public :: array_type, get_partial
  public :: array_context_type, default_context
  public :: set_default_context, reset_default_context
  public :: new_context, destroy_context, set_num_threads, get_num_threads
  public :: set_max_recursion_depth, team_size_for
  public :: context_of, canonical, live, discard_sources

  public :: operator(+), operator(-), operator(*), operator(/), operator(**)
  public :: sum, mean, spread, unspread, exp, log
  public :: sin, cos, tan
  public :: tanh, tanhf
  public :: matmul, operator(.mmul.), &
       outer_product, operator(.outer.), &
       dot_product, operator(.dot.), &
       transpose
  public :: concat, slice_left, slice_right, ltrim, rtrim, &
       operator(.index.), reverse_index, &
       pack, unpack, reshape
  public :: operator(.lt.), operator(.gt.), operator(.le.), operator(.ge.), merge
  public :: maxval, max
  public :: sqrt, sign, sigmoid, gaussian, abs, log10, squared, cubed

end module diffstruc
