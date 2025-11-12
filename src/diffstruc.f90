module diffstruc
  !! This is the top-level module for the diffstruc Fortran library.
  use coreutils, only: real32
  use diffstruc__global, only: max_recursion_depth, default_map_capacity
  use diffstruc__types, only: &
       array_type, get_partial, &
       operator(+), operator(-), operator(*), operator(/), operator(**), &
       sum, mean, spread, unspread, exp, log
  use diffstruc__operations_trig, only: sin, cos, tan
  use diffstruc__operations_hyp, only: tanh
  use diffstruc__operations_linalg, only: operator(.mmul.), operator(.outer.), transpose
  use diffstruc__operations_broadcast, only: &
       operator(.concat.), operator(.ltrim.), operator(.rtrim.), &
       operator(.index.), reverse_index, &
       pack, unpack
  use diffstruc__operations_comparison, only: operator(.lt.), operator(.gt.), merge
  use diffstruc__operations_reduction, only: maxval, max
  use diffstruc__operations_maths, only: sqrt, sign, sigmoid, gaussian
  implicit none

  private

  public :: real32
  public :: max_recursion_depth, default_map_capacity
  public :: array_type, get_partial

  public :: operator(+), operator(-), operator(*), operator(/), operator(**)
  public :: sum, mean, spread, unspread, exp, log
  public :: sin, cos, tan
  public :: tanh
  public :: operator(.mmul.), operator(.outer.), transpose
  public :: operator(.concat.), operator(.ltrim.), operator(.rtrim.), &
       operator(.index.), reverse_index, &
       pack, unpack
  public :: operator(.lt.), operator(.gt.), merge
  public :: maxval, max
  public :: sqrt, sign, sigmoid, gaussian

end module diffstruc
