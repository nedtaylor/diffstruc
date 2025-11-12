module diffstruc__global
  !! This module contains global parameters and settings for the diffstruc library.
  implicit none

  integer :: max_recursion_depth = 1000
  !! Recursion depth limit for operations that traverse the computation graph
  integer :: default_map_capacity = 16
  !! Default capacity for pointer mapping in graph operations

end module diffstruc__global
