module diffstruc__global
  !! This module contains global parameters and settings for the
  !! diffstruc library.
  implicit none

  integer :: diffstruc__max_recursion_depth = 1000
  !! Recursion depth limit for operations that traverse the computation
  !! graph

  integer :: diffstruc__pool_limit = 0
  !! Maximum number of pooled gradient buffers a context keeps for
  !! reuse (0 means unlimited)

  integer :: diffstruc__visit_counter = 0
  !! Monotonic counter used to stamp graph traversals without a visited set

  integer, parameter :: diffstruc__init_map_cap = 64
  !! Initial capacity of pointer maps used when copying graphs
  !! Maximum number of pooled gradient buffers a context keeps for
  !! reuse (0 means unlimited)

end module diffstruc__global
