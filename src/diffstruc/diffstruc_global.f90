module diffstruc__global
  !! Library-wide parameter constants.
  !!
  !! This module holds *parameters only*. All mutable state -- registries,
  !! pools, counters, phase tags, recursion limits and thread settings --
  !! lives in a context (see `array_context_type`), so a thread that owns a
  !! context touches no shared variable.
  implicit none

  integer, parameter :: diffstruc__init_map_cap = 64
  !! Initial capacity of pointer maps used when copying graphs

  integer, parameter :: diffstruc__default_max_recursion_depth = 1000
  !! Default recursion depth limit for graph traversals; a context may lower
  !! or raise its own copy

  integer, parameter :: diffstruc__parallel_min_elems = 262144
  !! Minimum number of elements (elements x samples) before a kernel asks for
  !! an OpenMP team. Below this a parallel region costs more than it saves.

  integer, parameter :: diffstruc__parallel_elems_per_thread = 262144
  !! Work (elements x samples) each thread should carry. The team size is
  !! `work / this` clamped to the available threads, so a small node stays
  !! serial and a large one scales with its size.

end module diffstruc__global
