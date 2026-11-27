module diffstruc__parallel
  !! Optional OpenMP support.
  !!
  !! The library builds and runs without OpenMP: every `!$omp` directive is a
  !! comment and every team size is 1. Compiled with `-fopenmp`, a kernel asks
  !! for a team only when its work exceeds `diffstruc__parallel_min_elems`.
  !!
  !! No mutable state lives here. The requested thread count and the runtime
  !! default are captured per context when it is created, so threads using
  !! different contexts never affect each other. `team_size` is pure, which
  !! lets the reverse-mode partial kernels (whose interfaces are `pure`) use
  !! it.
  use, intrinsic :: iso_fortran_env, only: int64
  use diffstruc__global, only: diffstruc__parallel_min_elems, &
       diffstruc__parallel_elems_per_thread
#ifdef _OPENMP
  use omp_lib, only: omp_get_max_threads, omp_in_parallel
#endif
  implicit none
  private

  public :: team_size, available_threads, in_parallel

contains

  pure integer function team_size(n_elem, n_samples, n_requested, &
       n_available) result(nth)
    !! Threads a kernel should request (1 means run serial).
    integer, intent(in) :: n_elem
    !! Elements per sample
    integer, intent(in) :: n_samples
    !! Samples
    integer, intent(in) :: n_requested
    !! Threads requested by the context (0 = runtime default)
    integer, intent(in) :: n_available
    !! Runtime default captured by the context (>= 1)

    integer :: n
    integer(int64) :: work

    n = max(1, n_available)
    if(n_requested > 0) n = n_requested
    if(n <= 1) then
       nth = 1
       return
    end if
    work = int(max(1, n_elem), int64) * int(max(1, n_samples), int64)
    if(work < int(diffstruc__parallel_min_elems, int64)) then
       nth = 1
       return
    end if
    nth = int(max(1_int64, work / &
         int(diffstruc__parallel_elems_per_thread, int64)))
    nth = min(n, nth)
  end function team_size

  integer function available_threads()
    !! The runtime's default OpenMP team size (1 without OpenMP).
#ifdef _OPENMP
    available_threads = max(1, omp_get_max_threads())
#else
    available_threads = 1
#endif
  end function available_threads

  logical function in_parallel()
    !! Whether the caller is already inside an OpenMP parallel region.
#ifdef _OPENMP
    in_parallel = omp_in_parallel()
#else
    in_parallel = .false.
#endif
  end function in_parallel

end module diffstruc__parallel
