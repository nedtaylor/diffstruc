program test_threads
  !! Thread-safety check: several threads run independent graphs at once.
  !!
  !! Two layouts are exercised:
  !!
  !!   1. each thread uses its own *default* context (thread-private);
  !!   2. each thread creates an explicit context with `new_context`.
  !!
  !! Every thread must reproduce the serial checksum bit for bit. The arrays are
  !! filled deterministically, so the only source of a difference would be a
  !! shared variable or a racing accumulation.
  use coreutils, only: real32
  use diffstruc
#ifdef _OPENMP
  use omp_lib, only: omp_get_max_threads, omp_get_thread_num
#endif
  implicit none

  integer, parameter :: N = 512       !! elements
  integer, parameter :: BATCH = 512   !! samples (N*BATCH clears the parallel threshold)
  integer, parameter :: MAX_THREADS = 8
  integer, parameter :: PASSES = 20

  real(real32) :: reference, checks(MAX_THREADS)
  integer :: nthreads, t

#ifdef _OPENMP
  nthreads = min(MAX_THREADS, max(1, omp_get_max_threads()))
#else
  nthreads = 1
#endif

  reference = chain_checksum(0)
  checks = -1.0_real32

  !$omp parallel num_threads(nthreads)
  block
    integer :: me
    me = 1
#ifdef _OPENMP
    me = omp_get_thread_num() + 1
#endif
    if(me <= MAX_THREADS) checks(me) = chain_checksum(0)
  end block
  !$omp end parallel

  call compare_checks('thread-private default contexts', reference, checks, nthreads)

  ! Explicit contexts: each thread owns (and destroys) its own.
  checks = -1.0_real32
  !$omp parallel num_threads(nthreads)
  block
    integer :: me
    me = 1
#ifdef _OPENMP
    me = omp_get_thread_num() + 1
#endif
    if(me <= MAX_THREADS) checks(me) = chain_checksum(1)
  end block
  !$omp end parallel

  call compare_checks('explicit new_context', reference, checks, nthreads)

  write(*, '(A,I0,A)') 'thread checks passed (', nthreads, ' threads)'

contains

  subroutine compare_checks(label, reference, checks, nthreads)
    character(len=*), intent(in) :: label
    real(real32), intent(in) :: reference
    real(real32), intent(in) :: checks(:)
    integer, intent(in) :: nthreads
    integer :: i

    do i = 1, nthreads
       if(checks(i) == -1.0_real32) then
          write(*, *) 'thread did not run: ', label, i
          stop 1
       end if
       if(checks(i) /= reference) then
          write(*, '(A,A)') 'checksum mismatch: ', label
          write(*, '(A,I0,2(1X,ES16.8))') 'thread ', i, checks(i), reference
          stop 1
       end if
    end do
  end subroutine compare_checks

  function chain_checksum(explicit_context) result(chk)
    !! loss = sum((W*x+b)^2) for one graph; returns the gradient checksum.
    integer, intent(in) :: explicit_context
    !! 0 = use this thread's default context, 1 = create and destroy one
    real(real32) :: chk

    type(array_type) :: w, x, b
    type(array_type), pointer :: y, loss
    type(array_context_type) :: ctx
    integer :: i, s, pass

    if(explicit_context /= 0) then
       call new_context(ctx, n_threads=1)
       w%ctx = ctx
       x%ctx = ctx
       b%ctx = ctx
    end if

    call w%allocate(array_shape=[N, 1], sample_dependent=.false., &
         temporary=.false.)
    call x%allocate(array_shape=[N, BATCH], sample_dependent=.true., &
         temporary=.false.)
    call b%allocate(array_shape=[N, 1], sample_dependent=.false., &
         temporary=.false.)
    call w%set_requires_grad(.true.)
    call x%set_requires_grad(.true.)
    call b%set_requires_grad(.true.)

    do i = 1, N
       w%val(i, 1) = real(mod(7*i, 17), real32) / 17.0_real32
       b%val(i, 1) = real(mod(3*i, 11), real32) / 110.0_real32
    end do
    do s = 1, BATCH
       do i = 1, N
          x%val(i, s) = real(mod(13*i + 29*s, 31), real32) / 31.0_real32
       end do
    end do

    chk = 0.0_real32
    do pass = 1, PASSES
       y => squared(w * x + b)
       loss => sum(y)
       loss%is_temporary = .false.
       call loss%grad_reverse(reset_graph=.true.)
       chk = chk + sum(w%grad%val) + sum(x%grad%val) + sum(b%grad%val)
       call loss%nullify_graph()
    end do

    call w%deallocate()
    call x%deallocate()
    call b%deallocate()
    if(explicit_context /= 0) call destroy_context(ctx)
  end function chain_checksum

end program test_threads
