program test_benchmark_threads
  !! Thread-scaling benchmark for the OpenMP kernels.
  !!
  !! The sizes are chosen so that one node's storage is a few MiB, which is
  !! where the per-node parallel regions pay off. Run it under different
  !! `OMP_NUM_THREADS` values (see `bench/threads.sh`) and compare the reported
  !! times; the checksum column must not change.
  !!
  !! Chains:
  !!   A - loss = sum((W*x+b)^2)
  !!   B - loss = sum(tanh(W2*(W1*x+b1)+b2))
  use coreutils, only: real32
  use diffstruc
#ifdef _OPENMP
  use omp_lib, only: omp_get_max_threads
#endif
  implicit none

  integer, parameter :: N_SIZES = 2
  integer, parameter :: DIMS(N_SIZES) = [2048, 2048]
  integer, parameter :: BATCHES(N_SIZES) = [2048, 4096]
  integer, parameter :: N_REV = 5

  integer :: s, d, batch, threads, seed_size
  integer, allocatable :: seed_vals(:)

#ifdef _OPENMP
  threads = omp_get_max_threads()
#else
  threads = 1
#endif

  call random_seed(size=seed_size)
  allocate(seed_vals(seed_size))
  seed_vals = 42
  call random_seed(put=seed_vals)
  write(*, '(A,I0)') "threads: ", threads
  write(*, '(A6,A8,A8,A6,3(A14))') "dim", "batch", "elem", "chain", "ms", "checksum"
  write(*, '(A)') repeat('-', 62)

  do s = 1, N_SIZES
     d = DIMS(s)
     batch = BATCHES(s)
     call bench_a(d, batch, threads)
     call bench_b(d, batch, threads)
  end do

contains

  subroutine bench_a(d, batch, threads)
    integer, intent(in) :: d, batch, threads
    type(array_type) :: w, x, b
    type(array_type), pointer :: y, loss
    real :: t0, t1
    real(real32) :: chk
    integer :: r

    call w%allocate(array_shape=[d, 1], sample_dependent=.false., &
         temporary=.false.)
    call x%allocate(array_shape=[d, batch], sample_dependent=.true., &
         temporary=.false.)
    call b%allocate(array_shape=[d, 1], sample_dependent=.false., &
         temporary=.false.)
    call w%set_requires_grad(.true.)
    call x%set_requires_grad(.true.)
    call b%set_requires_grad(.true.)
    call random_number(w%val); w%val = w%val * 0.1_real32
    call random_number(x%val)
    call random_number(b%val); b%val = b%val * 0.01_real32

    chk = 0.0_real32
    call cpu_time(t0)
    do r = 1, N_REV
       y => squared(w * x + b)
       loss => sum(y)
       loss%is_temporary = .false.
       call loss%grad_reverse(reset_graph=.true.)
       chk = chk + sum(x%grad%val)
       call loss%nullify_graph()
    end do
    call cpu_time(t1)
    call report(d, batch, "A", (t1 - t0) / real(N_REV) * 1000.0, chk, threads)

    call w%deallocate(); call x%deallocate(); call b%deallocate()
  end subroutine bench_a

  subroutine bench_b(d, batch, threads)
    integer, intent(in) :: d, batch, threads
    type(array_type) :: w1, w2, x, b1, b2
    type(array_type), pointer :: y, loss, h
    real :: t0, t1
    real(real32) :: chk
    integer :: r

    call w1%allocate(array_shape=[d, 1], sample_dependent=.false., &
         temporary=.false.)
    call w2%allocate(array_shape=[d, 1], sample_dependent=.false., &
         temporary=.false.)
    call x%allocate(array_shape=[d, batch], sample_dependent=.true., &
         temporary=.false.)
    call b1%allocate(array_shape=[d, 1], sample_dependent=.false., &
         temporary=.false.)
    call b2%allocate(array_shape=[d, 1], sample_dependent=.false., &
         temporary=.false.)
    call w1%set_requires_grad(.true.)
    call w2%set_requires_grad(.true.)
    call x%set_requires_grad(.true.)
    call b1%set_requires_grad(.true.)
    call b2%set_requires_grad(.true.)
    call random_number(w1%val); w1%val = w1%val * 0.1_real32
    call random_number(w2%val); w2%val = w2%val * 0.1_real32
    call random_number(x%val)
    call random_number(b1%val); b1%val = b1%val * 0.01_real32
    call random_number(b2%val); b2%val = b2%val * 0.01_real32

    chk = 0.0_real32
    call cpu_time(t0)
    do r = 1, N_REV
       h => w1 * x + b1
       y => tanh(w2 * h + b2)
       loss => sum(y)
       loss%is_temporary = .false.
       call loss%grad_reverse(reset_graph=.true.)
       chk = chk + sum(x%grad%val)
       call loss%nullify_graph()
    end do
    call cpu_time(t1)
    call report(d, batch, "B", (t1 - t0) / real(N_REV) * 1000.0, chk, threads)

    call w1%deallocate(); call w2%deallocate(); call x%deallocate()
    call b1%deallocate(); call b2%deallocate()
  end subroutine bench_b

  subroutine report(d, batch, chain, ms, chk, threads)
    integer, intent(in) :: d, batch, threads
    character(len=*), intent(in) :: chain
    real, intent(in) :: ms
    real(real32), intent(in) :: chk

    write(*, '(I6,I8,I8,A6,2X,F12.3,2X,ES14.6)') d, batch, d*batch, chain, ms, chk
  end subroutine report

end program test_benchmark_threads
