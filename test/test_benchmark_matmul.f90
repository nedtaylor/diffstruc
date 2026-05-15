program test_benchmark_matmul
  !! Benchmark matmul forward pass and reverse-mode gradients.
  !!
  !! The benchmark iterates over the available dense backends:
  !!   legacy BLAS, Accelerate, Metal (feature builds), and auto dispatch.
  !!
  !! Two scenarios are timed at each size for each backend:
  !!
  !!   Scenario A — type x type: Y = matmul(W_ad, X_ad)
  !!     W_ad is non-sample-dependent (shape [out,in]), X_ad is sample-dependent
  !!     (shape [in], batch samples).  W_ad%requires_grad = .false. so only
  !!     dL/dX is computed in the backward pass via get_partial_matmul_right_val,
  !!     which uses SGEMM('T','N') in the USE_BLAS path.
  !!
  !!   Scenario B — real x type: Y = matmul(W_real, X_ad)
  !!     Exercises real2d_matmul.  Forward uses SGEMM('N','N'); backward
  !!     (dL/dX) uses SGEMM('N','N') in get_partial_matmul_left_val.
  !!
  use coreutils, only: real32
  use, intrinsic :: iso_fortran_env, only: real64
  use diffstruc
  use diffstruc__backend_linalg, only: &
       backend_available, backend_name, reset_backend_mode, resolve_gemm_backend, &
       set_backend_mode, diffstruc_backend_accelerate, diffstruc_backend_auto, &
       diffstruc_backend_legacy, diffstruc_backend_metal
  implicit none

  !----------------------------------------------------------------
  ! Benchmark parameters
  !----------------------------------------------------------------
  integer, parameter :: N_SIZES = 7
  integer, parameter :: MAX_BACKENDS = 4
  integer, parameter :: IN_DIMS(N_SIZES)  = [64,  256,  512, 1024, 2048, 4096, 4096]
  integer, parameter :: OUT_DIMS(N_SIZES) = [64,  256,  512, 1024, 2048, 4096, 4096]
  integer, parameter :: BATCHES(N_SIZES)  = [512, 256,  128,   64,   32,   32, 64]
  integer, parameter :: N_FWD = 50   ! repetitions for forward pass
  integer, parameter :: N_REV = 20   ! repetitions for backward pass

  integer :: backend_modes(MAX_BACKENDS), num_backends, backend_index

  call collect_backends(backend_modes, num_backends)

  do backend_index = 1, num_backends
     call run_backend_benchmark(backend_modes(backend_index))
  end do

  call reset_backend_mode()

contains

  subroutine collect_backends(modes_out, backend_count)
    implicit none
    integer, intent(out) :: modes_out(:)
    integer, intent(out) :: backend_count

    backend_count = 0
    backend_count = backend_count + 1
    modes_out(backend_count) = diffstruc_backend_legacy
    if(backend_available(diffstruc_backend_accelerate))then
       backend_count = backend_count + 1
       modes_out(backend_count) = diffstruc_backend_accelerate
    end if
    if(backend_available(diffstruc_backend_metal))then
       backend_count = backend_count + 1
       modes_out(backend_count) = diffstruc_backend_metal
    end if
    backend_count = backend_count + 1
    modes_out(backend_count) = diffstruc_backend_auto
  end subroutine collect_backends

  subroutine run_backend_benchmark(backend_mode)
    implicit none
    integer, intent(in) :: backend_mode

    integer  :: s, r, in_d, out_d, batch, effective_mode
    real(real64) :: t0, t1
    real(real32) :: t_fwdA, t_revA, t_fwdB, t_revB
    real     :: checksum
    type(array_type)          :: W_ad, X_ad
    type(array_type), pointer :: Y_ad
    real(real32), allocatable :: W_real(:,:), X_real(:,:)

    call set_backend_mode(backend_mode)
    checksum = 0.0

    write(*,'(A)') ""
    write(*,'(A,A,A)') "=== matmul benchmark  [backend = ", trim(backend_name(backend_mode)), "] ==="
    write(*,'(A)') ""
    write(*,'(A5,A6,A6,2(A13),2(A13),2X,A9)') &
         "out", "in", "batch", "AdFwd(ms)", "AdRev(ms)", "ReFwd(ms)", "ReRev(ms)", "chosen"
    write(*,'(A)') repeat('-', 72)

    do s = 1, N_SIZES
       in_d  = IN_DIMS(s)
       out_d = OUT_DIMS(s)
       batch = BATCHES(s)
       effective_mode = resolve_gemm_backend(backend_mode, out_d, batch, in_d, .false.)

       call W_ad%allocate(array_shape=[out_d * in_d, 1])
       W_ad%shape = [out_d, in_d]
       W_ad%is_sample_dependent = .false.
       W_ad%is_temporary = .false.
       W_ad%requires_grad = .false.
       call random_number(W_ad%val)

       call X_ad%allocate(array_shape=[in_d, batch])
       X_ad%is_sample_dependent = .true.
       X_ad%is_temporary = .false.
       call X_ad%set_requires_grad(.true.)
       call random_number(X_ad%val)

       call warmup_type_left_case(W_ad, X_ad)

       t0 = wall_time_seconds()
       do r = 1, N_FWD
          Y_ad => matmul(W_ad, X_ad)
          checksum = checksum + Y_ad%val(1,1)
          Y_ad%is_temporary = .true.
          call Y_ad%deallocate()
          deallocate(Y_ad)
       end do
       t1 = wall_time_seconds()
       t_fwdA = real((t1 - t0) / real(N_FWD, real64) * 1000.0_real64, real32)

       t0 = wall_time_seconds()
       do r = 1, N_REV
          Y_ad => matmul(W_ad, X_ad)
          Y_ad%is_temporary = .false.
          call Y_ad%grad_reverse(reset_graph=.true.)
          call Y_ad%nullify_graph()
          if(associated(X_ad%grad)) then
             call X_ad%grad%deallocate()
             deallocate(X_ad%grad)
             nullify(X_ad%grad)
          end if
          Y_ad%is_temporary = .true.
          call Y_ad%deallocate()
          deallocate(Y_ad)
       end do
       t1 = wall_time_seconds()
       t_revA = real((t1 - t0) / real(N_REV, real64) * 1000.0_real64, real32)

       call W_ad%deallocate()
       call X_ad%deallocate()

       allocate(W_real(out_d, in_d), X_real(in_d, batch))
       call random_number(W_real)
       call random_number(X_real)

       call X_ad%allocate(array_shape=[in_d, batch])
       X_ad%is_sample_dependent = .true.
       X_ad%is_temporary = .false.
       call X_ad%set_requires_grad(.true.)
       X_ad%val = X_real

       call warmup_real_left_case(W_real, X_ad)

       t0 = wall_time_seconds()
       do r = 1, N_FWD
          Y_ad => matmul(W_real, X_ad)
          checksum = checksum + Y_ad%val(1,1)
          Y_ad%is_temporary = .true.
          call Y_ad%deallocate()
          deallocate(Y_ad)
       end do
       t1 = wall_time_seconds()
       t_fwdB = real((t1 - t0) / real(N_FWD, real64) * 1000.0_real64, real32)

       t0 = wall_time_seconds()
       do r = 1, N_REV
          Y_ad => matmul(W_real, X_ad)
          Y_ad%is_temporary = .false.
          call Y_ad%grad_reverse(reset_graph=.true.)
          call Y_ad%nullify_graph()
          if(associated(X_ad%grad)) then
             call X_ad%grad%deallocate()
             deallocate(X_ad%grad)
             nullify(X_ad%grad)
          end if
          Y_ad%is_temporary = .true.
          call Y_ad%deallocate()
          deallocate(Y_ad)
       end do
       t1 = wall_time_seconds()
       t_revB = real((t1 - t0) / real(N_REV, real64) * 1000.0_real64, real32)

       call X_ad%deallocate()
       deallocate(W_real, X_real)

       write(*,'(I5,I6,I6,4F13.3,2X,A9)') &
            out_d, in_d, batch, t_fwdA, t_revA, t_fwdB, t_revB, trim(backend_name(effective_mode))
    end do

    write(*,'(A)') repeat('-', 72)
    write(*,'(A,E12.4)') "Checksum (anti-DCE): ", checksum
    write(*,'(A)') &
         "Columns: AdFwd=autodiff fwd, AdRev=autodiff rev, " // &
         "ReFwd=real-matrix fwd, ReRev=real-matrix rev"
    write(*,'(A)') ""
  end subroutine run_backend_benchmark

  subroutine warmup_type_left_case(w_ad, x_ad)
    implicit none
    type(array_type), intent(inout) :: w_ad, x_ad
    type(array_type), pointer :: y_ad

    y_ad => matmul(w_ad, x_ad)
    y_ad%is_temporary = .false.
    call y_ad%grad_reverse(reset_graph=.true.)
    call y_ad%nullify_graph()
    if(associated(x_ad%grad)) then
       call x_ad%grad%deallocate()
       deallocate(x_ad%grad)
       nullify(x_ad%grad)
    end if
    y_ad%is_temporary = .true.
    call y_ad%deallocate()
    deallocate(y_ad)
  end subroutine warmup_type_left_case

  subroutine warmup_real_left_case(w_real, x_ad)
    implicit none
    real(real32), intent(in) :: w_real(:,:)
    type(array_type), intent(inout) :: x_ad
    type(array_type), pointer :: y_ad

    y_ad => matmul(w_real, x_ad)
    y_ad%is_temporary = .false.
    call y_ad%grad_reverse(reset_graph=.true.)
    call y_ad%nullify_graph()
    if(associated(x_ad%grad)) then
       call x_ad%grad%deallocate()
       deallocate(x_ad%grad)
       nullify(x_ad%grad)
    end if
    y_ad%is_temporary = .true.
    call y_ad%deallocate()
    deallocate(y_ad)
  end subroutine warmup_real_left_case

  real(real64) function wall_time_seconds() result(seconds)
    implicit none
    integer :: count, count_rate

    call system_clock(count, count_rate)
    seconds = real(count, real64) / real(count_rate, real64)
  end function wall_time_seconds

end program test_benchmark_matmul
