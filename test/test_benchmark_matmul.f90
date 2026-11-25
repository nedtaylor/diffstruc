program test_benchmark_matmul
  !! Benchmark matmul forward pass and reverse-mode gradients.
  !!
  !! Two scenarios are timed at each size:
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
  !! Compile with -DUSE_BLAS to enable the BLAS path.
  use coreutils, only: real32
  use diffstruc
  implicit none

  !----------------------------------------------------------------
  ! Benchmark parameters
  !----------------------------------------------------------------
  integer, parameter :: N_SIZES = 4
  integer, parameter :: IN_DIMS(N_SIZES)  = [64,  256,  512, 1024]
  integer, parameter :: OUT_DIMS(N_SIZES) = [64,  256,  512, 1024]
  integer, parameter :: BATCHES(N_SIZES)  = [512, 256,  128,   64]
  integer, parameter :: N_FWD = 50   ! repetitions for forward pass
  integer, parameter :: N_REV = 20   ! repetitions for backward pass

  integer  :: s, r, in_d, out_d, batch
  real     :: t0, t1, t_fwdA, t_revA, t_fwdB, t_revB
  real     :: checksum   ! prevents dead-code elimination

  type(array_type)          :: W_ad, X_ad
  type(array_type), pointer :: Y_ad
  real(real32), allocatable :: W_real(:,:), X_real(:,:)

  checksum = 0.0

  write(*,'(A)') ""
#ifndef NO_BLAS
  write(*,'(A)') "=== matmul benchmark  [BLAS = ON] ==="
#else
  write(*,'(A)') "=== matmul benchmark  [BLAS = OFF] ==="
#endif
  write(*,'(A)') ""
  write(*,'(A5,A6,A6,2(A13),2(A13))') &
       "out", "in", "batch", "AdFwd(ms)", "AdRev(ms)", "ReFwd(ms)", "ReRev(ms)"
  write(*,'(A)') repeat('-', 60)

  do s = 1, N_SIZES
     in_d  = IN_DIMS(s)
     out_d = OUT_DIMS(s)
     batch = BATCHES(s)

     !-------------------------------------------------------------
     ! Build arrays for scenario A:  Y = matmul(W_ad, X_ad)
     !
     !  W_ad – 2D weight matrix [out_d, in_d], non-sample-dependent.
     !          allocate(array_shape=[out_d, in_d, 1]) sets
     !            val   ← (out_d*in_d, 1)
     !            shape ← [out_d, in_d]      (leading dims; last entry is samples)
     !          so the matmul branch sees a 2D matrix directly.
     !
     !  X_ad – 1D input [in_d] per sample.
     !          allocate(array_shape=[in_d, batch]) sets
     !            val   ← (in_d, batch)
     !            shape ← [in_d]             (1D, batch samples)
     !-------------------------------------------------------------
     call W_ad%allocate(array_shape=[out_d, in_d, 1], &
          sample_dependent=.false., temporary=.false.)
     ! W does not require grad: only dL/dX is computed in backward,
     ! which is the path exercised by BLAS sgemm('T','N').
     W_ad%requires_grad = .false.
     call random_number(W_ad%val)

     call X_ad%allocate(array_shape=[in_d, batch], sample_dependent=.true., &
          temporary=.false.)
     call X_ad%set_requires_grad(.true.)
     call random_number(X_ad%val)

     !--- Forward A ---
     call cpu_time(t0)
     do r = 1, N_FWD
        Y_ad => matmul(W_ad, X_ad)
        checksum = checksum + Y_ad%val(1,1)
        Y_ad%is_temporary = .true.
     end do
     call cpu_time(t1)
     t_fwdA = (t1 - t0) / real(N_FWD) * 1000.0

     !--- Backward A ---
     call cpu_time(t0)
     do r = 1, N_REV
        Y_ad => matmul(W_ad, X_ad)
        Y_ad%is_temporary = .false.
        call Y_ad%grad_reverse(reset_graph=.true.)
        call Y_ad%nullify_graph()
        Y_ad%is_temporary = .true.
     end do
     call cpu_time(t1)
     t_revA = (t1 - t0) / real(N_REV) * 1000.0

     call W_ad%deallocate()
     call X_ad%deallocate()

     !-------------------------------------------------------------
     ! Build arrays for scenario B:  Y = matmul(W_real, X_ad)
     !
     !  Calls real2d_matmul.  W_real is a plain Fortran array — no
     !  autodiff overhead for W.  Forward: SGEMM('N','N').
     !  Backward dL/dX: SGEMM('N','N') in get_partial_matmul_left_val
     !  (since the internally-created a_array has 2D shape [out,in]).
     !-------------------------------------------------------------
     allocate(W_real(out_d, in_d), X_real(in_d, batch))
     call random_number(W_real)
     call random_number(X_real)

     call X_ad%allocate(array_shape=[in_d, batch], sample_dependent=.true., &
          temporary=.false.)
     call X_ad%set_requires_grad(.true.)
     X_ad%val = X_real

     !--- Forward B ---
     call cpu_time(t0)
     do r = 1, N_FWD
        Y_ad => matmul(W_real, X_ad)
        checksum = checksum + Y_ad%val(1,1)
        Y_ad%is_temporary = .true.
     end do
     call cpu_time(t1)
     t_fwdB = (t1 - t0) / real(N_FWD) * 1000.0

     !--- Backward B ---
     call cpu_time(t0)
     do r = 1, N_REV
        Y_ad => matmul(W_real, X_ad)
        Y_ad%is_temporary = .false.
        call Y_ad%grad_reverse(reset_graph=.true.)
        call Y_ad%nullify_graph()
        Y_ad%is_temporary = .true.
     end do
     call cpu_time(t1)
     t_revB = (t1 - t0) / real(N_REV) * 1000.0
     call X_ad%deallocate()
     deallocate(W_real, X_real)

     write(*,'(I5,I6,I6,4F13.3)') &
          out_d, in_d, batch, t_fwdA, t_revA, t_fwdB, t_revB
  end do

  write(*,'(A)') repeat('-', 60)
  write(*,'(A,E12.4)') "Checksum (anti-DCE): ", checksum
  write(*,'(A)') &
       "Columns: AdFwd=autodiff fwd, AdRev=autodiff rev, " // &
       "ReFwd=real-matrix fwd, ReRev=real-matrix rev"
  write(*,'(A)') ""

end program test_benchmark_matmul
