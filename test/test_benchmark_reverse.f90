program test_benchmark_reverse
  !! Benchmark reverse-mode gradient accumulation routines.
  !!
  !! Tests accumulate_gradient_single and accumulate_gradient_samples
  !! via composite computation graphs at various sizes.
  !!
  !! Scenarios:
  !!   A — Element-wise chain: loss = sum( (W * x + b)^2 )
  !!       W non-sample-dependent, x sample-dependent, b non-sample-dependent
  !!   B — Deeper chain: loss = sum( tanh(W2 * (W1 * x + b1) + b2) )
  !!       Exercises multiple recursion levels
  !!
  !! Reports: backward pass time (ms), correctness checksums
  use coreutils, only: real32
  use diffstruc
  implicit none

  !----------------------------------------------------------------
  ! Benchmark parameters
  !----------------------------------------------------------------
  integer, parameter :: N_SIZES = 5
  integer, parameter :: DIMS(N_SIZES)    = [32,   128,   512,  1024,  2048]
  integer, parameter :: BATCHES(N_SIZES) = [512,  256,   128,    64,    32]
  integer, parameter :: N_REV = 200  ! repetitions for backward pass

  integer  :: s, r, d, batch, seed_size
  integer, allocatable :: seed_vals(:)
  real     :: t0, t1, t_revA, t_revB, t_revC
  real(real32) :: checksum_a, checksum_b, checksum_c
  real(real32) :: ref_checksum_a, ref_checksum_b, ref_checksum_c

  type(array_type)          :: W, x, b, a1, a2
  type(array_type)          :: W1, W2, b1, b2
  type(array_type), pointer :: y, loss
  type(array_type), pointer :: h, y2, loss2

  logical :: first_pass

  ! Fixed seed for reproducibility
  call random_seed(size=seed_size)
  allocate(seed_vals(seed_size))
  seed_vals = 42
  call random_seed(put=seed_vals)

  write(*,'(A)') ""
  write(*,'(A)') "=== reverse-mode gradient accumulation benchmark ==="
  write(*,'(A)') "Scenario A: loss = sum((W*x+b)^2)   — mixed sample/non-sample"
  write(*,'(A)') "Scenario B: loss = sum(tanh(W2*(W1*x+b1)+b2)) — deeper chain"
  write(*,'(A)') "Scenario C: loss = sum((a1+a2)^2)    — non-sample-dependent only"
  write(*,'(A)') ""
  write(*,'(A5,A6,3(A13))') &
       "dim", "batch", "ChainA(ms)", "ChainB(ms)", "ChainC(ms)"
  write(*,'(A)') repeat('-', 56)

  do s = 1, N_SIZES
     d     = DIMS(s)
     batch = BATCHES(s)

     !-------------------------------------------------------------
     ! Scenario A: loss = sum( (W * x + b)^2 )
     !   W: non-sample-dependent [d, 1]
     !   x: sample-dependent [d, batch]
     !   b: non-sample-dependent [d, 1]
     !-------------------------------------------------------------
     call W%allocate(array_shape=[d, 1])
     W%is_sample_dependent = .false.
     W%is_temporary = .false.
     call W%set_requires_grad(.true.)
     call random_number(W%val)
     W%val = W%val * 0.1_real32

     call x%allocate(array_shape=[d, batch])
     x%is_sample_dependent = .true.
     x%is_temporary = .false.
     call x%set_requires_grad(.true.)
     call random_number(x%val)

     call b%allocate(array_shape=[d, 1])
     b%is_sample_dependent = .false.
     b%is_temporary = .false.
     call b%set_requires_grad(.true.)
     call random_number(b%val)
     b%val = b%val * 0.01_real32

     checksum_a = 0.0_real32
     first_pass = .true.

     !--- Backward A ---
     call cpu_time(t0)
     do r = 1, N_REV
        y => (W * x + b) ** 2._real32
        loss => sum(y)
        loss%is_temporary = .false.
        call loss%grad_reverse(reset_graph=.true.)

        ! Save checksum from first pass for correctness reference
        if(first_pass)then
           ref_checksum_a = 0.0_real32
           if(associated(W%grad)) ref_checksum_a = ref_checksum_a + sum(W%grad%val)
           if(associated(x%grad)) ref_checksum_a = ref_checksum_a + sum(x%grad%val)
           if(associated(b%grad)) ref_checksum_a = ref_checksum_a + sum(b%grad%val)
           first_pass = .false.
        end if

        ! Accumulate checksum (anti-DCE)
        if(associated(W%grad)) checksum_a = checksum_a + sum(W%grad%val)
        if(associated(x%grad)) checksum_a = checksum_a + sum(x%grad%val)
        if(associated(b%grad)) checksum_a = checksum_a + sum(b%grad%val)

        ! Cleanup
        call loss%nullify_graph()
        if(associated(W%grad))then
           call W%grad%deallocate(); deallocate(W%grad); nullify(W%grad)
           W%owns_gradient = .false.
        end if
        if(associated(x%grad))then
           call x%grad%deallocate(); deallocate(x%grad); nullify(x%grad)
           x%owns_gradient = .false.
        end if
        if(associated(b%grad))then
           call b%grad%deallocate(); deallocate(b%grad); nullify(b%grad)
           b%owns_gradient = .false.
        end if
     end do
     call cpu_time(t1)
     t_revA = (t1 - t0) / real(N_REV) * 1000.0

     call W%deallocate()
     call x%deallocate()
     call b%deallocate()

     !-------------------------------------------------------------
     ! Scenario B: loss = sum( tanh(W2 * (W1 * x + b1) + b2) )
     !   W1, W2: non-sample-dependent [d, 1]
     !   x: sample-dependent [d, batch]
     !   b1, b2: non-sample-dependent [d, 1]
     !-------------------------------------------------------------
     call W1%allocate(array_shape=[d, 1])
     W1%is_sample_dependent = .false.
     W1%is_temporary = .false.
     call W1%set_requires_grad(.true.)
     call random_number(W1%val)
     W1%val = W1%val * 0.1_real32

     call W2%allocate(array_shape=[d, 1])
     W2%is_sample_dependent = .false.
     W2%is_temporary = .false.
     call W2%set_requires_grad(.true.)
     call random_number(W2%val)
     W2%val = W2%val * 0.1_real32

     call x%allocate(array_shape=[d, batch])
     x%is_sample_dependent = .true.
     x%is_temporary = .false.
     call x%set_requires_grad(.true.)
     call random_number(x%val)

     call b1%allocate(array_shape=[d, 1])
     b1%is_sample_dependent = .false.
     b1%is_temporary = .false.
     call b1%set_requires_grad(.true.)
     call random_number(b1%val)
     b1%val = b1%val * 0.01_real32

     call b2%allocate(array_shape=[d, 1])
     b2%is_sample_dependent = .false.
     b2%is_temporary = .false.
     call b2%set_requires_grad(.true.)
     call random_number(b2%val)
     b2%val = b2%val * 0.01_real32

     checksum_b = 0.0_real32
     first_pass = .true.

     !--- Backward B ---
     call cpu_time(t0)
     do r = 1, N_REV
        h => W1 * x + b1
        y2 => tanh(W2 * h + b2)
        loss2 => sum(y2)
        loss2%is_temporary = .false.
        call loss2%grad_reverse(reset_graph=.true.)

        if(first_pass)then
           ref_checksum_b = 0.0_real32
           if(associated(W1%grad)) ref_checksum_b = ref_checksum_b + sum(W1%grad%val)
           if(associated(W2%grad)) ref_checksum_b = ref_checksum_b + sum(W2%grad%val)
           if(associated(x%grad))  ref_checksum_b = ref_checksum_b + sum(x%grad%val)
           if(associated(b1%grad)) ref_checksum_b = ref_checksum_b + sum(b1%grad%val)
           if(associated(b2%grad)) ref_checksum_b = ref_checksum_b + sum(b2%grad%val)
           first_pass = .false.
        end if

        if(associated(W1%grad)) checksum_b = checksum_b + sum(W1%grad%val)
        if(associated(W2%grad)) checksum_b = checksum_b + sum(W2%grad%val)
        if(associated(x%grad))  checksum_b = checksum_b + sum(x%grad%val)
        if(associated(b1%grad)) checksum_b = checksum_b + sum(b1%grad%val)
        if(associated(b2%grad)) checksum_b = checksum_b + sum(b2%grad%val)

        call loss2%nullify_graph()
        if(associated(W1%grad))then
           call W1%grad%deallocate(); deallocate(W1%grad); nullify(W1%grad)
           W1%owns_gradient = .false.
        end if
        if(associated(W2%grad))then
           call W2%grad%deallocate(); deallocate(W2%grad); nullify(W2%grad)
           W2%owns_gradient = .false.
        end if
        if(associated(x%grad))then
           call x%grad%deallocate(); deallocate(x%grad); nullify(x%grad)
           x%owns_gradient = .false.
        end if
        if(associated(b1%grad))then
           call b1%grad%deallocate(); deallocate(b1%grad); nullify(b1%grad)
           b1%owns_gradient = .false.
        end if
        if(associated(b2%grad))then
           call b2%grad%deallocate(); deallocate(b2%grad); nullify(b2%grad)
           b2%owns_gradient = .false.
        end if
     end do
     call cpu_time(t1)
     t_revB = (t1 - t0) / real(N_REV) * 1000.0

     call W1%deallocate()
     call W2%deallocate()
     call x%deallocate()
     call b1%deallocate()
     call b2%deallocate()

     !-------------------------------------------------------------
     ! Scenario C: loss = sum( (a1 + a2)^2 )
     !   a1, a2: non-sample-dependent [d, 1] but upstream_grad
     !   comes from sample-dependent context via broadcasting.
     !   This isolates accumulate_gradient_single with batch samples.
     !-------------------------------------------------------------
     call a1%allocate(array_shape=[d, batch])
     a1%is_sample_dependent = .true.
     a1%is_temporary = .false.
     call a1%set_requires_grad(.true.)
     call random_number(a1%val)
     a1%val = a1%val * 0.1_real32

     call a2%allocate(array_shape=[d, batch])
     a2%is_sample_dependent = .true.
     a2%is_temporary = .false.
     call a2%set_requires_grad(.true.)
     call random_number(a2%val)
     a2%val = a2%val * 0.1_real32

     checksum_c = 0.0_real32
     first_pass = .true.

     !--- Backward C ---
     call cpu_time(t0)
     do r = 1, N_REV
        y => (a1 + a2) ** 2._real32
        loss => sum(y)
        loss%is_temporary = .false.
        call loss%grad_reverse(reset_graph=.true.)

        if(first_pass)then
           ref_checksum_c = 0.0_real32
           if(associated(a1%grad)) ref_checksum_c = ref_checksum_c + sum(a1%grad%val)
           if(associated(a2%grad)) ref_checksum_c = ref_checksum_c + sum(a2%grad%val)
           first_pass = .false.
        end if

        if(associated(a1%grad)) checksum_c = checksum_c + sum(a1%grad%val)
        if(associated(a2%grad)) checksum_c = checksum_c + sum(a2%grad%val)

        call loss%nullify_graph()
        if(associated(a1%grad))then
           call a1%grad%deallocate(); deallocate(a1%grad); nullify(a1%grad)
           a1%owns_gradient = .false.
        end if
        if(associated(a2%grad))then
           call a2%grad%deallocate(); deallocate(a2%grad); nullify(a2%grad)
           a2%owns_gradient = .false.
        end if
     end do
     call cpu_time(t1)
     t_revC = (t1 - t0) / real(N_REV) * 1000.0

     call a1%deallocate()
     call a2%deallocate()

     write(*,'(I5,I6,3F13.3)') &
          d, batch, t_revA, t_revB, t_revC
  end do

  write(*,'(A)') repeat('-', 56)
  write(*,'(A,E16.6)') "Ref checksum A (first pass): ", ref_checksum_a
  write(*,'(A,E16.6)') "Ref checksum B (first pass): ", ref_checksum_b
  write(*,'(A,E16.6)') "Ref checksum C (first pass): ", ref_checksum_c
  write(*,'(A)') ""

end program test_benchmark_reverse
