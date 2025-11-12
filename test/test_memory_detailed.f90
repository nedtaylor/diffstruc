program test_memory_detailed
  !! Detailed memory leak test with explicit cleanup verification
  use coreutils, only: real32
  use diffstruc
  implicit none

  type(array_type) :: x, y, f
  type(array_type), pointer :: temp, xgrad, xgradgrad
  integer :: i, n_iterations
  logical :: do_forward, do_second_order


  do_forward = .true.
  do_second_order = .true.
  n_iterations = 10000
  x%is_temporary = .false.
  y%is_temporary = .false.
  f%is_temporary = .false.

  write(*,*) "=== Detailed Memory Test ==="
  write(*,*) "Testing different cleanup strategies..."
  write(*,*) ""

  ! Test 1: Assignment with reset_graph
  write(*,*) "Test 1: Using f = expr with reset_graph()"
  call x%allocate(array_shape=[1000, 100])
  call y%allocate(array_shape=[1000, 100])
  x%val = 1.0_real32
  y%val = 2.0_real32
  call x%set_requires_grad(.true.)
  call y%set_requires_grad(.true.)

  do i = 1, n_iterations
     ! This creates a pointer, assigns it to f, but the pointer is lost
     temp => x**2 + y * x + exp(x * 0.01_real32)
     call f%assign_and_deallocate_source(temp)
     f%is_temporary = .false.
     call f%grad_reverse(record_graph=.false., reset_graph=.true.)
     call f%nullify_graph()
     call f%deallocate()
     if (mod(i, 10) == 0) then
        write(*,'(A,I0)') "  Iteration ", i
     end if
  end do
  call f%print_graph()
  !write(*,*) "  Final gradient x:", x%grad%val(1,1)
  write(*,*) f%owns_left_operand, f%owns_right_operand, f%owns_gradient
  call f%nullify_graph()
  write(*,*) f%owns_left_operand, f%owns_right_operand, f%owns_gradient
  call x%deallocate()
  call y%deallocate()
  call f%deallocate()
  write(*,*) "  Test 1 complete"
  write(*,*) ""

  ! Test 2: Pointer assignment with explicit deallocation
  write(*,*) "Test 2: Using temp => expr with explicit deallocate"
  call x%allocate(array_shape=[1000, 100])
  call y%allocate(array_shape=[1000, 100])
  x%val = 1.0_real32
  y%val = 2.0_real32
  x%id = 12
  y%id = 13
  call x%set_requires_grad(.true.)
  call y%set_requires_grad(.true.)

  do i = 1, n_iterations
     ! Use pointer and explicitly deallocate
     !allocate(temp)
     temp => x**2 + y * x + exp(x * 0.01_real32)
     temp%is_temporary = .false.
     temp%id = 15
     if(do_forward)then
        xgrad => temp%grad_forward(x)
        xgrad%is_temporary = .false.
        if(do_second_order)then
           xgradgrad => xgrad%grad_forward(x)
           xgradgrad%is_temporary = .false.
        end if
     else
        call temp%grad_reverse(record_graph=.true., reset_graph=.true.)
     end if

     ! Explicit cleanup of temp (THIS IS KEY TO AVOIDING LEAKS)
     call temp%nullify_graph()
     call temp%deallocate()
     if(do_forward)then
        call xgrad%nullify_graph()
        call xgrad%deallocate()
        if(do_second_order)then
           call xgradgrad%nullify_graph()
           call xgradgrad%deallocate()
        end if
     end if

     deallocate(temp)
     if(do_forward)then
        deallocate(xgrad)
        if(do_second_order)then
           deallocate(xgradgrad)
        end if
     end if

     if (mod(i, 10) == 0) then
        write(*,'(A,I0)') "  Iteration ", i
     end if
  end do
  ! write(*,*) "  Final gradient x:", x%grad%val(1,1)
  call x%deallocate()
  call y%deallocate()
  write(*,*) "  Test 2 complete"
  write(*,*) ""

  ! ! Test 3: Multiple operations with nullify_graph
  ! write(*,*) "Test 3: Multiple operations with nullify_graph()"
  ! call x%allocate(array_shape=[1000, 100])
  ! call y%allocate(array_shape=[1000, 100])
  ! x%val = 1.0_real32
  ! y%val = 2.0_real32
  ! call x%set_requires_grad(.true.)
  ! call y%set_requires_grad(.true.)

  ! do i = 1, n_iterations
  !    f = x**2 + y * x + exp(x * 0.01_real32)!((x**2 + y) * x + y**2) / (x + 1.0_real32)
  !    call f%grad_reverse(record_graph=.false., reset_graph=.false.)

  !    ! Aggressive cleanup
  !    call f%nullify_graph()

  !    if (mod(i, 10) == 0) then
  !       write(*,'(A,I0)') "  Iteration ", i
  !    end if
  ! end do
  ! write(*,*) "associated(x%grad):", associated(x%grad)
  ! !write(*,*) "  Final gradient x:", x%grad%val(1,1)
  ! call x%deallocate()
  ! call y%deallocate()
  ! write(*,*) "  Test 3 complete"
  ! write(*,*) ""

  write(*,*) "=== All Tests Complete ==="
  write(*,*) "Monitor memory usage - it should remain stable."
  write(*,*) "If memory grows during tests, there are still leaks!"

end program test_memory_detailed
