program test_matmul_mixed
  use coreutils, only: real32
  use diffstruc
  implicit none

  type(array_type) :: inputs, weights
  type(array_type), pointer :: outputs
  real(real32), parameter :: tol = 1.0e-5_real32
  real(real32), dimension(2, 3) :: weight_matrix
  real(real32), dimension(2, 2) :: expected_input_grad
  real(real32), dimension(6) :: expected_weight_grad

  call inputs%allocate(array_shape=[2, 2])
  inputs%is_sample_dependent = .true.
  inputs%is_temporary = .false.
  call inputs%set_requires_grad(.true.)
  inputs%val(:,1) = [1.0_real32, 2.0_real32]
  inputs%val(:,2) = [3.0_real32, 4.0_real32]

  weight_matrix = reshape([ &
       10.0_real32, 40.0_real32, &
       20.0_real32, 50.0_real32, &
       30.0_real32, 60.0_real32], [2, 3])

  call weights%allocate(array_shape=[size(weight_matrix), 1])
  weights%shape = shape(weight_matrix)
  weights%is_sample_dependent = .false.
  weights%is_temporary = .false.
  call weights%set_requires_grad(.true.)
  weights%val(:,1) = reshape(weight_matrix, [size(weight_matrix)])

  outputs => matmul(inputs, weights)
  outputs%is_temporary = .false.
  call outputs%grad_reverse(reset_graph=.true.)

  if(.not. associated(inputs%grad)) then
     write(*,*) 'inputs gradient was not allocated'
     error stop 1
  end if
  if(.not. associated(weights%grad)) then
     write(*,*) 'weights gradient was not allocated'
     error stop 1
  end if

  expected_input_grad(:,1) = [60.0_real32, 150.0_real32]
  expected_input_grad(:,2) = [60.0_real32, 150.0_real32]
  expected_weight_grad = [ &
       4.0_real32, 6.0_real32, 4.0_real32, 6.0_real32, 4.0_real32, 6.0_real32]

  if(any(abs(inputs%grad%val - expected_input_grad) > tol)) then
     write(*,*) 'unexpected input gradient'
     write(*,*) 'expected:'
     write(*,'(2F10.4)') expected_input_grad(:,1)
     write(*,'(2F10.4)') expected_input_grad(:,2)
     write(*,*) 'actual:'
     write(*,'(2F10.4)') inputs%grad%val(:,1)
     write(*,'(2F10.4)') inputs%grad%val(:,2)
     error stop 1
  end if

  if(any(abs(weights%grad%val(:,1) - expected_weight_grad) > tol)) then
     write(*,*) 'unexpected weight gradient'
     write(*,*) 'expected:', expected_weight_grad
     write(*,*) 'actual:  ', weights%grad%val(:,1)
     error stop 1
  end if

  write(*,*) 'matmul mixed gradient test passed'

  call outputs%nullify_graph(ignore_ownership=.true.)
  if(associated(inputs%grad)) then
     call inputs%grad%deallocate()
     deallocate(inputs%grad)
     nullify(inputs%grad)
  end if
  if(associated(weights%grad)) then
     call weights%grad%deallocate()
     deallocate(weights%grad)
     nullify(weights%grad)
  end if
  call inputs%deallocate()
  call weights%deallocate()
  call outputs%deallocate()
  deallocate(outputs)

end program test_matmul_mixed
