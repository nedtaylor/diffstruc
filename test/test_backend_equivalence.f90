program test_backend_equivalence
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  use coreutils, only: real32
  use diffstruc
  use diffstruc__backend_linalg, only: &
       backend_available, backend_name, reset_backend_mode, set_backend_mode, &
       diffstruc_backend_accelerate, diffstruc_backend_legacy, diffstruc_backend_metal
  implicit none

  integer, parameter :: MAX_BACKENDS = 3
  real(real32), parameter :: atol = 3.0e-4_real32
  real(real32), parameter :: rtol = 3.0e-4_real32
  integer :: backend_modes(MAX_BACKENDS), num_backends

  call collect_backends(backend_modes, num_backends)
  if(num_backends.lt.2)then
     write(*,'(A)') 'backend equivalence: no alternate accelerated backend available; skipping'
     stop 0
  end if

  call check_type_left_case(backend_modes, num_backends)
  call check_type_right_case(backend_modes, num_backends)
  call check_real_left_case(backend_modes, num_backends)
  call check_real_right_case(backend_modes, num_backends)
  call reset_backend_mode()

  write(*,'(A,I0,A)') 'backend equivalence checks passed for ', num_backends, ' backends'

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
  end subroutine collect_backends

  subroutine check_type_left_case(modes, backend_count)
    implicit none
    integer, intent(in) :: modes(:), backend_count
    integer, parameter :: out_d = 96, in_d = 80, batch = 48
    integer :: backend_index
    real(real32), allocatable :: w_seed(:,:), x_seed(:,:)
    real(real32), allocatable :: y_ref(:,:), w_grad_ref(:,:), x_grad_ref(:,:)
    real(real32), allocatable :: y_val(:,:), w_grad(:,:), x_grad(:,:)

    allocate(w_seed(out_d, in_d), x_seed(in_d, batch))
    call fill_matrix(w_seed, 0.013_real32)
    call fill_matrix(x_seed, 0.019_real32)

    do backend_index = 1, backend_count
       call run_type_left_case(modes(backend_index), w_seed, x_seed, y_val, w_grad, x_grad)
       if(backend_index.eq.1)then
          allocate(y_ref(size(y_val,1), size(y_val,2)))
          allocate(w_grad_ref(size(w_grad,1), size(w_grad,2)))
          allocate(x_grad_ref(size(x_grad,1), size(x_grad,2)))
          y_ref = y_val
          w_grad_ref = w_grad
          x_grad_ref = x_grad
       else
          call assert_close('type-left forward', modes(backend_index), y_val, y_ref)
          call assert_close('type-left left grad', modes(backend_index), w_grad, w_grad_ref)
          call assert_close('type-left right grad', modes(backend_index), x_grad, x_grad_ref)
       end if
       deallocate(y_val, w_grad, x_grad)
    end do

    deallocate(y_ref, w_grad_ref, x_grad_ref, w_seed, x_seed)
  end subroutine check_type_left_case

  subroutine check_type_right_case(modes, backend_count)
    implicit none
    integer, intent(in) :: modes(:), backend_count
    integer, parameter :: in_d = 72, out_d = 56, batch = 40
    integer :: backend_index
    real(real32), allocatable :: x_seed(:,:), w_seed(:,:)
    real(real32), allocatable :: y_ref(:,:), x_grad_ref(:,:), w_grad_ref(:,:)
    real(real32), allocatable :: y_val(:,:), x_grad(:,:), w_grad(:,:)

    allocate(x_seed(in_d, batch), w_seed(in_d, out_d))
    call fill_matrix(x_seed, 0.017_real32)
    call fill_matrix(w_seed, 0.023_real32)

    do backend_index = 1, backend_count
       call run_type_right_case(modes(backend_index), x_seed, w_seed, y_val, x_grad, w_grad)
       if(backend_index.eq.1)then
          allocate(y_ref(size(y_val,1), size(y_val,2)))
          allocate(x_grad_ref(size(x_grad,1), size(x_grad,2)))
          allocate(w_grad_ref(size(w_grad,1), size(w_grad,2)))
          y_ref = y_val
          x_grad_ref = x_grad
          w_grad_ref = w_grad
       else
          call assert_close('type-right forward', modes(backend_index), y_val, y_ref)
          call assert_close('type-right left grad', modes(backend_index), x_grad, x_grad_ref)
          call assert_close('type-right right grad', modes(backend_index), w_grad, w_grad_ref)
       end if
       deallocate(y_val, x_grad, w_grad)
    end do

    deallocate(y_ref, x_grad_ref, w_grad_ref, x_seed, w_seed)
  end subroutine check_type_right_case

  subroutine check_real_left_case(modes, backend_count)
    implicit none
    integer, intent(in) :: modes(:), backend_count
    integer, parameter :: out_d = 88, in_d = 64, batch = 44
    integer :: backend_index
    real(real32), allocatable :: w_seed(:,:), x_seed(:,:)
    real(real32), allocatable :: y_ref(:,:), x_grad_ref(:,:)
    real(real32), allocatable :: y_val(:,:), x_grad(:,:)

    allocate(w_seed(out_d, in_d), x_seed(in_d, batch))
    call fill_matrix(w_seed, 0.029_real32)
    call fill_matrix(x_seed, 0.031_real32)

    do backend_index = 1, backend_count
       call run_real_left_case(modes(backend_index), w_seed, x_seed, y_val, x_grad)
       if(backend_index.eq.1)then
          allocate(y_ref(size(y_val,1), size(y_val,2)))
          allocate(x_grad_ref(size(x_grad,1), size(x_grad,2)))
          y_ref = y_val
          x_grad_ref = x_grad
       else
          call assert_close('real-left forward', modes(backend_index), y_val, y_ref)
          call assert_close('real-left grad', modes(backend_index), x_grad, x_grad_ref)
       end if
       deallocate(y_val, x_grad)
    end do

    deallocate(y_ref, x_grad_ref, w_seed, x_seed)
  end subroutine check_real_left_case

  subroutine check_real_right_case(modes, backend_count)
    implicit none
    integer, intent(in) :: modes(:), backend_count
    integer, parameter :: in_d = 68, out_d = 52, batch = 36
    integer :: backend_index
    real(real32), allocatable :: x_seed(:,:), w_seed(:,:)
    real(real32), allocatable :: y_ref(:,:), x_grad_ref(:,:)
    real(real32), allocatable :: y_val(:,:), x_grad(:,:)

    allocate(x_seed(in_d, batch), w_seed(in_d, out_d))
    call fill_matrix(x_seed, 0.037_real32)
    call fill_matrix(w_seed, 0.041_real32)

    do backend_index = 1, backend_count
       call run_real_right_case(modes(backend_index), x_seed, w_seed, y_val, x_grad)
       if(backend_index.eq.1)then
          allocate(y_ref(size(y_val,1), size(y_val,2)))
          allocate(x_grad_ref(size(x_grad,1), size(x_grad,2)))
          y_ref = y_val
          x_grad_ref = x_grad
       else
          call assert_close('real-right forward', modes(backend_index), y_val, y_ref)
          call assert_close('real-right grad', modes(backend_index), x_grad, x_grad_ref)
       end if
       deallocate(y_val, x_grad)
    end do

    deallocate(y_ref, x_grad_ref, x_seed, w_seed)
  end subroutine check_real_right_case

  subroutine run_type_left_case(backend_mode, w_seed, x_seed, y_out, w_grad_out, x_grad_out)
    implicit none
    integer, intent(in) :: backend_mode
    real(real32), intent(in) :: w_seed(:,:), x_seed(:,:)
    real(real32), allocatable, intent(out) :: y_out(:,:), w_grad_out(:,:), x_grad_out(:,:)
    type(array_type) :: w_ad, x_ad
    type(array_type), pointer :: y_ad, loss

    call set_backend_mode(backend_mode)
    call init_matrix_operand(w_ad, w_seed, requires_grad=.true.)
    call init_vector_batch_operand(x_ad, x_seed, requires_grad=.true.)

    y_ad => matmul(w_ad, x_ad)
    y_ad%is_temporary = .false.
    allocate(y_out(size(y_ad%val,1), size(y_ad%val,2)))
    y_out = y_ad%val

    loss => sum(y_ad, dim=1)
    loss%is_temporary = .false.
    call loss%grad_reverse(reset_graph=.true.)

    allocate(w_grad_out(size(w_ad%grad%val,1), size(w_ad%grad%val,2)))
    allocate(x_grad_out(size(x_ad%grad%val,1), size(x_ad%grad%val,2)))
    w_grad_out = w_ad%grad%val
    x_grad_out = x_ad%grad%val

    call loss%nullify_graph(ignore_ownership=.true.)
    call cleanup_array(w_ad)
    call cleanup_array(x_ad)
  end subroutine run_type_left_case

  subroutine run_type_right_case(backend_mode, x_seed, w_seed, y_out, x_grad_out, w_grad_out)
    implicit none
    integer, intent(in) :: backend_mode
    real(real32), intent(in) :: x_seed(:,:), w_seed(:,:)
    real(real32), allocatable, intent(out) :: y_out(:,:), x_grad_out(:,:), w_grad_out(:,:)
    type(array_type) :: x_ad, w_ad
    type(array_type), pointer :: y_ad, loss

    call set_backend_mode(backend_mode)
    call init_vector_batch_operand(x_ad, x_seed, requires_grad=.true.)
    call init_matrix_operand(w_ad, w_seed, requires_grad=.true.)

    y_ad => matmul(x_ad, w_ad)
    y_ad%is_temporary = .false.
    allocate(y_out(size(y_ad%val,1), size(y_ad%val,2)))
    y_out = y_ad%val

    loss => sum(y_ad, dim=1)
    loss%is_temporary = .false.
    call loss%grad_reverse(reset_graph=.true.)

    allocate(x_grad_out(size(x_ad%grad%val,1), size(x_ad%grad%val,2)))
    allocate(w_grad_out(size(w_ad%grad%val,1), size(w_ad%grad%val,2)))
    x_grad_out = x_ad%grad%val
    w_grad_out = w_ad%grad%val

    call loss%nullify_graph(ignore_ownership=.true.)
    call cleanup_array(x_ad)
    call cleanup_array(w_ad)
  end subroutine run_type_right_case

  subroutine run_real_left_case(backend_mode, w_seed, x_seed, y_out, x_grad_out)
    implicit none
    integer, intent(in) :: backend_mode
    real(real32), intent(in) :: w_seed(:,:), x_seed(:,:)
    real(real32), allocatable, intent(out) :: y_out(:,:), x_grad_out(:,:)
    type(array_type) :: x_ad
    type(array_type), pointer :: y_ad, loss

    call set_backend_mode(backend_mode)
    call init_vector_batch_operand(x_ad, x_seed, requires_grad=.true.)

    y_ad => matmul(w_seed, x_ad)
    y_ad%is_temporary = .false.
    allocate(y_out(size(y_ad%val,1), size(y_ad%val,2)))
    y_out = y_ad%val

    loss => sum(y_ad, dim=1)
    loss%is_temporary = .false.
    call loss%grad_reverse(reset_graph=.true.)

    allocate(x_grad_out(size(x_ad%grad%val,1), size(x_ad%grad%val,2)))
    x_grad_out = x_ad%grad%val

    call loss%nullify_graph(ignore_ownership=.true.)
    call cleanup_array(x_ad)
  end subroutine run_real_left_case

  subroutine run_real_right_case(backend_mode, x_seed, w_seed, y_out, x_grad_out)
    implicit none
    integer, intent(in) :: backend_mode
    real(real32), intent(in) :: x_seed(:,:), w_seed(:,:)
    real(real32), allocatable, intent(out) :: y_out(:,:), x_grad_out(:,:)
    type(array_type) :: x_ad
    type(array_type), pointer :: y_ad, loss

    call set_backend_mode(backend_mode)
    call init_vector_batch_operand(x_ad, x_seed, requires_grad=.true.)

    y_ad => matmul(x_ad, w_seed)
    y_ad%is_temporary = .false.
    allocate(y_out(size(y_ad%val,1), size(y_ad%val,2)))
    y_out = y_ad%val

    loss => sum(y_ad, dim=1)
    loss%is_temporary = .false.
    call loss%grad_reverse(reset_graph=.true.)

    allocate(x_grad_out(size(x_ad%grad%val,1), size(x_ad%grad%val,2)))
    x_grad_out = x_ad%grad%val

    call loss%nullify_graph(ignore_ownership=.true.)
    call cleanup_array(x_ad)
  end subroutine run_real_right_case

  subroutine init_matrix_operand(array, values, requires_grad)
    implicit none
    type(array_type), intent(out) :: array
    real(real32), intent(in) :: values(:,:)
    logical, intent(in) :: requires_grad

    call array%allocate(array_shape=[size(values), 1])
    array%shape = shape(values)
    array%is_sample_dependent = .false.
    array%is_temporary = .false.
    call array%set_requires_grad(requires_grad)
    array%val(:,1) = reshape(values, [size(values)])
  end subroutine init_matrix_operand

  subroutine init_vector_batch_operand(array, values, requires_grad)
    implicit none
    type(array_type), intent(out) :: array
    real(real32), intent(in) :: values(:,:)
    logical, intent(in) :: requires_grad

    call array%allocate(array_shape=[size(values,1), size(values,2)])
    array%is_sample_dependent = .true.
    array%is_temporary = .false.
    call array%set_requires_grad(requires_grad)
    array%val = values
  end subroutine init_vector_batch_operand

  subroutine cleanup_array(array)
    implicit none
    type(array_type), intent(inout) :: array

    if(associated(array%grad))then
       call array%grad%deallocate()
       deallocate(array%grad)
       nullify(array%grad)
    end if
    call array%deallocate()
  end subroutine cleanup_array

  subroutine fill_matrix(values, scale)
    implicit none
    real(real32), intent(out) :: values(:,:)
    real(real32), intent(in) :: scale
    integer :: i, j

    do j = 1, size(values, 2)
       do i = 1, size(values, 1)
          values(i,j) = sin(scale * real(i, real32)) + cos(scale * 0.5_real32 * real(j, real32))
       end do
    end do
  end subroutine fill_matrix

  subroutine assert_close(label, backend_mode, observed, reference)
    implicit none
    character(len=*), intent(in) :: label
    integer, intent(in) :: backend_mode
    real(real32), intent(in) :: observed(:,:), reference(:,:)
    real(real32) :: max_abs, max_ref, tolerance

    if(any(shape(observed).ne.shape(reference)))then
       write(*,'(A,1X,A,1X,A)') 'shape mismatch for', trim(label), trim(backend_name(backend_mode))
       stop 1
    end if
    if(any(ieee_is_nan(observed)))then
       write(*,'(A,1X,A,1X,A)') 'NaN detected for', trim(label), trim(backend_name(backend_mode))
       stop 1
    end if

    max_abs = maxval(abs(observed - reference))
    max_ref = max(1.0_real32, maxval(abs(reference)))
    tolerance = atol + rtol * max_ref
    if(max_abs.gt.tolerance)then
       write(*,'(A,1X,A,1X,A,1X,A,1X,ES12.5,1X,A,1X,ES12.5)') &
            'equivalence failure for', trim(label), 'backend', trim(backend_name(backend_mode)), &
            max_abs, 'tolerance', tolerance
       stop 1
    end if
  end subroutine assert_close

end program test_backend_equivalence
