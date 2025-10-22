submodule(diffstruc__operations_linalg) diffstruc__operations_linalg_sub
  !! Submodule containing implementations of linear algebra operations
  use coreutils, only: stop_program


contains

!###############################################################################
  function matmul_arrays(a, b) result(c)
    !! Matrix multiplication of two autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    integer :: s
    real(real32), pointer :: temp(:,:)

    if(.not.a%is_sample_dependent)then
       if(size(b%shape).ne.1)then
          call stop_program( &
               'Matrix multiplication not implemented for these shapes yet' )
       end if
       c => a%create_result(array_shape=[a%shape(1), size(b%val,2)])
       temp(1:a%shape(1), 1:a%shape(2)) => a%val
       do concurrent(s=1:size(b%val,2))
          c%val(:,s) = matmul(temp, b%val(:,s))
       end do
    elseif(.not.b%is_sample_dependent)then
       if(size(a%shape).ne.1)then
          call stop_program( &
               'Matrix multiplication not implemented for these shapes yet' )
       end if
       c => b%create_result(array_shape=[b%shape(2), size(a%val,2)])
       temp(1:b%shape(1), 1:b%shape(2)) => b%val
       do concurrent(s=1:size(a%val,2))
          c%val(:,s) = matmul(a%val(:,s), temp)
       end do
    else
       write(0,*) "NOT SURE WHAT TO DO YET"
       stop 0
    end if

    c%is_sample_dependent = .true.
    c%get_partial_left => get_partial_matmul_left
    c%get_partial_right => get_partial_matmul_right
    if(a%requires_grad .or. b%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%is_leaf = .false.
       c%operation = 'matmul'
       c%left_operand => a
       c%right_operand => b
    end if
  end function matmul_arrays
!-------------------------------------------------------------------------------
  function matmul_real2d(a, b) result(c)
    !! Matrix multiplication of a real array and an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), dimension(:,:), intent(in) :: b
    type(array_type), pointer :: c
    type(array_type), pointer :: b_array

    integer :: s, i

    c => a%create_result(array_shape = [size(b,2), size(a%val,2)])
    do concurrent(s=1:size(a%val,2))
       c%val(:,s) = matmul(a%val(:,s), b)
    end do

    c%is_sample_dependent = a%is_sample_dependent
    c%get_partial_left => get_partial_matmul_left
    c%get_partial_right => get_partial_matmul_right
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'matmul_scalar'
       c%left_operand => a
    end if
    allocate(b_array)
    b_array%is_sample_dependent = .false.
    b_array%shape = shape(b)
    b_array%requires_grad = .false.
    b_array%is_leaf = .false.
    call b_array%allocate(array_shape=[size(b,1), size(b,2), 1])
    do i = 1, size(b,2)
       b_array%val((i-1)*size(b,1)+1:i*size(b,1), 1) = b(:,i)
    end do
    c%right_operand => b_array
    c%owns_right_operand = .true.
  end function matmul_real2d
!-------------------------------------------------------------------------------
  function real2d_matmul(a, b) result(c)
    !! Matrix multiplication of two autodiff arrays
    implicit none
    real(real32), dimension(:,:), intent(in) :: a
    class(array_type), intent(in), target :: b
    type(array_type), pointer :: c
    type(array_type), pointer :: a_array

    integer :: s, i

    c => b%create_result(array_shape = [size(a,1), size(b%val,2)])
    do concurrent(s=1:size(b%val,2))
       c%val(:,s) = matmul(a, b%val(:,s))
    end do

    c%is_sample_dependent = b%is_sample_dependent
    c%get_partial_left => get_partial_matmul_left
    c%get_partial_right => get_partial_matmul_right
    if(b%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = b%is_forward
       c%is_leaf = .false.
       c%operation = 'matmul_scalar'
       c%right_operand => b
    end if
    allocate(a_array)
    a_array%is_sample_dependent = .false.
    a_array%shape = shape(a)
    a_array%requires_grad = .false.
    a_array%is_leaf = .false.
    call a_array%allocate(array_shape=[size(a,1), size(a,2), 1])
    do i = 1, size(a,2)
       a_array%val((i-1)*size(a,1)+1:i*size(a,1), 1) = a(:,i)
    end do
    c%left_operand => a_array
    c%owns_left_operand = .true.
  end function real2d_matmul
!-------------------------------------------------------------------------------
  function get_partial_matmul_left(this, upstream_grad) result(output)
    !! Get partial derivative with respect to left operand of matmul
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    type(array_type), pointer :: ptr

    if(size(this%right_operand%shape).eq.2)then
       if(this%is_forward)then
          ptr => upstream_grad .mmul. this%right_operand
       else
          ptr => upstream_grad .mmul. transpose(this%right_operand)
          ptr%owns_right_operand = .true.
       end if
    elseif(size(upstream_grad%shape).eq.2)then
       if(this%is_forward)then
          ptr => upstream_grad .mmul. this%right_operand
       else
          ptr => transpose(upstream_grad) .mmul. this%right_operand
          ptr%owns_left_operand = .true.
       end if
    else
       ptr => upstream_grad .outer. this%right_operand
    end if
    call output%assign_and_deallocate_source(ptr)

  end function get_partial_matmul_left
!-------------------------------------------------------------------------------
  function get_partial_matmul_right(this, upstream_grad) result(output)
    !! Get partial derivative with respect to right operand of matmul
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    type(array_type), pointer :: ptr

    if(size(this%left_operand%shape).eq.2)then
       if(this%is_forward)then
          ptr => this%left_operand .mmul. upstream_grad
       else
          ptr => transpose(this%left_operand) .mmul. upstream_grad
          ptr%owns_left_operand = .true.
       end if
    elseif(size(upstream_grad%shape).eq.2)then
       if(this%is_forward)then
          ptr => this%left_operand .mmul. upstream_grad
       else
          ptr => this%left_operand .mmul. transpose(upstream_grad)
          ptr%owns_right_operand = .true.
       end if
    else
       ptr => this%left_operand .outer. upstream_grad
    end if
    call output%assign_and_deallocate_source(ptr)

  end function get_partial_matmul_right
!###############################################################################


!###############################################################################
  function outer_product_arrays(a, b) result(c)
    !! Outer product of two autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    integer :: i, j, s

    c => a%create_result(array_shape = [size(a%val,1), size(b%val,1), size(a%val,2)])
    ! outer product 1D array by using shape to swap dimensions
    do concurrent(s=1:size(a%val,2))
       do concurrent(i=1:size(a%val,1), j=1:size(b%val,1))
          c%val(i + (j-1)*size(a%val,1),s) = a%val(i,s) * b%val(j,s)
       end do
    end do

    if(a%requires_grad .or. b%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%is_leaf = .false.
       c%operation = 'outer_product'
       c%left_operand => a
       c%right_operand => b
    end if
  end function outer_product_arrays
!-------------------------------------------------------------------------------
!###############################################################################


!###############################################################################
  function transpose_array(a) result(c)
    !! Transpose an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    integer :: i, j, s

    if(size(a%shape) .ne. 2)then
       write(*,*) "ashape", a%shape
       call stop_program("transpose_array: only 2D arrays can be transposed")
    end if
    c => a%create_result(array_shape=[a%shape(2), a%shape(1), size(a%val,2)])
    ! transpose 1D array by using shape to swap dimensions
    do concurrent(s=1:size(a%val,2))
       do concurrent(i=1:a%shape(1), j=1:a%shape(2))
          c%val( (i-1)*a%shape(2) + j, s) = a%val( (j-1)*a%shape(1) + i, s)
       end do
    end do

    c%get_partial_left => get_partial_transpose_left
    ! c%get_partial_right => get_partial_transpose_right
    c%is_sample_dependent = a%is_sample_dependent
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'transpose'
       c%left_operand => a
    end if
  end function transpose_array
!-------------------------------------------------------------------------------
  function get_partial_transpose_left(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    if(this%requires_grad) then
       output = transpose(upstream_grad)
    else
       output%val = transpose(upstream_grad%val)
    end if

  end function get_partial_transpose_left
! !-------------------------------------------------------------------------------
! !   function get_partial_transpose_right(this, upstream_grad) result(output)
! !     class(array_type), intent(inout) :: this
! !     type(array_type), intent(in) :: upstream_grad
! !     type(array_type) :: output

! !     output = transpose(this%left_operand)

! !   end function get_partial_transpose_right
!###############################################################################



end submodule diffstruc__operations_linalg_sub
