submodule(diffstruc__operations_linalg) diffstruc__operations_linalg_sub
  !! Submodule containing implementations of linear algebra operations
  use coreutils, only: stop_program


contains

!###############################################################################
  module function matmul_arrays(a, b) result(c)
    !! Matrix multiplication of two autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    integer :: s
    character(len=128) :: err_msg
    real(real32), pointer :: temp(:,:)

    if(.not.a%is_sample_dependent)then
       if(size(b%shape).ne.1)then
          write(err_msg,'("Matrix multiplication not implemented for array ''b'' &
               &rank: ",I0)') size(b%shape)
          call stop_program(err_msg)
          return
       end if
       c => a%create_result(array_shape=[a%shape(1), size(b%val,2)])
       temp(1:a%shape(1), 1:a%shape(2)) => a%val
       do concurrent(s=1:size(b%val,2))
          c%val(:,s) = matmul(temp, b%val(:,s))
       end do
    elseif(.not.b%is_sample_dependent)then
       if(size(a%shape).ne.1)then
          write(err_msg,'("Matrix multiplication not implemented for array ''a'' &
               &rank: ",I0)') size(a%shape)
          call stop_program(err_msg)
          return
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
    c%get_partial_left_val => get_partial_matmul_left_val
    c%get_partial_right_val => get_partial_matmul_right_val
    if(a%requires_grad .or. b%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%operation = 'matmul'
       c%left_operand => a
       c%right_operand => b
       c%owns_left_operand = a%is_temporary
       c%owns_right_operand = b%is_temporary
    end if
  end function matmul_arrays
!-------------------------------------------------------------------------------
  module function matmul_real2d(a, b) result(c)
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
    c%get_partial_left_val => get_partial_matmul_left_val
    c%get_partial_right_val => get_partial_matmul_right_val
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'matmul_scalar'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
    allocate(b_array)
    b_array%is_sample_dependent = .false.
    b_array%shape = shape(b)
    b_array%requires_grad = .false.
    call b_array%allocate(array_shape=[size(b,1), size(b,2), 1])
    do i = 1, size(b,2)
       b_array%val((i-1)*size(b,1)+1:i*size(b,1), 1) = b(:,i)
    end do
    c%right_operand => b_array
    c%owns_right_operand = .true.
  end function matmul_real2d
!-------------------------------------------------------------------------------
  module function real2d_matmul(a, b) result(c)
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
    c%get_partial_left_val => get_partial_matmul_left_val
    c%get_partial_right_val => get_partial_matmul_right_val
    if(b%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = b%is_forward
       c%operation = 'matmul_scalar'
       c%right_operand => b
       c%owns_right_operand = b%is_temporary
    end if
    allocate(a_array)
    a_array%is_sample_dependent = .false.
    a_array%shape = shape(a)
    a_array%requires_grad = .false.
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

    logical :: right_is_temporary_local
    type(array_type), pointer :: ptr

    right_is_temporary_local = this%right_operand%is_temporary
    this%right_operand%is_temporary = .false.
    if(size(this%right_operand%shape).eq.2)then
       if(this%is_forward)then
          ptr => matmul( upstream_grad, this%right_operand )
       else
          ptr => matmul( upstream_grad, transpose(this%right_operand) )
       end if
    elseif(size(upstream_grad%shape).eq.2)then
       if(this%is_forward)then
          ptr => matmul( upstream_grad, this%right_operand )
       else
          ptr => matmul( transpose(upstream_grad), this%right_operand )
       end if
    else
       ptr => upstream_grad .outer. this%right_operand
    end if
    this%right_operand%is_temporary = right_is_temporary_local
    call output%assign_and_deallocate_source(ptr)

  end function get_partial_matmul_left
!-------------------------------------------------------------------------------
  function get_partial_matmul_right(this, upstream_grad) result(output)
    !! Get partial derivative with respect to right operand of matmul
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    logical :: left_is_temporary_local
    type(array_type), pointer :: ptr

    left_is_temporary_local = this%left_operand%is_temporary
    this%left_operand%is_temporary = .false.
    if(size(this%left_operand%shape).eq.2)then
       if(this%is_forward)then
          ptr => matmul(this%left_operand, upstream_grad)
       else
          ptr => matmul(transpose(this%left_operand), upstream_grad)
       end if
    elseif(size(upstream_grad%shape).eq.2)then
       if(this%is_forward)then
          ptr => matmul(this%left_operand, upstream_grad)
       else
          ptr => matmul(this%left_operand, transpose(upstream_grad))
       end if
    else
       ptr => this%left_operand .outer. upstream_grad
    end if
    this%left_operand%is_temporary = left_is_temporary_local
    call output%assign_and_deallocate_source(ptr)

  end function get_partial_matmul_right
!-------------------------------------------------------------------------------
  subroutine get_partial_matmul_left_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(inout) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: i, j, s, num_elements
    real(real32), pointer :: temp(:,:)

    if(size(this%right_operand%shape).eq.2)then
       if(this%right_operand%is_sample_dependent)then
          do concurrent(s=1:size(upstream_grad,2))
             temp(1:this%right_operand%shape(1), 1:this%right_operand%shape(2)) => &
                  this%right_operand%val(:,s)
             output(:,s) = matmul(upstream_grad(:,s), transpose(temp))
          end do
       else
          temp(1:this%right_operand%shape(1), 1:this%right_operand%shape(2)) => &
               this%right_operand%val(:,1)
          do concurrent(s=1:size(upstream_grad,2))
             output(:,s) = matmul(upstream_grad(:,s), transpose(temp))
          end do
       end if
    else
       num_elements = size(upstream_grad,1)
       if(this%right_operand%is_sample_dependent)then
          do concurrent(s=1:size(upstream_grad,2), i=1:num_elements, &
               j=1:size(this%right_operand%val,1))
             output(i + (j-1)*num_elements,s) = &
                  upstream_grad(i,s) * this%right_operand%val(j,s)
          end do
       else
          do concurrent(s=1:size(upstream_grad,2), i=1:num_elements, &
               j=1:size(this%right_operand%val,1))
             output(i + (j-1)*num_elements,s) = &
                  upstream_grad(i,s) * this%right_operand%val(j,1)
          end do
       end if
    end if

  end subroutine get_partial_matmul_left_val
!-------------------------------------------------------------------------------
  subroutine get_partial_matmul_right_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(inout) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: i, j, s, num_elements
    real(real32), pointer :: temp(:,:)

    if(size(this%left_operand%shape).eq.2)then
       if(this%left_operand%is_sample_dependent)then
          do concurrent(s=1:size(upstream_grad,2))
             temp(1:this%left_operand%shape(1), 1:this%left_operand%shape(2)) => &
                  this%left_operand%val(:,s)
             output(:,s) = matmul(transpose(temp), upstream_grad(:,s))
          end do
       else
          temp(1:this%left_operand%shape(1), 1:this%left_operand%shape(2)) => &
               this%left_operand%val(:,1)
          do concurrent(s=1:size(upstream_grad,2))
             output(:,s) = matmul(transpose(temp), upstream_grad(:,s))
          end do
       end if
    else
       num_elements = size(this%left_operand%val,1)
       if(this%left_operand%is_sample_dependent)then
          do concurrent(s=1:size(upstream_grad,2), i=1:num_elements, &
               j=1:size(upstream_grad,1))
             output(i + (j-1)*num_elements,s) = &
                  this%left_operand%val(i,s) * upstream_grad(j,s)
          end do
       else
          do concurrent(s=1:size(upstream_grad,2), i=1:num_elements, &
               j=1:size(upstream_grad,1))
             output(i + (j-1)*num_elements,s) = &
                  this%left_operand%val(i,1) * upstream_grad(j,s)
          end do
       end if
    end if

  end subroutine get_partial_matmul_right_val
!###############################################################################


!###############################################################################
  module function outer_product_arrays(a, b) result(c)
    !! Outer product of two autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    integer :: i, j, s

    ! check shapes
    if(size(a%shape).ne.1 .or. size(b%shape).ne.1)then
       call stop_program("dot_product_arrays: only 1D arrays supported")
    elseif(size(a%val,2).ne.size(b%val,2))then
       call stop_program("dot_product_arrays: array length mismatch")
    end if

    c => a%create_result(array_shape = [size(a%val,1), size(b%val,1), size(a%val,2)])
    ! outer product 1D array by using shape to swap dimensions
    do concurrent(s=1:size(a%val,2))
       do concurrent(i=1:size(a%val,1), j=1:size(b%val,1))
          c%val(i + (j-1)*size(a%val,1),s) = a%val(i,s) * b%val(j,s)
       end do
    end do

    c%get_partial_left => get_partial_outer_product_left
    c%get_partial_right => get_partial_outer_product_right
    c%is_sample_dependent = a%is_sample_dependent
    if(a%requires_grad .or. b%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%operation = 'outer_product'
       c%left_operand => a
       c%right_operand => b
       c%owns_left_operand = a%is_temporary
       c%owns_right_operand = b%is_temporary
    end if
  end function outer_product_arrays
!-------------------------------------------------------------------------------
  function get_partial_outer_product_left(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    logical :: right_is_temporary_local
    type(array_type), pointer :: ptr

    right_is_temporary_local = this%right_operand%is_temporary
    this%right_operand%is_temporary = .false.
    if(this%is_forward)then
       ptr => upstream_grad .outer. this%right_operand
    else
       ptr => matmul(upstream_grad, this%right_operand)
    end if
    this%right_operand%is_temporary = right_is_temporary_local
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_outer_product_left
!-------------------------------------------------------------------------------
  function get_partial_outer_product_right(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    logical :: left_is_temporary_local
    type(array_type), pointer :: ptr

    left_is_temporary_local = this%left_operand%is_temporary
    this%left_operand%is_temporary = .false.
    if(this%is_forward)then
       ptr => this%left_operand .outer. upstream_grad
    else
       ! mathematically should be ptr => transpose(upstream_grad) .mmul. this%left_operand
       ! but for how we store vectors, this SHOULD BE equivalent
       ptr => matmul(this%left_operand, upstream_grad)
    end if
    this%left_operand%is_temporary = left_is_temporary_local
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_outer_product_right
!###############################################################################


!###############################################################################
  module function dot_product_arrays(a, b) result(c)
    !! Dot product of two autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    integer :: s

    ! check shapes
    if(size(a%shape).ne.1 .or. size(b%shape).ne.1)then
       call stop_program("dot_product_arrays: only 1D arrays supported")
    elseif(any(shape(a%val).ne.shape(b%val)))then
       call stop_program("dot_product_arrays: array length mismatch")
    end if

    c => a%create_result(array_shape = [1, size(a%val,2)])
    do concurrent(s=1:size(a%val,2))
       c%val(1,s) = dot_product(a%val(:,s), b%val(:,s))
    end do

    c%get_partial_left => get_partial_dot_product_left
    c%get_partial_right => get_partial_dot_product_right
    c%is_sample_dependent = a%is_sample_dependent
    if(a%requires_grad .or. b%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%operation = 'dot_product'
       c%left_operand => a
       c%right_operand => b
       c%owns_left_operand = a%is_temporary
       c%owns_right_operand = b%is_temporary
    end if
  end function dot_product_arrays
!-------------------------------------------------------------------------------
  function get_partial_dot_product_left(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    logical :: right_is_temporary_local
    type(array_type), pointer :: ptr

    right_is_temporary_local = this%right_operand%is_temporary
    this%right_operand%is_temporary = .false.
    if(this%is_forward)then
       ptr => dot_product(upstream_grad, this%right_operand)
    else
       ptr => upstream_grad * this%right_operand
    end if
    this%right_operand%is_temporary = right_is_temporary_local
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_dot_product_left
!-------------------------------------------------------------------------------
  function get_partial_dot_product_right(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    logical :: left_is_temporary_local
    type(array_type), pointer :: ptr

    left_is_temporary_local = this%left_operand%is_temporary
    this%left_operand%is_temporary = .false.
    if(this%is_forward)then
       ptr => dot_product(this%left_operand, upstream_grad)
    else
       ptr => upstream_grad * this%left_operand
    end if
    this%left_operand%is_temporary = left_is_temporary_local
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_dot_product_right
!###############################################################################


!###############################################################################
  module function transpose_array(a) result(c)
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
       c%operation = 'transpose'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function transpose_array
!-------------------------------------------------------------------------------
  function get_partial_transpose_left(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    ptr => transpose(upstream_grad)
    call output%assign_and_deallocate_source(ptr)
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
