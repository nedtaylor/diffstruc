module diffstruc__operations_reduction
  !! This module contains reduction operations for the diffstruc library.
  use coreutils, only: real32, stop_program
  use diffstruc__types, only: array_type, get_partial, &
       operator(+), operator(-), operator(*), operator(**)
  implicit none


  private

  public :: maxval, max

  ! Operation interfaces
  !-----------------------------------------------------------------------------
  interface maxval
     module procedure maxval_array
  end interface

  interface max
     module procedure max_array
     module procedure max_scalar
  end interface


contains

!###############################################################################
  function maxval_array(a, dim) result(c)
    !! Find maximum value along a dimension
    implicit none
    class(array_type), intent(in), target :: a
    integer, intent(in) :: dim
    real(real32), dimension(:), allocatable :: c

    integer :: i, s

    if(size(a%shape) .ne. 1)then
       call stop_program("maxval: only 1D arrays can be used")
    end if

    if(dim.eq.1)then
       allocate(c(size(a%val,2)))
       do concurrent(s=1:size(a%val,2))
          c(s) = maxval(a%val(:,s))
       end do
    else if(dim.eq.2)then
       allocate(c(size(a%val,1)))
       do concurrent(i=1:size(a%val,1))
          c(i) = maxval(a%val(i,:))
       end do
    else
       call stop_program("maxval: only 1 or 2 dimensions are supported")
    end if

  end function maxval_array
!###############################################################################


!###############################################################################
  function max_array(a, b) result(c)
    !! Find maximum value between two autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = max(a%val, b%val)

    c%get_partial_left => get_partial_max_left
    c%get_partial_right => get_partial_max_right
    c%get_partial_left_val => get_partial_max_left_val
    c%get_partial_right_val => get_partial_max_right_val
    if(a%requires_grad .or. b%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%operation = 'max'
       c%left_operand => a
       c%right_operand => b
       c%owns_left_operand = a%is_temporary
       c%owns_right_operand = b%is_temporary
    end if
  end function max_array
!-------------------------------------------------------------------------------
  function max_scalar(a, scalar) result(c)
    !! Find maximum value between an autodiff array and a scalar
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), intent(in) :: scalar
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = max(a%val, scalar)

    c%get_partial_left => get_partial_max_left
    c%get_partial_left_val => get_partial_max_left_val
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'max_scalar'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function max_scalar
!-------------------------------------------------------------------------------
  function get_partial_max_left(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    ptr => upstream_grad * (abs( this%val - this%left_operand%val ) .lt. 1.E-6_real32)
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_max_left
!-------------------------------------------------------------------------------
  function get_partial_max_right(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    ptr => upstream_grad * (abs( this%val - this%right_operand%val ) .lt. 1.E-6_real32)
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_max_right
!-------------------------------------------------------------------------------
  pure subroutine get_partial_max_left_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: i, j

    output = 0._real32
    do concurrent(i=1:size(upstream_grad,1), j=1:size(upstream_grad,2))
       if( abs( this%val(i,j) - this%left_operand%val(i,j) ) .lt. 1.E-6_real32 ) then
          output(i,j) = upstream_grad(i,j)
       end if
    end do

  end subroutine get_partial_max_left_val
!-------------------------------------------------------------------------------
  pure subroutine get_partial_max_right_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: i, j

    output = 0._real32
    do concurrent(i=1:size(upstream_grad,1), j=1:size(upstream_grad,2))
       if( abs( this%val(i,j) - this%right_operand%val(i,j) ) .lt. 1.E-6_real32 ) then
          output(i,j) = upstream_grad(i,j)
       end if
    end do

  end subroutine get_partial_max_right_val
!###############################################################################

end module diffstruc__operations_reduction
