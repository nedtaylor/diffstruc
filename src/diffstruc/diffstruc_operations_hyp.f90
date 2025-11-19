module diffstruc__operations_hyp
  !! This module contains hyperbolic operations for the diffstruc library.
  use coreutils, only: real32
  use diffstruc__types, only: array_type, get_partial, &
       operator(+), operator(-), operator(*), operator(**)
  implicit none


  private

  public :: tanh


  ! Operation interfaces
  !-----------------------------------------------------------------------------
  interface tanh
     module procedure tanh_array
  end interface


contains

!###############################################################################
  function tanh_array(a) result(c)
    !! Hyperbolic tangent function for autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    integer :: i, s

    c => a%create_result()
    do concurrent(s = 1:size(a%val, 2), i = 1:size(a%val,1))
      c%val(i,s) = tanh(a%val(i,s))
    end do
    !c%val = tanh(a%val)

    c%get_partial_left => get_partial_tanh
    c%get_partial_left_val => get_partial_tanh_val
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'tanh'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function tanh_array
!-------------------------------------------------------------------------------
  function get_partial_tanh(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    logical :: this_is_temporary_local
    type(array_type), pointer :: ptr

    this_is_temporary_local = this%is_temporary
    this%is_temporary = .false.
    ! derivative of tanh(x) is (1 - tanh(x)^2)
    ptr => upstream_grad * tanh_reverse_array(this)
    ! ptr => upstream_grad * (1._real32 - this ** 2._real32)
    this%is_temporary = this_is_temporary_local
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_tanh
!-------------------------------------------------------------------------------
  subroutine get_partial_tanh_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(inout) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    output = upstream_grad * (1._real32 - this%val ** 2._real32)
  end subroutine get_partial_tanh_val
!###############################################################################


!###############################################################################
  function tanh_reverse_array(a) result(c)
    !! Reverse mode for tanh function
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    integer :: i, s

    c => a%create_result()
    do concurrent(s = 1:size(a%val, 2), i = 1:size(a%val,1))
      c%val(i,s) = 1._real32 - (a%val(i,s) ** 2._real32)
    end do

    c%get_partial_left => get_partial_tanh_reverse
    c%get_partial_left_val => get_partial_tanh_reverse_val
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'tanh_reverse'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function tanh_reverse_array
!-------------------------------------------------------------------------------
  function get_partial_tanh_reverse(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    logical :: left_is_temporary_local
    type(array_type), pointer :: ptr

    left_is_temporary_local = this%left_operand%is_temporary
    this%left_operand%is_temporary = .false.
    ptr => (-2._real32) * upstream_grad * this%left_operand
    this%left_operand%is_temporary = left_is_temporary_local
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_tanh_reverse
!-------------------------------------------------------------------------------
  subroutine get_partial_tanh_reverse_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(inout) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    output = (-2._real32) * upstream_grad * this%left_operand%val
  end subroutine get_partial_tanh_reverse_val
!###############################################################################

end module diffstruc__operations_hyp
