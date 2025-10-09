module diffstruc__operations_hyp
  !! This module contains hyperbolic operations for the diffstruc library.
  use corestruc, only: real32
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

    c => a%create_result()
    c%val = tanh(a%val)

    c%get_partial_left => get_partial_tanh
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'tanh'
       c%left_operand => a
    end if
  end function tanh_array
!-------------------------------------------------------------------------------
  function get_partial_tanh(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    ! derivative of tanh(x) is (1 - tanh(x)^2)
    output = upstream_grad * (1._real32 - this ** 2._real32)
    ! output = upstream_grad * tanh_reverse_array( this )
  end function get_partial_tanh
!###############################################################################


!###############################################################################
  function tanh_reverse_array(a) result(c)
    !! Reverse mode for tanh function
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    !  allocate(output)
    c => a%create_result()
    c%val = (1._real32 - a%val ** 2._real32)

    c%get_partial_left => get_partial_tanh_reverse
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'tanh_reverse'
       c%left_operand => a
    end if

  end function tanh_reverse_array
!-------------------------------------------------------------------------------
  function get_partial_tanh_reverse(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: left

    allocate(left)
    left = -2._real32 * this%left_operand
    output = left * this

  end function get_partial_tanh_reverse
!###############################################################################

end module diffstruc__operations_hyp
