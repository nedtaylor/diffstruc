module diffstruc__operations_trig
  !! This module contains trigonometric operations for the diffstruc library.
  use corestruc, only: real32
  use diffstruc__types, only: array_type, get_partial, &
       operator(+), operator(-), operator(*), operator(/), operator(**)
  implicit none


  private

  public :: sin, cos, tan


  ! Operation interfaces
  !-----------------------------------------------------------------------------
  interface sin
     module procedure sin_array
  end interface

  interface cos
     module procedure cos_array
  end interface

  interface tan
     module procedure tan_array
  end interface


contains

!###############################################################################
  function sin_array(a) result(c)
    !! Sine function for autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    !  allocate(c)
    c => a%create_result()
    c%val = sin(a%val)

    c%get_partial_left => get_partial_sin
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'sin'
       c%left_operand => a
    end if
  end function sin_array
!-------------------------------------------------------------------------------
  function get_partial_sin(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = upstream_grad * cos( this%left_operand )

  end function get_partial_sin
!###############################################################################


!###############################################################################
  function cos_array(a) result(c)
    !! Cosine function for autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    !  allocate(c)
    c => a%create_result()
    c%val = cos(a%val)

    c%get_partial_left => get_partial_cos
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'cos'
       c%left_operand => a
    end if
  end function cos_array
!-------------------------------------------------------------------------------
  function get_partial_cos(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = -upstream_grad * sin( this%left_operand )

  end function get_partial_cos
!###############################################################################


!###############################################################################
  function tan_array(a) result(c)
    !! Tangent function for autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = tan(a%val)

    c%get_partial_left => get_partial_tan
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'tan'
       c%left_operand => a
    end if
  end function tan_array
!-------------------------------------------------------------------------------
  function get_partial_tan(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = upstream_grad / ( cos( this%left_operand ) ** 2._real32 )

  end function get_partial_tan
!###############################################################################

end module diffstruc__operations_trig
