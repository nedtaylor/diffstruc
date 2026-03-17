module diffstruc__operations_trig
  !! This module contains trigonometric operations for the diffstruc library.
  use coreutils, only: real32
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

    c => a%create_result()
    c%val = sin(a%val)

    c%get_partial_left => get_partial_sin
    c%get_partial_left_val => get_partial_sin_val
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'sin'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function sin_array
!-------------------------------------------------------------------------------
  function get_partial_sin(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    logical :: left_is_temporary_local
    type(array_type), pointer :: ptr

    left_is_temporary_local = this%left_operand%is_temporary
    this%left_operand%is_temporary = .false.
    ptr => upstream_grad * cos( this%left_operand )
    this%left_operand%is_temporary = left_is_temporary_local
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_sin
!-------------------------------------------------------------------------------
  pure subroutine get_partial_sin_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    output = upstream_grad * cos(this%left_operand%val)
  end subroutine get_partial_sin_val
!###############################################################################


!###############################################################################
  function cos_array(a) result(c)
    !! Cosine function for autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = cos(a%val)

    c%get_partial_left => get_partial_cos
    c%get_partial_left_val => get_partial_cos_val
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'cos'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function cos_array
!-------------------------------------------------------------------------------
  function get_partial_cos(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    logical :: left_is_temporary_local
    type(array_type), pointer :: ptr

    left_is_temporary_local = this%left_operand%is_temporary
    this%left_operand%is_temporary = .false.
    ptr => -upstream_grad * sin( this%left_operand )
    this%left_operand%is_temporary = left_is_temporary_local
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_cos
!-------------------------------------------------------------------------------
  pure subroutine get_partial_cos_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    output = -upstream_grad * sin(this%left_operand%val)
  end subroutine get_partial_cos_val
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
    c%get_partial_left_val => get_partial_tan_val
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'tan'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function tan_array
!-------------------------------------------------------------------------------
  function get_partial_tan(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    logical :: left_is_temporary_local
    type(array_type), pointer :: ptr

    left_is_temporary_local = this%left_operand%is_temporary
    this%left_operand%is_temporary = .false.
    ptr => upstream_grad / ( cos( this%left_operand ) ** 2._real32 )
    this%left_operand%is_temporary = left_is_temporary_local
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_tan
!-------------------------------------------------------------------------------
  pure subroutine get_partial_tan_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    real(real32) :: cos_val
    integer :: i, s

    do concurrent(s = 1:size(upstream_grad,2), i = 1:size(upstream_grad,1))
       cos_val = cos(this%left_operand%val(i,s))
       output(i,s) = upstream_grad(i,s) / (cos_val * cos_val)
    end do
  end subroutine get_partial_tan_val
!###############################################################################

end module diffstruc__operations_trig
