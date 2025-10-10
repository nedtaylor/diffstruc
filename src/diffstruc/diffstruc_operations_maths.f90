module diffstruc__operations_maths
  !! This module contains maths operations for the diffstruc library.
  use coreutils, only: real32
  use diffstruc__types, only: array_type, get_partial, &
       operator(+), operator(-), operator(*), operator(/)
  implicit none


  private

  public :: sqrt, sign


  ! Operation interfaces
  !-----------------------------------------------------------------------------
  interface sqrt
     module procedure sqrt_array
  end interface

  interface sign
     module procedure sign_array
  end interface


contains

!###############################################################################
  function sqrt_array(a) result(c)
    !! Square root function for autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = sqrt(a%val)

    c%get_partial_left => get_partial_sqrt
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%is_leaf = .false.
       c%operation = 'sqrt'
       c%left_operand => a
    end if
  end function sqrt_array
!-------------------------------------------------------------------------------
  function get_partial_sqrt(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = upstream_grad / ( 2._real32 * this )

  end function get_partial_sqrt
!###############################################################################


!###############################################################################
  function sign_array(scalar, array) result(c)
    !! Add a scalar sign to an autodiff array
    real(real32), intent(in) :: scalar
    class(array_type), intent(in), target :: array
    real(real32), dimension(:,:), allocatable :: c
    ! type(array_type), pointer :: c

    allocate(c(size(array%val,1), size(array%val,2)))
    c = sign(scalar, array%val)
    ! allocate(c)
    ! call c%allocate(array_shape=array%shape)
    ! c%val = sign(scalar, array%val)

    ! if(array%requires_grad) then
    !    c%requires_grad = .true.
    !    c%is_leaf = .false.
    !    c%operation = 'sign'
    !    c%left_operand => array
    ! end if
  end function sign_array
!###############################################################################

end module diffstruc__operations_maths
