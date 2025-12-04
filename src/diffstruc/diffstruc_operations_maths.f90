module diffstruc__operations_maths
  !! This module contains maths operations for the diffstruc library.
  use coreutils, only: real32
  use diffstruc__types, only: array_type
  implicit none


  private

  public :: sqrt, sign, sigmoid, gaussian, abs, log10


  ! Operation interfaces
  !-----------------------------------------------------------------------------
  interface abs
     module function abs_array(a) result(c)
       class(array_type), intent(in), target :: a
       type(array_type), pointer :: c
     end function abs_array
  end interface

  interface sqrt
     module function sqrt_array(a) result(c)
       class(array_type), intent(in), target :: a
       type(array_type), pointer :: c
     end function sqrt_array
  end interface

  interface sign
     module function sign_array(scalar, array) result(c)
       real(real32), intent(in) :: scalar
       class(array_type), intent(in), target :: array
       real(real32), dimension(:,:), allocatable :: c
     end function sign_array
  end interface

  interface sigmoid
     module function sigmoid_array(a) result(c)
       class(array_type), intent(in), target :: a
       type(array_type), pointer :: c
     end function sigmoid_array
  end interface

  interface gaussian
     module function gaussian_array(a, mu, sigma) result(c)
       class(array_type), intent(in), target :: a
       real(real32), intent(in) :: mu, sigma
       type(array_type), pointer :: c
     end function gaussian_array
  end interface

  interface log10
     module function log10_array(a) result(c)
       class(array_type), intent(in), target :: a
       type(array_type), pointer :: c
     end function log10_array
  end interface

end module diffstruc__operations_maths
