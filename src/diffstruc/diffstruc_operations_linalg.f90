module diffstruc__operations_linalg
  !! This module contains linear algebra operations for the diffstruc library.
  use coreutils, only: real32
  use diffstruc__types, only: array_type, operator(*)
  implicit none

  private

  public :: matmul, operator(.mmul.)
  public :: outer_product, operator(.outer.)
  public :: dot_product, operator(.dot.)
  public :: transpose

  ! Operation interfaces
  !-----------------------------------------------------------------------------
  interface matmul
     module function matmul_arrays(a, b) result(c)
       class(array_type), intent(in), target :: a, b
       type(array_type), pointer :: c
     end function matmul_arrays

     module function matmul_real2d(a, b) result(c)
       class(array_type), intent(in), target :: a
       real(real32), dimension(:,:), intent(in) :: b
       type(array_type), pointer :: c
     end function matmul_real2d
     module function real2d_matmul(a, b) result(c)
       real(real32), dimension(:,:), intent(in) :: a
       class(array_type), intent(in), target :: b
       type(array_type), pointer :: c
     end function real2d_matmul
  end interface

  interface operator(.mmul.)
    module procedure matmul_arrays, matmul_real2d, real2d_matmul
  end interface

  interface outer_product
     module function outer_product_arrays(a, b) result(c)
       class(array_type), intent(in), target :: a, b
       type(array_type), pointer :: c
     end function outer_product_arrays
  end interface

  interface operator(.outer.)
    module procedure outer_product_arrays
  end interface

  interface operator(.dot.)
    module procedure dot_product_arrays
  end interface

  interface dot_product
     module function dot_product_arrays(a, b) result(c)
       class(array_type), intent(in), target :: a, b
       type(array_type), pointer :: c
     end function dot_product_arrays
  end interface

  interface transpose
     module function transpose_array(a) result(c)
       class(array_type), intent(in), target :: a
       type(array_type), pointer :: c
     end function transpose_array
  end interface

end module diffstruc__operations_linalg
