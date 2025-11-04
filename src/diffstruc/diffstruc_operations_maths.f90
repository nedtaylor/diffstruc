module diffstruc__operations_maths
  !! This module contains maths operations for the diffstruc library.
  use coreutils, only: real32, pi
  use diffstruc__types, only: array_type, get_partial, &
       operator(+), operator(-), operator(*), operator(/), operator(**), exp
  implicit none


  private

  public :: sqrt, sign, sigmoid, gaussian


  ! Operation interfaces
  !-----------------------------------------------------------------------------
  interface sqrt
     module procedure sqrt_array
  end interface

  interface sign
     module procedure sign_array
  end interface

  interface sigmoid
     module procedure sigmoid_array
  end interface

  interface gaussian
     module procedure gaussian_array
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
    !    c%operation = 'sign'
    !    c%left_operand => array
    ! end if
  end function sign_array
!###############################################################################


!###############################################################################
  function sigmoid_array(a) result(c)
    !! Sigmoid function for autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = 1.0_real32 / (1.0_real32 + exp(-a%val))

    c%get_partial_left => get_partial_sigmoid
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'sigmoid'
       c%left_operand => a
    end if
  end function sigmoid_array
!-------------------------------------------------------------------------------
  function get_partial_sigmoid(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: sigmoid_val

    sigmoid_val => sigmoid_array(this)
    output = upstream_grad * sigmoid_val * (1.0_real32 - sigmoid_val)

  end function get_partial_sigmoid
!###############################################################################


!###############################################################################
  function gaussian_array(a, mu, sigma) result(c)
    !! Generate a Gaussian random autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), intent(in) :: mu, sigma
    type(array_type), pointer :: c
    type(array_type), pointer :: b_array

    c => a%create_result()
    c%val = 1._real32/(sqrt(2*pi)*sigma) * exp( -0.5_real32 * ((a%val - mu)/sigma)**2 )

    c%get_partial_left => get_partial_gaussian
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'gaussian'
       c%left_operand => a
    end if
    allocate(b_array)
    b_array%is_sample_dependent = .false.
    b_array%requires_grad = .false.
    call b_array%allocate(array_shape=[2, 1])
    b_array%val(1,1) = mu
    b_array%val(2,1) = sigma
    c%right_operand => b_array
    c%owns_left_operand = .true.

  end function gaussian_array
!-------------------------------------------------------------------------------
  function get_partial_gaussian(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    real(real32) :: rcoeff1
    type(array_type) :: coeff2, exponent

    rcoeff1 = - 1._real32 / ( sqrt(2._real32 * pi ) * this%right_operand%val(2,1)**3 )
    coeff2 = this - this%right_operand%val(1,1)
    exponent = -0.5_real32 * ( coeff2 / this%right_operand%val(2,1) ) ** 2
    output = upstream_grad * rcoeff1 * coeff2 * exp(exponent)

  end function get_partial_gaussian
!###############################################################################

end module diffstruc__operations_maths
