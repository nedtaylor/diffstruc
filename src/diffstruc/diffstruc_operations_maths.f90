module diffstruc__operations_maths
  !! This module contains maths operations for the diffstruc library.
  use coreutils, only: real32, pi
  use diffstruc__types, only: array_type, get_partial, &
       operator(+), operator(-), operator(*), operator(/), operator(**), exp
  implicit none


  private

  public :: sqrt, sign, sigmoid, gaussian, abs


  ! Operation interfaces
  !-----------------------------------------------------------------------------
  interface abs
     module procedure abs_array
  end interface

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
  function abs_array(a) result(c)
    !! Square root function for autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = abs(a%val)

    c%get_partial_left => get_partial_abs
    c%get_partial_left_val => get_partial_abs_val
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'abs'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function abs_array
!-------------------------------------------------------------------------------
  function get_partial_abs(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = upstream_grad * sign(1._real32, this)

  end function get_partial_abs
!-------------------------------------------------------------------------------
  pure subroutine get_partial_abs_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    output = sign(1._real32, this%val) * upstream_grad

  end subroutine get_partial_abs_val
!###############################################################################


!###############################################################################
  function sqrt_array(a) result(c)
    !! Square root function for autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = sqrt(a%val)

    c%get_partial_left => get_partial_sqrt
    c%get_partial_left_val => get_partial_sqrt_val
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'sqrt'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
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
!-------------------------------------------------------------------------------
  pure subroutine get_partial_sqrt_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    output = upstream_grad / ( 2._real32 * this%val )

  end subroutine get_partial_sqrt_val
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
    !    c%owns_left_operand = array%is_temporary
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
    c%get_partial_left_val => get_partial_sigmoid_val
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'sigmoid'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function sigmoid_array
!-------------------------------------------------------------------------------
  function get_partial_sigmoid(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    logical :: this_is_temporary_local
    type(array_type), pointer :: ptr

    this_is_temporary_local = this%is_temporary
    this%is_temporary = .false.
    ptr => upstream_grad * this * (1.0_real32 - this)
    this%is_temporary = this_is_temporary_local
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_sigmoid
!-------------------------------------------------------------------------------
  pure subroutine get_partial_sigmoid_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    output = upstream_grad * this%val * (1.0_real32 - this%val)
  end subroutine get_partial_sigmoid_val
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
    c%get_partial_left_val => get_partial_gaussian_val
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'gaussian'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
    allocate(b_array)
    b_array%is_sample_dependent = .false.
    b_array%requires_grad = .false.
    call b_array%allocate(array_shape=[2, 1])
    b_array%val(1,1) = mu
    b_array%val(2,1) = sigma
    c%right_operand => b_array
    c%owns_right_operand = .true.

  end function gaussian_array
!-------------------------------------------------------------------------------
  function get_partial_gaussian(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    logical :: this_is_temporary_local
    real(real32) :: coeff
    type(array_type), pointer :: ptr1, ptr2, ptr3

    this_is_temporary_local = this%is_temporary
    this%is_temporary = .false.
    coeff = - 1._real32 / ( sqrt(2._real32 * pi ) * this%right_operand%val(2,1)**3 )

    ptr1 => this - this%right_operand%val(1,1)
    ptr2 => -0.5_real32 * ( ptr1 / this%right_operand%val(2,1) ) ** 2
    ptr1%is_temporary = .false.
    ptr3 => upstream_grad * coeff * ptr1 * exp(ptr2)
    ptr1%is_temporary = .true.
    this%is_temporary = this_is_temporary_local
    call output%assign_and_deallocate_source(ptr3)
  end function get_partial_gaussian
!-------------------------------------------------------------------------------
  pure subroutine get_partial_gaussian_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output
    real(real32) :: coeff

    coeff = - 1._real32 / ( sqrt(2._real32 * pi ) * this%right_operand%val(2,1)**3 )
    output = upstream_grad * coeff * (this%val - this%right_operand%val(1,1)) * &
         exp( -0.5_real32 * ( &
              (this%val - this%right_operand%val(1,1)) / &
              this%right_operand%val(2,1) &
         )**2 )
  end subroutine get_partial_gaussian_val
!###############################################################################

end module diffstruc__operations_maths
