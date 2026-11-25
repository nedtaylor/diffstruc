module diffstruc__operations_hyp
  !! This module contains hyperbolic operations for the diffstruc library.
  use coreutils, only: real32
  use diffstruc__types, only: canonical, constant_node, array_type, get_partial, &
       team_size_for, operator(+), operator(-), operator(*), operator(**)
  implicit none


  private

  public :: tanh, tanhf


  ! Operation interfaces
  !-----------------------------------------------------------------------------
  interface tanh
     module procedure tanh_array
  end interface

  interface tanhf
     module procedure tanhf_array
  end interface


contains

!###############################################################################
  function tanhf_array(a) result(c)
    !! Fast, vectorised hyperbolic tangent: same API, gradients and graph
    !! structure as `tanh`, with the [7/6] Pade approximation (maximum
    !! absolute error 8.5e-5).
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c
    type(array_type), pointer :: na, nc
    integer :: i, s, nth
    real(real32) :: av, x2, p, q

    na => canonical(a)
    c => a%create_result()
    nc => c
    nth = team_size_for(a, size(na%store, 1), size(na%store, 2))
    !$omp parallel do if(nth > 1) num_threads(nth) schedule(static)
    do s = 1, size(na%store, 2)
       !$omp simd
       do i = 1, size(na%store, 1)
          av = max(-4.9_real32, min(4.9_real32, na%store(i,s)))
          x2 = av * av
          p = av * (135135.0_real32 + x2 * (17325.0_real32 + x2 * &
               (378.0_real32 + x2)))
          q = 135135.0_real32 + x2 * (62370.0_real32 + x2 * &
               (3150.0_real32 + x2 * 28.0_real32))
          nc%store(i,s) = max(-1.0_real32, min(1.0_real32, p / q))
       end do
    end do

    c%get_partial_left => get_partial_tanh
    c%get_partial_left_val => get_partial_tanh_val
    c%get_partial_left_val_sum => get_partial_tanh_val_sum
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'tanh'
       c%left_operand => canonical(a)
    end if
  end function tanhf_array
!###############################################################################


!###############################################################################
  function tanh_array(a) result(c)
    !! Hyperbolic tangent function for autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c
    type(array_type), pointer :: na, nc
    integer :: i, s, nth
    real(real32) :: av, x2, p, q

    na => canonical(a)
    c => a%create_result()
    nc => c
    nth = team_size_for(a, size(na%store, 1), size(na%store, 2))
    !$omp parallel do if(nth > 1) num_threads(nth) schedule(static)
    do s = 1, size(na%store, 2)
       !$omp simd
       do i = 1, size(na%store, 1)
#ifdef DIFFSTRUC_FAST_TANH
          av = max(-4.9_real32, min(4.9_real32, na%store(i,s)))
          x2 = av * av
          p = av * (135135.0_real32 + x2 * (17325.0_real32 + x2 * &
               (378.0_real32 + x2)))
          q = 135135.0_real32 + x2 * (62370.0_real32 + x2 * &
               (3150.0_real32 + x2 * 28.0_real32))
          nc%store(i,s) = max(-1.0_real32, min(1.0_real32, p / q))
#else
          nc%store(i,s) = tanh(na%store(i,s))
#endif
       end do
    end do
    !c%val = tanh(a%val)

    c%get_partial_left => get_partial_tanh
    c%get_partial_left_val => get_partial_tanh_val
    c%get_partial_left_val_sum => get_partial_tanh_val_sum
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'tanh'
       c%left_operand => canonical(a)
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
    output = ptr
  end function get_partial_tanh
!-------------------------------------------------------------------------------
  pure subroutine get_partial_tanh_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: i, s
    do s = 1, size(output, 2)
       !$omp simd
       do i = 1, size(output, 1)
          output(i,s) = upstream_grad(i,s) * &
               (1._real32 - this%val(i,s) * this%val(i,s))
       end do
    end do
  end subroutine get_partial_tanh_val
!-------------------------------------------------------------------------------
  pure subroutine get_partial_tanh_val_sum(this, upstream_grad, output)
    !! Fused partial+sum for tanh: output(:,1) = sum_s(upstream * (1 - val^2))
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:), intent(out) :: output

    integer :: s, n_samples

    n_samples = size(upstream_grad, 2)
    output(:) = upstream_grad(:,1) * &
         (1._real32 - this%val(:,1) * this%val(:,1))
    do s = 2, n_samples
       output(:) = output(:) + upstream_grad(:,s) * &
            (1._real32 - this%val(:,s) * this%val(:,s))
    end do
  end subroutine get_partial_tanh_val_sum
!###############################################################################


!###############################################################################
  function tanh_reverse_array(a) result(c)
    !! Reverse mode for tanh function
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    integer :: i, s

    c => a%create_result()
    do s = 1,size(a%val, 2)
       !$omp simd
       do i = 1,size(a%val, 1)
          c%val(i,s) = 1._real32 - a%val(i,s) * a%val(i,s)
       end do
    end do

    c%get_partial_left => get_partial_tanh_reverse
    c%get_partial_left_val => get_partial_tanh_reverse_val
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'tanh_reverse'
       c%left_operand => canonical(a)
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
    output = ptr
  end function get_partial_tanh_reverse
!-------------------------------------------------------------------------------
  pure subroutine get_partial_tanh_reverse_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    output = (-2._real32) * upstream_grad * this%left_operand%val
  end subroutine get_partial_tanh_reverse_val
!###############################################################################



end module diffstruc__operations_hyp
