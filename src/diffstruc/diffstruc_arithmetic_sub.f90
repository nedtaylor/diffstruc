!> Arithmetic operations and their partial derivatives for
!! `diffstruc__types`. The results and operands are all context-owned
!! records; the procedures here only create result nodes and views.
submodule(diffstruc__types) diffstruc__arithmetic_impl
  implicit none

contains

  module function add_arrays(a, b) result(c)
    !! Add two autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    integer :: i, s
    type(array_type), pointer :: na, nb, nc
    na => canonical(a)
    nb => canonical(b)
    c => a%create_result()
    nc => c
    if(b%is_sample_dependent)then
       do s = 1, size(na%store, 2)
          !$omp simd
          do i = 1, size(na%store, 1)
             nc%store(i,s) = na%store(i,s) + nb%store(i,s)
          end do
       end do
    else
       do s = 1, size(na%store, 2)
          !$omp simd
          do i = 1, size(na%store, 1)
             nc%store(i,s) = na%store(i,s) + nb%store(i,1)
          end do
       end do
    end if
    c%get_partial_left => get_partial_add
    c%get_partial_right => get_partial_add
    c%get_partial_left_val => get_partial_add_val
    c%get_partial_right_val => get_partial_add_val
    c%get_partial_left_val_sum => get_partial_add_val_sum
    c%get_partial_right_val_sum => get_partial_add_val_sum
    ! Set up computation graph
    if(a%requires_grad .or. b%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%operation = 'add'
       c%left_operand => canonical(a)
       c%right_operand => canonical(b)
    end if
  end function add_arrays
!-------------------------------------------------------------------------------
  module function add_real2d(a, b) result(c)
    !! Add a real array to an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), dimension(:,:), intent(in) :: b
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = a%val + b

    c%get_partial_left => get_partial_add
    c%get_partial_left_val => get_partial_add_val
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'add'
       c%left_operand => canonical(a)
    end if
  end function add_real2d
!-------------------------------------------------------------------------------
  module function real2d_add(a, b) result(c)
    !! Add a real array to an autodiff array
    implicit none
    real(real32), dimension(:,:), intent(in) :: a
    class(array_type), intent(in), target :: b
    type(array_type), pointer :: c

    c => add_real2d(b, a)
  end function real2d_add
!-------------------------------------------------------------------------------
  module function add_real1d(a, b) result(c)
    !! Add a real array to an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), dimension(:), intent(in) :: b
    type(array_type), pointer :: c

    integer :: s

    c => a%create_result()
    do s = 1,size(a%val, 2)
       c%val(:,s) = a%val(:,s) + b(:)
    end do

    c%get_partial_left => get_partial_add
    c%get_partial_left_val => get_partial_add_val
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'add'
       c%left_operand => canonical(a)
    end if
  end function add_real1d
!-------------------------------------------------------------------------------
  module function real1d_add(a, b) result(c)
    !! Add a real array to an autodiff array
    implicit none
    real(real32), dimension(:), intent(in) :: a
    class(array_type), intent(in), target :: b
    type(array_type), pointer :: c

    c => add_real1d(b, a)
  end function real1d_add
!-------------------------------------------------------------------------------
  module function add_scalar(a, b) result(c)
    !! Add a scalar to an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), intent(in) :: b
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = a%val + b

    c%get_partial_left => get_partial_add
    c%get_partial_left_val => get_partial_add_val
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'add'
       c%left_operand => canonical(a)
    end if
  end function add_scalar
!-------------------------------------------------------------------------------
  module function scalar_add(a, b) result(c)
    !! Add a scalar to an autodiff array
    implicit none
    real(real32), intent(in) :: a
    class(array_type), intent(in), target :: b
    type(array_type), pointer :: c

    c => add_scalar(b, a)
  end function scalar_add
!-------------------------------------------------------------------------------
  function get_partial_add(this, upstream_grad) result(output)
    !! Get partial derivative with respect to left operand
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = upstream_grad
  end function get_partial_add
!-------------------------------------------------------------------------------
  pure subroutine get_partial_add_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: i, s

    if(size(upstream_grad,2).ne.size(output,2))then
       if(size(output,1).eq.1)then
          output(1,1) = sum(upstream_grad)
       else
          output(:,1) = sum(upstream_grad, dim=2)
       end if
    else
       if(size(output,1).eq.1.and.size(output,1).ne.size(upstream_grad,1))then
          output(1,:) = sum(upstream_grad,1)
       else
          do s = 1, size(output, 2)
             !$omp simd
             do i = 1, size(output, 1)
                output(i,s) = upstream_grad(i,s)
             end do
          end do
       end if
    end if
  end subroutine get_partial_add_val
!-------------------------------------------------------------------------------
  pure subroutine get_partial_add_val_sum(this, upstream_grad, output)
    !! Fused partial+sum for add: output(:,1) = sum_over_samples(upstream_grad)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:), intent(out) :: output

    integer :: s, n_samples

    n_samples = size(upstream_grad, 2)
    if(size(output,1).eq.1 .and. size(upstream_grad,1).gt.1)then
       output(1) = sum(upstream_grad)
    else if(n_samples .eq. 1)then
       output(:) = upstream_grad(:,1)
    else
       output(:) = upstream_grad(:,1)
       do s = 2, n_samples
          output(:) = output(:) + upstream_grad(:,s)
       end do
    end if
  end subroutine get_partial_add_val_sum
!###############################################################################


!###############################################################################
  module function subtract_arrays(a, b) result(c)
    !! Subtract two autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    integer :: i, s

    c => a%create_result()
    if(.not.b%is_sample_dependent)then
       do s = 1,size(a%val, 2)
          !$omp simd
          do i = 1,size(a%val, 1)
             c%val(i,s) = a%val(i,s) - b%val(i,1)
          end do
       end do
    else
       do s = 1,size(a%val, 2)
          !$omp simd
          do i = 1,size(a%val, 1)
             c%val(i,s) = a%val(i,s) - b%val(i,s)
          end do
       end do
    end if
    c%get_partial_left => get_partial_add
    c%get_partial_right => get_partial_negate
    c%get_partial_left_val => get_partial_add_val
    c%get_partial_right_val => get_partial_negate_val
    c%get_partial_left_val_sum => get_partial_add_val_sum
    c%get_partial_right_val_sum => get_partial_negate_val_sum
    if(a%requires_grad .or. b%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%operation = 'subtract'
       c%left_operand => canonical(a)
       c%right_operand => canonical(b)
    end if
  end function subtract_arrays
!-------------------------------------------------------------------------------
  module function subtract_real2d(a, b) result(c)
    !! Subtract a real array from an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), dimension(:,:), intent(in) :: b
    type(array_type), pointer :: c

    integer :: i, s

    c => a%create_result()
    do s = 1,size(a%val, 2)
       !$omp simd
       do i = 1,size(a%val, 1)
          c%val(i,s) = a%val(i,s) - b(i,s)
       end do
    end do

    c%get_partial_left => get_partial_add
    c%get_partial_left_val => get_partial_add_val
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'subtract_scalar'
       c%left_operand => canonical(a)
    end if
  end function subtract_real2d
!-------------------------------------------------------------------------------
  module function subtract_real1d(a, b) result(c)
    !! Subtract a real array from an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), dimension(:), intent(in) :: b
    type(array_type), pointer :: c

    integer :: s

    c => a%create_result()
    do s = 1,size(a%val, 2)
       c%val(:,s) = a%val(:,s) - b(s)
    end do

    c%get_partial_left => get_partial_add
    c%get_partial_left_val => get_partial_negate_val
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'subtract_scalar'
       c%left_operand => canonical(a)
    end if
  end function subtract_real1d
!-------------------------------------------------------------------------------
  module function subtract_scalar(a, b) result(c)
    !! Subtract a scalar from an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), intent(in) :: b
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = a%val - b

    c%get_partial_left => get_partial_add
    c%get_partial_left_val => get_partial_add_val
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'subtract_scalar'
       c%left_operand => canonical(a)
    end if
  end function subtract_scalar
!-------------------------------------------------------------------------------
  module function scalar_subtract(a, b) result(c)
    !! Subtract an autodiff array from a scalar
    implicit none
    real(real32), intent(in) :: a
    class(array_type), intent(in), target :: b
    type(array_type), pointer :: c

    c => negate_array(b)
    c%val = a + c%val

    if(b%requires_grad)then
       c%operation = 'subtract_scalar'
    end if
  end function scalar_subtract
!-------------------------------------------------------------------------------
  module function negate_array(a) result(c)
    !! Negate an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = -a%val

    c%get_partial_left => get_partial_negate
    c%get_partial_left_val => get_partial_negate_val
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'negate'
       c%left_operand => canonical(a)
    end if
  end function negate_array
!-------------------------------------------------------------------------------
  function get_partial_negate(this, upstream_grad) result(output)
    !! Get partial derivative with respect to left operand
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    ptr => -upstream_grad
    output = ptr
  end function get_partial_negate
!-------------------------------------------------------------------------------
  pure subroutine get_partial_negate_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    if(size(upstream_grad,2).ne.size(output,2))then
       if(size(output,1).eq.1)then
          output(1,1) = -sum(upstream_grad)
       else
          output(:,1) = -sum(upstream_grad, dim=2)
       end if
    else
       if(size(output,1).eq.1.and.size(output,1).ne.size(upstream_grad,1))then
          output(1,:) = -sum(upstream_grad,1)
       else
          output = -upstream_grad
       end if
    end if
  end subroutine get_partial_negate_val
!-------------------------------------------------------------------------------
  pure subroutine get_partial_negate_val_sum(this, upstream_grad, output)
    !! Fused partial+sum for negate: output(:,1) = -sum_over_samples(upstream_grad)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:), intent(out) :: output

    integer :: s, n_samples

    n_samples = size(upstream_grad, 2)
    if(size(output,1).eq.1 .and. size(upstream_grad,1).gt.1)then
       output(1) = -sum(upstream_grad)
    else if(n_samples .eq. 1)then
       output(:) = -upstream_grad(:,1)
    else
       output(:) = -upstream_grad(:,1)
       do s = 2, n_samples
          output(:) = output(:) - upstream_grad(:,s)
       end do
    end if
  end subroutine get_partial_negate_val_sum
!###############################################################################


!###############################################################################
  module function multiply_arrays(a, b) result(c)
    !! Multiply two autodiff arrays (element-wise)
    implicit none
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    integer :: i, s

    if(b%is_scalar)then
       c => a%create_result()
       do s = 1, size(a%val, 2)
          !$omp simd
          do i = 1, size(a%val, 1)
             c%val(i,s) = a%val(i,s) * b%val(1,1)
          end do
       end do
    elseif(.not.b%is_sample_dependent)then
       c => a%create_result()
       do s = 1,size(a%val, 2)
          !$omp simd
          do i = 1,size(a%val, 1)
             c%val(i,s) = a%val(i,s) * b%val(i,1)
          end do
       end do
    elseif(.not.a%is_sample_dependent)then
       c => b%create_result()
       do s = 1,size(b%val, 2)
          !$omp simd
          do i = 1,size(b%val, 1)
             c%val(i,s) = a%val(1,1) * b%val(i,s)
          end do
       end do
    elseif(size(a%val,1).ne.size(b%val,1).and.size(a%val,2).eq.size(b%val,2))then
       if(size(a%val,1) .eq. 1)then
          c => b%create_result()
          do s = 1,size(a%val, 2)
             !$omp simd
             do i = 1,size(b%val, 1)
                c%val(i,s) = a%val(1,s) * b%val(i,s)
             end do
          end do
       elseif(size(b%val,1) .eq. 1)then
          c => a%create_result()
          do s = 1,size(a%val, 2)
             !$omp simd
             do i = 1,size(a%val, 1)
                c%val(i,s) = a%val(i,s) * b%val(1,s)
             end do
          end do
       end if
    else
       c => a%create_result()
       do s = 1, size(a%val, 2)
          !$omp simd
          do i = 1, size(a%val, 1)
             c%val(i,s) = a%val(i,s) * b%val(i,s)
          end do
       end do
    end if

    c%get_partial_left => get_partial_multiply_left
    c%get_partial_right => get_partial_multiply_right
    c%get_partial_left_val => get_partial_multiply_left_val
    c%get_partial_right_val => get_partial_multiply_right_val
    c%get_partial_left_val_sum => get_partial_multiply_left_val_sum
    c%get_partial_right_val_sum => get_partial_multiply_right_val_sum
    if(a%requires_grad .or. b%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%operation = 'multiply'
       c%left_operand => canonical(a)
       c%right_operand => canonical(b)
    end if
  end function multiply_arrays
!-------------------------------------------------------------------------------
  module function multiply_real2d(a, b) result(c)
    !! Multiply autodiff array by real array (element-wise)
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), dimension(:,:), intent(in) :: b
    type(array_type), pointer :: c

    integer :: s, i

    c => a%create_result()
    do s = 1,size(a%val, 2)
       !$omp simd
       do i = 1,size(a%val, 1)
          c%val(i,s) = a%val(i,s) * b(i,s)
       end do
    end do

    c%get_partial_left => get_partial_multiply_left
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'multiply_real2d'
       c%left_operand => canonical(a)
    end if
    c%right_operand => constant_node(a, b)
  end function multiply_real2d
!-------------------------------------------------------------------------------
  module function real2d_multiply(a, b) result(c)
    !! Multiply scalar by autodiff array
    implicit none
    real(real32), dimension(:,:), intent(in) :: a
    class(array_type), intent(in), target :: b
    type(array_type), pointer :: c

    c => multiply_real2d(b, a)
  end function real2d_multiply
!-------------------------------------------------------------------------------
  module function multiply_scalar(a, scalar) result(c)
    !! Multiply autodiff array by scalar
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), intent(in) :: scalar
    type(array_type), pointer :: c

    integer :: i, s
    type(array_type), pointer :: na, nc

    na => canonical(a)
    c => a%create_result()
    nc => c
    do s = 1, size(na%store, 2)
       !$omp simd
       do i = 1, size(na%store, 1)
          nc%store(i,s) = na%store(i,s) * scalar
       end do
    end do

    c%get_partial_left => get_partial_multiply_left
    c%get_partial_left_val => get_partial_multiply_left_val
    c%get_partial_left_val_sum => get_partial_multiply_left_val_sum
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'multiply_scalar'
       c%left_operand => canonical(a)
    end if
    c%right_operand => constant_node(a, reshape([scalar], [1, 1]))
  end function multiply_scalar
!-------------------------------------------------------------------------------
  module function scalar_multiply(scalar, a) result(c)
    !! Multiply scalar by autodiff array
    implicit none
    real(real32), intent(in) :: scalar
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    c => multiply_scalar(a, scalar)
  end function scalar_multiply
!-------------------------------------------------------------------------------
  module function multiply_logical(a, b) result(c)
    !! Multiply two logical arrays (element-wise)
    implicit none
    class(array_type), intent(in), target :: a
    logical, dimension(:,:), intent(in) :: b
    type(array_type), pointer :: c

    integer :: s, i

    c => a%create_result()
    do s = 1,size(a%val, 2)
       !$omp simd
       do i = 1,size(a%val, 1)
          if(b(i,s))then
             c%val(i,s) = a%val(i,s)
          else
             c%val(i,s) = 0._real32
          end if
       end do
    end do
    c%mask = b

    c%get_partial_left => get_partial_multiply_logical_left
    c%get_partial_left_val => get_partial_multiply_logical_left_val
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'multiply_logical'
       c%left_operand => canonical(a)
    end if

  end function multiply_logical
!-------------------------------------------------------------------------------
  function get_partial_multiply_left(this, upstream_grad) result(output)
    !! Get partial derivative with respect to left operand
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    logical :: right_is_temporary_local
    type(array_type), pointer :: ptr

    right_is_temporary_local = this%right_operand%is_temporary
    this%right_operand%is_temporary = .false.
    if(this%right_operand%is_scalar)then
       ptr => upstream_grad * this%right_operand%val(1,1)
    else
       ptr => upstream_grad * this%right_operand
    end if
    this%right_operand%is_temporary = right_is_temporary_local
    output = ptr
  end function get_partial_multiply_left
!-------------------------------------------------------------------------------
  function get_partial_multiply_right(this, upstream_grad) result(output)
    !! Get partial derivative with respect to right operand
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    logical :: left_is_temporary_local
    type(array_type), pointer :: ptr

    left_is_temporary_local = this%left_operand%is_temporary
    this%left_operand%is_temporary = .false.
    if(this%left_operand%is_scalar)then
       ptr => upstream_grad * this%left_operand%store(1,1)
    else
       ptr => upstream_grad * this%left_operand
    end if
    this%left_operand%is_temporary = left_is_temporary_local
    output = ptr
  end function get_partial_multiply_right
!-------------------------------------------------------------------------------
  pure subroutine get_partial_multiply_left_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: i, s
    if(this%right_operand%is_scalar)then
       do s = 1, size(output, 2)
          !$omp simd
          do i = 1, size(output, 1)
             output(i,s) = upstream_grad(i,s) * this%right_operand%store(1,1)
          end do
       end do
    elseif(.not.this%left_operand%is_sample_dependent)then
       do s = 1 , size(output, 2)
          !$omp simd
          do i = 1 , size(output, 1)
             output(i,s) = upstream_grad(i,s) * this%right_operand%store(i,1)
          end do
       end do
    elseif(.not.this%right_operand%is_sample_dependent)then
       do s = 1 , size(output, 2)
          !$omp simd
          do i = 1 , size(output, 1)
             output(i,s) = upstream_grad(i,s) * this%right_operand%store(i,1)
          end do
       end do
    elseif(size(upstream_grad,2).ne.size(output,2))then
       do s = 1 , size(output, 2)
          !$omp simd
          do i = 1 , size(output, 1)
             output(i,s) = upstream_grad(i,s) * this%right_operand%val(i,1)
          end do
       end do
    else
       do s = 1, size(output, 2)
          !$omp simd
          do i = 1, size(output, 1)
             output(i,s) = upstream_grad(i,s) * this%right_operand%val(i,s)
          end do
       end do
    end if
  end subroutine get_partial_multiply_left_val
!-------------------------------------------------------------------------------
  pure subroutine get_partial_multiply_right_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: i, s
    if(this%left_operand%is_scalar)then
       do s = 1, size(output, 2)
          !$omp simd
          do i = 1, size(output, 1)
             output(i,s) = upstream_grad(i,s) * this%left_operand%store(1,1)
          end do
       end do
    elseif(.not.this%left_operand%is_sample_dependent)then
       do s = 1 , size(output, 2)
          !$omp simd
          do i = 1 , size(output, 1)
             output(i,s) = upstream_grad(i,s) * this%left_operand%store(i,1)
          end do
       end do
    elseif(.not.this%right_operand%is_sample_dependent)then
       do s = 1 , size(output, 2)
          !$omp simd
          do i = 1 , size(output, 1)
             output(i,s) = upstream_grad(i,s) * this%left_operand%store(i,1)
          end do
       end do
    elseif(size(upstream_grad,2).ne.size(output,2))then
       do s = 1 , size(output, 2)
          !$omp simd
          do i = 1 , size(output, 1)
             output(i,s) = upstream_grad(i,s) * this%left_operand%store(i,1)
          end do
       end do
    else
       do s = 1, size(output, 2)
          !$omp simd
          do i = 1, size(output, 1)
             output(i,s) = upstream_grad(i,s) * this%left_operand%val(i,s)
          end do
       end do
    end if
  end subroutine get_partial_multiply_right_val
!-------------------------------------------------------------------------------
  pure subroutine get_partial_multiply_left_val_sum(this, upstream_grad, output)
    !! Fused partial+sum for multiply left: output(:,1) = sum_s(upstream*right_op)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:), intent(out) :: output

    integer :: s, n_samples

    n_samples = size(upstream_grad, 2)
    if(this%right_operand%is_scalar)then
       output(:) = upstream_grad(:,1)
       do s = 2, n_samples
          output(:) = output(:) + upstream_grad(:,s)
       end do
       output(:) = output(:) * this%right_operand%val(1,1)
    elseif(.not.this%left_operand%is_sample_dependent .or. &
         .not.this%right_operand%is_sample_dependent)then
       output(:) = upstream_grad(:,1) * this%right_operand%val(:,1)
       do s = 2, n_samples
          output(:) = output(:) + upstream_grad(:,s) * this%right_operand%val(:,1)
       end do
    else
       output(:) = upstream_grad(:,1) * this%right_operand%val(:,1)
       do s = 2, n_samples
          output(:) = output(:) + upstream_grad(:,s) * this%right_operand%val(:,s)
       end do
    end if
  end subroutine get_partial_multiply_left_val_sum
!-------------------------------------------------------------------------------
  pure subroutine get_partial_multiply_right_val_sum(this, upstream_grad, output)
    !! Fused partial+sum for multiply right: output(:,1) = sum_s(upstream*left_op)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:), intent(out) :: output

    integer :: s, n_samples

    n_samples = size(upstream_grad, 2)
    if(this%left_operand%is_scalar)then
       output(:) = upstream_grad(:,1)
       do s = 2, n_samples
          output(:) = output(:) + upstream_grad(:,s)
       end do
       output(:) = output(:) * this%left_operand%store(1,1)
    elseif(.not.this%left_operand%is_sample_dependent .or. &
         .not.this%right_operand%is_sample_dependent)then
       output(:) = upstream_grad(:,1) * this%left_operand%val(:,1)
       do s = 2, n_samples
          output(:) = output(:) + upstream_grad(:,s) * this%left_operand%val(:,1)
       end do
    else
       output(:) = upstream_grad(:,1) * this%left_operand%val(:,1)
       do s = 2, n_samples
          output(:) = output(:) + upstream_grad(:,s) * this%left_operand%val(:,s)
       end do
    end if
  end subroutine get_partial_multiply_right_val_sum
!-------------------------------------------------------------------------------
  function get_partial_multiply_logical_left(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    ptr => upstream_grad * this%mask
    output = ptr
  end function get_partial_multiply_logical_left
!-------------------------------------------------------------------------------
  pure subroutine get_partial_multiply_logical_left_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: i, s

    do s = 1,size(output, 2)
       !$omp simd
       do i = 1,size(output, 1)
          if(this%mask(i,s))then
             output(i,s) = upstream_grad(i,s)
          else
             output(i,s) = 0._real32
          end if
       end do
    end do
  end subroutine get_partial_multiply_logical_left_val
!###############################################################################


!###############################################################################
  module function divide_arrays(a, b) result(c)
    !! Divide two autodiff arrays (element-wise)
    implicit none
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    integer :: i, s

    if(.not.b%is_sample_dependent)then
       c => a%create_result()
       do s = 1,size(a%val, 2)
          !$omp simd
          do i = 1,size(a%val, 1)
             c%val(i,s) = a%val(i,s) / b%val(i,1)
          end do
       end do
    elseif(all(shape(a%val) .eq. shape(b%val)))then
       c => a%create_result()
       c%val = a%val / b%val
    elseif(size(a%val,1).ne.size(b%val,1).and.size(a%val,2).eq.size(b%val,2))then
       if(size(a%val,1) .eq. 1)then
          c => b%create_result()
          do s = 1,size(a%val, 2)
             !$omp simd
             do i = 1,size(b%val, 1)
                c%val(i,s) = a%val(1,s) / b%val(i,s)
             end do
          end do
       elseif(size(b%val,1) .eq. 1)then
          c => a%create_result()
          do s = 1,size(a%val, 2)
             !$omp simd
             do i = 1,size(a%val, 1)
                c%val(i,s) = a%val(i,s) / b%val(1,s)
             end do
          end do
       end if
    end if

    c%get_partial_left => get_partial_divide_left
    c%get_partial_right => get_partial_divide_right
    c%get_partial_left_val => get_partial_divide_left_val
    c%get_partial_right_val => get_partial_divide_right_val
    if(a%requires_grad .or. b%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%operation = 'divide'
       c%left_operand => canonical(a)
       c%right_operand => canonical(b)
    end if
  end function divide_arrays
!-------------------------------------------------------------------------------
  module function divide_scalar(a, scalar) result(c)
    !! Divide autodiff array by scalar
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), intent(in) :: scalar
    type(array_type), pointer :: c

    integer :: i, s
    real(real32) :: scalar_tmp

    c => a%create_result()
    scalar_tmp = 1._real32 / scalar
    do i = 1,size(a%val, 1)
       do s = 1,size(a%val, 2)
          c%val(i,s) = a%val(i,s) * scalar_tmp
       end do
    end do

    c%get_partial_left => get_partial_divide_left
    c%get_partial_left_val => get_partial_divide_left_val
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'divide_scalar'
       c%left_operand => canonical(a)
    end if
    c%right_operand => constant_node(a, reshape([scalar], [1, 1]))
  end function divide_scalar
!-------------------------------------------------------------------------------
  module function scalar_divide(scalar, a) result(c)
    !! Divide scalar by autodiff array
    implicit none
    real(real32), intent(in) :: scalar
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = scalar / a%val

    c%get_partial_right => get_partial_divide_right
    c%get_partial_right_val => get_partial_divide_right_val
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'scalar_divide'
       c%right_operand => canonical(a)
    end if
    c%left_operand => constant_node(a, reshape([scalar], [1, 1]))
  end function scalar_divide
!-------------------------------------------------------------------------------
  module function divide_real1d(a, b) result(c)
    !! Divide autodiff array by a real array
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), dimension(:), intent(in) :: b
    type(array_type), pointer :: c

    integer :: i, s

    c => a%create_result()
    do s = 1,size(a%val, 2)
       !$omp simd
       do i = 1,size(a%val, 1)
          c%val(i,s) = a%val(i,s) / b(s)
       end do
    end do

    c%get_partial_left => get_partial_divide_left
    c%get_partial_left_val => get_partial_divide_left_val
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'divide_real1d'
       c%left_operand => canonical(a)
    end if
    c%right_operand => constant_node(a, reshape(b, [1, size(b)]))
  end function divide_real1d
!-------------------------------------------------------------------------------
  function get_partial_divide_left(this, upstream_grad) result(output)
    !! Get partial derivative with respect to left operand
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    logical :: right_is_temporary_local
    type(array_type), pointer :: ptr

    right_is_temporary_local = this%right_operand%is_temporary
    this%right_operand%is_temporary = .false.
    if(this%right_operand%is_scalar)then
       ptr => upstream_grad / this%right_operand%val(1,1)
    else
       ptr => upstream_grad / this%right_operand
    end if
    this%right_operand%is_temporary = right_is_temporary_local
    output = ptr
  end function get_partial_divide_left
!-------------------------------------------------------------------------------
  function get_partial_divide_right(this, upstream_grad) result(output)
    !! Get partial derivative with respect to right operand
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    logical :: left_is_temporary_local, right_is_temporary_local
    type(array_type), pointer :: ptr

    left_is_temporary_local = this%left_operand%is_temporary
    right_is_temporary_local = this%right_operand%is_temporary
    this%left_operand%is_temporary = .false.
    this%right_operand%is_temporary = .false.
    if(this%left_operand%is_scalar)then
       ptr => (-upstream_grad * this%left_operand%store(1,1)) / &
            (this%right_operand * this%right_operand)
    else
       ptr => (-upstream_grad * this%left_operand) / &
            (this%right_operand * this%right_operand)
    end if
    this%left_operand%is_temporary = left_is_temporary_local
    this%right_operand%is_temporary = right_is_temporary_local
    output = ptr
  end function get_partial_divide_right
!-------------------------------------------------------------------------------
  pure subroutine get_partial_divide_left_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: i, s

    if(this%right_operand%is_scalar)then
       output = upstream_grad / this%right_operand%val(1,1)
    elseif(.not.this%right_operand%is_sample_dependent)then
       do s = 1 , size(output, 2)
          !$omp simd
          do i = 1 , size(output, 1)
             output(i,s) = upstream_grad(i,s) / this%right_operand%val(i,1)
          end do
       end do
    elseif(size(upstream_grad,2).ne.size(output,2))then
       do s = 1 , size(output, 2)
          !$omp simd
          do i = 1 , size(output, 1)
             output(i,s) = upstream_grad(i,s) / this%right_operand%val(i,1)
          end do
       end do
    else
       output = upstream_grad / this%right_operand%val
    end if
  end subroutine get_partial_divide_left_val
!-------------------------------------------------------------------------------
  pure subroutine get_partial_divide_right_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: i, s

    if(this%left_operand%is_scalar)then
       do s = 1 , size(output, 2)
          !$omp simd
          do i = 1 , size(output, 1)
             output(i,s) = (-upstream_grad(i,s) * this%left_operand%store(1,1)) / &
                  (this%right_operand%val(i,1) * this%right_operand%val(i,1))
          end do
       end do
    elseif(.not.this%left_operand%is_sample_dependent)then
       do s = 1 , size(output, 2)
          !$omp simd
          do i = 1 , size(output, 1)
             output(i,s) = (-upstream_grad(i,s) * this%left_operand%store(i,1)) / &
                  (this%right_operand%val(i,1) * this%right_operand%val(i,1))
          end do
       end do
    elseif(.not.this%right_operand%is_sample_dependent)then
       do s = 1 , size(output, 2)
          !$omp simd
          do i = 1 , size(output, 1)
             output(i,s) = (-upstream_grad(i,s) * this%left_operand%store(i,s)) / &
                  (this%right_operand%val(i,1) * this%right_operand%val(i,1))
          end do
       end do
    else
       do s = 1 , size(output, 2)
          !$omp simd
          do i = 1 , size(output, 1)
             output(i,s) = (-upstream_grad(i,s) * this%left_operand%store(i,s)) / &
                  (this%right_operand%val(i,s) * this%right_operand%val(i,s))
          end do
       end do
    end if
  end subroutine get_partial_divide_right_val
!###############################################################################


!###############################################################################
  module function power_arrays(a, b) result(c)
    !! Raise autodiff array to power of another array
    implicit none
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = a%val ** b%val

    c%get_partial_left => get_partial_power_base
    c%get_partial_right => get_partial_power_exponent
    if(a%requires_grad .or. b%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%operation = 'power'
       c%left_operand => canonical(a)
       c%right_operand => canonical(b)
    end if
  end function power_arrays
!-------------------------------------------------------------------------------
  module function power_real_scalar(a, scalar) result(c)
    !! Raise autodiff array to scalar power
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), intent(in) :: scalar
    type(array_type), pointer :: c

    integer :: i, s

    c => a%create_result()
    do s = 1,size(a%val, 2)
       !$omp simd
       do i = 1,size(a%val, 1)
          c%val(i,s) = a%val(i,s) ** scalar
       end do
    end do

    c%get_partial_left => get_partial_power_base
    c%get_partial_left_val => get_partial_power_base_val
    c%get_partial_left_val_sum => get_partial_power_base_val_sum
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'power_scalar'
       c%left_operand => canonical(a)
    end if
    c%right_operand => constant_node(a, reshape([scalar], [1, 1]))

  end function power_real_scalar
!-------------------------------------------------------------------------------
  module function power_int_scalar(a, scalar) result(c)
    !! Raise autodiff array to scalar power
    implicit none
    class(array_type), intent(in), target :: a
    integer, intent(in) :: scalar
    type(array_type), pointer :: c

    c => power_real_scalar(a, real(scalar, real32))
  end function power_int_scalar
!-------------------------------------------------------------------------------
  module function scalar_power(scalar, a) result(c)
    implicit none
    real(real32), intent(in) :: scalar
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = scalar ** a%val

    c%get_partial_left => get_partial_power_base
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'scalar_power'
       c%right_operand => canonical(a)
    end if
    c%left_operand => constant_node(a, reshape([scalar], [1, 1]))

  end function scalar_power
!-------------------------------------------------------------------------------
  module function int_scalar_power(scalar, a) result(c)
    implicit none
    integer, intent(in) :: scalar
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    c => scalar_power(real(scalar, real32), a)
  end function int_scalar_power
!-------------------------------------------------------------------------------
  function get_partial_power_base(this, upstream_grad) result(output)
    !! Get the partial gradient with respect to the base of the power operation
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    logical :: left_is_temporary_local, right_is_temporary_local
    type(array_type), pointer :: ptr

    left_is_temporary_local = this%left_operand%is_temporary
    right_is_temporary_local = this%right_operand%is_temporary
    this%left_operand%is_temporary = .false.
    this%right_operand%is_temporary = .false.
    if(all(abs(this%right_operand%val - 1._real32).lt.1.E-6_real32))then
       output = upstream_grad
       return
    elseif(all(abs(this%right_operand%val - 2._real32).lt.1.E-6_real32))then
       ptr => upstream_grad * this%left_operand * 2._real32
    else
       if(this%right_operand%is_scalar)then
          ptr => upstream_grad * this%right_operand%val(1,1) * &
               this%left_operand ** ( this%right_operand%val(1,1) - 1.0_real32 )
       else
          ptr => upstream_grad * this%right_operand * &
               this%left_operand ** ( this%right_operand - 1.0_real32 )
       end if
    end if
    this%left_operand%is_temporary = left_is_temporary_local
    this%right_operand%is_temporary = right_is_temporary_local
    output = ptr
  end function get_partial_power_base
!-------------------------------------------------------------------------------
  function get_partial_power_exponent(this, upstream_grad) result(output)
    !! Get the partial gradient with respect to the exponent of the power operation
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    logical :: left_is_temporary_local, this_is_temporary_local
    type(array_type), pointer :: ptr

    left_is_temporary_local = this%left_operand%is_temporary
    this_is_temporary_local = this%is_temporary
    this%left_operand%is_temporary = .false.
    this%is_temporary = .false.
    if(this%left_operand%is_scalar)then
       ptr => upstream_grad * log(this%left_operand%store(1,1)) * this
    else
       ptr => upstream_grad * log(this%left_operand) * this
    end if
    this%left_operand%is_temporary = left_is_temporary_local
    this%is_temporary = this_is_temporary_local
    output = ptr
  end function get_partial_power_exponent
!-------------------------------------------------------------------------------
  pure subroutine get_partial_power_base_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    if(all(abs(this%right_operand%val - 1._real32).lt.1.E-6_real32))then
       output = upstream_grad
    elseif(all(abs(this%right_operand%val - 2._real32).lt.1.E-6_real32))then
       output = upstream_grad * this%left_operand%val * 2._real32
    else
       if(this%right_operand%is_scalar)then
          output = upstream_grad * this%right_operand%val(1,1) * &
               this%left_operand%val ** ( this%right_operand%val(1,1) - 1.0_real32 )
       else
          output = upstream_grad * this%right_operand%val * &
               this%left_operand%val ** ( this%right_operand%val - 1.0_real32 )
       end if
    end if
  end subroutine get_partial_power_base_val
!-------------------------------------------------------------------------------
  pure subroutine get_partial_power_base_val_sum(this, upstream_grad, output)
    !! Fused partial+sum for power base: output(:,1) = sum_s(d(a^n)/da * upstream)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:), intent(out) :: output

    integer :: s, n_samples
    real(real32) :: exp_val

    n_samples = size(upstream_grad, 2)
    if(all(abs(this%right_operand%val - 1._real32).lt.1.E-6_real32))then
       output(:) = upstream_grad(:,1)
       do s = 2, n_samples
          output(:) = output(:) + upstream_grad(:,s)
       end do
    elseif(all(abs(this%right_operand%val - 2._real32).lt.1.E-6_real32))then
       if(.not.this%left_operand%is_sample_dependent)then
          output(:) = upstream_grad(:,1)
          do s = 2, n_samples
             output(:) = output(:) + upstream_grad(:,s)
          end do
          output(:) = output(:) * this%left_operand%val(:,1) * 2._real32
       else
          output(:) = upstream_grad(:,1) * this%left_operand%val(:,1) * 2._real32
          do s = 2, n_samples
             output(:) = output(:) + &
                  upstream_grad(:,s) * this%left_operand%val(:,s) * 2._real32
          end do
       end if
    else
       exp_val = this%right_operand%val(1,1)
       if(this%right_operand%is_scalar)then
          if(.not.this%left_operand%is_sample_dependent)then
             output(:) = upstream_grad(:,1)
             do s = 2, n_samples
                output(:) = output(:) + upstream_grad(:,s)
             end do
             output(:) = output(:) * exp_val * &
                  this%left_operand%val(:,1) ** (exp_val - 1._real32)
          else
             output(:) = upstream_grad(:,1) * exp_val * &
                  this%left_operand%val(:,1) ** (exp_val - 1._real32)
             do s = 2, n_samples
                output(:) = output(:) + upstream_grad(:,s) * exp_val * &
                     this%left_operand%val(:,s) ** (exp_val - 1._real32)
             end do
          end if
       else
          output(:) = upstream_grad(:,1) * this%right_operand%val(:,1) * &
               this%left_operand%val(:,1) ** (this%right_operand%val(:,1) - 1._real32)
          do s = 2, n_samples
             output(:) = output(:) + upstream_grad(:,s) * &
                  this%right_operand%val(:,s) * &
                  this%left_operand%val(:,s) ** ( &
                       this%right_operand%val(:,s) - 1._real32 &
                  )
          end do
       end if
    end if
  end subroutine get_partial_power_base_val_sum
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!



!###############################################################################


!###############################################################################


!###############################################################################





end submodule diffstruc__arithmetic_impl
