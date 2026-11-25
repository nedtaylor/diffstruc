program test_porting
  !! Checks the allocation, assignment and custom-operation behaviour that
  !! user code relies on after the context-ownership migration.
  !!
  !! Each check corresponds to an item in `docs/source/porting.rst`.
  use coreutils, only: real32
  use diffstruc
  implicit none

  call check_allocation()
  call check_set_shape()
  call check_source()
  call check_extract()
  call check_assignment()
  call check_lifecycle()
  call check_custom_operation()
  write(*, '(A)') 'porting checks passed'

contains

  subroutine check_allocation()
    type(array_type) :: w
    call w%allocate(array_shape=[2, 2, 1], sample_dependent=.false., &
         temporary=.false.)
    if(any(w%shape /= [2, 2])) then
       write(*, *) 'allocate([2,2,1]) did not set shape=[2,2]'
       stop 1
    end if
    if(w%rank /= 2) then
       write(*, *) 'allocate([2,2,1]) did not set rank=2'
       stop 1
    end if
    if(any(shape(w%val) /= [4, 1])) then
       write(*, *) 'allocate([2,2,1]) did not give (4,1) storage'
       stop 1
    end if
    if(w%is_sample_dependent) then
       write(*, *) 'sample_dependent=.false. was not applied'
       stop 1
    end if
    if(w%is_temporary) then
       write(*, *) 'temporary=.false. was not applied'
       stop 1
    end if
    call w%deallocate()
  end subroutine check_allocation

  subroutine check_set_shape()
    type(array_type) :: w
    call w%allocate(array_shape=[4, 1])
    call w%set_shape([2, 2])
    if(any(w%shape /= [2, 2]) .or. w%rank /= 2) then
       write(*, *) 'set_shape did not update shape and rank'
       stop 1
    end if
    call w%deallocate()
  end subroutine check_set_shape

  subroutine check_source()
    type(array_type) :: a, b
    real(real32) :: matrix(2, 2)
    real(real32) :: column(4)
    integer :: i

    do i = 1, 4
       column(i) = real(i, real32)
    end do
    matrix = reshape(column, [2, 2])

    ! A logical (2,2) source is accepted for the flattened storage view.
    call a%allocate(array_shape=[2, 2, 1], source=matrix)
    if(any(a%val(:, 1) /= column)) then
       write(*, *) 'matrix source was not flattened column-major'
       stop 1
    end if
    call a%deallocate()

    ! Exact-shape sources keep working.
    call b%allocate(array_shape=[4, 1], source=reshape(column, [4, 1]))
    if(any(b%val(:, 1) /= column)) then
       write(*, *) 'exact-shape source was not copied'
       stop 1
    end if
    call b%deallocate()
  end subroutine check_source

  subroutine check_extract()
    type(array_type) :: m, v
    real(real32), allocatable :: a1(:), a2(:,:), a3(:,:,:)
    integer :: i

    call m%allocate(array_shape=[2, 3, 1])
    m%val(:, 1) = [(real(i, real32), i = 1, 6)]

    call m%extract(a3)
    if(any(shape(a3) /= [2, 3, 1])) then
       write(*, *) 'extract to rank 3 did not restore the logical shape'
       stop 1
    end if
    if(a3(1, 1, 1) /= 1.0_real32 .or. a3(2, 3, 1) /= 6.0_real32) then
       write(*, *) 'extract to rank 3 returned the wrong values'
       stop 1
    end if
    call m%extract(a2)
    if(any(shape(a2) /= [2, 3]) .or. a2(2, 3) /= 6.0_real32) then
       write(*, *) 'extract to rank 2 did not restore a single-sample matrix'
       stop 1
    end if
    call m%deallocate()

    ! A rank-1 array keeps its sample dimension.
    call v%allocate(array_shape=[4, 2])
    v%val = 7.0_real32
    call v%extract(a2)
    if(any(shape(a2) /= [4, 2])) then
       write(*, *) 'extract to rank 2 lost the sample dimension'
       stop 1
    end if
    call v%extract(a1)
    if(size(a1) /= 8) then
       write(*, *) 'extract to rank 1 did not flatten the array'
       stop 1
    end if
    call v%deallocate()
  end subroutine check_extract

  subroutine check_assignment()
    type(array_type) :: x, shared, copied

    call x%allocate(array_shape=[3, 1])
    x%val(:, 1) = [1.0_real32, 2.0_real32, 3.0_real32]

    ! Assignment to an unbound handle shares the context-owned record.
    shared = x
    shared%val(1, 1) = 99.0_real32
    if(x%val(1, 1) /= 99.0_real32) then
       write(*, *) 'unbound assignment did not share the record'
       stop 1
    end if
    x%val(1, 1) = 1.0_real32

    ! Assignment into an already-bound handle copies the values.
    call copied%allocate(array_shape=[3, 1])
    copied = x
    copied%val(1, 1) = 77.0_real32
    if(x%val(1, 1) /= 1.0_real32) then
       write(*, *) 'bound assignment did not copy the values'
       stop 1
    end if
    call copied%deallocate()
    call x%deallocate()
  end subroutine check_assignment

  subroutine check_lifecycle()
    type(array_type) :: x
    type(array_type), pointer :: f

    call x%allocate(array_shape=[2, 1], temporary=.false.)
    x%val(:, 1) = [1.0_real32, 2.0_real32]
    call x%set_requires_grad(.true.)
    f => x * 2.0_real32
    call f%nullify_graph()
    ! Releasing a handle that was already unbound by nullify_graph is a no-op.
    call f%deallocate()
    ! The root variable is still alive and usable.
    if(any(x%val(:, 1) /= [1.0_real32, 2.0_real32])) then
       write(*, *) 'root variable storage did not survive nullify_graph'
       stop 1
    end if
    call x%deallocate()
  end subroutine check_lifecycle

  subroutine check_custom_operation()
    type(array_type) :: x
    type(array_type), pointer :: y, loss

    call x%allocate(array_shape=[3, 1], sample_dependent=.true., &
         temporary=.false.)
    x%val(:, 1) = [1.0_real32, 2.0_real32, 3.0_real32]
    call x%set_requires_grad(.true.)

    y => my_square(x)
    loss => sum(y)
    loss%is_temporary = .false.
    call loss%grad_reverse(reset_graph=.true.)

    ! d/dx sum(x^2) = 2x
    if(any(abs(x%grad%val(:, 1) - 2.0_real32 * x%val(:, 1)) > 1.0e-6_real32)) then
       write(*, *) 'custom operation gradient is wrong'
       stop 1
    end if
    call loss%nullify_graph()
    call x%deallocate()
  end subroutine check_custom_operation

  function my_square(a) result(c)
    !! Minimal custom operation following docs/source/tutorials/operations.rst.
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = a%val * a%val

    c%get_partial_left => my_square_partial
    c%get_partial_left_val => my_square_partial_val
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'my_square'
       c%left_operand => a
    end if
  end function my_square

  function my_square_partial(this, upstream_grad) result(output)
    !! Graph partial derivative used by reverse mode.
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    ptr => 2.0_real32 * upstream_grad * this%left_operand
    call output%assign_and_deallocate_source(ptr)
  end function my_square_partial

  pure subroutine my_square_partial_val(this, upstream_grad, output)
    !! Value partial derivative (reserved for built-in operations).
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    output = 2.0_real32 * upstream_grad * this%left_operand%val
  end subroutine my_square_partial_val

end program test_porting
