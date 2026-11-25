program test_context
  !! Exercises context ownership of array handles.
  !!
  !! With no argument the program checks that handles are non-owning views:
  !! copying a handle, reassigning it, and letting it go out of scope never
  !! frees context storage, and releasing a graph leaves root variables
  !! usable. With an argument it exercises a failure mode that must abort:
  !!
  !!   stale  -- use a handle to a node the context has released
  !!   guard  -- use a handle that was never bound to a node
  use coreutils, only: real32
  use diffstruc
  implicit none

  type(array_type) :: x, y
  type(array_type), pointer :: f, g, released
  character(len=32) :: mode

  mode = ''
  if(command_argument_count() >= 1) call get_command_argument(1, mode)

  select case(trim(mode))
  case('stale')
     call make_released(released)
     if(associated(released%val)) then
        write(*, *) 'released node still owns storage'
        stop 1
     end if
     ! This must abort: the handle refers to a released record.
     call released%set_shape([2, 1])
     write(*, *) 'handle to a released record was accepted'
  case('guard')
     ! This must abort: the handle was never bound to a node.
     call x%set_shape([2, 1])
     write(*, *) 'unbound handle was accepted'
  case default
     call check_ownership()
  end select

contains

  subroutine make_released(handle)
    !! Builds a two-operand expression and releases its graph, handing back a
    !! handle that referred to one of the released operands.
    type(array_type), pointer, intent(out) :: handle

    call x%allocate(array_shape=[2, 1])
    call y%allocate(array_shape=[2, 1])
    x%val = 1.0_real32
    y%val = 2.0_real32
    call x%set_requires_grad(.true.)
    call y%set_requires_grad(.true.)

    f => x * y + x
    handle => f%left_operand
    call f%nullify_graph()
  end subroutine make_released

  subroutine check_ownership()
    !! Verifies the non-owning handle semantics on a real computation.
    real(real32) :: before(2, 1), after(2, 1), gradient(2, 1)
    type(array_type) :: copy

    call x%allocate(array_shape=[2, 1])
    call y%allocate(array_shape=[2, 1])
    x%val(:, 1) = [1.0_real32, 2.0_real32]
    y%val(:, 1) = [3.0_real32, 4.0_real32]
    call x%set_requires_grad(.true.)
    call y%set_requires_grad(.true.)

    ! Assigning a handle to an unbound handle shares the context-owned record:
    ! no storage is copied, and both handles view the same values.
    copy = x
    if(copy%id /= x%id) then
       write(*, *) 'assignment did not share the record'
       stop 1
    end if
    copy%val(1, 1) = 9.0_real32
    if(abs(x%val(1, 1) - 9.0_real32) > 1.0e-6_real32) then
       write(*, *) 'shared record was not visible through the source handle'
       stop 1
    end if
    x%val(1, 1) = 1.0_real32

    ! Differentiation leaves the inputs intact and their storage context-owned.
    f => x * y + x
    before = x%val
    call f%grad_reverse(reset_graph=.true.)
    after = x%val
    if(any(abs(before - after) > 1.0e-6_real32)) then
       write(*, *) 'inputs changed during differentiation'
       stop 1
    end if
    if(.not. associated(x%grad)) then
       write(*, *) 'no gradient accumulated'
       stop 1
    end if
    gradient = x%grad%val

    ! Releasing the graph keeps the root variables alive.
    g => x + y
    call g%nullify_graph()
    x%val(2, 1) = 5.0_real32
    y%val(2, 1) = 6.0_real32
    if(abs(x%val(2, 1) - 5.0_real32) > 1.0e-6_real32) then
       write(*, *) 'root variable storage was released'
       stop 1
    end if

    write(*, '(A)') 'context ownership checks passed'
    write(*, '(A,2F8.3)') ' x = ', x%val(:, 1)
    write(*, '(A,2F8.3)') ' y = ', y%val(:, 1)
    write(*, '(A,2F8.3)') ' d(x*y+x)/dx = ', gradient(:, 1)
  end subroutine check_ownership

end program test_context
