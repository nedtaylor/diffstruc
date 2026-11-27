program test_context
  !! Exercises context ownership of array handles.
  !!
  !! With no argument the program checks that handles are non-owning views:
  !! copying a handle, reassigning it, and letting it go out of scope never
  !! frees context storage, and releasing a graph leaves root variables
  !! usable. With an argument it exercises a failure mode that must abort:
  !!
  !!   stale    -- use a handle to a node the context has released
  !!   recycled -- use a released handle after its slot was reused
  !!   guard    -- use a handle that was never bound to a node
  use coreutils, only: real32
  use diffstruc
  implicit none

  type(array_type) :: x, y, saved, fresh
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
  case('recycled')
     ! A released record's slot is recycled by the next allocation. A handle
     ! that still points at the old record must be rejected, not resolve to
     ! the new occupant of the slot.
     call x%allocate(array_shape=[2, 1])
     x%val = 1.0_real32
     saved = x
     call x%deallocate(recycle=.true.)
     call fresh%allocate(array_shape=[4, 1])
     ! This must abort: the handle refers to the record that used the slot.
     call saved%set_shape([2, 1])
     write(*, *) 'handle to a recycled record was accepted'
  case('guard')
     ! This must abort: the handle was never bound to a node.
     call x%set_shape([2, 1])
     write(*, *) 'unbound handle was accepted'
  case('release')
     call check_release_retention()
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

  subroutine check_release_retention()
    !! Releasing a consumed loss must not retain its root gradient record.
    integer, parameter :: warmup = 3
    integer, parameter :: iterations = 5000
    integer :: c0, l0, f0, c1, l1, f1

    call x%allocate(array_shape=[2, 1], temporary=.false.)
    call y%allocate(array_shape=[2, 1], temporary=.false.)
    x%val = 1.0_real32
    y%val = 2.0_real32
    call x%set_requires_grad(.true.)
    call y%set_requires_grad(.true.)

    ! Temporary expression root, as used by a manual training loop.
    call run_release_loop(.false., warmup)
    call context_stats(context_of(x), c0, l0, f0)
    call run_release_loop(.false., iterations)
    call context_stats(context_of(x), c1, l1, f1)
    write(*, '(A,3I8,A,3I8)') &
         'temporary root: before chunks/live/free ', c0, l0, f0, &
         ', after ', c1, l1, f1
    ! The free list is diagnostic only; its length may vary with allocation
    ! order. Chunks and live records must be exactly flat in steady state.
    if(c1 /= c0 .or. l1 /= l0) then
       write(*, *) 'temporary root retained context records'
       stop 1
    end if

    ! Non-temporary root, as used by network%train.
    call run_release_loop(.true., warmup)
    call context_stats(context_of(x), c0, l0, f0)
    call run_release_loop(.true., iterations)
    call context_stats(context_of(x), c1, l1, f1)
    write(*, '(A,3I8,A,3I8)') &
         'train-style root: before chunks/live/free ', c0, l0, f0, &
         ', after ', c1, l1, f1
    if(c1 /= c0 .or. l1 /= l0) then
       write(*, *) 'train-style root retained context records'
       stop 1
    end if

    write(*, '(A)') 'release retention checks passed'
  end subroutine check_release_retention

  subroutine run_release_loop(train_root, count)
    !! Runs a consumed-loss loop, optionally marking the root non-temporary.
    logical, intent(in) :: train_root
    integer, intent(in) :: count
    type(array_type), pointer :: loss
    integer :: i

    do i = 1, count
       loss => sum(squared(x * y + x))
       if(train_root) then
          if(associated(loss%is_temporary)) loss%is_temporary = .false.
       end if
       call loss%grad_reverse()
       call loss%release_expression()
       nullify(loss)
    end do
  end subroutine run_release_loop

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
