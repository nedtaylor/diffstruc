!> The autodiff engine: construction, forward-mode differentiation,
!! reverse-mode differentiation, graph lifecycle, and the reduction
!! operations of `diffstruc__types`.
!!
!! Every node the engine creates lives in a context. The engine never frees
!! a node a handle may still view; instead it releases whole regions when a
!! graph is detached (`nullify_graph`) or when the context is rolled back.
submodule(diffstruc__types) diffstruc__types_submodule
  use, intrinsic :: iso_fortran_env, only: error_unit
  use diffstruc__kernels, only: copy2, axpy2, sum_columns
  implicit none

contains


!###############################################################################
  integer function next_visit(node) result(tag)
    !! Issues the next traversal stamp of a node's context.
    type(array_type), intent(in) :: node
    !! Node whose context supplies the counter

    type(context_state), pointer :: state

    state => context_state_of(context_of_node(node))
    state%visit_counter = state%visit_counter + 1
    tag = state%visit_counter
  end function next_visit
!###############################################################################


!###############################################################################
  integer function recursion_limit(node) result(limit)
    !! The graph traversal depth limit of a node's context.
    type(array_type), intent(in) :: node
    !! Node whose context supplies the limit

    type(context_state), pointer :: state

    state => context_state_of(context_of_node(node))
    limit = state%max_recursion_depth
  end function recursion_limit


!###############################################################################
  module function kernel_team(this, n_elem, n_samples) result(nth)
    !! Threads a kernel should use, or 1 inside an existing team.
    class(array_type), intent(in) :: this
    integer, intent(in) :: n_elem, n_samples

    integer :: nth

    nth = team_size_for(this, n_elem, n_samples)
    if(nth > 1) then
       if(in_parallel()) nth = 1
    end if
  end function kernel_team
!###############################################################################


!###############################################################################
  pure module function team_size_for(this, n_elem, n_samples) result(nth)
    !! Threads a kernel on this array should request (1 = serial).
    class(array_type), intent(in) :: this
    integer, intent(in) :: n_elem, n_samples

    integer :: requested, available, nth

    requested = 0
    available = 1
    if(associated(this%ctx%state)) then
       requested = this%ctx%state%n_threads
       available = this%ctx%state%n_available
    end if
    nth = team_size(n_elem, n_samples, requested, available)
  end function team_size_for
!###############################################################################


!###############################################################################
  module function constructor() result(this)
    !! Constructor for array_type: an unbound handle.
    type(array_type) :: this

    this%id = 0
    nullify(this%ctx%state)
    this%ctx%gen = 1
    this%gen = 1
  end function constructor
!###############################################################################


!###############################################################################
  module subroutine set_requires_grad(this, requires_grad)
    !! Marks this array as an independent variable of the graph.
    class(array_type), intent(inout) :: this
    !! Handle to mark
    logical, intent(in) :: requires_grad
    !! Whether gradients are required

    ! Local variables
    type(array_type), pointer :: node

    node => live(this)
    node%requires_grad = requires_grad
    if(requires_grad .and. .not. associated(node%grad)) then
       node%grad => ensure_gradient(context_of_node(node), node, size(node%val, 2))
    end if
    call bind_handle(this, node)
  end subroutine set_requires_grad
!###############################################################################


!###############################################################################
  module subroutine set_shape(this, array_shape)
    !! Updates the logical shape of an array in its context record.
    class(array_type), intent(inout) :: this
    !! Handle to update
    integer, dimension(:), intent(in) :: array_shape
    !! New logical shape

    ! Local variables
    type(array_type), pointer :: node

    node => live(this)
    node%shape = array_shape
    node%size = product(array_shape)
    node%rank = max(1, size(array_shape))
    call bind_handle(this, node)
  end subroutine set_shape
!###############################################################################


!###############################################################################
  module subroutine set_direction(this, direction)
    !! Sets the forward-mode seed direction of a variable.
    class(array_type), intent(inout) :: this
    !! Variable to seed
    real(real32), dimension(:), intent(in) :: direction
    !! The seed vector

    ! Local variables
    type(array_type), pointer :: node

    node => live(this)
    if(allocated(node%direction)) deallocate(node%direction)
    allocate(node%direction(size(direction)))
    node%direction = direction
  end subroutine set_direction
!###############################################################################


!###############################################################################
  pure logical function is_variable(node)
    !! Whether a record is a root variable of the graph.
    type(array_type), intent(in) :: node
    !! Record to test

    is_variable = node%requires_grad .and. .not. associated(node%left_operand) &
         .and. .not. associated(node%right_operand)
  end function is_variable
!###############################################################################


!###############################################################################
  logical function same_node(a, b)
    !! Whether two handles refer to the same context node.
    class(array_type), intent(in) :: a, b
    !! Handles to compare

    same_node = a%id > 0 .and. a%id == b%id .and. a%gen == b%gen
  end function same_node
!###############################################################################


!###############################################################################
  module function grad_forward(this, variable) result(output)
    !! Forward-mode differentiation: the derivative of this array with
    !! respect to a variable.
    class(array_type), intent(inout) :: this
    !! Array to differentiate
    type(array_type), intent(in) :: variable
    !! Variable to differentiate with respect to
    type(array_type), pointer :: output
    !! The derivative

    ! Local variables
    integer :: depth, visit
    type(array_type), pointer :: node

    nullify(output)
    if(.not. variable%requires_grad) then
       call print_warning('Variable does not require grad in forward mode')
       return
    end if
    node => canonical(this)
    if(associated(node)) then
       call checkpoint_context(context_of_node(node))
    end if
    depth = 0
    output => forward_over_reverse(this, variable, depth)
    if(associated(output)) then
       output%is_forward = .true.
       ! Release every node the traversal built and then discarded.
       visit = next_visit(output)
       call mark_graph(output, visit)
       call sweep_region(context_of_node(node), visit)
    end if
    if(associated(node)) then
       node%requires_grad = .true.
       call bind_handle(this, node)
    end if
  end function grad_forward
!###############################################################################


!###############################################################################
  recursive function forward_over_reverse(this, variable, depth) result(output)
    !! Recursive forward-mode traversal of the computation graph.
    type(array_type), intent(inout) :: this
    !! Node being differentiated
    type(array_type), intent(in) :: variable
    !! Variable to differentiate with respect to
    integer, intent(inout) :: depth
    !! Current recursion depth
    type(array_type), pointer :: output
    !! The derivative node

    ! Local variables
    type(array_type), pointer :: node, left_deriv, right_deriv
    type(array_type), pointer :: left_tmp, right_tmp
    logical :: was_forward, left_is_variable, right_is_variable
    logical :: has_left, has_right, has_left_partial, has_right_partial
    integer :: s, i, n_elem, n_samples

    depth = depth + 1
    if(depth > recursion_limit(this)) then
       write(error_unit, '(A,I0)') 'MAX RECURSION DEPTH REACHED ', depth
       nullify(output)
       return
    end if

    node => live(this)
    was_forward = node%is_forward
    node%is_forward = .true.
    has_left = associated(node%left_operand)
    has_right = associated(node%right_operand)

    if(same_node(this, variable)) then
       output => clone_handle(node)
       output%is_temporary = .true.
       n_elem = size(output%val, 1)
       n_samples = size(output%val, 2)
       if(allocated(node%direction)) then
          do concurrent(s = 1:n_samples, i = 1:n_elem)
             output%val(i, s) = node%direction(i)
          end do
       else
          output%val = 1.0_real32
       end if
       if(allocated(output%direction)) deallocate(output%direction)
       output%requires_grad = .false.
       output%grad => null()

    else if(has_left .or. has_right) then
       has_left_partial = associated(node%get_partial_left)
       has_right_partial = associated(node%get_partial_right)
       left_is_variable = .false.
       right_is_variable = .false.

       if(has_left .and. has_left_partial) then
          left_is_variable = .true.
          left_tmp => forward_over_reverse(node%left_operand, variable, depth)
          if(trim(node%operation) == 'add') then
             left_deriv => left_tmp
          else if(has_right_partial .and. .not. has_right) then
             left_deriv => partial_node(node, left_tmp, 2)
          else
             left_deriv => partial_node(node, left_tmp, 1)
          end if
       end if

       if(has_right .and. has_right_partial) then
          right_is_variable = .true.
          right_tmp => forward_over_reverse(node%right_operand, variable, depth)
          if(trim(node%operation) == 'add') then
             right_deriv => right_tmp
          else
             right_deriv => partial_node(node, right_tmp, 2)
          end if
       end if

       if(left_is_variable .and. right_is_variable) then
          output => left_deriv + right_deriv
       else if(left_is_variable) then
          output => left_deriv
       else if(right_is_variable) then
          output => right_deriv
       else
          ! This subtree does not depend on the variable: its derivative is zero.
          output => clone_handle(node)
          output%is_temporary = .true.
          output%val = 0.0_real32
          output%requires_grad = .false.
          output%grad => null()
       end if

    else
       output => clone_handle(node)
       output%is_temporary = .true.
       output%requires_grad = .false.
       output%val = 0.0_real32
    end if

    node%is_forward = was_forward
    if(associated(output)) output%is_forward = .true.
  end function forward_over_reverse
!###############################################################################


!###############################################################################
  function partial_node(node, upstream, side) result(deriv)
    !! The derivative of a node's operation with respect to one operand,
    !! evaluated at the given upstream derivative.
    type(array_type), intent(inout), target :: node
    !! Operation node
    type(array_type), intent(in) :: upstream
    !! Derivative of the node's value
    integer, intent(in) :: side
    !! 1 for the left operand, 2 for the right operand
    type(array_type), pointer :: deriv
    !! The derivative node

    ! Local variables
    type(array_type) :: tmp

    if(side == 1) then
       tmp = node%get_partial_left(upstream)
    else
       tmp = node%get_partial_right(upstream)
    end if
    deriv => canonical(tmp)
    if(.not. associated(deriv)) then
       deriv => clone_handle(tmp)
    end if
  end function partial_node
!###############################################################################


!###############################################################################
  module subroutine grad_reverse(this, reset_graph)
    !! Reverse-mode differentiation: accumulates the gradient of this array
    !! with respect to every variable in its graph.
    class(array_type), intent(inout) :: this
    !! Array to differentiate
    logical, intent(in), optional :: reset_graph
    !! Whether to clear existing gradients first

    ! Local variables
    type(array_type), pointer :: node, grad
    type(array_context_type) :: ctx
    type(context_state), pointer :: state
    integer :: visit

    node => live(this)
    if(present(reset_graph)) then
       if(reset_graph) continue
    end if
    ctx = context_of_node(node)
    state => context_state_of(ctx)
    ! A fresh pass tag makes the first contribution to each accumulator
    ! overwrite it, so existing gradient contents are irrelevant: no zeroing
    ! sweep and no read-modify-write of the gradient arrays is needed.
    state%accumulate_phase = state%accumulate_phase + 1
    state%scratch_nodes_created = .false.
#ifdef DIFFSTRUC_SLOW_REVERSE
    call zero_variable_grads(node, state%accumulate_phase)
#endif

    if(.not. associated(node%grad)) then
       ctx = context_of(this)
       grad => allocate_value_storage(ctx, 1, &
            [node%size, size(node%val, 2)])
       grad%is_scalar = node%is_scalar
       grad%is_sample_dependent = node%is_sample_dependent
       grad%requires_grad = .true.
       grad%operation = 'none'
       if(allocated(node%indices)) then
          allocate(grad%indices(size(node%indices)))
          grad%indices = node%indices
       end if
       grad%val = 1.0_real32
       node%grad => grad
    end if

    call checkpoint_context(ctx)
    if(size(node%grad%val) == 1) then
       call reverse_accumulate(node, node%grad%val, 0, .true., node%grad%val(1,1))
    else
       call reverse_accumulate(node, node%grad%val, 0)
    end if

    ! Release every record the backward pass created that is not part of the
    ! value graph. Only the records issued since the checkpoint are scanned, so
    ! this is cheap (a few dozen records) and keeps repeated passes in bounded
    ! memory. Do not guard it with the scratch-node flag: graph-building
    ! partials can allocate nodes without that flag being observable here.
    visit = next_visit(node)
    call mark_graph(node, visit)
    if(associated(node%grad)) node%grad%visit_tag = visit
    call sweep_region(ctx, visit)
  end subroutine grad_reverse
!###############################################################################


!###############################################################################
  recursive subroutine reverse_accumulate(node, upstream, depth, uniform, uvalue)
    !! Accumulates gradients by walking the graph backwards.
    type(array_type), intent(inout), target :: node
    !! Node whose operands receive gradient
    real(real32), dimension(:,:), intent(in) :: upstream
    !! Gradient of the node's value
    integer, intent(in) :: depth
    !! Current recursion depth
    logical, intent(in), optional :: uniform
    !! Whether the upstream gradient is the same value at every element
    real(real32), intent(in), optional :: uvalue
    !! That value, when `uniform` is present and true

    if(depth > recursion_limit(node)) then
       write(error_unit, '(A,I0)') 'MAX RECURSION DEPTH IN REVERSE MODE ', depth
       return
    end if

    node%is_forward = .false.

    call accumulate_side(node, node%left_operand, 1, upstream, depth, uniform, uvalue)
    call accumulate_side(node, node%right_operand, 2, upstream, depth, uniform, uvalue)
  end subroutine reverse_accumulate
!###############################################################################


!###############################################################################
  subroutine fuse_uniform_unary(node, child, uvalue, n_elem, n_samples, &
       depth, handled)
    !! Fuses `unary partial -> add (pass-through) -> multiply partial`.
    !!
    !! With a scalar upstream gradient, the derivative of
    !!
    !!   unary( w * d + other )
    !!
    !! (with `w` a single-column, non-sample-dependent operand and `d` the
    !! sample-dependent one) is evaluated in a single pass over the data: the
    !! unary partial `g`, the full partial of `d` (`g*w`), and the reduced
    !! gradients `sum_s(g*d)` and `sum_s(g)` of `w` and of the remaining addend.
    !! Only `g*w` is written; both reductions stay in registers. The
    !! intermediate unary partial is therefore never materialised and then read
    !! back three times.
    !!
    !! Every per-element expression and the order of every summation match the
    !! separate kernels exactly, including the reduction layout of a broadcast
    !! multiply (which reduces against the first column of the sample-dependent
    !! operand). The order in which operands receive their contribution also
    !! follows the unfused traversal, so repeated contributions to one
    !! accumulator are summed in the same order.
    type(array_type), intent(inout), target :: node
    !! The unary operation node
    type(array_type), pointer, intent(in) :: child
    !! Its operand (the addition)
    real(real32), intent(in) :: uvalue
    !! The (uniform) upstream gradient
    integer, intent(in) :: n_elem
    !! Number of elements
    integer, intent(in) :: n_samples
    !! Number of samples
    integer, intent(in) :: depth
    !! Current recursion depth
    logical, intent(out) :: handled
    !! Whether the fusion applied

    ! Local variables
    type(array_type), pointer :: mult, other, ml, mr, d, w
    type(array_context_type) :: ctx
    type(context_state), pointer :: state
    real(real32), dimension(:,:), pointer :: dd, dwp, oap, basis
    integer :: op, i
    logical :: mult_left

    handled = .false.
    if(n_samples <= 1) return
    if(.not. associated(child)) return
    if(trim(child%operation) /= 'add') return
    if(allocated(child%direction)) return
    select case(trim(node%operation))
    case('tanh')
       op = 1
       basis => node%val
    case('squared')
       op = 2
       basis => child%val
    case('exp')
       op = 3
       basis => node%val
    case('negate')
       op = 4
       basis => node%val
    case default
       return
    end select
    if(size(basis, 1) /= n_elem .or. size(basis, 2) /= n_samples) return

    ! The addition must have a multiply operand; the other operand (if any) is
    ! the broadcast addend.
    nullify(mult, other)
    mult_left = .true.
    if(associated(child%left_operand)) then
       if(trim(child%left_operand%operation) == 'multiply') then
          mult => child%left_operand
          other => child%right_operand
       end if
    end if
    if(.not. associated(mult)) then
       if(associated(child%right_operand)) then
          if(trim(child%right_operand%operation) == 'multiply') then
             mult => child%right_operand
             other => child%left_operand
             mult_left = .false.
          end if
       end if
    end if
    if(.not. associated(mult)) return
    if(allocated(mult%direction)) return
    ml => mult%left_operand
    mr => mult%right_operand
    if(.not. associated(ml) .or. .not. associated(mr)) return
    ! Only the `broadcast * sample-dependent` order is fused: it is the shape
    ! the value-level kernels are written for, and it fixes which reduction
    ! layout (and which of the two operands carries the samples) applies.
    if(ml%is_sample_dependent) return
    if(.not. mr%is_sample_dependent) return
    w => ml
    d => mr
    if(w%is_scalar) return
    if(w%size /= n_elem .or. size(w%val, 2) /= 1) return
    if(d%size /= n_elem .or. size(d%val, 2) /= n_samples) return
    if(associated(other)) then
       if(other%is_sample_dependent) return
       if(other%size /= n_elem) return
       if(size(other%val, 2) /= 1) return
    end if

    ctx = context_of_node(node)
    state => context_state_of(ctx)
    dd => acquire_scratch(ctx, [n_elem, n_samples])
    dwp => acquire_scratch(ctx, [n_elem, 1])
    oap => acquire_scratch(ctx, [n_elem, 1])
    dwp = 0.0_real32
    oap = 0.0_real32

    call unary_add_mult_kernel(op, n_elem, n_samples, uvalue, basis, &
         w%val, d%val, dd, dwp(:, 1), oap(:, 1), &
         kernel_team(node, n_elem, n_samples))

    ! Contributions in the same order as the unfused traversal: the add visits
    ! its left operand first and the multiply visits its left operand first.
    if(.not. mult_left) call fuse_push_other()
    call fuse_push_w()
    call fuse_push_d()
    if(mult_left) call fuse_push_other()

    call release_scratch(ctx, dd)
    call release_scratch(ctx, dwp)
    call release_scratch(ctx, oap)
    handled = .true.

  contains

    subroutine fuse_push_d()
      if(.not. d%requires_grad) return
      if(is_variable(d)) then
         d%grad => ensure_gradient(ctx, d, n_samples)
         call add_to_gradient(state%accumulate_phase, d%grad, dd)
      end if
      if(associated(d%left_operand) .or. associated(d%right_operand)) then
         call reverse_accumulate(d, dd, depth + 1)
      end if
    end subroutine fuse_push_d

    subroutine fuse_push_w()
      if(.not. w%requires_grad) return
      if(is_variable(w)) then
         w%grad => ensure_gradient(ctx, w, 1)
         call add_to_gradient(state%accumulate_phase, w%grad, dwp)
      end if
      if(associated(w%left_operand) .or. associated(w%right_operand)) then
         call reverse_accumulate(w, dwp, depth + 1)
      end if
    end subroutine fuse_push_w

    subroutine fuse_push_other()
      if(.not. associated(other)) return
      if(.not. other%requires_grad) return
      if(is_variable(other)) then
         other%grad => ensure_gradient(ctx, other, 1)
         call add_to_gradient(state%accumulate_phase, other%grad, oap)
      end if
      if(associated(other%left_operand) .or. &
           associated(other%right_operand)) then
         call reverse_accumulate(other, oap, depth + 1)
      end if
    end subroutine fuse_push_other



  end subroutine fuse_uniform_unary
!###############################################################################


!###############################################################################
  subroutine unary_add_mult_kernel(op, n_elem, n_samples, uvalue, t, &
       w, d, dd, dw, oa, nth)
    !! Unary partial, multiply partial and both reductions.
    !!
    !! Two passes keep both loops vectorisable and the summation order intact:
    !! the first writes the full partial `dd` (parallel over samples), the
    !! second accumulates the reduced gradients `dw` and `oa` (parallel over
    !! elements, samples in ascending order per element).
    integer, intent(in) :: op
    integer, intent(in) :: n_elem
    integer, intent(in) :: n_samples
    real(real32), intent(in) :: uvalue
    real(real32), dimension(:,:), intent(in) :: t
    real(real32), dimension(:,:), intent(in) :: w
    real(real32), dimension(:,:), intent(in) :: d
    real(real32), dimension(:,:), intent(out) :: dd
    real(real32), dimension(:), intent(inout) :: dw
    real(real32), dimension(:), intent(inout) :: oa
    integer, intent(in) :: nth
    real(real32) :: av
    integer :: i, s, ib, iend

    select case(op)
    case(1)
       !$omp parallel do if(nth > 1) num_threads(nth) schedule(static)
       do s = 1, n_samples
          !$omp simd
          do i = 1, n_elem
             av = uvalue * (1.0_real32 - t(i,s) * t(i,s))
             dd(i,s) = av * w(i,1)
          end do
       end do
       if(nth > 1) then
          !$omp parallel do if(nth > 1) num_threads(nth) schedule(static)
          do ib = 1, n_elem, 128
             iend = min(n_elem, ib + 127)
             do s = 1, n_samples
                !$omp simd
                do i = ib, iend
                   av = uvalue * (1.0_real32 - t(i,s) * t(i,s))
                   dw(i) = dw(i) + av * d(i,1)
                   oa(i) = oa(i) + av
                end do
             end do
          end do
       else
          do s = 1, n_samples
             !$omp simd
             do i = 1, n_elem
                av = uvalue * (1.0_real32 - t(i,s) * t(i,s))
                dw(i) = dw(i) + av * d(i,1)
                oa(i) = oa(i) + av
             end do
          end do
       end if
    case(2)
       !$omp parallel do if(nth > 1) num_threads(nth) schedule(static)
       do s = 1, n_samples
          !$omp simd
          do i = 1, n_elem
             av = uvalue * t(i,s) * 2.0_real32
             dd(i,s) = av * w(i,1)
          end do
       end do
       if(nth > 1) then
          !$omp parallel do if(nth > 1) num_threads(nth) schedule(static)
          do ib = 1, n_elem, 128
             iend = min(n_elem, ib + 127)
             do s = 1, n_samples
                !$omp simd
                do i = ib, iend
                   av = uvalue * t(i,s) * 2.0_real32
                   dw(i) = dw(i) + av * d(i,1)
                   oa(i) = oa(i) + av
                end do
             end do
          end do
       else
          do s = 1, n_samples
             !$omp simd
             do i = 1, n_elem
                av = uvalue * t(i,s) * 2.0_real32
                dw(i) = dw(i) + av * d(i,1)
                oa(i) = oa(i) + av
             end do
          end do
       end if
    case(3)
       !$omp parallel do if(nth > 1) num_threads(nth) schedule(static)
       do s = 1, n_samples
          !$omp simd
          do i = 1, n_elem
             av = uvalue * t(i,s)
             dd(i,s) = av * w(i,1)
          end do
       end do
       if(nth > 1) then
          !$omp parallel do if(nth > 1) num_threads(nth) schedule(static)
          do ib = 1, n_elem, 128
             iend = min(n_elem, ib + 127)
             do s = 1, n_samples
                !$omp simd
                do i = ib, iend
                   av = uvalue * t(i,s)
                   dw(i) = dw(i) + av * d(i,1)
                   oa(i) = oa(i) + av
                end do
             end do
          end do
       else
          do s = 1, n_samples
             !$omp simd
             do i = 1, n_elem
                av = uvalue * t(i,s)
                dw(i) = dw(i) + av * d(i,1)
                oa(i) = oa(i) + av
             end do
          end do
       end if
    case default
       !$omp parallel do if(nth > 1) num_threads(nth) schedule(static)
       do s = 1, n_samples
          !$omp simd
          do i = 1, n_elem
             av = -uvalue
             dd(i,s) = av * w(i,1)
          end do
       end do
       if(nth > 1) then
          !$omp parallel do if(nth > 1) num_threads(nth) schedule(static)
          do ib = 1, n_elem, 128
             iend = min(n_elem, ib + 127)
             do s = 1, n_samples
                !$omp simd
                do i = ib, iend
                   av = -uvalue
                   dw(i) = dw(i) + av * d(i,1)
                   oa(i) = oa(i) + av
                end do
             end do
          end do
       else
          do s = 1, n_samples
             !$omp simd
             do i = 1, n_elem
                av = -uvalue
                dw(i) = dw(i) + av * d(i,1)
                oa(i) = oa(i) + av
             end do
          end do
       end if
    end select
  end subroutine unary_add_mult_kernel


  subroutine add_to_gradient(phase, grad, values)
    !! Adds values to a gradient accumulator, overwriting stale contents.
    !!
    !! The first contribution of a pass overwrites the accumulator, so a pass
    !! never has to zero (or read) the gradient array first. Later
    !! contributions from other paths add to it. `phase` is the owning
    !! context's pass tag.
    integer, intent(in) :: phase
    !! Tag of the current backward pass
    type(array_type), intent(inout) :: grad
    !! Gradient accumulator
    real(real32), dimension(:,:), intent(in) :: values
    !! Contribution

    integer :: i, n, s, ns, nth

    n = min(size(grad%store, 2), size(values, 2))
    ns = n
    nth = kernel_team(grad, size(grad%store, 1), ns)
#ifdef DIFFSTRUC_SLOW_REVERSE
    do s = 1, n
       !$omp simd
       do i = 1, size(grad%store, 1)
          grad%store(i,s) = grad%store(i,s) + values(i,s)
       end do
    end do
#else
    if(grad%phase /= phase) then
       call copy2(grad%store, values, size(grad%store, 1), ns, nth)
       grad%phase = phase
    else
       call axpy2(grad%store, values, size(grad%store, 1), ns, nth)
    end if
#endif
  end subroutine add_to_gradient
!###############################################################################


!###############################################################################
  recursive subroutine accumulate_side(node, child, side, upstream, depth, &
       uniform, uvalue)
    !! Accumulates the contribution of one operand and recurses into it.
    !!
    !! Three paths, in order of cost:
    !!
    !! 1. A non-sample-dependent operand whose operation has a sum-reduced
    !!    value partial: the reduced gradient is written straight into the
    !!    operand's accumulator, so no (elements x samples) buffer exists.
    !! 2. A value-level partial evaluated into one pooled scratch buffer.
    !! 3. The graph-building partial, used for layouts the value-level
    !!    procedures do not cover (matmul, outer products, ...).
    type(array_type), intent(inout), target :: node
    !! Parent operation node
    type(array_type), pointer :: child
    !! The operand (may be null)
    integer, intent(in) :: side
    !! 1 for the left operand, 2 for the right operand
    real(real32), dimension(:,:), intent(in) :: upstream
    !! Gradient of the parent's value
    integer, intent(in) :: depth
    !! Current recursion depth
    logical, intent(in), optional :: uniform
    !! Whether the upstream gradient is the same value at every element
    real(real32), intent(in), optional :: uvalue
    !! That value, when `uniform` is present and true

    ! Local variables
    type(array_type) :: tmp
    type(array_type), pointer :: upnode, nleft
    type(array_context_type) :: ctx
    type(context_state), pointer :: state
    real(real32), dimension(:, :), pointer :: pval
    real(real32), dimension(:, :), pointer :: reduced
    real(real32), dimension(:, :), allocatable :: partial
    real(real32), dimension(:), allocatable :: column
    integer :: i, n_elem, ncols, n_samples, s, nth
    logical :: has_direction, value_ok, sum_ok, needs_recurse, done

    if(.not. associated(child)) return
    if(.not. child%requires_grad) return

    n_elem = child%size
    n_samples = max(size(child%val, 2), size(upstream, 2))
    has_direction = allocated(child%direction)
    needs_recurse = associated(child%left_operand) .or. &
         associated(child%right_operand)
    ctx = context_of_node(child)
    state => context_state_of(ctx)

    if(side == 1) then
       value_ok = associated(node%get_partial_left_val)
       sum_ok = associated(node%get_partial_left_val_sum)
    else
       value_ok = associated(node%get_partial_right_val)
       sum_ok = associated(node%get_partial_right_val_sum)
    end if
    value_ok = value_ok .and. (has_value_partial(node%operation) .or. &
         node%value_partial_ok)
    ! The sum-reduced partials of matmul write a vector of the operand's
    ! element count, which is exactly the layout a non-sample-dependent operand
    ! needs; they are usable even though the full value partial is not.
    sum_ok = sum_ok .and. (has_value_partial(node%operation) .or. &
         node%value_partial_ok .or. has_sum_partial(node%operation))
#ifdef DIFFSTRUC_SLOW_REVERSE
    ! Benchmark reference path: the pre-optimisation engine, which built a
    ! derivative node for every partial instead of evaluating values directly.
    value_ok = .false.
    sum_ok = .false.
#endif

    ! Fused uniform upstream: a scalar loss (`sum(y)`, `mean(y)`, `sum(y, dim)`
    ! collapsing to one element) makes the partial a single value. Consuming
    ! that value directly saves materialising a full broadcast buffer and then
    ! reading it back.
#ifndef DIFFSTRUC_SLOW_REVERSE
    if(present(uniform)) then
       if(uniform .and. .not. has_direction .and. present(uvalue)) then
          select case(trim(node%operation))
          case('sum_all_array')
             if(is_variable(child)) then
                child%grad => ensure_gradient(ctx, child, 1)
                if(child%grad%phase /= state%accumulate_phase) then
                   child%grad%val(:, 1) = uvalue * real(n_samples, real32)
                   child%grad%phase = state%accumulate_phase
                else
                   child%grad%val(:, 1) = child%grad%val(:, 1) + &
                        uvalue * real(n_samples, real32)
                end if
             end if
             if(needs_recurse) then
                call reverse_accumulate(child, upstream, depth + 1, &
                     .true., uvalue)
             end if
             return
          case('tanh', 'squared', 'exp', 'negate')
             ! A unary node whose operand is `multiply(w, d) + other` with a
             ! broadcast `w` is the common chain shape. Evaluate the unary
             ! partial, the multiply partial and both reductions in one pass.
#ifndef DIFFSTRUC_NO_FUSE
             call fuse_uniform_unary(node, child, uvalue, n_elem, n_samples, &
                  depth, done)
             if(done) return
#endif
             pval => acquire_scratch(ctx, [n_elem, n_samples])
             select case(trim(node%operation))
             case('tanh')
                do s = 1, n_samples
                   !$omp simd
                   do i = 1, n_elem
                      pval(i,s) = uvalue * &
                           (1.0_real32 - node%val(i,s) * node%val(i,s))
                   end do
                end do
             case('squared')
                ! The derivative of x*x is 2*x, i.e. twice the *operand* value.
                nleft => node%left_operand
                do s = 1, n_samples
                   !$omp simd
                   do i = 1, n_elem
                      pval(i,s) = uvalue * nleft%val(i,s) * 2.0_real32
                   end do
                end do
             case('exp')
                do s = 1, n_samples
                   !$omp simd
                   do i = 1, n_elem
                      pval(i,s) = uvalue * node%val(i,s)
                   end do
                end do
             case default
                do s = 1, n_samples
                   !$omp simd
                   do i = 1, n_elem
                      pval(i,s) = -uvalue
                   end do
                end do
             end select
             if(child%is_sample_dependent .and. n_samples > 1) then
                if(is_variable(child)) then
                   child%grad => ensure_gradient(ctx, child, n_samples)
                   call add_to_gradient(state%accumulate_phase, child%grad, pval)
                end if
                if(needs_recurse) call reverse_accumulate(child, pval, depth + 1)
             else
                allocate(column(n_elem))
                column = sum(pval, dim=2)
                if(is_variable(child)) then
                   child%grad => ensure_gradient(ctx, child, 1)
                   call add_to_gradient( &
                        state%accumulate_phase, child%grad, &
                        reshape(column, [n_elem, 1]) &
                   )
                end if
                if(needs_recurse) call reverse_accumulate(child, &
                     reshape(column, [n_elem, 1]), depth + 1)
                deallocate(column)
             end if
             call release_scratch(ctx, pval)
             return
          end select
       end if
    end if
#endif

    ! Addition passes the upstream gradient through unchanged, so the operand
    ! can consume it directly: no partial buffer is needed at all.
#ifndef DIFFSTRUC_SLOW_REVERSE
    if(trim(node%operation) == 'add' .or. &
         (side == 1 .and. trim(node%operation) == 'add_bias')) then
       if(has_direction .or. (.not. child%is_sample_dependent .and. &
            n_samples > 1)) then
          ! fall through to the buffered value path, which scales the values
       else
          if(is_variable(child)) then
             child%grad => ensure_gradient(ctx, child, max(1, n_samples))
             if(child%grad%phase /= state%accumulate_phase) then
                ncols = min(n_samples, size(child%grad%store, 2))
                do s = 1, ncols
                   !$omp simd
                   do i = 1, size(child%grad%store, 1)
                      child%grad%store(i,s) = upstream(i,s)
                   end do
                end do
                child%grad%phase = state%accumulate_phase
             else
                call add_to_gradient(state%accumulate_phase, child%grad, upstream)
             end if
          end if
          if(needs_recurse) call reverse_accumulate(child, upstream, depth + 1)
          return
       end if
    end if
#endif

    ! Path 1: reduced accumulation directly into the accumulator.
    if(sum_ok .and. .not. has_direction .and. n_samples > 1 .and. &
         .not. child%is_sample_dependent) then
#ifndef DIFFSTRUC_SLOW_REVERSE
       if(is_variable(child)) then
          child%grad => ensure_gradient(ctx, child, 1)
          if(child%grad%phase /= state%accumulate_phase) then
             ! First contribution of the pass: write it straight into the
             ! accumulator instead of buffering and adding.
             if(side == 1) then
                call node%get_partial_left_val_sum(upstream, child%grad%val(:, 1))
             else
                call node%get_partial_right_val_sum(upstream, child%grad%val(:, 1))
             end if
             child%grad%phase = state%accumulate_phase
             if(needs_recurse) call reverse_accumulate(child, &
                  spread(child%grad%val(:, 1), 2, 1), depth + 1)
             return
          end if
       end if
#endif
       reduced => acquire_scratch(ctx, [n_elem, 1])
       if(side == 1) then
          call node%get_partial_left_val_sum(upstream, reduced(:, 1))
       else
          call node%get_partial_right_val_sum(upstream, reduced(:, 1))
       end if
       if(is_variable(child)) &
            call add_to_gradient(state%accumulate_phase, child%grad, reduced)
       if(needs_recurse) call reverse_accumulate(child, reduced, depth + 1)
       call release_scratch(ctx, reduced)
       return
    end if

    ! Path 2: value-level partial into one pooled scratch buffer.
    if(value_ok) then
       ! First contribution to a variable whose accumulator already has the
       ! right width can be written straight into the accumulator, which saves
       ! a buffer and a separate add pass.
#ifndef DIFFSTRUC_SLOW_REVERSE
       if(is_variable(child)) then
          child%grad => ensure_gradient(ctx, child, max(1, n_samples))
          if(child%grad%phase /= state%accumulate_phase .and. &
               n_samples == size(child%grad%val, 2)) then
             if(side == 1) then
                call node%get_partial_left_val(upstream, child%grad%val)
             else
                call node%get_partial_right_val(upstream, child%grad%val)
             end if
             if(has_direction) then
                do s = 1, n_samples
                   child%grad%val(:, s) = child%grad%val(:, s) * child%direction
                end do
             end if
             child%grad%phase = state%accumulate_phase
             if(needs_recurse) call reverse_accumulate(child, &
                  child%grad%val, depth + 1)
             return
          end if
       end if
#endif
       pval => acquire_scratch(ctx, [n_elem, n_samples])
       if(side == 1) then
          call node%get_partial_left_val(upstream, pval)
       else
          call node%get_partial_right_val(upstream, pval)
       end if
       if(has_direction) then
          do s = 1, n_samples
             pval(:, s) = pval(:, s) * child%direction
          end do
       end if
       if(child%is_sample_dependent .and. n_samples > 1) then
          if(is_variable(child)) then
             child%grad => ensure_gradient(ctx, child, n_samples)
             call add_to_gradient(state%accumulate_phase, child%grad, pval)
          end if
          if(needs_recurse) call reverse_accumulate(child, pval, depth + 1)
       else
          allocate(column(n_elem))
          column = sum(pval, dim=2)
          if(is_variable(child)) then
             child%grad => ensure_gradient(ctx, child, 1)
             call add_to_gradient( &
                  state%accumulate_phase, child%grad, reshape(column, [n_elem, 1]) &
             )
          end if
          if(needs_recurse) call reverse_accumulate(child, &
               reshape(column, [n_elem, 1]), depth + 1)
          deallocate(column)
       end if
       call release_scratch(ctx, pval)
       return
    end if

    ! Path 3: graph-building partial (any layout).
    upnode => allocate_value_storage(ctx, 1, &
         [size(upstream, 1), size(upstream, 2)])
    upnode%val = upstream
    if(side == 1) then
       tmp = node%get_partial_left(upnode)
    else
       tmp = node%get_partial_right(upnode)
    end if
    pval => null()
    if(associated(tmp%val)) pval => tmp%val
    if(.not. associated(pval)) return

    if(size(pval, 1) /= child%size) then
       allocate(partial(child%size, n_samples))
       partial = 0.0_real32
       call partial_values(node, upstream, side, partial)
    else
       allocate(partial(size(pval, 1), size(pval, 2)))
       partial = pval
    end if
    pval => null()

    if(has_direction) then
       n_samples = size(partial, 2)
       do s = 1, n_samples
          partial(:, s) = partial(:, s) * child%direction
       end do
    end if

    n_elem = size(partial, 1)
    n_samples = size(partial, 2)
    if(child%is_sample_dependent .and. n_samples > 1) then
       if(is_variable(child)) then
          child%grad => ensure_gradient(ctx, child, n_samples)
          call add_to_gradient(state%accumulate_phase, child%grad, partial)
       end if
       if(needs_recurse) call reverse_accumulate(child, partial, depth + 1)
    else
       if(is_variable(child)) then
          child%grad => ensure_gradient(ctx, child, 1)
          allocate(column(n_elem))
          column = sum(partial, dim=2)
          call add_to_gradient( &
               state%accumulate_phase, child%grad, reshape(column, [n_elem, 1]) &
          )
          deallocate(column)
       end if
       if(needs_recurse) then
          partial = spread(sum(partial, dim=2), 2, 1)
          call reverse_accumulate(child, partial, depth + 1)
       end if
    end if
  end subroutine accumulate_side


!###############################################################################
  pure logical function has_sum_partial(operation)
    !! Whether a sum-reduced value partial is available for the operation.
    character(len=*), intent(in) :: operation

    select case(trim(operation))
    case('matmul', 'matmul_scalar')
       has_sum_partial = .true.
    case default
       has_sum_partial = .false.
    end select
  end function has_sum_partial
!###############################################################################


!###############################################################################
  pure logical function has_value_partial(operation)
    !! Whether a value-level partial derivative is available and matches the
    !! operand's storage layout for the given operation.
    character(len=*), intent(in) :: operation
    !! Operation name of the node

    select case(trim(operation))
    case('add', 'subtract', 'negate', 'multiply', 'divide', 'power', &
         'power_scalar', 'scalar_power', 'exp', 'log', 'sin', 'cos', 'tan', &
         'tanh', 'abs', 'sqrt', 'sigmoid', 'squared', 'cubed', 'log10', &
         'gaussian', 'sum_array', 'sum_all_array', 'mean_array', &
         'mean_all_array', 'multiply_scalar', 'multiply_real2d', &
         'multiply_logical', 'divide_scalar', 'divide_real1d', &
         'scalar_divide', 'subtract_scalar', 'max', 'max_scalar')
       has_value_partial = .true.
    case default
       has_value_partial = .false.
    end select
  end function has_value_partial
!###############################################################################


!###############################################################################
  function ensure_gradient(ctx, node, n_samples) result(grad)
    !! Ensures a variable has a gradient accumulator of the given width.
    type(array_context_type), intent(in) :: ctx
    !! The owning context
    type(array_type), intent(inout), target :: node
    !! The variable
    integer, intent(in) :: n_samples
    !! Number of samples the accumulator holds
    type(array_type), pointer :: grad
    !! The accumulator

    if(associated(node%grad)) then
       grad => node%grad
       return
    end if
    ! Storage is (elements, samples); flatten the logical shape so that a
    ! rank-2 variable does not get an (rows, cols) accumulator.
    grad => allocate_value_storage(ctx, 1, [node%size, max(1, n_samples)])
    grad%shape = node%shape
    grad%size = node%size
    grad%rank = node%rank
    grad%is_scalar = node%is_scalar
    grad%is_sample_dependent = node%is_sample_dependent
    grad%requires_grad = .not. node%is_scalar
    grad%operation = 'none'
    grad%val = 0.0_real32
    node%grad => grad
  end function ensure_gradient
!###############################################################################


!###############################################################################
  function context_of_node(node) result(ctx)
    !! The context that owns a record.
    type(array_type), intent(in) :: node
    !! The record
    type(array_context_type) :: ctx
    !! Its context

    if(associated(node%ctx%state)) then
       ctx%state => node%ctx%state
       ctx%gen = node%ctx%gen
    else
       ctx = default_context_of()
    end if
  end function context_of_node
!###############################################################################


!###############################################################################
  subroutine partial_values(node, upstream, side, partial)
    !! Evaluates a node's partial derivative numerically into a buffer.
    type(array_type), intent(inout), target :: node
    !! Operation node
    real(real32), dimension(:,:), intent(in) :: upstream
    !! Gradient of the node's value
    integer, intent(in) :: side
    !! Which operand
    real(real32), dimension(:,:), intent(out) :: partial
    !! Receives the partial derivative

    ! Local variables
    type(array_type) :: tmp
    type(array_type), pointer :: handle
    type(array_context_type) :: ctx
    type(context_state), pointer :: state
    logical :: has_val_partial

    if(side == 1) then
       has_val_partial = associated(node%get_partial_left_val)
    else
       has_val_partial = associated(node%get_partial_right_val)
    end if

    if(has_val_partial) then
       if(side == 1) then
          call node%get_partial_left_val(upstream, partial)
       else
          call node%get_partial_right_val(upstream, partial)
       end if
       return
    end if


    ! Fall back to the graph-building partial derivative.
    ctx = context_of_node(node)
    state => context_state_of(ctx)
    state%scratch_nodes_created = .true.
    handle => allocate_value_storage(ctx, 1, &
         [size(upstream, 1), size(upstream, 2)])
    handle%val = upstream
    if(side == 1) then
       tmp = node%get_partial_left(handle)
    else
       tmp = node%get_partial_right(handle)
    end if
    partial = tmp%val
  end subroutine partial_values
!###############################################################################


!###############################################################################
  module subroutine zero_grad(this)
    !! Zeroes this array's gradient.
    class(array_type), intent(inout) :: this
    !! Handle whose gradient is zeroed

    ! Local variables
    type(array_type), pointer :: node

    node => canonical(this)
    if(associated(node)) then
       if(associated(node%grad)) then
          node%grad%val = 0.0_real32
          node%grad%phase = 0
       end if
    end if
  end subroutine zero_grad
!###############################################################################


!###############################################################################
  recursive module subroutine zero_all_grads(this)
    !! Zeroes the gradients of every node reachable from this array.
    class(array_type), intent(inout) :: this
    !! Root of the graph

    ! Local variables
    type(array_type), pointer :: node
    integer :: visit

    node => canonical(this)
    if(.not. associated(node)) return
    visit = next_visit(node)
    call zero_grads_inner(node, visit)
  end subroutine zero_all_grads
!###############################################################################


!###############################################################################
  recursive subroutine zero_grads_inner(node, visit)
    !! Zeroes gradients below a node, visiting each record once.
    type(array_type), intent(inout) :: node
    !! Current node
    integer, intent(in) :: visit
    !! Traversal marker

    if(node%visit_tag == visit) return
    node%visit_tag = visit
    if(associated(node%left_operand)) call zero_grads_inner(node%left_operand, visit)
    if(associated(node%right_operand)) call zero_grads_inner(node%right_operand, visit)
    if(associated(node%grad)) then
       if(associated(node%grad%val)) node%grad%val = 0.0_real32
       node%grad%phase = 0
    end if
  end subroutine zero_grads_inner
!###############################################################################


!###############################################################################
  recursive subroutine mark_graph(node, visit)
    !! Marks a graph and the gradient accumulators it references.
    type(array_type), intent(inout), target :: node
    !! Current node
    integer, intent(in) :: visit
    !! Traversal stamp

    if(node%visit_tag == visit) return
    node%visit_tag = visit
    if(associated(node%grad)) node%grad%visit_tag = visit
    if(associated(node%left_operand)) call mark_graph(node%left_operand, visit)
    if(associated(node%right_operand)) call mark_graph(node%right_operand, visit)
  end subroutine mark_graph
!###############################################################################


!###############################################################################
  recursive subroutine zero_variable_grads(node, visit)
    !! Zeroes the gradient accumulators of every variable below a node.
    type(array_type), intent(inout) :: node
    !! Current node
    integer, intent(in) :: visit
    !! Traversal marker

    if(node%visit_tag == visit) return
    node%visit_tag = visit
    if(associated(node%left_operand)) &
         call zero_variable_grads(node%left_operand, visit)
    if(associated(node%right_operand)) &
         call zero_variable_grads(node%right_operand, visit)
    if(is_variable(node) .and. associated(node%grad)) node%grad%val = 0.0_real32
  end subroutine zero_variable_grads
!###############################################################################


!###############################################################################
  module subroutine reset_graph(this)
    !! Releases the gradient side of this array's graph.
    class(array_type), intent(inout) :: this
    !! Root of the graph

    call reset_graph_of(this)
  end subroutine reset_graph
!###############################################################################


!###############################################################################
  subroutine invalidate_gradients(this)
    !! Makes every gradient accumulator in the graph stale for the next pass.
    !!
    !! Because a pass overwrites each accumulator on its first contribution,
    !! clearing gradients only needs to reset a per-pass tag; no array is
    !! touched.
    class(array_type), intent(inout) :: this
    !! Root of the graph

    ! Local variables
    type(array_type), pointer :: node
    integer :: visit

    node => canonical(this)
    if(.not. associated(node)) return
    visit = next_visit(node)
    call invalidate_inner(node, visit)
  end subroutine invalidate_gradients
!###############################################################################


!###############################################################################
  recursive subroutine invalidate_inner(node, visit)
    !! Resets the pass tag of every accumulator in a graph.
    type(array_type), intent(inout) :: node
    !! Current node
    integer, intent(in) :: visit
    !! Traversal marker

    if(node%visit_tag == visit) return
    node%visit_tag = visit
    if(associated(node%grad)) node%grad%phase = 0
    if(associated(node%left_operand)) call invalidate_inner(node%left_operand, visit)
    if(associated(node%right_operand)) call invalidate_inner(node%right_operand, visit)
  end subroutine invalidate_inner
!###############################################################################


!###############################################################################
  subroutine reset_graph_of(this)
    !! Implementation of `reset_graph`.
    class(array_type), intent(inout) :: this
    !! Root of the graph

    ! Local variables
    type(array_type), pointer :: node
    integer :: visit

    node => canonical(this)
    if(.not. associated(node)) return
    visit = next_visit(node)
    call reset_grads_inner(node, visit)
  end subroutine reset_graph_of
!###############################################################################


!###############################################################################
  recursive subroutine reset_grads_inner(node, visit)
    !! Releases non-variable gradients and zeroes variable gradients.
    type(array_type), intent(inout), target :: node
    !! Current node
    integer, intent(in) :: visit
    !! Traversal marker

    ! Local variables
    type(array_context_type) :: ctx
    logical :: is_var

    if(node%visit_tag == visit) return
    node%visit_tag = visit
    is_var = is_variable(node)
    if(associated(node%left_operand)) call reset_grads_inner(node%left_operand, visit)
    if(associated(node%right_operand)) call reset_grads_inner(node%right_operand, visit)
    if(associated(node%grad)) then
       if(is_var) then
          if(associated(node%grad%val)) node%grad%val = 0.0_real32
          node%grad%phase = 0
       else
          ctx = context_of_node(node)
          call release_node(ctx, node%grad)
          node%grad => null()
       end if
    end if
  end subroutine reset_grads_inner
!###############################################################################


!###############################################################################
  module subroutine discard_sources(this, leaves)
    !! Releases the temporary expression tree retained by a bound record.
    !!
    !! By default only temporary operation nodes are released: temporary leaves
    !! are kept because another part of the same step may still read their
    !! values. Pass `leaves=.true.` when the whole expression is dead, which
    !! also releases temporary input copies.
    class(array_type), intent(inout), target :: this
    logical, intent(in), optional :: leaves
    type(array_type), pointer :: node
    logical :: release_leaves
    integer :: visit

    release_leaves = .false.
    if(present(leaves)) release_leaves = leaves
    node => canonical(this)
    if(.not. associated(node)) return
    visit = next_visit(node)
    call discard_children(node, visit, release_leaves)
  end subroutine discard_sources
!-------------------------------------------------------------------------------
  recursive subroutine discard_children(node, visit, release_leaves)
    type(array_type), intent(inout) :: node
    integer, intent(in) :: visit
    logical, intent(in) :: release_leaves

    if(node%visit_tag == visit) return
    node%visit_tag = visit
    call discard_child(node%left_operand, visit, release_leaves)
    call discard_child(node%right_operand, visit, release_leaves)
  end subroutine discard_children
!-------------------------------------------------------------------------------
  recursive subroutine discard_child(child, visit, release_leaves)
    type(array_type), pointer, intent(inout) :: child
    integer, intent(in) :: visit
    logical, intent(in) :: release_leaves

    if(.not. associated(child)) return
    if(child%is_temporary .and. (release_leaves .or. &
         associated(child%left_operand) .or. associated(child%right_operand))) then
       call discard_children(child, visit, release_leaves)
       call release_node(context_of_node(child), child, recycle=.true.)
    end if
    child => null()
  end subroutine discard_child
!###############################################################################


!###############################################################################
  module subroutine nullify_graph(this, ignore_ownership)
    !! Releases the graph structure rooted at this array.
    !!
    !! Operand and gradient records are returned to the context and the links
    !! are nullified. Root variables of the graph are kept alive: they are the
    !! persistent inputs a caller builds new expressions from. The optional
    !! `ignore_ownership` argument is accepted for source compatibility; the
    !! context owns every record, so ownership never restricts the release.
    class(array_type), intent(inout), target :: this
    !! Root of the graph to release
    logical, intent(in), optional :: ignore_ownership
    !! Ignored; retained for source compatibility

    ! Local variables
    type(array_type), pointer :: node
    integer :: visit
    logical :: release_root

    node => canonical(this)
    if(.not. associated(node)) return
    ! Decide this before the operands are released: nullifying them would make
    ! the root look like a root variable even when it is a temporary.
    release_root = node%is_temporary .and. .not. is_variable(node)
    visit = next_visit(node)
    call release_operands(node, visit)
    if(release_root) then
       ! The root of the released graph is a temporary expression node: the
       ! context takes its storage back and the handle is unbound.
       call release_node(context_of_node(node), node)
       this%id = 0
       this%allocated = .false.
       this%val => null()
       this%grad => null()
       this%left_operand => null()
       this%right_operand => null()
    else
       call bind_handle(this, node)
    end if
  end subroutine nullify_graph
!###############################################################################


!###############################################################################
  module subroutine release_expression(this)
    !! Releases an expression that has been fully consumed.
    !!
    !! Unlike `nullify_graph`, which keeps a root that looks like an owned
    !! output, this frees the whole expression and recycles every record. Use
    !! it for a loss or residual that will not be read again. The root must be
    !! a consumed expression, not a variable or handle whose storage must
    !! survive.
    class(array_type), intent(inout), target :: this
    type(array_type), pointer :: node, grad
    logical :: was_variable

    node => canonical(this)
    if(.not. associated(node)) then
       this%id = 0
       this%allocated = .false.
       this%val => null()
       return
    end if
    ! Decide this before `discard_sources` nullifies the root operands. After
    ! that call the root looks like a variable (requires_grad, no operands),
    ! so `nullify_graph` would zero rather than release its gradient and the
    ! root gradient record would leak on every pass.
    was_variable = is_variable(node)
    if(.not. was_variable .and. associated(node%grad)) then
       ! The reverse-mode root gradient is an accumulator leaf: it owns no
       ! operand graph, so detaching and recycling it here is enough.
       grad => node%grad
       node%grad => null()
       call release_node(context_of_node(node), grad, recycle=.true.)
    end if
    call discard_sources(this, leaves=.true.)
    call nullify_graph(this)
    call deallocate_array(this, recycle=.true.)
  end subroutine release_expression
!###############################################################################


!###############################################################################
  recursive subroutine release_operands(node, visit)
    !! Releases the operand subtrees of a node, keeping root variables.
    type(array_type), intent(inout), target :: node
    !! Node whose operands are released
    integer, intent(in) :: visit
    !! Traversal marker
    logical :: was_variable

    if(node%id <= 0) return
    if(node%visit_tag == visit) return
    node%visit_tag = visit
    ! Decide before the children are released: nullifying the operands would
    ! otherwise make every operation node look like a root variable.
    was_variable = is_variable(node)
    call release_child(node%left_operand, visit)
    call release_child(node%right_operand, visit)
    if(associated(node%grad)) then
       if(was_variable) then
          if(associated(node%grad%val)) node%grad%val = 0.0_real32
       else
          call release_operands(node%grad, visit)
          call release_node(context_of_node(node), node%grad)
          node%grad => null()
       end if
    end if
    node%get_partial_left => null()
    node%get_partial_right => null()
    node%get_partial_left_val => null()
    node%get_partial_right_val => null()
    node%get_partial_left_val_sum => null()
    node%get_partial_right_val_sum => null()
  end subroutine release_operands
!###############################################################################


!###############################################################################
  recursive subroutine release_child(child, visit)
    !! Releases one operand subtree, keeping root variables alive.
    type(array_type), pointer, intent(inout) :: child
    !! The operand to release
    integer, intent(in) :: visit
    !! Traversal marker
    type(array_context_type) :: ctx

    if(.not. associated(child)) return
    if(is_variable(child) .or. .not. child%is_temporary) then
       ! Keep the variable's accumulator: the next backward pass overwrites it
       ! on its first contribution, so nothing has to be cleared here.
       if(associated(child%grad)) child%grad%phase = 0
    else
       call release_operands(child, visit)
       ctx = context_of_node(child)
       call release_node(ctx, child)
    end if
    child => null()
  end subroutine release_child
!###############################################################################


!###############################################################################
  module function get_ptr_from_id(this, id) result(ptr)
    !! Finds the node of this array's graph with the given id.
    class(array_type), intent(in), target :: this
    !! Any handle of the graph
    integer, intent(in) :: id
    !! Id to look for
    type(array_type), pointer :: ptr
    !! The record

    ! Local variables
    type(array_context_type) :: ctx

    ctx = context_of(this)
    ptr => find_node(ctx, id)
  end function get_ptr_from_id
!###############################################################################


!###############################################################################
  module function duplicate_graph(this) result(output_ptr)
    !! Deep-copies this array's computation graph.
    !!
    !! Every node reachable from this array is copied into a new record of the
    !! same context. Root variables (independent inputs) are shared, so the
    !! copy differentiates with respect to the same variables.
    class(array_type), intent(inout) :: this
    !! Root of the graph to copy
    type(array_type), pointer :: output_ptr
    !! The copy

    ! Local variables
    type(array_type), pointer :: node
    type(array_context_type) :: ctx
    integer, allocatable :: map(:)

    node => live(this)
    ctx = context_of(this)
    allocate(map(max(1, node%id)))
    map = 0
    output_ptr => copy_node(node, ctx, map)

  contains

    recursive function copy_node(source, ctx, map) result(copy)
      type(array_type), intent(inout), target :: source
      type(array_context_type), intent(in) :: ctx
      integer, intent(inout) :: map(:)
      type(array_type), pointer :: copy

      if(is_variable(source)) then
         copy => ctx_find(ctx, source%id)
         return
      end if
      if(source%id > 0 .and. source%id <= size(map)) then
         if(map(source%id) > 0) then
            copy => ctx_find(ctx, map(source%id))
            return
         end if
      end if
      copy => allocate_node(ctx)
      call clone_record(copy, source)
      ! A duplicated graph is transient: it exists only for as long as the
      ! caller keeps it, so its records must not look like owned outputs when
      ! the graph is released.
      copy%is_temporary = .true.
      if(source%id > 0 .and. source%id <= size(map)) map(source%id) = copy%id
      if(associated(source%left_operand)) &
           copy%left_operand => copy_node(source%left_operand, ctx, map)
      if(associated(source%right_operand)) &
           copy%right_operand => copy_node(source%right_operand, ctx, map)
      if(associated(source%grad)) &
           copy%grad => copy_node(source%grad, ctx, map)
    end function copy_node

  end function duplicate_graph
!###############################################################################


!###############################################################################
  function ctx_find(ctx, id) result(node)
    !! Finds a record of a context by id.
    type(array_context_type), intent(in) :: ctx
    integer, intent(in) :: id
    type(array_type), pointer :: node

    node => find_node(ctx, id)
  end function ctx_find
!###############################################################################


!###############################################################################
  module subroutine print_graph(this, print_location)
    !! Prints the computation graph rooted at this array.
    class(array_type), intent(in) :: this
    !! Root of the graph
    logical, intent(in), optional :: print_location
    !! Whether to print node addresses

    ! Local variables
    type(array_type), pointer :: node
    logical :: show_location

    show_location = .false.
    if(present(print_location)) show_location = print_location
    node => canonical(this)
    if(.not. associated(node)) then
       write(*, '(A)') 'empty graph'
       return
    end if
    call print_graph_inner(node, next_visit(node), 0, show_location)
  end subroutine print_graph
!###############################################################################


!###############################################################################
  recursive subroutine print_graph_inner(node, visit, depth, show_location)
    !! Prints one node and its operands.
    type(array_type), intent(inout), target :: node
    !! Current node
    integer, intent(in) :: visit
    !! Traversal marker
    integer, intent(in) :: depth
    !! Indentation depth
    logical, intent(in) :: show_location
    !! Whether to print node ids

    character(len=256) :: line

    if(node%visit_tag == visit) return
    node%visit_tag = visit
    if(show_location) then
       write(line, '(A,I0,A,I0,A,A)') repeat(' ', depth), node%id, &
            ' (', depth, ') ', trim(node%operation)
    else
       write(line, '(A,A)') repeat(' ', depth), trim(node%operation)
    end if
    write(*, '(A)') trim(line)
    if(associated(node%left_operand)) &
         call print_graph_inner(node%left_operand, visit, depth + 2, show_location)
    if(associated(node%right_operand)) &
         call print_graph_inner(node%right_operand, visit, depth + 2, show_location)
  end subroutine print_graph_inner
!###############################################################################


!###############################################################################
  module function get_partial(this, upstream_grad) result(output)
    !! The partial derivative of this node's operation.
    !!
    !! This is the default forward-mode partial used when a caller asks a node
    !! directly. Operation nodes bind their own partial procedures; for a node
    !! without one (a variable or a constant) the derivative is the upstream
    !! gradient itself.
    class(array_type), intent(inout) :: this
    !! Operation node
    type(array_type), intent(in) :: upstream_grad
    !! Gradient of the node's value
    type(array_type) :: output
    !! The partial derivative

    ! Local variables
    type(array_type), pointer :: deriv

    if(associated(this%get_partial_left)) then
       deriv => partial_node(this, upstream_grad, 1)
    else if(associated(this%get_partial_right)) then
       deriv => partial_node(this, upstream_grad, 2)
    else
       deriv => canonical(upstream_grad)
    end if
    output = deriv
  end function get_partial
!###############################################################################


!###############################################################################
  module function exp_array(a) result(c)
    !! Exponential function for autodiff arrays.
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c
    type(array_type), pointer :: na, nc
    integer :: i, s

    c => a%create_result()
    na => canonical(a)
    nc => c
    do s = 1, size(na%store, 2)
       !$omp simd
       do i = 1, size(na%store, 1)
          nc%store(i,s) = exp(na%store(i,s))
       end do
    end do
    c%get_partial_left => get_partial_exp
    c%get_partial_left_val => get_partial_exp_val
    c%get_partial_left_val_sum => get_partial_exp_val_sum
    call link_unary(c, a, 'exp')
  end function exp_array
!-------------------------------------------------------------------------------
  function get_partial_exp(this, upstream_grad) result(output)
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    ptr => upstream_grad * this
    output = ptr
  end function get_partial_exp
!-------------------------------------------------------------------------------
  pure subroutine get_partial_exp_val(this, upstream_grad, output)
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: i, s

    do s = 1, size(output, 2)
       !$omp simd
       do i = 1, size(output, 1)
          output(i,s) = upstream_grad(i,s) * this%val(i,s)
       end do
    end do
  end subroutine get_partial_exp_val
!-------------------------------------------------------------------------------
  pure subroutine get_partial_exp_val_sum(this, upstream_grad, output)
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:), intent(out) :: output

    integer :: s

    output = 0.0_real32
    do s = 1, size(upstream_grad, 2)
       output(:) = output(:) + upstream_grad(:, s) * &
            this%val(:, min(s, size(this%val, 2)))
    end do
  end subroutine get_partial_exp_val_sum
!###############################################################################


!###############################################################################
  module function log_array(a) result(c)
    !! Natural logarithm for autodiff arrays.
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c
    type(array_type), pointer :: na, nc
    integer :: i, s

    c => a%create_result()
    na => canonical(a)
    nc => c
    do s = 1, size(na%store, 2)
       !$omp simd
       do i = 1, size(na%store, 1)
          nc%store(i,s) = log(na%store(i,s))
       end do
    end do
    c%get_partial_left => get_partial_log
    c%get_partial_left_val => get_partial_log_val
    call link_unary(c, a, 'log')
  end function log_array
!-------------------------------------------------------------------------------
  function get_partial_log(this, upstream_grad) result(output)
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    ptr => upstream_grad / this%left_operand
    output = ptr
  end function get_partial_log
!-------------------------------------------------------------------------------
  pure subroutine get_partial_log_val(this, upstream_grad, output)
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: i, s

    do s = 1, size(output, 2)
       !$omp simd
       do i = 1, size(output, 1)
          output(i,s) = upstream_grad(i,s) / this%left_operand%val(i,s)
       end do
    end do
  end subroutine get_partial_log_val
!###############################################################################


!###############################################################################
  subroutine link_unary(c, a, operation)
    !! Records the graph edge of a unary elementwise operation.
    type(array_type), pointer, intent(inout) :: c
    !! Result node
    class(array_type), intent(in), target :: a
    !! Operand
    character(len=*), intent(in) :: operation
    !! Operation name

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = operation
       c%left_operand => canonical(a)
    else
       c%operation = operation
    end if
  end subroutine link_unary
!###############################################################################


!###############################################################################
  subroutine link_binary(c, a, b, operation)
    !! Records the graph edges of a binary elementwise operation.
    type(array_type), pointer, intent(inout) :: c
    !! Result node
    class(array_type), intent(in), target :: a, b
    !! Operands
    character(len=*), intent(in) :: operation
    !! Operation name

    if(a%requires_grad .or. b%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%operation = operation
       c%left_operand => canonical(a)
       c%right_operand => canonical(b)
    else
       c%operation = operation
    end if
  end subroutine link_binary
!###############################################################################


!###############################################################################
  module function sum_array(a, dim) result(c)
    !! Sums values along a dimension.
    class(array_type), intent(in), target :: a
    integer, intent(in) :: dim
    type(array_type), pointer :: c

    integer :: n_cols

    n_cols = size(a%val, 2)
    if(dim == 1) then
       c => a%create_result(array_shape=[1, n_cols])
       c%val(1, :) = sum(a%val, dim=1)
    else if(dim == 2) then
       c => a%create_result(array_shape=[a%shape, 1])
       c%val(:, 1) = sum(a%val, dim=2)
       c%is_sample_dependent = .false.
    else
       call stop_program('sum_array: only 1 or 2 dimensions are supported')
       nullify(c)
       return
    end if
    c%indices = [dim, 1]

    c%get_partial_left => get_partial_sum
    c%get_partial_left_val => get_partial_sum_val
    c%get_partial_left_val_sum => get_partial_sum_val_sum
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'sum_array'
       c%left_operand => canonical(a)
    end if
  end function sum_array
!-------------------------------------------------------------------------------
  function get_partial_sum(this, upstream_grad) result(output)
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    if(this%is_forward) then
       ptr => sum(upstream_grad, dim=this%indices(1))
    else
       ptr => spread(upstream_grad, dim=this%indices(1), index=this%indices(2), &
            ncopies=size(this%left_operand%val, this%indices(1)))
    end if
    output = ptr
  end function get_partial_sum
!-------------------------------------------------------------------------------
  pure subroutine get_partial_sum_val(this, upstream_grad, output)
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: i, s, dim, n_rows, n_cols

    dim = this%indices(1)
    n_rows = size(output, 1)
    n_cols = size(output, 2)
    if(dim == 1) then
       do concurrent(s = 1:n_cols, i = 1:n_rows)
          output(i, s) = upstream_grad(1, s)
       end do
    else
       do concurrent(s = 1:n_cols, i = 1:n_rows)
          output(i, s) = upstream_grad(i, 1)
       end do
    end if
  end subroutine get_partial_sum_val
!-------------------------------------------------------------------------------
  pure subroutine get_partial_sum_val_sum(this, upstream_grad, output)
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:), intent(out) :: output

    integer :: dim, n_samples

    dim = this%indices(1)
    n_samples = size(upstream_grad, 2)
    if(dim == 1) then
       output(:) = upstream_grad(1, 1) * real(n_samples, real32)
    else
       output(:) = upstream_grad(:, 1) * real(n_samples, real32)
    end if
  end subroutine get_partial_sum_val_sum
!###############################################################################


!###############################################################################
  module function sum_all_array(a) result(c)
    !! Sums all values of an autodiff array.
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c
    type(array_type), pointer :: na

    na => canonical(a)
    c => a%create_result(array_shape=[1, 1])
#ifdef DIFFSTRUC_FAST_REDUCTION
    block
      integer :: n, ns, nth
      real(real32), allocatable :: colsum(:)
      n = size(na%store, 1)
      ns = size(na%store, 2)
      nth = kernel_team(a, n, ns)
      allocate(colsum(ns))
      call sum_columns(colsum, na%store, n, ns, nth)
      c%store(1, 1) = sum(colsum)
      deallocate(colsum)
    end block
#else
    c%store(1, 1) = sum(na%store)
#endif
    allocate(c%indices(2))
    c%indices = [1, 1]

    c%get_partial_left => get_partial_sum_all
    c%get_partial_left_val => get_partial_sum_all_val
    c%get_partial_left_val_sum => get_partial_sum_all_val_sum
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'sum_all_array'
       c%left_operand => canonical(a)
    end if
  end function sum_all_array
!-------------------------------------------------------------------------------
  function get_partial_sum_all(this, upstream_grad) result(output)
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    if(this%is_forward) then
       ptr => sum(upstream_grad)
    else
       ptr => spread(upstream_grad, dim=1, index=1, &
            ncopies=size(this%left_operand%val, 1))
       ptr => spread(ptr, dim=2, index=1, &
            ncopies=size(this%left_operand%val, 2))
    end if
    output = ptr
  end function get_partial_sum_all
!-------------------------------------------------------------------------------
  pure subroutine get_partial_sum_all_val(this, upstream_grad, output)
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: i, s

    do s = 1, size(output, 2)
       !$omp simd
       do i = 1, size(output, 1)
          output(i,s) = upstream_grad(1, 1)
       end do
    end do
  end subroutine get_partial_sum_all_val
!-------------------------------------------------------------------------------
  pure subroutine get_partial_sum_all_val_sum(this, upstream_grad, output)
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:), intent(out) :: output

    output(:) = upstream_grad(1, 1) * real(size(upstream_grad, 2), real32)
  end subroutine get_partial_sum_all_val_sum
!###############################################################################


!###############################################################################
  module function sum_and_pad_array(a, dim, new_dim_index, new_dim_size) result(c)
    !! Sums a 1-D array along a dimension and places the result at an index.
    class(array_type), intent(in), target :: a
    integer, intent(in) :: dim
    integer, intent(in) :: new_dim_index
    integer, intent(in) :: new_dim_size
    type(array_type), pointer :: c

    if(size(a%shape) /= 1) then
       call stop_program('sum_and_pad_array: only 1D arrays can be used')
    end if
    if(dim == 1) then
       c => a%create_result(array_shape=[new_dim_size, size(a%val, 2)])
       c%val = 0.0_real32
       c%val(new_dim_index, :) = sum(a%val, dim=1)
    else if(dim == 2) then
       c => a%create_result(array_shape=[size(a%val, 1), new_dim_size])
       c%val = 0.0_real32
       c%val(:, new_dim_index) = sum(a%val, dim=2)
    else
       call stop_program('sum_and_pad_array: only 1 or 2 dimensions are supported')
       nullify(c)
       return
    end if
    allocate(c%indices(2))
    c%indices = [dim, new_dim_index]
    c%is_sample_dependent = a%is_sample_dependent

    c%get_partial_left => get_partial_sum_and_pad
    c%get_partial_left_val => get_partial_sum_and_pad_val
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'sum_and_pad_array'
       c%left_operand => canonical(a)
    end if
  end function sum_and_pad_array
!-------------------------------------------------------------------------------
  function get_partial_sum_and_pad(this, upstream_grad) result(output)
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    if(this%is_forward) then
       ptr => sum(upstream_grad, dim=this%indices(1), &
            new_dim_index=this%indices(2), &
            new_dim_size=size(this%val, this%indices(1)))
    else
       ptr => spread(upstream_grad, dim=this%indices(1), index=this%indices(2), &
            ncopies=size(this%left_operand%val, this%indices(1)))
    end if
    output = ptr
  end function get_partial_sum_and_pad
!-------------------------------------------------------------------------------
  pure subroutine get_partial_sum_and_pad_val(this, upstream_grad, output)
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: i, s, dim, n_rows, n_cols

    dim = this%indices(1)
    n_rows = size(output, 1)
    n_cols = size(output, 2)
    if(dim == 1) then
       do concurrent(s = 1:n_cols, i = 1:n_rows)
          output(i, s) = upstream_grad(1, this%indices(2))
       end do
    else
       do concurrent(s = 1:n_cols, i = 1:n_rows)
          output(i, s) = upstream_grad(this%indices(2), 1)
       end do
    end if
  end subroutine get_partial_sum_and_pad_val
!###############################################################################


!###############################################################################
  module function mean_array(a, dim) result(c)
    !! Computes the mean along a dimension.
    class(array_type), intent(in), target :: a
    integer, intent(in) :: dim
    type(array_type), pointer :: c

    integer :: n_cols
    real(real32) :: inv_count

    n_cols = size(a%val, 2)
    if(dim == 1) then
       c => a%create_result(array_shape=[1, n_cols])
       c%val(1, :) = sum(a%val, dim=1) / real(size(a%val, 1), real32)
    else if(dim == 2) then
       c => a%create_result(array_shape=[a%shape, 1])
       inv_count = 1.0_real32 / real(n_cols, real32)
       c%val(:, 1) = sum(a%val, dim=2) * inv_count
       c%is_sample_dependent = .false.
    else
       call stop_program('mean_array: only 1 or 2 dimensions are supported')
       nullify(c)
       return
    end if
    allocate(c%indices(2))
    c%indices = [dim, 1]

    c%get_partial_left => get_partial_mean
    c%get_partial_left_val => get_partial_mean_val
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'mean_array'
       c%left_operand => canonical(a)
    end if
  end function mean_array
!-------------------------------------------------------------------------------
  function get_partial_mean(this, upstream_grad) result(output)
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    real(real32) :: count
    type(array_type), pointer :: ptr

    count = real(size(this%left_operand%val, this%indices(1)), real32)
    if(this%is_forward) then
       ptr => sum(upstream_grad, dim=this%indices(1)) / count
    else
       ptr => spread(upstream_grad / count, dim=this%indices(1), &
            index=this%indices(2), &
            ncopies=size(this%left_operand%val, this%indices(1)))
    end if
    output = ptr
  end function get_partial_mean
!-------------------------------------------------------------------------------
  pure subroutine get_partial_mean_val(this, upstream_grad, output)
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: i, s, dim, n_rows, n_cols
    real(real32) :: inv_count

    dim = this%indices(1)
    n_rows = size(output, 1)
    n_cols = size(output, 2)
    inv_count = 1.0_real32 / real(size(this%left_operand%val, dim), real32)
    if(dim == 1) then
       do concurrent(s = 1:n_cols, i = 1:n_rows)
          output(i, s) = upstream_grad(1, s) * inv_count
       end do
    else
       do concurrent(s = 1:n_cols, i = 1:n_rows)
          output(i, s) = upstream_grad(i, 1) * inv_count
       end do
    end if
  end subroutine get_partial_mean_val
!###############################################################################


!###############################################################################
  module function mean_all_array(a) result(c)
    !! Computes the mean of all elements.
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    real(real32) :: inv_count

    c => a%create_result(array_shape=[1, 1])
    inv_count = 1.0_real32 / real(size(a%val), real32)
#ifdef DIFFSTRUC_FAST_REDUCTION
    block
      integer :: n, ns, nth
      real(real32), allocatable :: colsum(:)
      n = size(a%val, 1)
      ns = size(a%val, 2)
      nth = kernel_team(a, n, ns)
      allocate(colsum(ns))
      call sum_columns(colsum, a%val, n, ns, nth)
      c%val(1, 1) = sum(colsum) * inv_count
      deallocate(colsum)
    end block
#else
    c%val(1, 1) = sum(a%val) * inv_count
#endif
    allocate(c%indices(2))
    c%indices = [1, 1]

    c%get_partial_left => get_partial_mean_all
    c%get_partial_left_val => get_partial_mean_all_val
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'mean_all_array'
       c%left_operand => canonical(a)
    end if
  end function mean_all_array
!-------------------------------------------------------------------------------
  function get_partial_mean_all(this, upstream_grad) result(output)
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    real(real32) :: inv_count
    type(array_type), pointer :: ptr

    inv_count = 1.0_real32 / real(size(this%left_operand%val), real32)
    if(this%is_forward) then
       ptr => sum(upstream_grad) * inv_count
    else
       ptr => spread(upstream_grad * inv_count, dim=1, index=1, &
            ncopies=size(this%left_operand%val, 1))
       ptr => spread(ptr, dim=2, index=1, &
            ncopies=size(this%left_operand%val, 2))
    end if
    output = ptr
  end function get_partial_mean_all
!-------------------------------------------------------------------------------
  pure subroutine get_partial_mean_all_val(this, upstream_grad, output)
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    real(real32) :: inv_count

    inv_count = 1.0_real32 / real(size(this%left_operand%val), real32)
    output = upstream_grad(1, 1) * inv_count
  end subroutine get_partial_mean_all_val
!###############################################################################


!###############################################################################
  module function spread_array(source, dim, index, ncopies) result(c)
    !! Spreads a 1-D autodiff array along a dimension.
    class(array_type), intent(in), target :: source
    integer, intent(in) :: dim
    integer, intent(in) :: index
    integer, intent(in) :: ncopies
    type(array_type), pointer :: c

    integer :: s

    if(dim == 1) then
       c => source%create_result(array_shape=[ncopies, size(source%val, 2)])
       do concurrent(s = 1:ncopies)
          c%val(s, :) = source%val(index, :)
       end do
    else if(dim == 2) then
       c => source%create_result(array_shape=[size(source%val, 1), ncopies])
       do concurrent(s = 1:ncopies)
          c%val(:, s) = source%val(:, index)
       end do
    else
       call stop_program('spread: only 1 or 2 dimensions are supported')
       nullify(c)
       return
    end if
    allocate(c%indices(1))
    c%indices = [index]
    allocate(c%adj_ja(2, 1))
    c%adj_ja(:, 1) = [dim, size(source%val, dim)]

    c%get_partial_left => get_partial_spread
    if(source%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = source%is_forward
       c%operation = 'spread'
       c%left_operand => canonical(source)
    end if
  end function spread_array
!-------------------------------------------------------------------------------
  function get_partial_spread(this, upstream_grad) result(output)
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    ptr => unspread(upstream_grad, this%indices(1), this%adj_ja(1, 1), &
         this%adj_ja(2, 1))
    output = ptr
  end function get_partial_spread
!###############################################################################


!###############################################################################
  module function unspread_array(source, index, dim, new_size) result(c)
    !! Gathers an autodiff array onto one index of a larger array.
    class(array_type), intent(in), target :: source
    integer, intent(in) :: index
    integer, intent(in) :: new_size, dim
    type(array_type), pointer :: c

    integer :: i, s

    if(dim == 1) then
       c => source%create_result(array_shape=[new_size, size(source%val, 2)])
       c%val = 0.0_real32
       do i = 1, size(source%val, 1)
          do s = 1, size(source%val, 2)
             c%val(index, s) = c%val(index, s) + source%val(i, s)
          end do
       end do
    else if(dim == 2) then
       c => source%create_result(array_shape=[size(source%val, 1), new_size])
       c%val = 0.0_real32
       do i = 1, size(source%val, 1)
          do s = 1, size(source%val, 2)
             c%val(i, index) = c%val(i, index) + source%val(i, s)
          end do
       end do
    else
       call stop_program('unspread: only 1 or 2 dimensions are supported')
       nullify(c)
       return
    end if
    allocate(c%indices(1))
    c%indices = [index]
    allocate(c%adj_ja(2, 1))
    c%adj_ja(:, 1) = [dim, new_size]

    c%get_partial_left => get_partial_unspread
    if(source%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = source%is_forward
       c%operation = 'unspread'
       c%left_operand => canonical(source)
    end if
  end function unspread_array
!-------------------------------------------------------------------------------
  function get_partial_unspread(this, upstream_grad) result(output)
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    ptr => spread(upstream_grad, this%indices(1), this%adj_ja(1, 1), &
         this%adj_ja(2, 1))
    output = ptr
  end function get_partial_unspread
!###############################################################################


end submodule diffstruc__types_submodule
