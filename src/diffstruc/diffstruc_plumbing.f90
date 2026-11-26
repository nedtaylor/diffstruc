!> Handle resolution, validation and the value-access procedures of
!! `diffstruc__types`. These procedures are the bridge between the
!! lightweight handle and the context-owned record: they never allocate or
!! free storage that a handle depends on, they only take views of it.
submodule(diffstruc__types) diffstruc__plumbing_impl
  implicit none

contains


!###############################################################################
  module function context_of(this) result(ctx)
    !! The context that owns a handle's node.
    class(array_type), intent(in) :: this
    !! Handle to query
    type(array_context_type) :: ctx
    !! Its context

    if(associated(this%ctx%state)) then
       ctx%state => this%ctx%state
       ctx%gen = this%ctx%gen
    else
       ctx = default_context_of()
    end if
  end function context_of
!###############################################################################


!###############################################################################
  module function canonical(this) result(node)
    !! The context-owned node record a handle refers to (null when unbound
    !! or stale).
    class(array_type), intent(in) :: this
    !! Handle to resolve
    type(array_type), pointer :: node
    !! The record

    ! Local variables
    type(array_context_type) :: ctx
    type(context_state), pointer :: state

    nullify(node)
    if(this%id <= 0) return
    ! Fast path: a bound handle remembers the record it views, so no registry
    ! lookup is needed. The id check rejects a record the context released.
    if(associated(this%self)) then
       if(this%self%id == this%id .and. this%self%gen == this%gen) then
          node => this%self
          return
       end if
    end if
    ctx = context_of(this)
    state => context_state_of(ctx)
    if(this%gen /= state%gen) return
    node => find_node(ctx, this%id)
  end function canonical
!###############################################################################


!###############################################################################
  module function live(this) result(node)
    !! The context-owned node record a handle refers to, stopping the program
    !! on stale or unbound access.
    class(array_type), intent(in) :: this
    !! Handle to resolve
    type(array_type), pointer :: node
    !! The record

    node => canonical(this)
    if(.not. associated(node)) then
       call stop_program('diffstruc: use of an unallocated or released array')
    end if
  end function live
!###############################################################################


!###############################################################################
  module subroutine bind_handle(this, node)
    !! Binds a handle to a context-owned record, taking views of its storage.
    class(array_type), intent(inout) :: this
    !! Handle to bind
    type(array_type), intent(in), target :: node
    !! Record to view

    this%self => node
    this%id = node%id
    this%gen = node%gen
    this%ctx = node%ctx
    this%rank = node%rank
    this%size = node%size
    this%is_sample_dependent => node%is_sample_dependent
    this%is_forward = node%is_forward
    this%is_scalar = node%is_scalar
    this%allocated = node%allocated
    this%requires_grad = node%requires_grad
    this%value_partial_ok = node%value_partial_ok
    this%operation = node%operation
    this%is_temporary => node%is_temporary
    this%visit_tag = node%visit_tag
    if(allocated(node%shape)) this%shape = node%shape
    this%val => node%val
    if(allocated(node%indices)) this%indices = node%indices
    if(allocated(node%adj_ja)) this%adj_ja = node%adj_ja
    if(allocated(node%mask)) this%mask = node%mask
    if(allocated(node%direction)) this%direction = node%direction
    this%grad => node%grad
    this%left_operand => node%left_operand
    this%right_operand => node%right_operand
    this%get_partial_left => node%get_partial_left
    this%get_partial_right => node%get_partial_right
    this%get_partial_left_val => node%get_partial_left_val
    this%get_partial_right_val => node%get_partial_right_val
    this%get_partial_left_val_sum => node%get_partial_left_val_sum
    this%get_partial_right_val_sum => node%get_partial_right_val_sum
  end subroutine bind_handle
!###############################################################################


!###############################################################################
  module subroutine clone_record(dst, src)
    !! Copies the content of one record into another.
    !!
    !! Value, shape and auxiliary buffers are copied into fresh storage owned
    !! by the destination's context; graph links are shared, exactly as a
    !! handle copy is expected to share them. The destination keeps its own
    !! identity.
    type(array_type), intent(inout), target :: dst
    !! Record to overwrite
    type(array_type), intent(in), target :: src
    !! Record to copy

    ! Local variables
    type(array_context_type) :: ctx, keep_ctx
    integer :: keep_id, keep_gen

    keep_id = dst%id
    keep_gen = dst%gen
    keep_ctx = dst%ctx
    ctx = context_of(dst)
    call release_node(ctx, dst)
    dst%id = keep_id
    dst%gen = keep_gen
    dst%ctx = keep_ctx

    dst%rank = src%rank
    dst%size = src%size
    dst%is_sample_dependent = src%is_sample_dependent
    dst%is_forward = src%is_forward
    dst%is_scalar = src%is_scalar
    dst%allocated = src%allocated
    dst%requires_grad = src%requires_grad
    dst%value_partial_ok = src%value_partial_ok
    dst%operation = src%operation
    dst%is_temporary = src%is_temporary
    dst%visit_tag = 0

    if(allocated(src%shape)) dst%shape = src%shape
    if(allocated(src%store)) then
       call take_value(ctx, dst%store, shape(src%store))
       dst%store = src%store
       dst%val => dst%store
    end if
    if(allocated(src%indices)) dst%indices = src%indices
    if(allocated(src%adj_ja)) dst%adj_ja = src%adj_ja
    if(allocated(src%mask)) dst%mask = src%mask
    if(allocated(src%direction)) dst%direction = src%direction

    dst%left_operand => src%left_operand
    dst%right_operand => src%right_operand
    dst%get_partial_left => src%get_partial_left
    dst%get_partial_right => src%get_partial_right
    dst%get_partial_left_val => src%get_partial_left_val
    dst%get_partial_right_val => src%get_partial_right_val
    dst%get_partial_left_val_sum => src%get_partial_left_val_sum
    dst%get_partial_right_val_sum => src%get_partial_right_val_sum
  end subroutine clone_record
!###############################################################################


!###############################################################################
  module function clone_handle(source) result(copy)
    !! Creates a new context-owned record holding a copy of a handle's content.
    class(array_type), intent(in) :: source
    !! Handle to copy
    type(array_type), pointer :: copy
    !! The new record

    ! Local variables
    type(array_context_type) :: ctx
    type(array_type), pointer :: src

    src => live(source)
    ctx = context_of(source)
    copy => allocate_node(ctx)
    call clone_record(copy, src)
  end function clone_handle
!###############################################################################


!###############################################################################
  module function attach_gradient(this) result(grad)
    !! Ensures a node has a gradient accumulator of its own shape.
    class(array_type), intent(in), target :: this
    !! Handle whose gradient is required
    type(array_type), pointer :: grad
    !! The accumulator

    ! Local variables
    type(array_context_type) :: ctx
    type(array_type), pointer :: node

    node => live(this)
    if(associated(node%grad)) then
       grad => node%grad
       return
    end if
    ctx = context_of(this)
    grad => allocate_value_storage(ctx, 1, [node%size, size(node%val, 2)])
    grad%shape = node%shape
    grad%size = node%size
    grad%is_scalar = node%is_scalar
    grad%is_sample_dependent = node%is_sample_dependent
    grad%requires_grad = .not. node%is_scalar
    grad%operation = 'none'
    grad%val = 0.0_real32
    if(allocated(node%indices)) grad%indices = node%indices
    node%grad => grad
  end function attach_gradient
!###############################################################################


!###############################################################################
  module function constant_node(template, values) result(node)
    !! Creates a constant node in a template handle's context.
    class(array_type), intent(in) :: template
    !! Handle whose context owns the constant
    real(real32), dimension(:,:), intent(in) :: values
    !! Constant values
    type(array_type), pointer :: node
    !! The constant node

    ! Local variables
    type(array_context_type) :: ctx

    ctx = context_of(template)
    node => allocate_value_storage(ctx, 1, [size(values, 1), size(values, 2)])
    node%val = values
    node%is_sample_dependent = .false.
    node%is_scalar = size(values) == 1
    node%requires_grad = .false.
    node%operation = 'none'
  end function constant_node
!###############################################################################


!###############################################################################
  module subroutine allocate_array(this, array_shape, source, &
       sample_dependent, temporary)
    !! Allocates the array's value storage in its context.
    !!
    !! `array_shape` follows the library's storage convention: the last entry
    !! is the number of samples and the leading entries are the logical shape.
    !! An `n`-element request therefore allocates `(product(array_shape(1:n-1)),
    !! array_shape(n))` and gives the node the logical shape
    !! `array_shape(1:n-1)`. For example `[d, batch]` is a rank-1 array of `d`
    !! elements with `batch` samples, and `[rows, cols, 1]` is a rank-2 array
    !! with one sample.
    class(array_type), intent(inout), target :: this
    !! Handle to allocate
    integer, dimension(:), intent(in), optional :: array_shape
    !! Shape of the array (last entry is the sample count)
    class(*), dimension(..), intent(in), optional :: source
    !! Values to store
    logical, intent(in), optional :: sample_dependent
    !! Whether the array varies over samples (default true). Setting it here is
    !! equivalent to assigning `is_sample_dependent` after allocation, but
    !! avoids the window in which the flag has no target.
    logical, intent(in), optional :: temporary
    !! Whether the array is a disposable expression temporary (default true)

    ! Local variables
    type(array_context_type) :: ctx
    type(array_type), pointer :: node
    integer :: n

    if(this%allocated) then
       call stop_program('diffstruc: array is already allocated')
       return
    end if
    if(.not. present(array_shape) .and. .not. present(source)) then
       call stop_program('diffstruc: no shape or source provided')
       return
    end if

    ctx = context_of(this)
    if(present(array_shape)) then
       n = size(array_shape)
       if(n < 1) then
          call stop_program('diffstruc: array_shape must not be empty')
          return
       end if
       node => allocate_value_storage(ctx, 1, [ &
            product(array_shape(1:n-1)), array_shape(n) &
       ])
       ! The logical shape is the leading dimensions; the final entry is the
       ! number of samples.
       node%shape = array_shape(1:n-1)
       node%size = product(array_shape(1:n-1))
       node%rank = max(1, n - 1)
    else
       select rank(source)
       rank(2)
          node => allocate_value_storage(ctx, 1, shape(source))
       rank default
          call stop_program('diffstruc: source must be rank 2')
          return
       end select
    end if
    node%val = 0.0_real32
    if(present(sample_dependent)) then
       node%sample_dependent_flag = sample_dependent
    end if
    if(present(temporary)) node%temporary_flag = temporary
    call bind_handle(this, node)

    if(present(source)) then
       select rank(source)
       rank(0)
          select type(source)
          type is (real(real32))
             node%val = source
          end select
       rank(2)
          select type(source)
          type is (real(real32))
             ! Accept a logical (row, column) source for a flattened storage
             ! view, e.g. `allocate([rows, cols, 1], source=matrix)`.
             if(all(shape(source) == shape(node%val))) then
                node%val = source
             else if(size(source) == size(node%val)) then
                node%val = reshape(source, shape(node%val))
             else
                call stop_program('diffstruc: source shape does not match array shape')
             end if
          end select
       rank default
          call stop_program('diffstruc: unrecognised source rank')
       end select
    end if
  end subroutine allocate_array
!###############################################################################


!###############################################################################
  module subroutine deallocate_array(this, keep_shape)
    !! Returns this handle's node storage to its context and unbinds the
    !! handle.
    !!
    !! Because node storage is context-owned and there is no reference count,
    !! `deallocate` is the one explicit request to give a node's storage back.
    !! Any other handle that still refers to the same node becomes stale and is
    !! rejected on access. Use `nullify_graph` to release an expression's
    !! operands while keeping its root, or `reset_context` to release everything.
    class(array_type), intent(inout) :: this
    !! Handle to release
    logical, intent(in), optional :: keep_shape
    !! Retained for source compatibility; the context owns the shape

    ! Local variables
    type(array_type), pointer :: node
    type(array_context_type) :: ctx

    node => canonical(this)
    if(associated(node)) then
       ctx = context_of(this)
       call release_node(ctx, node)
    end if

    this%id = 0
    this%allocated = .false.
    if(allocated(this%shape)) deallocate(this%shape)
    if(allocated(this%indices)) deallocate(this%indices)
    if(allocated(this%adj_ja)) deallocate(this%adj_ja)
    if(allocated(this%mask)) deallocate(this%mask)
    if(allocated(this%direction)) deallocate(this%direction)
    this%val => null()
    this%grad => null()
    this%left_operand => null()
    this%right_operand => null()
  end subroutine deallocate_array
!###############################################################################


!###############################################################################
  pure module function flatten_array(this) result(output)
    !! Copies a handle's values into a flat rank-1 array.
    class(array_type), intent(in) :: this
    !! Handle to read
    real(real32), dimension(this%size) :: output
    !! The flattened values

    output = reshape(this%val, [this%size])
  end function flatten_array
!###############################################################################


!###############################################################################
  recursive module subroutine assign_array(this, input)
    !! Copies the content of a source array into this array.
    !!
    !! If this handle is unbound it is bound to a new context-owned record.
    !! If it is already bound -- for instance because it is the target of a
    !! pointer that the statement assigns through -- the record it refers to
    !! is overwritten in place, so every view of it stays coherent.
    class(array_type), intent(inout), target :: this
    !! Handle to write
    type(array_type), intent(in) :: input
    !! Array to copy

    ! Local variables
    type(array_context_type) :: ctx
    type(array_type), pointer :: src, dst

    src => canonical(input)
    if(.not. associated(src)) then
       ! Assigning from an unbound handle leaves the target unbound too. This
       ! happens when a derived type that contains an allocated-but-unbound
       ! array_type component is copied, so it must not be treated as an error.
       this%id = 0
       this%allocated = .false.
       this%self => null()
       this%val => null()
       this%grad => null()
       this%left_operand => null()
       this%right_operand => null()
       this%is_temporary => null()
       this%is_sample_dependent => null()
       return
    end if
    dst => canonical(this)
    if(associated(dst)) then
       call clone_record(dst, src)
       call bind_handle(this, dst)
    else
       ! Binding an unbound handle is a pure handle copy: the target shares the
       ! source's context-owned record. Nothing is allocated or copied.
       call bind_handle(this, src)
    end if
  end subroutine assign_array
!###############################################################################


!###############################################################################
  module subroutine assign_shallow(this, source)
    !! Binds this handle to another handle's node.
    class(array_type), intent(inout) :: this
    !! Handle to bind
    class(array_type), intent(in), target :: source
    !! Handle whose node is shared

    ! Local variables
    type(array_type), pointer :: node

    node => canonical(source)
    if(.not. associated(node)) then
       ! Nothing to share: leave this handle unbound.
       this%id = 0
       this%allocated = .false.
       this%self => null()
       this%val => null()
       this%grad => null()
       this%left_operand => null()
       this%right_operand => null()
       this%is_temporary => null()
       this%is_sample_dependent => null()
       return
    end if
    call bind_handle(this, node)
  end subroutine assign_shallow
!###############################################################################


!###############################################################################
  module subroutine assign_and_deallocate_source(this, source)
    !! Copies a source array into this handle and releases the source node.
    !!
    !! The source must be a temporary expression node. Root variables and
    !! nodes explicitly marked as non-temporary are copied but kept alive,
    !! because other handles may still view them.
    class(array_type), intent(inout) :: this
    !! Handle to write
    type(array_type), intent(inout), pointer :: source
    !! Source array

    ! Local variables
    type(array_type), pointer :: src, node, dst
    type(array_context_type) :: ctx

    src => canonical(source)
    if(.not. associated(src)) return
    ! Copy into a record of its own: the source is about to be released, so
    ! binding this handle to it would leave a dangling handle.
    ! (Nested tests: Fortran does not short-circuit `.and.`, so `dst%id` must
    ! not be evaluated while `dst` may be unassociated.)
    dst => canonical(this)
    if(associated(dst)) then
       if(dst%id == src%id) dst => null()
    end if
    if(.not. associated(dst)) then
       ctx = context_of(this)
       dst => allocate_node(ctx)
    end if
    call clone_record(dst, src)
    call bind_handle(this, dst)
    if(src%requires_grad .and. .not. associated(src%left_operand) .and. &
         .not. associated(src%right_operand)) return
    if(.not. src%is_temporary) return
    ctx = context_of(source)
    call release_node(ctx, src)
    source%id = 0
    source%allocated = .false.
    source%val => null()
    source%left_operand => null()
    source%right_operand => null()
    source%grad => null()
  end subroutine assign_and_deallocate_source
!###############################################################################


!###############################################################################
  module function create_result_array(this, array_shape) result(result_ptr)
    !! Creates a result node of the given shape in a handle's context.
    class(array_type), intent(in), target :: this
    !! Handle whose context allocates the result
    integer, dimension(:), intent(in), optional :: array_shape
    !! Shape of the result array
    type(array_type), pointer :: result_ptr
    !! The allocated result

    ! Local variables
    type(array_context_type) :: ctx
    type(array_type), pointer :: node
    integer :: n

    ctx = context_of(this)
    if(present(array_shape)) then
       n = size(array_shape)
       node => allocate_value_storage(ctx, 1, [ &
            product(array_shape(1:n-1)), array_shape(n) &
       ])
       node%shape = array_shape(1:n-1)
       node%size = product(array_shape(1:n-1))
       node%rank = 1
    else if(allocated(this%shape)) then
       ! Storage is always (elements, samples); flatten the logical shape.
       node => allocate_value_storage(ctx, 1, [this%size, size(this%val, 2)])
       node%shape = this%shape
       node%size = this%size
       node%rank = max(1, size(this%shape))
    else
       node => allocate_value_storage(ctx, 1, shape(this%val))
    end if

    node%requires_grad = .false.
    node%is_scalar = this%is_scalar
    node%is_sample_dependent = this%is_sample_dependent
    node%is_forward = this%is_forward
    node%operation = 'none'
    result_ptr => node
  end function create_result_array
!###############################################################################


!###############################################################################
  module subroutine set_array(this, input, set_id)
    !! Assigns array values, preserving the array's identity.
    class(array_type), intent(inout) :: this
    !! Handle to write
    real(real32), dimension(..), intent(in) :: input
    !! Values to store
    logical, intent(in), optional :: set_id
    !! Ignored; retained for source compatibility

    if(.not. associated(this%val)) return
    select rank(input)
    rank(0)
       this%val = input
    rank(1)
       if(size(input) == size(this%val, 1)) then
          this%val(:, 1) = input
       else
          this%val = reshape(input, shape(this%val))
       end if
    rank(2)
       if(all(shape(input) == shape(this%val))) then
          this%val = input
       else if(size(input, 1) == size(this%val, 1)) then
          this%val = input
       else
          this%val = reshape(input, shape(this%val))
       end if
    rank(3)
       this%val = reshape(input, shape(this%val))
    rank(4)
       this%val = reshape(input, shape(this%val))
    rank(5)
       this%val = reshape(input, shape(this%val))
    rank(6)
       this%val = reshape(input, shape(this%val))
    rank(7)
       this%val = reshape(input, shape(this%val))
    rank default
       call stop_program('diffstruc: unsupported input rank')
    end select
  end subroutine set_array
!###############################################################################


!###############################################################################
  module subroutine extract_array(this, output)
    !! Copies a handle's values into a standard real array.
    !!
    !! The requested rank selects how much of the logical shape is restored:
    !!
    !! * rank ``size(shape) + 1`` gives the logical shape plus the sample
    !!   dimension, e.g. a ``[rows, cols]`` array with one sample extracts to
    !!   ``(rows, cols, 1)``;
    !! * a rank equal to ``size(shape)`` is accepted when the array has one
    !!   sample, e.g. ``(rows, cols)``;
    !! * rank 1 always flattens (element index varying fastest, then samples).
    class(array_type), intent(in) :: this
    !! Handle to read
    real(real32), dimension(..), allocatable, intent(out) :: output
    !! Copy of the values

    ! Local variables
    integer :: ndim, ns

    if(.not. this%allocated .or. .not. associated(this%val)) then
       call print_warning('diffstruc: cannot extract from an unallocated array')
       return
    end if
    ndim = 0
    if(allocated(this%shape)) ndim = size(this%shape)
    ns = size(this%val, 2)

    select rank(output)
    rank(0)
       output = this%val(1, 1)
    rank(1)
       allocate(output(max(1, product(this%shape)) * ns))
       output = reshape(this%val, shape(output))
    rank(2)
       if(ndim == 2 .and. ns == 1) then
          allocate(output(this%shape(1), this%shape(2)))
       else if(ndim <= 1) then
          allocate(output(size(this%val, 1), ns))
       else
          call print_warning('diffstruc: output rank is too small for this array')
          return
       end if
       output = reshape(this%val, shape(output))
    rank(3)
       if(ndim /= 2) then
          call print_warning('diffstruc: output rank does not match array shape')
          return
       end if
       allocate(output(this%shape(1), this%shape(2), ns))
       output = reshape(this%val, shape(output))
    rank(4)
       if(ndim /= 3) then
          call print_warning('diffstruc: output rank does not match array shape')
          return
       end if
       allocate(output(this%shape(1), this%shape(2), this%shape(3), ns))
       output = reshape(this%val, shape(output))
    rank(5)
       if(ndim /= 4) then
          call print_warning('diffstruc: output rank does not match array shape')
          return
       end if
       allocate(output(this%shape(1), this%shape(2), this%shape(3), &
            this%shape(4), ns))
       output = reshape(this%val, shape(output))
    rank default
       call print_warning('diffstruc: unsupported output rank for extract')
    end select
  end subroutine extract_array
!###############################################################################


!###############################################################################
  module subroutine detach(this)
    !! Detaches a handle from its computation-graph structure.
    class(array_type), intent(inout) :: this
    !! Handle to detach

    ! Local variables
    type(array_type), pointer :: node

    node => canonical(this)
    if(associated(node)) then
       node%left_operand => null()
       node%right_operand => null()
       node%get_partial_left => null()
       node%get_partial_right => null()
       node%get_partial_left_val => null()
       node%get_partial_right_val => null()
       node%get_partial_left_val_sum => null()
       node%get_partial_right_val_sum => null()
       node%operation = 'none'
       call bind_handle(this, node)
    end if
  end subroutine detach
!###############################################################################


end submodule diffstruc__plumbing_impl
