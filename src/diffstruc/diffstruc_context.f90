!> Context ownership for `diffstruc__types`.
!!
!! The context is the single owner of every node record, every value and
!! gradient buffer, and every seed direction. A handle never allocates or
!! frees anything; it only carries an id plus views into the context's
!! memory. Records live in fixed-size chunks that are never moved, so a
!! pointer into a record (and a view of its storage) stays valid for as long
!! as the record is live.
submodule(diffstruc__types) diffstruc__context_impl
  implicit none

  type(context_state), target :: context_states(max_contexts)
  !! Private state of every context the process can address.

contains


!###############################################################################
  module function context_state_of(ctx) result(state)
    !! The private state of a context.
    type(array_context_type), intent(in) :: ctx
    !! Context handle
    type(context_state), pointer :: state
    !! Its state

    state => context_states(max(1, min(max_contexts, ctx%slot)))
  end function context_state_of
!###############################################################################


!###############################################################################
  module function allocate_node(ctx) result(node)
    !! Allocates an empty node record in the context's registry.
    !!
    !! The record lives in a chunk that is never reallocated, so the returned
    !! pointer stays valid until the context releases the record.
    type(array_context_type), intent(in) :: ctx
    !! The owning context
    type(array_type), pointer :: node
    !! The new record

    ! Local variables
    type(context_state), pointer :: state
    type(ad_registry), pointer :: chunk, fresh
    type(chunk_slot), dimension(:), allocatable :: grown
    logical :: fresh_needed
    integer :: i, n

    state => context_state_of(ctx)
    ! Chunks are appended in order, so the last one is the only candidate for
    ! free slots. Walking the list here made every allocation cost grow with
    ! the number of chunks ever created.
    chunk => state%tail

    fresh_needed = .not. associated(chunk)
    if(.not. fresh_needed) fresh_needed = state%free_slot > size(chunk%nodes)
    if(fresh_needed) then
       allocate(fresh)
       allocate(fresh%nodes(node_chunk_size))
       n = node_chunk_size
       do i = 1, n
          call clear_record(fresh%nodes(i))
       end do
       fresh%base = state%last_id
       nullify(fresh%next)
       if(associated(chunk)) then
          chunk%next => fresh
       else
          state%head => fresh
       end if
       chunk => fresh
       state%tail => fresh
       state%free_slot = 1
       ! Record the chunk so `find_node` can address a record arithmetically.
       if(.not. allocated(state%chunks)) then
          allocate(state%chunks(16))
       else if(state%n_chunks == size(state%chunks)) then
          allocate(grown(2*state%n_chunks))
          do i = 1, state%n_chunks
             grown(i)%chunk => state%chunks(i)%chunk
          end do
          call move_alloc(grown, state%chunks)
       end if
       state%n_chunks = state%n_chunks + 1
       state%chunks(state%n_chunks)%chunk => fresh
    end if

    i = state%free_slot
    state%free_slot = i + 1
    node => chunk%nodes(i)
    call clear_record(node)

    state%last_id = state%last_id + 1_int64
    node%id = int(state%last_id)
    node%gen = state%gen
    node%ctx = max(1, min(max_contexts, ctx%slot))
  end function allocate_node
!###############################################################################


!###############################################################################
  module function allocate_value_storage(ctx, rank, array_shape) result(node)
    !! Allocates a node record together with value storage.
    !!
    !! `array_shape` gives the *storage* dimensions, either `[n]` or
    !! `[n, samples]`. Storage is always rank 2, so a rank-1 logical array of
    !! n elements becomes an (n, 1) view. The logical shape is set to `[n]`
    !! here; callers that represent a rank-2 logical array override it.
    type(array_context_type), intent(in) :: ctx
    !! The owning context
    integer, intent(in) :: rank
    !! Logical rank of the array
    integer, dimension(:), intent(in) :: array_shape
    !! Storage dimensions
    type(array_type), pointer :: node
    !! The new node

    node => allocate_node(ctx)
    if(size(array_shape) == 1) then
       call take_value(ctx, node%store, [array_shape(1), 1])
    else
       call take_value(ctx, node%store, [array_shape(1), array_shape(2)])
    end if
    node%val => node%store
    node%allocated = .true.
    node%rank = rank
    node%size = array_shape(1)
    node%is_sample_dependent = .true.
    allocate(node%shape(rank))
    node%shape = array_shape(1:rank)
  end function allocate_value_storage
!###############################################################################


!###############################################################################
  module function find_node(ctx, id) result(node)
    !! Finds the live node record with the given id.
    !!
    !! Ids are issued in order and every chunk holds a consecutive id range,
    !! so a record is addressed directly from its id.
    type(array_context_type), intent(in) :: ctx
    !! The owning context
    integer, intent(in) :: id
    !! Id of the record
    type(array_type), pointer :: node
    !! The record, or null when it is not live

    ! Local variables
    type(context_state), pointer :: state
    type(ad_registry), pointer :: chunk
    integer :: i, k

    nullify(node)
    if(id <= 0) return
    state => context_state_of(ctx)
    if(id > state%last_id) return
    ! Every chunk but the last holds exactly node_chunk_size records carrying
    ! consecutive ids, so the owning chunk follows from the id.
    k = int((id - 1) / node_chunk_size) + 1
    if(k < 1 .or. k > state%n_chunks) return
    chunk => state%chunks(k)%chunk
    if(.not. associated(chunk)) return
    i = int(int(id, int64) - chunk%base)
    if(i < 1 .or. i > size(chunk%nodes)) return
    if(chunk%nodes(i)%id == id .and. chunk%nodes(i)%gen == state%gen) then
       node => chunk%nodes(i)
    end if
  end function find_node
!###############################################################################


!###############################################################################
  module subroutine take_value(ctx, store, array_shape)
    !! Allocates a value buffer, reusing a released one when possible.
    !!
    !! The buffer is moved (no copy) out of the context's cache, so repeated
    !! graph builds do not allocate and fault in fresh pages every time.
    type(array_context_type), intent(in) :: ctx
    integer, dimension(2), intent(in) :: array_shape
    real(real32), dimension(:,:), allocatable, intent(out) :: store

    ! Local variables
    type(context_state), pointer :: state
    integer :: i

    state => context_state_of(ctx)
    do i = state%value_cache_n, 1, -1
       if(allocated(state%value_cache(i)%data)) then
          if(all(state%value_cache(i)%array_shape == array_shape)) then
             call move_alloc(state%value_cache(i)%data, store)
             ! Move the last entry into the hole so nothing is lost.
             if(i /= state%value_cache_n) then
                call move_alloc(state%value_cache(state%value_cache_n)%data, &
                     state%value_cache(i)%data)
                state%value_cache(i)%array_shape = &
                     state%value_cache(state%value_cache_n)%array_shape
             end if
             state%value_cache(state%value_cache_n)%array_shape = 0
             state%value_cache_n = state%value_cache_n - 1
             return
          end if
       end if
    end do
    allocate(store(array_shape(1), array_shape(2)))
  end subroutine take_value
!###############################################################################


!###############################################################################
  module subroutine give_value(ctx, store)
    !! Returns a value buffer to the context for reuse.
    type(array_context_type), intent(in) :: ctx
    real(real32), dimension(:,:), allocatable, intent(inout) :: store

    ! Local variables
    type(context_state), pointer :: state
    integer :: i

    if(.not. allocated(store)) return
    state => context_state_of(ctx)
    i = state%value_cache_n + 1
    if(i > value_cache_size) then
       deallocate(store)
       return
    end if
    state%value_cache(i)%array_shape = shape(store)
    call move_alloc(store, state%value_cache(i)%data)
    state%value_cache_n = i
  end subroutine give_value
!###############################################################################


!###############################################################################
  module subroutine release_node(ctx, node)
    !! Releases every buffer a node record owns and empties the record.
    !!
    !! This is the only place storage is returned to the runtime. It is called
    !! by the context when a region is discarded; callers must not keep any
    !! handle to the node afterwards.
    type(array_context_type), intent(in) :: ctx
    !! The owning context
    type(array_type), intent(inout) :: node
    !! The record to empty

    call give_value(ctx, node%store)
    node%val => null()
    if(allocated(node%shape)) deallocate(node%shape)
    if(allocated(node%indices)) deallocate(node%indices)
    if(allocated(node%adj_ja)) deallocate(node%adj_ja)
    if(allocated(node%mask)) deallocate(node%mask)
    if(allocated(node%direction)) deallocate(node%direction)
    ! Records are never reused (slots are retired, not recycled), so only the
    ! identity and the links need clearing: `find_node` treats id=0 as dead and
    ! no live handle may point at this record any more.
    node%id = 0
    node%grad => null()
    node%left_operand => null()
    node%right_operand => null()
    node%get_partial_left => null()
    node%get_partial_right => null()
    node%get_partial_left_val => null()
    node%get_partial_right_val => null()
    node%get_partial_left_val_sum => null()
    node%get_partial_right_val_sum => null()
  end subroutine release_node
!###############################################################################


!###############################################################################
  module subroutine reset_context(ctx)
    !! Releases every node record and every scratch buffer of a context.
    type(array_context_type), intent(in) :: ctx
    !! The context to reset

    ! Local variables
    type(context_state), pointer :: state
    type(ad_registry), pointer :: chunk, next
    integer :: i

    state => context_state_of(ctx)
    chunk => state%head
    do while(associated(chunk))
       next => chunk%next
       if(associated(chunk%nodes)) then
          do i = 1, size(chunk%nodes)
             call release_node(ctx, chunk%nodes(i))
          end do
          deallocate(chunk%nodes)
       end if
       deallocate(chunk)
       chunk => next
    end do
    nullify(state%head)
    nullify(state%tail)
    nullify(state%mark_chunk)
    if(allocated(state%chunks)) deallocate(state%chunks)
    state%n_chunks = 0
    if(associated(state%pool)) then
       do i = 1, state%pool_top
          if(associated(state%pool(i)%array)) deallocate(state%pool(i)%array)
       end do
       deallocate(state%pool)
       nullify(state%pool)
    end if
    do i = 1, state%value_cache_n
       if(allocated(state%value_cache(i)%data)) &
            deallocate(state%value_cache(i)%data)
       state%value_cache(i)%array_shape = 0
    end do
    state%value_cache_n = 0
    state%pool_top = 0
    state%pool_cap = 0
    state%last_id = 0_int64
    state%free_slot = 1
    state%gen = state%gen + 1
    state%mark_id = 0_int64
    state%mark_slot = 1
    state%mark_gen = state%gen
  end subroutine reset_context
!###############################################################################


!###############################################################################
  module subroutine checkpoint_context(ctx)
    !! Marks the current allocation state for a later rollback.
    type(array_context_type), intent(in) :: ctx
    !! The context to checkpoint

    ! Local variables
    type(context_state), pointer :: state

    state => context_state_of(ctx)
    state%mark_id = state%last_id
    state%mark_slot = state%free_slot
    state%mark_gen = state%gen
    state%mark_chunk => state%tail
  end subroutine checkpoint_context
!###############################################################################


!###############################################################################
  module subroutine rollback_context(ctx)
    !! Releases every node created since the last checkpoint.
    !!
    !! Records created before the checkpoint are untouched, so every handle
    !! bound to them keeps its views valid. The registry chunks themselves are
    !! kept, so earlier records never move.
    type(array_context_type), intent(in) :: ctx
    !! The context to roll back

    ! Local variables
    type(context_state), pointer :: state
    type(ad_registry), pointer :: chunk
    integer :: i

    state => context_state_of(ctx)
    if(state%gen /= state%mark_gen) return
    chunk => state%mark_chunk
    if(.not. associated(chunk)) return
    i = state%mark_slot
    do while(associated(chunk))
       if(associated(chunk%nodes)) then
          do while(i <= size(chunk%nodes))
             if(chunk%nodes(i)%id > 0) then
                if(int(chunk%nodes(i)%id, int64) > state%mark_id) then
                   call release_node(ctx, chunk%nodes(i))
                end if
             end if
             i = i + 1
          end do
       end if
       chunk => chunk%next
       i = 1
    end do
    ! Slots are deliberately not reused: ids must stay monotonic so that a
    ! stale handle is detected, and a record is never overwritten while some
    ! pointer into the registry may still refer to it.
    state%mark_slot = state%free_slot
    if(associated(state%pool)) then
       do i = 1, state%pool_top
          state%pool(i)%in_use = .false.
       end do
       state%pool_top = 0
    end if
  end subroutine rollback_context
!###############################################################################


!###############################################################################
  module subroutine sweep_region(ctx, keep_tag)
    !! Releases every record created since a checkpoint that is not marked.
    !!
    !! `keep_tag` is the traversal stamp written by the caller's mark pass; any
    !! record issued after `mark_id` whose stamp differs is transient and is
    !! returned to the context. This is the bulk reclamation that keeps repeated
    !! forward and backward passes in bounded memory.
    type(array_context_type), intent(in) :: ctx
    !! The owning context
    integer, intent(in) :: keep_tag
    !! The traversal stamp that marks live records

    ! Local variables
    type(context_state), pointer :: state
    type(ad_registry), pointer :: chunk
    integer :: i

    state => context_state_of(ctx)
    chunk => state%mark_chunk
    if(.not. associated(chunk)) return
    ! Only the records issued after the checkpoint can be transient: the first
    ! scan starts at the checkpoint's slot, every later chunk is scanned whole.
    i = state%mark_slot
    do while(associated(chunk))
       if(associated(chunk%nodes)) then
          do while(i <= size(chunk%nodes))
             if(chunk%nodes(i)%id > 0) then
                if(chunk%nodes(i)%visit_tag /= keep_tag) then
                   call release_node(ctx, chunk%nodes(i))
                end if
             end if
             i = i + 1
          end do
       end if
       chunk => chunk%next
       i = 1
    end do
  end subroutine sweep_region
!###############################################################################


!###############################################################################
  module function acquire_scratch(ctx, array_shape) result(buffer)
    !! Checks out a pooled rank-2 scratch buffer of the given shape.
    type(array_context_type), intent(in) :: ctx
    !! The owning context
    integer, dimension(2), intent(in) :: array_shape
    !! Shape of the scratch buffer
    real(real32), dimension(:, :), pointer :: buffer
    !! The scratch buffer

    ! Local variables
    type(context_state), pointer :: state
    integer :: i

    state => context_state_of(ctx)
    if(.not. associated(state%pool)) then
       state%pool_cap = 8
       allocate(state%pool(state%pool_cap))
       state%pool_top = 0
    end if
    do i = 1, state%pool_top
       if(.not. state%pool(i)%in_use) then
          if(all(state%pool(i)%array_shape == array_shape)) then
             state%pool(i)%in_use = .true.
             buffer => state%pool(i)%array
             return
          end if
       end if
    end do
    if(state%pool_top >= state%pool_cap) then
       call grow_pool(state)
    end if
    state%pool_top = state%pool_top + 1
    i = state%pool_top
    if(associated(state%pool(i)%array)) deallocate(state%pool(i)%array)
    allocate(state%pool(i)%array(array_shape(1), array_shape(2)))
    state%pool(i)%array = 0.0_real32
    state%pool(i)%array_shape = array_shape
    state%pool(i)%in_use = .true.
    buffer => state%pool(i)%array
  end function acquire_scratch
!###############################################################################


!###############################################################################
  module subroutine release_scratch(ctx, buffer)
    !! Returns a pooled scratch buffer to the context.
    type(array_context_type), intent(in) :: ctx
    !! The owning context
    real(real32), dimension(:, :), pointer :: buffer
    !! The buffer to return

    ! Local variables
    type(context_state), pointer :: state
    integer :: i

    state => context_state_of(ctx)
    do i = 1, state%pool_top
       if(associated(state%pool(i)%array, buffer)) then
          state%pool(i)%in_use = .false.
          return
       end if
    end do
    nullify(buffer)
  end subroutine release_scratch
!###############################################################################


!###############################################################################
  subroutine grow_pool(state)
    !! Doubles the capacity of a context's scratch pool.
    type(context_state), intent(inout) :: state
    !! The context state to grow

    ! Local variables
    type(pooled_buffer), dimension(:), pointer :: bigger

    allocate(bigger(state%pool_cap * 2))
    bigger(1:state%pool_top) = state%pool(1:state%pool_top)
    deallocate(state%pool)
    state%pool => bigger
    state%pool_cap = state%pool_cap * 2
  end subroutine grow_pool
!###############################################################################


!###############################################################################
  subroutine clear_record(node)
    !! Empties every component of a node record without freeing storage.
    type(array_type), intent(inout), target :: node
    !! The record to empty

    node%id = 0
    node%gen = 1
    node%ctx = 1
    node%rank = 0
    node%size = 0
    node%sample_dependent_flag = .true.
    node%is_sample_dependent => node%sample_dependent_flag
    node%is_scalar = .false.
    node%is_forward = .false.
    node%allocated = .false.
    node%requires_grad = .false.
    node%operation = 'none'
    node%temporary_flag = .true.
    node%is_temporary => node%temporary_flag
    node%visit_tag = 0
    node%phase = 0
    node%self => node
    if(allocated(node%shape)) deallocate(node%shape)
    if(allocated(node%indices)) deallocate(node%indices)
    if(allocated(node%adj_ja)) deallocate(node%adj_ja)
    if(allocated(node%mask)) deallocate(node%mask)
    if(allocated(node%direction)) deallocate(node%direction)
    node%val => null()
    node%grad => null()
    node%left_operand => null()
    node%right_operand => null()
    node%get_partial_left => null()
    node%get_partial_right => null()
    node%get_partial_left_val => null()
    node%get_partial_right_val => null()
    node%get_partial_left_val_sum => null()
    node%get_partial_right_val_sum => null()
  end subroutine clear_record
!###############################################################################


!###############################################################################
  subroutine set_default_context(ctx)
    !! Replaces the process-wide default context.
    !!
    !! The previous default context is reset, releasing every node it owned.
    type(array_context_type), intent(in) :: ctx
    !! The context to install as the default

    call reset_context(default_context)
    default_context%slot = max(1, min(max_contexts, ctx%slot))
    default_context%gen = context_states(default_context%slot)%gen
  end subroutine set_default_context
!###############################################################################


!###############################################################################
  subroutine reset_default_context()
    !! Resets the process-wide default context, releasing all its memory.

    call reset_context(default_context)
  end subroutine reset_default_context
!###############################################################################


end submodule diffstruc__context_impl
