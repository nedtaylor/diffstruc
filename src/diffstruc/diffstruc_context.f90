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

contains


!###############################################################################
  module function context_state_of(ctx) result(state)
    !! The private state of a context.
    type(array_context_type), intent(in) :: ctx
    !! Context handle
    type(context_state), pointer :: state
    !! Its state

    state => ctx%state
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
    integer :: i, n, k, slot

    state => context_state_of(ctx)

    if(state%free_n > 0) then
       ! Reuse a released record slot. Ids stay monotonic, so a stale handle
       ! pointing at this slot still fails the id check in `find_node`.
       slot = state%free_slots(state%free_n)
       state%free_n = state%free_n - 1
       k = int((slot - 1) / node_chunk_size) + 1
       chunk => state%chunks(k)%chunk
       i = slot - (k - 1) * node_chunk_size
    else
       ! Chunks are appended in order, so the last one is the only candidate
       ! for fresh slots.
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
          nullify(fresh%next)
          if(associated(chunk)) then
             chunk%next => fresh
          else
             state%head => fresh
          end if
          chunk => fresh
          state%tail => fresh
          state%free_slot = 1
          ! Record the chunk so a slot can be resolved from its global index.
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
       slot = (state%n_chunks - 1) * node_chunk_size + i
    end if

    node => chunk%nodes(i)
    call clear_record(node)

    state%last_id = state%last_id + 1_int64
    node%id = int(state%last_id)
    node%slot = slot
    node%gen = state%gen
    node%ctx%state => state
    node%ctx%gen = state%gen
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
    !! Ids are issued monotonically and never reused, but slots are recycled,
    !! so the record is located by scanning the registry. This is only a
    !! fallback path (a bound handle already knows its record), which keeps
    !! registry memory proportional to the live graph rather than to the
    !! number of records ever issued.
    type(array_context_type), intent(in) :: ctx
    !! The owning context
    integer, intent(in) :: id
    !! Id of the record
    type(array_type), pointer :: node
    !! The record, or null when it is not live

    ! Local variables
    type(context_state), pointer :: state
    type(ad_registry), pointer :: chunk
    integer :: i

    nullify(node)
    if(id <= 0) return
    state => context_state_of(ctx)
    if(id > state%last_id) return
    chunk => state%head
    do while(associated(chunk))
       if(associated(chunk%nodes)) then
          do i = 1, size(chunk%nodes)
             if(chunk%nodes(i)%id == id .and. chunk%nodes(i)%gen == state%gen) then
                node => chunk%nodes(i)
                return
             end if
          end do
       end if
       chunk => chunk%next
    end do
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
             state%value_cache_bytes = state%value_cache_bytes - &
                  int(size(state%value_cache(i)%data), int64) * 4_int64
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
    if(state%value_cache_bytes + int(size(store), int64) * 4_int64 > &
         value_cache_max_bytes) then
       deallocate(store)
       return
    end if
    state%value_cache(i)%array_shape = shape(store)
    state%value_cache_bytes = state%value_cache_bytes + &
         int(size(store), int64) * 4_int64
    call move_alloc(store, state%value_cache(i)%data)
    state%value_cache_n = i
  end subroutine give_value
!###############################################################################


!###############################################################################
  module subroutine release_node(ctx, node, recycle)
    !! Releases every buffer a node record owns and empties the record.
    type(array_context_type), intent(in) :: ctx
    !! The owning context
    type(array_type), intent(inout) :: node
    !! The record to empty
    logical, intent(in), optional :: recycle
    !! Whether the slot may be reused immediately

    ! Local variables
    type(context_state), pointer :: state
    logical :: recycle_

    if(node%id <= 0) return
    recycle_ = .false.
    if(present(recycle)) recycle_ = recycle
    if(recycle_ .and. node%slot > 0) then
       state => context_state_of(ctx)
       call push_free_slot(state, node%slot)
    end if

    call give_value(ctx, node%store)
    node%val => null()
    if(allocated(node%shape)) deallocate(node%shape)
    if(allocated(node%indices)) deallocate(node%indices)
    if(allocated(node%adj_ja)) deallocate(node%adj_ja)
    if(allocated(node%mask)) deallocate(node%mask)
    if(allocated(node%direction)) deallocate(node%direction)
    ! The slot may be on the free list now; only the identity and links need
    ! clearing. `find_node` treats id=0 as dead and ids are never reused.
    node%id = 0
    node%slot = 0
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
             ! The chunks themselves are dropped below, so the slots do not
             ! need to go on the free list.
             call release_node(ctx, chunk%nodes(i))
          end do
          deallocate(chunk%nodes)
       end if
       deallocate(chunk)
       chunk => next
    end do
    nullify(state%head)
    nullify(state%tail)
    if(allocated(state%chunks)) deallocate(state%chunks)
    if(allocated(state%free_slots)) deallocate(state%free_slots)
    state%free_n = 0
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
    state%value_cache_bytes = 0_int64
    state%pool_top = 0
    state%pool_cap = 0
    state%last_id = 0_int64
    state%free_slot = 1
    state%gen = state%gen + 1
    state%mark_id = 0_int64
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
    state%mark_gen = state%gen
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
    ! Slots are recycled, so a record created after the checkpoint may live in
    ! any chunk/slot. Select records by id instead of by registry position.
    chunk => state%head
    do while(associated(chunk))
       if(associated(chunk%nodes)) then
          do i = 1, size(chunk%nodes)
             if(chunk%nodes(i)%id > 0) then
                if(int(chunk%nodes(i)%id, int64) > state%mark_id) then
                   call release_node(ctx, chunk%nodes(i), recycle=.true.)
                end if
             end if
          end do
       end if
       chunk => chunk%next
    end do
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
    ! Slots are recycled, so scan every chunk and select by id.
    chunk => state%head
    do while(associated(chunk))
       if(associated(chunk%nodes)) then
          do i = 1, size(chunk%nodes)
             if(chunk%nodes(i)%id > 0) then
                if(int(chunk%nodes(i)%id, int64) > state%mark_id) then
                   if(chunk%nodes(i)%visit_tag /= keep_tag) then
                      call release_node(ctx, chunk%nodes(i), recycle=.true.)
                   end if
                end if
             end if
          end do
       end if
       chunk => chunk%next
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
  subroutine push_free_slot(state, slot)
    !! Records a released slot for reuse.
    type(context_state), intent(inout) :: state
    integer, intent(in) :: slot
    integer, allocatable :: grown(:)

    if(.not. allocated(state%free_slots)) then
       allocate(state%free_slots(1024))
       state%free_n = 0
    else if(state%free_n == size(state%free_slots)) then
       allocate(grown(2*size(state%free_slots)))
       grown(1:state%free_n) = state%free_slots(1:state%free_n)
       call move_alloc(grown, state%free_slots)
    end if
    state%free_n = state%free_n + 1
    state%free_slots(state%free_n) = slot
  end subroutine push_free_slot
!###############################################################################


!###############################################################################
  subroutine clear_record(node)
    !! Empties every component of a node record without freeing storage.
    type(array_type), intent(inout), target :: node
    !! The record to empty

    node%id = 0
    node%slot = 0
    node%gen = 1
    nullify(node%ctx%state)
    node%ctx%gen = 1
    node%rank = 0
    node%size = 0
    node%sample_dependent_flag = .true.
    node%is_sample_dependent => node%sample_dependent_flag
    node%is_scalar = .false.
    node%is_forward = .false.
    node%allocated = .false.
    node%requires_grad = .false.
    node%value_partial_ok = .false.
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
  subroutine ensure_default_context()
    !! Allocates this thread's default context state on first use.
    !!
    !! `default_context` is thread-private, so every thread gets an independent
    !! registry and pool. The state is never freed by the library (it stays
    !! reachable until the process ends, which is what a default context is
    !! for); `destroy_context` can free an explicit context.

    if(associated(default_context%state)) return
    allocate(default_context%state)
    default_context%state%n_available = available_threads()
    default_context%gen = default_context%state%gen
  end subroutine ensure_default_context
!###############################################################################


!###############################################################################
  module function default_context_of() result(ctx)
    !! This thread's default context, allocated on first use.
    type(array_context_type) :: ctx

    call ensure_default_context()
    ctx = default_context
  end function default_context_of
!###############################################################################


!###############################################################################
  module subroutine new_context(ctx, n_threads, max_recursion_depth)
    !! Creates an independent context with its own registry and pools.
    type(array_context_type), intent(out) :: ctx
    integer, intent(in), optional :: n_threads
    integer, intent(in), optional :: max_recursion_depth

    if(associated(ctx%state)) then
       call stop_program('diffstruc: context is already created')
       return
    end if
    allocate(ctx%state)
    ctx%state%n_available = available_threads()
    if(present(n_threads)) ctx%state%n_threads = max(0, n_threads)
    if(present(max_recursion_depth)) then
       ctx%state%max_recursion_depth = max(1, max_recursion_depth)
    end if
    ctx%gen = ctx%state%gen
  end subroutine new_context
!###############################################################################


!###############################################################################
  module subroutine destroy_context(ctx)
    !! Releases everything a context owns and frees its state.
    !!
    !! Handles that referred to the context become unbound/stale. Do not call
    !! this on the thread's default context while it is still in use.
    type(array_context_type), intent(inout) :: ctx

    if(.not. associated(ctx%state)) return
    call reset_context(ctx)
    deallocate(ctx%state)
    nullify(ctx%state)
    ctx%gen = 1
  end subroutine destroy_context
!###############################################################################


!###############################################################################
  module subroutine set_num_threads(ctx, n_threads)
    !! Sets the thread count used by this context's kernels.
    type(array_context_type), intent(in) :: ctx
    integer, intent(in) :: n_threads

    if(associated(ctx%state)) ctx%state%n_threads = max(0, n_threads)
  end subroutine set_num_threads
!###############################################################################


!###############################################################################
  module function get_num_threads(ctx) result(n_threads)
    !! The thread count this context's kernels request (0 = OpenMP default).
    type(array_context_type), intent(in) :: ctx
    integer :: n_threads

    n_threads = 0
    if(associated(ctx%state)) n_threads = ctx%state%n_threads
  end function get_num_threads
!###############################################################################


!###############################################################################
  module subroutine set_max_recursion_depth(ctx, max_depth)
    !! Sets the graph traversal depth limit for this context.
    type(array_context_type), intent(in) :: ctx
    integer, intent(in) :: max_depth

    if(associated(ctx%state)) ctx%state%max_recursion_depth = max(1, max_depth)
  end subroutine set_max_recursion_depth
!###############################################################################


!###############################################################################
  subroutine set_default_context(ctx)
    !! Replaces this thread's default context.
    !!
    !! The previous default context is reset, releasing every node it owned;
    !! its state is kept (not freed), so existing node pointers remain safe and
    !! stale handles are rejected by the generation check.
    type(array_context_type), intent(in) :: ctx

    call ensure_default_context()
    if(associated(default_context%state)) then
       if(.not. associated(default_context%state, ctx%state)) then
          call reset_context(default_context)
       end if
    end if
    default_context = ctx
  end subroutine set_default_context
!###############################################################################


!###############################################################################
  subroutine reset_default_context()
    !! Resets this thread's default context, releasing all its memory.

    call ensure_default_context()
    call reset_context(default_context)
  end subroutine reset_default_context
!###############################################################################


end submodule diffstruc__context_impl
