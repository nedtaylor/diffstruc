module diffstruc__types
  !! This module contains the automatic differentiation data structures.
  !!
  !! `array_type` is a lightweight, non-owning handle. It carries the id of
  !! a node record plus a set of views (pointers) into memory that the
  !! owning `array_context_type` allocated. Copying a handle copies the
  !! views, never the storage, so a handle can be returned from a function,
  !! passed by value, or allowed to go out of scope without any risk of a
  !! dangling pointer or a double free. All memory -- node records, value
  !! storage, gradients and scratch buffers -- belongs to the context and is
  !! released by the context in bulk (`reset`) or back to a checkpoint
  !! (`rollback_region`). There is no reference counting and no per-handle
  !! deallocation anywhere in the library.
  use, intrinsic :: iso_fortran_env, only: error_unit, int64, real32
  use coreutils, only: stop_program, print_warning
  implicit none

  private

  public :: array_type, array_context_type, get_partial
  public :: default_context, set_default_context, reset_default_context
  public :: canonical, constant_node, context_of, live

  public :: operator(+), operator(-), operator(*), operator(/), operator(**)
  public :: sum, mean, spread, unspread, exp, log

  integer, parameter :: max_contexts = 8
  !! Number of independent contexts the library can address.
  integer, parameter :: node_chunk_size = 128
  !! Number of node records allocated per registry chunk.


!-------------------------------------------------------------------------------
! Context ownership
!-------------------------------------------------------------------------------
  type :: array_context_type
     !! A handle to a memory-owning context.
     !!
     !! The context's state (node registry, generation, scratch pool) is held
     !! in module storage indexed by `slot`. A context value is therefore
     !! trivially copyable: copying it names the same owner, it does not
     !! duplicate or fork the storage.
     integer :: slot = 1
     !! The module-storage slot of the owning context
     integer :: gen = 1
     !! The allocation generation of the context when it was captured
  end type array_context_type

  type :: pooled_buffer
     !! A pooled rank-2 real32 scratch buffer. Buffers hold only transient
     !! scratch data, never graph-node storage, and are reused across
     !! operations to keep steady-state allocation at zero.
     integer, dimension(2) :: array_shape = 0
     logical :: in_use = .false.
     real(real32), dimension(:, :), pointer :: array => null()
  end type pooled_buffer

  type :: ad_registry
     !! A chunk of node records owned by a context.
     !!
     !! A chunk is allocated once and never grown or moved, so a pointer into
     !! one of its records stays valid until the owning context releases it.
     !! Ids are issued in order, so the records of a chunk carry a consecutive
     !! range of ids and a node can be addressed from its id alone.
     type(array_type), dimension(:), pointer :: nodes => null()
     integer(int64) :: base = 0_int64
     type(ad_registry), pointer :: next => null()
  end type ad_registry

  type :: chunk_slot
     !! A pointer to one registry chunk, for direct addressing by id.
     !!
     !! Chunks are created in order and hold a fixed number of records each,
     !! so the chunk that owns an id is found by arithmetic instead of by
     !! walking the linked list. The list itself is still used to release a
     !! context, where the order does not matter.
     type(ad_registry), pointer :: chunk => null()
  end type chunk_slot

  type :: cached_value
     !! A released value buffer kept for reuse by the context.
     real(real32), dimension(:,:), allocatable :: data
     integer, dimension(2) :: array_shape = 0
  end type cached_value

  integer, parameter :: value_cache_size = 256
  !! Number of released value buffers a context keeps for reuse.

  type :: context_state
     !! The private state of one context: its node registry and scratch pool.
     type(ad_registry), pointer :: head => null()
     type(ad_registry), pointer :: tail => null()
     type(ad_registry), pointer :: mark_chunk => null()
     type(chunk_slot), dimension(:), allocatable :: chunks
     !! Direct index into the registry chunks, addressed by id
     integer :: n_chunks = 0
     !! Number of entries used in `chunks`
     integer(int64) :: last_id = 0_int64
     integer :: free_slot = 1
     integer :: gen = 1
     integer(int64) :: mark_id = 0_int64
     integer :: mark_slot = 1
     integer :: mark_gen = 1
     integer :: visit_counter = 0
     type(pooled_buffer), dimension(:), pointer :: pool => null()
     integer :: pool_top = 0
     integer :: pool_cap = 0
     type(cached_value) :: value_cache(value_cache_size)
     integer :: value_cache_n = 0
  end type context_state


!-------------------------------------------------------------------------------
! The non-owning handle
!-------------------------------------------------------------------------------
  type :: array_type
     !! A lightweight, non-owning handle to an autodiff array.
     !!
     !! The first three components identify the context-owned node record.
     !! Every remaining component is a view: a pointer into storage owned by
     !! that node's context, or a pointer to another context-owned node. A
     !! handle never owns what it points at, so no destructor, finaliser or
     !! reference count is needed.
     integer :: id = 0
     !! The id of the referenced node record (zero when unbound)
     integer :: gen = 1
     !! The context generation the handle was bound in
     integer :: ctx = 1
     !! The context slot that owns the referenced node

     integer :: rank = 0
     !! Rank of the array
     integer, dimension(:), allocatable :: shape
     !! Shape of the array (context-owned)
     integer :: size = 0
     !! Number of elements per sample
     logical :: sample_dependent_flag = .true.
     !! Context-owned storage of the sample-dependence flag
     logical, pointer :: is_sample_dependent => null()
     !! Whether the array is sample-dependent (view of the owned flag)
     logical :: is_scalar = .false.
     !! Whether the array represents a scalar value
     logical :: is_forward = .false.
     !! Whether the node was produced in forward mode
     logical :: allocated = .false.
     !! Whether value storage is live
     real(real32), dimension(:, :), allocatable :: store
     !! Array values owned by the node record (rank 2, element x sample)
     real(real32), dimension(:, :), contiguous, pointer :: val => null()
     !! View of `store` for handle-based access (never owns memory)
     integer, dimension(:), allocatable :: indices
     !! Indices used for gradient accumulation
     integer, dimension(:, :), allocatable :: adj_ja
     !! Auxiliary integer data for an operation
     logical, dimension(:, :), allocatable :: mask
     !! Mask data for an operation
     logical :: requires_grad = .false.
     !! Whether this node is an independent variable of the graph
     type(array_type), pointer :: grad => null()
     !! Gradient accumulator node
     type(array_type), pointer :: left_operand => null()
     !! Left operand of the operation
     type(array_type), pointer :: right_operand => null()
     !! Right operand of the operation
     character(len=32) :: operation = 'none'
     !! Name of the operation that produced this node
     logical :: temporary_flag = .true.
     !! Context-owned storage of the temporary flag
     logical, pointer :: is_temporary => null()
     !! Whether the array is a disposable expression temporary
     integer :: visit_tag = 0
     !! Traversal marker for graph walks
     type(array_type), pointer :: self => null()
     !! Direct pointer to the context-owned record this handle views
     integer :: phase = 0
     !! Accumulation-pass tag: a gradient is overwritten on its first
     !! contribution in a backward pass and added to afterwards
     real(real32), dimension(:), allocatable :: direction
     !! Forward-mode seed direction (context-owned)

     procedure(get_partial), pass(this), pointer :: get_partial_left => null()
     !! Partial derivative with respect to the left operand
     procedure(get_partial), pass(this), pointer :: get_partial_right => null()
     !! Partial derivative with respect to the right operand

     procedure(get_partial_val), pass(this), pointer :: get_partial_left_val => null()
     !! Value partial derivative with respect to the left operand
     procedure(get_partial_val), pass(this), pointer :: get_partial_right_val => null()
     !! Value partial derivative with respect to the right operand

     procedure(get_partial_val_sum), pass(this), pointer :: &
          get_partial_left_val_sum => null()
     !! Sum-reduced value partial derivative with respect to the left operand
     procedure(get_partial_val_sum), pass(this), pointer :: &
          get_partial_right_val_sum => null()
     !! Sum-reduced value partial derivative with respect to the right operand

   contains
     procedure, pass(this) :: allocate => allocate_array
     !! Allocates the array's value storage in its context
     procedure, pass(this) :: deallocate => deallocate_array
     !! Releases this handle's binding (the context keeps the storage)
     procedure, pass(this) :: flatten => flatten_array
     !! Copies the values into a flat rank-1 array
     procedure, pass(this) :: assign_shallow
     !! Binds this handle to another handle's node
     procedure, pass(this) :: assign_and_deallocate_source
     !! Copies a source array into this handle
     procedure, pass(this) :: set => set_array
     !! Assigns array values
     procedure, pass(this) :: extract => extract_array
     !! Copies the values into a standard real array
     procedure, pass(this) :: set_shape
     !! Updates the logical shape of the array
     procedure, pass(this) :: set_direction
     !! Sets the forward-mode seed direction of a variable
     procedure, pass(this) :: set_requires_grad
     !! Marks the array as an independent variable
     procedure, pass(this) :: grad_reverse
     !! Reverse-mode: accumulates gradients with respect to all inputs
     procedure, pass(this) :: grad_forward
     !! Forward-mode: derivative with respect to a variable
     procedure, pass(this) :: zero_grad
     !! Zeroes this array's gradient
     procedure, pass(this) :: zero_all_grads
     !! Zeroes the gradients of this array's graph
     procedure, pass(this) :: reset_graph
     !! Releases the gradient side of this array's graph
     procedure, pass(this) :: duplicate_graph
     !! Deep-copies this array's computation graph
     procedure, pass(this) :: nullify_graph
     !! Releases the graph structure rooted at this array
     procedure, pass(this) :: get_ptr_from_id
     !! Finds the node of this array's graph with the given id
     procedure, pass(this) :: detach
     !! Detaches this array from the computation graph
     procedure, pass(this) :: create_result => create_result_array
     !! Creates a result node in this array's context
     procedure, pass(this) :: canonical
     !! The context-owned node record this handle refers to
     procedure, pass(this) :: live
     !! The context-owned node record, stopping on stale access
     procedure, pass(this) :: print_graph
     !! Prints the computation graph

     procedure :: add_arrays, add_real1d, add_real2d, add_scalar
     generic, public :: operator(+) => &
          add_arrays, add_real1d, add_real2d, add_scalar
     !! Overloaded addition operator

     procedure :: subtract_arrays, subtract_real2d, subtract_real1d, &
          subtract_scalar, negate_array
     generic, public :: operator(-) => &
          subtract_arrays, subtract_real2d, subtract_real1d, &
          subtract_scalar, negate_array
     !! Overloaded subtraction operator

     procedure :: multiply_arrays, multiply_real2d, &
          multiply_scalar, multiply_logical
     generic, public :: operator(*) => &
          multiply_arrays, multiply_real2d, &
          multiply_scalar, multiply_logical
     !! Overloaded multiplication operator

     procedure :: divide_arrays, divide_scalar, divide_real1d
     generic, public :: operator(/) => divide_arrays, divide_scalar, divide_real1d
     !! Overloaded division operator

     procedure :: power_arrays, power_real_scalar, power_int_scalar
     generic, public :: operator(**) => &
          power_arrays, power_real_scalar, power_int_scalar
     !! Overloaded power operator

     procedure :: assign => assign_array
     generic, public :: assignment(=) => assign
     !! Overloaded assignment operator
  end type array_type


  ! Constructor interface
  !-----------------------------------------------------------------------------
  interface array_type
     module function constructor() result(this)
       type(array_type) :: this
     end function constructor
  end interface array_type


  ! Lifecycle and graph interfaces
  !-----------------------------------------------------------------------------
  interface
     module subroutine allocate_array(this, array_shape, source, &
          sample_dependent, temporary)
       class(array_type), intent(inout), target :: this
       integer, dimension(:), intent(in), optional :: array_shape
       class(*), dimension(..), intent(in), optional :: source
       logical, intent(in), optional :: sample_dependent
       logical, intent(in), optional :: temporary
     end subroutine allocate_array

     module subroutine deallocate_array(this, keep_shape)
       !! Releases this handle's binding; the context keeps the storage.
       class(array_type), intent(inout) :: this
       logical, intent(in), optional :: keep_shape
     end subroutine deallocate_array

     pure module function flatten_array(this) result(output)
       class(array_type), intent(in) :: this
       real(real32), dimension(this%size) :: output
     end function flatten_array

     module subroutine assign_shallow(this, source)
       class(array_type), intent(inout) :: this
       class(array_type), intent(in), target :: source
     end subroutine assign_shallow

     module subroutine assign_and_deallocate_source(this, source)
       class(array_type), intent(inout) :: this
       type(array_type), intent(inout), pointer :: source
     end subroutine assign_and_deallocate_source

     recursive module subroutine assign_array(this, input)
       class(array_type), intent(inout), target :: this
       type(array_type), intent(in) :: input
     end subroutine assign_array

     module function create_result_array(this, array_shape) result(result_ptr)
       class(array_type), intent(in), target :: this
       integer, dimension(:), intent(in), optional :: array_shape
       type(array_type), pointer :: result_ptr
     end function create_result_array

     module function canonical(this) result(node)
       !! The context-owned node record this handle refers to.
       class(array_type), intent(in) :: this
       type(array_type), pointer :: node
     end function canonical

     module function live(this) result(node)
       !! The context-owned node record, stopping on stale access.
       class(array_type), intent(in) :: this
       type(array_type), pointer :: node
     end function live

     module subroutine set_array(this, input, set_id)
       class(array_type), intent(inout) :: this
       real(real32), dimension(..), intent(in) :: input
       logical, intent(in), optional :: set_id
     end subroutine set_array

     module subroutine extract_array(this, output)
       class(array_type), intent(in) :: this
       real(real32), dimension(..), allocatable, intent(out) :: output
     end subroutine extract_array
  end interface

  interface
     module function grad_forward(this, variable) result(output)
       class(array_type), intent(inout) :: this
       type(array_type), intent(in) :: variable
       type(array_type), pointer :: output
     end function grad_forward

     module subroutine grad_reverse(this, reset_graph)
       class(array_type), intent(inout) :: this
       logical, intent(in), optional :: reset_graph
     end subroutine grad_reverse
  end interface


  interface
     module subroutine reset_graph(this)
       !! Releases the gradient side of this array's graph
       class(array_type), intent(inout) :: this
     end subroutine reset_graph

     module subroutine nullify_graph(this, ignore_ownership)
       !! Releases the graph structure rooted at this array
       class(array_type), intent(inout), target :: this
       logical, intent(in), optional :: ignore_ownership
     end subroutine nullify_graph

     module subroutine zero_grad(this)
       !! Zeroes this array's gradient
       class(array_type), intent(inout) :: this
     end subroutine zero_grad

     recursive module subroutine zero_all_grads(this)
       !! Zeroes the gradients of this array's graph
       class(array_type), intent(inout) :: this
     end subroutine zero_all_grads
  end interface

  interface
     module function get_ptr_from_id(this, id) result(ptr)
       class(array_type), intent(in), target :: this
       integer, intent(in) :: id
       type(array_type), pointer :: ptr
     end function get_ptr_from_id
  end interface

  interface
     module function duplicate_graph(this) result(output_ptr)
       class(array_type), intent(inout) :: this
       type(array_type), pointer :: output_ptr
     end function duplicate_graph
  end interface

  interface
     module subroutine detach(this)
       !! Detaches this array from the computation graph
       class(array_type), intent(inout) :: this
     end subroutine detach
  end interface

  interface
     module subroutine set_requires_grad(this, requires_grad)
       class(array_type), intent(inout) :: this
       logical, intent(in) :: requires_grad
     end subroutine set_requires_grad

     module subroutine set_shape(this, array_shape)
       class(array_type), intent(inout) :: this
       integer, dimension(:), intent(in) :: array_shape
     end subroutine set_shape

     module subroutine set_direction(this, direction)
       class(array_type), intent(inout) :: this
       real(real32), dimension(:), intent(in) :: direction
     end subroutine set_direction
  end interface

  interface
     module function get_partial(this, upstream_grad) result(output)
       class(array_type), intent(inout) :: this
       type(array_type), intent(in) :: upstream_grad
       type(array_type) :: output
     end function get_partial

#ifdef USE_BLAS
     module subroutine get_partial_val(this, upstream_grad, output)
#else
     pure module subroutine get_partial_val(this, upstream_grad, output)
#endif
       class(array_type), intent(in) :: this
       real(real32), dimension(:,:), intent(in) :: upstream_grad
       real(real32), dimension(:,:), intent(out) :: output
     end subroutine get_partial_val

#ifdef USE_BLAS
     module subroutine get_partial_val_sum(this, upstream_grad, output)
#else
     pure module subroutine get_partial_val_sum(this, upstream_grad, output)
#endif
       !! Sum-reduced partial derivative: output = sum(partial, dim=2)
       class(array_type), intent(in) :: this
       real(real32), dimension(:,:), intent(in) :: upstream_grad
       real(real32), dimension(:), intent(out) :: output
     end subroutine get_partial_val_sum
  end interface

  interface
     module subroutine print_graph(this, print_location)
       class(array_type), intent(in) :: this
       logical, intent(in), optional :: print_location
     end subroutine print_graph
  end interface
!-------------------------------------------------------------------------------


!-------------------------------------------------------------------------------
! Pointer wrapper type
!-------------------------------------------------------------------------------
  type :: array_ptr
     type(array_type), pointer :: p => null()
  end type array_ptr
!-------------------------------------------------------------------------------


!-------------------------------------------------------------------------------
! Reduction interfaces that are directly overloaded into array_type
!-----------------------------------------------------------------------------
  interface mean
     module function mean_array(a, dim) result(c)
       class(array_type), intent(in), target :: a
       integer, intent(in) :: dim
       type(array_type), pointer :: c
     end function mean_array

     module function mean_all_array(a) result(c)
       class(array_type), intent(in), target :: a
       type(array_type), pointer :: c
     end function mean_all_array
  end interface


  interface sum
     module function sum_array(a, dim) result(c)
       class(array_type), intent(in), target :: a
       integer, intent(in) :: dim
       type(array_type), pointer :: c
     end function sum_array

     module function sum_all_array(a) result(c)
       class(array_type), intent(in), target :: a
       type(array_type), pointer :: c
     end function sum_all_array

     module function sum_and_pad_array(a, dim, new_dim_index, new_dim_size) &
          result(c)
       class(array_type), intent(in), target :: a
       integer, intent(in) :: dim
       integer, intent(in) :: new_dim_index
       integer, intent(in) :: new_dim_size
       type(array_type), pointer :: c
     end function sum_and_pad_array
  end interface


  interface spread
     module function spread_array(source, dim, index, ncopies) result(c)
       class(array_type), intent(in), target :: source
       integer, intent(in) :: dim
       integer, intent(in) :: index
       integer, intent(in) :: ncopies
       type(array_type), pointer :: c
     end function spread_array
  end interface


  interface unspread
     module function unspread_array(source, index, dim, new_size) result(c)
       class(array_type), intent(in), target :: source
       integer, intent(in) :: index
       integer, intent(in) :: new_size, dim
       type(array_type), pointer :: c
     end function unspread_array
  end interface


  ! Arithmetic operator interfaces that are directly overloaded into array_type
  !-----------------------------------------------------------------------------
  interface
     module function add_arrays(a, b) result(c)
       class(array_type), intent(in), target :: a, b
       type(array_type), pointer :: c
     end function add_arrays

     module function add_real2d(a, b) result(c)
       class(array_type), intent(in), target :: a
       real(real32), dimension(:,:), intent(in) :: b
       type(array_type), pointer :: c
     end function add_real2d

     module function add_real1d(a, b) result(c)
       class(array_type), intent(in), target :: a
       real(real32), dimension(:), intent(in) :: b
       type(array_type), pointer :: c
     end function add_real1d

     module function add_scalar(a, b) result(c)
       class(array_type), intent(in), target :: a
       real(real32), intent(in) :: b
       type(array_type), pointer :: c
     end function add_scalar

     module function subtract_arrays(a, b) result(c)
       class(array_type), intent(in), target :: a, b
       type(array_type), pointer :: c
     end function subtract_arrays

     module function subtract_real2d(a, b) result(c)
       class(array_type), intent(in), target :: a
       real(real32), dimension(:,:), intent(in) :: b
       type(array_type), pointer :: c
     end function subtract_real2d

     module function subtract_real1d(a, b) result(c)
       class(array_type), intent(in), target :: a
       real(real32), dimension(:), intent(in) :: b
       type(array_type), pointer :: c
     end function subtract_real1d

     module function subtract_scalar(a, b) result(c)
       class(array_type), intent(in), target :: a
       real(real32), intent(in) :: b
       type(array_type), pointer :: c
     end function subtract_scalar

     module function negate_array(a) result(c)
       class(array_type), intent(in), target :: a
       type(array_type), pointer :: c
     end function negate_array

     module function multiply_arrays(a, b) result(c)
       class(array_type), intent(in), target :: a, b
       type(array_type), pointer :: c
     end function multiply_arrays

     module function multiply_real2d(a, b) result(c)
       class(array_type), intent(in), target :: a
       real(real32), dimension(:,:), intent(in) :: b
       type(array_type), pointer :: c
     end function multiply_real2d

     module function multiply_scalar(a, scalar) result(c)
       class(array_type), intent(in), target :: a
       real(real32), intent(in) :: scalar
       type(array_type), pointer :: c
     end function multiply_scalar

     module function multiply_logical(a, b) result(c)
       class(array_type), intent(in), target :: a
       logical, dimension(:,:), intent(in) :: b
       type(array_type), pointer :: c
     end function multiply_logical

     module function divide_arrays(a, b) result(c)
       class(array_type), intent(in), target :: a, b
       type(array_type), pointer :: c
     end function divide_arrays

     module function divide_scalar(a, scalar) result(c)
       class(array_type), intent(in), target :: a
       real(real32), intent(in) :: scalar
       type(array_type), pointer :: c
     end function divide_scalar

     module function divide_real1d(a, b) result(c)
       class(array_type), intent(in), target :: a
       real(real32), dimension(:), intent(in) :: b
       type(array_type), pointer :: c
     end function divide_real1d

     module function power_arrays(a, b) result(c)
       class(array_type), intent(in), target :: a, b
       type(array_type), pointer :: c
     end function power_arrays

     module function power_real_scalar(a, scalar) result(c)
       class(array_type), intent(in), target :: a
       real(real32), intent(in) :: scalar
       type(array_type), pointer :: c
     end function power_real_scalar

     module function power_int_scalar(a, scalar) result(c)
       class(array_type), intent(in), target :: a
       integer, intent(in) :: scalar
       type(array_type), pointer :: c
     end function power_int_scalar
  end interface


  ! Arithmetic operator interfaces with a plain array or scalar on the left
  interface operator(+)
     module function real2d_add(a, b) result(c)
       real(real32), dimension(:,:), intent(in) :: a
       class(array_type), intent(in), target :: b
       type(array_type), pointer :: c
     end function real2d_add

     module function real1d_add(a, b) result(c)
       real(real32), dimension(:), intent(in) :: a
       class(array_type), intent(in), target :: b
       type(array_type), pointer :: c
     end function real1d_add

     module function scalar_add(a, b) result(c)
       real(real32), intent(in) :: a
       class(array_type), intent(in), target :: b
       type(array_type), pointer :: c
     end function scalar_add
  end interface


  interface operator(-)
     module function scalar_subtract(a, b) result(c)
       real(real32), intent(in) :: a
       class(array_type), intent(in), target :: b
       type(array_type), pointer :: c
     end function scalar_subtract
  end interface


  interface operator(*)
     module function real2d_multiply(a, b) result(c)
       real(real32), dimension(:,:), intent(in) :: a
       class(array_type), intent(in), target :: b
       type(array_type), pointer :: c
     end function real2d_multiply

     module function scalar_multiply(scalar, a) result(c)
       real(real32), intent(in) :: scalar
       class(array_type), intent(in), target :: a
       type(array_type), pointer :: c
     end function scalar_multiply
  end interface


  interface operator(/)
     module function scalar_divide(scalar, a) result(c)
       real(real32), intent(in) :: scalar
       class(array_type), intent(in), target :: a
       type(array_type), pointer :: c
     end function scalar_divide
  end interface


  interface operator(**)
     module function scalar_power(scalar, a) result(c)
       real(real32), intent(in) :: scalar
       class(array_type), intent(in), target :: a
       type(array_type), pointer :: c
     end function scalar_power

     module function int_scalar_power(scalar, a) result(c)
       integer, intent(in) :: scalar
       class(array_type), intent(in), target :: a
       type(array_type), pointer :: c
     end function int_scalar_power
  end interface


  interface exp
     module function exp_array(a) result(c)
       class(array_type), intent(in), target :: a
       type(array_type), pointer :: c
     end function exp_array
  end interface

  interface log
     module function log_array(a) result(c)
       class(array_type), intent(in), target :: a
       type(array_type), pointer :: c
     end function log_array
  end interface
!-------------------------------------------------------------------------------

  !-----------------------------------------------------------------------------
  ! Context ownership procedures (module procedures of this module).
  !-----------------------------------------------------------------------------
  interface
     module function context_state_of(ctx) result(state)
       type(array_context_type), intent(in) :: ctx
       type(context_state), pointer :: state
     end function context_state_of

     module function allocate_node(ctx) result(node)
       type(array_context_type), intent(in) :: ctx
       type(array_type), pointer :: node
     end function allocate_node

     module subroutine take_value(ctx, store, array_shape)
       type(array_context_type), intent(in) :: ctx
       real(real32), dimension(:,:), allocatable, intent(out) :: store
       integer, dimension(2), intent(in) :: array_shape
     end subroutine take_value

     module subroutine give_value(ctx, store)
       type(array_context_type), intent(in) :: ctx
       real(real32), dimension(:,:), allocatable, intent(inout) :: store
     end subroutine give_value

     module function allocate_value_storage(ctx, rank, array_shape) result(node)
       type(array_context_type), intent(in) :: ctx
       integer, intent(in) :: rank
       integer, dimension(:), intent(in) :: array_shape
       type(array_type), pointer :: node
     end function allocate_value_storage

     module function find_node(ctx, id) result(node)
       type(array_context_type), intent(in) :: ctx
       integer, intent(in) :: id
       type(array_type), pointer :: node
     end function find_node

     module subroutine release_node(ctx, node)
       type(array_context_type), intent(in) :: ctx
       type(array_type), intent(inout) :: node
     end subroutine release_node

     module subroutine reset_context(ctx)
       type(array_context_type), intent(in) :: ctx
     end subroutine reset_context

     module subroutine checkpoint_context(ctx)
       type(array_context_type), intent(in) :: ctx
     end subroutine checkpoint_context

     module subroutine rollback_context(ctx)
       type(array_context_type), intent(in) :: ctx
     end subroutine rollback_context

     module subroutine sweep_region(ctx, keep_tag)
       type(array_context_type), intent(in) :: ctx
       integer, intent(in) :: keep_tag
     end subroutine sweep_region

     module function acquire_scratch(ctx, array_shape) result(buffer)
       type(array_context_type), intent(in) :: ctx
       integer, dimension(2), intent(in) :: array_shape
       real(real32), dimension(:, :), pointer :: buffer
     end function acquire_scratch

     module subroutine release_scratch(ctx, buffer)
       type(array_context_type), intent(in) :: ctx
       real(real32), dimension(:, :), pointer :: buffer
     end subroutine release_scratch

     module subroutine set_default_context(ctx)
       type(array_context_type), intent(in) :: ctx
     end subroutine set_default_context

     module subroutine reset_default_context()
     end subroutine reset_default_context


     module function context_of(this) result(ctx)
       class(array_type), intent(in) :: this
       type(array_context_type) :: ctx
     end function context_of

     module subroutine bind_handle(this, node)
       class(array_type), intent(inout) :: this
       type(array_type), intent(in), target :: node
     end subroutine bind_handle

     module subroutine clone_record(dst, src)
       type(array_type), intent(inout), target :: dst
       type(array_type), intent(in), target :: src
     end subroutine clone_record

     module function clone_handle(source) result(copy)
       class(array_type), intent(in) :: source
       type(array_type), pointer :: copy
     end function clone_handle

     module function attach_gradient(this) result(grad)
       class(array_type), intent(in), target :: this
       type(array_type), pointer :: grad
     end function attach_gradient

     module function constant_node(template, values) result(node)
       class(array_type), intent(in) :: template
       real(real32), dimension(:,:), intent(in) :: values
       type(array_type), pointer :: node
     end function constant_node
  end interface

  type(array_context_type), target :: default_context = array_context_type()
  !! The context every operation uses unless another one is bound to a handle.

end module diffstruc__types
