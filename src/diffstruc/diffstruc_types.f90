module diffstruc__types
  !! This module contains the automatic differentiation data structures.
  !!
  !! The module provides a derived type implementation for arrays that
  !! supports automatic differentiation, including forward and reverse mode.
  use coreutils, only: real32
  implicit none

  private

  public :: array_type, get_partial

  public :: operator(+), operator(-), operator(*), operator(/), operator(**)
  public :: sum, mean, spread, unspread, exp, log


!-------------------------------------------------------------------------------
! Automatic differentiation derived type
!-------------------------------------------------------------------------------
  type :: array_type
     !! Abstract type for array operations
     integer :: id = -1
     integer :: rank
     !! Rank of the array
     integer, dimension(:), allocatable :: shape
     !! Shape of the array
     integer :: size
     !! Size of the array
     logical :: is_sample_dependent = .true.
     !! Boolean whether array is sample-dependent
     logical :: is_scalar = .false.
     !! Boolean whether array is contains a scalar value
     logical :: is_forward = .false.
     !! Boolean whether operation is forward-mode
     logical :: allocated = .false.
     !! Logical flag for array allocation
     real(real32), dimension(:,:), allocatable :: val
     !! Array values in rank 2 (sample, batch)
     integer, dimension(:), allocatable :: indices ! store_1
     !! Indices for gradient accumulation
     integer, dimension(:,:), allocatable :: adj_ja ! store_2
     !! Sparse adjacency matrix for graph structure
     logical, dimension(:,:), allocatable :: mask
     !! Mask for operation
     logical :: requires_grad = .false.
     !! Flag indicating if gradients should be computed
     logical :: is_leaf = .true.
     !! Flag indicating if this is a leaf node (parameter)
     type(array_type), pointer :: grad => null()
     !! Gradient array (same type as value)
     type(array_type), pointer :: left_operand => null()
     !! Left operand for backward pass
     type(array_type), pointer :: right_operand => null()
     !! Right operand for backward pass
     character(len=32) :: operation = 'none'
     logical :: owns_gradient = .true.
     !! Flag indicating if this array owns its gradient memory
     logical :: fix_pointer = .false.

     real(real32), dimension(:), allocatable :: direction

     procedure(get_partial), pass(this), pointer :: get_partial_left => null()
     procedure(get_partial), pass(this), pointer :: get_partial_right => null()

   contains
     procedure, pass(this) :: allocate => allocate_array
     !! Abstract procedure for allocating array
     procedure, pass(this) :: deallocate => deallocate_array
     !! Abstract procedure for deallocating array
     procedure, pass(this) :: flatten => flatten_array
     !! Procedure for flattening array
     procedure :: assign => assign_array
     generic, public :: assignment(=) => assign
     !! Overloaded assignment operator
     procedure, pass(this) :: set => set_array
     !! Procedure for setting array

     procedure, pass(this) :: set_direction
     procedure, pass(this) :: grad_reverse
     !! Reverse-mode: accumulate gradients wrt all inputs
     procedure, pass(this) :: grad_forward
     !! Forward-mode: return derivative wrt variable pointer

     !! Backward pass for gradient computation
     procedure, pass(this) :: zero_grad
     procedure, pass(this) :: zero_all_grads
     !! Zero the gradients
     procedure, pass(this) :: reset_graph
     procedure, pass(this) :: duplicate_graph
     procedure, pass(this) :: nullify_graph
     !   procedure, pass(this) :: duplicate_graph_ptrs
     procedure, pass(this) :: get_ptr_from_id
     procedure, pass(this) :: detach
     !! Detach from computation graph
     procedure, pass(this) :: set_requires_grad
     !! Set requires_grad flag
     procedure :: create_result => create_result_array
     !! Helper to safely create result arrays

     procedure, pass(this) :: print_graph

     ! final :: finalise_array
     ! !! Finaliser for array type
  end type array_type


  ! Interfaces
  !-----------------------------------------------------------------------------
  interface
     module subroutine allocate_array(this, array_shape, source)
       class(array_type), intent(inout), target :: this
       integer, dimension(:), intent(in), optional :: array_shape
       class(*), dimension(..), intent(in), optional :: source
     end subroutine allocate_array

     module recursive subroutine deallocate_array(this, keep_shape)
       class(array_type), intent(inout) :: this
       logical, intent(in), optional :: keep_shape
     end subroutine deallocate_array

     module recursive subroutine finalise_array(this)
       type(array_type), intent(inout) :: this
     end subroutine finalise_array
  end interface

  interface
     pure module function flatten_array(this) result(output)
       class(array_type), intent(in) :: this
       real(real32), dimension(this%size) :: output
     end function flatten_array

     module subroutine assign_array(this, input)
       class(array_type), intent(out), target :: this
       type(array_type), intent(in) :: input
     end subroutine assign_array

     module function create_result_array(this, array_shape) result(result_ptr)
       class(array_type), intent(in) :: this
       integer, dimension(:), intent(in), optional :: array_shape
       type(array_type), pointer :: result_ptr
     end function create_result_array

     pure module subroutine set_array(this, input)
       class(array_type), intent(inout) :: this
       real(real32), dimension(..), intent(in) :: input
     end subroutine set_array
  end interface

  interface
     module function grad_forward(this, variable) result(output)
       class(array_type), intent(inout) :: this
       type(array_type), intent(in) :: variable
       type(array_type), pointer :: output
     end function grad_forward

     module subroutine grad_reverse(this, record_graph, reset_graph)
       class(array_type), intent(inout) :: this
       logical, intent(in), optional :: record_graph
       logical, intent(in), optional :: reset_graph
     end subroutine grad_reverse
  end interface


  interface
     module recursive subroutine reset_graph(this)
       !! Reset the gradients of this array
       class(array_type), intent(inout) :: this
     end subroutine reset_graph

     module recursive subroutine nullify_graph(this)
       !! Nullify the gradients of this array
       class(array_type), intent(inout) :: this
     end subroutine nullify_graph

     module subroutine zero_grad(this)
       !! Zero the gradients of this array
       class(array_type), intent(inout) :: this
     end subroutine zero_grad

     module recursive subroutine zero_all_grads(this)
       !! Zero the gradients of this array
       class(array_type), intent(inout) :: this
     end subroutine zero_all_grads
  end interface

  interface
     module recursive function get_ptr_from_id(this, id) result(ptr)
       use iso_c_binding
       class(array_type), intent(in), target :: this
       integer, intent(in) :: id
       type(array_type), pointer :: ptr
     end function get_ptr_from_id
  end interface

  interface
     module subroutine duplicate_graph(this)
       class(array_type), intent(inout) :: this
     end subroutine duplicate_graph
  end interface

  interface
     module subroutine detach(this)
       !! Detach this array from the computation graph
       class(array_type), intent(inout) :: this
     end subroutine detach
  end interface

  interface
     module subroutine set_requires_grad(this, requires_grad)
       class(array_type), intent(inout) :: this
       logical, intent(in) :: requires_grad
     end subroutine set_requires_grad

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
  end interface

  interface
     module subroutine print_graph(this)
       class(array_type), intent(in) :: this
     end subroutine print_graph
  end interface


  ! Operation interfaces
  !-----------------------------------------------------------------------------
  interface mean
     module function mean_array(a, dim) result(c)
       class(array_type), intent(in), target :: a
       integer, intent(in) :: dim
       type(array_type), pointer :: c
     end function mean_array
  end interface


  interface sum
     module function sum_array(a, dim) result(c)
       class(array_type), intent(in), target :: a
       integer, intent(in) :: dim
       type(array_type), pointer :: c
     end function sum_array

     module function sum_array_output_array(a, dim, new_dim_index, new_dim_size) &
          result(c)
       class(array_type), intent(in), target :: a
       integer, intent(in) :: dim
       integer, intent(in) :: new_dim_index
       integer, intent(in) :: new_dim_size
       type(array_type), pointer :: c
     end function sum_array_output_array
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


  interface operator(+)
     module function add_arrays(a, b) result(c)
       class(array_type), intent(in), target :: a, b
       type(array_type), pointer :: c
     end function add_arrays

     module function add_real2d(a, b) result(c)
       class(array_type), intent(in), target :: a
       real(real32), dimension(:,:), intent(in) :: b
       type(array_type), pointer :: c
     end function add_real2d

     module function real2d_add(a, b) result(c)
       real(real32), dimension(:,:), intent(in) :: a
       class(array_type), intent(in), target :: b
       type(array_type), pointer :: c
     end function real2d_add

     module function add_real1d(a, b) result(c)
       class(array_type), intent(in), target :: a
       real(real32), dimension(:), intent(in) :: b
       type(array_type), pointer :: c
     end function add_real1d

     module function real1d_add(a, b) result(c)
       real(real32), dimension(:), intent(in) :: a
       class(array_type), intent(in), target :: b
       type(array_type), pointer :: c
     end function real1d_add

     module function add_scalar(a, b) result(c)
       class(array_type), intent(in), target :: a
       real(real32), intent(in) :: b
       type(array_type), pointer :: c
     end function add_scalar

     module function scalar_add(a, b) result(c)
       real(real32), intent(in) :: a
       class(array_type), intent(in), target :: b
       type(array_type), pointer :: c
     end function scalar_add
  end interface


  interface operator(-)
     module function subtract_arrays(a, b) result(c)
       class(array_type), intent(in), target :: a, b
       type(array_type), pointer :: c
     end function subtract_arrays

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

     module function scalar_subtract(a, b) result(c)
       real(real32), intent(in) :: a
       class(array_type), intent(in), target :: b
       type(array_type), pointer :: c
     end function scalar_subtract

     module function negate_array(a) result(c)
       class(array_type), intent(in), target :: a
       type(array_type), pointer :: c
     end function negate_array
  end interface


  interface operator(*)
     module function multiply_arrays(a, b) result(c)
       class(array_type), intent(in), target :: a, b
       type(array_type), pointer :: c
     end function multiply_arrays

     module function multiply_scalar(a, scalar) result(c)
       class(array_type), intent(in), target :: a
       real(real32), intent(in) :: scalar
       type(array_type), pointer :: c
     end function multiply_scalar

     module function scalar_multiply(scalar, a) result(c)
       real(real32), intent(in) :: scalar
       class(array_type), intent(in), target :: a
       type(array_type), pointer :: c
     end function scalar_multiply

     module function multiply_logical(a, b) result(c)
       class(array_type), intent(in), target :: a
       logical, dimension(:,:), intent(in) :: b
       type(array_type), pointer :: c
     end function multiply_logical
  end interface


  interface operator(/)
     module function divide_arrays(a, b) result(c)
       class(array_type), intent(in), target :: a, b
       type(array_type), pointer :: c
     end function divide_arrays

     module function divide_scalar(a, scalar) result(c)
       class(array_type), intent(in), target :: a
       real(real32), intent(in) :: scalar
       type(array_type), pointer :: c
     end function divide_scalar

     module function scalar_divide(scalar, a) result(c)
       real(real32), intent(in) :: scalar
       class(array_type), intent(in), target :: a
       type(array_type), pointer :: c
     end function scalar_divide

     module function divide_real1d(a, b) result(c)
       class(array_type), intent(in), target :: a
       real(real32), dimension(:), intent(in) :: b
       type(array_type), pointer :: c
     end function divide_real1d
  end interface


  interface operator(**)
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

end module diffstruc__types
