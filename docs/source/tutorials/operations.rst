Supported Operations
====================

diffstruc supports a wide range of mathematical operations, all with automatic differentiation capabilities.

Below is a table of the supported operations, along with notes on how to build your own custom operations if needed.

If there are operations you need that are not listed here and would be useful to others, please consider contributing them via a pull request on the `diffstruc GitHub repository <https://github.com/nedtaylor/diffstruc>`_.

Operation Summary Table
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 50 40

   * - Category
     - Operations
     - Notes
   * - Arithmetic
     - ``+``, ``-``, ``*``, ``/``, ``**``
     - Supports scalars and arrays
   * - Trigonometric
     - ``sin``, ``cos``, ``tan``
     - Input in radians, same as Fortran intrinsic
   * - Hyperbolic
     - ``tanh``
     - Common in neural networks
   * - Exponential
     - ``exp``, ``log``, ``log10``
     - log is natural logarithm
   * - Linear Algebra
     - ``dot_product``, ``outer_product``, ``matmul``, ``transpose``
     - Matrix operations
   * - Reduction
     - ``sum``, ``mean``, ``unspread``, ``max``, ``maxval``
     - Collapse dimensions
   * - Comparison
     - ``.lt.``, ``.gt.``, ``merge``
     - Element-wise comparisons
   * - Broadcast
     - ``spread``, ``concat``, ``slice_left``, ``slice_right``, ``ltrim``, ``rtrim``, ``.index.``, ``reverse_index``, ``pack``, ``unpack``, ``reshape``
     - Broadcasting and indexing
   * - Other
     - ``sign``, ``sqrt``, ``sigmoid``, ``gaussian``
     - Element-wise operations


Custom Operations
-----------------

If you need an operation not provided, you can implement it providing a custom Fortran function and defining the partial derivative procedures.
The custom function can take any form you need.
However, the partial derivative functions must conform to the interface expected by diffstruc.

The interface for the ``get_partial_left`` and ``get_partial_right`` functions are:

.. code-block:: fortran

  interface
     module function get_partial(this, upstream_grad) result(output)
       class(array_type), intent(inout) :: this
       type(array_type), intent(in) :: upstream_grad
       type(array_type) :: output
     end function get_partial
  end interface

The interface for the ``get_partial_left_val`` and ``get_partial_right_val`` functions are:

.. code-block:: fortran

  interface
     pure subroutine get_partial_val(this, upstream_grad, output)
       class(array_type), intent(in) :: this
       real(real32), dimension(:,:), intent(in) :: upstream_grad
       real(real32), dimension(:,:), intent(out) :: output
     end subroutine get_partial_val
  end interface

The former set are used for forward mode differentiation, while the latter are used exclusively for reverse mode differentiation.
A future release may use ``get_partial`` for reverse mode also (if a computational graph needs to be built during reverse mode traversal), but for now it is only used in forward mode.
The reason that some operations in diffstruc still define a ``get_partial`` function is a legacy reason in case this reverse mode graph building needs to be reintroduced in the future.

Depending on the operation, you might only need to define one of these (priority is given to the left operand if only one is defined).

A simple example of a custom operation that takes in one operand and computes the cosine is shown below.
Focus on the parts marked with comments.

.. code-block:: fortran

  module custom_operations
    use diffstruc
    implicit none

    interface operation_name
      module procedure my_custom_op
    end interface

  contains

    function my_custom_op(a) result(c)
      !! This is a custom operation example, it can take any form you need.
      implicit none
      class(array_type), intent(in), target :: a
      type(array_type), pointer :: c

      !! Allocates result array to the same shape as input
      c => a%create_result()
      ! !! An alternative is to provide it with a different shape, do not forget the batch_size final dimension
      ! c => a%create_result([desired_shape])

      !!-----------------------------------------------
      !! YOUR CUSTOM OPERATION
      !!-----------------------------------------------
      c%val = cos(a%val)
      !!-----------------------------------------------

      c%get_partial_left => get_partial_left_custom_op
      c%get_partial_left_val => get_partial_left_custom_op_val
      if(a%requires_grad) then
        c%requires_grad = .true.
        c%is_forward = a%is_forward
        c%operation = 'cos'
        c%left_operand => a
        c%owns_left_operand = a%is_temporary
      end if
    end function my_custom_op

    function get_partial_left_custom_op(this, upstream_grad) result(output)
      !! Partial derivative of custom operation w.r.t. left operand
      !! This has to conform to the interface expected by diffstruc
      implicit none
      class(array_type), intent(inout) :: this
      type(array_type), intent(in) :: upstream_grad
      type(array_type) :: output

      logical :: left_is_temporary_local
      type(array_type), pointer :: ptr

      !! Save and temporarily disable the temporary status of the left operand
      left_is_temporary_local = this%left_operand%is_temporary
      this%left_operand%is_temporary = .false.

      !!-----------------------------------------------
      !! YOUR CUSTOM PARTIAL DERIVATIVE
      !!-----------------------------------------------
      ptr => -upstream_grad * sin( this%left_operand )
      !!-----------------------------------------------

      !! Restore the temporary status of the left operand
      this%left_operand%is_temporary = left_is_temporary_local

      call output%assign_and_deallocate_source(ptr)
    end function get_partial_left_custom_op


    pure subroutine get_partial_left_custom_op_val(this, upstream_grad, output)
      implicit none
      class(array_type), intent(in) :: this
      real(real32), dimension(:,:), intent(in) :: upstream_grad
      real(real32), dimension(:,:), intent(out) :: output

      !!-----------------------------------------------
      !! YOUR CUSTOM PARTIAL DERIVATIVE
      !!-----------------------------------------------
      output = -upstream_grad * sin( this%left_operand%val )
      !!-----------------------------------------------
    end subroutine get_partial_left_custom_op_val

  end module custom_operations

For one with two operands, you would similarly define ``get_partial_right_custom_op``, ``get_partial_right_custom_op_val``, and associate the ``right_operand`` pointer of the result.
For how this works, see the built-in matmul operation in the source code (:git:`diffstruc_operations_linalg_sub.f90<src/diffstruc/diffstruc_operations_linalg_sub.f90>`)
