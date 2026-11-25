Porting from the owning implementation
======================================

`array_type` used to own its value storage, gradients and operands, and it had
to be cleaned up by hand. It is now a lightweight handle into a context that
owns all of that memory. Maths code is unchanged; what changes is the lifetime
of results and a few data members.

This page is the checklist. The complete memory model is described in
:doc:`tutorials/memory_management`.

Summary
-------

.. list-table::
   :header-rows: 1

   * - Still valid
     - No longer valid
   * - ``use diffstruc``, ``type(array_type)``, all operators
     - ``deallocate(handle)`` on a pointer result
   * - ``x%allocate(...)``, ``x%val`` read/write, ``x%grad%val``
     - ``allocated(x%val)`` / ``deallocate(x%val)`` (``%val`` is a view)
   * - ``grad_forward``, ``grad_reverse``, ``set_direction``
     - ``owns_left_operand``, ``owns_right_operand``, ``owns_gradient``,
       ``fix_pointer``
   * - ``nullify_graph()``, ``%deallocate()``
     - setting ``is_temporary`` / ``is_sample_dependent`` before ``allocate``
   * - ``sum``, ``mean``, ``exp``, ``log``, trig, ``matmul``, ...
     - deep-copy semantics of assignment into an unbound handle

Lifetime and cleanup
--------------------

The context owns every record, value buffer and gradient. A pointer result is
a view of a context-owned record, so it must never be passed to Fortran's
``deallocate``:

.. code-block:: fortran

   f => x**2 + x
   call f%grad_reverse(reset_graph=.true.)
   print *, x%grad%val(:, 1)

   call f%nullify_graph()   ! releases the expression and its operands
   ! deallocate(f)          ! WRONG: aborts, the record is context-owned

``call f%deallocate()`` still exists. It gives *that node's* storage back to
the context and unbinds the handle; calling it after ``nullify_graph`` is a
harmless no-op. To release everything a context owns (for example between
independent runs), use ``reset_default_context()`` or
``checkpoint_context``/``rollback_context`` for a region.

``nullify_graph`` keeps root variables: nodes marked ``requires_grad`` that
have no operands. That is why ``x`` above is still usable afterwards. Nodes
that are not roots are released, and any handle that still refers to them is
rejected as stale instead of being dereferenced.

Assignment
----------

Assignment to an **unbound** handle binds the handle to the source's record.
Nothing is copied, so both handles view the same values:

.. code-block:: fortran

   type(array_type) :: saved
   saved = x              ! shares x's record
   saved%val(1, 1) = 99   ! x%val(1, 1) is now 99

To copy values, bind the target first. Assignment into an already-bound handle
copies into its existing record:

.. code-block:: fortran

   call copy%allocate(array_shape=shape(x))   ! bind the target
   copy = x                                   ! value copy
   copy%val(1, 1) = 99                        ! x is unchanged

For graph expressions, ``p => expr%duplicate_graph()`` deep-copies the graph
(root variables are shared, so the copy differentiates the same inputs).

Allocation
----------

``array_shape`` follows the storage convention: **the last entry is the number
of samples, the leading entries are the logical shape**. ``%val`` is always
rank 2, ``(elements, samples)``, whatever the logical shape is.

.. code-block:: fortran

   call x%allocate(array_shape=[d, batch])       ! rank 1, d elements, batch samples
   call w%allocate(array_shape=[rows, cols, 1])  ! rank 2, one sample

The flags can be set in the same call, which avoids touching them before the
handle is bound:

.. code-block:: fortran

   call w%allocate(array_shape=[rows, cols, 1], &
        sample_dependent=.false., temporary=.false.)

``source=`` accepts a value of the same total size, so a logical matrix can be
passed directly:

.. code-block:: fortran

   call w%allocate(array_shape=[rows, cols, 1], source=matrix)

``set_shape`` is only needed to re-interpret an existing allocation (for
example after allocating a flattened ``[rows*cols, 1]`` view).

.. warning::

   ``is_sample_dependent`` and ``is_temporary`` are views of flags stored in
   the context record. They are unbound until the handle refers to a record,
   so assigning them *before* ``allocate`` is a null-pointer dereference. Set
   them after ``allocate`` (as the tutorials do) or pass the keyword arguments
   shown above.

Custom operations
-----------------

The shape of a custom operation is unchanged; only the ownership bookkeeping is
removed:

.. code-block:: fortran

   function my_op(a) result(c)
     class(array_type), intent(in), target :: a
     type(array_type), pointer :: c

     c => a%create_result()          ! same shape as a, context-owned
     c%val = cos(a%val)              ! your operation

     c%get_partial_left => my_op_partial
     c%get_partial_left_val => my_op_partial_val
     if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'my_op'
       c%left_operand => a
       ! c%owns_left_operand = ...  <- removed; delete this line
     end if
   end function my_op

Checklist for a custom operation:

* Delete ``owns_left_operand`` / ``owns_right_operand`` / ``owns_gradient`` /
  ``fix_pointer``. The context releases operands; nothing is reference counted.
* Give the operation a name that is **not** a built-in operation name. Reverse
  mode dispatches on the string, and a custom ``'tanh'`` would inherit the
  built-in derivative.
* Reverse mode uses the graph partial ``get_partial_left`` (or
  ``get_partial_right``). Keep the documented signature and finish it with
  ``output%assign_and_deallocate_source(ptr)``.
* ``get_partial_left_val`` / ``_val_sum`` are wired up for the library's
  built-in operations only. For a custom operation name they are ignored and
  the graph partial is used, which is correct but slower. (Defining the value
  partial is harmless and future-proof.)
* Do not keep a pointer to ``c%val`` beyond the life of the graph; use
  ``c%extract(values)`` or assign to a Fortran array to keep the numbers.

Calling code
------------

* Operators, functions, reductions, ``matmul``, forward and reverse mode keep
  their signatures; results are still ``type(array_type), pointer``.
* ``%val`` is a pointer view now: use ``associated(x%val)`` rather than
  ``allocated(x%val)``, and never deallocate it.
* ``nullify_graph(ignore_ownership=...)`` still accepts the argument, but it is
  ignored: ownership is not tracked per handle.
* Gradients are still read as ``x%grad%val``. Their layout follows the
  variable: a rank-1 variable of ``d`` elements with ``n`` samples keeps a
  ``(d, n)`` gradient, and a ``rows x cols`` variable with one sample keeps a
  ``(rows, cols)`` gradient.
* New helpers that replacement code may find useful: ``context_of``,
  ``canonical``, ``live``, ``default_context``, ``reset_default_context``,
  ``set_shape``, and the ``tanhf`` operation (approximate, see the README).

Verification
------------

``test/test_porting.f90`` exercises every item above (keyword allocation,
matrix ``source=``, ``set_shape`` rank, sharing versus copying, cleanup order
and a custom operation with a graph partial).
