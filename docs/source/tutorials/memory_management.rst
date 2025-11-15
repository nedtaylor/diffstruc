Memory Management
=================

Proper memory management is crucial when using diffstruc to avoid memory leaks and ensure efficient performance.


Each ``array_type`` variable manages:

1. **Data array** (``val``) - The actual values
2. **Gradient** (``grad``) - Pointer to gradient array
3. **Left operand** (``left_operand``) - Pointer to left operand array
4. **Right operand** (``right_operand``) - Pointer to right operand array
5. **Computation graph** - Links to operands for differentiation
6. **Ownership flags** - Tracks what should be deallocated

Depending on what operations are performed, multiple intermediate arrays and pointers may be created.
These will be directly accessible via the ``array_type`` structure through the ``left_operand`` and ``right_operand`` pointers.

It is important to manage the lifecycle of these components carefully.

1. **Allocation**: Create and allocate memory
2. **Computation**: Build computation graph
3. **Differentiation**: Compute gradients
4. **Cleanup**: Deallocate memory and nullify pointers

Essential Cleanup Operations
-----------------------------

Always perform these two operations when done with computed results:

.. code-block:: fortran

   ! 1. Nullify computation graph
   call f%nullify_graph()

   ! 2. Deallocate data arrays
   call f%deallocate()

   ! 3. Deallocate the pointer itself
   deallocate(f)


The nullify_graph Procedure
----------------------------

``nullify_graph()`` traverses the computation graph and:

* Nullifies all pointer connections
* Deallocates temporary intermediate results
* Prevents memory leaks from circular references

Call ``nullify_graph()`` on:

* **All** pointer results from operations
* Both function values and gradient results
* Before deallocating the variable

.. code-block:: fortran

   f => x**2 + y**2
   df_dx => f%grad_forward(x)

   ! Clean up BOTH
   call f%nullify_graph()      ! Clean function graph
   call df_dx%nullify_graph()  ! Clean gradient graph

Sometimes it is necessary to use the ``ignore_ownership`` argument of ``nullify_graph()`` to prevent repeated deallocation attempts of shared nodes.
Setting ``ignore_ownership = .false.`` skips deallocation of nodes not owned by the current variable.
By default, ``ignore_ownership = .true.`` (unless the variable has been generated from ``grad_forward``).
As such, it might sometimes be useful to do the following:

.. code-block:: fortran

   call f%nullify_graph(ignore_ownership = .true.)
   call df_dx%nullify_graph(ignore_ownership = .false.)

where the first call cleans up all nodes owned by ``f``, and the second call only cleans up nodes owned by ``df_dx``.

Common Mistake
~~~~~~~~~~~~~~

.. code-block:: fortran

   ! WRONG - Only cleaning one
   call f%nullify_graph()
   call f%deallocate()
   deallocate(f)
   ! df_dx is leaked!

   ! CORRECT - Clean both
   call f%nullify_graph()
   call df_dx%nullify_graph()
   deallocate(f, df_dx)


The is_temporary Flag
---------------------

* ``is_temporary = .false.``: Variable you created explicitly
* ``is_temporary = .true.``: Intermediate computation result (automatic)

.. code-block:: fortran

   x%is_temporary = .false.   ! Explicit variable
   y%is_temporary = .false.   ! Explicit variable

   f => x + y                 ! f%is_temporary = .true. (automatic)

Set ``is_temporary = .false.`` for:

* Input variables
* Parameters (weights, biases)
* Any variable you explicitly create

.. code-block:: fortran

   call x%allocate([1, 1, 1], source=1.0)
   call x%set_requires_grad(.true.)
   x%is_temporary = .false.  ! Set this!

Memory Leak Patterns
---------------------

Pattern 1: Forgetting Cleanup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Forgetting to clean up inside loops or repeated computations can lead to significant memory leaks over time.

.. code-block:: fortran

   ! BAD - Memory leak!
   do i = 1, 1000
     f => x**2
     ! No cleanup - leaks every iteration
   end do

   ! GOOD
   do i = 1, 1000
     f => x**2
     call f%nullify_graph()
     deallocate(f)
   end do

Pattern 2: Forgetting Gradient Cleanup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because `f` is part of the `df_dx` computation graph, it is likely that only cleaning up `df_dx` is necessary.
However, to be safe and avoid leaks in complex graphs, always clean up both.

.. code-block:: fortran

   ! BAD - Gradient result leaked!
   f => x**2
   df_dx => f%grad_forward(x)
   call f%nullify_graph()
   call f%deallocate()
   deallocate(f)
   ! df_dx still allocated!

   ! GOOD
   call f%nullify_graph()
   call df_dx%nullify_graph()
   deallocate(f, df_dx)

Pattern 3: Circular References
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Circular references are common in computation graphs.
These can lead to memory not being freed if not handled correctly.
The ``nullify_graph()`` procedure is designed to handle these cases safely.

.. code-block:: fortran

   ! Computation graphs can have cycles
   ! This is why nullify_graph() is essential

   f => x * y + x  ! x appears twice - creates shared nodes

   ! nullify_graph() handles cycles correctly
   call f%nullify_graph()

Efficient Memory Usage
----------------------

Reusing Variables
~~~~~~~~~~~~~~~~~

When possible, reuse allocated variables:

.. code-block:: fortran

   ! Allocate once
   call workspace%allocate([100, 100, 1])

   do iter = 1, 1000
     ! Reuse workspace
     workspace%val = compute_something()
     ! No allocation/deallocation overhead
   end do

   ! Deallocate once at end
   call workspace%deallocate()

In-Place Operations
~~~~~~~~~~~~~~~~~~~

For non-differentiable updates, modify in-place.
Also, try to prioritise pointer assignment over normal assignment.

.. code-block:: fortran

   ! Instead of creating new arrays
   x = x + delta + y  ! In-place update

   ! Rather than
   temp = x + delta
   x = temp + y

Global Variables
----------------

diffstruc has two global variables to help manage memory:

* ``diffstruc__max_recursion_depth``: Maximum recursion depth for graph traversal (default 1000)
* ``diffstruc__init_map_cap``: Default capacity for internal hash maps (default 32)

These can be adjusted by importing them from diffstruc and modifying them in your program before performing large computations.


Debugging Memory Issues
-----------------------

Checking for Leaks
~~~~~~~~~~~~~~~~~~

Use valgrind (Linux):

.. code-block:: bash

   valgrind --leak-check=full --show-leak-kinds=all ./your_program

For macOS, use Instruments or Activity Monitor to track memory usage.
However, to do so, you need to use the ``codesign`` to sign your executable for profiling tools to work correctly.
