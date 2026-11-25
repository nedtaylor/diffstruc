Memory Management
=================

Every record, value buffer, gradient and scratch buffer belongs to a context
(``array_context_type``). An ``array_type`` is a non-owning handle: it carries
an id plus views into that memory. Copying a handle copies the views, never the
storage, and no handle has a finaliser or a reference count.

This means the usual Fortran cleanup rule is inverted: you do not free values,
and you must not call ``deallocate`` on a pointer result. You release *regions*
of the context, either explicitly (``nullify_graph``/``deallocate``) or in bulk
(``reset``).

What the context owns
---------------------

* Node records (including ``shape``, ``indices``, masks and directions)
* Value storage and its ``%val`` view
* Gradient accumulators (``%grad``)
* Scratch buffers used by the reverse-mode kernels
* The links between operation nodes (``left_operand``/``right_operand``)

``x%val``, ``x%grad%val``, ``x%left_operand`` and friends are all views of
context-owned memory. Reading and writing them is unchanged.

Lifecycle
---------

1. **Allocate** an input with ``x%allocate(...)`` and mark it
   ``set_requires_grad(.true.)`` if it is a differentiation variable.
2. **Compute**: operators/functions create expression nodes in the context.
3. **Differentiate**: ``grad_forward``/``grad_reverse`` fill ``x%grad``.
4. **Release**: ``call expr%nullify_graph()`` for an expression, or
   ``reset_default_context()`` to release everything.

.. code-block:: fortran

   call x%allocate(array_shape=[3, 1], temporary=.false.)
   x%val(:, 1) = [1.0_real32, 2.0_real32, 3.0_real32]
   call x%set_requires_grad(.true.)

   f => x**2 + x
   call f%grad_reverse(reset_graph=.true.)
   print *, x%grad%val(:, 1)     ! 3.0, 5.0, 7.0

   call f%nullify_graph()        ! expression and its operands released
   call x%deallocate()           ! optional: give the input's storage back

The nullify_graph Procedure
---------------------------

``nullify_graph()`` walks the graph rooted at a handle and:

* releases every operand record that is not a root variable, plus the gradients
  of transient nodes,
* releases the root itself when it is a temporary (an expression), or keeps it
  when it is a variable,
* unbinds the handle if its own record was released.

Root variables are nodes with ``requires_grad = .true.`` and no operands, i.e.
the arrays the caller allocated. That is why ``x`` above survives and can be
reused for the next expression.

The optional ``ignore_ownership`` argument is accepted for source
compatibility. It has no effect: ownership is not tracked per handle.

The deallocate Procedure
------------------------

``call h%deallocate()`` returns *that node's* storage to the context and
unbinds the handle. Other handles referring to the same node become stale and
are rejected on access. Calling it after ``nullify_graph`` is a harmless
no-op. Use it for inputs you are done with; you do not need it for expressions.

Releasing whole regions
~~~~~~~~~~~~~~~~~~~~~~~

* ``reset_default_context()`` releases every record the global context owns.
* ``checkpoint_context(ctx)`` marks a point and ``rollback_context(ctx)``
  releases every node created after it, without invalidating earlier records.
  This is what the reverse-mode scratch region uses internally.
* ``set_default_context(other)`` switches the process-wide context.

The is_temporary Flag
---------------------

* ``is_temporary = .false.``: a variable the caller created explicitly.
* ``is_temporary = .true.``: an intermediate expression result (the default for
  every operation result).

``nullify_graph`` uses the flag to decide whether the root of the graph should
be released. Set it, or pass the keyword to ``allocate``:

.. code-block:: fortran

   call x%allocate(array_shape=[1, 1, 1], source=1.0, temporary=.false.)
   ! or, after allocation:
   x%is_temporary = .false.

.. warning::

   ``is_temporary`` and ``is_sample_dependent`` are views of flags stored in
   the context record. They have no target until the handle is bound, so set
   them after ``allocate`` (or use the ``temporary=``/``sample_dependent=``
   keywords).

Assignment and copies
---------------------

Assignment into an **unbound** handle binds it to the source's record, sharing
the values. Assignment into a handle that is **already bound** copies the
values into its record:

.. code-block:: fortran

   type(array_type) :: shared, copied

   shared = x                              ! shares x's record
   call copied%allocate(array_shape=shape(x))
   copied = x                              ! value copy

For expressions use ``p => expr%duplicate_graph()``, which deep-copies the
graph while sharing the root variables.

Common mistakes
---------------

.. code-block:: fortran

   ! WRONG: pointer results are context-owned
   f => x**2
   call f%nullify_graph()
   deallocate(f)                  ! aborts

   ! WRONG: %val is a view, not an allocation
   deallocate(x%val)

   ! WRONG: the flag has no target before binding
   type(array_type) :: y
   y%is_temporary = .false.       ! null-pointer dereference
   call y%allocate(...)

   ! RIGHT
   f => x**2
   call f%nullify_graph()

Memory growth in loops
----------------------

Expression records are released by ``nullify_graph``, so repeated graph builds
reuse the context's value buffers and stay in bounded memory:

.. code-block:: fortran

   do i = 1, 1000
     f => x**2 + x
     call f%grad_reverse(reset_graph=.true.)
     call f%nullify_graph()       ! keeps memory flat
   end do

Records are retired rather than recycled, so the registry itself grows with the
total number of nodes a context has ever issued (a few hundred bytes per node).
``reset_default_context()`` after a batch of work returns all of it. The
``test_memory`` program checks that a long loop does not grow the *value*
buffers; the number of iterations can be set with the
``DIFFSTRUC_MEMORY_ITERATIONS`` environment variable.

Efficient usage
---------------

Reuse an allocated input instead of allocating a new one; ``%set`` writes into
the existing record:

.. code-block:: fortran

   call workspace%allocate(array_shape=[100, 100, 1])
   do iter = 1, 1000
     call workspace%set(compute_something())
   end do
   call workspace%deallocate()

Diagnostics
-----------

* ``context_of(handle)`` returns the owning context.
* ``canonical(handle)`` resolves the handle to its record, or null when it is
  unbound/stale.
* ``live(handle)`` is the checked variant used internally: it rejects a stale
  or unbound handle with a diagnostic instead of dereferencing it.

Debugging memory issues
-----------------------

Checking for leaks
~~~~~~~~~~~~~~~~~~

Use valgrind (Linux):

.. code-block:: bash

   valgrind --leak-check=full --show-leak-kinds=all ./your_program

On macOS use ``leaks`` with the malloc stack logging environment variable:

.. code-block:: bash

   MallocStackLogging=1 leaks -atExit -- ./your_program

The library's own test does exactly this:

.. code-block:: bash

   MallocStackLogging=1 leaks -atExit -- build/testbin/test_memory
