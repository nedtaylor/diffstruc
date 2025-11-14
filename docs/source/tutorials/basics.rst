Getting Started: The Basics
===========================

This tutorial introduces the fundamental concepts of using diffstruc for automatic differentiation.

What is Automatic Differentiation?
-----------------------------------

Automatic differentiation (AD) is a technique for efficiently and accurately computing derivatives of functions.
Unlike numerical differentiation (finite differences) or symbolic differentiation, AD provides:

* **Machine precision** accuracy (no approximation errors)
* **Efficiency** - computes derivatives in the same order of magnitude as the original function
* **Automatic** - no manual derivation needed

The ``array_type``
------------------

The core of diffstruc is the ``array_type`` derived type.
This type wraps your data and tracks operations to enable automatic differentiation.

Basic Structure
~~~~~~~~~~~~~~~

.. code-block:: fortran

   type(array_type) :: x

   ! Allocate with shape [height, width, batch_size]
   call x%allocate([2, 3, 1], source=1.0)

   ! Access the values
   write(*,*) x%val(:, 1)  ! First sample in batch

Unless working with neural networks (i.e., via `athena <https://github.com/nedtaylor/athena>`_), you will typically set ``batch_size = 1`` for most applications.
What this means is that, for most use cases, the final dimension of the array shape will be set to 1, indicating a single batch of data.
An ``array_shape`` argument of dimension ``[n, m, 1]`` corresponds to a 2D array of size ``n x m``.
``array_type`` currently supports array ranks of up to 5 (i.e., shapes like ``[d1, d2, d3, d4, d5, batch_size]``).

Key Components
~~~~~~~~~~~~~~

The main components of ``array_type`` are:

* ``val`` - The actual array values
* ``requires_grad`` - Flag to enable gradient tracking
* ``grad`` - Pointer to the gradient (derivative) array
* ``is_temporary`` - Flag indicating if this is a temporary computation result


Worked Example
--------------

Here we want to compute the derivative of $f(x) = x^2$ at $x = 3$.

A full code example is provided below.

.. code-block:: fortran

   program first_derivative
     use diffstruc
     implicit none

     type(array_type) :: x
     type(array_type), pointer :: f, df_dx

     ! Step 1: Create and initialise the input
     call x%allocate([1, 1], source=3.0)
     call x%set_requires_grad(.true.)
     x%is_temporary = .false.

     ! Step 2: Define the function f(x) = x^2
     f => x ** 2

     ! Step 3: Compute the derivative df/dx
     df_dx => f%grad_forward(x)

     ! Step 4: Display results
     write(*,*) 'x =', x%val(1, 1)
     write(*,*) 'f(x) = x^2 =', f%val(1, 1)
     write(*,*) 'df/dx = 2x =', df_dx%val(1, 1)
     ! Expected: df/dx = 2*3 = 6

     ! Step 5: Clean up
     call f%nullify_graph()
     call df_dx%nullify_graph()
     deallocate(f, df_dx)

   end program first_derivative

**Expected output:**

.. code-block:: text

   x =    3.00000000
   f(x) = x^2 =    9.00000000
   df/dx = 2x =    6.00000000


.. _:

In-depth Explanation
~~~~~~~~~~~~~~~~~~~~~

1. **Initialise the input**

   .. code-block:: fortran

      call x%allocate([1, 1], source=3.0)
      call x%set_requires_grad(.true.)
      x%is_temporary = .false.

   * Allocate a 1D array (with length 1 and batch size of 1) with value 3.0
   * Enable gradient tracking with ``set_requires_grad(.true.)``
   * Mark as non-temporary (important for graph tracking), but not strictly necessary here

2. **Define the function**

   .. code-block:: fortran

      f => x ** 2

   * The ``=>`` pointer assignment creates a new ``array_type``
   * diffstruc automatically builds a computation graph
   * Pointers allow efficient memory usage, avoiding unnecessary copies

3. **Compute the derivative**

   .. code-block:: fortran

      df_dx => f%grad_forward(x)

   * ``grad_forward`` computes $\frac{\partial f}{\partial x}$ using forward mode
   * Returns a pointer to the result

4. **Clean up memory**

   .. code-block:: fortran

      call f%nullify_graph()
      call df_dx%nullify_graph()
      deallocate(f, df_dx)

   * Always clean up computation graphs and deallocate memory
   * See :doc:`memory_management` for details

Multi-Dimensional Arrays
-------------------------

diffstruc works seamlessly with arrays of any size up to rank 5.

An example of computing gradients for multiple input points simultaneously is shown below.


.. code-block:: fortran

   program batch_derivative
     use diffstruc
     implicit none

     type(array_type) :: x
     type(array_type), pointer :: f, df_dx
     real :: values(4)
     integer :: i

     ! Initialise with 4 different x values: [1, 2, 3, 4]
     values = [1.0, 2.0, 3.0, 4.0]
     call x%allocate([4, 1])
     call x%set(values)
     call x%set_requires_grad(.true.)
     x%is_temporary = .false.

     ! Compute f(x) = x^2 for all values at once
     f => x ** 2

     ! Compute derivatives for all values
     df_dx => f%grad_forward(x)

     ! Display results
     write(*,*) 'x values:', x%val(:, 1)
     write(*,*) 'f(x) = x^2:', f%val(:, 1)
     write(*,*) 'df/dx = 2x:', df_dx%val(:, 1)

     ! Clean up
     call f%nullify_graph()
     call df_dx%nullify_graph()
     deallocate(f, df_dx)

   end program batch_derivative

**Expected output:**

.. code-block:: text

   x values:    1.00000000       2.00000000       3.00000000       4.00000000
   f(x) = x^2:    1.00000000       4.00000000       9.00000000       16.0000000
   df/dx = 2x:    2.00000000       4.00000000       6.00000000       8.00000000

Key Concepts
------------

Shape Convention
~~~~~~~~~~~~~~~~

diffstruc uses the shape ``[SHAPE_LIST**, batch_size]``:

* **SHAPE_LIST**: Dimensions of your data (e.g., height, width, channels), can be 1D, 2D, up to 5D
* **batch_size**: Number of samples in the batch (for most use cases, set to 1)

For simple cases, use ``[n, 1]`` where ``n`` is your data size.

requires_grad Flag
~~~~~~~~~~~~~~~~~~

Only variables with ``requires_grad = .true.`` will have gradients computed:

.. code-block:: fortran

   call x%set_requires_grad(.true.)   ! Compute gradients w.r.t. x
   call y%set_requires_grad(.false.)  ! Don't compute gradients w.r.t. y

is_temporary Flag
~~~~~~~~~~~~~~~~~

* Set to ``.false.`` for variables you create explicitly
* Intermediate results in expressions automatically have ``is_temporary = .true.``
* Temporary variables may be automatically cleaned up

.. code-block:: fortran

   x%is_temporary = .false.  ! Explicit variable
   f => x ** 2               ! f%is_temporary = .true. (automatic)



Common Issues
-------------

Forgetting to Enable Gradients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For any variable you want to differentiate with respect to, you must enable gradient tracking:

.. code-block:: fortran

   ! WRONG - no gradients computed
   call x%allocate([1, 1], source=3.0)
   f => x ** 2
   df_dx => f%grad_forward(x)  ! Will not work correctly!

   ! CORRECT
   call x%allocate([1, 1], source=3.0)
   call x%set_requires_grad(.true.)  ! Enable gradients
   f => x ** 2
   df_dx => f%grad_forward(x)

Not Cleaning Up Memory
~~~~~~~~~~~~~~~~~~~~~~

It is crucial to clean up computation graphs and deallocate memory to avoid leaks.
For small programs, this may not be noticeable, but for larger applications or long-running processes, failing to do so can lead to excessive memory usage.
It is especially important in loops or repeated computations.

.. code-block:: fortran

   ! WRONG - memory leak!
   f => x ** 2
   df_dx => f%grad_forward(x)
   ! Program ends without cleanup

   ! CORRECT
   f => x ** 2
   df_dx => f%grad_forward(x)
   call f%nullify_graph()
   call df_dx%nullify_graph()
   deallocate(f, df_dx)
