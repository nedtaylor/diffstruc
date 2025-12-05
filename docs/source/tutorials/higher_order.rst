Higher-Order Derivatives
========================

diffstruc supports computation of second, third, and higher-order derivatives through repeated application of forward-mode differentiation.

Computing Second Derivatives
----------------------------

An example of computing the second derivative of a function is shown below.

.. code-block:: fortran

   program second_derivative
     use diffstruc
     implicit none

     type(array_type) :: x
     type(array_type), pointer :: f, df_dx, d2f_dx2

     ! f(x) = x^4
     ! f'(x) = 4x^3
     ! f''(x) = 12x^2

     call x%allocate([1, 1], source=2.0)
     call x%set_requires_grad(.true.)
     x%is_temporary = .false.

     ! First derivative
     f => x**4
     df_dx => f%grad_forward(x)

     ! Second derivative: differentiate df_dx with respect to x
     d2f_dx2 => df_dx%grad_forward(x)

     write(*,*) 'f(2) = 2^4 =', f%val(1, 1)              ! 16
     write(*,*) "f'(2) = 4*8 =", df_dx%val(1, 1)         ! 32
     write(*,*) "f''(2) = 12*4 =", d2f_dx2%val(1, 1)     ! 48

     ! Cleanup
     call f%nullify_graph()
     call df_dx%nullify_graph()
     call d2f_dx2%nullify_graph()
     deallocate(f, df_dx, d2f_dx2)

   end program second_derivative

As shown, the second derivative is computed by applying ``grad_forward()`` twice:

1. Compute first derivative: ``df_dx => f%grad_forward(x)``
2. Compute second derivative: ``d2f_dx2 => df_dx%grad_forward(x)``

Each call to ``grad_forward`` adds another order of differentiation.

**Third-Order Derivatives**

An example of computing the third derivative of a function is shown below.

.. code-block:: fortran

   program third_derivative
     use diffstruc
     implicit none

     type(array_type) :: x
     type(array_type), pointer :: f, df, d2f, d3f

     ! f(x) = x^5
     ! f'(x) = 5x^4
     ! f''(x) = 20x^3
     ! f'''(x) = 60x^2

     call x%allocate([1, 1], source=2.0)
     call x%set_requires_grad(.true.)
     x%is_temporary = .false.

     f => x**5
     df => f%grad_forward(x)         ! First derivative
     d2f => df%grad_forward(x)       ! Second derivative
     d3f => d2f%grad_forward(x)      ! Third derivative

     write(*,*) 'f(2) =', f%val(1, 1)       ! 32
     write(*,*) "f'(2) =", df%val(1, 1)     ! 80
     write(*,*) "f''(2) =", d2f%val(1, 1)   ! 160
     write(*,*) "f'''(2) =", d3f%val(1, 1)  ! 240

     ! Cleanup
     call f%nullify_graph()
     call df%nullify_graph()
     call d2f%nullify_graph()
     call d3f%nullify_graph()
     deallocate(f, df, d2f, d3f)

   end program third_derivative



Partial Derivatives
-------------------

For functions of multiple variables, you can compute mixed partials:

.. math::

   \frac{\partial^2 f}{\partial x \partial y}, \quad \frac{\partial^2 f}{\partial y \partial x}

An example of computing mixed partial derivatives is shown below.

.. code-block:: fortran

   program mixed_partials
     use diffstruc
     implicit none

     type(array_type) :: x, y
     type(array_type), pointer :: f, df_dx, d2f_dxdy

     ! f(x, y) = x^2 * y^3
     ! ∂f/∂x = 2x * y^3
     ! ∂²f/∂x∂y = 6x * y^2

     call x%allocate([1, 1], source=2.0)
     call y%allocate([1, 1], source=3.0)
     call x%set_requires_grad(.true.)
     call y%set_requires_grad(.true.)
     x%is_temporary = .false.
     y%is_temporary = .false.

     f => x**2 * y**3
     df_dx => f%grad_forward(x)          ! ∂f/∂x
     d2f_dxdy => df_dx%grad_forward(y)   ! ∂²f/∂x∂y

     write(*,*) 'f(2,3) =', f%val(1, 1)            ! 4 * 27 = 108
     write(*,*) '∂f/∂x =', df_dx%val(1, 1)         ! 2*2 * 27 = 108
     write(*,*) '∂²f/∂x∂y =', d2f_dxdy%val(1, 1)   ! 6*2 * 9 = 108

     ! Cleanup
     call f%nullify_graph()
     call df_dx%nullify_graph()
     call d2f_dxdy%nullify_graph()
     deallocate(f, df_dx, d2f_dxdy)

   end program mixed_partials

This can be extended further to calculate the full Hessian matrix:

.. math::

   H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}

An example is provided below.

.. code-block:: fortran

   program hessian_example
     use diffstruc
     implicit none

     type(array_type) :: x, y
     type(array_type), pointer :: f
     type(array_type), pointer :: df_dx, df_dy
     type(array_type), pointer :: d2f_dx2, d2f_dxdy, d2f_dydx, d2f_dy2
     real :: hessian(2, 2)

     ! f(x, y) = x^2 + x*y + y^2

     call x%allocate([1, 1], source=1.0)
     call y%allocate([1, 1], source=1.0)
     call x%set_requires_grad(.true.)
     call y%set_requires_grad(.true.)
     x%is_temporary = .false.
     y%is_temporary = .false.

     f => x**2 + x*y + y**2

     ! First derivatives
     df_dx => f%grad_forward(x)   ! 2x + y
     df_dy => f%grad_forward(y)   ! x + 2y

     ! Second derivatives
     d2f_dx2 => df_dx%grad_forward(x)   ! ∂²f/∂x²
     d2f_dxdy => df_dx%grad_forward(y)  ! ∂²f/∂x∂y
     d2f_dydx => df_dy%grad_forward(x)  ! ∂²f/∂y∂x
     d2f_dy2 => df_dy%grad_forward(y)   ! ∂²f/∂y²

     ! Build Hessian matrix
     hessian(1, 1) = d2f_dx2%val(1, 1)   ! 2
     hessian(1, 2) = d2f_dxdy%val(1, 1)  ! 1
     hessian(2, 1) = d2f_dydx%val(1, 1)  ! 1
     hessian(2, 2) = d2f_dy2%val(1, 1)   ! 2

     write(*,*) 'Hessian matrix:'
     write(*,*) hessian(1, :)
     write(*,*) hessian(2, :)
     ! Output:
     !  2.0  1.0
     !  1.0  2.0

     ! Cleanup
     call f%nullify_graph()
     call df_dx%nullify_graph()
     call df_dy%nullify_graph()
     call d2f_dx2%nullify_graph()
     call d2f_dxdy%nullify_graph()
     call d2f_dydx%nullify_graph()
     call d2f_dy2%nullify_graph()
     deallocate(f, df_dx, df_dy, d2f_dx2, d2f_dxdy, d2f_dydx, d2f_dy2)

   end program hessian_example


Summary
-------

Memory and Computational Cost
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each order of differentiation requires additional memory.
For high-order derivatives with large arrays, memory can become limiting.

* $n$-th order derivative requires $n$ forward passes
* Cost likely increases greater than that due to the growing computation graph that must be traversed multiple times

It is advisable to:

1. Clean up intermediate results promptly
2. Consider numerical methods for very high orders (>3)
3. Test with small arrays first

Key Points
~~~~~~~~~~

* Higher-order derivatives use repeated ``grad_forward()`` calls
* Second derivative: ``d2f => df%grad_forward(x)``
* Mixed partials: Differentiate with respect to different variables
* Hessian: Matrix of all second-order partials
* Memory scales with derivative order
