Forward-Mode Differentiation
============================

Forward-mode automatic differentiation computes derivatives by propagating derivatives forward through the computation graph alongside the function values.

Many great resources exist online explaining the theory behind forward-mode automatic differentiation.
Here, we focus on how to use diffstruc to perform forward-mode differentiation in Fortran.

Basic Usage
-----------

A simple example of using forward mode to compute the derivative of a function is shown below.

.. code-block:: fortran

   program forward_example
     use diffstruc
     implicit none

     type(array_type) :: x
     type(array_type), pointer :: f, df_dx

     ! Initialize
     call x%allocate([1, 1], source=2.0)
     call x%set_requires_grad(.true.)
     x%is_temporary = .false.

     ! f(x) = x^3 + 2*x^2 + x
     f => x**3 + 2.0*x**2 + x

     ! Compute df/dx using forward mode
     df_dx => f%grad_forward(x)

     write(*,*) 'f(2) =', f%val(1, 1)          ! 8 + 8 + 2 = 18
     write(*,*) 'df/dx(2) =', df_dx%val(1, 1)  ! 3*4 + 4*2 + 1 = 21

     ! Cleanup
     call f%nullify_graph()
     call df_dx%nullify_graph()
     deallocate(f, df_dx)

   end program forward_example


Partial Derivatives
-------------------

When you have multiple input variables, forward mode computes one partial derivative at a time.
The derivative of a function with respect to each input variable can be computed separately by calling ``grad_forward`` with the desired input.

.. code-block:: fortran

   program partial_derivatives
     use diffstruc
     implicit none

     type(array_type) :: x, y
     type(array_type), pointer :: f, df_dx, df_dy

     ! Initialize two variables
     call x%allocate([1, 1], source=3.0)
     call y%allocate([1, 1], source=4.0)
     call x%set_requires_grad(.true.)
     call y%set_requires_grad(.true.)
     x%is_temporary = .false.
     y%is_temporary = .false.

     ! f(x, y) = x^2 + x*y + y^2
     f => x**2 + x*y + y**2

     ! Compute partial derivatives
     df_dx => f%grad_forward(x)  ! ∂f/∂x = 2x + y = 6 + 4 = 10
     df_dy => f%grad_forward(y)  ! ∂f/∂y = x + 2y = 3 + 8 = 11

     write(*,*) 'f(3, 4) =', f%val(1, 1)        ! 9 + 12 + 16 = 37
     write(*,*) '∂f/∂x =', df_dx%val(1, 1)      ! 10
     write(*,*) '∂f/∂y =', df_dy%val(1, 1)      ! 11

     ! Cleanup
     call f%nullify_graph()
     call df_dx%nullify_graph()
     call df_dy%nullify_graph()
     deallocate(f, df_dx, df_dy)

   end program partial_derivatives

Additionally, the gradient vector can be computed efficiently.
For a function :math:`f: \mathbb{R}^n \rightarrow \mathbb{R}`, the gradient is:

.. math::

   \nabla f = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right]

Here, we compute the full gradient vector using forward mode, i.e. differentiating with respect to all elements of the input array.

.. code-block:: fortran

   program gradient_vector
     use diffstruc
     implicit none

     type(array_type) :: x
     type(array_type), pointer :: f, grad_f
     real :: x_values(3)

     ! f(x1, x2, x3) = x1^2 + x2^2 + x3^2
     x_values = [1.0, 2.0, 3.0]
     call x%allocate([3, 1])
     call x%set(x_values)
     call x%set_requires_grad(.true.)
     x%is_temporary = .false.

     ! Sum of squares
     f => sum(x**2, dim=2)

     ! Gradient: [2*x1, 2*x2, 2*x3] = [2, 4, 6]
     grad_f => f%grad_forward(x)

     write(*,*) 'f =', f%val(1, 1)          ! 1 + 4 + 9 = 14
     write(*,*) '∇f =', grad_f%val(:, 1)    ! [2, 4, 6]

     ! Cleanup
     call f%nullify_graph()
     call grad_f%nullify_graph()
     deallocate(f, grad_f)

   end program gradient_vector


Directional Derivatives
-----------------------

To calculate the directional derivative of a function at a point in a specified direction, forward mode can be used effectively.

The directional derivative of :math:`f` at :math:`x` in direction :math:`v` is:

.. math::

   D_v f(x) = \nabla f(x) \cdot v = \lim_{h \to 0} \frac{f(x + hv) - f(x)}{h}

To achieve this, we must set the direction vector appropriately.
This can be done using the ``set_direction`` method on the input variable.

Below is an example of computing a directional derivative using forward mode.


.. code-block:: fortran

   program directional_derivative
     use diffstruc
     implicit none

     type(array_type) :: x
     type(array_type), pointer :: f, dir_deriv
     real :: x_vals(2), direction(2)

     ! Point: x = [1, 2]
     x_vals = [1.0, 2.0]
     call x%allocate([2, 1])
     call x%set(x_vals)
     call x%set_requires_grad(.true.)
     x%is_temporary = .false.

     ! Direction: v = [1, 0] (along x1 axis)
     direction = [1.0, 0.0]
     call x%set_direction(direction)

     ! f(x1, x2) = x1^2 + x2^2
     f => sum(x**2, dim=1)

     ! Directional derivative
     dir_deriv => f%grad_forward(x)

     ! ∇f = [2*x1, 2*x2] = [2, 4]
     ! D_v f = ∇f · v = [2, 4] · [1, 0] = 2
     write(*,*) 'Directional derivative:', dir_deriv%val(1, 1)

     ! Cleanup
     call f%nullify_graph()
     call dir_deriv%nullify_graph()
     deallocate(f, dir_deriv)

   end program directional_derivative


Complex Functions
-----------------

Forward mode works with any differentiable function composition.
Here, we show examples with trigonometric, exponential, and logarithmic functions.

**Trigonometric Functions**

.. code-block:: fortran

   program trig_derivatives
     use diffstruc
     implicit none

     type(array_type) :: x
     type(array_type), pointer :: f1, f2, df1, df2
     real, parameter :: pi = 3.14159265359

     call x%allocate([1, 1], source=pi/4.0)  ! 45 degrees
     call x%set_requires_grad(.true.)
     x%is_temporary = .false.

     ! f1(x) = sin(x), f1'(x) = cos(x)
     f1 => sin(x)
     df1 => f1%grad_forward(x)

     write(*,*) 'sin(π/4) =', f1%val(1, 1)      ! ≈ 0.707
     write(*,*) 'd/dx sin =', df1%val(1, 1)     ! ≈ 0.707 (cos(π/4))

     ! f2(x) = cos(x), f2'(x) = -sin(x)
     f2 => cos(x)
     df2 => f2%grad_forward(x)

     write(*,*) 'cos(π/4) =', f2%val(1, 1)      ! ≈ 0.707
     write(*,*) 'd/dx cos =', df2%val(1, 1)     ! ≈ -0.707

     ! Cleanup
     call f1%nullify_graph()
     call f2%nullify_graph()
     call df1%nullify_graph()
     call df2%nullify_graph()
     deallocate(f1, f2, df1, df2)

   end program trig_derivatives


**Exponential and Logarithmic**

.. code-block:: fortran

   program exp_log_derivatives
     use diffstruc
     implicit none

     type(array_type) :: x
     type(array_type), pointer :: f_exp, f_log, df_exp, df_log

     call x%allocate([1, 1], source=2.0)
     call x%set_requires_grad(.true.)
     x%is_temporary = .false.

     ! f(x) = exp(x), f'(x) = exp(x)
     f_exp => exp(x)
     df_exp => f_exp%grad_forward(x)

     write(*,*) 'exp(2) =', f_exp%val(1, 1)        ! ≈ 7.389
     write(*,*) "d/dx exp(x) =", df_exp%val(1, 1)  ! ≈ 7.389

     ! f(x) = log(x), f'(x) = 1/x
     f_log => log(x)
     df_log => f_log%grad_forward(x)

     write(*,*) 'log(2) =', f_log%val(1, 1)        ! ≈ 0.693
     write(*,*) "d/dx log(x) =", df_log%val(1, 1)  ! = 0.5

     ! Cleanup
     call f_exp%nullify_graph()
     call f_log%nullify_graph()
     call df_exp%nullify_graph()
     call df_log%nullify_graph()
     deallocate(f_exp, f_log, df_exp, df_log)

   end program exp_log_derivatives

Iterative Use Cases
-------------------

Here, we show how forward mode can be applied in iterative algorithms like Newton's method.

In these iterative approaches, it is essential that the memory is correctly managed to avoid leaks.

.. code-block:: fortran

   program newtons_method
     use diffstruc
     implicit none

     type(array_type) :: x
     type(array_type), pointer :: f, df
     real :: x_val
     integer :: iter
     integer, parameter :: max_iter = 10
     real, parameter :: tol = 1.0e-6

     ! Find root of f(x) = x^2 - 2 (i.e., sqrt(2))
     x_val = 1.0  ! Initial guess

     do iter = 1, max_iter
       ! Setup
       call x%allocate([1, 1], source=x_val)
       call x%set_requires_grad(.true.)
       x%is_temporary = .false.

       ! Function and derivative
       f => x**2 - 2.0
       df => f%grad_forward(x)

       ! Newton update: x_new = x - f(x)/f'(x)
       x_val = x_val - f%val(1, 1) / df%val(1, 1)

       write(*,'(A,I2,A,F12.8)') 'Iteration', iter, ': x =', x_val

       ! Cleanup for next iteration
       call f%nullify_graph()
       call df%nullify_graph()
       deallocate(f, df)
       call x%deallocate()

       if (abs(x_val**2 - 2.0) < tol) then
         write(*,*) 'Converged! sqrt(2) ≈', x_val
         exit
       end if
     end do

   end program newtons_method


Summary
-------

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

Forward mode has time complexity O(n) per derivative, where n is the number of operations.

* **Best for**: Few inputs, many outputs
* **Cost**: One forward pass per input variable
* **Memory**: Stores intermediate derivative values

For functions with many inputs but few outputs, consider using :doc:`reverse_mode` instead.
Note, however, that forward mode is required for higher-order derivatives.


Key Points
~~~~~~~~~~

Here are some key points regarding forward-mode differentiation that one should consider (some of which have not been discussed here):

* Forward mode propagates derivatives from inputs to outputs
* Use ``grad_forward(x)`` to compute $\frac{\partial f}{\partial x}$
* Efficient for few inputs, many outputs
* Natural for directional derivatives
* Required for higher-order derivatives
