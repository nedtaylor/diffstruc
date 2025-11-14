Reverse-Mode Differentiation (Backpropagation)
==============================================

Reverse-mode automatic differentiation computes derivatives by propagating gradients backward through the computation graph.
This is the foundation of backpropagation used in neural networks.

The reverse mode algorithm applies the chain rule in reverse order, starting from the output and moving back to the inputs.
It is more efficient than forward mode when there are many input variables but few output variables (particularly for scalar outputs of neural networks).

However, reverse mode cannot handle higher order derivatives directly (it can be applied directly to a forward mode-derived derivative to return a higher order derivative).

Basic Usage
-----------

A simple example of using reverse mode to compute the derivative of a function is shown below.

.. code-block:: fortran

   program reverse_example
     use diffstruc
     implicit none

     type(array_type) :: x
     type(array_type), pointer :: f

     ! Initialise
     call x%allocate([1, 1, 1], source=3.0)
     call x%set_requires_grad(.true.)
     x%is_temporary = .false.

     ! f(x) = x^3 + 2*x^2
     f => x**3 + 2.0*x**2

     ! Compute gradient using reverse mode
     call f%grad_reverse(record_graph=.true.)

     write(*,*) 'f(3) =', f%val(1, 1)          ! 27 + 18 = 45
     write(*,*) 'df/dx =', x%grad%val(1, 1)    ! 3*9 + 4*3 = 39

     ! Cleanup
     call f%nullify_graph()
     deallocate(f)

   end program reverse_example

Notice that in reverse mode:

* We call ``grad_reverse()`` on the **output** (``f``)
* The gradient is stored in the **input's** ``grad`` field (``x%grad``)
* No pointer assignment for the gradient result

.. code-block:: fortran

   ! Forward mode: gradient is returned
   df_dx => f%grad_forward(x)
   write(*,*) df_dx%val

   ! Reverse mode: gradient stored in input
   call f%grad_reverse(record_graph=.true.)
   write(*,*) x%grad%val

All of the input gradients are computed in a single backward pass, making reverse mode very efficient for functions with many inputs and few outputs.


Partial Derivatives
-------------------

Here, we compute gradients with respect to multiple input variables in one backward pass:

.. code-block:: fortran

   program reverse_multiple
     use diffstruc
     implicit none

     type(array_type) :: x, y, z
     type(array_type), pointer :: f

     ! Initialise three variables
     call x%allocate([1, 1, 1], source=2.0)
     call y%allocate([1, 1, 1], source=3.0)
     call z%allocate([1, 1, 1], source=4.0)
     call x%set_requires_grad(.true.)
     call y%set_requires_grad(.true.)
     call z%set_requires_grad(.true.)
     x%is_temporary = .false.
     y%is_temporary = .false.
     z%is_temporary = .false.

     ! f(x,y,z) = x*y + y*z + x*z
     f => x*y + y*z + x*z

     ! One backward pass computes all gradients!
     call f%grad_reverse(record_graph=.true.)

     write(*,*) 'f =', f%val(1, 1)             ! 6 + 12 + 8 = 26
     write(*,*) '∂f/∂x =', x%grad%val(1, 1)    ! y + z = 7
     write(*,*) '∂f/∂y =', y%grad%val(1, 1)    ! x + z = 6
     write(*,*) '∂f/∂z =', z%grad%val(1, 1)    ! y + x = 5

     ! Cleanup
     call f%nullify_graph()
     deallocate(f)

   end program reverse_multiple

Another example with high-dimensional inputs:

.. code-block:: fortran

   program high_dimensional
     use diffstruc
     implicit none

     type(array_type) :: params
     type(array_type), pointer :: loss
     real :: param_vals(1000)
     integer :: i

     ! Initialise 1000 parameters
     do i = 1, 1000
       param_vals(i) = real(i) / 1000.0
     end do

     call params%allocate([1000, 1, 1])
     call params%set(param_vals)
     call params%set_requires_grad(.true.)
     params%is_temporary = .false.

     ! Loss function: sum of squares
     loss => sum(params**2, dim=1)

     ! One backward pass computes all 1000 gradients!
     call loss%grad_reverse()

     write(*,*) 'Loss =', loss%val(1, 1)
     write(*,*) 'First 5 gradients:', params%grad%val(1:5, 1)
     ! Gradient of x^2 is 2x, so we expect [2/1000, 4/1000, ...]

     ! Cleanup
     call loss%nullify_graph()
     call loss%deallocate()
     deallocate(loss)

   end program high_dimensional


Practical Application: Gradient Descent
----------------------------------------

Reverse mode is perfect for optimisation algorithms.

Simple Gradient Descent
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: fortran

   program gradient_descent
     use diffstruc
     implicit none

     type(array_type) :: x
     type(array_type), pointer :: loss
     real :: learning_rate = 0.1
     integer :: epoch

     ! Minimise f(x) = (x - 5)^2
     ! Minimum at x = 5
     call x%allocate([1, 1, 1], source=0.0)  ! Start at x=0
     call x%set_requires_grad(.true.)
     x%is_temporary = .false.

     do epoch = 1, 50
       ! Forward pass
       loss => (x - 5.0)**2

       ! Backward pass
       call loss%grad_reverse()

       ! Gradient descent update: x = x - lr * grad
       x%val(1, 1) = x%val(1, 1) - learning_rate * x%grad%val(1, 1)

       if (mod(epoch, 10) == 0) then
         write(*,'(A,I3,A,F8.4,A,F8.4)') &
           'Epoch ', epoch, ': x = ', x%val(1, 1), &
           ', loss = ', loss%val(1, 1)
       end if

       ! Cleanup for next iteration
       call x%zero_grad()  ! Reset gradient
       call loss%nullify_graph()
       call loss%deallocate()
       deallocate(loss)
     end do

     write(*,*) 'Final x =', x%val(1, 1)  ! Should be ≈ 5.0

   end program gradient_descent

Multi-Parameter Optimisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: fortran

   program optimise_2d
     use diffstruc
     implicit none

     type(array_type) :: params
     type(array_type), pointer :: loss
     real :: param_vals(2), learning_rate = 0.1
     integer :: epoch

     ! Minimise: (x-3)^2 + (y-4)^2
     ! Minimum at (3, 4)
     param_vals = [0.0, 0.0]  ! Start at origin
     call params%allocate([2, 1])
     call params%set(param_vals)
     call params%set_requires_grad(.true.)
     params%is_temporary = .false.

     do epoch = 1, 100
       ! Extract x and y
       loss => (pack(params,[1], dim=1) - 3.0)**2 + &
               (pack(params,[2], dim=1) - 4.0)**2

       ! Backward pass computes both gradients
       call loss%grad_reverse()

       ! Update both parameters
       params%val(:, 1) = params%val(:, 1) - &
                          learning_rate * params%grad%val(:, 1)

       if (mod(epoch, 20) == 0) then
         write(*,'(A,I3,A,2F8.4,A,F10.6)') &
           'Epoch ', epoch, ': (x,y) = (', params%val(:, 1), &
           '), loss = ', loss%val(1, 1)
       end if

       ! Cleanup
       call params%zero_grad()
       call loss%nullify_graph()
       call loss%deallocate()
       deallocate(loss)
     end do

     write(*,*) 'Final params:', params%val(:, 1)

   end program optimise_2d

Summary
-------

Key Points
~~~~~~~~~~

Here are the main takeaways for reverse-mode differentiation:

* Reverse mode propagates gradients from outputs to inputs
* Use reverse mode for many parameters, few outputs
* Gradients stored in ``input%grad``
* One pass computes all input gradients
* Call ``zero_grad()`` between iterations to reset gradients
* Clean up graphs with ``nullify_graph()``
