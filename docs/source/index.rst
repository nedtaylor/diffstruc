Welcome to diffstruc's documentation!
=====================================

**diffstruc** is a Fortran library providing the framework for automatic diferentiation.

The code is provided freely available under the `MIT License <https://opensource.org/licenses/MIT>`_.

The documentation provided here does not intend to be an explanation of the theory behind automatic differentiation; many great resources exist online for this purpose.

An example of how to use the library is shown below:


.. code-block:: fortran

   program test_diffstruc
     use diffstruc
     implicit none

     type(array_type) :: x, y
     type(array_type), pointer :: f, xgrad

     call x%allocate([2,2,1], source=2.0) ! Allocate a 2x2 array with 1 sample
     call y%allocate([2,2,1], source=10.0) ! Allocate a 2x2 array with 1 sample
     call x%set_requires_grad(.true.)
     x%is_temporary = .false.
     y%is_temporary = .false.

     f => x * y + sin(x)
     write(*,*) 'Value of f:', f%val(:,1)

     ! Perform differentiation
     xgrad => f%grad_forward(x)
     write(*,*) 'Gradient of f w.r.t x:', xgrad%val(:,1)

     ! Clean up
     call f%nullify_graph()
     call xgrad%nullify_graph()
   end program test_diffstruc


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   about
   install
   tutorials
   Fortran API <api>
