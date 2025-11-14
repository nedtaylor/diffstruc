Tutorials
=========

These tutorials will guide you through using diffstruc for automatic differentiation in Fortran.

Overview
--------

The tutorials are organised to build your understanding progressively:

1. **Basics** - Get started with automatic differentiation using diffstruc
2. **Forward Mode** - Learn how to use forward-mode automatic differentiation
3. **Reverse Mode** - Learn how to use reverse-mode (backpropagation) automatic differentiation
4. **Higher-Order Derivatives** - Learn how to compute second and higher-order derivatives
5. **Operations** - Learn about the supported mathematical operations and how to define custom ones
6. **Memory Management** - Learn how to manage memory effectively when using diffstruc

Prerequisites
-------------

Before starting these tutorials, make sure you have:

* Installed diffstruc (see :doc:`install`)
* A working Fortran compiler (gfortran >= 14.3.0 or ifx >= 2025.2.0)
* Basic understanding of Fortran programming
* Familiarity with calculus and derivatives


.. toctree::
   :maxdepth: 2
   :caption: Tutorial Contents:

   tutorials/basics
   tutorials/forward_mode
   tutorials/reverse_mode
   tutorials/higher_order
   tutorials/operations
   tutorials/memory_management
