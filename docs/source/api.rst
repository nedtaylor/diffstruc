.. _api:

API Reference
=============

The full Fortran API documentation is generated using FORD (Fortran Automatic Documentation).

You can browse the complete API documentation here:

.. raw:: html

   <div style="margin: 20px 0;">
      <a href="_static/ford/index.html" target="_blank" class="btn btn-primary">
         📚 Open FORD API Documentation
      </a>
   </div>

The API documentation includes:

* **Module Reference**: Detailed documentation of all modules
* **Type Definitions**: Documentation of derived types like ``array_type``
* **Procedures**: All public procedures and their interfaces
* **Source Code**: Annotated source code with cross-references
* **Call Graphs**: Visual representation of procedure dependencies

Main Modules
------------

diffstruc
~~~~~~~~~

The main module providing automatic differentiation functionality.

Key types:

* ``array_type``: The core type for differentiable arrays

  * ``allocate()``: Allocate array storage
  * ``deallocate()``: Free array storage
  * ``set_requires_grad()``: Enable gradient tracking
  * ``grad_forward()``: Compute gradients using forward mode
  * ``grad_reverse()``: Compute gradients using reverse mode (backpropagation)
  * ``nullify_graph()``: Clean up computation graph
  * ``duplicate_graph()``: Duplicate a computation graph

Key operations supported:

* Arithmetic: ``+``, ``-``, ``*``, ``/``, ``**``
* Trigonometric: ``sin``, ``cos``, ``tan``, ``asin``, ``acos``, ``atan``
* Hyperbolic: ``sinh``, ``cosh``, ``tanh``
* Exponential/Logarithmic: ``exp``, ``log``, ``log10``
* Linear Algebra: ``matmul``, ``transpose``
* Reduction: ``sum``, ``mean``

For complete details and source code, please refer to the FORD documentation linked above.
