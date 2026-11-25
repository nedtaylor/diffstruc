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

  * ``allocate()``: Allocate array storage (last shape entry = sample count;
    ``sample_dependent=``/``temporary=`` keyword flags)
  * ``deallocate()``: Release this handle's node storage back to its context
  * ``set_requires_grad()``: Enable gradient tracking
  * ``grad_forward()``: Compute gradients using forward mode
  * ``grad_reverse()``: Compute gradients using reverse mode (backpropagation)
  * ``nullify_graph()``: Clean up computation graph
  * ``duplicate_graph()``: Duplicate a computation graph

Key operations supported:

* Arithmetic: ``+``, ``-``, ``*``, ``/``, ``**``
* Trigonometric: ``sin``, ``cos``, ``tan``, ``asin``, ``acos``, ``atan``
* Hyperbolic: ``sinh``, ``cosh``, ``tanh``, ``tanhf`` (approximate kernel)
* Exponential/Logarithmic: ``exp``, ``log``, ``log10``
* Linear Algebra: ``matmul``, ``transpose``
* Reduction: ``sum``, ``mean``

Memory ownership:

* ``array_context_type``: The owner of records, values and gradients
* ``context_of()``, ``canonical()``, ``live()``: Resolve a handle to its record
* ``default_context``, ``set_default_context()``, ``reset_default_context()``:
  The process-wide context and bulk release

For complete details and source code, please refer to the FORD documentation linked above.
