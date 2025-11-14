=====
About
=====


diffstruc is an open-source Fortran library for automatic differentiation (AD).
It does this by providing a derived type ``array_type`` that encapsulates arrays with gradient tracking capabilities.
The library supports both forward and reverse mode AD, enabling efficient computation of derivatives for a wide range of applications including optimization, machine learning, and scientific computing.

The library is designed to be user-friendly, allowing users to easily integrate AD into their Fortran programs with minimal code changes.
By default, the library supports a variety of mathematical operations including arithmetic, trigonometric functions, exponentials, logarithms, and linear algebra operations.
One of the core philosophies of the library is to enable efficient extension of the set of supported operations by users without need to modify the library source code; this is enabled through users employing interfaces and operator overloading.
Another core philosophy is to make this as user-friendly as possible, with clear documentation and examples provided.

The library utilises modern Fortran features such as modules, derived types, and pointers to manage memory and data structures effectively.
Due to the complexity of managing computation graphs and memory in AD, there is a chance of memory leaks if users do not properly clean up temporary objects created during differentiation.

This documentation does not intend to be an explanation of the theory behind automatic differentiation as this is well-covered by many resources online.

This library has been developed first and foremost for the `athena <https://github.com/nedtaylor/athena>`_ neural network library, but is designed to be general-purpose and reusable in other Fortran projects requiring automatic differentiation.
If you have any interest in contributing to the library or have suggestions for improvements, we are happy to have you get involved; for more information on contributing, please refer to the (:git:`contributing guidelines<CONTRIBUTING.md>`).

The code is freely available under the `MIT License <https://opensource.org/licenses/MIT>`_.
