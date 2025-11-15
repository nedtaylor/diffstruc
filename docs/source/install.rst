Installation
============

This guide will help you install and set up the **diffstruc** library for automatic differentiation in Fortran.

.. contents::
   :local:
   :depth: 2


Getting the Source Code
-----------------------

The diffstruc library can be obtained from the GitHub repository:

.. code-block:: bash

   git clone https://github.com/nedtaylor/diffstruc.git
   cd diffstruc

It is planned to add diffstruc to the Fortran Package Manager (fpm) registry in the future for easier installation.
Until then, please follow the instructions below to build and install the library manually.

Prerequisites
-------------

Required Dependencies
~~~~~~~~~~~~~~~~~~~~~

To build and use diffstruc, you need:

1. **A Fortran Compiler** (compatible with Fortran 2018 or later)
2. **Fortran Package Manager (fpm)** - https://github.com/fortran-lang/fpm
3. **coreutils** - (dependency handled automatically by fpm) https://github.com/nedtaylor/coreutils

.. important::
   diffstruc is known to be **incompatible** with all versions of the gfortran compiler below ``14.3.0`` due to issues with the calling of the ``final`` procedure of ``array_type``.

coreutils is a lightweight Fortran library that provides essential precision types, mathematical constants, and utility functions.
The installation of coreutils is managed automatically by fpm when building diffstruc.

Supported Compilers
~~~~~~~~~~~~~~~~~~~

The library has been developed and tested with:

* **gfortran** -- GCC 15.2.0
* **ifx** -- Intel Fortran Compiler 2025.2.0

Installing Dependencies
-----------------------

Installing fpm
~~~~~~~~~~~~~~

**Linux/macOS:**

You can install fpm using one of the following methods:

.. code-block:: bash

   # Using conda
   conda install -c conda-forge fpm

   # Or download pre-built binary from GitHub releases
   # https://github.com/fortran-lang/fpm/releases

**Manual Installation:**

.. code-block:: bash

   git clone https://github.com/fortran-lang/fpm
   cd fpm
   ./install.sh

See the `fpm documentation <https://fpm.fortran-lang.org/install/index.html>`_ for detailed installation instructions.

Installing a Fortran Compiler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Ubuntu/Debian:**

.. code-block:: bash

   # For GCC/gfortran (ensure version >= 14.3.0)
   sudo apt-get update
   sudo apt-get install gfortran

**macOS:**

.. code-block:: bash

   # Using Homebrew
   brew install gcc

   # This typically installs as gfortran-<version>
   # Check your version
   gfortran --version

**Intel Fortran (ifx):**

Download from the `Intel oneAPI website <https://www.intel.com/content/www/us/en/developer/tools/oneapi/fortran-compiler.html>`_.

Installing coreutils
~~~~~~~~~~~~~~~~~~~~

coreutils will be installed automatically by fpm when you build diffstruc, so no manual installation is necessary.


Building diffstruc
------------------

Once you have installed the prerequisites, building diffstruc is straightforward using fpm.

Basic Build
~~~~~~~~~~~

In the repository main directory, run:

.. code-block:: bash

   fpm build --profile release

This will compile the library with optimization flags for production use.

Development Build
~~~~~~~~~~~~~~~~~

For development and debugging, you can build without the release profile:

.. code-block:: bash

   fpm build

This compiles faster but without optimizations.


Testing the Installation
------------------------

To verify that diffstruc has been installed correctly and works as expected, run the test suite:

.. code-block:: bash

   fpm test

This runs a set of test programs (found in the ``test/`` directory) to ensure:

* Core functionality works correctly
* Memory management is functioning properly
* Forward and reverse mode differentiation produce correct results
* Higher-order derivatives are computed accurately

If all tests pass, your installation is successful!

.. note::
   Some tests may take a few minutes to complete, especially memory stress tests with many iterations.


Using diffstruc in Your Project
--------------------------------

With fpm
~~~~~~~~

The easiest way to use diffstruc in your own fpm project is to add it as a dependency in your ``fpm.toml``:

.. code-block:: toml

   [dependencies]
   diffstruc = { git = "https://github.com/nedtaylor/diffstruc.git" }

Then in your Fortran code:

.. code-block:: fortran

   program my_program
     use diffstruc
     implicit none

     type(array_type) :: x, y
     type(array_type), pointer :: f

     ! Your code here...
   end program my_program


Quick Start Example
--------------------

Here's a simple example to verify your installation:

.. code-block:: fortran

   program test_diffstruc
     use diffstruc
     implicit none

     type(array_type) :: x, y
     type(array_type), pointer :: f, xgrad

     ! Allocate arrays
     call x%allocate([2,2,1], source=2.0)
     call y%allocate([2,2,1], source=10.0)

     ! Enable gradient tracking
     call x%set_requires_grad(.true.)
     x%is_temporary = .false.
     y%is_temporary = .false.

     ! Compute function: f = x * y + sin(x)
     f => x * y + sin(x)
     write(*,*) 'Value of f:', f%val(:,1)

     ! Compute gradient df/dx
     xgrad => f%grad_forward(x)
     write(*,*) 'Gradient of f w.r.t x:', xgrad%val(:,1)

     ! Clean up
     call f%nullify_graph()
     call f%deallocate()
     call xgrad%nullify_graph()
     call xgrad%deallocate()

     deallocate(f, xgrad)
   end program test_diffstruc

Save this as ``test.f90`` and compile with:

.. code-block:: bash

   fpm run test


Troubleshooting
---------------

Compiler Version Issues
~~~~~~~~~~~~~~~~~~~~~~~

If you encounter errors related to ``final`` or ``finalise_array``, this will likely be due to using an outdated Fortran compiler.

**Solution:** Ensure your gfortran version is at least 14.3.0:

.. code-block:: bash

   gfortran --version

If your version is older, upgrade your compiler.

fpm Not Found
~~~~~~~~~~~~~

If you get ``fpm: command not found``:

**Solution:** Ensure fpm is installed and in your PATH:

.. code-block:: bash

   which fpm
   # If not found, install fpm or add its location to PATH
   export PATH="$HOME/.local/bin:$PATH"

Module File Errors
~~~~~~~~~~~~~~~~~~

If you see errors about missing ``.mod`` files:

**Solution:** Clean the build directory and rebuild:

.. code-block:: bash

   rm -rf build/
   fpm build --profile release


Getting Help
------------

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/nedtaylor/diffstruc/issues>`_ page
2. Review the :doc:`api` documentation
3. Open an issue on the GitHub issue tracker, making sure to follow the (:git:`contributing guidelines<CONTRIBUTING.md>`).
