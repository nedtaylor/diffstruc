[![MIT workflow](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/mit/ "View MIT license")
[![Latest Release](https://img.shields.io/github/v/release/nedtaylor/diffstruc?sort=semver)](https://github.com/nedtaylor/diffstruc/releases "View on GitHub")
[![Documentation Status](https://readthedocs.org/projects/diffstruc/badge/?version=latest)](https://diffstruc.readthedocs.io/en/latest/?badge=latest "diffstruc ReadTheDocs")
[![FPM](https://img.shields.io/badge/fpm-0.12.0-purple)](https://github.com/fortran-lang/fpm "View Fortran Package Manager")
[![GCC compatibility](https://img.shields.io/badge/gcc-15.2.0-green)](https://gcc.gnu.org/gcc-15/ "View GCC")
[![IFX compatibility](https://img.shields.io/badge/ifx-2025.2.0-green)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/fortran-compiler.html "View ifx")

# diffstruc

by Ned Thaddeus Taylor

diffstruc is a Fortran library that provides automatic differentiation capabilities through use of an array derived type.

The library has implemented both forward and reverse mode automatic differentiation.
Through repetitive use of the forward mode differentiation procedure, any higher order partial differentiation is achievable (note, memory usage will increase for higher order differentials) and has been tested up to third order.

---

diffstruc is distributed with the following directories:

| Directory | Description |
|---|---|
|  _docs/_ |    Compilable documentation |
|  _src/_ |      Source code  |
|  _test/_ |    A set of unit test programs to check functionality of the library works after compilation |



Documentation
-----

Tutorials and documentation are provided on the [docs](http://diffstruc.readthedocs.io/) website.

Refer to the [API Documentation section](#api-documentation) later in this document to see how to access the API-specific documentation.

Setup
-----

The diffstruc library can be obtained from the git repository.
Use the following commands to get started:

```
  git clone https://github.com/nedtaylor/diffstruc.git
  cd diffstruc
```


### Dependencies

The library has the following dependencies
- A Fortran compiler (compatible with Fortran 2018 or later)
- [fpm](https://github.com/fortran-lang/fpm)

The library has been developed and tested using the following compilers:
- gfortran -- gcc 14.3.0, 15.2.0
- ifx -- ifx 2025.2.0
- flang -- flang 22.1.1

> **_NOTE:_** `array_type` has no finaliser, so the compiler limitation that
> affected older gfortran versions no longer applies. The library is tested with
> the compilers listed above.



### Building with fpm

The library is set up to work with the Fortran Package Manager (fpm).

Run the following command in the repository main directory:
```
  fpm build --profile release
```

#### Testing with fpm

To check whether diffstruc has installed correctly and that the compilation works as expected, the following command can be run:
```
  fpm test
```

This runs a set of test programs (found within the test/ directory) to ensure the expected output occurs when layers and networks are set up.

### Context-owned arrays

`array_type` is a lightweight, non-owning handle to an array node. All memory
-- node records, values, gradients, shapes and forward-mode seeds -- belongs to
an `array_context_type`. A handle carries only the id of its node, the context
generation it was bound in, and views of the context-owned data. Copying a
handle therefore never copies or frees storage, and no destructor, finaliser or
reference count is needed.

```fortran
use diffstruc, only: real32, array_type

type(array_type) :: x
type(array_type), pointer :: f

call x%allocate(array_shape=[2, 1])
x%val(:, 1) = [1.0_real32, 2.0_real32]
call x%set_requires_grad(.true.)

f => x**2 + x
call f%grad_reverse(reset_graph=.true.)
print *, x%grad%val(:, 1)     ! 3.0, 5.0
call f%nullify_graph()        ! releases the expression nodes
```

Semantics:

- Assignment (`f = expr`) binds the handle to the expression's node: no value
  storage is copied and both handles view the same values. Assigning into a
  handle that is already bound (for instance the target of a pointer) copies
  the values into that handle's existing record.
- `allocate`, `set`, `extract`, `flatten` and the arithmetic operators all work
  on the context-owned storage; `%val` is a view of it.
- `nullify_graph` releases the operand and gradient records of an expression.
  Root variables (nodes marked `requires_grad` with no operands) are kept, so
  they can be reused to build the next expression. Other nodes are released and
  any handle that still refers to them is detected as stale.
- `reset_graph` only touches gradients: variable gradients are zeroed and
  transient gradient records are released.
- `deallocate` returns that handle's node storage to its context and unbinds
  the handle; calling it after `nullify_graph` is a harmless no-op. It never
  releases the whole context: use `reset_default_context()` (or `reset` on a
  named context) for that.
- The context bumps a generation counter on reset. A handle bound before the
  reset is rejected with a diagnostic instead of being dereferenced, and a
  handle that was never bound is rejected the same way.

The global context is `default_context`; `set_default_context` installs another
one and `reset_default_context` releases all of its memory. The memory model is
single-owner with checkpoint/rollback: `checkpoint_context` marks a point and
`rollback_context` releases every node created after it in bulk, without
invalidating records created before it.

The legacy ownership members (`owns_left_operand`, `owns_right_operand`,
`owns_gradient`, `fix_pointer`) and the type finaliser have been removed.
See `MIGRATION_VERIFICATION.md` for the migration record, the verification
matrix and the source audit.
Expression temporaries are released by `nullify_graph` or by a context reset.
`assign_and_deallocate_source` copies its source into a record of its own and
releases the source node when the source is a disposable temporary.

Allocation takes the flags as keywords, so they never have to be set before
the handle is bound, and `source=` accepts any value with the same number of
elements:

```fortran
call w%allocate(array_shape=[rows, cols, 1], source=matrix, &
     sample_dependent=.false., temporary=.false.)
```

See `docs/source/porting.rst` for the full migration checklist.

### Hyperbolic tangent: accuracy versus speed

Two operations are provided:

| Operation | Kernel | Accuracy vs libm `tanhf` | Speed |
|---|---|---|---|
| `tanh(x)` | libm `tanhf` | bit-identical | 1x |
| `tanhf(x)` | [7/6] Pade approximation | max abs. error 8.5e-5 | about 10x on the kernel |

`tanh` is the default inside the library (`tanh_array`) and reproduces the
reference results exactly. `tanhf` is the vectorised approximation used by
several libraries for neural-network training (the same [7/6] Pade form as
Eigen's `fast_tanh` for `float`); it is bounded by +/-1, has no NaNs, and its
maximum absolute error on [-9, 9] is 8.5e-5, i.e. about three digits below
real32's 1e-7 relative precision scale. Both operations produce the same graph
structure and use the same partial derivative (`1 - tanh(x)^2`), so they can be
swapped freely.

```fortran
f => tanh(x)     ! accurate, bit-identical to the reference implementation
g => tanhf(x)    ! fast approximation, same API and gradients
```

A build can also switch the library's own `tanh` to the fast kernel with the
preprocessor option `-DDIFFSTRUC_FAST_TANH` (for example
`fpm build --flag "-DDIFFSTRUC_FAST_TANH"`). With the option, chain B of the
reverse benchmark falls from 0.394 ms to 0.299 ms; without it every printed
value and checksum is unchanged.

`test/test_tanh.f90` checks both the accuracy of `tanhf` against `tanh` and
that their gradients agree.

Known issues
------------

#### Compiler note: bracketed expressions and temporaries

> **_WARNING:_** When chaining overloaded operators on `array_type`, expressions that use brackets to group intermediate results may cause compiler-generated temporaries to be created. For example:
>
> ```fortran
> y => (w * x)**2
> ```
>
> This pattern works correctly with **gfortran**, but can behave incorrectly with **flang** because the temporary result produced by `(w * x)` may not propagate the `is_temporary` flag correctly through the chained operator call, leading to incorrect gradient computation or memory issues.
>
> The recommended safe pattern is to store intermediate results explicitly:
>
> ```fortran
> tmp => w * x
> y => tmp**2
> ```
>
> This avoids reliance on compiler-specific handling of temporaries in operator-overloaded expressions and is the portable approach across all supported compilers.

API documentation
-----------------

API documentation can be generated using FORD (Fortran Documenter).
The library has a compilable documentation this can be accessed with the [FORD (FORtran Documenter)](https://forddocs.readthedocs.io/en/stable/) tool.
The documentation can be compiled using the following terminal command in the root directory of the repository:

```
  ford ford.md
```

This will generate the `docs/html` directory, inside which, you will find `index.html`.
By opening this file in a browser window, you will be able to view a nagivable documentation.

Contributing
------------

Please note that this project adheres to the [Contributing Guide](CONTRIBUTING.md). If you want to contribute to this project, please first read through the guide.


License
-------
This work is licensed under an [MIT license](https://opensource.org/license/mit/).
