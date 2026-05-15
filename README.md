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

> **_NOTE:_** diffstruc is known to be incompatible with all versions of the gfortran compiler below `14.3.0` due to issues with the calling of the `final` procedure of `array_type`.



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

#### Apple Silicon dense backend selection

On macOS, dense GEMM and GEMV paths can be steered at runtime with the
`DIFFSTRUC_LINALG_BACKEND` environment variable:

```
  DIFFSTRUC_LINALG_BACKEND=legacy      # existing BLAS / LAPACK path
  DIFFSTRUC_LINALG_BACKEND=accelerate  # Accelerate.framework CBLAS path
  DIFFSTRUC_LINALG_BACKEND=metal       # Metal / MPS GEMM path (feature build)
  DIFFSTRUC_LINALG_BACKEND=auto        # size-based CPU/GPU dispatch
```

The default mode is `auto`. On standard macOS builds this selects the
Accelerate path. On `metal` feature builds it still prefers Accelerate for the
measured dense training/inference sizes used by diffstruc and only promotes to
Metal / MPS for very large GEMM workloads. Explicit `metal` mode always forces
the MPS path when available.

Optional Metal support can be enabled in fpm with:

```
  fpm test --profile release-metal
```

The `test_benchmark_matmul` target reports per-backend forward and reverse-mode
timings across a range of dense matrix sizes. The automatic Metal threshold can
be tuned with `DIFFSTRUC_METAL_MIN_OPS` when benchmarking.

#### Reproducible Apple Silicon validation

The commands below were used for validation in the `raffle_env` conda
environment on Apple Silicon:

```bash
  rm -rf build/gfortran_* build/compile_commands.json
  conda run -n raffle_env fpm test --profile release
  rm -rf build/gfortran_* build/compile_commands.json
  conda run -n raffle_env fpm test --profile release-metal
  rm -rf build/gfortran_* build/compile_commands.json
  conda run -n raffle_env fpm test --target test_backend_equivalence --profile release-metal
  conda run -n raffle_env fpm test --target test_benchmark_matmul --profile release-metal
  conda run -n raffle_env fpm test --target test_benchmark_reverse --profile release-metal
```

When switching between `release` and `release-metal`, clear the generated
`build/gfortran_*` artifacts first. With the current `fpm` + `conda run`
workflow, profile changes can otherwise reuse stale module objects and hide the
Metal feature during incremental rebuilds.

`test_backend_equivalence` compares forward values and reverse-mode gradients
between the legacy BLAS path, Accelerate, and Metal backends within tight
tolerances.

`test_benchmark_matmul` performs wall-clock dense GEMM benchmarking for:

```bash
  DIFFSTRUC_LINALG_BACKEND=legacy
  DIFFSTRUC_LINALG_BACKEND=accelerate
  DIFFSTRUC_LINALG_BACKEND=metal
  DIFFSTRUC_LINALG_BACKEND=auto
```

The benchmark includes warmup passes and reports autodiff forward/reverse and
real-matrix forward/reverse timings across increasing matrix sizes.

For backend bring-up or threshold tuning, the following optional environment
variables are available:

```bash
  DIFFSTRUC_METAL_MIN_OPS=268435456   # override the auto-dispatch crossover
  DIFFSTRUC_BACKEND_DEBUG=1           # log backend probing and Metal setup
```


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
