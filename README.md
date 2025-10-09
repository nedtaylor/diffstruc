[![MIT workflow](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/mit/ "View MIT license")
[![Latest Release](https://img.shields.io/github/v/release/nedtaylor/diffstruc?sort=semver)](https://github.com/nedtaylor/diffstruc/releases "View on GitHub")
[![FPM](https://img.shields.io/badge/fpm-0.12.0-purple)](https://github.com/fortran-lang/fpm "View Fortran Package Manager")
[![CMAKE](https://img.shields.io/badge/cmake-3.17.5-red)](https://github.com/Kitware/CMake/releases/tag/v3.17.5 "View cmake")
[![GCC compatibility](https://img.shields.io/badge/gcc-15.1.0-green)](https://gcc.gnu.org/gcc-15/ "View GCC")

# diffstruc

by Ned Thaddeus Taylor

diffstruc is a Fortran library that provides automatic differentiation capabilities through use of an array derived type.

---

diffstruc is distributed with the following directories:

| Directory | Description |
|---|---|
|  _src/_ |      Source code  |
|  _test/_  |    A set of unit test programs to check functionality of the library works after compilation |


Documentation
-----

The library has a compilable documentation this can be accessed with the [FORD (FORtran Documenter)](https://forddocs.readthedocs.io/en/stable/) tool.
The documentation can be compiled using the following terminal command in the root directory of the repository:

```
  ford ford.md
```

This will generate the `doc/html` directory, inside which, you will find `index.html`.
By opening this file in a browser window, you will be able to view a nagivable documentation.


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
- gfortran -- gcc 15.1.0


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


Developers
----------
- Ned Thaddeus Taylor

Contributing
------------

Please note that this project adheres to the [Contributing Guide](CONTRIBUTING.md). If you want to contribute to this project, please first read through the guide.


License
-------
This work is licensed under an [MIT license](https://opensource.org/license/mit/).
