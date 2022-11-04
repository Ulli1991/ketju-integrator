# Ketju-integrator (MSTAR)

A highly parallel regularized N-body integrator library.


The interface to the integrator is defined in
`include/ketju_integrator/ketju_integrator.h`.
See the comments there for detailed explanations of the code interface.

## Building

To build the integrator library first configure the CMake build system by running

    mkdir build
    cd build
    cmake .. [CMAKE_OPTIONS]

Configuration options such specific compiler optimization flags can then be
changed by editing the `CMakeCache.txt` file either directly or using e.g. `ccmake`,
but doing this is not typically required.

After configuration, the library, a python shim and the standalone driver 
program can be built by running `make` in the build directory.
The standalone `nbody` program is found under `bin/`.
See the [`nbody` README](nbody/README.md) for details on using it.

## Usage in other codes

The public interface of the library is included in the file 
`include/ketju_integrator/ketju_integrator.h`,
so in your C or C++ code add  
`#include <ketju_integrator/ketju_integrator.h>`  
to use the definitions.

If you're building your project with `cmake`, you can use the following commands
in your `CMakeLists.txt` to add the necessary include and linker flags:  

    set(ketju-integrator_DIR "/path/to/new-ketju-integrator/") 
    find_package(ketju-integrator 1.0 REQUIRED)  
    add_executable(nbody "main.c") # using the nbody program as an example  
    target_link_libraries(nbody ketju-integrator)

Otherwise simply add (assuming gcc/clang)   
`-I/path/to/new-ketju-integrator/include/`  
to your compiler flags and    
`-L/path/to/new-ketju-integrator/lib -lketju-integrator`  
to your linker flags.


## Python bindings

Simple Python 3 bindings depending only on the standard library and NumPy are
included as the `ketju_integrator` module under `python_bindings`.
They function via a small C library shim and allow parallel integration by
running the integrator in subprocesses with MPI.
An example of using the bindings is in
`python_bindings/ketju_integrator_example.py`.
For more convenient usage add the `python_bindings` directory to the `PYTHONPATH`.

These bindings may not work on all systems, as internally the code calls `mpirun`
to launch the integrator subprocess.
They might however be useful for small experiments on e.g. desktop systems.

## Test suite

If you are making changes to the integrator code itself, it is useful to also
build and run the included test suite.
This can be useful also for checking that the code functions correctly on your system.
To build the test suite, run
    
    git submodule update --init

to pull in the unit test libraries required for building the tests.
Then set `UNIT_TESTS=ON` in the CMake configuration and run the build as normal.

The tests can be run using `make run-unit-tests` or by running the test binary
found under `bin/` directly.
Note that some of the are currently somewhat sensitive to rounding errors caused
by different compilers and optimizations, so in case the tests fail inspect the
output to see if the errors in the results are only slightly over the assigned tolerances.


## License

This software is licensed under the GPLv3 license, see [LICENSE.md](LICENSE.md) for details.

The code under `thirdparty` is distributed under the licenses specified there.
