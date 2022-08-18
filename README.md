# Ketju-integrator (MSTAR)

A highly parallel regularized N-body integrator library.


The interface to the integrator is defined in  
`include/ketju_integrator/ketju_integrator.h`  
see the comments there for detailed explanations.

## Building
Depending on whether you want to build the included unit tests 
(recommended for developing the code), you will either first need to run
    
    git submodule update --init

to pull in the unit test libraries required for building the tests;
or pass `-DUNIT_TESTS=OFF` as an option to `cmake` in the commands below
to disable the tests.

Run

    mkdir build
    cd build
    cmake .. [CMAKE_OPTIONS]

to configure the build system.
Configuration options can then be changed by editing the `CMakeCache.txt` file
either directly or using e.g. `ccmake`, but doing this is not typically required.

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

## License

This software is licensed under the GPLv3 license, see [LICENSE.md](LICENSE.md) for details.
