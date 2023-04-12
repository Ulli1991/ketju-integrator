# Ketju-integrator (MSTAR)

A highly parallel regularized N-body integrator library.

The interface to the integrator is defined in
`include/ketju_integrator/ketju_integrator.h`.
See the comments there for explanations of the code interface.
A simple example program is included in the `nbody/` directory.
For a more complex example of using the library, see the [gadget4-ketju repository](https://bitbucket.org/helsinkiextragalacticgroup/gadget4-ketju).

The main numerical methods used in the code are explained in the [code paper](https://ui.adsabs.harvard.edu/abs/2020MNRAS.492.4131R).
Additional details and references are given in the comments within the code.


## Building
The code requires an MPI library/environment (e.g. OpenMPI) to be installed before building.

To build the integrator library, first configure the CMake build system by running  
```
    mkdir build
    cd build
    cmake .. [CMAKE_OPTIONS]
```
Configuration options such specific compiler optimization flags can then be
changed by editing the `CMakeCache.txt` file either directly or using e.g. `ccmake`,
but doing this is not typically required.

After configuration, the library and other included components can be
built by running `make` in the build directory.
The `nbody` executable is then found under `bin/`.
See the [`nbody` README](nbody/README.md) for details on using it.

## Usage in other codes

The public interface of the library is included in the file 
`include/ketju_integrator/ketju_integrator.h`,
so in your C or C++ code add  
`#include <ketju_integrator/ketju_integrator.h>`  
to use the definitions.
For use with other languages,
you may need to write a small wrapper for the library using C first.
You also need to include the MPI library in your code and perform the necessary
initialization for it before calling the integrator.

The general outline of using the integrator looks something like this:
```C++
    // Set up your MPI environment etc. with mpi_communicator being a communicator
    // over the tasks you want to use for the integration (can be MPI_COMM_WORLD).
    // All the tasks on this communicator must call the following functions and
    // modify the integrator setting and initial state in the same way.

    // Create the integrator data structures
    // num_pn_particles is the number of particles (usually black holes)
    // with PN terms enabled (depending on options).
    // num_other_particles is the number of particles (usually stars)
    // using only Newtonian gravity, possibly with softening.
    struct ketju_system R;
    ketju_create_system(&R, num_pn_particles, num_other_particles, mpi_communicator);

    // Set the configuration in R.options and R.constants if you want to change the defaults.
    // Initialize values in R.physical_state:
    //   Set mass, pos, vel for R.num_particles (= num_pn_particles + num_other_particles) entries.
    //   Set spins for the first num_pn_particles.
    ...

    // Optional:
    // If the initial data you just set is far from the center-of-mass frame of the system,
    // you may benefit from shifting the system to its CoM frame using
    double CoM_pos[3], CoM_vel[3];
    ketju_into_CoM_frame(&R, CoM_pos, CoM_vel);
    // to maintain the best numerical resolution for the system coordinates during the integration.
    // CoM_pos and CoM_vel give the original coordinates of the CoM for later processing.

    // Repeat as desired:
    // Run the integration
    ketju_run_integrator(&R, integration_time);
    // Read the resulting state from R.physical_state and process it
    ...

    // Finally free the allocated memory and communicators
    ketju_free_system(&R);
```
You can have multiple independent integrator systems within your program at the same time,
as all of the integrator state is contained within the `ketju_system` struct.

Finally, to build your code that uses the integrator, you need to make some changes to
your build system.
If you're building your code with `cmake`, you can use the following commands
in your `CMakeLists.txt` to add the necessary include and linker flags:  
```cmake
    set(ketju-integrator_DIR "/path/to/ketju-integrator/") 
    find_package(ketju-integrator 1.0 REQUIRED)  
    add_executable(nbody "main.c") # using the nbody program as an example  
    target_link_libraries(nbody ketju-integrator)
```
Otherwise, simply add (assuming gcc/clang)   
`-I/path/to/ketju-integrator/include/`  
to your compiler flags and    
`-L/path/to/ketju-integrator/lib -lketju-integrator`  
to your linker flags.
And naturally you will need to include the necessary MPI flags or use the 
MPI compiler wrapper to compile your code to link the necessary MPI library.



## Python bindings

Simple Python 3 bindings depending only on the standard library and NumPy are
included as the `ketju_integrator` module under `python_bindings`.

They function via a small C library shim and allow parallel integration by
running the integrator in subprocesses with MPI, or by running python directly with MPI.
Examples of using the bindings are given in
`python_bindings/ketju_integrator_example.py` and `python_bindings/ketju_integrator_example_mpi.py`.
After building the library as described above,
you can install install the python bindings with `pip` by running `python3 -m pip install --user ./python_bindings/`
(you can omit `--user` if in a venv), you may need to first upgrade `pip` with
`python3 -m pip install --upgrade pip`.

## Test suite

If you are making changes to the integrator code itself, it is useful to also
build and run the included test suite.
This can be useful also for checking that the code functions correctly on your system.
To build the test suite, run `git submodule update --init`
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
