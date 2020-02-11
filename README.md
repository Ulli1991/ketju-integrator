# Ketju-integrator (MSTAR)

A highly parallel regularized N-body integrator library.

## Building

Run

    mkdir build
    cd build
    cmake .. [CMAKE_OPTIONS]

to configure the build system.
Configuration options can then be changed by editing the `CMakeCache.txt` file
either directly or using e.g. `ccmake`.

After configuration, the library and the standalone driver program can be built
by running `make` in the build directory.
The standalone `nbody` program is found under `bin/`.
*TODO: improve and document usage.*

## Usage in other codes

The public interface of the library is included in the file 
`include/ketju_integrator/ketju_integrator.h`,
so `#include <ketju_integrator/ketju_integrator.h>` and add
`-I/path/to/include/` to your compiler flags.
To link the library, add `-L/path/to/lib -lketju-integrator` to your linker
flags.
*TODO: usage in cmake projects*
