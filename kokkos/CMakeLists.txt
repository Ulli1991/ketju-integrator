# C++/Kokkos MSTAR library (GPLv3, see ../LICENSE.md): C++ rewrite of the MSTAR regularised integrator with Kokkos
# support for GPUs, by Ulrich Steinwandel. Built from the top level with -DKETJU_KOKKOS=ON, or standalone
# (cmake -S kokkos). Kokkos is found with find_package(Kokkos); the default execution space is the device one.
#
#   mstar (mstar::mstar)       the library: include/mstar/mstar.hpp, src/
#   mstar-driver, mstar-bench  examples/: standalone driver (KETJU's nbody IC format), scaling benchmark
#   tests (MSTAR_TESTS)        tests/: C vs Kokkos comparisons, the KETJU unit tests incl. GIZMO's additions on the C
#                              library, the GIZMO-addition tests on the Kokkos library, extra forces, embedded tide
cmake_minimum_required(VERSION 3.16)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(mstar-kokkos VERSION 1.0.0 LANGUAGES C CXX)
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    endif()
    include(CTest)
else()
    enable_language(CXX)
    enable_testing()
endif()
# the top level writes binaries into the source tree; keep ours in the build tree
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})

set(MSTAR_VERSION 1.0.0)
find_package(Kokkos REQUIRED)

add_library(mstar ${CMAKE_CURRENT_SOURCE_DIR}/src/engine.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/mergers.cpp)
add_library(mstar::mstar ALIAS mstar)
target_include_directories(mstar
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(mstar PUBLIC Kokkos::kokkos)
target_compile_features(mstar PUBLIC cxx_std_17)
set_target_properties(mstar PROPERTIES POSITION_INDEPENDENT_CODE ON VERSION ${MSTAR_VERSION})

# ---- install: headers, library, CMake package (find_package(mstar) -> mstar::mstar)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(TARGETS mstar EXPORT mstarTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT mstarTargets NAMESPACE mstar:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mstar)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/mstarConfig.cmake.in
     "@PACKAGE_INIT@\ninclude(CMakeFindDependencyMacro)\nfind_dependency(Kokkos)\n"
     "include(\"\${CMAKE_CURRENT_LIST_DIR}/mstarTargets.cmake\")\n")
configure_package_config_file(${CMAKE_CURRENT_BINARY_DIR}/mstarConfig.cmake.in
                              ${CMAKE_CURRENT_BINARY_DIR}/mstarConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mstar)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/mstarConfigVersion.cmake VERSION ${MSTAR_VERSION}
                                 COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mstarConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/mstarConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mstar)

# ---- the C library as a test reference: built here from ../src with the single-task MPI stand-in
set(KETJU_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
enable_language(C)
function(ketju_reference name dir)
    add_library(${name} STATIC ${dir}/src/init.c ${dir}/src/mst.c ${dir}/src/integration.c ${dir}/src/util.c
                ${dir}/src/pn.c ${dir}/src/bh_mergers.c ${dir}/src/debug.c)
    set_target_properties(${name} PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
    target_include_directories(${name} PUBLIC ${dir}/include ${KETJU_ROOT}/thirdparty/serial_mpi)
    target_link_libraries(${name} PUBLIC m)
endfunction()

add_executable(mstar-driver ${CMAKE_CURRENT_SOURCE_DIR}/examples/mstar_driver.cpp)
target_link_libraries(mstar-driver mstar::mstar)

option(MSTAR_TESTS "Build the tests and the benchmark of the Kokkos MSTAR library (needs a C compiler)" ON)
if(MSTAR_TESTS)
    set(T ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    ketju_reference(ketju-ref ${KETJU_ROOT})
    ketju_reference(ketju-extra-force-fwd-ref ${T}/reference/ketju-extra-force-fwd)
    set(_env "OMP_NUM_THREADS=1;OMP_PROC_BIND=false")

    # KETJU's unit tests (unit-tests/, GIZMO's additions included) on the C library, GoogleTest stand-in
    add_executable(ketju-unit-c ${T}/ketju_unit_main.cpp ${KETJU_ROOT}/unit-tests/integration.cpp
                   ${KETJU_ROOT}/unit-tests/mergers.cpp ${KETJU_ROOT}/unit-tests/mst_construction.cpp
                   ${KETJU_ROOT}/unit-tests/gizmo_additions.cpp ${KETJU_ROOT}/unit-tests/softening_kink.cpp)
    target_include_directories(ketju-unit-c PRIVATE ${T}/gtest_shim)
    target_link_libraries(ketju-unit-c ketju-ref)
    target_compile_features(ketju-unit-c PRIVATE cxx_std_17)
    add_test(NAME ketju_unit_c COMMAND ketju-unit-c)

    # the same GIZMO-addition and softening-landing tests on the Kokkos library
    add_executable(mstar-gizmo-additions ${T}/gizmo_additions_kokkos.cpp ${T}/softening_kink_kokkos.cpp)
    target_include_directories(mstar-gizmo-additions PRIVATE ${T}/gtest_shim)
    target_link_libraries(mstar-gizmo-additions mstar::mstar)
    add_test(NAME mstar_gizmo_additions COMMAND mstar-gizmo-additions)
    set_tests_properties(mstar_gizmo_additions PROPERTIES ENVIRONMENT "${_env}")

    # C vs Kokkos: Kepler e = 0.9, Pythagorean, Plummer cores, PN + spin binary, 2-BH PN cusp, BH mergers, collisions,
    # tide, softening
    add_executable(mstar-compare-c ${T}/compare_c.cpp)
    target_link_libraries(mstar-compare-c mstar::mstar ketju-ref)
    add_test(NAME mstar_compare_c COMMAND mstar-compare-c)
    set_tests_properties(mstar_compare_c PROPERTIES ENVIRONMENT "${_env}")

    # extra forces: against the extra-force branch patch forward-ported onto eb96fd0, and independent references
    add_executable(mstar-extra-force ${T}/extra_force.cpp)
    target_link_libraries(mstar-extra-force mstar::mstar ketju-extra-force-fwd-ref)
    target_compile_definitions(mstar-extra-force PRIVATE MSTAR_SAMPLE_IC="${KETJU_ROOT}/nbody/nbody_sample_ic.txt")
    add_test(NAME mstar_extra_force COMMAND mstar-extra-force)
    set_tests_properties(mstar_extra_force PROPERTIES ENVIRONMENT "${_env}")

    # embedded group: impulsive member kicks against c.o.m. kicks + continuous tide (design measurement)
    add_executable(mstar-embedded-tide ${T}/embedded_tide.cpp)
    target_link_libraries(mstar-embedded-tide mstar::mstar)
    add_test(NAME mstar_embedded_tide COMMAND mstar-embedded-tide)
    set_tests_properties(mstar_embedded_tide PROPERTIES ENVIRONMENT "${_env}")

    # the standalone driver on KETJU's sample IC (smoke test)
    add_test(NAME mstar_driver COMMAND mstar-driver ${KETJU_ROOT}/nbody/nbody_sample_ic.txt 1e-6 4)
    set_tests_properties(mstar_driver PROPERTIES ENVIRONMENT "${_env}")

    # scaling benchmark (N_sub 16..4096, C library at small N) and launch latency; not a ctest
    add_executable(mstar-bench ${CMAKE_CURRENT_SOURCE_DIR}/examples/mstar_bench.cpp)
    target_link_libraries(mstar-bench mstar::mstar ketju-ref)
endif()
