# C++/Kokkos MSTAR library (GPLv3, see LICENSE.md): C++ rewrite of the MSTAR regularised integrator with Kokkos
# support for GPUs, by Ulrich Steinwandel. Kokkos is found with find_package(Kokkos); the default execution space is
# the device one. The C library is on branch ketju-c.
#
#   mstar (mstar::mstar)       the library: include/mstar/mstar.hpp, src/
#   mstar-driver, mstar-bench  examples/: standalone driver (KETJU's nbody IC format), scaling benchmark
#   tests (MSTAR_TESTS)        tests/: comparisons with the C library's stored results (tests/reference_data), KETJU's
#                              unit tests and GIZMO's additions, extra forces, embedded tide
cmake_minimum_required(VERSION 3.16)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(mstar VERSION 1.0.0 LANGUAGES CXX)
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    endif()
    include(CTest)
else()
    enable_language(CXX)
    enable_testing()
endif()
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

add_executable(mstar-driver ${CMAKE_CURRENT_SOURCE_DIR}/examples/mstar_driver.cpp)
target_link_libraries(mstar-driver mstar::mstar)
add_executable(mstar-bench ${CMAKE_CURRENT_SOURCE_DIR}/examples/mstar_bench.cpp)
target_link_libraries(mstar-bench mstar::mstar)

option(MSTAR_TESTS "Build the tests of the MSTAR library" ON)
if(MSTAR_TESTS)
    set(T ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    set(_env "OMP_NUM_THREADS=1;OMP_PROC_BIND=false")
    # results of the C library (branch ketju-c) that the comparison tests read
    set(MSTAR_REF_DEFS MSTAR_REF_DATA="${T}/reference_data")

    # KETJU's unit tests (integration, mergers, MST) and GIZMO's additions (contact collisions, collision energy with
    # tide and softening, pericentre guard, tide, collision_radius, softening kink) on a GoogleTest stand-in
    add_executable(mstar-unit-tests ${T}/gizmo_additions_kokkos.cpp ${T}/softening_kink_kokkos.cpp ${T}/unit_tests_kokkos.cpp)
    target_include_directories(mstar-unit-tests PRIVATE ${T}/gtest_shim)
    target_link_libraries(mstar-unit-tests mstar::mstar)
    add_test(NAME mstar_unit_tests COMMAND mstar-unit-tests)
    set_tests_properties(mstar_unit_tests PROPERTIES ENVIRONMENT "${_env}")

    # against the C library's stored results: Kepler e = 0.9, Pythagorean, Plummer cores, PN + spin binary, 2-BH PN
    # cusp, BH mergers, collisions, tide, softening
    add_executable(mstar-compare-c ${T}/compare_c.cpp)
    target_include_directories(mstar-compare-c PRIVATE ${T})
    target_compile_definitions(mstar-compare-c PRIVATE ${MSTAR_REF_DEFS})
    target_link_libraries(mstar-compare-c mstar::mstar)
    add_test(NAME mstar_compare_c COMMAND mstar-compare-c)
    set_tests_properties(mstar_compare_c PROPERTIES ENVIRONMENT "${_env}")

    # extra forces: against the stored extra-force branch results (patch forward-ported onto eb96fd0), and
    # independent references
    add_executable(mstar-extra-force ${T}/extra_force.cpp)
    target_include_directories(mstar-extra-force PRIVATE ${T})
    target_link_libraries(mstar-extra-force mstar::mstar)
    target_compile_definitions(mstar-extra-force PRIVATE ${MSTAR_REF_DEFS}
                               MSTAR_SAMPLE_IC="${CMAKE_CURRENT_SOURCE_DIR}/examples/nbody_sample_ic.txt")
    add_test(NAME mstar_extra_force COMMAND mstar-extra-force)
    set_tests_properties(mstar_extra_force PROPERTIES ENVIRONMENT "${_env}")

    # embedded group: impulsive member kicks against c.o.m. kicks + continuous tide (design measurement)
    add_executable(mstar-embedded-tide ${T}/embedded_tide.cpp)
    target_link_libraries(mstar-embedded-tide mstar::mstar)
    add_test(NAME mstar_embedded_tide COMMAND mstar-embedded-tide)
    set_tests_properties(mstar_embedded_tide PROPERTIES ENVIRONMENT "${_env}")

    # the standalone driver on KETJU's sample IC (smoke test)
    add_test(NAME mstar_driver COMMAND mstar-driver ${CMAKE_CURRENT_SOURCE_DIR}/examples/nbody_sample_ic.txt 1e-6 4)
    set_tests_properties(mstar_driver PROPERTIES ENVIRONMENT "${_env}")
endif()
