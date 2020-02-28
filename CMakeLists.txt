cmake_minimum_required(VERSION 3.0 FATAL_ERROR)


# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Set C++ standard (used for tests)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build (Release, RelWithDebInfo, Debug or MinSizeRel)."
      FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

# Define project
#
project(ketju-integrator C CXX)

set(PROJECT_NAME "Ketju-integrator")
set(PROJECT_COPYRIGHT_YEARS "2019-2020")
set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 1)
set(PROJECT_VERSION_PATCH 0)
set(PROJECT_VERSION_TWEAK -dev)
set(PROJECT_VERSION_STRING "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}${PROJECT_VERSION_TWEAK}")

# Project output paths
#
set(MAINFOLDER ${PROJECT_SOURCE_DIR})
set(EXECUTABLE_OUTPUT_PATH "${MAINFOLDER}/bin")
set(LIBRARY_OUTPUT_PATH "${MAINFOLDER}/lib")

# Project Search Paths
#
LIST (APPEND CMAKE_PREFIX_PATH "${MAINFOLDER}")
LIST (APPEND CMAKE_PREFIX_PATH "${MAINFOLDER}/thirdparty")

#TODO is this useful
# generate appinfo.h
#configure_file("${MAINFOLDER}/include/appinfo.h.in.cmake" 
#               "${MAINFOLDER}/include/appinfo.h")


# get the library sources
# TODO see if this can be done in a better way
file(GLOB lib_headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/ketju_integrator/*.h)
file(GLOB lib_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/*.c)
source_group("" FILES ${lib_headers} ${lib_sources})



# Create library target and add flags we want to enable always

#
# Generate an exportable library target with the required paths etc.
#
add_library(ketju-integrator ${lib_headers} ${lib_sources})

target_include_directories(ketju-integrator
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

target_compile_options(ketju-integrator
    PRIVATE
        $<$<C_COMPILER_ID:GNU>:
            -Wall -Wextra -Wpedantic -Wno-unknown-pragmas
        >
        $<$<OR:$<C_COMPILER_ID:Clang>,$<C_COMPILER_ID:AppleClang>>:
            -Wall -Wextra -Wpedantic -Wno-missing-braces
        >
)

target_link_libraries(ketju-integrator PUBLIC m) # math lib


# Find packages

find_package(MPI REQUIRED)
target_include_directories(ketju-integrator PUBLIC ${MPI_C_INCLUDE_PATH})
target_link_libraries(ketju-integrator PUBLIC ${MPI_C_LIBRARIES})
# Need the linker flags also, target_link_options is only available in very
# new cmake versions, so use this.
# First need to strip extra whitespace in the MPI variable which causes errors
string(STRIP ${MPI_C_LINK_FLAGS} MPI_C_LINK_FLAGS)
target_link_libraries(ketju-integrator PUBLIC ${MPI_C_LINK_FLAGS})

# standalone executable

add_executable(nbody "nbody/main.c")
target_link_libraries(nbody ketju-integrator)



## Generate install cmake files
##
#include(CMakePackageConfigHelpers)
#
#write_basic_package_version_file(
#    "${MAINFOLDER}/ketju-integrator-config-version.cmake"
#    VERSION ${PROJECT_VERSION_STRING}
#    COMPATIBILITY AnyNewerVersion
#)
#
#export(TARGETS ketju-integrator
#    FILE "${MAINFOLDER}/ketju-integrator-targets.cmake"
#)
#
## unit tests if enabled
option(UNIT_TESTS "Build and run unit-tests" ON)
if(UNIT_TESTS)
    add_subdirectory("unit-tests")
endif()


##
## Documentation Target
#
#option(BUILD_DOC "Build Doxygen documentation" ON)
#
#if (BUILD_DOC)
## check if Doxygen is installed
#    find_package(Doxygen)
#    if (DOXYGEN_FOUND)
#        # set input and output files for autogeneration
#        set(DOXYGEN_IN ${MAINFOLDER}/docs/Doxyfile.in)
#        set(DOXYGEN_OUT ${MAINFOLDER}/docs/Doxyfile)
#
#        # generate the Doxyfile
#        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
#
#        # generate mainpage.md
#        configure_file(${MAINFOLDER}/docs/mainpage.md.in
#                       ${MAINFOLDER}/docs/mainpage.md
#                      )
#
#        add_custom_target( doc_doxygen ALL
#                           COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
#                           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#                           COMMENT "Generating API documentation with Doxygen"
#                           VERBATIM )
#    else (DOXYGEN_FOUND)
#        message(
#            "Doxygen not found, skipping generation of documentation")
#    endif (DOXYGEN_FOUND)
#endif (BUILD_DOC)
