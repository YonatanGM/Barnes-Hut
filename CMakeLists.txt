cmake_minimum_required(VERSION 3.16.3)
project(ParallelNBodySimulation VERSION 1.0 LANGUAGES CXX)

# default to Release build if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

# macOS OpenMP flags
if(APPLE)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/Cellar/libomp/19.1.4/include")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/Cellar/libomp/19.1.4/include")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/usr/local/Cellar/libomp/19.1.4/lib/libomp.dylib")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/usr/local/Cellar/libomp/19.1.4/lib -lomp")
endif()

# core library
add_library(solar_system_lib STATIC
  src/octree.cpp
  src/barnes_hut.cpp
  src/integration.cpp
  src/io.cpp
  src/kepler_to_cartesian.cpp
  src/tinyxml2.cpp
)
target_compile_features(solar_system_lib PUBLIC cxx_std_17)
target_include_directories(solar_system_lib PUBLIC include)
target_compile_options(solar_system_lib PUBLIC
  $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wall -Wextra -Wpedantic -Werror>
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
)
target_compile_definitions(solar_system_lib PUBLIC _USE_MATH_DEFINES)

# OpenMP and MPI
find_package(OpenMP REQUIRED)
target_link_libraries(solar_system_lib PUBLIC OpenMP::OpenMP_CXX)

find_package(MPI REQUIRED)
target_link_libraries(solar_system_lib PUBLIC MPI::MPI_CXX)

# main executable
add_executable(simulate src/main.cpp)
target_link_libraries(simulate PRIVATE solar_system_lib)

# test and coverage options
option(WITH_TESTS      "Build and run unit tests"      ON)
option(ENABLE_COVERAGE "Enable code coverage reporting" OFF)

# coverage setup (forces Debug build)
if(ENABLE_COVERAGE)
  message(STATUS "Coverage enabled — forcing Debug + coverage flags")
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "" FORCE)
  include(${CMAKE_SOURCE_DIR}/test/CodeCoverage.cmake)
  APPEND_COVERAGE_COMPILER_FLAGS()
endif()

# tests
if(WITH_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()


