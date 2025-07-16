cmake_minimum_required(VERSION 3.16.3)
project(ParallelNBodySimulation VERSION 1.0 LANGUAGES CXX)

# Default to a Release build for performance if not specified.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build" FORCE)
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
  src/linear_octree.cpp
  src/morton_keys.cpp
  src/bounding_box.cpp
  src/load_balancing.cpp
  src/exchange.cpp
  src/io.cpp
  src/traversal.cpp
  src/tinyxml2.cpp
)

target_compile_features(solar_system_lib PUBLIC cxx_std_17)
target_include_directories(solar_system_lib PUBLIC include)
target_compile_options(solar_system_lib PUBLIC
  # warnings  (all configs, GCC/Clang)
  $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:
     -Wall -Wextra -Wpedantic -Werror>

  $<$<CXX_COMPILER_ID:GNU,Clang>:-include ${CMAKE_SOURCE_DIR}/include/fix_includes.h>

  # warnings  (all configs, MSVC)
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>



  # performance flags – **only in Release** and only for GCC/Clang
  $<$<AND:$<CONFIG:Release>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>>:
     -march=native -ffast-math>
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

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Configured")