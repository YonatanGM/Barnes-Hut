cmake_minimum_required(VERSION 3.16.3)

project(ParallelNBodySimulation
        VERSION 1.0
        LANGUAGES CXX)

# Set default build type to Release
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

include(${CMAKE_SOURCE_DIR}/test/CodeCoverage.cmake)
option(ENABLE_COVERAGE
       "Enable code coverage (forces Debug build)"
       OFF)

if(ENABLE_COVERAGE)
    message(STATUS "Coverage enabled—forcing Debug + -g --coverage")
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "" FORCE)

    APPEND_COVERAGE_COMPILER_FLAGS()
endif()




# Define the library
set(LIBRARY_SOURCES
    src/octree.cpp
    src/barnes_hut.cpp
    src/integration.cpp
    src/io.cpp
    src/kepler_to_cartesian.cpp
    src/tinyxml2.cpp
)

set(LIBRARY_NAME solar_system_lib)
add_library(${LIBRARY_NAME} STATIC ${LIBRARY_SOURCES})

# Set compiler flags
target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_17)
target_include_directories(${LIBRARY_NAME} PUBLIC include)
target_compile_options(${LIBRARY_NAME} PUBLIC
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Werror>
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
)
target_compile_definitions(${LIBRARY_NAME} PUBLIC _USE_MATH_DEFINES)

# Check for optional and necessary dependencies
if(APPLE)
    # set(CMAKE_C_COMPILER "/usr/local/bin/gcc")
    # set(CMAKE_CXX_COMPILER "/usr/local/bin/g++")

    # OpenMP configuration
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/Cellar/libomp/19.1.4/include")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/Cellar/libomp/19.1.4/include")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/usr/local/Cellar/libomp/19.1.4/lib/libomp.dylib")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/usr/local/Cellar/libomp/19.1.4/lib -lomp")
endif()

find_package(OpenMP REQUIRED)
target_link_libraries(${LIBRARY_NAME} PUBLIC OpenMP::OpenMP_CXX)

find_package(MPI REQUIRED)
target_link_libraries(${LIBRARY_NAME} PUBLIC MPI::MPI_CXX)

# Create the executable
add_executable(simulate src/main.cpp)
target_link_libraries(simulate PRIVATE ${LIBRARY_NAME})

# Add tests
option(WITH_TESTS "Build the tests." ON)
if(WITH_TESTS)
    message(STATUS "Enable tests!")
    enable_testing()
    add_subdirectory(test)
endif()
