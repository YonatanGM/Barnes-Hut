project(ParallelNBodySimulationTests LANGUAGES C CXX)

cmake_minimum_required(VERSION 3.16.3)
# Enable testing
enable_testing()

# Include CMake’s GoogleTest helper module for discovery
include(GoogleTest)

# Always fetch and integrate Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        main
)
# Set Google Test options to avoid unnecessary installations
set(BUILD_GMOCK OFF CACHE BOOL "Build Google Mock" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "Disable install for GoogleTest" FORCE)

FetchContent_MakeAvailable(googletest)

# Include directories for the code under test
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)

# Define tests for the project
add_executable(test_parse_time test_parse_time.cpp)
target_link_libraries(test_parse_time PRIVATE gtest gtest_main)
gtest_discover_tests(test_parse_time)

add_executable(test_io test_io.cpp ../src/io.cpp)
target_link_libraries(test_io PRIVATE solar_system_lib gtest gtest_main)
gtest_discover_tests(test_io)

add_executable(test_integration test_integration.cpp ../src/integration.cpp)
target_link_libraries(test_integration PRIVATE solar_system_lib gtest gtest_main)
gtest_discover_tests(test_integration)

add_executable(test_octree test_octree.cpp ../src/octree.cpp)
target_link_libraries(test_octree PRIVATE solar_system_lib gtest gtest_main)
gtest_discover_tests(test_octree)

add_executable(test_barnes_hut test_barnes_hut.cpp ../src/barnes_hut.cpp ../src/octree.cpp)
target_link_libraries(test_barnes_hut PRIVATE solar_system_lib gtest gtest_main)
gtest_discover_tests(test_barnes_hut)


if(ENABLE_COVERAGE)
  setup_target_for_coverage_lcov(
      NAME coverage
      EXECUTABLE ctest --output-on-failure || true
      DEPENDENCIES
          test_parse_time
          test_io
          test_integration
          test_octree
          test_barnes_hut

      LCOV_ARGS "--ignore-errors" "inconsistent,unused,mismatch"

      EXCLUDE
        "${CMAKE_BINARY_DIR}/*"  # Exclude build directory
        "${CMAKE_BINARY_DIR}/_deps/*"  # Exclude _deps directory
        "/usr/*"                        # Exclude system headers
        "${CMAKE_SOURCE_DIR}/test/*"
        "${CMAKE_SOURCE_DIR}/include/*"  # exclude include directory
  )
endif()
