cmake_minimum_required(VERSION 3.16.3)
# Enable testing
enable_testing()

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

# Define tests for the project
add_executable(test_main test_main.cpp ../src/main.cpp)
target_link_libraries(test_main PRIVATE solar_system_lib gtest gtest_main)
add_test(NAME test_main COMMAND test_main)

add_executable(test_io test_io.cpp ../src/io.cpp)
target_link_libraries(test_io PRIVATE solar_system_lib gtest gtest_main)
add_test(NAME test_io COMMAND test_io)

add_executable(test_integration test_integration.cpp ../src/integration.cpp)
target_link_libraries(test_integration PRIVATE solar_system_lib gtest gtest_main)
add_test(NAME test_integration COMMAND test_integration)

add_executable(test_octree test_octree.cpp ../src/octree.cpp)
target_link_libraries(test_octree PRIVATE solar_system_lib gtest gtest_main)
add_test(NAME test_octree COMMAND test_octree)

add_executable(test_barnes_hut test_barnes_hut.cpp ../src/barnes_hut.cpp ../src/octree.cpp)
target_link_libraries(test_barnes_hut PRIVATE solar_system_lib gtest gtest_main)
add_test(NAME test_barnes_hut COMMAND test_barnes_hut)

# Include CodeCoverage.cmake
# include(${CMAKE_SOURCE_DIR}/CodeCoverage.cmake)
# include(CodeCoverage.cmake)
# # Append coverage compiler flags
# APPEND_COVERAGE_COMPILER_FLAGS()

# setup_target_for_coverage_lcov(
#     NAME coverage                # Name of the coverage target
#     EXECUTABLE ctest -j 4 # Runs tests using ctest
#     DEPENDENCIES                 # Dependencies (all your test targets)
#         test_main
#         test_io
#         test_integration
#         test_octree
#         test_barnes_hut
#     # Optional: Add EXCLUDE arguments if needed
#     # EXCLUDE "${PROJECT_SOURCE_DIR}/src/dir1/*" "/path/to/my/src/dir2/*"
# )