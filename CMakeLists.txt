cmake_minimum_required(VERSION 3.10)
project(SolarSystemSimulation)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(include)

# Source files
set(SRC_FILES
    src/main.cpp
    src/octree.cpp
    src/barnes_hut.cpp
    src/integration.cpp
    src/io.cpp
    src/kepler_to_cartesian.cpp
)

# Add the executable
add_executable(simulate ${SRC_FILES})

# Link any necessary libraries (e.g., filesystem for C++17)
# On some systems, you may need to link stdc++fs for filesystem support
# Uncomment the following line if you get linker errors related to filesystem
# target_link_libraries(simulate stdc++fs)
