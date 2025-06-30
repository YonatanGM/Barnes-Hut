#pragma once

#include <vector>
#include <string>
#include "body.h"
#include "tinyxml2.h"
#include "cxxopts.hpp" 


// Reads body data (mass, position, velocity) from a specified CSV file.
// Returns true on success, false on failure.
bool readCSV(
    const std::string& filename,
    std::vector<uint64_t>& ids,
    std::vector<double>& masses,
    std::vector<Position>& positions,
    std::vector<Velocity>& velocities,
    int body_count = -1 // limit on the number of bodies to read
);

void writeSnapshot(int rank, int vs_counter,
                   const std::vector<uint64_t>& local_ids,
                   const std::vector<double>& local_masses,
                   const std::vector<Position>& local_positions,
                   const std::vector<Velocity>& local_velocities,
                   const std::vector<Acceleration>& local_accelerations,
                   const std::string& vs_dir);

void updatePVDFile(const cxxopts::ParseResult& args,
                   int size,
                   int vs_counter,
                   double current_time,
                   const std::string& vs_dir);
