#pragma once

#include <vector>
#include <string>
#include "body.h"
#include "tinyxml2.h"
#include "cxxopts.hpp"
#include "linear_octree.h"
#include "bounding_box.h"

// Reads body data (mass, position, velocity) from a specified CSV file
// Returns true on success, false on failure
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

void writeReceivedLETs(int rank, int vis_step,
                       const std::vector<NodeRecord>& received_nodes,
                       const std::vector<int>& recv_counts,
                       const BoundingBox& global_bb,
                       const std::string& vs_dir);

void updateReceivedLETPVDFile(const cxxopts::ParseResult& args,
                              int size,
                              int vs_counter,
                              double current_time,
                              const std::string& vs_dir);

void writeHistogram(int vis_step,
                    const std::vector<std::pair<long long, int>>& hist,
                    const BoundingBox& global_bb,
                    const std::string& out_dir);

void updateHistogramPVDFile(const cxxopts::ParseResult& args,
                            int vs_counter,
                            double current_time,
                            const std::string& out_dir);