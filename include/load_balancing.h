#pragma once

#include "body.h"
#include "morton_keys.h"
#include <vector>


// void rebalance_bodies(
//     int rank, int size,
//     const CodesAndNorm &cn,
//     std::vector<Position> &local_pos,
//     std::vector<double> &local_mass,
//     std::vector<Velocity> &local_vel,
//     std::vector<uint64_t> &local_ids);


void rebalance_bodies(
    int rank, int size,
    const CodesAndNorm &cn,
    std::vector<Position> &local_pos,
    std::vector<double> &local_mass,
    std::vector<Velocity> &local_vel,
    std::vector<uint64_t> &local_ids,
    std::vector<std::vector<uint64_t>>& rank_domain_keys);