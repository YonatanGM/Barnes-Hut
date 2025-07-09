#pragma once

#include "linear_octree.h"
#include "body.h"
#include "bounding_box.h"


void bhAccelerations(
    const OctreeMap&                tree,
    const std::vector<uint64_t>&    key,       // one Morton key per body
    const std::vector<Position>&    pos,       // physical coords (e.g. AU)
    double                          theta,
    double                          G,
    double                          soft2,     // ε²
    const BoundingBox&              global_bb, // from rebalance_bodies
    std::vector<Acceleration>&      out);

// void bhAccelerations(
//     const OctreeMap&                local_tree,
//     const std::vector<NodeRecord>&  remote_nodes,
//     const std::vector<uint64_t>&    key,
//     const std::vector<Position>&    pos,
//     double                          theta,
//     double                          G,
//     double                          soft2,
//     const BoundingBox&              global_bb,
//     std::vector<Acceleration>&      out);

void bhAccelerations_dual_walk(
const OctreeMap&                local_tree,
const std::vector<NodeRecord>&  remote_nodes,
const std::vector<uint64_t>&    key,
const std::vector<Position>&    pos,
double                          theta,
double                          G,
double                          soft2,
const BoundingBox&              global_bb,
std::vector<Acceleration>&      out);