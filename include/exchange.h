#pragma once
#include <vector>
#include <mpi.h>

#include "policy.h"
#include "linear_octree.h"
#include "bounding_box.h"



void exchange_whole_trees(const OctreeMap &local_tree, OctreeMap &full_tree,
                          MPI_Datatype node_type, int rank, int size);


std::vector<NodeRecord> generate_interaction_list(
        const OctreeMap&                      tree,
        const std::vector<BoundingBox>&       remote_boxes,   // all buckets of dest rank
        const BoundingBox&                    global_bb,
        double                                theta2);


void exchange_LET_gather_remotes(
    const OctreeMap&                        local_tree,
    std::vector<NodeRecord>&                remote_nodes, // Output
    std::vector<int>&                       recv_counts, //output
    const BoundingBox&                      global_bb,
    double                                  theta,
    MPI_Datatype                            node_type,
    int                                     rank,
    int                                     size,
    const std::vector<std::vector<uint64_t>>& rank_domain_keys);

// std::vector<NodeRecord> generate_interaction_list(
//     const OctreeMap& local_tree,
//     const std::vector<Position>& remote_bucket_positions,
//     const BoundingBox& global_bb,
//     double theta);
