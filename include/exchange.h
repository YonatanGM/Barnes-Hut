#pragma once
#include <vector>
#include <mpi.h>

#include "policy.h"
#include "linear_octree.h"
#include "bounding_box.h"


/* ------------ TREES (broadcast every node) ----------------------- */

void exchange_whole_trees(const OctreeMap &local_tree, OctreeMap &full_tree,
                          MPI_Datatype node_type, int rank, int size);


/**
 * @brief Exchanges essential tree data using a sender-initiated ("push") LET protocol.
 */
void exchange_LET(const OctreeMap& local_tree, OctreeMap& full_tree,
                  const BoundingBox& local_bb, const BoundingBox& global_bb,
                  double theta, MPI_Datatype node_type,  MPI_Datatype bb_type, int rank, int size);

void exchange_LET(const OctreeMap& local_tree,  std::vector<NodeRecord>& remote_nodes,
                  const BoundingBox& local_bb, const BoundingBox& global_bb,
                  double theta, MPI_Datatype node_type,  MPI_Datatype bb_type, int rank, int size);

void exchange_LET(
    const OctreeMap& local_tree,
    OctreeMap& full_tree,
    const BoundingBox& global_bb,
    double theta,
    MPI_Datatype node_type,
    int rank,
    int size,
    const std::vector<std::vector<uint64_t>>& rank_domain_keys);


std::vector<NodeRecord> generate_interaction_list(
    const OctreeMap& local_tree,
    const std::vector<Position>& remote_bucket_positions,
    const BoundingBox& global_bb,
    double theta);
/**
 * @brief (Internal) Generates the list of essential nodes for a remote domain.
 */
std::vector<NodeRecord>
generate_interaction_list(const OctreeMap& local_tree,
                          const BoundingBox& remote_bb,
                          const BoundingBox& global_bb,
                          double theta_sq);

std::vector<NodeRecord> generate_interaction_list(
        const OctreeMap&                      tree,
        const std::vector<BoundingBox>&       remote_boxes,   // all buckets of dest rank
        const BoundingBox&                    global_bb,
        double                                theta2);

void recompute_ancestor_moments(OctreeMap& tree, const std::vector<NodeRecord>& merged_nodes);
void full_recompute_moments(OctreeMap &tree);


void exchange_LET_gather_remotes(
    const OctreeMap&                        local_tree,
    std::vector<NodeRecord>&                remote_nodes, // Output
    const BoundingBox&                      global_bb,
    double                                  theta,
    MPI_Datatype                            node_type,
    int                                     rank,
    int                                     size,
    const std::vector<std::vector<uint64_t>>& rank_domain_keys);