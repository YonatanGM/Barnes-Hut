#include "exchange.h"
#include <numeric>
#include <stack>
#include <iostream>

static inline uint64_t mortonPrefix(uint64_t code, int depth) {
    if (depth == 0) return 0;
    // FIX: The shift was off by one. To keep the top 3*depth bits of a 64-bit
    // integer, the shift amount must be 64 - 3*depth. The previous value of
    // 63 - 3*depth was keeping one extra bit, leading to incorrect prefixes.
    int shift = 63 - 3 * depth;
    return code & (~0ULL << shift);
}

void exchange_whole_trees(const OctreeMap &local_tree, OctreeMap &full_tree,
                          MPI_Datatype node_type, int /*rank*/, int size) {
    std::vector<NodeRecord> send_buf = flattenTree(local_tree);
    int send_cnt = static_cast<int>(send_buf.size());

    std::vector<int> recv_counts(size);
    MPI_Allgather(&send_cnt, 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
                  MPI_COMM_WORLD);

    std::vector<int> displs(size, 0);
    for (int i = 1; i < size; ++i) displs[i] = displs[i - 1] + recv_counts[i - 1];
    int total_nodes = displs.back() + recv_counts.back();

    std::vector<NodeRecord> recv_buf(total_nodes);
    MPI_Allgatherv(send_buf.data(), send_cnt, node_type,
                   recv_buf.data(), recv_counts.data(), displs.data(), node_type,
                   MPI_COMM_WORLD);

    full_tree.clear();
    mergeIntoTree(full_tree, recv_buf);
}

/**
 * @brief Exchanges essential tree data using a sender-initiated ("push") LET protocol.
 * This function populates the `remote_nodes` vector with the essential nodes
 * received from all other MPI ranks.
 */
void exchange_LET(const OctreeMap& local_tree, std::vector<NodeRecord>& remote_nodes,
                  const BoundingBox& local_bb, const BoundingBox& global_bb,
                  double theta, MPI_Datatype node_type, MPI_Datatype bb_type, int rank, int size) {

    // 1. Exchange Domain Bounding Boxes
    std::vector<BoundingBox> all_domain_bbs(size);
    MPI_Allgather(&local_bb, 1, bb_type,
                  all_domain_bbs.data(), 1, bb_type, MPI_COMM_WORLD);

    // 2. Generate Interaction Lists (Local Computation)
    std::vector<std::vector<NodeRecord>> send_lists(size);
    double theta_sq = theta * theta;
    for (int j = 0; j < size; ++j) {
        if (j == rank) continue;
        send_lists[j] = generate_interaction_list(local_tree, all_domain_bbs[j], global_bb, theta_sq);
    }

    // 3. Exchange Interaction Counts
    std::vector<int> send_counts(size);
    for (int i = 0; i < size; ++i) {
        send_counts[i] = send_lists[i].size();
    }
    std::vector<int> recv_counts(size);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // 4. Exchange LET Data with MPI_Alltoallv
    std::vector<NodeRecord> send_buf;
    std::vector<int> sdispls(size, 0);
    int total_to_send = 0;
    for (int i = 0; i < size; ++i) {
        sdispls[i] = total_to_send;
        total_to_send += send_counts[i];
    }
    send_buf.resize(total_to_send);
    for(int i = 0; i < size; ++i) {
        if(send_counts[i] > 0) {
            std::copy(send_lists[i].begin(), send_lists[i].end(), send_buf.begin() + sdispls[i]);
        }
    }

    std::vector<int> rdispls(size, 0);
    int total_to_recv = 0;
    for (int i = 0; i < size; ++i) {
        rdispls[i] = total_to_recv;
        total_to_recv += recv_counts[i];
    }

    // Resize the output parameter directly to serve as the receive buffer.
    remote_nodes.resize(total_to_recv);

    MPI_Alltoallv(send_buf.data(), send_counts.data(), sdispls.data(), node_type,
                  remote_nodes.data(), recv_counts.data(), rdispls.data(), node_type,
                  MPI_COMM_WORLD);
}

/**
 * @brief Exchanges essential tree data using a sender-initiated ("push") LET protocol.
 */
void exchange_LET(const OctreeMap& local_tree, OctreeMap& full_tree,
                  const BoundingBox& local_bb, const BoundingBox& global_bb,
                  double theta, MPI_Datatype node_type, MPI_Datatype bb_type, int rank, int size) {

    // 1. Exchange Domain Bounding Boxes
    std::vector<BoundingBox> all_domain_bbs(size);

    // --- DEBUG PRINT ---
    // std::cout << "[Rank " << rank << "] Starting MPI_Allgather for bounding boxes." << std::endl;
    // MPI_Barrier(MPI_COMM_WORLD); // Synchronize before the call for cleaner logs

    MPI_Allgather(&local_bb, 1, bb_type, all_domain_bbs.data(), 1, bb_type, MPI_COMM_WORLD);
    // --- DEBUG PRINT ---
    // MPI_Barrier(MPI_COMM_WORLD); // Synchronize after the call
    // std::cout << "[Rank " << rank << "] Finished MPI_Allgather." << std::endl;

    // 2. Generate Interaction Lists (Local Computation)
    std::vector<std::vector<NodeRecord>> send_lists(size);
    double theta_sq = theta * theta;

    for (int j = 0; j < size; ++j) {
        if (j == rank) continue;
        send_lists[j] = generate_interaction_list(local_tree, all_domain_bbs[j], global_bb, theta_sq);
    }

    // 3. Exchange Interaction Counts
    std::vector<int> send_counts(size);
    for (int i = 0; i < size; ++i) {
        send_counts[i] = send_lists[i].size();
    }
    std::vector<int> recv_counts(size);

    // --- DEBUG PRINT ---
    // std::cout << "[Rank " << rank << "] Starting MPI_Alltoall for counts." << std::endl;
    // MPI_Barrier(MPI_COMM_WORLD);

    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // --- DEBUG PRINT ---
    // MPI_Barrier(MPI_COMM_WORLD);
    // std::cout << "[Rank " << rank << "] Finished MPI_Alltoall." << std::endl;


    // 4. Exchange LET Data with MPI_Alltoallv
    std::vector<NodeRecord> send_buf;
    std::vector<int> sdispls(size, 0);
    int total_to_send = 0;
    for (int i = 0; i < size; ++i) {
        sdispls[i] = total_to_send;
        total_to_send += send_counts[i];
    }
    send_buf.resize(total_to_send);
    for(int i = 0; i < size; ++i) {
        if(send_counts[i] > 0) {
            std::copy(send_lists[i].begin(), send_lists[i].end(), send_buf.begin() + sdispls[i]);
        }
    }

    std::vector<int> rdispls(size, 0);
    int total_to_recv = 0;
    for (int i = 0; i < size; ++i) {
        rdispls[i] = total_to_recv;
        total_to_recv += recv_counts[i];
    }
    std::vector<NodeRecord> recv_buf(total_to_recv);

    // --- DEBUG PRINT ---
    // std::cout << "[Rank " << rank << "] Starting MPI_Alltoallv. Sending " << total_to_send
    //           << " nodes, receiving " << total_to_recv << " nodes." << std::endl;
    // MPI_Barrier(MPI_COMM_WORLD);

    MPI_Alltoallv(send_buf.data(), send_counts.data(), sdispls.data(), node_type,
                  recv_buf.data(), recv_counts.data(), rdispls.data(), node_type,
                  MPI_COMM_WORLD);

    // --- DEBUG PRINT ---
    // MPI_Barrier(MPI_COMM_WORLD);
    // std::cout << "[Rank " << rank << "] Finished MPI_Alltoallv." << std::endl;

    // Construct the full tree by merging the local tree and received nodes
    full_tree = local_tree;
    mergeIntoTree(full_tree, recv_buf);
}

// Generates the list of essential nodes for a remote domain
/**
 * @brief (Internal) Generates the list of essential nodes for a remote domain.
 *
 * This function implements the canonical "path-to-root" export. When a node
 * is found to be essential, it and its entire chain of ancestors are added
 * to the export list. This ensures the receiver can reconstruct a fully
 * connected tree segment.
 */
std::vector<NodeRecord>
generate_interaction_list(const OctreeMap& local_tree,
                          const BoundingBox& remote_bb,
                          const BoundingBox& global_bb,
                          double theta_sq) {

    // Use a hash set to store the unique keys of essential nodes and their ancestors.
    robin_hood::unordered_set<OctreeKey, OctreeKeyHash> essential_keys;
    if (local_tree.empty()) return {};

    std::stack<OctreeKey> stack;
    // Start traversal from the root, if it exists.
    if (local_tree.count({0, 0})) {
        stack.push({0, 0});
    }

    while (!stack.empty()) {
        OctreeKey key = stack.top();
        stack.pop();

        auto it = local_tree.find(key);
        if (it == local_tree.end()) continue;

        BoundingBox node_bb = key_to_bounding_box(key, global_bb);
        double s = node_bb.max.x - node_bb.min.x;
        double d_sq = min_distance_sq(node_bb, remote_bb);

        if (s * s < theta_sq * d_sq) {
            // This node is essential. Add it AND all its ancestors to the set.
            OctreeKey current_key = key;
            while (true) {
                // The set automatically handles duplicates.
                essential_keys.insert(current_key);
                if (current_key.depth == 0) break;

                uint8_t p_dep = current_key.depth - 1;
                uint64_t p_prefix = mortonPrefix(current_key.prefix, p_dep);
                current_key = {p_prefix, p_dep};
            }
        } else {
            // Node is too close, open it and check its children.
            if (key.depth < 21) {
                for (int octant = 0; octant < 8; ++octant) {
                    uint64_t child_prefix = key.prefix | (static_cast<uint64_t>(octant) << (63 - 3 * (key.depth + 1)));
                    OctreeKey child_key = {child_prefix, (uint8_t)(key.depth + 1)};
                    if (local_tree.count(child_key)) {
                        stack.push(child_key);
                    }
                }
            }
        }
    }

    // Convert the set of unique keys into a vector of NodeRecords to be sent.
    std::vector<NodeRecord> essential_nodes;
    essential_nodes.reserve(essential_keys.size());
    for (const auto& key : essential_keys) {
        auto it = local_tree.find(key);
        if (it != local_tree.end()) {
            const auto& node = it->second;
            essential_nodes.push_back({key.prefix, key.depth, node.mass, node.comX, node.comY, node.comZ});
        }
    }
    return essential_nodes;
}

void recompute_ancestor_moments(OctreeMap& tree, const std::vector<NodeRecord>& merged_nodes) {
    if (merged_nodes.empty()) return;

    // 1. Collect all unique parents of the nodes that were just merged.
    robin_hood::unordered_set<OctreeKey, OctreeKeyHash> parents_to_update;
    for (const auto& rec : merged_nodes) {
        if (rec.depth > 0) {
            uint8_t p_dep = rec.depth - 1;
            uint64_t p_prefix = mortonPrefix(rec.prefix, p_dep);
            parents_to_update.insert({p_prefix, p_dep});
        }
    }

    // 2. Create a list of all ancestors that need updating, up to the root.
    robin_hood::unordered_set<OctreeKey, OctreeKeyHash> all_ancestors = parents_to_update;
    for (const auto& key : parents_to_update) {
        OctreeKey current_key = key;
        while (current_key.depth > 0) {
            uint8_t p_dep = current_key.depth - 1;
            uint64_t p_prefix = mortonPrefix(current_key.prefix, p_dep);
            current_key = {p_prefix, p_dep};
            all_ancestors.insert(current_key);
        }
    }

    // 3. Process the ancestors from bottom to top to ensure correctness.
    std::vector<OctreeKey> sorted_ancestors(all_ancestors.begin(), all_ancestors.end());
    std::sort(sorted_ancestors.begin(), sorted_ancestors.end(), [](const auto& a, const auto& b) {
        return a.depth > b.depth; // Start with the deepest nodes first
    });

    for (const auto& p_key : sorted_ancestors) {
        auto it = tree.find(p_key);
        if (it == tree.end()) continue;

        OctreeNode& parent_node = it->second;
        // Reset parent moments before re-calculating
        parent_node.mass = 0.0;
        parent_node.comX = 0.0;
        parent_node.comY = 0.0;
        parent_node.comZ = 0.0;

        // Sum moments from all 8 potential children
        uint8_t c_dep = p_key.depth + 1;
        for (int octant = 0; octant < 8; ++octant) {
            uint64_t c_prefix = p_key.prefix | (static_cast<uint64_t>(octant) << (63 - 3 * c_dep));
            auto child_it = tree.find({c_prefix, c_dep});
            if (child_it != tree.end() && child_it->second.mass > 0) {
                const OctreeNode& child_node = child_it->second;
                parent_node.mass += child_node.mass;
                parent_node.comX += child_node.comX * child_node.mass; // Use weighted sum
                parent_node.comY += child_node.comY * child_node.mass;
                parent_node.comZ += child_node.comZ * child_node.mass;
            }
        }

        // Finalize the parent's Center of Mass
        if (parent_node.mass > 0) {
            parent_node.comX /= parent_node.mass;
            parent_node.comY /= parent_node.mass;
            parent_node.comZ /= parent_node.mass;
        }
    }
}

// std::vector<NodeRecord>
// generate_interaction_list(const OctreeMap& local_tree,
//                           const BoundingBox& remote_bb,
//                           const BoundingBox& global_bb,
//                           double theta_sq) {
//     std::vector<NodeRecord> essential_nodes;
//     if (local_tree.empty()) return essential_nodes;

//     std::stack<OctreeKey> stack;
//     // Start traversal at the root node {prefix=0, depth=0}
//     stack.push({0, 0});

//     while (!stack.empty()) {
//         OctreeKey key = stack.top();
//         stack.pop();

//         auto it = local_tree.find(key);
//         if (it == local_tree.end()) continue; // Should not happen if we start from an existing root

//         const OctreeNode& node = it->second;
//         BoundingBox node_bb = key_to_bounding_box(key, global_bb);
//         double s = node_bb.max.x - node_bb.min.x; // Side length
//         double d_sq = min_distance_sq(node_bb, remote_bb);

//         // Group multipole-acceptance criterion: s^2 / d^2 < theta^2
//         if (s * s < theta_sq * d_sq) {
//             // Node is far enough away, add it to the list and prune the walk
//             essential_nodes.push_back({key.prefix, key.depth, node.mass, node.comX, node.comY, node.comZ});
//         } else {
//             // Node is too close, open it by pushing its children onto the stack
//             if (key.depth < 21) {
//                 for (int octant = 0; octant < 8; ++octant) {
//                     uint64_t child_prefix = key.prefix | (static_cast<uint64_t>(octant) << (63 - 3 * (key.depth + 1)));
//                     OctreeKey child_key = {child_prefix, (uint8_t)(key.depth + 1)};
//                     if (local_tree.count(child_key)) {
//                         stack.push(child_key);
//                     }
//                 }
//             }
//         }
//     }
//     return essential_nodes;
// }
