#include "exchange.h"
#include "morton_keys.h"
#include <numeric>
#include <stack>
#include <iostream>



void exchangeFullTrees(const OctreeMap &local_tree, OctreeMap &full_tree,
                          MPI_Datatype node_type, int /*rank*/, int size) {
    std::vector<NodeRecord> send_buf = serializeTreeToRecords(local_tree);
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
    mergeRecordsIntoTree(full_tree, recv_buf);
}


void exchangeEssentialTrees(
    const OctreeMap&                        local_tree,
    std::vector<NodeRecord>&                remote_nodes,
    std::vector<int>&                       recv_counts,
    const BoundingBox&                      global_bb,
    double                                  theta,
    MPI_Datatype                            node_type,
    int                                     rank,
    int                                     size,
    const std::vector<std::vector<uint64_t>>& rank_domain_keys,
    int                                     max_traversal_depth,
    int                                     bucket_bits)
{

    int bucket_depth = bucket_bits/3;

    // build per rank bucket AABBs with correct anisotropic dimensions
    std::vector<std::vector<BoundingBox>> bucket_boxes(size);
    double dx   = global_bb.max.x - global_bb.min.x;
    double dy   = global_bb.max.y - global_bb.min.y;
    double dz   = global_bb.max.z - global_bb.min.z;

    double cellX = dx / (1 << bucket_depth);
    double cellY = dy / (1 << bucket_depth);
    double cellZ = dz / (1 << bucket_depth);

    for(int r=0; r<size; ++r){
      bucket_boxes[r].reserve(rank_domain_keys[r].size());
      for(auto pref: rank_domain_keys[r]){
        Position n = key_to_normalized_position(pref, bucket_depth);
        BoundingBox bb;
        bb.min.x = global_bb.min.x + n.x * dx;
        bb.min.y = global_bb.min.y + n.y * dy;
        bb.min.z = global_bb.min.z + n.z * dz;
        bb.max.x = bb.min.x + cellX;
        bb.max.y = bb.min.y + cellY;
        bb.max.z = bb.min.z + cellZ;
        bucket_boxes[r].push_back(bb);
      }
    }

    // Generate per destination send lists using the corrected interaction list function
    std::vector<std::vector<NodeRecord>> send_lists(size);
    double theta2 = theta*theta;
    for(int dest=0; dest<size; ++dest){
        if(dest==rank) continue;
        send_lists[dest] = createInteractionListForRank(local_tree, bucket_boxes[dest], global_bb, theta2, max_traversal_depth);
    }

    // Perform MPI_Alltoallv to exchange LET data
    std::vector<int> send_counts(size);
    recv_counts.resize(size);

    for(int r=0; r<size; ++r) send_counts[r] = int(send_lists[r].size());
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> sdispls(size,0), rdispls(size,0);
    int total_to_send = 0;
    int total_to_recv = 0;

    for(int i=0; i<size; ++i) {
        sdispls[i] = total_to_send;
        rdispls[i] = total_to_recv;
        total_to_send += send_counts[i];
        total_to_recv += recv_counts[i];
    }

    std::vector<NodeRecord> send_buffer(total_to_send);
    for(int i=0; i<size; ++i) {
      if(send_counts[i] > 0) {
        std::copy(send_lists[i].begin(), send_lists[i].end(), send_buffer.begin() + sdispls[i]);
      }
    }

    // Resize the output vector to exactly hold the incoming data
    remote_nodes.resize(total_to_recv);

    MPI_Alltoallv(send_buffer.data(), send_counts.data(), sdispls.data(), node_type,
                  remote_nodes.data(), recv_counts.data(), rdispls.data(), node_type,
                  MPI_COMM_WORLD);

}


std::vector<NodeRecord> createInteractionListForRank(
    const OctreeMap&                      tree,
    const std::vector<BoundingBox>&       remote_boxes,
    const BoundingBox&                    global_bb,
    double                                theta2,
    int                                   max_traversal_depth)
{
    if (tree.empty() || remote_boxes.empty())
        return {};

    robin_hood::unordered_set<OctreeKey, OctreeKeyHash> keep;

    //  256 seems safe for depth ≤ 21 (7*d+1 bound)
    std::array<OctreeKey, 256> stk;
    int top = 0;
    if (tree.count({0ULL, 0}))
        stk[top++] = {0ULL, 0}; // push root

    while (top > 0) {
        OctreeKey k = stk[--top];
        BoundingBox node_bb = getBoundingBoxForCell(k, global_bb);

        // Barnes Hut opening test
        double sx = node_bb.max.x - node_bb.min.x;
        double sy = node_bb.max.y - node_bb.min.y;
        double sz = node_bb.max.z - node_bb.min.z;
        double s  = std::max({sx, sy, sz});
        double s2 = s * s;                // largest side squared

        bool bhFails = false;
        for (const auto& bucket : remote_boxes) {
            double d2 = min_distance_sq(node_bb, bucket);
            if (s2 >= theta2 * d2) {      // Barnes–Hut criterion
                bhFails = true;
                break;
            }
        }

        if (bhFails && k.depth < max_traversal_depth) {
            // criterion failed  and we are still allowed to descend
            uint8_t cd = k.depth + 1;
            uint64_t stride = 1ULL << (63 - 3 * cd);
            for (uint8_t oc = 0; oc < 8; ++oc) {
                uint64_t child_prefix = k.prefix | (static_cast<uint64_t>(oc) * stride);
                OctreeKey ch{child_prefix, cd};
                if (tree.count(ch) && top < 256) {
                    stk[top++] = ch;
                }
            }
        } else {
            // Accept, either BH test passed or we hit max_traversal_depth.
            keep.insert(k);
        }
    }

    std::vector<NodeRecord> out;
    out.reserve(keep.size());
    for (const auto& k : keep) {
        const auto& n = tree.at(k);
        out.push_back({k.prefix, k.depth, n.mass, n.comX, n.comY, n.comZ});
    }
    return out;
}

