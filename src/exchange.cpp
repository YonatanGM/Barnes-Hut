#include "exchange.h"
#include "morton_keys.h"
#include <numeric>
#include <stack>
#include <iostream>


static inline uint64_t mortonPrefix(uint64_t code, int depth) {
    if (depth == 0) return 0;
    int shift = 63 - 3 * depth;
    return code & (~0ULL << shift);
}

// Pulls the 21 x, y, or z bits back out of a 63-bit Morton word.
static inline uint32_t compact21(uint64_t m) {
    m &= 0x1249249249249249ULL;
    m = (m ^ (m >>  2)) & 0x10c30c30c30c30c3ULL;
    m = (m ^ (m >>  4)) & 0x100f00f00f00f00fULL;
    m = (m ^ (m >>  8)) & 0x1f0000ff0000ffULL;
    m = (m ^ (m >> 16)) & 0x1f00000000ffffULL;
    m = (m ^ (m >> 32)) & 0x1fffff;
    return static_cast<uint32_t>(m);
}

// De-interleaves the top 3*depth bits of the prefix into (ix,iy,iz) integer coordinates.
static inline void decodePrefix(uint64_t prefix, int depth, uint32_t &ix, uint32_t &iy, uint32_t &iz) {
    if (depth == 0) { ix = iy = iz = 0; return; }

    // Move the relevant 3*d bits down to the LSBs and un-shuffle.
    uint64_t bits = prefix >> (63 - 3 * depth);
    ix = compact21(bits >> 0);
    iy = compact21(bits >> 1);
    iz = compact21(bits >> 2);
}

// Decodes a Morton key to the position of its minimum corner in normalized [0,1) space.
static inline Position key_to_normalized_position(uint64_t prefix, int depth) {
    uint32_t ix, iy, iz;
    decodePrefix(prefix, depth, ix, iy, iz);

    // Normalize the integer coordinates of the minimum corner.
    constexpr double Q_INV = 1.0 / (1ULL << 21);
    return {
        static_cast<double>(ix) * Q_INV,
        static_cast<double>(iy) * Q_INV,
        static_cast<double>(iz) * Q_INV
    };
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


// solar-sim/src/exchange.cpp

// (generate_interaction_list should remain as we fixed it before)

// Add this new function
void exchange_LET_gather_remotes(
    const OctreeMap&                        local_tree,
    std::vector<NodeRecord>&                remote_nodes, // Output
    const BoundingBox&                      global_bb,
    double                                  theta,
    MPI_Datatype                            node_type,
    int                                     rank,
    int                                     size,
    const std::vector<std::vector<uint64_t>>& rank_domain_keys)
{
    constexpr int BUCKET_BITS  = 18;
    constexpr int BUCKET_DEPTH = BUCKET_BITS/3;

    // 1) Build per-rank bucket AABBs with correct anisotropic dimensions
    std::vector<std::vector<BoundingBox>> bucket_boxes(size);
    double dx   = global_bb.max.x - global_bb.min.x;
    double dy   = global_bb.max.y - global_bb.min.y;
    double dz   = global_bb.max.z - global_bb.min.z;

    double cellX = dx / (1 << BUCKET_DEPTH);
    double cellY = dy / (1 << BUCKET_DEPTH);
    double cellZ = dz / (1 << BUCKET_DEPTH);

    for(int r=0; r<size; ++r){
      bucket_boxes[r].reserve(rank_domain_keys[r].size());
      for(auto pref: rank_domain_keys[r]){
        Position n = key_to_normalized_position(pref, BUCKET_DEPTH);
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

    // 2) Generate per-destination send-lists using the corrected interaction list function
    std::vector<std::vector<NodeRecord>> send_lists(size);
    double theta2 = theta*theta;
    for(int dest=0; dest<size; ++dest){
      if(dest==rank) continue;
      send_lists[dest] = generate_interaction_list(
        local_tree, bucket_boxes[dest], global_bb, theta2);
    }

    // 3) Perform MPI_Alltoallv to exchange LET data
    std::vector<int> send_counts(size), recv_counts(size);
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

/// ---------------------------------------------------------------------------
// STEP 1–4: exchange LET by building per‐rank bucket AABBs (using dx,dy,dz),
// generating per‐dest send‐lists via box–box MAC, then an MPI_Alltoallv.
// ---------------------------------------------------------------------------
void exchange_LET(
    const OctreeMap&                        local_tree,
    OctreeMap&                              full_tree,
    const BoundingBox&                      global_bb,
    double                                  theta,
    MPI_Datatype                            node_type,
    int                                     rank,
    int                                     size,
    const std::vector<std::vector<uint64_t>>& rank_domain_keys)
{
    constexpr int BUCKET_BITS  = 18;            // 6 levels ⇒ 18 bits
    constexpr int BUCKET_DEPTH = BUCKET_BITS/3; // =6

    // 1) build per‐rank bucket AABBs with correct dx,dy,dz
    std::vector<std::vector<BoundingBox>> bucket_boxes(size);
    double dx   = global_bb.max.x - global_bb.min.x;
    double dy   = global_bb.max.y - global_bb.min.y;
    double dz   = global_bb.max.z - global_bb.min.z;

    double cellX = dx / (1 << BUCKET_DEPTH);
    double cellY = dy / (1 << BUCKET_DEPTH);
    double cellZ = dz / (1 << BUCKET_DEPTH);

    for(int r=0; r<size; ++r){
      bucket_boxes[r].reserve(rank_domain_keys[r].size());
      for(auto pref: rank_domain_keys[r]){
        Position n = key_to_normalized_position(pref, BUCKET_DEPTH);
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



    // 2) generate per‐dest send‐lists
    std::vector<std::vector<NodeRecord>> send_lists(size);
    double theta2 = theta*theta;
    for(int dest=0; dest<size; ++dest){
      if(dest==rank) continue;
      send_lists[dest] = generate_interaction_list(
        local_tree, bucket_boxes[dest], global_bb, theta2);
    }

    // debug: how many nodes total will be sent?
    long long total_send = 0;
    for(auto &v: send_lists) total_send += v.size();
    std::cerr<<"[Rank "<<rank<<"] send-list total = "<<total_send<<"\n";

    // 3) all‐to‐all counts + Alltoallv
    std::vector<int> send_counts(size), recv_counts(size);
    for(int r=0;r<size;++r) send_counts[r] = int(send_lists[r].size());
    MPI_Alltoall(send_counts.data(),1,MPI_INT,
                 recv_counts.data(),1,MPI_INT,
                 MPI_COMM_WORLD);

    std::vector<int> sdis(size,0), rdis(size,0);
    int tot_s=0, tot_r=0;
    for(int r=1;r<size;++r){
      sdis[r] = tot_s += send_counts[r-1];
      rdis[r] = tot_r += recv_counts[r-1];
    }
    tot_s += send_counts.back();
    tot_r += recv_counts.back();

    std::vector<NodeRecord> sbuf(tot_s), rbuf(tot_r);
    for(int r=0;r<size;++r){
      std::copy(send_lists[r].begin(), send_lists[r].end(),
                sbuf.begin()+sdis[r]);
    }

    MPI_Alltoallv(sbuf.data(), send_counts.data(), sdis.data(), node_type,
                  rbuf.data(), recv_counts.data(), rdis.data(), node_type,
                  MPI_COMM_WORLD);

    std::cerr<<"[Rank "<<rank<<"] sent "<<tot_s
             <<"  received "<<tot_r<<"\n";

    // 4) merge + repair
    full_tree = local_tree;
    mergeIntoTree(full_tree, rbuf);
    // full_recompute_moments(full_tree);
}

// solar-sim/src/exchange.cpp

// ... (other functions in exchange.cpp like mortonPrefix, key_to_normalized_position, etc. remain) ...

// Replace the existing generate_interaction_list with this version.
std::vector<NodeRecord> generate_interaction_list(
    const OctreeMap&                      tree,
    const std::vector<BoundingBox>&       remote_boxes,
    const BoundingBox&                    global_bb,
    double                                theta2)
{
    if (tree.empty() || remote_boxes.empty())
        return {};

    robin_hood::unordered_set<OctreeKey, OctreeKeyHash> keep;
    std::array<OctreeKey, 256> stk;
    int top = 0;
    if (tree.count({0ULL, 0}))
        stk[top++] = {0ULL, 0};

    while (top > 0) {
        OctreeKey k = stk[--top];
        BoundingBox node_bb = key_to_bounding_box(k, global_bb);

        // Use the largest side length squared for the most conservative MAC.
        double sx = node_bb.max.x - node_bb.min.x;
        double sy = node_bb.max.y - node_bb.min.y;
        double sz = node_bb.max.z - node_bb.min.z;
        double s = std::max({sx, sy, sz});
        double s_squared = s * s;

        bool should_open = false;
        if (k.depth < 21) { // Max depth is 21, so leaves can't be opened
            for (const auto& remote_bucket : remote_boxes) {
                double d2 = min_distance_sq(node_bb, remote_bucket);
                if (s_squared >= theta2 * d2) {
                    should_open = true;
                    break;
                }
            }
        }

        if (should_open) {
            // OPEN: Node is too close. Push its children to continue the test.
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
            // ACCEPT: This node is a valid pseudoleaf for the entire remote domain.
            // Add ONLY this node to the keep set.
            keep.insert(k);
        }
    }

    // Serialize...
    std::vector<NodeRecord> out;
    out.reserve(keep.size());
    for (auto& k : keep) {
        auto& n = tree.at(k);
        out.push_back({k.prefix, k.depth, n.mass, n.comX, n.comY, n.comZ});
    }
    return out;
}
// ---------------------------------------------------------------------------
// Builds the source‐node list for *one* destination rank, using BOX–BOX MAC.
// ---------------------------------------------------------------------------
// std::vector<NodeRecord> generate_interaction_list(
//     const OctreeMap&                      tree,
//     const std::vector<BoundingBox>&       remote_boxes,
//     const BoundingBox&                    global_bb,
//     double                                theta2)
// {
//     if(tree.empty() || remote_boxes.empty())
//       return {};

//     robin_hood::unordered_set<OctreeKey,OctreeKeyHash> keep;
//     std::array<OctreeKey,256> stk; int top=0;
//     if(tree.count({0ULL,0})) stk[top++] = {0ULL,0};

//     while(top){
//       OctreeKey k = stk[--top];

//       BoundingBox node_bb = key_to_bounding_box(k, global_bb);

//       // Use the largest side length squared for the most conservative MAC.
//       double sx = node_bb.max.x - node_bb.min.x;
//       double sy = node_bb.max.y - node_bb.min.y;
//       double sz = node_bb.max.z - node_bb.min.z;
//       double s = std::max({sx, sy, sz});
//       double s_squared = s * s;

//       bool should_open = false;
//       if (k.depth < 21) { // Max depth is 21, so leaves can't be opened
//           for (const auto& remote_bucket : remote_boxes) {
//               double d2 = min_distance_sq(node_bb, remote_bucket);
//               if (s_squared >= theta2 * d2) {
//                   should_open = true;
//                   break;
//               }
//           }
//       }

//       if (should_open) {
//         // Open: push children.
//         uint8_t cd = k.depth + 1;
//         uint64_t stride = 1ULL << (63 - 3*cd);
//         for (uint8_t oc = 0; oc < 8; ++oc) {
//             // FIX: Use bitwise OR for clarity and robustness.
//             uint64_t child_prefix = k.prefix | (static_cast<uint64_t>(oc) * stride);
//             OctreeKey ch{ child_prefix, cd };
//             if (tree.count(ch) && top < 256) {
//                  stk[top++] = ch;
//             }
//         }
//       }
//       else {
//         // Accept: Add this node and its ancestors to the keep set.
//         for(OctreeKey cur=k;;){
//           if(!keep.insert(cur).second || cur.depth==0) break;
//           auto pd = uint8_t(cur.depth-1);
//           cur = { mortonPrefix(cur.prefix,pd), pd };
//         }
//       }
//     }

//     // Serialize...
//     std::vector<NodeRecord> out;
//     out.reserve(keep.size());
//     for(auto &k: keep){
//       auto &n = tree.at(k);
//       out.push_back({k.prefix, k.depth, n.mass, n.comX, n.comY, n.comZ});
//     }
//     return out;
// }


/**
 * @brief Exchanges essential tree data using a sender-initiated ("push") LET protocol.
 * This function populates the `remote_nodes` vector with the essential nodes
 * received from all other MPI ranks.
 */
// void exchange_LET(const OctreeMap& local_tree, std::vector<NodeRecord>& remote_nodes,
//                   const BoundingBox& local_bb, const BoundingBox& global_bb,
//                   double theta, MPI_Datatype node_type, MPI_Datatype bb_type, int rank, int size) {

//     // 1. Exchange Domain Bounding Boxes
//     std::vector<BoundingBox> all_domain_bbs(size);
//     MPI_Allgather(&local_bb, 1, bb_type,
//                   all_domain_bbs.data(), 1, bb_type, MPI_COMM_WORLD);

//     // 2. Generate Interaction Lists (Local Computation)
//     std::vector<std::vector<NodeRecord>> send_lists(size);
//     double theta_sq = theta * theta;
//     for (int j = 0; j < size; ++j) {
//         if (j == rank) continue;
//         send_lists[j] = generate_interaction_list(local_tree, all_domain_bbs[j], global_bb, theta_sq);
//     }

//     // 3. Exchange Interaction Counts
//     std::vector<int> send_counts(size);
//     for (int i = 0; i < size; ++i) {
//         send_counts[i] = send_lists[i].size();
//     }
//     std::vector<int> recv_counts(size);
//     MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

//     // 4. Exchange LET Data with MPI_Alltoallv
//     std::vector<NodeRecord> send_buf;
//     std::vector<int> sdispls(size, 0);
//     int total_to_send = 0;
//     for (int i = 0; i < size; ++i) {
//         sdispls[i] = total_to_send;
//         total_to_send += send_counts[i];
//     }
//     send_buf.resize(total_to_send);
//     for(int i = 0; i < size; ++i) {
//         if(send_counts[i] > 0) {
//             std::copy(send_lists[i].begin(), send_lists[i].end(), send_buf.begin() + sdispls[i]);
//         }
//     }

//     std::vector<int> rdispls(size, 0);
//     int total_to_recv = 0;
//     for (int i = 0; i < size; ++i) {
//         rdispls[i] = total_to_recv;
//         total_to_recv += recv_counts[i];
//     }

//     // Resize the output parameter directly to serve as the receive buffer.
//     remote_nodes.resize(total_to_recv);

//     MPI_Alltoallv(send_buf.data(), send_counts.data(), sdispls.data(), node_type,
//                   remote_nodes.data(), recv_counts.data(), rdispls.data(), node_type,
//                   MPI_COMM_WORLD);
// }

/**
 * @brief Exchanges essential tree data using a sender-initiated ("push") LET protocol.
 */
// void exchange_LET(const OctreeMap& local_tree, OctreeMap& full_tree,
//                   const BoundingBox& local_bb, const BoundingBox& global_bb,
//                   double theta, MPI_Datatype node_type, MPI_Datatype bb_type, int rank, int size) {

//     // 1. Exchange Domain Bounding Boxes
//     std::vector<BoundingBox> all_domain_bbs(size);

//     // --- DEBUG PRINT ---
//     std::cout << "[Rank " << rank << "] Starting MPI_Allgather for bounding boxes." << std::endl;
//     MPI_Barrier(MPI_COMM_WORLD); // Synchronize before the call for cleaner logs

//     MPI_Allgather(&local_bb, 1, bb_type, all_domain_bbs.data(), 1, bb_type, MPI_COMM_WORLD);
//     // --- DEBUG PRINT ---
//     MPI_Barrier(MPI_COMM_WORLD); // Synchronize after the call
//     std::cout << "[Rank " << rank << "] Finished MPI_Allgather." << std::endl;

//     // 2. Generate Interaction Lists (Local Computation)
//     std::vector<std::vector<NodeRecord>> send_lists(size);
//     double theta_sq = theta * theta;

//     for (int j = 0; j < size; ++j) {
//         if (j == rank) continue;
//         send_lists[j] = generate_interaction_list(local_tree, all_domain_bbs[j], global_bb, theta_sq);
//     }

//     // 3. Exchange Interaction Counts
//     std::vector<int> send_counts(size);
//     for (int i = 0; i < size; ++i) {
//         send_counts[i] = send_lists[i].size();
//     }
//     std::vector<int> recv_counts(size);

//     // --- DEBUG PRINT ---
//     std::cout << "[Rank " << rank << "] Starting MPI_Alltoall for counts." << std::endl;
//     MPI_Barrier(MPI_COMM_WORLD);

//     MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

//     // --- DEBUG PRINT ---
//     MPI_Barrier(MPI_COMM_WORLD);
//     std::cout << "[Rank " << rank << "] Finished MPI_Alltoall." << std::endl;


//     // 4. Exchange LET Data with MPI_Alltoallv
//     std::vector<NodeRecord> send_buf;
//     std::vector<int> sdispls(size, 0);
//     int total_to_send = 0;
//     for (int i = 0; i < size; ++i) {
//         sdispls[i] = total_to_send;
//         total_to_send += send_counts[i];
//     }
//     send_buf.resize(total_to_send);
//     for(int i = 0; i < size; ++i) {
//         if(send_counts[i] > 0) {
//             std::copy(send_lists[i].begin(), send_lists[i].end(), send_buf.begin() + sdispls[i]);
//         }
//     }

//     std::vector<int> rdispls(size, 0);
//     int total_to_recv = 0;
//     for (int i = 0; i < size; ++i) {
//         rdispls[i] = total_to_recv;
//         total_to_recv += recv_counts[i];
//     }
//     std::vector<NodeRecord> recv_buf(total_to_recv);

//     // --- DEBUG PRINT ---
//     std::cout << "[Rank " << rank << "] Starting MPI_Alltoallv. Sending " << total_to_send
//               << " nodes, receiving " << total_to_recv << " nodes." << std::endl;
//     MPI_Barrier(MPI_COMM_WORLD);

//     MPI_Alltoallv(send_buf.data(), send_counts.data(), sdispls.data(), node_type,
//                   recv_buf.data(), recv_counts.data(), rdispls.data(), node_type,
//                   MPI_COMM_WORLD);

//     // --- DEBUG PRINT ---
//     MPI_Barrier(MPI_COMM_WORLD);
//     std::cout << "[Rank " << rank << "] Finished MPI_Alltoallv." << std::endl;

//     // Construct the full tree by merging the local tree and received nodes
//     full_tree = local_tree;
//     mergeIntoTree(full_tree, recv_buf);
// }

// inline double pointAABB2(double x,double y,double z,
//                          const BoundingBox& bb)
// {
//     double dx = std::max({0.0, bb.min.x - x, x - bb.max.x});
//     double dy = std::max({0.0, bb.min.y - y, y - bb.max.y});
//     double dz = std::max({0.0, bb.min.z - z, z - bb.max.z});
//     return dx*dx + dy*dy + dz*dz;
// }


// std::vector<NodeRecord> generate_interaction_list(
//         const OctreeMap&   tree,
//         const BoundingBox& remote_bb,
//         const BoundingBox& global_bb,
//         double             theta2)                 // already θ²
// {
//     if (tree.empty()) return {};

//     constexpr int MAX_L = 21;
//     double L = std::max({global_bb.max.x - global_bb.min.x,
//                          global_bb.max.y - global_bb.min.y,
//                          global_bb.max.z - global_bb.min.z});

//     /* side² for every depth */
//     std::array<double, MAX_L+1> side2;
//     for (int d=0; d<=MAX_L; ++d) {
//         double s = L / double(1ULL<<d);
//         side2[d] = s*s;
//     }

//     robin_hood::unordered_set<OctreeKey,OctreeKeyHash> essential;
//     constexpr int STK = 256;
//     std::array<OctreeKey,STK> stack;
//     int top = 0;
//     if (tree.count({0,0})) stack[top++] = {0ULL,0};

//     while (top) {
//         OctreeKey k = stack[--top];
//         const OctreeNode& n = tree.at(k);

//         double d2 = pointAABB2(n.comX,n.comY,n.comZ, remote_bb);

//         if (k.depth==MAX_L || side2[k.depth] < theta2 * d2) {
//             /* accept, prune descendants */
//             for (int d=k.depth+1; d<=MAX_L; ++d) {
//                 essential.erase({ mortonPrefix(k.prefix,d), uint8_t(d) });
//             }
//             /* add node + ancestors */
//             for (OctreeKey cur=k;;) {
//                 if (!essential.insert(cur).second || cur.depth==0) break;
//                 uint8_t pd = cur.depth-1;
//                 cur = { mortonPrefix(cur.prefix,pd), pd };
//             }
//         } else if (k.depth < MAX_L) {          // open node
//             uint8_t cd = k.depth+1;
//             uint64_t stride = 1ULL << (63-3*cd);
//             for (uint8_t oc=0; oc<8; ++oc) {
//                 OctreeKey child{ k.prefix | uint64_t(oc)*stride, cd };
//                 if (tree.count(child)) {
//                     if (top>=STK) throw std::runtime_error("LET stack ovfl");
//                     stack[top++] = child;
//                 }
//             }
//         }
//     }

//     std::vector<NodeRecord> out;
//     out.reserve(essential.size());
//     for (auto& k : essential) {
//         const OctreeNode& n = tree.at(k);
//         out.push_back({k.prefix,k.depth,n.mass,n.comX,n.comY,n.comZ});
//     }
//     return out;
// }
// -----------------------------------------------------------------------------
// exchange_LET  ────────────────────────────────────────────────────────────────
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// exchange_LET  — with compact debug prints
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// exchange_LET  — with compact debug prints
// -----------------------------------------------------------------------------
// void exchange_LET(
//     const OctreeMap&                       local_tree,
//     OctreeMap&                             full_tree,
//     const BoundingBox&                     global_bb,
//     double                                 theta,
//     MPI_Datatype                           node_type,
//     int                                    rank,
//     int                                    size,
//     const std::vector<std::vector<uint64_t>>& rank_domain_keys)
// {
//     constexpr int BUCKET_BITS  = 18;                 // depth-6 ⇒ 18 bits
//     constexpr int BUCKET_DEPTH = BUCKET_BITS / 3;    // = 6
//     double dx = global_bb.max.x - global_bb.min.x;
//     double dy = global_bb.max.y - global_bb.min.y;
//     double dz = global_bb.max.z - global_bb.min.z;

//     // ── part 1: bucket-centres ------------------------------------------------
//     std::vector<std::vector<Position>> remote_pos(size);
//     for (int r = 0; r < size; ++r) {
//         remote_pos[r].reserve(rank_domain_keys[r].size());
//         for (uint64_t p : rank_domain_keys[r]) {
//             Position n = key_to_normalized_position(p, BUCKET_DEPTH);
//             remote_pos[r].push_back({ global_bb.min.x + n.x*dx,
//                                       global_bb.min.y + n.y*dy,
//                                       global_bb.min.z + n.z*dz });
//         }
//     }

//     // ── part 2: build send-lists --------------------------------------------
//     std::vector<std::vector<NodeRecord>> send_lists(size);
//     for (int dest = 0; dest < size; ++dest)
//         if (dest != rank)
//             send_lists[dest] = generate_interaction_list(
//                                    local_tree, remote_pos[dest],
//                                    global_bb, theta);

//     /* debug: total nodes we will send */
//     {
//         long long tot = 0;
//         for (auto& v : send_lists) tot += v.size();
//         std::cerr << "[Rank " << rank
//                   << "] send-list total = " << tot << '\n';
//     }

//     // ── part 3: Alltoallv counts + data -------------------------------------
//     std::vector<int> send_counts(size), recv_counts(size);
//     for (int r = 0; r < size; ++r) send_counts[r] = send_lists[r].size();
//     MPI_Alltoall(send_counts.data(), 1, MPI_INT,
//                  recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

//     std::vector<int> sdis(size,0), rdis(size,0);
//     int total_s = 0, total_r = 0;
//     for (int r = 1; r < size; ++r) {
//         sdis[r] = (total_s += send_counts[r-1]);
//         rdis[r] = (total_r += recv_counts[r-1]);
//     }
//     total_s += send_counts.back();
//     total_r += recv_counts.back();

//     std::vector<NodeRecord> sbuf(total_s), rbuf(total_r);
//     for (int r = 0; r < size; ++r)
//         std::copy(send_lists[r].begin(), send_lists[r].end(),
//                   sbuf.begin() + sdis[r]);

//     MPI_Alltoallv(sbuf.data(), send_counts.data(), sdis.data(), node_type,
//                   rbuf.data(), recv_counts.data(), rdis.data(), node_type,
//                   MPI_COMM_WORLD);

//     /* debug: one-line summary of traffic */
//     std::cerr << "[Rank " << rank << "] sent " << total_s
//               << "  received " << total_r << '\n';

//     // ── part 4: merge + repair ---------------------------------------------
//     full_tree = local_tree;
//     mergeIntoTree(full_tree, rbuf);
//     recompute_ancestor_moments(full_tree, rbuf);
// }



// -----------------------------------------------------------------------------
// generate_interaction_list  (LET exporter)  – fully conservative MAC
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// generate_interaction_list  – conservative MAC + duplicate pruning
// -----------------------------------------------------------------------------
// std::vector<NodeRecord> generate_interaction_list(
//         const OctreeMap&              tree,
//         const std::vector<Position>&  remote_bucket_pos,
//         const BoundingBox&            global_bb,
//         double                        theta)
// {
//     if (tree.empty() || remote_bucket_pos.empty()) return {};

//     constexpr int MAX_L = 21;
//     constexpr int BUCKET_DEPTH = 6;           // depth-6 buckets
//     const double theta2 = theta * theta;

//     double L = std::max({global_bb.max.x - global_bb.min.x,
//                          global_bb.max.y - global_bb.min.y,
//                          global_bb.max.z - global_bb.min.z});

//     /* cell side² for every depth */
//     std::array<double, MAX_L+1> side2;
//     for (int d = 0; d <= MAX_L; ++d) {
//         double s = L / double(1ULL << d);
//         side2[d] = s*s;
//     }

//     /* bounding-sphere radius² of one depth-6 bucket */
//     const double sideB = L / 64.0;            // 2^6 = 64
//     const double rB2   = 0.75 * sideB * sideB;  // (√3/2)^2 = 0.75

//     robin_hood::unordered_set<OctreeKey,OctreeKeyHash> essential;

//     constexpr int STACK_MAX = 256;
//     std::array<OctreeKey,STACK_MAX> stack;
//     int top = 0;
//     if (tree.count({0,0})) stack[top++] = {0ULL,0};

//     while (top) {
//         OctreeKey key = stack[--top];
//         auto it = tree.find(key);
//         if (it == tree.end()) continue;
//         const OctreeNode& node = it->second;

//         /* min distance² from node COM to any bucket centre */
//         double min_r2 = std::numeric_limits<double>::infinity();
//         for (const auto& bp : remote_bucket_pos) {
//             double dx = node.comX - bp.x;
//             double dy = node.comY - bp.y;
//             double dz = node.comZ - bp.z;
//             double d2 = dx*dx + dy*dy + dz*dz;
//             if (d2 < min_r2) min_r2 = d2;
//         }
//         double d2_eff = std::max(0.0, min_r2 - rB2);   // conservative

//         if (key.depth == MAX_L || side2[key.depth] < theta2 * d2_eff) {
//             /* accept node, but first drop any earlier-accepted descendants */
//             for (int d = key.depth+1; d <= MAX_L; ++d) {
//                 OctreeKey k{ mortonPrefix(key.prefix,d), uint8_t(d) };
//                 essential.erase(k);
//             }
//             /* add node & ancestors up to the root */
//             for (OctreeKey cur = key;;) {
//                 if (!essential.insert(cur).second || cur.depth == 0) break;
//                 uint8_t pd = cur.depth - 1;
//                 cur = { mortonPrefix(cur.prefix,pd), pd };
//             }
//         } else if (key.depth < MAX_L) {
//             /* open the node */
//             uint8_t cd = key.depth + 1;
//             uint64_t stride = 1ULL << (63 - 3*cd);
//             for (uint8_t oc = 0; oc < 8; ++oc) {
//                 OctreeKey child{ key.prefix | uint64_t(oc)*stride, cd };
//                 if (tree.count(child)) {
//                     if (top >= STACK_MAX) throw std::runtime_error("LET stack overflow");
//                     stack[top++] = child;
//                 }
//             }
//         }
//     }

//     /* pack result ----------------------------------------------------------- */
//     std::vector<NodeRecord> out;
//     out.reserve(essential.size());
//     for (const auto& k : essential) {
//         const OctreeNode& n = tree.at(k);
//         out.push_back({k.prefix, k.depth,
//                        n.mass, n.comX, n.comY, n.comZ});
//     }
//     return out;
// }



// Generates the list of essential nodes for a remote domain
/**
 * @brief (Internal) Generates the list of essential nodes for a remote domain.
 *
 * This function implements the canonical "path-to-root" export. When a node
 * is found to be essential, it and its entire chain of ancestors are added
 * to the export list. This ensures the receiver can reconstruct a fully
 * connected tree segment.
 */
// std::vector<NodeRecord>
// generate_interaction_list(const OctreeMap& local_tree,
//                           const BoundingBox& remote_bb,
//                           const BoundingBox& global_bb,
//                           double theta_sq) {

//     // Use a hash set to store the unique keys of essential nodes and their ancestors.
//     robin_hood::unordered_set<OctreeKey, OctreeKeyHash> essential_keys;
//     if (local_tree.empty()) return {};

//     std::stack<OctreeKey> stack;
//     // Start traversal from the root, if it exists.
//     if (local_tree.count({0, 0})) {
//         stack.push({0, 0});
//     }

//     while (!stack.empty()) {
//         OctreeKey key = stack.top();
//         stack.pop();

//         auto it = local_tree.find(key);
//         if (it == local_tree.end()) continue;

//         BoundingBox node_bb = key_to_bounding_box(key, global_bb);
//         double s = node_bb.max.x - node_bb.min.x;
//         double d_sq = min_distance_sq(node_bb, remote_bb);

//         if (s * s < theta_sq * d_sq) {
//             // This node is essential. Add it AND all its ancestors to the set.
//             OctreeKey current_key = key;
//             while (true) {
//                 // The set automatically handles duplicates.
//                 essential_keys.insert(current_key);
//                 if (current_key.depth == 0) break;

//                 uint8_t p_dep = current_key.depth - 1;
//                 uint64_t p_prefix = mortonPrefix(current_key.prefix, p_dep);
//                 current_key = {p_prefix, p_dep};
//             }
//         } else {
//             // Node is too close, open it and check its children.
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

//     // Convert the set of unique keys into a vector of NodeRecords to be sent.
//     std::vector<NodeRecord> essential_nodes;
//     essential_nodes.reserve(essential_keys.size());
//     for (const auto& key : essential_keys) {
//         auto it = local_tree.find(key);
//         if (it != local_tree.end()) {
//             const auto& node = it->second;
//             essential_nodes.push_back({key.prefix, key.depth, node.mass, node.comX, node.comY, node.comZ});
//         }
//     }
//     return essential_nodes;
// }
// call this *after* mergeIntoTree(full_tree, rbuf)
void full_recompute_moments(OctreeMap &tree) {
    constexpr int MAX_L = 21;

    // for each depth from deepest non‐leaf up to 0 (the root)
    for (int depth = MAX_L-1; depth >= 0; --depth) {
        // for every node in the tree
        for (auto &it : tree) {
            const OctreeKey &key = it.first;
            // only rebuild internal nodes at this depth
            if ((int)key.depth != depth) continue;

            OctreeNode &node = it.second;
            // accumulate from children
            double m_sum = 0, x_sum = 0, y_sum = 0, z_sum = 0;
            uint8_t cd = depth + 1;
            uint64_t stride = 1ULL << (63 - 3*cd);

            for (uint8_t oc = 0; oc < 8; ++oc) {
                OctreeKey child{ key.prefix + stride*oc, cd };
                auto cit = tree.find(child);
                if (cit != tree.end() && cit->second.mass > 0.0) {
                    const auto &c = cit->second;
                    m_sum   += c.mass;
                    x_sum   += c.mass * c.comX;
                    y_sum   += c.mass * c.comY;
                    z_sum   += c.mass * c.comZ;
                }
            }

            if (m_sum > 0.0) {
                node.mass = m_sum;
                node.comX = x_sum / m_sum;
                node.comY = y_sum / m_sum;
                node.comZ = z_sum / m_sum;
            }
        }
    }
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
