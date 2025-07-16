#include "traversal.h"

#include <array>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <omp.h>
#include <algorithm>
#include <iostream>
#include <iomanip>


void computeAccelerations(
    const OctreeMap&                tree,
    const std::vector<uint64_t>&    key,
    const std::vector<Position>&    pos,
    double                          theta,
    double                          G,
    double                          soft2,
    const BoundingBox&              global_bb,
    std::vector<Acceleration>&      out,
    int                             rank)
{
    const size_t N = pos.size();
    out.resize(N);
    const double theta2 = theta * theta;

    constexpr int MAX_L = 21;

    // root side length
    double dx = global_bb.max.x - global_bb.min.x;
    double dy = global_bb.max.y - global_bb.min.y;
    double dz = global_bb.max.z - global_bb.min.z;
    double L  = std::max({dx, dy, dz});   // physical side length of root cell

    // precompute side^2 & stride per depth
    std::array<double,   MAX_L+1> side2;
    std::array<uint64_t, MAX_L+1> stride;
    for (int d = 0; d <= MAX_L; ++d) {
        double side = L / double(1ULL << d);
        side2[d] = side * side;
        stride[d] = 1ULL << (63 - 3*d);
    }

    // build sort order once (serial)
    struct Ord { uint64_t k; size_t i; };
    std::vector<Ord> order(N);
    for (size_t i=0;i<N;++i) order[i] = {key[i], i};
    std::sort(order.begin(), order.end(), [](auto&a,auto&b){return a.k<b.k;});

    constexpr int STACK_MAX = 256; // safe for depth <=21

    #pragma omp parallel for
    for (size_t oi = 0; oi < N; ++oi) {
        const size_t i = order[oi].i;  // actual body index

        double ax = 0.0, ay = 0.0, az = 0.0;

        std::array<std::pair<uint64_t,uint8_t>, STACK_MAX> stack;
        int top = 0;
        stack[top++] = {0ULL, 0};  // push root

        while (top) {
            auto [pref, dep] = stack[--top];
            auto it = tree.find(OctreeKey{pref, dep});
            if (it == tree.end() || it->second.mass == 0.0) continue;
            const OctreeNode& node = it->second;

            // skip self interaction at leaf
            if (dep == MAX_L && pref == key[i]) continue;

            double dx = node.comX - pos[i].x;
            double dy = node.comY - pos[i].y;
            double dz = node.comZ - pos[i].z;
            double r2 = dx*dx + dy*dy + dz*dz + soft2;

            // Barnes–Hut acceptance
            if (node.is_pseudo || dep == MAX_L || side2[dep] < theta2 * r2) {
                double invR  = 1.0 / std::sqrt(r2);
                double invR3 = invR * invR * invR;
                double s     = G * node.mass * invR3;
                ax += s * dx;  ay += s * dy;  az += s * dz;
            } else {
                // open cell, push its children
                const uint8_t  cdep = dep + 1;
                const uint64_t s    = stride[cdep];
                for (uint8_t oc = 0; oc < 8; ++oc) {
                    const uint64_t child = pref + s*oc;  // unchanged
                    auto itc = tree.find(OctreeKey{child, cdep});
                    if (itc != tree.end()) {
                        if (top < STACK_MAX) {
                            stack[top++] = {child, cdep};
                        } else {
                            throw std::runtime_error("BH stack overflow");
                        }
                    }
                }
            }
        }

        out[i] = {ax, ay, az};
    }
}



void computeAccelerationsWithLET(
    OctreeMap&                     local_tree,
    const std::vector<NodeRecord>& remote_nodes,
    const std::vector<uint64_t>&   bodyKey,
    const std::vector<Position>&   pos,
    double                         theta,
    double                         G,
    double                         soft2,
    const BoundingBox&             global_bb,
    std::vector<Acceleration>&     out,
    int                            rank)
{
    OctreeMap remote_tree = buildTreeFromPseudoLeaves(remote_nodes);
    if (!remote_nodes.empty()) {
        OctreeMap remote_tree = buildTreeFromPseudoLeaves(remote_nodes);
        mergeRecordsIntoTree(local_tree, serializeTreeToRecords(remote_tree));
    }

    computeAccelerations(local_tree, bodyKey, pos, theta, G, soft2, global_bb, out, rank);

}

/**
 * @brief Computes accelerations using a local BH traversal and a remote direct force summation.
 *
 * This function provides a hybrid approach for force calculation. It performs:
 * 1. A standard Barnes-Hut traversal on the local_tree for local interactions.
 * 2. A direct O(N*M) summation where the force from each of the M remote_nodes
 *    is calculated for each of the N local bodies.
 *
 * This method avoids the overhead of building a secondary tree for remote nodes, which
 * can be faster if the number of remote pseudo-leaves is small
 *
 * @param local_tree The pre-built octree for the local rank's particles
 * @param remote_nodes The vector of pseudo-leaves received from other ranks
 * @param key Morton keys of the local bodies
 * @param pos Positions of the local bodies
 * @param theta The Barnes-Hut opening angle for the local traversal
 * @param G The gravitational constant
 * @param soft2 The softening factor squared
 * @param[out] out The output vector where the final calculated accelerations are stored
 */
void computeAccelerationsWithRemoteDirectSum(
    const OctreeMap&               local_tree,
    const std::vector<NodeRecord>& remote_nodes,
    const std::vector<uint64_t>&   key,
    const std::vector<Position>&   pos,
    double                         theta,
    double                         G,
    double                         soft2,
    const BoundingBox&             global_bb,
    std::vector<Acceleration>&     out,
    int                            rank)
{
    // const auto t0 = std::chrono::high_resolution_clock::now();
    // const auto p1_start = std::chrono::high_resolution_clock::now();

    // Pass 1 local Barnes–Hut walk
    computeAccelerations(local_tree, key, pos, theta, G, soft2, global_bb, out,
                    rank);

    // const double p1_us = std::chrono::duration_cast<std::chrono::microseconds>(
    //                          std::chrono::high_resolution_clock::now() -
    //                          p1_start)
    //                          .count();
    // const auto p2_start = std::chrono::high_resolution_clock::now();

    // Pass 2 direct forces from remote pseudoleaves
    // using omp simd
    // build a structure of arrays copy so the compiler sees unit stride
    const std::size_t M = remote_nodes.size();
    std::vector<double> rx(M), ry(M), rz(M), rm(M);
    for (std::size_t j = 0; j < M; ++j) {
        rx[j] = remote_nodes[j].comX;
        ry[j] = remote_nodes[j].comY;
        rz[j] = remote_nodes[j].comZ;
        rm[j] = remote_nodes[j].mass * G;
    }

    // loop over local bodies
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < pos.size(); ++i) {

        double ax = 0.0, ay = 0.0, az = 0.0;
        const double px = pos[i].x;
        const double py = pos[i].y;
        const double pz = pos[i].z;

        // vectorised inner loop over remote nodes
        #pragma omp simd reduction(+:ax,ay,az)
        for (std::size_t j = 0; j < M; ++j) {
            double dx = rx[j] - px;
            double dy = ry[j] - py;
            double dz = rz[j] - pz;
            double r2 = dx*dx + dy*dy + dz*dz + soft2;
            if (r2 <= 1e-12) continue;

            double invR  = 1.0 / std::sqrt(r2);
            double invR3 = invR * invR * invR;
            double s     = rm[j] * invR3;
            ax += s * dx;
            ay += s * dy;
            az += s * dz;
        }

        out[i].x += ax;
        out[i].y += ay;
        out[i].z += az;
    }

    // const double p2_us = std::chrono::duration_cast<std::chrono::microseconds>(
    //                          std::chrono::high_resolution_clock::now() -
    //                          p2_start)
    //                          .count();

    // const double total_us =
    //     std::chrono::duration_cast<std::chrono::microseconds>(
    //         std::chrono::high_resolution_clock::now() - t0)
    //         .count();

    // if (rank == 0) {
    //     std::cout << std::fixed << std::setprecision(1)
    //               << "    - BH remote sum (Pass 2): " << p2_us << " µs ("
    //               << pct(p2_us, total_us) << "%)\n"
    //               << "    - BH local  walk (Pass 1): " << p1_us << " µs ("
    //               << pct(p1_us, total_us) << "%)\n"
    //               << "    - BH Dual Walk (total)   : " << total_us << " µs\n";
    // }
}


// static inline double pct(double part, double whole) {
//     return whole > 0.0 ? 100.0 * part / whole : 0.0;
// }
// static inline double ns2us(long long ns) { return ns / 1000.0; }

// void computeAccelerationsWithLET(
//     OctreeMap&                     local_tree,
//     const std::vector<NodeRecord>& remote_nodes,
//     const std::vector<uint64_t>&   bodyKey,
//     const std::vector<Position>&   pos,
//     double                         theta,
//     double                         G,
//     double                         soft2,
//     const BoundingBox&             global_bb,
//     std::vector<Acceleration>&     out,
//     int                            rank)
// {
//     // Local traversal
//     computeAccelerations(local_tree, bodyKey, pos, theta, G, soft2, global_bb, out, rank);

//     if (remote_nodes.empty()) {
//         return; // All force calculations are complete.
//     }

//     // Remote traversal
//     OctreeMap remote_tree = buildTreeFromPseudoLeaves(remote_nodes);

//     // Create a temporary vector to hold the accelerations from remote nodes.
//     std::vector<Acceleration> remote_acc;

//     // We need a dummy key vector for the remote traversal, since none of our
//     // local bodies are actually in the remote tree. This prevents the
//     // self interaction check from ever being true
//     std::vector<uint64_t> dummy_keys(pos.size(), ~uint64_t(0));

//     computeAccelerations(remote_tree, dummy_keys, pos, theta, G, soft2, global_bb, remote_acc, rank);

//     // Add the remote force contributions to the local ones
//     #pragma omp parallel for
//     for (size_t i = 0; i < pos.size(); ++i) {
//         out[i].x += remote_acc[i].x;
//         out[i].y += remote_acc[i].y;
//         out[i].z += remote_acc[i].z;
//     }
// }
