#include "traversal.h"

#include <array>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <omp.h>


// void bhAccelerations(
//     const OctreeMap&                tree,
//     const std::vector<uint64_t>&    key,
//     const std::vector<Position>&    pos,
//     double                          theta,
//     double                          G,
//     double                          soft2,
//     const BoundingBox&              global_bb,
//     std::vector<Acceleration>&      out)
// {
//     const size_t N = pos.size();
//     out.resize(N);
//     const double theta2 = theta * theta;

//     constexpr int MAX_L = 21;
//     double L = std::max({global_bb.max.x - global_bb.min.x,
//                          global_bb.max.y - global_bb.min.y,
//                          global_bb.max.z - global_bb.min.z});

//     std::array<double, MAX_L + 1> side2;
//     for (int d = 0; d <= MAX_L; ++d) {
//         double side = L / double(1ULL << d);
//         side2[d] = side * side;
//     }

//     // FIX #1: Identify all roots of disconnected tree fragments.
//     // An orphan node is a "root" if its parent does not exist in the tree.
//     std::vector<OctreeKey> tree_roots;
//     if (!tree.empty()) {
//         for (const auto& [node_key, node_val] : tree) {
//             if (node_key.depth == 0) {
//                 tree_roots.push_back(node_key); // The main root
//             } else {
//                 uint8_t p_dep = node_key.depth - 1;
//                 uint64_t p_prefix = node_key.prefix & (~0ULL << (63 - 3 * p_dep));
//                 if (tree.find({p_prefix, p_dep}) == tree.end()) {
//                     tree_roots.push_back(node_key); // This is an orphan root
//                 }
//             }
//         }
//     }

//     constexpr int STACK_MAX = 256;

//     #pragma omp parallel for schedule(dynamic, 256)
//     for (size_t i = 0; i < N; ++i) {
//         double ax = 0.0, ay = 0.0, az = 0.0;
//         std::array<OctreeKey, STACK_MAX> stack;
//         int top = 0;

//         // FIX #2: Start traversal from ALL identified roots.
//         for (const auto& root_key : tree_roots) {
//             if (top < STACK_MAX) stack[top++] = root_key;
//             else throw std::runtime_error("BH stack overflow on root push");
//         }

//         while (top > 0) {
//             OctreeKey current_key = stack[--top];
//             auto it = tree.find(current_key);
//             if (it == tree.end() || it->second.mass == 0.0) continue;
//             const OctreeNode& node = it->second;

//             if (current_key.depth == MAX_L && current_key.prefix == key[i]) continue;

//             double dx = node.comX - pos[i].x;
//             double dy = node.comY - pos[i].y;
//             double dz = node.comZ - pos[i].z;
//             double r2 = dx * dx + dy * dy + dz * dz + soft2;

//             // FIX #3: Check for children before deciding to open a node.
//             bool has_children = false;
//             if (current_key.depth < MAX_L) {
//                 uint8_t cdep = current_key.depth + 1;
//                 uint64_t stride = 1ULL << (63 - 3 * cdep);
//                 for (uint8_t oc = 0; oc < 8; ++oc) {
//                     if (tree.count({current_key.prefix | (stride * oc), cdep})) {
//                         has_children = true;
//                         break;
//                     }
//                 }
//             }

//             // New acceptance criterion: Use the node if it's far enough away
//             // OR if it's a "pseudo-leaf" (a pruned remote node with no children).
//             if (!has_children || side2[current_key.depth] < theta2 * r2) {
//                 double invR  = 1.0 / std::sqrt(r2);
//                 double invR3 = invR * invR * invR;
//                 double s     = G * node.mass * invR3;
//                 ax += s * dx;  ay += s * dy;  az += s * dz;
//             } else {
//                 // This block is now only entered if the node is too close AND has children.
//                 uint8_t cdep = current_key.depth + 1;
//                 uint64_t stride = 1ULL << (63 - 3*cdep);
//                 for (uint8_t oc = 0; oc < 8; ++oc) {
//                     OctreeKey child_key = {current_key.prefix | (stride * oc), cdep};
//                     if (tree.count(child_key)) {
//                          if (top < STACK_MAX) stack[top++] = child_key;
//                          else throw std::runtime_error("BH stack overflow");
//                     }
//                 }
//             }
//         }
//         out[i] = {ax, ay, az};
//     }
// }

void bhAccelerations(
    const OctreeMap&                tree,
    const std::vector<uint64_t>&    key,       // one Morton key per body
    const std::vector<Position>&    pos,       // physical coords (e.g. AU)
    double                          theta,
    double                          G,
    double                          soft2,     // ε²
    const BoundingBox&              global_bb, // from rebalance_bodies
    std::vector<Acceleration>&      out)
{
    const size_t N = pos.size();
    out.resize(N);
    const double theta2 = theta * theta;

    // --- Compute physical cell side² for depths 0…MAX_L ---
    constexpr int MAX_L = 21;
    // find the largest dimension of the global box
    double dx = global_bb.max.x - global_bb.min.x;
    double dy = global_bb.max.y - global_bb.min.y;
    double dz = global_bb.max.z - global_bb.min.z;
    double L  = std::max({dx, dy, dz});       // physical side-length of root cell

    std::array<double, MAX_L+1> side2;
    for (int d = 0; d <= MAX_L; ++d) {
        double side = L / double(1ULL << d);
        side2[d] = side * side;
    }

    constexpr int STACK_MAX = 256; // safe for depth ≤21

    #pragma omp parallel for schedule(dynamic,256)
    for (size_t i = 0; i < N; ++i) {
        double ax = 0.0, ay = 0.0, az = 0.0;

        std::array<std::pair<uint64_t,uint8_t>, STACK_MAX> stack;
        int top = 0;
        stack[top++] = {0ULL, 0};  // push root (prefix=0,depth=0)

        while (top) {
            auto [pref, dep] = stack[--top];
            auto it = tree.find(OctreeKey{pref, dep});
            if (it == tree.end() || it->second.mass == 0.0) continue;
            const OctreeNode& node = it->second;

            // skip self‐interaction at leaf
            if (dep == MAX_L && pref == key[i]) continue;

            double dx = node.comX - pos[i].x;
            double dy = node.comY - pos[i].y;
            double dz = node.comZ - pos[i].z;
            double r2 = dx*dx + dy*dy + dz*dz + soft2;

            // Barnes–Hut acceptance: cell is “small” relative to distance
            if (dep == MAX_L || side2[dep] < theta2 * r2) {
                double invR  = 1.0 / std::sqrt(r2);
                double invR3 = invR * invR * invR;
                double s     = G * node.mass * invR3;
                ax += s * dx;  ay += s * dy;  az += s * dz;
            } else {
                // open cell: push its children
                uint8_t cdep = dep + 1;
                uint64_t stride = 1ULL << (63 - 3*cdep);
                for (uint8_t oc = 0; oc < 8; ++oc) {
                    uint64_t child = pref + stride*oc;
                    if (tree.count(OctreeKey{child, cdep})) {
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



void bhAccelerations(
    const OctreeMap&                tree,
    const std::vector<NodeRecord>&  remote_nodes,
    const std::vector<uint64_t>&    key,
    const std::vector<Position>&    pos,
    double                          theta,
    double                          G,
    double                          soft2,
    const BoundingBox&              global_bb,
    std::vector<Acceleration>&      out)
{
    const size_t N = pos.size();
    out.resize(N);
    const double theta2 = theta * theta;

    // --- Compute physical cell side² for depths 0…MAX_L ---
    constexpr int MAX_L = 21;
    // find the largest dimension of the global box
    double dx = global_bb.max.x - global_bb.min.x;
    double dy = global_bb.max.y - global_bb.min.y;
    double dz = global_bb.max.z - global_bb.min.z;
    double L  = std::max({dx, dy, dz});       // physical side-length of root cell

    std::array<double, MAX_L+1> side2;
    for (int d = 0; d <= MAX_L; ++d) {
        double side = L / double(1ULL << d);
        side2[d] = side * side;
    }

    constexpr int STACK_MAX = 256; // safe for depth ≤21

    #pragma omp parallel for schedule(dynamic,256)
    for (size_t i = 0; i < N; ++i) {
        double ax = 0.0, ay = 0.0, az = 0.0;

        std::array<std::pair<uint64_t,uint8_t>, STACK_MAX> stack;
        int top = 0;
        stack[top++] = {0ULL, 0};  // push root (prefix=0,depth=0)

        while (top) {
            auto [pref, dep] = stack[--top];
            auto it = tree.find(OctreeKey{pref, dep});
            if (it == tree.end() || it->second.mass == 0.0) continue;
            const OctreeNode& node = it->second;

            // skip self‐interaction at leaf
            if (dep == MAX_L && pref == key[i]) continue;

            double dx = node.comX - pos[i].x;
            double dy = node.comY - pos[i].y;
            double dz = node.comZ - pos[i].z;
            double r2 = dx*dx + dy*dy + dz*dz + soft2;

            // Barnes–Hut acceptance: cell is “small” relative to distance
            if (dep == MAX_L || side2[dep] < theta2 * r2) {
                double invR  = 1.0 / std::sqrt(r2);
                double invR3 = invR * invR * invR;
                double s     = G * node.mass * invR3;
                ax += s * dx;  ay += s * dy;  az += s * dz;
            } else {
                // open cell: push its children
                uint8_t cdep = dep + 1;
                uint64_t stride = 1ULL << (63 - 3*cdep);
                for (uint8_t oc = 0; oc < 8; ++oc) {
                    uint64_t child = pref + stride*oc;
                    if (tree.count(OctreeKey{child, cdep})) {
                        if (top < STACK_MAX) {
                            stack[top++] = {child, cdep};
                        } else {
                            throw std::runtime_error("BH stack overflow");
                        }
                    }
                }
            }
        }
        // --- PASS 2: Direct force calculation for all remote nodes ---
        for (const auto& node : remote_nodes) {
            double dx = node.comX - pos[i].x;
            double dy = node.comY - pos[i].y;
            double dz = node.comZ - pos[i].z;
            double r2 = dx*dx + dy*dy + dz*dz + soft2;

            double invR  = 1.0 / std::sqrt(r2);
            double invR3 = invR * invR * invR;
            double s     = G * node.mass * invR3;
            ax += s * dx;
            ay += s * dy;
            az += s * dz;
        }

        out[i] = {ax, ay, az};
    }
}

// solar-sim/src/traversal.cpp

// ... keep the original bhAccelerations function implementation ...

// Add this new function
// Add this new function to solar-sim/src/traversal.cpp
void bhAccelerations_dual_walk(
    const OctreeMap&                local_tree,
    const std::vector<NodeRecord>&  remote_nodes,
    const std::vector<uint64_t>&    key,
    const std::vector<Position>&    pos,
    double                          theta,
    double                          G,
    double                          soft2,
    const BoundingBox&              global_bb,
    std::vector<Acceleration>&      out)
{
    const size_t N = pos.size();
    out.resize(N);

    // Pass 1: Standard BH walk on the local tree for local-local forces.
    bhAccelerations(local_tree, key, pos, theta, G, soft2, global_bb, out);

    // Pass 2: Direct force summation over the list of remote pseudoleaves.
    // This is correct because the list no longer contains hierarchical overlaps.
    #pragma omp parallel for
    for (size_t i = 0; i < N; ++i) {
        for (const auto& r_node : remote_nodes) {
            double dx = r_node.comX - pos[i].x;
            double dy = r_node.comY - pos[i].y;
            double dz = r_node.comZ - pos[i].z;
            double r2 = dx * dx + dy * dy + dz * dz + soft2;

            if (r2 > 1e-12) {
                double invR = 1.0 / std::sqrt(r2);
                double invR3 = invR * invR * invR;
                double s = G * r_node.mass * invR3;
                out[i].x += s * dx;
                out[i].y += s * dy;
                out[i].z += s * dz;
            }
        }
    }
}