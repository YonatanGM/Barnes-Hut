#include <array>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <omp.h>

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
