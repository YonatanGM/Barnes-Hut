#pragma once

#include <array>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <omp.h>

#include "linear_octree.h"   // OctreeMap, OctreeKey, OctreeNode
#include "body.h"            // Position, Acceleration

inline void bhAccelerations(const OctreeMap&             tree,
                            const std::vector<uint64_t>& key,   // one per body
                            const std::vector<Position>& pos,   // physical coords
                            double                       theta,
                            double                       G,
                            double                       soft2, // ε² 
                            std::vector<Acceleration>&   out)
{
    const std::size_t N = pos.size();
    out.resize(N);

    const double theta2 = theta * theta;

    // side2[d] = (cell side length)² in the unit cube
    std::array<double, 22> side2{};
    side2[0] = 1.0;
    for (int d = 1; d <= 21; ++d) side2[d] = side2[d-1] * 0.25;   // (½)²

    constexpr int STACK_MAX = 256;            // 1 + 7·21 = 148 ≤ 256

    #pragma omp parallel for schedule(dynamic,256)
    for (std::size_t i = 0; i < N; ++i)
    {
        double ax = 0.0, ay = 0.0, az = 0.0;

        std::array<std::pair<uint64_t,uint8_t>, STACK_MAX> stack;
        int top = 0;
        stack[top++] = {0ULL, 0};             // push root

        while (top)
        {
            auto [pref, dep] = stack[--top];

            auto it = tree.find(OctreeKey{pref, dep});
            if (it == tree.end() || it->second.mass == 0.0) continue;
            const OctreeNode& node = it->second;

            // self-interaction guard (one body per leaf)
            if (dep == 21 && pref == key[i]) continue;

            double dx = node.comX - pos[i].x;
            double dy = node.comY - pos[i].y;
            double dz = node.comZ - pos[i].z;
            double r2 = dx*dx + dy*dy + dz*dz + soft2;   // softened distance²

            if (dep == 21 || side2[dep] < theta2 * r2)
            {
                double invR  = 1.0 / std::sqrt(r2);
                double invR3 = invR * invR * invR;
                double s     = G * node.mass * invR3;
                ax += s * dx; ay += s * dy; az += s * dz;
            }
            else
            {
                uint8_t  cdep   = dep + 1;
                uint64_t stride = 1ULL << (63 - 3 * cdep);
                for (uint8_t oc = 0; oc < 8; ++oc)
                    if (tree.count(OctreeKey{pref + stride*oc, cdep}))
                    {
                        if (top < STACK_MAX)
                            stack[top++] = {pref + stride*oc, cdep};
                        else
                            throw std::runtime_error("BH stack overflow");
                    }
            }
        }
        out[i] = {ax, ay, az};
    }
}
