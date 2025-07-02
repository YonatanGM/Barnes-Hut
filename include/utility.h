#pragma once
#include <vector>
#include <numeric>
#include <algorithm>
#include <limits>
#include <mpi.h>
#include "body.h"
#include "morton_keys.h"

// Sorts all local particle data arrays in-place based on Morton key order.
inline void sort_local_data(
    CodesAndNorm&         cn,
    std::vector<Position>& pos,
    std::vector<double>&   mass,
    std::vector<Velocity>& vel,
    std::vector<uint64_t>& ids)
{
    const size_t N = pos.size();
    if (N == 0) return;

    std::vector<size_t> sort_idx(N);
    std::iota(sort_idx.begin(), sort_idx.end(), 0);

    std::sort(sort_idx.begin(), sort_idx.end(), [&](size_t a, size_t b) {
        return cn.code[a] < cn.code[b];
    });

    auto permute = [&](auto &vec) {
        std::vector<std::decay_t<decltype(vec[0])>> tmp(vec.size());
        for (size_t i = 0; i < vec.size(); ++i) tmp[i] = vec[sort_idx[i]];
        vec.swap(tmp);
    };

    // Permute all data arrays to match the sorted key order
    permute(pos);
    permute(mass);
    permute(vel);
    permute(ids);
    permute(cn.code);
    permute(cn.norm);
}

inline BoundingBox compute_global_bbox(const std::vector<Position>& local_pos)
{
    BoundingBox local_bb;
    // Each rank finds the bounding box of its own local particles
    if (!local_pos.empty()) {
        for (const auto &p : local_pos) {
            local_bb.min.x = std::min(local_bb.min.x, p.x);
            local_bb.min.y = std::min(local_bb.min.y, p.y);
            local_bb.min.z = std::min(local_bb.min.z, p.z);
            local_bb.max.x = std::max(local_bb.max.x, p.x);
            local_bb.max.y = std::max(local_bb.max.y, p.y);
            local_bb.max.z = std::max(local_bb.max.z, p.z);
        }
    }

    BoundingBox global_bb;
    // Use MPI_Allreduce to find the global min and max corners
    MPI_Allreduce(&local_bb.min, &global_bb.min, 3, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_bb.max, &global_bb.max, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    return global_bb;
}