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
