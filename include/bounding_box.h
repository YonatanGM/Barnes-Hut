#pragma once
#include "body.h"
#include "linear_octree.h"

#include <vector>
#include <mpi.h>
#include <limits>
#include <algorithm>

/**
 * @brief Represents an axis-aligned bounding box (AABB) in 3D space.
 */
struct BoundingBox {
    Position min{ std::numeric_limits<double>::infinity(),
                  std::numeric_limits<double>::infinity(),
                  std::numeric_limits<double>::infinity() };
    Position max{ -std::numeric_limits<double>::infinity(),
                  -std::numeric_limits<double>::infinity(),
                  -std::numeric_limits<double>::infinity() };
};

/**
 * @brief Calculates the squared minimum Euclidean distance between two AABBs.
 *
 * @param b1 The first bounding box.
 * @param b2 The second bounding box.
 * @return The squared minimum distance between the boxes.
 */
inline double min_distance_sq(const BoundingBox& b1, const BoundingBox& b2) {
    // For each axis, find the separation distance. If the boxes overlap on an
    // axis, the distance is 0.
    double dx = std::max(0.0, std::max(b1.min.x - b2.max.x, b2.min.x - b1.max.x));
    double dy = std::max(0.0, std::max(b1.min.y - b2.max.y, b2.min.y - b1.max.y));
    double dz = std::max(0.0, std::max(b1.min.z - b2.max.z, b2.min.z - b1.max.z));

    return dx * dx + dy * dy + dz * dz;
}

/**
 * @brief Computes the tight bounding box for a vector of local positions.
 */
inline BoundingBox compute_local_bbox(const std::vector<Position>& local_pos) {
    BoundingBox local_bb;
    if (local_pos.empty()) {
        // Return a default-constructed "invalid" box if there are no points.
        // The MPI_Allreduce will correctly handle this.
        return local_bb;
    }

    // Initialize with the first point
    local_bb.min = local_pos[0];
    local_bb.max = local_pos[0];

    // Expand the box to include all other points
    for (size_t i = 1; i < local_pos.size(); ++i) {
        const auto& p = local_pos[i];
        local_bb.min.x = std::min(local_bb.min.x, p.x);
        local_bb.min.y = std::min(local_bb.min.y, p.y);
        local_bb.min.z = std::min(local_bb.min.z, p.z);
        local_bb.max.x = std::max(local_bb.max.x, p.x);
        local_bb.max.y = std::max(local_bb.max.y, p.y);
        local_bb.max.z = std::max(local_bb.max.z, p.z);
    }
    return local_bb;
}

inline BoundingBox compute_global_bbox(const BoundingBox& local_bb) {
    BoundingBox global_bb;
    // Use MPI_Allreduce to find the global min and max corners
    MPI_Allreduce(&local_bb.min, &global_bb.min, 3, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_bb.max, &global_bb.max, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    return global_bb;
}

inline BoundingBox key_to_bounding_box(const OctreeKey& key, const BoundingBox& global_bb) {
    const double size = (global_bb.max.x - global_bb.min.x) / (1 << key.depth);
    double x = global_bb.min.x;
    double y = global_bb.min.y;
    double z = global_bb.min.z;

    uint64_t prefix = key.prefix;
    for (int d = 0; d < key.depth; ++d) {
        // Extract the 3 bits for this level from the Morton code
        int octant = (prefix >> (63 - 3 * (d + 1))) & 0x7;
        double step = (global_bb.max.x - global_bb.min.x) / (2 << d);
        if (octant & 1) x += step; // Check x-bit
        if (octant & 2) y += step; // Check y-bit
        if (octant & 4) z += step; // Check z-bit
    }

    return {{x, y, z}, {x + size, y + size, z + size}};
}
