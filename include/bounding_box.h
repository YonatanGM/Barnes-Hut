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

// solar-sim/include/bounding_box.h

inline BoundingBox key_to_bounding_box(const OctreeKey& k,
                                       const BoundingBox& global_bb)
{
    // FIX: Use anisotropic dimensions, not a single cubic 'L'.
    double dx = global_bb.max.x - global_bb.min.x;
    double dy = global_bb.max.y - global_bb.min.y;
    double dz = global_bb.max.z - global_bb.min.z;

    // The size of a cell at depth 'd' is the parent size / 2.
    double sizeX = dx / (1 << k.depth);
    double sizeY = dy / (1 << k.depth);
    double sizeZ = dz / (1 << k.depth);

    // Calculate the minimum corner of the box from the Morton prefix.
    double x = global_bb.min.x;
    double y = global_bb.min.y;
    double z = global_bb.min.z;

    double stepX = dx;
    double stepY = dy;
    double stepZ = dz;

    for (int d = 0; d < k.depth; ++d) {
        stepX /= 2.0;
        stepY /= 2.0;
        stepZ /= 2.0;
        int oct = (k.prefix >> (63 - 3*(d+1))) & 7;
        if (oct & 1) x += stepX;
        if (oct & 2) y += stepY;
        if (oct & 4) z += stepZ;
    }

    return {{x, y, z}, {x + sizeX, y + sizeY, z + sizeZ}};
}

inline double point_box_sq(double x,double y,double z, const BoundingBox& b)
{
    /* distance along each axis (0 if inside) */
    double dx = std::max(0.0, std::max(b.min.x - x, x - b.max.x));
    double dy = std::max(0.0, std::max(b.min.y - y, y - b.max.y));
    double dz = std::max(0.0, std::max(b.min.z - z, z - b.max.z));
    return dx*dx + dy*dy + dz*dz;
}

