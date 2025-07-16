#include "body.h"
#include "linear_octree.h"
#include "morton_keys.h"
#include <vector>
#include <mpi.h>
#include <limits>
#include <algorithm>


double min_distance_sq(const BoundingBox& b1, const BoundingBox& b2) {
    // For each axis, find the separation distance. If the boxes overlap on an
    // axis, the distance is 0
    double dx = std::max(0.0, std::max(b1.min.x - b2.max.x, b2.min.x - b1.max.x));
    double dy = std::max(0.0, std::max(b1.min.y - b2.max.y, b2.min.y - b1.max.y));
    double dz = std::max(0.0, std::max(b1.min.z - b2.max.z, b2.min.z - b1.max.z));

    return dx * dx + dy * dy + dz * dz;
}

/**
 * @brief Computes the tight bounding box for a vector of local positions.
 */
BoundingBox compute_local_bbox(const std::vector<Position>& local_pos) {
    BoundingBox local_bb;
    if (local_pos.empty()) {
        return local_bb;
    }

    // initialize with the first point
    local_bb.min = local_pos[0];
    local_bb.max = local_pos[0];

    // expand the box to include all other points
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

BoundingBox compute_global_bbox(const BoundingBox& local_bb) {
    BoundingBox global_bb;
    MPI_Allreduce(&local_bb.min, &global_bb.min, 3, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_bb.max, &global_bb.max, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    return global_bb;
}


BoundingBox getBoundingBoxForCell(const OctreeKey& k,
                                const BoundingBox& global_bb) {
    // Get the real world dimensions of the global box
    double dx = global_bb.max.x - global_bb.min.x;
    double dy = global_bb.max.y - global_bb.min.y;
    double dz = global_bb.max.z - global_bb.min.z;

    // Calculate the real world size of a cell at this depth
    double sizeX = dx / (1 << k.depth);
    double sizeY = dy / (1 << k.depth);
    double sizeZ = dz / (1 << k.depth);

    Position norm_min = key_to_normalized_position(k.prefix, k.depth);

    // Scale the normalized coordinates to the real-world minimum corner position
    double min_x = global_bb.min.x + norm_min.x * dx;
    double min_y = global_bb.min.y + norm_min.y * dy;
    double min_z = global_bb.min.z + norm_min.z * dz;

    return {{min_x, min_y, min_z}, {min_x + sizeX, min_y + sizeY, min_z + sizeZ}};
}

double pointBoxDistanceSq(double x,double y,double z, const BoundingBox& b) {
    /* distance along each axis (0 if inside) */
    double dx = std::max(0.0, std::max(b.min.x - x, x - b.max.x));
    double dy = std::max(0.0, std::max(b.min.y - y, y - b.max.y));
    double dz = std::max(0.0, std::max(b.min.z - z, z - b.max.z));
    return dx*dx + dy*dy + dz*dz;
}


