#pragma once

#include "body.h"
#include "bounding_box.h"
#include "linear_octree.h"
#include <vector>
#include <limits>

/* ------------------------------------------------------------------ */
/*  Axis-aligned bounding box                                         */
/* ------------------------------------------------------------------ */
struct BoundingBox
{
    Position min{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity() };

    Position max{
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity() };
};
/* ------------------------------------------------------------------ */
/*  Pure-geometry helpers                                             */
/* ------------------------------------------------------------------ */
double min_distance_sq(const BoundingBox& b1, const BoundingBox& b2);

BoundingBox compute_local_bbox(const std::vector<Position>& points);

BoundingBox compute_global_bbox(const BoundingBox& local_bb);

double point_box_sq(double x, double y, double z, const BoundingBox& b);

/* ------------------------------------------------------------------ */
/*  Octree helper (needs Morton code; definition lives in .cpp)       */
/* ------------------------------------------------------------------ */
BoundingBox key_to_bounding_box(const OctreeKey& cell,
                                const BoundingBox& global_bb);
