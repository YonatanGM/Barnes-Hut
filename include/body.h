#pragma once
#include <limits>

// Basic 3D vector for position, velocity, and acceleration
struct Vec3 {
    double x, y, z;
};

using Position = Vec3;
using Velocity = Vec3;
using Acceleration = Vec3;


struct BoundingBox {
    Position min{ std::numeric_limits<double>::infinity(),
                  std::numeric_limits<double>::infinity(),
                  std::numeric_limits<double>::infinity() };
    Position max{ -std::numeric_limits<double>::infinity(),
                  -std::numeric_limits<double>::infinity(),
                  -std::numeric_limits<double>::infinity() };
};