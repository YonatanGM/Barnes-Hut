#pragma once

// Basic vector for position, velocity, and acceleration
struct Vec3 {
    double x, y, z;
};

using Position = Vec3;
using Velocity = Vec3;
using Acceleration = Vec3;