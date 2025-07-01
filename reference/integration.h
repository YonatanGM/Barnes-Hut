#pragma once

#include "body.h"
#include <vector>

// Performs one step of leapfrog integration (Kick-Drift-Kick).
// This variant assumes it's part of a KDK loop where the initial kick
// for the step has already been applied. It performs the drift and the
// second kick.
inline void leapfrogIntegration(
    std::vector<Position>& positions,
    std::vector<Velocity>& velocities,
    const std::vector<Acceleration>& accelerations,
    double dt) {
    
    const size_t n = positions.size();

    // Note: A full KDK loop would look like:
    // 1. Kick (update v by a*dt/2)
    // 2. Drift (update p by v*dt)
    // 3. Compute new accelerations a'
    // 4. Kick (update v by a'*dt/2)
    
    // This function performs steps 2 and a modified step 1/4 combined.
    // It's a common simplification in main loops.
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        // First, half-step velocity update (kick)
        velocities[i].x += accelerations[i].x * dt * 0.5;
        velocities[i].y += accelerations[i].y * dt * 0.5;
        velocities[i].z += accelerations[i].z * dt * 0.5;

        // Full-step position update (drift)
        positions[i].x += velocities[i].x * dt;
        positions[i].y += velocities[i].y * dt;
        positions[i].z += velocities[i].z * dt;
    }
}