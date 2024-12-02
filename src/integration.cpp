#include "integration.h"

// Leapfrog integration method
void leapfrogIntegration(std::vector<Body>& bodies, double dt) {
    // First half-step: update velocities
    for (auto& body : bodies) {
        body.vx += body.ax * dt * 0.5;
        body.vy += body.ay * dt * 0.5;
        body.vz += body.az * dt * 0.5;
    }


    // Full-step: update positions
    for (auto& body : bodies) {
        body.x += body.vx * dt;
        body.y += body.vy * dt;
        body.z += body.vz * dt;
    }

    // Accelerations will be recomputed after updating positions
}
