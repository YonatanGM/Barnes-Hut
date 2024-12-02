#include "integration.h"

// Leapfrog integration method
void leapfrogIntegration(std::vector<Body>& bodies, double dt) {

    #pragma omp parallel for
    for (size_t i = 0; i < bodies.size(); ++i) {
        // First, half-step velocity updates
        bodies[i].vx += bodies[i].ax * dt * 0.5;
        bodies[i].vy += bodies[i].ay * dt * 0.5;
        bodies[i].vz += bodies[i].az * dt * 0.5;
        
        // update positions by a full step
        bodies[i].x += bodies[i].vx * dt;
        bodies[i].y += bodies[i].vy * dt;
        bodies[i].z += bodies[i].vz * dt;
    }

    // Accelerations will be recomputed after updating positions
    // and we do another half-step velocity update 
}
