#include "integration.h"
#include <omp.h>

void leapfrogIntegration(std::vector<Position>& positions,
                         std::vector<Velocity>& velocities,
                         const std::vector<Acceleration>& accelerations,
                         double dt) {
    int n = velocities.size();

    
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {

        // First, half-step velocity updates
        velocities[i].vx += accelerations[i].ax * dt * 0.5;
        velocities[i].vy += accelerations[i].ay * dt * 0.5;
        velocities[i].vz += accelerations[i].az * dt * 0.5;
        
        // Update positions by a full step 
        positions[i].x += velocities[i].vx * dt;
        positions[i].y += velocities[i].vy * dt;
        positions[i].z += velocities[i].vz * dt;
    }



    // Accelerations will be recomputed after updating positions
}
