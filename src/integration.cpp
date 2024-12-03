#include "integration.h"
#include <omp.h>


/**
 * @brief Performs leapfrog integration on the given positions and velocities.
 *
 * This function updates velocities by a half step and positions by a full step using the leapfrog integration method.
 * It uses the provided accelerations and time step.
 *
 * @param positions Vector of positions to be updated.
 * @param velocities Vector of velocities to be updated.
 * @param accelerations Vector of accelerations.
 * @param dt Time step for the integration.
 */
void leapfrogIntegration(std::vector<Position>& positions,
                         std::vector<Velocity>& velocities,
                         const std::vector<Acceleration>& accelerations,
                         double dt) {
    int n = velocities.size();

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        // Only update velocity if acceleration is non-zero
        if (accelerations[i].ax != 0 || accelerations[i].ay != 0 || accelerations[i].az != 0) {
            // Half-step velocity updates based on acceleration
            velocities[i].vx += accelerations[i].ax * dt * 0.5;
            velocities[i].vy += accelerations[i].ay * dt * 0.5;
            velocities[i].vz += accelerations[i].az * dt * 0.5;
        }

        // Regardless of acceleration, update position based on velocity
        positions[i].x += velocities[i].vx * dt;
        positions[i].y += velocities[i].vy * dt;
        positions[i].z += velocities[i].vz * dt;
    }

    // Accelerations will be recomputed after updating positions if necessary
}
