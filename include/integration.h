#ifndef INTEGRATION_H
#define INTEGRATION_H

#include <vector>
#include "body.h"

/**
 * @brief Performs leapfrog integration on the local bodies.
 *
 * @param positions     Vector of positions.
 * @param velocities    Vector of velocities.
 * @param accelerations Vector of accelerations.
 * @param dt            Time step.
 */
void leapfrogIntegration(std::vector<Position>& positions,
                         std::vector<Velocity>& velocities,
                         const std::vector<Acceleration>& accelerations,
                         double dt);

#endif // INTEGRATION_H
