#ifndef BARNES_HUT_H
#define BARNES_HUT_H

#include <vector>
#include "body.h"
#include "octree.h"

/**
 * @brief Computes accelerations for local bodies using the Barnes-Hut algorithm.
 *
 * @param masses           Vector of all masses.
 * @param positions        Vector of all positions.
 * @param local_masses     Vector of local masses.
 * @param local_positions  Vector of local positions.
 * @param local_accelerations Vector to store computed accelerations for local bodies.
 * @param G                Gravitational constant.
 * @param theta            Barnes-Hut opening angle parameter.
 * @param softening        Softening parameter to prevent singularities.
 */
void computeAccelerations(const std::vector<double>& masses,
                          const std::vector<Position>& positions,
                          const std::vector<double>& local_masses,
                          const std::vector<Position>& local_positions,
                          std::vector<Acceleration>& local_accelerations,
                          double G, double theta, double softening);



void computeForceBarnesHut(double mass, const Position& position, Acceleration& acceleration,
                           OctreeNode* node, double G, double theta, double softening);

void buildOctree(const std::vector<double>& masses, const std::vector<Position>& positions, OctreeNode*& root);


#endif 



