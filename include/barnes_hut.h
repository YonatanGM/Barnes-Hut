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


/**
 * @brief Builds the octree using parallelism.
 *
 * Constructs the Barnes-Hut octree by recursively dividing the space
 * and inserting bodies into the tree. Parallelism is achieved using
 * OpenMP tasks, and task creation is controlled based on the current
 * depth to balance overhead and performance.
 *
 * @param masses Vector of masses of the bodies.
 * @param positions Vector of positions of the bodies.
 * @param root Reference to the pointer of the root node (will be created).
 */
void buildOctreeParallel(const std::vector<double>& masses,
                 const std::vector<Position>& positions,
                 OctreeNode*& root);

/**
 * @brief Recursively builds the octree nodes.
 *
 * Divides the bodies among child nodes and recursively builds the octree.
 * Uses OpenMP tasks to parallelize the construction up to a certain depth.
 *
 * @param node Pointer to the current octree node.
 * @param masses Vector of masses of the bodies.
 * @param positions Vector of positions of the bodies.
 * @param bodyIndices Indices of the bodies to be inserted into this node.
 * @param currentDepth Current depth in the octree.
 * @param MAX_DEPTH_FOR_TASKS Maximum depth to create tasks.
 */
void buildOctreeNode(OctreeNode* node,
                     const std::vector<double>& masses,
                     const std::vector<Position>& positions,
                     const std::vector<int>& bodyIndices,
                     int currentDepth, 
                     int MAX_DEPTH_FOR_TASKS);

#endif // BARNES_HUT_H



