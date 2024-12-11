#ifndef BARNES_HUT_H
#define BARNES_HUT_H

#include <vector>
#include "body.h"
#include "octree.h"

/**
 * @brief Computes accelerations for local bodies using the Barnes-Hut algorithm.
 *
 * @param masses               Vector of all masses.
 * @param positions            Vector of all positions.
 * @param local_masses         Vector of local masses.
 * @param local_positions      Vector of local positions.
 * @param local_accelerations  Vector to store computed accelerations for local bodies.
 * @param G                    Gravitational constant.
 * @param theta                Barnes-Hut opening angle parameter.
 * @param softening            Softening parameter to prevent singularities.
 */
void computeAccelerations(const std::vector<double>& masses,
                          const std::vector<Position>& positions,
                          const std::vector<double>& local_masses,
                          const std::vector<Position>& local_positions,
                          std::vector<Acceleration>& local_accelerations,
                          double G, double theta, double softening);

/**
 * @brief Computes the gravitational force on a single body
 *
 * @param mass           Mass of the body.
 * @param position       Position of the body.
 * @param acceleration    Acceleration to be updated based on the computed force.
 * @param node            Current node in the octree.
 * @param G               Gravitational constant.
 * @param theta           Barnes-Hut opening angle parameter.
 * @param softening       Softening parameter to prevent singularities.
 */
void computeForceOnBody(double mass, const Position& position, Acceleration& acceleration,
                           OctreeNode* node, double G, double theta, double softening);

/**
 * @brief Builds the Barnes-Hut octree.
 *
 * Constructs the octree by inserting all bodies into the appropriate nodes.
 *
 * @param masses     Vector of masses of the bodies.
 * @param positions  Vector of positions of the bodies.
 * @param root       Reference to the pointer of the root node (will be created).
 */
void buildOctree(const std::vector<double>& masses, const std::vector<Position>& positions, OctreeNode*& root);

/**
 * @brief Builds the octree in parallel.
 *
 * Constructs the Barnes-Hut octree by recursively dividing the space
 * and inserting bodies into the tree. Parallelism is achieved using
 * OpenMP tasks, and task creation is controlled based on the current
 * depth to balance overhead and performance.
 *
 * @param masses     Vector of masses of the bodies.
 * @param positions  Vector of positions of the bodies.
 * @param root       Reference to the pointer of the root node (will be created).
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
 * @param node                 Pointer to the current octree node.
 * @param masses               Vector of masses of the bodies.
 * @param positions            Vector of positions of the bodies.
 * @param bodyIndices          Indices of the bodies to be inserted into this node.
 * @param currentDepth         Current depth in the octree.
 * @param MAX_DEPTH_FOR_TASKS  Maximum depth to create tasks.
 */
void buildOctreeNode(OctreeNode* node,
                     const std::vector<double>& masses,
                     const std::vector<Position>& positions,
                     const std::vector<int>& bodyIndices,
                     int currentDepth, 
                     int MAX_DEPTH_FOR_TASKS);

/**
 * @brief Computes the global kinetic and potential energies of the N-body system.
 *
 * This function calculates the kinetic and potential energies by aggregating
 * contributions from all MPI processes. It uses OpenMP for parallel computation
 * within each process and MPI_Allreduce to sum the local energies globally.
 *
 * @param masses               Vector of masses for all bodies.
 * @param positions            Vector of positions for all bodies.
 * @param local_velocities     Vector of velocities for bodies assigned to this process.
 * @param G                    Gravitational constant.
 * @param rank                 MPI process rank.
 * @param size                 Total number of MPI processes.
 * @param displs               Displacements for each process's data segment.
 * @param sendcounts           Number of elements each process handles.
 * @param local_n              Number of bodies handled by this process.
 * @param kinetic_energy       Reference to store the computed global kinetic energy.
 * @param potential_energy     Reference to store the computed global potential energy.
 * @param total_energy         Reference to store the sum of kinetic and potential energies.
 * @param virial_equilibrium   Reference to store the virial equilibrium ratio.
 */
void computeGlobalEnergiesParallel(const std::vector<double>& masses,
                                   const std::vector<Position>& positions,
                                   const std::vector<Velocity>& local_velocities,
                                   double G,
                                   int rank, int size,
                                   const std::vector<int>& displs,
                                   const std::vector<int>& sendcounts,
                                   int local_n,
                                   double& kinetic_energy,
                                   double& potential_energy,
                                   double& total_energy,
                                   double& virial_equilibrium);

#endif // BARNES_HUT_H
