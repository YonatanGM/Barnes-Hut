#include "barnes_hut.h"
#include <cmath>
#include <limits>
#include <vector>
#include <omp.h>
#include <mpi.h>
#include <atomic>
#include <iostream>
#include <algorithm>


// Computes accelerations for local bodies using the Barnes-Hut algorithm.
void computeAccelerations(const std::vector<double>& masses,
                          const std::vector<Position>& positions,
                          const std::vector<double>& local_masses,
                          const std::vector<Position>& local_positions,
                          std::vector<Acceleration>& local_accelerations,
                          double G, double theta, double softening) {
    OctreeNode* root = nullptr;
    buildOctree(masses, positions, root);
    // Print the total number of tasks created and maximum depth
    // std::cout << "Total tasks created: " << totalTasksCreated.load() << std::endl;
    // std::cout << "Maximum depth reached: " << maxDepthReached.load() << std::endl;
    size_t n = local_positions.size();
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        local_accelerations[i].ax = 0.0;
        local_accelerations[i].ay = 0.0;
        local_accelerations[i].az = 0.0;
        computeForceOnBody(local_masses[i], local_positions[i], local_accelerations[i], root, G, theta, softening);
    }

    delete root;
}



/**
 * @brief Computes forces on a body using the Barnes-Hut algorithm.
 *
 * @param mass         Mass of the body.
 * @param position     Position of the body.
 * @param acceleration Acceleration to be updated.
 * @param node         Pointer to the current octree node.
 * @param G            Gravitational constant.
 * @param theta        Barnes-Hut opening angle parameter.
 * @param softening    Softening parameter to prevent singularities.
 */
void computeForceOnBody(double mass, const Position& position, Acceleration& acceleration,
                           OctreeNode* node, double G, double theta, double softening) {
    if (node == nullptr) {
        return;
    }

    double dx = node->comX - position.x;
    double dy = node->comY - position.y;
    double dz = node->comZ - position.z;
    double distSqr = dx * dx + dy * dy + dz * dz + softening * softening;
    double distance = sqrt(distSqr);

    if (node->isLeaf) {
        if (node->bodyIndex >= 0 && node->bodyIndex != -1) {
            // Avoid self-interaction
            if (dx == 0.0 && dy == 0.0 && dz == 0.0) {
                return;
            }

            double invDist = 1.0 / distance;
            double invDist3 = invDist * invDist * invDist;
            double force = G * node->mass * invDist3;

            acceleration.ax += force * dx;
            acceleration.ay += force * dy;
            acceleration.az += force * dz;
        }
    } else {
        if ((node->size / distance) < theta) {
            double invDist = 1.0 / distance;
            double invDist3 = invDist * invDist * invDist;
            double force = G * node->mass * invDist3;

            acceleration.ax += force * dx;
            acceleration.ay += force * dy;
            acceleration.az += force * dz;
        } else {
            for (int i = 0; i < 8; ++i) {
                computeForceOnBody(mass, position, acceleration, node->children[i], G, theta, softening);
            }
        }
    }
}


/**
 * @brief Builds the octree from the list of bodies.
 *
 * @param masses    Vector of masses.
 * @param positions Vector of positions.
 * @param root      Reference to the root node pointer (will be created).
 */
void buildOctree(const std::vector<double>& masses, const std::vector<Position>& positions, OctreeNode*& root) {
    // Determine the bounding cube that contains all bodies
    double xMin = std::numeric_limits<double>::max();
    double xMax = std::numeric_limits<double>::lowest();
    double yMin = xMin, yMax = xMax;
    double zMin = xMin, zMax = xMax;

    for (const auto& pos : positions) {
        if (pos.x < xMin) xMin = pos.x;
        if (pos.x > xMax) xMax = pos.x;
        if (pos.y < yMin) yMin = pos.y;
        if (pos.y > yMax) yMax = pos.y;
        if (pos.z < zMin) zMin = pos.z;
        if (pos.z > zMax) zMax = pos.z;
    }

    double size = std::max({ xMax - xMin, yMax - yMin, zMax - zMin });
    double xCenter = (xMin + xMax) / 2.0;
    double yCenter = (yMin + yMax) / 2.0;
    double zCenter = (zMin + zMax) / 2.0;

    // Create the root node
    if (root != nullptr) {
        root->clear();
        delete root;
    }
    root = new OctreeNode(xCenter, yCenter, zCenter, size);

    // Insert bodies into the octree
    for (size_t i = 0; i < positions.size(); ++i) {
        root->insertBody(i, masses, positions);
    }
}

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
                                   double& virial_equilibrium) {
    size_t N = masses.size();
    int start_i = displs[rank];
    int end_i = start_i + local_n;

    double local_kinetic = 0.0;
    double local_potential = 0.0;

    // Compute local kinetic energy
    #pragma omp parallel for reduction(+:local_kinetic)
    for (int i = 0; i < local_n; i++) {
        double v2 = local_velocities[i].vx * local_velocities[i].vx +
                    local_velocities[i].vy * local_velocities[i].vy +
                    local_velocities[i].vz * local_velocities[i].vz;
        local_kinetic += 0.5 * masses[start_i + i] * v2;
    }

    // accumulate potential energy, ensuring each pair is counted only once
    // If rank 0 handles bodies 0-1, it computes pairs: (0,1), (0,2), (0,3), (1,2), (1,3)
    // If rank 2 handles bodies 2-3, it computes pair: (2,3)
    #pragma omp parallel for reduction(+:local_potential) schedule(dynamic)
    for (int i = start_i; i < end_i; i++) {
        for (int j = i + 1; j < (int)N; j++) {
            double dx = positions[j].x - positions[i].x;
            double dy = positions[j].y - positions[i].y;
            double dz = positions[j].z - positions[i].z;
            double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (dist > 0.0) {
                local_potential += -G * masses[i] * masses[j] / dist;
            }
        }
    }

    // Aggregate energies globally
    double global_kinetic = 0.0, global_potential = 0.0;
    MPI_Allreduce(&local_kinetic, &global_kinetic, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_potential, &global_potential, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    kinetic_energy = global_kinetic;
    potential_energy = global_potential;
    total_energy = kinetic_energy + potential_energy;
    virial_equilibrium = (potential_energy != 0.0) ? -2.0 * (kinetic_energy / potential_energy) : 0.0;
}


/**
 * @brief Builds the octree in parallel.
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
                 OctreeNode*& root) {
    // determine the bounding cube that contains all bodies
    double xMin = std::numeric_limits<double>::max();
    double xMax = std::numeric_limits<double>::lowest();
    double yMin = xMin, yMax = xMax;
    double zMin = xMin, zMax = xMax;

    int num_bodies = positions.size();

    for (const auto& pos : positions) {
        if (pos.x < xMin) xMin = pos.x;
        if (pos.x > xMax) xMax = pos.x;
        if (pos.y < yMin) yMin = pos.y;
        if (pos.y > yMax) yMax = pos.y;
        if (pos.z < zMin) zMin = pos.z;
        if (pos.z > zMax) zMax = pos.z;
    }

    double size = std::max({ xMax - xMin, yMax - yMin, zMax - zMin });
    double xCenter = (xMin + xMax) / 2.0;
    double yCenter = (yMin + yMax) / 2.0;
    double zCenter = (zMin + zMax) / 2.0;

    // create the root node
    if (root != nullptr) {
        root->clear();
        delete root;
    }
    root = new OctreeNode(xCenter, yCenter, zCenter, size);

    // indices of bodies to insert
    std::vector<int> bodyIndices(num_bodies);
    for (int i = 0; i < num_bodies; ++i) {
        bodyIndices[i] = i;
    }

    // enable nested parallelism for OpenMP
    // omp_set_nested(1);
    

    // determine max depth for task creation based on max available threads
    int maxThreads = omp_get_max_threads();
    int MAX_DEPTH_FOR_TASKS = 0;
    int temp = 1;
    while (temp * 8 <= maxThreads) {
        temp *= 8; // each level increases the potential number of tasks by up to 8
        MAX_DEPTH_FOR_TASKS++;
    }

    // std::cout << "MAX: " << MAX_DEPTH_FOR_TASKS << std::endl;
    
    omp_set_max_active_levels(MAX_DEPTH_FOR_TASKS); 
    // start the parallel region
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            // Begin building the octree from the root node
            buildOctreeNode(root, masses, positions, bodyIndices, 0, MAX_DEPTH_FOR_TASKS);
        }
    }
}

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
                     int MAX_DEPTH_FOR_TASKS ) {
    if (bodyIndices.empty()) {
        return;
    }

    if (bodyIndices.size() == 1) {
        // Only one body; insert it into this node
        int idx = bodyIndices[0];
        node->insertBody(idx, masses, positions);
        return;
    } else {
        node->isLeaf = false;
        node->bodyIndex = -1;
    }

    // Divide bodies among the 8 octants
    std::vector<std::vector<int>> octantIndices(8);
    for (int idx : bodyIndices) {
        int octant = node->getOctant(positions[idx]);
        octantIndices[octant].push_back(idx);
    }

    double offset = node->size / 4.0;
    double childSize = node->size / 2.0;

    // Count non-empty child nodes
    int childCount = 0;
    for (const auto& indices : octantIndices) {
        if (!indices.empty()) {
            ++childCount;
        }
    }

    // Process each child node
    int processedChildren = 0;
    for (int i = 0; i < 8; ++i) {
        if (!octantIndices[i].empty()) {
            double childXCenter = node->xCenter + ((i & 1) ? offset : -offset);
            double childYCenter = node->yCenter + ((i & 2) ? offset : -offset);
            double childZCenter = node->zCenter + ((i & 4) ? offset : -offset);

            // Create the child node
            node->children[i] = new OctreeNode(childXCenter, childYCenter, childZCenter, childSize);

            if (currentDepth < MAX_DEPTH_FOR_TASKS) {
                // Create a task for this child
                // #pragma omp task default(none) firstprivate(i, currentDepth, MAX_DEPTH_FOR_TASKS) shared(node, masses, positions, octantIndices)
                #pragma omp task final(currentDepth >= MAX_DEPTH_FOR_TASKS) mergeable \
                default(none) firstprivate(i, currentDepth) shared(node, masses, positions, octantIndices, MAX_DEPTH_FOR_TASKS)
                {
                    buildOctreeNode(node->children[i], masses, positions, octantIndices[i], currentDepth + 1, MAX_DEPTH_FOR_TASKS);
                }
            } else {
                // Process in the current thread
                buildOctreeNode(node->children[i], masses, positions, octantIndices[i], currentDepth + 1, MAX_DEPTH_FOR_TASKS);
            }

            ++processedChildren;
        }
    }

    // Wait for all tasks to complete
    #pragma omp taskwait

    // Update node's mass and center of mass
    node->mass = 0.0;
    node->comX = 0.0;
    node->comY = 0.0;
    node->comZ = 0.0;

    for (int i = 0; i < 8; ++i) {
        if (node->children[i] != nullptr) {
            node->mass += node->children[i]->mass;
            node->comX += node->children[i]->mass * node->children[i]->comX;
            node->comY += node->children[i]->mass * node->children[i]->comY;
            node->comZ += node->children[i]->mass * node->children[i]->comZ;
        }
    }

    if (node->mass > 0.0) {
        node->comX /= node->mass;
        node->comY /= node->mass;
        node->comZ /= node->mass;
    }
}


//
// We want to determine the maximum depth to create tasks so that we don't create more tasks than the
// available number of threads, to avoid excessive overhead.
//
// Since at each level of the octree, a node can have up to 8 children, the maximum number of tasks
// at depth 'd' is 8^d.
//
// We want the total number of tasks not to exceed the max number of threads. So we find the maximum
// depth 'D' such that 8^D <= maxThreads.
//
// The calculation is done by starting with temp = 1, and multiplying temp by 8 until temp * 8 exceeds
// maxThreads. The value of MAX_DEPTH_FOR_TASKS is the depth 'D' at which we should stop creating new tasks.
//
// For example, if maxThreads = 64:
//
// temp = 1         (depth 0)
// temp = 8         (depth 1)
// temp = 64        (depth 2)
// temp = 512       (depth 3) // Exceeds maxThreads
//
// So MAX_DEPTH_FOR_TASKS = 2 (we can create tasks up to depth 2)

//
// Octree Construction with Parallelism:
//
// Each node can have up to 8 children (octants).
//
// Level 0: Root Node
//          |
//          +-- Level 1: Up to 8 child nodes (tasks created if currentDepth < MAX_DEPTH_FOR_TASKS)
//                |
//                +-- Level 2: Up to 8 child nodes per parent node
//                      |
//                      +-- Level 3: Up to 8 child nodes per parent node
//                            |
//                            ...
//
// At each level, the number of nodes (and potential tasks) increases by a factor of up to 8.
//
// The depth at which we stop creating new tasks is determined by MAX_DEPTH_FOR_TASKS, calculated based
// on the maximum number of available threads. This ensures that we don't create more tasks than can be
// efficiently handled.
//
// By limiting task creation to this depth, we balance between utilizing available threads and avoiding
// the overhead of creating too many tasks.
//
// The bodies are inserted recursively into the octree, and the mass and center of mass are computed
// after all child nodes have been processed.