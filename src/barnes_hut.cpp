#include "barnes_hut.h"
#include <cmath>
#include <limits>
#include <vector>
#include <omp.h>

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
                          double G, double theta, double softening) {
    OctreeNode* root = nullptr;
    buildOctree(masses, positions, root);

    size_t n = local_positions.size();
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        local_accelerations[i].ax = 0.0;
        local_accelerations[i].ay = 0.0;
        local_accelerations[i].az = 0.0;
        computeForceBarnesHut(local_masses[i], local_positions[i], local_accelerations[i], root, G, theta, softening);
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
void computeForceBarnesHut(double mass, const Position& position, Acceleration& acceleration,
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
                computeForceBarnesHut(mass, position, acceleration, node->children[i], G, theta, softening);
            }
        }
    }
}
