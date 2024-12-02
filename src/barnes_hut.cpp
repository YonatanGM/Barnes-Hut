// barnes_hut.cpp
#include "barnes_hut.h"
#include <cmath>
#include <limits>
#include <vector>

// Build the octree from the list of bodies
void buildOctree(const std::vector<Body>& bodies, OctreeNode*& root) {
    // Determine the bounding cube that contains all bodies
    double xMin = std::numeric_limits<double>::max();
    double xMax = std::numeric_limits<double>::lowest();
    double yMin = xMin, yMax = xMax;
    double zMin = xMin, zMax = xMax;

    for (const auto& body : bodies) {
        if (body.x < xMin) xMin = body.x;
        if (body.x > xMax) xMax = body.x;
        if (body.y < yMin) yMin = body.y;
        if (body.y > yMax) yMax = body.y;
        if (body.z < zMin) zMin = body.z;
        if (body.z > zMax) zMax = body.z;
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
    for (size_t i = 0; i < bodies.size(); ++i) {
        root->insertBody(static_cast<int>(i), bodies);
    }
}


void computeAccelerations(std::vector<Body>& bodies, std::vector<Body>& local_bodies, double G, double theta, double softening) {
    OctreeNode* root = nullptr;
    buildOctree(bodies, root);

    for (auto& body : local_bodies) {
        body.ax = body.ay = body.az = 0.0;
        computeForcesBarnesHut(body, root, theta, G, softening, bodies);
    }

    delete root;
}

// Compute forces on a body using the Barnes-Hut algorithm
void computeForcesBarnesHut(Body& body, OctreeNode* node, double theta, double G, double softening, const std::vector<Body>& bodies) {
    if (node == nullptr || (node->isLeaf && node->bodyIndex == body.index)) {
        return;
    }

    double dx = node->comX - body.x;
    double dy = node->comY - body.y;
    double dz = node->comZ - body.z;
    double distSqr = dx * dx + dy * dy + dz * dz + softening * softening;
    double distance = sqrt(distSqr);

    if (node->isLeaf && node->bodyIndex != body.index) {
        const Body& otherBody = bodies[node->bodyIndex];
        double invDist = 1.0 / distance;
        double invDist3 = invDist * invDist * invDist;
        double force = G * body.mass * otherBody.mass * invDist3;

        body.ax += force * dx / body.mass;
        body.ay += force * dy / body.mass;
        body.az += force * dz / body.mass;
    } else if (!node->isLeaf) {
        if ((node->size / distance) < theta) {
            double invDist = 1.0 / distance;
            double invDist3 = invDist * invDist * invDist;
            double force = G * body.mass * node->mass * invDist3;

            body.ax += force * dx / body.mass;
            body.ay += force * dy / body.mass;
            body.az += force * dz / body.mass;
        } else {
            for (int i = 0; i < 8; ++i) {
                computeForcesBarnesHut(body, node->children[i], theta, G, softening, bodies);
            }
        }
    }
}


