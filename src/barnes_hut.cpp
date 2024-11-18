#include "barnes_hut.h"
#include <cmath>
#include <limits>

// Build the octree from the list of bodies
void buildOctree(const std::vector<Body*>& bodies, OctreeNode*& root) {
    // Determine the bounding cube that contains all bodies
    double xMin = std::numeric_limits<double>::max();
    double xMax = std::numeric_limits<double>::lowest();
    double yMin = xMin, yMax = xMax;
    double zMin = xMin, zMax = xMax;

    for (const auto& body : bodies) {
        if (body->x < xMin) xMin = body->x;
        if (body->x > xMax) xMax = body->x;
        if (body->y < yMin) yMin = body->y;
        if (body->y > yMax) yMax = body->y;
        if (body->z < zMin) zMin = body->z;
        if (body->z > zMax) zMax = body->z;
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
    for (const auto& body : bodies) {
        root->insertBody(body);
    }
}

// Compute forces on a body using the Barnes-Hut algorithm
void computeForcesBarnesHut(Body& body, OctreeNode* node, double theta, double G, double softening) {
    if (node == nullptr || (node->isLeafNode() && node->body == &body)) {
        return;
    }

    // Compute distance between the body and the node's center of mass
    double dx = node->getCOMX() - body.x;
    double dy = node->getCOMY() - body.y;
    double dz = node->getCOMZ() - body.z;
    double distSqr = dx * dx + dy * dy + dz * dz + (softening * softening);
    double distance = sqrt(distSqr);

    if (node->isLeafNode() && node->body != &body) {
        // Node is a leaf and not the same body; compute direct force
        double invDist = 1.0 / distance;
        double invDist3 = invDist * invDist * invDist;
        double force = G * body.mass * node->getMass() * invDist3;

        body.ax += force * dx / body.mass;
        body.ay += force * dy / body.mass;
        body.az += force * dz / body.mass;
    } else if (!node->isLeafNode()) {
        if ((node->getSize() / distance) < theta) {
            // Node is sufficiently far, approximate as a single body
            double invDist = 1.0 / distance;
            double invDist3 = invDist * invDist * invDist;
            double force = G * body.mass * node->getMass() * invDist3;

            body.ax += force * dx / body.mass;
            body.ay += force * dy / body.mass;
            body.az += force * dz / body.mass;
        } else {
            // Node too close, recurse into children
            for (int i = 0; i < 8; ++i) {
                computeForcesBarnesHut(body, node->getChild(i), theta, G, softening);
            }
        }
    }
}
