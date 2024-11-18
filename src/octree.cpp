#include "octree.h"
#include <cmath>
#include <limits>


OctreeNode::OctreeNode(double xCenter, double yCenter, double zCenter, double size)
    : mass(0.0), comX(0.0), comY(0.0), comZ(0.0), size(size),
      xCenter(xCenter), yCenter(yCenter), zCenter(zCenter),
      body(nullptr), isLeaf(true) {
    children.fill(nullptr);
}

OctreeNode::~OctreeNode() {
    clear();
}

// Insert a body into the octree
void OctreeNode::insertBody(Body* newBody) {
    if (isLeaf) {
        if (body == nullptr) {
            // Leaf node is empty; insert the body here
            body = newBody;
            mass = newBody->mass;
            comX = newBody->x;
            comY = newBody->y;
            comZ = newBody->z;
        } else {
            // Leaf node already has a body; need to subdivide
            Body* existingBody = body;
            body = nullptr;
            isLeaf = false;

            // Re-insert the existing body
            int existingOctant = getOctant(existingBody);
            createChild(existingOctant);
            children[existingOctant]->insertBody(existingBody);

            // Insert the new body
            int newOctant = getOctant(newBody);
            if (children[newOctant] == nullptr) {
                createChild(newOctant);
            }
            children[newOctant]->insertBody(newBody);

            // Update mass and center of mass
            mass = existingBody->mass + newBody->mass;
            comX = (existingBody->mass * existingBody->x + newBody->mass * newBody->x) / mass;
            comY = (existingBody->mass * existingBody->y + newBody->mass * newBody->y) / mass;
            comZ = (existingBody->mass * existingBody->z + newBody->mass * newBody->z) / mass;
        }
    } else {
        // Internal node; update mass and center of mass
        mass += newBody->mass;
        comX = (comX * (mass - newBody->mass) + newBody->mass * newBody->x) / mass;
        comY = (comY * (mass - newBody->mass) + newBody->mass * newBody->y) / mass;
        comZ = (comZ * (mass - newBody->mass) + newBody->mass * newBody->z) / mass;

        // Insert the body into the appropriate child
        int octant = getOctant(newBody);
        if (children[octant] == nullptr) {
            createChild(octant);
        }
        children[octant]->insertBody(newBody);
    }
}

// Determine which octant a body belongs to
int OctreeNode::getOctant(Body* body) {
    int octant = 0;
    if (body->x >= xCenter) octant |= 1;
    if (body->y >= yCenter) octant |= 2;
    if (body->z >= zCenter) octant |= 4;
    return octant;
}

// Create a child node at the given index
void OctreeNode::createChild(int index) {
    double offset = size / 4.0;
    double childSize = size / 2.0;
    double childXCenter = xCenter + ((index & 1) ? offset : -offset);
    double childYCenter = yCenter + ((index & 2) ? offset : -offset);
    double childZCenter = zCenter + ((index & 4) ? offset : -offset);
    children[index] = new OctreeNode(childXCenter, childYCenter, childZCenter, childSize);
}

// Clear the octree node and its children
void OctreeNode::clear() {
    for (auto& child : children) {
        if (child) {
            child->clear();
            delete child;
            child = nullptr;
        }
    }
    body = nullptr;
    mass = 0.0;
    comX = comY = comZ = 0.0;
    isLeaf = true;
}
