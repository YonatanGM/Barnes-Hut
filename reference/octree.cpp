#include "octree.h"

/**
 * @brief Constructor for OctreeNode.
 *
 * @param xCenter X-coordinate of the node's center
 * @param yCenter Y-coordinate of the node's center
 * @param zCenter Z-coordinate of the node's center
 * @param size    Size of the node (length of one side)
 */
OctreeNode::OctreeNode(double xCenter, double yCenter, double zCenter, double size)
    : mass(0.0), comX(0.0), comY(0.0), comZ(0.0), size(size),
      xCenter(xCenter), yCenter(yCenter), zCenter(zCenter),
      bodyIndex(-1), isLeaf(true) {
    children.fill(nullptr);
}

/**
 * @brief Destructor for OctreeNode.
 */
OctreeNode::~OctreeNode() {
    clear();
}

/**
 * @brief Inserts a body into the octree node.
 *
 * @param newBodyIndex Index of the body to insert
 * @param masses       Vector of masses
 * @param positions    Vector of positions
 */
void OctreeNode::insertBody(int newBodyIndex, const std::vector<double>& masses, const std::vector<Position>& positions) {
    const Position& newBodyPos = positions[newBodyIndex];
    double newBodyMass = masses[newBodyIndex];

    if (isLeaf) {
        if (bodyIndex == -1) {
            // Leaf node is empty; insert the body here
            bodyIndex = newBodyIndex;
            mass = newBodyMass;
            comX = newBodyPos.x;
            comY = newBodyPos.y;
            comZ = newBodyPos.z;
        } else {
            // Leaf node already has a body; need to subdivide
            int existingBodyIndex = bodyIndex;
            const Position& existingBodyPos = positions[existingBodyIndex];
            double existingBodyMass = masses[existingBodyIndex];
            bodyIndex = -1;
            isLeaf = false;

            // Re-insert the existing body
            int existingOctant = getOctant(existingBodyPos);
            createChild(existingOctant);
            children[existingOctant]->insertBody(existingBodyIndex, masses, positions);

            // Insert the new body
            int newOctant = getOctant(newBodyPos);
            if (children[newOctant] == nullptr) {
                createChild(newOctant);
            }
            children[newOctant]->insertBody(newBodyIndex, masses, positions);

            // Update mass and center of mass
            mass = existingBodyMass + newBodyMass;
            comX = (existingBodyMass * existingBodyPos.x + newBodyMass * newBodyPos.x) / mass;
            comY = (existingBodyMass * existingBodyPos.y + newBodyMass * newBodyPos.y) / mass;
            comZ = (existingBodyMass * existingBodyPos.z + newBodyMass * newBodyPos.z) / mass;
        }
    } else {
        // Internal node; update mass and center of mass
        mass += newBodyMass;
        comX = (comX * (mass - newBodyMass) + newBodyMass * newBodyPos.x) / mass;
        comY = (comY * (mass - newBodyMass) + newBodyMass * newBodyPos.y) / mass;
        comZ = (comZ * (mass - newBodyMass) + newBodyMass * newBodyPos.z) / mass;

        // Insert the body into the appropriate child
        int octant = getOctant(newBodyPos);
        if (children[octant] == nullptr) {
            createChild(octant);
        }
        children[octant]->insertBody(newBodyIndex, masses, positions);
    }
}

/**
 * @brief Determines the octant for a body relative to this node.
 *
 * @param position Position of the body
 * @return Octant index (0-7)
 */
int OctreeNode::getOctant(const Position& position) {
    int octant = 0;
    if (position.x >= xCenter) octant |= 1;
    if (position.y >= yCenter) octant |= 2;
    if (position.z >= zCenter) octant |= 4;
    return octant;
}

/**
 * @brief Creates a child node in the specified octant.
 *
 * @param index Octant index (0-7)
 */
void OctreeNode::createChild(int index) {
    double offset = size / 4.0;
    double childSize = size / 2.0;
    double childXCenter = xCenter + ((index & 1) ? offset : -offset);
    double childYCenter = yCenter + ((index & 2) ? offset : -offset);
    double childZCenter = zCenter + ((index & 4) ? offset : -offset);
    children[index] = new OctreeNode(childXCenter, childYCenter, childZCenter, childSize);
}

/**
 * @brief Clears the octree node and its children.
 */
void OctreeNode::clear() {
    for (auto& child : children) {
        if (child) {
            child->clear();
            delete child;
            child = nullptr;
        }
    }
    bodyIndex = -1;
    mass = 0.0;
    comX = comY = comZ = 0.0;
    isLeaf = true;
}
