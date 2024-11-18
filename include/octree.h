// octree.h

#ifndef OCTREE_H
#define OCTREE_H

#include <array>
#include "body.h"

// Class representing a node in the octree
class OctreeNode {
public:
    OctreeNode(double xCenter, double yCenter, double zCenter, double size);
    ~OctreeNode();

    // Insert a body into the octree
    void insertBody(Body* body);

    // Clear the octree node and its children
    void clear();

    // Getters
    double getMass() const { return mass; }
    double getCOMX() const { return comX; }
    double getCOMY() const { return comY; }
    double getCOMZ() const { return comZ; }
    double getSize() const { return size; }
    bool isLeafNode() const { return isLeaf; }
    OctreeNode* getChild(int index) const { return children[index]; }

    // Pointer to body if this is a leaf node
    Body* body;

private:
    // Mass and center of mass
    double mass;
    double comX, comY, comZ;

    // Spatial properties
    double size;
    double xCenter, yCenter, zCenter;

    // Children nodes
    std::array<OctreeNode*, 8> children;

    // Flag indicating if the node is a leaf
    bool isLeaf;

    // Determine which octant a body belongs to
    int getOctant(Body* body);

    // Create a child node at the given index
    void createChild(int index);
};

#endif // OCTREE_H
