#ifndef OCTREE_H
#define OCTREE_H

#include <array>
#include <vector>
#include "body.h"

// Class representing a node in the octree
class OctreeNode {
public:
    OctreeNode(double xCenter, double yCenter, double zCenter, double size);
    ~OctreeNode();

    void insertBody(int bodyIndex, const std::vector<Body>& bodies);
    void clear();

    int bodyIndex; // Index of the body if this is a leaf node; -1 otherwise

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

private:
    int getOctant(int bodyIndex, const std::vector<Body>& bodies);
    void createChild(int index);
};


#endif // OCTREE_H
