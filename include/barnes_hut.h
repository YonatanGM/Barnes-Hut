#ifndef BARNES_HUT_H
#define BARNES_HUT_H

#include "body.h"
#include "octree.h"

// Build the octree from the list of bodies
void buildOctree(const std::vector<Body*>& bodies, OctreeNode*& root);

// Compute forces on a body using the Barnes-Hut algorithm
void computeForcesBarnesHut(Body& body, OctreeNode* node, double theta, double G, double softening);

#endif // BARNES_HUT_H
