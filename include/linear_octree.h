#pragma once

#include <vector>
#include <cstdint>
#include <unordered_map>
#include "robin_hood.h"
#include "body.h"

// Defines a unique key for a node in the octree using its Morton prefix and depth.
struct OctreeKey {
    uint64_t prefix; // Top 3 * depth bits of the Morton code
    uint8_t  depth;  // Depth of the node (0=root, 21=max)

    bool operator==(const OctreeKey& other) const noexcept {
        return prefix == other.prefix && depth == other.depth;
    }
};

// Custom hash function for OctreeKey to be used in std::unordered_map.
struct OctreeKeyHash {
    size_t operator()(OctreeKey k) const noexcept {
        // A simple hash combining prefix and depth
        return std::hash<uint64_t>()(k.prefix ^ (static_cast<uint64_t>(k.depth) << 56));
    }
};

// Represents a node in the Barnes-Hut tree, storing total mass and center of mass.
struct OctreeNode {
    double mass{0.0};
    double comX{0.0};
    double comY{0.0};
    double comZ{0.0};
};

// The linear octree is represented as a hash map from a key to a node.
// using OctreeMap = std::unordered_map<OctreeKey, OctreeNode, OctreeKeyHash>;

using OctreeMap = robin_hood::unordered_flat_map<
        OctreeKey, OctreeNode, OctreeKeyHash>;

// A plain data structure for sending/receiving tree nodes over MPI.
struct NodeRecord {
    uint64_t prefix;
    uint8_t  depth;
    double   mass;
    double   comX, comY, comZ;
};


OctreeMap buildOctree1(
    const std::vector<uint64_t>& mortonCodes,
    const std::vector<Position>& positions,
    const std::vector<double>&   masses);

// Builds a linear octree from a set of bodies (defined by their codes, positions, and masses).
// This is the parallel-friendly, two-pass construction algorithm.
OctreeMap buildOctree2(
    const std::vector<uint64_t>& mortonCodes,
    const std::vector<Position>& positions,
    const std::vector<double>&   masses);

// Flattens an OctreeMap into a vector of NodeRecord for easy MPI communication.
std::vector<NodeRecord> flattenTree(const OctreeMap& tree);

// Merges a vector of NodeRecord into an existing OctreeMap.
void mergeIntoTree(OctreeMap& tree, const std::vector<NodeRecord>& records);

void printTree(const char* title, const OctreeMap& tree);

void compareFlattened(const std::vector<NodeRecord> A,
                      const std::vector<NodeRecord> B,
                      double tol = 1e-9);