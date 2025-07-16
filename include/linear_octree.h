#pragma once

#include <vector>
#include <cstdint>
#include <unordered_map>
#include "robin_hood.h"
#include "body.h"

/**
 * @brief Defines a unique key for a node in the octree.
 *
 * A key is a combination of a Morton prefix and a depth. This uniquely identifies
 * any possible cell within the octree's spatial domain.
 */
struct OctreeKey {
    uint64_t prefix;
    uint8_t  depth;

    bool operator==(const OctreeKey& other) const noexcept {
        return prefix == other.prefix && depth == other.depth;
    }
};

// Custom hash function for OctreeKey to be used in std::unordered_map.
struct OctreeKeyHash {
    size_t operator()(const OctreeKey& k) const noexcept {
        // Pack depth into low bits so different depths never collide when prefixes match.
        // Then mix with a 64-bit finalizer (splitmix64 style).
        uint64_t x = k.prefix ^ (uint64_t{k.depth} * 0x9e3779b97f4a7c15ULL);
        // final avalanche
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return static_cast<size_t>(x);
    }
};

/**
 * @brief Represents a single node in the Barnes-Hut octree.
 *
 * Stores the consolidated mass properties (total mass and center of mass)
 * for all particles contained within the node's spatial volume. It also
 * tracks if the node is a pseudo-leaf from a remote rank.
 */
struct OctreeNode {
    double mass{0.0};
    double comX{0.0};
    double comY{0.0};
    double comZ{0.0};
    bool is_pseudo{false};
};

// The linear octree is represented as a hash map from a key to a node.
using OctreeMap = robin_hood::unordered_flat_map<
        OctreeKey, OctreeNode, OctreeKeyHash>;

// A plain data structure for sending/receiving tree nodes over MPI.
struct NodeRecord {
    uint64_t prefix;
    uint8_t  depth;
    double   mass;
    double   comX, comY, comZ;
};

/**
 * @brief Builds a linear octree from leaf nodes in a bottom-up fashion.
 *
 * This single pass algorithm first populates the tree with all leaf nodes
 * and then iteratively generates and aggregates parent nodes level-by-level
 * up to the root.
 *
 * @param mortonCodes A sorted vector of Morton keys for the bodies.
 * @param positions The positions of the bodies.
 * @param masses The masses of the bodies.
 * @return An OctreeMap representing the completed octree.
 */
OctreeMap buildOctreeBottomUp(
    const std::vector<uint64_t>& mortonCodes,
    const std::vector<Position>& positions,
    const std::vector<double>&   masses);

/**
 * @brief Constructs an octree from a list of received pseudo-leaves (NodeRecord).
 *
 * This function is used to reconstruct a remote tree structure on the receiving rank.
 * The nodes corresponding to the input records are marked as 'is_pseudo=true' to prevent
 * the traversal algorithm from opening them.
 *
 * @param remote_nodes A vector of nodes received from other ranks.
 * @return An OctreeMap representing the remote interaction tree.
 */
OctreeMap buildTreeFromPseudoLeaves(const std::vector<NodeRecord>& remote_nodes);


/**
 * @brief Serializes an OctreeMap into a flat vector of NodeRecord structs for communication.
 */
std::vector<NodeRecord> serializeTreeToRecords(const OctreeMap& tree);

/**
 * @brief Merges a vector of NodeRecord structs into an existing OctreeMap.
 *
 * This function adds the mass and combines the center-of-mass data for each record
 * into the corresponding node in the destination tree.
 */
void mergeRecordsIntoTree(OctreeMap& tree, const std::vector<NodeRecord>& records);

void printTree(const char* title, const OctreeMap& tree);

void compareFlattened(const std::vector<NodeRecord> A,
                      const std::vector<NodeRecord> B,
                      double tol = 1e-9);

