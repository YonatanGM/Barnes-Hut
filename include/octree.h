#ifndef OCTREE_H
#define OCTREE_H

#include <array>
#include <vector>
#include <body.h>

/**
 * @brief Class representing a node in the octree.
 */
class OctreeNode {
public:
    double mass;      ///< Total mass of bodies in this node
    double comX;      ///< Center of mass X-coordinate
    double comY;      ///< Center of mass Y-coordinate
    double comZ;      ///< Center of mass Z-coordinate
    double size;      ///< Size of the node (length of one side)
    double xCenter;   ///< X-coordinate of the node's center
    double yCenter;   ///< Y-coordinate of the node's center
    double zCenter;   ///< Z-coordinate of the node's center
    int bodyIndex;    ///< Index of the body if this is a leaf node
    bool isLeaf;      ///< Flag indicating if this is a leaf node

    std::array<OctreeNode*, 8> children; ///< Children of this node

    /**
     * @brief Constructor for OctreeNode.
     *
     * @param xCenter X-coordinate of the node's center
     * @param yCenter Y-coordinate of the node's center
     * @param zCenter Z-coordinate of the node's center
     * @param size    Size of the node (length of one side)
     */
    OctreeNode(double xCenter, double yCenter, double zCenter, double size);

    /**
     * @brief Destructor for OctreeNode.
     */
    ~OctreeNode();

    /**
     * @brief Inserts a body into the octree node.
     *
     * @param bodyIndex Index of the body to insert
     * @param masses    Vector of masses
     * @param positions Vector of positions
     */
    void insertBody(int bodyIndex, const std::vector<double>& masses, const std::vector<Position>& positions);

    /**
     * @brief Clears the octree node and its children.
     */
    void clear();

private:
    /**
     * @brief Determines the octant for a body relative to this node.
     *
     * @param position Position of the body
     * @return Octant index (0-7)
     */
    int getOctant(const Position& position);

    /**
     * @brief Creates a child node in the specified octant.
     *
     * @param index Octant index (0-7)
     */
    void createChild(int index);
};

#endif // OCTREE_H
