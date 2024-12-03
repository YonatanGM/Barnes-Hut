#include <gtest/gtest.h>
#include <vector>
#include "body.h"
#include "octree.h"



class OctreeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup can be done here if needed
    }

    void TearDown() override {
        // Common cleanup can be done here if needed
    }
};

TEST_F(OctreeTest, ConstructorInitialization) {
    double xCenter = 0.0;
    double yCenter = 0.0;
    double zCenter = 0.0;
    double size = 100.0;

    OctreeNode node(xCenter, yCenter, zCenter, size);

    EXPECT_DOUBLE_EQ(node.xCenter, xCenter);
    EXPECT_DOUBLE_EQ(node.yCenter, yCenter);
    EXPECT_DOUBLE_EQ(node.zCenter, zCenter);
    EXPECT_DOUBLE_EQ(node.size, size);
    EXPECT_EQ(node.mass, 0.0);
    EXPECT_EQ(node.bodyIndex, -1);
    EXPECT_TRUE(node.isLeaf);
    for (auto child : node.children) {
        EXPECT_EQ(child, nullptr);
    }
}

TEST_F(OctreeTest, GetOctantAllCases) {
    OctreeNode node(0.0, 0.0, 0.0, 100.0);

    Position positions[8] = {
        {-1, -1, -1}, // Octant 0
        {1, -1, -1},  // Octant 1
        {-1, 1, -1},  // Octant 2
        {1, 1, -1},   // Octant 3
        {-1, -1, 1},  // Octant 4
        {1, -1, 1},   // Octant 5
        {-1, 1, 1},   // Octant 6
        {1, 1, 1}     // Octant 7
    };

    for (int i = 0; i < 8; ++i) {
        int octant = node.getOctant(positions[i]);
        EXPECT_EQ(octant, i);
    }
}

TEST_F(OctreeTest, CreateChildCorrectly) {
    OctreeNode node(0.0, 0.0, 0.0, 100.0);

    int octant = 5;
    node.createChild(octant);

    EXPECT_NE(node.children[octant], nullptr);

    double expectedOffset = 100.0 / 4.0;
    double expectedSize = 100.0 / 2.0;

    EXPECT_DOUBLE_EQ(node.children[octant]->xCenter, node.xCenter + expectedOffset);
    EXPECT_DOUBLE_EQ(node.children[octant]->yCenter, node.yCenter - expectedOffset);
    EXPECT_DOUBLE_EQ(node.children[octant]->zCenter, node.zCenter + expectedOffset);
    EXPECT_DOUBLE_EQ(node.children[octant]->size, expectedSize);
}

TEST_F(OctreeTest, InsertSingleBody) {
    OctreeNode node(0.0, 0.0, 0.0, 100.0);
    std::vector<double> masses = {10.0};
    std::vector<Position> positions = {{10.0, 0.0, 0.0}};

    node.insertBody(0, masses, positions);

    EXPECT_EQ(node.bodyIndex, 0);
    EXPECT_DOUBLE_EQ(node.mass, 10.0);
    EXPECT_DOUBLE_EQ(node.comX, 10.0);
    EXPECT_DOUBLE_EQ(node.comY, 0.0);
    EXPECT_DOUBLE_EQ(node.comZ, 0.0);
    EXPECT_TRUE(node.isLeaf);
}

TEST_F(OctreeTest, InsertTwoBodiesSameOctant) {
    OctreeNode node(0.0, 0.0, 0.0, 100.0);
    std::vector<double> masses = {10.0, 20.0};
    std::vector<Position> positions = {{10.0, 10.0, 10.0}, {15.0, 15.0, 15.0}};

    node.insertBody(0, masses, positions);
    node.insertBody(1, masses, positions);

    EXPECT_EQ(node.bodyIndex, -1);
    EXPECT_FALSE(node.isLeaf);
    EXPECT_DOUBLE_EQ(node.mass, 30.0);

    // Center of mass calculation
    double comX = (10.0 * 10.0 + 20.0 * 15.0) / 30.0;
    double comY = (10.0 * 10.0 + 20.0 * 15.0) / 30.0;
    double comZ = (10.0 * 10.0 + 20.0 * 15.0) / 30.0;

    EXPECT_DOUBLE_EQ(node.comX, comX);
    EXPECT_DOUBLE_EQ(node.comY, comY);
    EXPECT_DOUBLE_EQ(node.comZ, comZ);
}

TEST_F(OctreeTest, InsertBodiesDifferentOctants) {
    OctreeNode node(0.0, 0.0, 0.0, 100.0);
    std::vector<double> masses = {10.0, 20.0};
    std::vector<Position> positions = {{-10.0, -10.0, -10.0}, {10.0, 10.0, 10.0}};

    node.insertBody(0, masses, positions);
    node.insertBody(1, masses, positions);

    EXPECT_EQ(node.bodyIndex, -1);
    EXPECT_FALSE(node.isLeaf);
    EXPECT_DOUBLE_EQ(node.mass, 30.0);

    double comX = (10.0 * (-10.0) + 20.0 * 10.0) / 30.0;
    double comY = (10.0 * (-10.0) + 20.0 * 10.0) / 30.0;
    double comZ = (10.0 * (-10.0) + 20.0 * 10.0) / 30.0;

    EXPECT_DOUBLE_EQ(node.comX, comX);
    EXPECT_DOUBLE_EQ(node.comY, comY);
    EXPECT_DOUBLE_EQ(node.comZ, comZ);

    // Check that children are created correctly
    int octant0 = node.getOctant(positions[0]);
    int octant1 = node.getOctant(positions[1]);

    EXPECT_NE(octant0, octant1);
    EXPECT_NE(node.children[octant0], nullptr);
    EXPECT_NE(node.children[octant1], nullptr);

    EXPECT_EQ(node.children[octant0]->bodyIndex, 0);
    EXPECT_EQ(node.children[octant1]->bodyIndex, 1);
}

TEST_F(OctreeTest, ClearNode) {
    OctreeNode node(0.0, 0.0, 0.0, 100.0);
    std::vector<double> masses = {10.0};
    std::vector<Position> positions = {{10.0, 0.0, 0.0}};

    node.insertBody(0, masses, positions);
    node.clear();

    EXPECT_EQ(node.bodyIndex, -1);
    EXPECT_EQ(node.mass, 0.0);
    EXPECT_TRUE(node.isLeaf);
    for (auto child : node.children) {
        EXPECT_EQ(child, nullptr);
    }
}

TEST_F(OctreeTest, DeepInsertion) {
    OctreeNode node(0.0, 0.0, 0.0, 100.0);
    std::vector<double> masses = {10.0, 20.0, 30.0, 40.0};
    std::vector<Position> positions = {
        {10.0, 10.0, 10.0},
        {15.0, 15.0, 15.0},
        {17.0, 17.0, 17.0},
        {19.0, 19.0, 19.0}
    };

    for (int i = 0; i < masses.size(); ++i) {
        node.insertBody(i, masses, positions);
    }

    EXPECT_EQ(node.bodyIndex, -1);
    EXPECT_FALSE(node.isLeaf);
    EXPECT_DOUBLE_EQ(node.mass, 100.0);

    // Since all bodies are in the same octant, the tree should have multiple levels
    int octant = node.getOctant(positions[0]);
    OctreeNode* child = node.children[octant];
    EXPECT_NE(child, nullptr);
    EXPECT_FALSE(child->isLeaf);
}

