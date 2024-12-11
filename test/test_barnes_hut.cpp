#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "barnes_hut.h"
#include "body.h"
#include <mpi.h>

class BarnesHutTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup
        G = 1.0;       // Simplify G for testing
        theta = 0.5;
        softening = 0.1;
    }

    void TearDown() override {
        // Common cleanup
    }

    double G;
    double theta;
    double softening;
};

TEST_F(BarnesHutTest, ComputeForceBarnesHutSingleBody) {
    // Create a simple tree with one other body
    OctreeNode* root = new OctreeNode(0.0, 0.0, 0.0, 100.0);
    std::vector<double> masses = {10.0};
    std::vector<Position> positions = {{10.0, 0.0, 0.0}};
    root->insertBody(0, masses, positions);

    double mass = 5.0;
    Position position = {0.0, 0.0, 0.0};
    Acceleration acceleration = {0.0, 0.0, 0.0};

    computeForceOnBody(mass, position, acceleration, root, G, theta, softening);

    // Compute expected acceleration directly
    double dx = positions[0].x - position.x;
    double dy = positions[0].y - position.y;
    double dz = positions[0].z - position.z;
    double distSqr = dx * dx + dy * dy + dz * dz + softening * softening;
    double distance = sqrt(distSqr);
    double invDist3 = 1.0 / (distance * distance * distance);
    double expectedForce = G * masses[0] * invDist3;

    EXPECT_NEAR(acceleration.ax, expectedForce * dx, 1e-6);
    EXPECT_NEAR(acceleration.ay, expectedForce * dy, 1e-6);
    EXPECT_NEAR(acceleration.az, expectedForce * dz, 1e-6);

    delete root;
}

TEST_F(BarnesHutTest, ComputeAccelerationsTwoBodies) {
    std::vector<double> masses = {10.0, 20.0};
    std::vector<Position> positions = {{-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};

    std::vector<double> local_masses = {10.0};
    std::vector<Position> local_positions = {{-1.0, 0.0, 0.0}};
    std::vector<Acceleration> local_accelerations(1);

    computeAccelerations(masses, positions, local_masses, local_positions, local_accelerations, G, theta, softening);

    // Compute expected acceleration directly
    double dx = positions[1].x - positions[0].x;
    double dy = positions[1].y - positions[0].y;
    double dz = positions[1].z - positions[0].z;
    double distSqr = dx * dx + dy * dy + dz * dz + softening * softening;
    double distance = sqrt(distSqr);
    double invDist3 = 1.0 / (distance * distance * distance);
    double expectedForce = G * masses[1] * invDist3;

    EXPECT_NEAR(local_accelerations[0].ax, expectedForce * dx, 1e-6);
    EXPECT_NEAR(local_accelerations[0].ay, expectedForce * dy, 1e-6);
    EXPECT_NEAR(local_accelerations[0].az, expectedForce * dz, 1e-6);
}

TEST_F(BarnesHutTest, BuildOctreeSimple) {
    std::vector<double> masses = {10.0, 20.0};
    std::vector<Position> positions = {{-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0}};
    OctreeNode* root = nullptr;

    buildOctree(masses, positions, root);

    EXPECT_NE(root, nullptr);
    EXPECT_DOUBLE_EQ(root->mass, 30.0);

    // Center of mass
    double comX = (10.0 * (-1.0) + 20.0 * 1.0) / 30.0;
    double comY = (10.0 * (-1.0) + 20.0 * 1.0) / 30.0;
    double comZ = (10.0 * (-1.0) + 20.0 * 1.0) / 30.0;

    EXPECT_DOUBLE_EQ(root->comX, comX);
    EXPECT_DOUBLE_EQ(root->comY, comY);
    EXPECT_DOUBLE_EQ(root->comZ, comZ);

    delete root;
}

TEST_F(BarnesHutTest, BuildOctreeParallelSimple) {
    std::vector<double> masses = {10.0, 20.0, 15.0};
    std::vector<Position> positions = {
        {-1.0, -1.0, -1.0},
        {1.0, 1.0, 1.0},
        {0.5, 0.5, 0.5}
    };
    OctreeNode* root = nullptr;

    buildOctreeParallel(masses, positions, root);

    EXPECT_NE(root, nullptr);
    EXPECT_DOUBLE_EQ(root->mass, 45.0);

    delete root;
}

TEST_F(BarnesHutTest, ComputeAccelerationsMultipleBodies) {
    std::vector<double> masses = {10.0, 20.0, 15.0};
    std::vector<Position> positions = {
        {-1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0}
    };

    std::vector<double> local_masses = {10.0};
    std::vector<Position> local_positions = {{-1.0, 0.0, 0.0}};
    std::vector<Acceleration> local_accelerations(1);

    computeAccelerations(masses, positions, local_masses, local_positions, local_accelerations, G, theta, softening);

    // Compute expected acceleration directly by summing forces from other bodies
    Acceleration expectedAcc = {0.0, 0.0, 0.0};
    for (size_t i = 1; i < positions.size(); ++i) {
        double dx = positions[i].x - positions[0].x;
        double dy = positions[i].y - positions[0].y;
        double dz = positions[i].z - positions[0].z;
        double distSqr = dx * dx + dy * dy + dz * dz + softening * softening;
        double distance = sqrt(distSqr);
        double invDist3 = 1.0 / (distance * distance * distance);
        double force = G * masses[i] * invDist3;
        expectedAcc.ax += force * dx;
        expectedAcc.ay += force * dy;
        expectedAcc.az += force * dz;
    }

    EXPECT_NEAR(local_accelerations[0].ax, expectedAcc.ax, 1e-6);
    EXPECT_NEAR(local_accelerations[0].ay, expectedAcc.ay, 1e-6);
    EXPECT_NEAR(local_accelerations[0].az, expectedAcc.az, 1e-6);
}

// TEST_F(BarnesHutTest, ComputeGlobalEnergiesParallel) {

//     std::vector<double> masses = {10.0, 20.0, 15.0};
//     std::vector<Position> positions = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
//     std::vector<Velocity> velocities = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

//     double kinetic_energy = 0.0;
//     double potential_energy = 0.0;
//     double total_energy = 0.0;
//     double virial_equilibrium = 0.0;

//     int rank = 0, size = 1;
//     std::vector<int> displs = {0};
//     std::vector<int> sendcounts = {3};
//     int local_n = 3;
//     MPI_Init(nullptr, nullptr); // Initialize MPI
//     computeGlobalEnergiesParallel(masses, positions, velocities, G, rank, size, displs, sendcounts, local_n,
//                                    kinetic_energy, potential_energy, total_energy, virial_equilibrium);

//     double expected_kinetic_energy = 0.5 * (10.0 + 20.0 + 15.0);
//     EXPECT_NEAR(kinetic_energy, expected_kinetic_energy, 1e-6);
//     EXPECT_LE(potential_energy, 0.0); // Potential energy should be non-positive
//     EXPECT_NEAR(total_energy, kinetic_energy + potential_energy, 1e-6);
//     EXPECT_GT(virial_equilibrium, 0.0); // Virial ratio should be positive
// }

TEST_F(BarnesHutTest, BuildOctreeComplex) {
    std::vector<double> masses = {10.0, 20.0, 15.0, 25.0};
    std::vector<Position> positions = {
        {-1.0, -1.0, -1.0},
        {1.0, 1.0, 1.0},
        {0.5, 0.5, 0.5},
        {-0.5, -0.5, -0.5}
    };

    OctreeNode* root = nullptr;
    buildOctree(masses, positions, root);

    EXPECT_NE(root, nullptr);
    EXPECT_DOUBLE_EQ(root->mass, 70.0);

    double comX = (-10.0 + 20.0 * 1.0 + 15.0 * 0.5 - 25.0 * 0.5) / 70.0;
    double comY = comX;
    double comZ = comX;

    EXPECT_DOUBLE_EQ(root->comX, comX);
    EXPECT_DOUBLE_EQ(root->comY, comY);
    EXPECT_DOUBLE_EQ(root->comZ, comZ);

    delete root;
}
