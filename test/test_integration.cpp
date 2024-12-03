#include <gtest/gtest.h>
#include <vector>
#include "body.h"

void leapfrogIntegration(std::vector<Position>& positions,
                         std::vector<Velocity>& velocities,
                         const std::vector<Acceleration>& accelerations,
                         double dt);

TEST(LeapfrogIntegrationTest, ZeroAcceleration) {
    std::vector<Position> positions = {{0, 0, 0}};
    std::vector<Velocity> velocities = {{1, 2, 3}};
    std::vector<Acceleration> accelerations = {{0, 0, 0}};
    double dt = 1.0;

    leapfrogIntegration(positions, velocities, accelerations, dt);

    EXPECT_DOUBLE_EQ(positions[0].x, 1.0);
    EXPECT_DOUBLE_EQ(positions[0].y, 2.0);
    EXPECT_DOUBLE_EQ(positions[0].z, 3.0);

    EXPECT_DOUBLE_EQ(velocities[0].vx, 1.0);
    EXPECT_DOUBLE_EQ(velocities[0].vy, 2.0);
    EXPECT_DOUBLE_EQ(velocities[0].vz, 3.0);
}

TEST(LeapfrogIntegrationTest, NegativeAcceleration) {
    std::vector<Position> positions = {{0, 0, 0}};
    std::vector<Velocity> velocities = {{1, 0, 0}};
    std::vector<Acceleration> accelerations = {{-1, 0, 0}};
    double dt = 1.0;

    leapfrogIntegration(positions, velocities, accelerations, dt);

    EXPECT_DOUBLE_EQ(positions[0].x, 0.25);
    EXPECT_DOUBLE_EQ(velocities[0].vx, 0.5);
}

TEST(LeapfrogIntegrationTest, MultipleBodies) {
    std::vector<Position> positions = {{0, 0, 0}, {1, 1, 1}};
    std::vector<Velocity> velocities = {{1, 0, 0}, {0, 1, 0}};
    std::vector<Acceleration> accelerations = {{0, 1, 0}, {1, 0, 0}};
    double dt = 1.0;

    leapfrogIntegration(positions, velocities, accelerations, dt);

    EXPECT_DOUBLE_EQ(positions[0].x, 1.0);
    EXPECT_DOUBLE_EQ(positions[0].y, 1.0);
    EXPECT_DOUBLE_EQ(positions[0].z, 0.0);

    EXPECT_DOUBLE_EQ(velocities[0].vx, 1.0);
    EXPECT_DOUBLE_EQ(velocities[0].vy, 0.5);
}

