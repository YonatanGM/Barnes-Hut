#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include "body.h"
#include <gtest/gtest.h>
#include "io.h"
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>

class IOTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir = "test_data";
        std::filesystem::create_directory(test_dir);
    }

    void TearDown() override {
        std::filesystem::remove_all(test_dir);
    }

    std::string test_dir;
};

TEST_F(IOTest, ReadCSV_ValidFile) {
    std::string csv_file = test_dir + "/test.csv";
    std::ofstream file(csv_file);
    file << "id,name,class,mass,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z\n";
    file << "1,Body1,ClassA,5.0,1.0,2.0,3.0,0.1,0.2,0.3\n";
    file << "2,Body2,ClassB,10.0,4.0,5.0,6.0,0.4,0.5,0.6\n";
    file.close();

    std::vector<double> masses;
    std::vector<Position> positions;
    std::vector<Velocity> velocities;
    std::vector<std::string> names;
    std::vector<int> orbit_classes;

    ASSERT_TRUE(readCSV(csv_file, masses, positions, velocities, names, orbit_classes, -1));

    EXPECT_EQ(masses.size(), 2);
    EXPECT_EQ(masses[0], 5.0);
    EXPECT_EQ(masses[1], 10.0);

    EXPECT_EQ(positions[0].x, 1.0);
    EXPECT_EQ(positions[1].y, 5.0);

    EXPECT_EQ(velocities[0].vx, 0.1);
    EXPECT_EQ(velocities[1].vz, 0.6);

    EXPECT_EQ(names[0], "Body1");
    EXPECT_EQ(names[1], "Body2");

    EXPECT_EQ(orbit_classes.size(), 2);
    EXPECT_NE(orbit_classes[0], orbit_classes[1]);
}

TEST_F(IOTest, SaveState_ValidFile) {
    std::string output_dir = test_dir;
    int counter = 1;

    std::vector<double> masses = {5.0, 10.0};
    std::vector<Position> positions = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    std::vector<Velocity> velocities = {{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}};

    saveState(output_dir, counter, masses, positions, velocities);

    std::string saved_file = output_dir + "/output_00001.csv";
    ASSERT_TRUE(std::filesystem::exists(saved_file));

    std::ifstream file(saved_file);
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    EXPECT_NE(content.find("id,mass,x,y,z,vx,vy,vz"), std::string::npos);
    EXPECT_NE(content.find("0,5,1,2,3,0.1,0.2,0.3"), std::string::npos);
    EXPECT_NE(content.find("1,10,4,5,6,0.4,0.5,0.6"), std::string::npos);
}

TEST_F(IOTest, WriteVTPFile_ValidFile) {
    int rank = 0;
    int counter = 1;
    std::vector<double> masses = {5.0, 10.0};
    std::vector<Position> positions = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    std::vector<Velocity> velocities = {{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}};
    std::vector<Acceleration> accelerations = {{0.01, 0.02, 0.03}, {0.04, 0.05, 0.06}};
    std::vector<std::string> names = {"Body1", "Body2"};
    std::vector<int> orbit_classes = {0, 1};

    std::string output_dir = test_dir;

    writeVTPFile(rank, counter, masses, positions, velocities, accelerations, names, orbit_classes, 0,
                 5.0, -10.0, -5.0, 2.0, output_dir);

    std::string vtp_file = output_dir + "/0/sim.1.vtp";
    ASSERT_TRUE(std::filesystem::exists(vtp_file));
}

TEST_F(IOTest, UpdatePVDFile_ValidFile) {
    std::string pvd_file = "simulation.pvd";
    double current_time = 0.0;

    updatePVDFile(pvd_file, 1, 0, current_time, test_dir);

    std::string saved_file = test_dir + "/simulation.pvd";
    ASSERT_TRUE(std::filesystem::exists(saved_file));

    std::ifstream file(saved_file);
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    EXPECT_NE(content.find("VTKFile"), std::string::npos);
    EXPECT_NE(content.find("Collection"), std::string::npos);
}