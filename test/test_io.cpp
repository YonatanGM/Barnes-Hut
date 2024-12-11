#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include "body.h"

// // include function prototypes
// bool readCSV(const std::string& filename,
//              std::vector<double>& masses,
//              std::vector<Position>& positions,
//              std::vector<Velocity>& velocities,
//              std::vector<std::string>& names);

// void saveState(const std::string& vs_dir, int vs_counter,
//                const std::vector<double>& masses,
//                const std::vector<Position>& positions,
//                const std::vector<Velocity>& velocities);

// namespace fs = std::filesystem;

// TEST(ReadCSVTest, EmptyFile) {
//     std::string tempFilename = "temp_empty.csv";
//     std::ofstream tempFile(tempFilename);
//     tempFile.close();

//     std::vector<double> masses;
//     std::vector<Position> positions;
//     std::vector<Velocity> velocities;
//     std::vector<std::string> names;   

//     EXPECT_FALSE(readCSV(tempFilename, masses, positions, velocities, names));

//     fs::remove(tempFilename);
// }

// TEST(ReadCSVTest, IncorrectDataTypes) {
//     std::string tempFilename = "temp_incorrect_types.csv";
//     std::ofstream tempFile(tempFilename);
//     tempFile << "mass,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z\n";
//     tempFile << "abc,0.0,0.0,0.0,1.0,0.0,0.0\n"; // Non-numeric mass
//     tempFile.close();

//     std::vector<double> masses;
//     std::vector<Position> positions;
//     std::vector<Velocity> velocities;
//     std::vector<std::string> names;   

//     EXPECT_TRUE(readCSV(tempFilename, masses, positions, velocities, names));
//     EXPECT_EQ(masses.size(), 0); // Should skip the invalid line

//     fs::remove(tempFilename);
// }

// TEST(SaveStateTest, InvalidDirectory) {
//     std::string vs_dir = "/invalid/directory/path";
//     std::vector<double> masses = {1.0};
//     std::vector<Position> positions = {{0, 0, 0}};
//     std::vector<Velocity> velocities = {{1, 0, 0}};

//     // Should not throw an exception but should handle the error internally
//     saveState(vs_dir, 0, masses, positions, velocities);
// }

