#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <unordered_map>
#include <algorithm>
#include <filesystem> // C++17 filesystem library
#include "body.h"
#include "io.h"
#include "barnes_hut.h"
#include "octree.h"
#include "integration.h"

namespace fs = std::filesystem;

// Function to parse time strings like "1h", "2.5d", "1y" into days
double parseTime(const std::string& timeStr) {
    double timeValue = std::stod(timeStr.substr(0, timeStr.size() - 1));
    char unit = timeStr.back();
    if (unit == 'h') {
        return timeValue / 24.0; // Convert hours to days
    } else if (unit == 'd') {
        return timeValue; // Days
    } else if (unit == 'y') {
        return timeValue * 365.25; // Convert years to days
    } else {
        std::cerr << "Unknown time unit: " << unit << std::endl;
        exit(1);
    }
}

// Function to parse command-line arguments
std::unordered_map<std::string, std::string> parseArguments(int argc, char* argv[]) {
    std::unordered_map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--", 0) == 0) {
            std::string key = arg.substr(2);
            if ((i + 1) < argc && argv[i + 1][0] != '-') {
                args[key] = argv[++i];
            } else {
                args[key] = ""; // Flag without value
            }
        } else {
            std::cerr << "Invalid argument: " << arg << std::endl;
            exit(1);
        }
    }
    return args;
}


void computeAccelerations(std::vector<Body*>& bodies, double G, double theta, double softening) {
    // Build the octree
    OctreeNode* root = nullptr;
    buildOctree(bodies, root);

    // Reset accelerations and compute forces in a single loop
    for (auto& body : bodies) {
        body->ax = 0.0;
        body->ay = 0.0;
        body->az = 0.0;

        // Compute forces using Barnes-Hut
        computeForcesBarnesHut(*body, root, theta, G, softening);
    }

    // Clean up
    delete root;
}


int main(int argc, char* argv[]) {
    // Adjusted G in AU^3 kg^-1 day^-2
    const double G = 1.48812e-34;
    double softening = 1e-11; // Softening parameter in AU
    double theta = 0.5; // Barnes-Hut threshold parameter
    double dt = 1.0; // Time step in days
    double t_end = 365.25; // End time in days (default 1 year)
    double vs = 1.0; // Visualization step width in days
    std::string vs_dir = "sim"; // Visualization output directory
    std::string filename; // Input file

    // Parse command-line arguments
    auto args = parseArguments(argc, argv);

    // Process arguments
    if (args.find("file") != args.end()) {
        filename = args["file"];
    } else {
        std::cerr << "Input file not specified. Use --file <filename>" << std::endl;
        return 1;
    }

    if (args.find("dt") != args.end()) {
        dt = parseTime(args["dt"]);
    }

    if (args.find("t_end") != args.end()) {
        t_end = parseTime(args["t_end"]);
    }

    if (args.find("vs") != args.end()) {
        vs = parseTime(args["vs"]);
    }

    if (args.find("vs_dir") != args.end()) {
        vs_dir = args["vs_dir"];
    }

    if (args.find("theta") != args.end()) {
        theta = std::stod(args["theta"]);
    }

    // Create visualization output directory if it doesn't exist
    fs::create_directories(vs_dir);

    // Read input data from file
    std::vector<Body*> bodies;
    if (!readCSV(filename, bodies)) {
        std::cerr << "Failed to read input data." << std::endl;
        return 1;
    }

    // Set the gravitational parameter mu = G * M_sun
    // double mu = G * 1.98847e30; // AU^3/day^2

    // Initialize simulation time
    double t = 0.0;

    
    // First, compute initial accelerations
    computeAccelerations(bodies, G, theta, softening);

    // Simulation loop
    int step = 0;
    int vs_counter = 0;


    while (t < t_end) {
        // Perform leapfrog integration (half-step velocity + position update)
        leapfrogIntegration(bodies, dt);

        // Advance time
        t += dt;
        step++;

        // Compute accelerations at new positions
        computeAccelerations(bodies, G, theta, softening);

        // Update velocities by another half step using new accelerations
        for (auto& body : bodies) {
            body->vx += body->ax * dt * 0.5;
            body->vy += body->ay * dt * 0.5;
            body->vz += body->az * dt * 0.5;
        }

        // Output visualization data 
        if (step % int(vs / dt) == 0) {
            saveState(vs_dir, vs_counter, bodies);
            vs_counter++;
        }

    }

    // Clean up bodies
    for (auto& body : bodies) {
        delete body;
    }

    return 0;
}
