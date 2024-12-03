#include "io.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <unordered_map>
#include <iomanip> 



/**
 * @brief Reads bodies data from a CSV file.
 *
 * Reads masses, positions, and velocities of bodies from the specified CSV file.
 *
 * @param filename Path to the CSV file.
 * @param masses Vector to store masses.
 * @param positions Vector to store positions.
 * @param velocities Vector to store velocities.
 * @return True if reading was successful, false otherwise.
 */
bool readCSV(const std::string& filename,
             std::vector<double>& masses,
             std::vector<Position>& positions,
             std::vector<Velocity>& velocities) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening CSV file: " << filename << std::endl;
        return false;
    }

    std::string line;

    // Read the header line
    if (!std::getline(file, line)) {
        std::cerr << "Empty file or error reading CSV file: " << filename << std::endl;
        return false;
    }

    // Process the header to map column names to indices
    std::istringstream headerStream(line);
    std::vector<std::string> headers;
    std::string header;
    while (std::getline(headerStream, header, ',')) {
        headers.push_back(header);
    }

    // Map headers to indices
    std::unordered_map<std::string, int> headerMap;
    for (size_t i = 0; i < headers.size(); ++i) {
        headerMap[headers[i]] = static_cast<int>(i);
    }

    // Validate required headers
    std::vector<std::string> requiredHeaders = {"mass", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"};
    for (const std::string& header : requiredHeaders) {
        if (headerMap.find(header) == headerMap.end()) {
            std::cerr << "Missing required header: " << header << "\n";
            return false;
        }
    }

    masses.clear();
    positions.clear();
    velocities.clear();

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string token;

        // Split the line into tokens
        while (std::getline(iss, token, ',')) {
            tokens.push_back(token);
        }

        try {
            // Mass
            double mass = std::stod(tokens.at(headerMap.at("mass")));
            masses.push_back(mass);

            // Position components (pos_x, pos_y, pos_z)
            double x = std::stod(tokens.at(headerMap.at("pos_x")));
            double y = std::stod(tokens.at(headerMap.at("pos_y")));
            double z = std::stod(tokens.at(headerMap.at("pos_z")));
            positions.push_back({x, y, z});

            // Velocity components (vel_x, vel_y, vel_z)
            double vx = std::stod(tokens.at(headerMap.at("vel_x")));
            double vy = std::stod(tokens.at(headerMap.at("vel_y")));
            double vz = std::stod(tokens.at(headerMap.at("vel_z")));
            velocities.push_back({vx, vy, vz});

        } catch (const std::exception& ex) {
            // Log error and skip this row
            std::cerr << "Error parsing line. Skipping: " << line << "\n";
            std::cerr << "Exception: " << ex.what() << "\n";
        }
    }

    file.close();
    return true;
}


/**
 * @brief Saves the current state of the simulation to a CSV file.
 *
 * Writes masses, positions, and velocities of all bodies to a CSV file for visualization.
 *
 * @param vs_dir Directory to save the visualization files.
 * @param vs_counter Counter for the visualization step (used in filename).
 * @param masses Vector of masses.
 * @param positions Vector of positions.
 * @param velocities Vector of velocities.
 */
void saveState(const std::string& vs_dir, int vs_counter,
               const std::vector<double>& masses,
               const std::vector<Position>& positions,
               const std::vector<Velocity>& velocities) {
    // Create the file path with zero-padded suffix
    std::ostringstream vs_filename_stream;
    vs_filename_stream << vs_dir << "/output_" << std::setw(5) << std::setfill('0') << vs_counter << ".csv";
    std::string vs_filename = vs_filename_stream.str();

    // Open the file
    std::ofstream vs_file(vs_filename);
    if (!vs_file.is_open()) {
        std::cerr << "Failed to open visualization file: " << vs_filename << std::endl;
        // exit(1); // Comment out exit to allow the simulation to continue
    } else {
        // Write header
        vs_file << "id,mass,x,y,z,vx,vy,vz\n";

        // Write data
        for (size_t i = 0; i < masses.size(); ++i) {
            vs_file << i << ","                       // Write id starting from 0
                    << masses[i] << ","
                    << positions[i].x << "," << positions[i].y << "," << positions[i].z << ","
                    << velocities[i].vx << "," << velocities[i].vy << "," << velocities[i].vz << "\n";
        }
        vs_file.close();
    }
}


