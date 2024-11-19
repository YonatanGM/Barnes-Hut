#include "io.h"
#include "kepler_to_cartesian.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>
#include <unordered_map>
#include <iomanip> 

using namespace std; 

// Function to parse Scenario 1 CSV files
bool readCSV(const std::string& filename, std::vector<Body*>& bodies) {
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

    // Debugging: Print parsed headers
    // std::cerr << "Parsed headers:\n";
    // for (const auto& [key, value] : headerMap) {
    //     std::cerr << "  " << key << ": " << value << "\n";
    // }

    // Validate required headers
    std::vector<std::string> requiredHeaders = {"mass", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"};
    for (const std::string& header : requiredHeaders) {
        if (headerMap.find(header) == headerMap.end()) {
            std::cerr << "Missing required header: " << header << "\n";
            return false;
        }
    }

    int bodyIndex = 0; // For assigning default names if missing

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string token;

        // Split the line into tokens
        while (std::getline(iss, token, ',')) {
            tokens.push_back(token);
        }

        // Create a new Body
        Body* body = new Body;

        try {
            // Name (optional)
            if (headerMap.count("name") && headerMap["name"] < tokens.size() && !tokens[headerMap["name"]].empty()) {
                body->name = tokens[headerMap["name"]];
            } else {
                // Assign a default name if missing
                body->name = "Body" + std::to_string(bodyIndex);
            }

            // Mass (required)
            body->mass = std::stod(tokens.at(headerMap.at("mass")));

            // Position components (pos_x, pos_y, pos_z - required)
            body->x = std::stod(tokens.at(headerMap.at("pos_x")));
            body->y = std::stod(tokens.at(headerMap.at("pos_y")));
            body->z = std::stod(tokens.at(headerMap.at("pos_z")));

            // Velocity components (vel_x, vel_y, vel_z - required)
            body->vx = std::stod(tokens.at(headerMap.at("vel_x")));
            body->vy = std::stod(tokens.at(headerMap.at("vel_y")));
            body->vz = std::stod(tokens.at(headerMap.at("vel_z")));

            // Initialize accelerations to zero
            body->ax = 0.0;
            body->ay = 0.0;
            body->az = 0.0;

            // Add the body to the list
            bodies.push_back(body);
            bodyIndex++; // Increment body index

        } catch (const std::exception& ex) {
            // Log error, delete body, and skip this row
            std::cerr << "Error parsing line. Skipping: " << line << "\n";
            std::cerr << "Exception: " << ex.what() << "\n";
            delete body;
        }
    }

    file.close();
    return true;
}

// Function to write the bodies to csv for visualization 
void saveState(const std::string& vs_dir, int vs_counter, const std::vector<Body*>& bodies) {
    // Create the file path with zero-padded suffix
    std::ostringstream vs_filename_stream;
    vs_filename_stream << vs_dir << "/output_" << std::setw(5) << std::setfill('0') << vs_counter << ".csv";
    std::string vs_filename = vs_filename_stream.str();

    // Open the file
    std::ofstream vs_file(vs_filename);
    if (!vs_file.is_open()) {
        std::cerr << "Failed to open visualization file: " << vs_filename << std::endl;
        exit(1);
    }

    // Write header
    vs_file << "name,x,y,z,vx,vy,vz\n";

    // Write data
    for (const auto& body : bodies) {
        vs_file << body->name << ","
                << body->x << ","
                << body->y << ","
                << body->z << ","
                << body->vx << ","
                << body->vy << ","
                << body->vz << "\n";
    }
    vs_file.close();
}


bool convertOrbitalElementsToCSV(const std::string& inputFilename, const std::string& outputFilename, double mu) {
    std::ifstream inputFile(inputFilename);
    if (!inputFile.is_open()) {
        std::cerr << "Error opening input file: " << inputFilename << std::endl;
        return false;
    }

    std::ofstream outputFile(outputFilename);
    if (!outputFile.is_open()) {
        std::cerr << "Error opening output file: " << outputFilename << std::endl;
        return false;
    }

    std::string line;

    // Read the header line
    if (!std::getline(inputFile, line)) {
        std::cerr << "Empty file or error reading input file: " << inputFilename << std::endl;
        return false;
    }

    // Process the header to map column names to indices
    std::istringstream headerStream(line);
    std::vector<std::string> headers;
    std::string header;
    while (std::getline(headerStream, header, ',')) {
        // Remove surrounding quotes from each header
        if (!header.empty() && header.front() == '"' && header.back() == '"') {
            header = header.substr(1, header.size() - 2);
        }
        headers.push_back(header);
    }

    // Map headers to indices
    std::unordered_map<std::string, int> headerMap;
    for (size_t i = 0; i < headers.size(); ++i) {
        headerMap[headers[i]] = static_cast<int>(i);
    }
    std::cerr << "Header Map:\n";
    for (const auto& [key, value] : headerMap) {
        std::cerr << "  " << key << " -> " << value << "\n";
    }

    outputFile << "name,class,mass,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z\n";

    // Random number generator for albedo approximation
    std::random_device rd;
    std::mt19937 gen(rd());

    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string token;

        // Split the line into tokens
        while (std::getline(iss, token, ',')) {
            tokens.push_back(token);
        }

        // Initialize essential variables
        std::string name = "";  // Default to empty string for missing names
        std::string classType = "";
        double mass = 0.0;
        double albedo = 0.0;
        double diameter = 0.0;
        bool hasEssentialFields = true;

        // Extract `name`
        if (headerMap.count("name") && headerMap["name"] < tokens.size()) {
            name = tokens[headerMap["name"]];
        }

        // Extract and validate `class`
        if (headerMap.count("class") && headerMap["class"] < tokens.size() && !tokens[headerMap["class"]].empty()) {
            classType = tokens[headerMap["class"]];
        } else {
            std::cerr << "Missing class field for row. Skipping..." << std::endl;
            continue; // Skip if `class` is missing
        }

        // Extract or approximate `mass`
        if (headerMap.count("mass") && headerMap["mass"] < tokens.size() && !tokens[headerMap["mass"]].empty()) {
            mass = std::stod(tokens[headerMap["mass"]]);
        }

        // Extract or approximate `albedo`
        if (headerMap.count("albedo") && headerMap["albedo"] < tokens.size() && !tokens[headerMap["albedo"]].empty()) {
            albedo = std::stod(tokens[headerMap["albedo"]]);
        } else {
            std::cout << "Missing albedo for body: " << name << ". Approximating based on class...\n";
            std::uniform_real_distribution<> dis(0.1, 0.2); // Default range
            if (classType == "AMO" || classType == "APO" || classType == "ATE" || classType == "IEO") {
                dis = std::uniform_real_distribution<>(0.450, 0.550);
            } else if (classType == "IMB") {
                dis = std::uniform_real_distribution<>(0.030, 0.103);
            } else if (classType == "MBA") {
                dis = std::uniform_real_distribution<>(0.097, 0.203);
            }
            albedo = dis(gen);
        }

        // Extract or approximate `diameter`
        if (headerMap.count("diameter") && headerMap["diameter"] < tokens.size() && !tokens[headerMap["diameter"]].empty()) {
            diameter = std::stod(tokens[headerMap["diameter"]]);
        } else if (headerMap.count("H") && headerMap["H"] < tokens.size() && !tokens[headerMap["H"]].empty()) {
            double H = std::stod(tokens[headerMap["H"]]);
            diameter = 1329 * std::pow(albedo, -0.5) * std::pow(10, -0.2 * H); // in km
            std::cout << "Missing diameter for body: " << name << ". Approximated using albedo and H-magnitude.\n";
        }

        // Calculate `mass` if still missing
        if (mass == 0.0 && diameter > 0.0) {
            double rho = (albedo < 0.1) ? 1380 : (albedo <= 0.2 ? 2710 : 5320); // density in kg/m^3
            double radius = (diameter / 2.0) * 1000.0; // Convert diameter to meters
            mass = (4.0 / 3.0) * M_PI * std::pow(radius, 3) * rho; // in kg
            std::cout << "Approximated mass for body: " << name << " using diameter and density.\n";
        }

        // Parse orbital elements
        OrbitalElements elem;
        if (headerMap.count("e") && headerMap["e"] < tokens.size() && !tokens[headerMap["e"]].empty() &&
            headerMap.count("a") && headerMap["a"] < tokens.size() && !tokens[headerMap["a"]].empty() &&
            headerMap.count("i") && headerMap["i"] < tokens.size() && !tokens[headerMap["i"]].empty() &&
            headerMap.count("om") && headerMap["om"] < tokens.size() && !tokens[headerMap["om"]].empty() &&
            headerMap.count("w") && headerMap["w"] < tokens.size() && !tokens[headerMap["w"]].empty() &&
            headerMap.count("ma") && headerMap["ma"] < tokens.size() && !tokens[headerMap["ma"]].empty()) {
            elem.eccentricity = std::stod(tokens[headerMap["e"]]);
            elem.semiMajorAxis = std::stod(tokens[headerMap["a"]]);
            elem.inclination = std::stod(tokens[headerMap["i"]]) * M_PI / 180.0;
            elem.longOfAscNode = std::stod(tokens[headerMap["om"]]) * M_PI / 180.0;
            elem.argOfPeriapsis = std::stod(tokens[headerMap["w"]]) * M_PI / 180.0;
            elem.meanAnomaly = std::stod(tokens[headerMap["ma"]]) * M_PI / 180.0;
            elem.mass = mass;
        } else {
            std::cerr << "Missing essential orbital elements for row: " << line << ". Skipping..." << std::endl;
            continue; // Skip if essential orbital elements are missing
        }

        // Convert orbital elements to Cartesian state vectors
        Body* body = convertKeplerToCartesian(elem, mu);

        // Write state vectors to the output file
        outputFile << name << ","    // name (can be empty string if missing)
                   << classType << "," // class
                   << body->mass << "," // mass
                   << body->x << ","    // pos_x
                   << body->y << ","    // pos_y
                   << body->z << ","    // pos_z
                   << body->vx << ","   // vel_x
                   << body->vy << ","   // vel_y
                   << body->vz << "\n"; // vel_z

        // Clean up the dynamically allocated body
        delete body;
    }

    inputFile.close();
    outputFile.close();

    return true;
}
