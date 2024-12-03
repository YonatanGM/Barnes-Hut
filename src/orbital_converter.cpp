#include <iostream>
#include "kepler_to_cartesian.h"

// g++ -std=c++11 -o orbital_converter src/orbital_converter.cpp src/kepler_to_cartesian.cpp -I./include
int main() {
    const std::string inputFilename = "data/scenario2_orbital_elements.csv"; // Input file with orbital elements
    const std::string outputFilename = "data/scenario2.csv";   // Output file for state vectors


    std::cout << "Converting orbital elements to state vectors..." << std::endl;

    if (!convertOrbitalElementsToCSV(inputFilename, outputFilename)) {
        std::cerr << "Conversion failed. Check the input file or parameters." << std::endl;
        return 1;
    }

    std::cout << "Conversion successful! Output written to: " << outputFilename << std::endl;

    // combineCSVFiles("data/planets_and_moons_state_vectors.csv", "state_vectors.csv", "planets_and_moons_and_asteroids_state_vectors.csv");

    return 0;
}
