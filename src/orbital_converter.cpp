#include <iostream>
#include "io.h"
#include "kepler_to_cartesian.h"

int main() {
    const std::string inputFilename = "data/scenario1_without_planets_and_moons.csv"; // Input file with orbital elements
    const std::string outputFilename = "state_vectors.csv";   // Output file for state vectors


    std::cout << "Converting orbital elements to state vectors..." << std::endl;

    if (!convertOrbitalElementsToCSV(inputFilename, outputFilename)) {
        std::cerr << "Conversion failed. Check the input file or parameters." << std::endl;
        return 1;
    }

    std::cout << "Conversion successful! Output written to: " << outputFilename << std::endl;

    combineCSVFiles("data/planets_and_moons_state_vectors.csv", "state_vectors.csv", "planets_and_moons_and_asteroids_state_vectors.csv");

    return 0;
}
