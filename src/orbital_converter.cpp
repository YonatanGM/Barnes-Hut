#include <iostream>
#include "io.h"
#include "kepler_to_cartesian.h"

int main() {
    const std::string inputFilename = "data/scenario1_without_planets_and_moons.csv"; // Input file with orbital elements
    const std::string outputFilename = "state_vectors.csv";   // Output file for state vectors

    // Constants
    const double G = 0.0002959122082855911; // Gravitational constant in AU^3 kg^-1 day^-2
    const double mass_sun = 1.98847e30;     // Mass of the Sun in kg

    // Dynamically calculate mu
    double mu = G * mass_sun; // AU^3/day^2

    std::cout << "Converting orbital elements to state vectors..." << std::endl;

    if (!convertOrbitalElementsToCSV(inputFilename, outputFilename, mu)) {
        std::cerr << "Conversion failed. Check the input file or parameters." << std::endl;
        return 1;
    }

    std::cout << "Conversion successful! Output written to: " << outputFilename << std::endl;
    return 0;
}
