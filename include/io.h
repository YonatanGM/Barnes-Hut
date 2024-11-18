#ifndef IO_H
#define IO_H

#include <vector>
#include <string>
#include "body.h"

// Functions to parse CSV file
bool readCSV(const std::string& filename, std::vector<Body*>& bodies);


// Function to parse Scenario 2 CSV files (Orbital Elements)
// bool parseScenario2(const std::string& filename, std::vector<Body*>& bodies);

void saveState(const std::string& vs_dir, int vs_counter, const std::vector<Body*>& bodies);

// Function to convert orbital elements CSV to state vector CSV
bool convertOrbitalElementsToCSV(const std::string& inputFilename, const std::string& outputFilename, double mu);
#endif // IO_H
