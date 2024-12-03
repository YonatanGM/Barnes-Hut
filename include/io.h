#ifndef IO_H
#define IO_H

#include <vector>
#include <string>
#include "body.h"

// function to read bodies from a CSV file
bool readCSV(const std::string& filename,
             std::vector<double>& masses,
             std::vector<Position>& positions,
             std::vector<Velocity>& velocities);

// function to save the state of bodies to a CSV file
void saveState(const std::string& vs_dir, int vs_counter,
               const std::vector<double>& masses,
               const std::vector<Position>& positions,
               const std::vector<Velocity>& velocities);


#endif 
