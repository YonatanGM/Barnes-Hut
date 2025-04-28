#ifndef IO_H
#define IO_H

#include <vector>
#include <string>
#include "body.h"
#include "tinyxml2.h"


// function to read csv and populate vectors, with optional max_bodies limit
bool readCSV(const std::string& filename,
             std::vector<double>& masses,
             std::vector<Position>& positions,
             std::vector<Velocity>& velocities,
             std::vector<std::string>& names,
             std::vector<int>& orbit_classes,
             int max_bodies);

// function to save intermediate simulation state for visualization
void saveState(const std::string& vs_dir, int vs_counter,
               const std::vector<double>& masses,
               const std::vector<Position>& positions,
               const std::vector<Velocity>& velocities);

// function to write a vtp file for visualization
void writeVTPFile(int rank, int vs_counter,
                 const std::vector<double> &local_masses,
                 const std::vector<Position> &local_positions,
                 const std::vector<Velocity> &local_velocities,
                 const std::vector<Acceleration> &local_accelerations,
                 const std::vector<std::string> &local_names,
                 const std::vector<int> &local_orbit_classes,
                 int body_id_offset,
                 double kinetic_energy,
                 double potential_energy,
                 double total_energy,
                 double virial_equilibrium,
                 const std::string &vs_dir);

 // function to update (or create) a pvd file that references all vtp files for the given timestep
 void updatePVDFile(const std::string &pvdFilename,
                   int size,
                   int vs_counter,
                   double current_time,
                   const std::string &vs_dir);


// Write one timestep’s positions to a small CSV named ref_step_<step>.csv
void saveReferenceStepCSV(const std::string& dir,
                          int step,
                          const std::vector<Position>& pos);

// Load exactly that one CSV back into a Position vector
std::vector<Position>
loadReferenceStepCSV(const std::string& dir,
                     int step,
                     int num_bodies);

// Compute the sum of per‐body Euclidean distances between two Position arrays
double computeDistanceSum(const std::vector<Position>& a,
                          const std::vector<Position>& b);



#endif
