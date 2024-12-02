#include <iostream>
#include <vector>
#include <string>
#include <mpi.h>
#include <omp.h>
#include <filesystem>
#include <cstddef> // For offsetof
#include "body.h"
#include "io.h"
#include "barnes_hut.h"
#include "integration.h"
#include "cxxopts.hpp"
#include "logger.h"

namespace fs = std::filesystem;

// Parses time strings like "1h", "2.5d", "1y" into days
double parseTime(const std::string& timeStr) {
    double timeValue = std::stod(timeStr.substr(0, timeStr.size() - 1)); 
    char unit = timeStr.back(); // get the unit character
    switch (unit) {
        case 'h': return timeValue / 24.0;         // convert hours to days
        case 'd': return timeValue;                // already in days
        case 'y': return timeValue * 365.25;       // convert years to days
        default:
            throw std::invalid_argument("Unknown time unit: " + timeStr);
    }
}


/**
 * @brief Creates an MPI datatype for the Position structure.
 * @param MPI_POSITION Pointer to the MPI_Datatype to be created.
 */
void createMPIPositionType(MPI_Datatype* MPI_POSITION) {
    int lengths[3] = {1, 1, 1};
    MPI_Aint displacements[3];
    MPI_Datatype types[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};

    displacements[0] = offsetof(Position, x);
    displacements[1] = offsetof(Position, y);
    displacements[2] = offsetof(Position, z);

    MPI_Type_create_struct(3, lengths, displacements, types, MPI_POSITION);
    MPI_Type_commit(MPI_POSITION);
}


/**
 * @brief Creates an MPI datatype for the Velocity structure.
 * @param MPI_VELOCITY Pointer to the MPI_Datatype to be created.
 */
void createMPIVelocityType(MPI_Datatype* MPI_VELOCITY) {
    int lengths[3] = {1, 1, 1};
    MPI_Aint displacements[3];
    MPI_Datatype types[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};

    displacements[0] = offsetof(Velocity, vx);
    displacements[1] = offsetof(Velocity, vy);
    displacements[2] = offsetof(Velocity, vz);

    MPI_Type_create_struct(3, lengths, displacements, types, MPI_VELOCITY);
    MPI_Type_commit(MPI_VELOCITY);
}


/**
 * @brief Creates an MPI datatype for the Acceleration structure.
 * @param MPI_ACCELERATION Pointer to the MPI_Datatype to be created.
 */
void createMPIAccelerationType(MPI_Datatype* MPI_ACCELERATION) {
    int lengths[3] = {1, 1, 1};
    MPI_Aint displacements[3];
    MPI_Datatype types[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};

    displacements[0] = offsetof(Acceleration, ax);
    displacements[1] = offsetof(Acceleration, ay);
    displacements[2] = offsetof(Acceleration, az);

    MPI_Type_create_struct(3, lengths, displacements, types, MPI_ACCELERATION);
    MPI_Type_commit(MPI_ACCELERATION);
}


/**
 * @brief Main function for the N-Body Simulation with MPI & Barnes-Hut algorithm.
 *
 * This function initializes MPI, parses command-line arguments, sets up the simulation parameters,
 * distributes data among processes, and runs the main simulation loop.
 */
int main(int argc, char* argv[]) {
    int provided;

    // Initialize MPI with thread support
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    // Check the level of thread support provided
    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "MPI does not provide the required threading level" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    auto total_start = MPI_Wtime(); // Start timing the total process

    // Create custom MPI datatypes
    MPI_Datatype MPI_POSITION, MPI_VELOCITY, MPI_ACCELERATION;
    createMPIPositionType(&MPI_POSITION);
    createMPIVelocityType(&MPI_VELOCITY);
    createMPIAccelerationType(&MPI_ACCELERATION);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get current process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get total number of processes

    // Default simulation parameters
    const double G = 1.48812e-34; // Gravitational constant in AU^3 kg^-1 day^-2
    double softening = 1e-11;     // Softening parameter in AU
    double theta = 0.5;           // Barnes-Hut opening angle parameter
    double dt = 1.0;              // Time step in days
    double t_end = 365.25;        // End time in days (default 1 year)
    double vs = 1.0;              // Visualization step in days
    std::string vs_dir = "sim";   // Output directory for visualization data
    std::string filename;         // Input filename

    // Command-line argument parsing
    try {
        // Set up command-line options using cxxopts
        cxxopts::Options options(argv[0], "N-Body Simulation with MPI & Barnes-Hut");
        options.add_options()
            // Define command-line options
            ("f,file", "Input file", cxxopts::value<std::string>())
            ("d,dt", "Time step (e.g., 1d)", cxxopts::value<std::string>()->default_value("1d"))
            ("t,t_end", "End time (e.g., 1y)", cxxopts::value<std::string>()->default_value("1y"))
            ("v,vs", "Visualization step", cxxopts::value<std::string>()->default_value("1d"))
            ("o,vs_dir", "Output directory", cxxopts::value<std::string>()->default_value("sim"))
            ("theta", "Barnes-Hut parameter", cxxopts::value<double>()->default_value("0.5"))
            ("s,softening", "Softening parameter", cxxopts::value<double>()->default_value("1e-11"))
            ("log", "Enable logging")
            ("h,help", "Show usage");

        // Parse the command-line arguments
        auto result = options.parse(argc, argv);

        // If help is requested, display usage and exit
        if (result.count("help")) {
            if (rank == 0) std::cout << options.help() << std::endl;
            MPI_Finalize();
            return 0;
        }

        // Check if logging is enabled
        logging_enabled = result.count("log");

        // Get input filename
        if (result.count("file")) {
            filename = result["file"].as<std::string>();
        } else {
            if (rank == 0) std::cerr << "Error: Input file not specified. Use --file <filename>" << std::endl;
            MPI_Finalize();
            return 1;
        }

        // Parse other parameters
        dt = parseTime(result["dt"].as<std::string>());
        t_end = parseTime(result["t_end"].as<std::string>());
        vs = parseTime(result["vs"].as<std::string>());
        vs_dir = result["vs_dir"].as<std::string>();
        theta = result["theta"].as<double>();
        softening = result["softening"].as<double>();

        // Validate parameters
        if (dt <= 0 || t_end <= 0 || vs <= 0 || theta <= 0 || softening < 0) {
            throw std::invalid_argument("Invalid or missing parameters.");
        }
    } catch (const std::exception& e) {
        // Handle exceptions during argument parsing
        if (rank == 0) std::cerr << "Error: " << e.what() << std::endl;
        MPI_Finalize();
        return 1;
    }

    // Log the maximum number of threads available
    LOG(rank, "Maximum threads available: " << omp_get_max_threads());

    // Log the number of threads in use
    #pragma omp parallel
    {
        #pragma omp single
        {
            LOG(rank, "Total threads: " << omp_get_num_threads());
        }
    }

    // Log simulation parameters
    if (rank == 0) {
        LOG(0, "Simulation parameters:");
        LOG(0, "  File: " << filename << ", dt: " << dt << ", t_end: " << t_end);
        LOG(0, "  theta: " << theta << ", softening: " << softening);
        LOG(0, "  Visualization every " << vs << " days to " << vs_dir);
    }

    // Create visualization directory if it doesn't exist
    if (rank == 0 && !fs::exists(vs_dir) && !fs::create_directories(vs_dir)) {
        std::cerr << "Error: Could not create directory: " << vs_dir << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Read input file and initialize masses, positions, and velocities
    std::vector<double> masses;            // Vector to store masses of bodies
    std::vector<Position> positions;       // Vector to store positions of bodies
    std::vector<Velocity> velocities;      // Vector to store velocities of bodies

    int num_bodies = 0; // Total number of bodies in the simulation
    if (rank == 0) {
        if (!readCSV(filename, masses, positions, velocities)) {
            std::cerr << "Error: Could not read file: " << filename << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        num_bodies = static_cast<int>(masses.size());
        LOG(0, "Loaded " << num_bodies << " bodies.");
    }

    // Broadcast number of bodies to all processes
    MPI_Bcast(&num_bodies, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Resize vectors on other processes
    if (rank != 0) {
        masses.resize(num_bodies);
        positions.resize(num_bodies);
        velocities.resize(num_bodies);
    }

    // Broadcast masses, positions, and velocities to all processes
    MPI_Bcast(masses.data(), num_bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(positions.data(), num_bodies, MPI_POSITION, 0, MPI_COMM_WORLD);
    MPI_Bcast(velocities.data(), num_bodies, MPI_VELOCITY, 0, MPI_COMM_WORLD);

    // Distribute work among processes
    int bodies_per_proc = num_bodies / size; // Base number of bodies per process
    int remainder = num_bodies % size;       // Extra bodies to distribute

    std::vector<int> sendcounts(size); // Number of bodies each process will receive
    std::vector<int> displs(size, 0);  // Displacement (starting index) for each process

    int offset = 0;
    for (int i = 0; i < size; ++i) {
        sendcounts[i] = bodies_per_proc + (i < remainder ? 1 : 0);
        displs[i] = offset;
        offset += sendcounts[i];
    }

    int local_n = sendcounts[rank]; // Number of bodies for this process

    // Local data for each process
    std::vector<double> local_masses(local_n);              // Local masses
    std::vector<Position> local_positions(local_n);         // Local positions
    std::vector<Velocity> local_velocities(local_n);        // Local velocities
    std::vector<Acceleration> local_accelerations(local_n); // Local accelerations

    // Scatter masses to all processes
    MPI_Scatterv(masses.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 local_masses.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter positions to all processes
    MPI_Scatterv(positions.data(), sendcounts.data(), displs.data(), MPI_POSITION,
                 local_positions.data(), local_n, MPI_POSITION, 0, MPI_COMM_WORLD);

    // Scatter velocities to all processes
    MPI_Scatterv(velocities.data(), sendcounts.data(), displs.data(), MPI_VELOCITY,
                 local_velocities.data(), local_n, MPI_VELOCITY, 0, MPI_COMM_WORLD);

    LOG(rank, "Process " << rank << " received " << local_n << " bodies starting at index " << displs[rank]);

    // All initialization is done, now start simulation timing
    auto simulation_start = MPI_Wtime(); 

    // Initialize simulation time and counters
    double t = 0.0;      // Current simulation time in days
    int step = 0;        // Simulation step counter
    int vs_counter = 0;  // Visualization counter

    // Compute initial accelerations
    computeAccelerations(masses, positions, local_masses, local_positions, local_accelerations, G, theta, softening);

    // Main simulation loop
    while (t < t_end) {
        // Perform leapfrog integration, update velocities by half step and positions by full step
        leapfrogIntegration(local_positions, local_velocities, local_accelerations, dt);

        // Advance simulation time
        t += dt;
        step++;

        // Gather updated positions from all processes to compute accelerations
        MPI_Allgatherv(local_positions.data(), local_n, MPI_POSITION,
                       positions.data(), sendcounts.data(), displs.data(), MPI_POSITION, MPI_COMM_WORLD);

        // Compute accelerations at new positions using Barnes-Hut algorithm
        computeAccelerations(masses, positions, local_masses, local_positions, local_accelerations, G, theta, softening);

        // Update velocities by another half step using new accelerations
        #pragma omp parallel for
        for (int i = 0; i < local_n; ++i) {
            local_velocities[i].vx += local_accelerations[i].ax * dt * 0.5;
            local_velocities[i].vy += local_accelerations[i].ay * dt * 0.5;
            local_velocities[i].vz += local_accelerations[i].az * dt * 0.5;
        }

        // Output visualization data at specified intervals
        if (t >= vs_counter * vs) {
            // Gather velocities to the main rank
            MPI_Gatherv(local_velocities.data(), local_n, MPI_VELOCITY,
                        velocities.data(), sendcounts.data(), displs.data(), MPI_VELOCITY, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                saveState(vs_dir, vs_counter, masses, positions, velocities); // Save current state to file
                LOG(0, "Saved state for visualization step " << vs_counter);
            }
            vs_counter++;
        }
    }

    auto simulation_end = MPI_Wtime();
    LOG(0, "Simulation completed.");

    // Output timing info
    if (rank == 0) {
        std::cout << "SIMULATION_TIME " << (simulation_end - simulation_start) << std::endl;
        std::cout << "TOTAL_TIME " << (MPI_Wtime() - total_start) << std::endl;
    }

    // Clean up MPI data types and finalize MPI
    MPI_Type_free(&MPI_POSITION);
    MPI_Type_free(&MPI_VELOCITY);
    MPI_Type_free(&MPI_ACCELERATION);
    MPI_Finalize();

    return 0;
}