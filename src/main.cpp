#include <iostream>
#include <vector>
#include <string>
#include <mpi.h>
#include <filesystem>
#include "body.h"
#include "io.h"
#include "barnes_hut.h"
#include "integration.h"
#include "logger.h"
#include "cxxopts.hpp"


// logging macro to output messages from a specific rank
#define LOG(rank_to_print, msg) \
    do { if (logging_enabled && rank == rank_to_print) std::cout << msg << std::endl; } while (0)

bool logging_enabled = false; 
namespace fs = std::filesystem;

// parses time strings like "1h", "2.5d", "1y" into days
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

// creates an MPI datatype for the Body structure
void createMPIBodyType(MPI_Datatype* MPI_BODY) {
    int lengths[11] = {1,1,1,1,1,1,1,1,1,1,1}; // number of elements in each block
    MPI_Aint offsets[11];
    MPI_Datatype types[11] = {MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
                              MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};

    // compute offsets of each member in the Body structure
    offsets[0] = offsetof(Body, index);
    offsets[1] = offsetof(Body, mass);
    offsets[2] = offsetof(Body, x);
    offsets[3] = offsetof(Body, y);
    offsets[4] = offsetof(Body, z);
    offsets[5] = offsetof(Body, vx);
    offsets[6] = offsetof(Body, vy);
    offsets[7] = offsetof(Body, vz);
    offsets[8] = offsetof(Body, ax);
    offsets[9] = offsetof(Body, ay);
    offsets[10] = offsetof(Body, az);

    // create the MPI datatype
    MPI_Type_create_struct(11, lengths, offsets, types, MPI_BODY);
    MPI_Type_commit(MPI_BODY);
}



int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv); // initialize MPI environment

    MPI_Datatype MPI_BODY;
    createMPIBodyType(&MPI_BODY); // create MPI datatype for Body structure

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get current process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get total number of processes

    // default simulation parameters
    const double G = 1.48812e-34; // gravitational constant in AU^3 kg^-1 day^-2
    double softening = 1e-11;     // softening parameter in AU
    double theta = 0.5;           // Barnes-Hut opening angle parameter
    double dt = 1.0;              // time step in days
    double t_end = 365.25;        // end time in days (default 1 year)
    double vs = 1.0;              // visualization step in days
    std::string vs_dir = "sim";   // output directory for visualization data
    std::string filename;         // input filename

    // command-line argument parsing
    try {
        cxxopts::Options options(argv[0], "N-Body Simulation with MPI & Barnes-Hut");
        options.add_options()
            ("f,file", "Input file", cxxopts::value<std::string>())
            ("d,dt", "Time step (e.g., 1d)", cxxopts::value<std::string>()->default_value("1d"))
            ("t,t_end", "End time (e.g., 1y)", cxxopts::value<std::string>()->default_value("1y"))
            ("v,vs", "Visualization step", cxxopts::value<std::string>()->default_value("1d"))
            ("o,vs_dir", "Output directory", cxxopts::value<std::string>()->default_value("sim"))
            ("theta", "Barnes-Hut parameter", cxxopts::value<double>()->default_value("0.5"))
            ("s,softening", "Softening parameter", cxxopts::value<double>()->default_value("1e-11"))
            ("log", "Enable logging")
            ("h,help", "Show usage");

        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            if (rank == 0) std::cout << options.help() << std::endl;
            MPI_Finalize();
            return 0;
        }

        // initialize logging
        logging_enabled = result.count("log");
        Logger::initialize(logging_enabled);

        // get input filename
        if (result.count("file")) {
            filename = result["file"].as<std::string>();
        } else {
            if (rank == 0) std::cerr << "Error: Input file not specified. Use --file <filename>" << std::endl;
            MPI_Finalize();
            return 1;
        }

        // parse other parameters
        dt = parseTime(result["dt"].as<std::string>());
        t_end = parseTime(result["t_end"].as<std::string>());
        vs = parseTime(result["vs"].as<std::string>());
        vs_dir = result["vs_dir"].as<std::string>();
        theta = result["theta"].as<double>();
        softening = result["softening"].as<double>();

        // validate parameters
        if (dt <= 0 || t_end <= 0 || vs <= 0 || theta <= 0 || softening < 0) {
            throw std::invalid_argument("Invalid or missing parameters.");
        }
    } catch (const std::exception& e) {
        // handle exceptions during argument parsing
        if (rank == 0) std::cerr << "Error: " << e.what() << std::endl;
        MPI_Finalize();
        return 1;
    }

    // log simulation parameters
    LOG(0, "Simulation parameters:");
    LOG(0, "  File: " << filename << ", dt: " << dt << ", t_end: " << t_end);
    LOG(0, "  theta: " << theta << ", softening: " << softening);
    LOG(0, "  Visualization every " << vs << " days to " << vs_dir);

    // create visualization directory if it doesn't exist
    if (rank == 0 && !fs::exists(vs_dir) && !fs::create_directories(vs_dir)) {
        std::cerr << "Error: Could not create directory: " << vs_dir << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // read input file and initialize bodies
    std::vector<Body> bodies;
    if (rank == 0) {
        if (!readCSV(filename, bodies)) {
            std::cerr << "Error: Could not read file: " << filename << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // assign indices to bodies
        for (size_t i = 0; i < bodies.size(); ++i) bodies[i].index = static_cast<int>(i);
        LOG(0, "Loaded " << bodies.size() << " bodies.");
    }

    // broadcast number of bodies to all processes
    int num_bodies = static_cast<int>(bodies.size());
    MPI_Bcast(&num_bodies, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) bodies.resize(num_bodies); // resize vector on other processes
    // broadcast bodies to all processes
    MPI_Bcast(bodies.data(), num_bodies, MPI_BODY, 0, MPI_COMM_WORLD);

    // distribute work among processes
    int bodies_per_proc = num_bodies / size; // base number of bodies per process
    int remainder = num_bodies % size;       // extra bodies to distribute

    std::vector<int> sendcounts(size); // number of bodies each process will receive
    std::vector<int> displs(size);     // displacement (starting index) for each process

    for (int i = 0; i < size; ++i) {
        sendcounts[i] = bodies_per_proc + (i < remainder ? 1 : 0); // distribute extra bodies to first 'remainder' processes
        displs[i] = (bodies_per_proc * i) + std::min(i, remainder); // compute displacement for each process
    }

    int local_n = sendcounts[rank];            // number of bodies for this process
    std::vector<Body> local_bodies(local_n);   // local bodies vector

    // MPI_Scatterv to distribute bodies to the processes
    MPI_Scatterv(bodies.data(), sendcounts.data(), displs.data(), MPI_BODY,
                 local_bodies.data(), local_n, MPI_BODY, 0, MPI_COMM_WORLD);
    LOG(rank, "Process " << rank << " received " << local_n << " bodies " << "at index " << displs[rank]);

    // initialize simulation time and counters
    double t = 0.0;      // current simulation time in days
    int step = 0;        // simulation step counter
    int vs_counter = 0;  // visualization counter

    // compute initial accelerations
    computeAccelerations(bodies, local_bodies, G, theta, softening);

    // main simulation loop
    while (t < t_end) {
        // perform leapfrog integration: update velocities by half step and positions by full step
        leapfrogIntegration(local_bodies, dt);

        // advance simulation time
        t += dt;
        step++;

        // gather updated positions from all processes to compute accelerations
        MPI_Allgatherv(local_bodies.data(), local_n, MPI_BODY,
                       bodies.data(), sendcounts.data(), displs.data(), MPI_BODY, MPI_COMM_WORLD);

        // compute accelerations at new positions using Barnes-Hut algorithm
        computeAccelerations(bodies, local_bodies, G, theta, softening);

        // update velocities by another half step using new accelerations
        for (auto& body : local_bodies) {
            body.vx += body.ax * dt * 0.5; // update velocity components
            body.vy += body.ay * dt * 0.5;
            body.vz += body.az * dt * 0.5;
        }

        // output visualization data at specified intervals
        if (rank == 0 && t >= vs_counter * vs) {
            saveState(vs_dir, vs_counter, bodies); // save current state to file
            vs_counter++;
        }
    }

    LOG(0, "Simulation completed.");


    // clean up MPI data types and finalize MPI
    MPI_Type_free(&MPI_BODY);
    MPI_Finalize();

    return 0;
}
