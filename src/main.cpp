#include <iostream>
#include <vector>
#include <string>
#include <mpi.h>
#include <omp.h>
#include <filesystem>
#include <cstddef> // for offsetof
#include <cmath>
#include <sstream>
#include <iomanip>
#include "body.h"
#include "io.h"
#include "barnes_hut.h"
#include "integration.h"
#include "cxxopts.hpp"
#include "logger.h"
#include "tinyxml2.h"
#include "parse_time.h"

namespace fs = std::filesystem;


int main(int argc, char* argv[]) {
    int provided;

    // initialize mpi with thread support
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    // check the level of thread support provided
    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "mpi does not provide the required threading level" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    auto total_start = MPI_Wtime(); // start timing the total process

    // custom mpi type for acceleration, position and velocities 
    MPI_Datatype MPI_VECTOR;
    MPI_Type_contiguous(3, MPI_DOUBLE, &MPI_VECTOR);
    MPI_Type_commit(&MPI_VECTOR);


    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get current process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get total number of processes

    // default simulation parameters
    const double G = 1.48812e-34; // gravitational constant in AU^3 kg^-1 day^-2
    double softening = 1e-11;     // softening parameter in AU
    double theta = 0.5;           // barnes-hut opening angle parameter
    double dt_val = 1.0;          // time step in days
    double t_end_val = 365.25;    // end time in days (default 1 year)
    double vs_val = 1.0;          // visualization step in days
    std::string vs_dir = "sim";   // output directory for visualization data
    std::string filename;         // input filename

    std::string dt_str;
    std::string t_end_str;
    std::string vs_str;
    int body_count = -1; // if not provided, use all bodies

    try {
        // set up command-line options using cxxopts
        cxxopts::Options options(argv[0], "n-body simulation with mpi & barnes-hut");
        options.add_options()
            // define command-line options
            ("f,file", "input file", cxxopts::value<std::string>())
            ("d,dt", "time step (e.g., 1d)", cxxopts::value<std::string>()->default_value("1d"))
            ("t,t_end", "end time (e.g., 1y)", cxxopts::value<std::string>()->default_value("1y"))
            ("v,vs", "visualization step", cxxopts::value<std::string>()->default_value("1d"))
            ("o,vs_dir", "output directory", cxxopts::value<std::string>()->default_value("sim"))
            ("theta", "barnes-hut parameter", cxxopts::value<double>()->default_value("0.5"))
            ("s,softening", "softening parameter", cxxopts::value<double>()->default_value("1e-11"))
            ("b,bodies", "number of bodies to simulate", cxxopts::value<int>()->default_value("-1"))
            ("log", "enable logging")
            ("h,help", "show usage");

        // parse the command-line arguments
        auto result = options.parse(argc, argv);

        // if help is requested, display usage and exit
        if (result.count("help")) {
            if (rank == 0) std::cout << options.help() << std::endl;
            MPI_Finalize();
            return 0;
        }

        // check if logging is enabled
        logging_enabled = result.count("log");

        // get input filename
        if (result.count("file")) {
            filename = result["file"].as<std::string>();
        } else {
            if (rank == 0) std::cerr << "error: input file not specified. use --file <filename>" << std::endl;
            MPI_Finalize();
            return 1;
        }

        // parse other parameters
        dt_str = result["dt"].as<std::string>();
        t_end_str = result["t_end"].as<std::string>();
        vs_str = result["vs"].as<std::string>();
        vs_dir = result["vs_dir"].as<std::string>();
        theta = result["theta"].as<double>();
        softening = result["softening"].as<double>();
        body_count = result["bodies"].as<int>();

        dt_val = parseTime(dt_str);
        t_end_val = parseTime(t_end_str);
        vs_val = parseTime(vs_str);

        // validate parameters
        if (dt_val <= 0 || t_end_val <= 0 || vs_val <= 0 || theta <= 0 || softening < 0) {
            throw std::invalid_argument("invalid or missing parameters.");
        }
    } catch (const std::exception& e) {
        // handle exceptions during argument parsing
        if (rank == 0) std::cerr << "error: " << e.what() << std::endl;
        MPI_Finalize();
        return 1;
    }

    // log the number of threads available
    LOG(rank, "threads available: " << omp_get_max_threads());

    // log simulation parameters
    if (rank == 0) {
        LOG(0, "simulation parameters:");
        LOG(0, "  file: " << filename << ", dt: " << dt_str << ", t_end: " << t_end_str);
        LOG(0, "  theta: " << theta << ", softening: " << softening);
        LOG(0, "  visualization every " << vs_str << " to " << vs_dir);
        if (body_count > 0) {
            LOG(0, "  limiting number of bodies to " << body_count);
        } else {
            LOG(0, "  using all bodies from csv");
        }
    }

    // create visualization directory if it doesn't exist
    if (rank == 0 && !fs::exists(vs_dir) && !fs::create_directories(vs_dir)) {
        std::cerr << "error: could not create directory: " << vs_dir << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

   

    std::vector<double> masses;            // vector to store masses of bodies
    std::vector<Position> positions;       // vector to store positions of bodies
    std::vector<Velocity> velocities;      // vector to store velocities of bodies
    std::vector<std::string> names;        // vector to store names of bodies
    std::vector<int> orbit_classes;        // vector to store orbit classes

    int num_bodies = 0; // total number of bodies
    if (rank == 0) {
        // read csv and limit the number of bodies if specified
        if (!readCSV(filename, masses, positions, velocities, names, orbit_classes, body_count)) {
            std::cerr << "error: could not read file: " << filename << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        num_bodies = (int)masses.size(); // get the actual number of bodies read

        LOG(0, "loaded " << num_bodies << " bodies" << ((body_count > 0 && body_count < num_bodies) ? " (truncated)" : "") << ".");
    }

     // construct pvd filename including body count
    std::string input_stem = fs::path(filename).stem().string();
    std::string pvdFilename = "sim_" + input_stem + "_dt" + dt_str + "_tend" + t_end_str + "_vs" + vs_str + "_bodies" + std::to_string(num_bodies) + ".pvd";

    // broadcast the number of bodies to all processes
    MPI_Bcast(&num_bodies, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // resize vectors on other processes based on the number of bodies
    if (rank != 0) {
        masses.resize(num_bodies);
        positions.resize(num_bodies);
        velocities.resize(num_bodies);
        names.resize(num_bodies);
        orbit_classes.resize(num_bodies);
    }

    // broadcast masses and positions to all processes
    MPI_Bcast(masses.data(), num_bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(positions.data(), num_bodies, MPI_VECTOR, 0, MPI_COMM_WORLD);

    // determine the number of bodies per process
    int bodies_per_proc = num_bodies / size; // base number of bodies per process
    int remainder = num_bodies % size;       // extra bodies to distribute

    std::vector<int> sendcounts(size), displs(size, 0);
    {
        int offset = 0;
        for (int i = 0; i < size; i++) {
            sendcounts[i] = bodies_per_proc + (i < remainder ? 1 : 0);
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    int local_n = sendcounts[rank]; // number of bodies for this process

    // prepare to scatter names
    int max_name_len = 0;
    if (rank == 0) {
        for (auto &nm : names) {
            int len = (int)nm.size();
            if (len > max_name_len) max_name_len = len;
        }
    }
    MPI_Bcast(&max_name_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int chunk_size = max_name_len + 1; // allocate extra space for null terminator
    std::vector<char> all_names_buffer;
    std::vector<int> name_sendcounts(size), name_displs(size, 0);
    if (rank == 0) {
        all_names_buffer.resize(num_bodies * chunk_size);
        for (int i = 0; i < num_bodies; i++) {
            std::string &nm = names[i];
            int len = (int)nm.size();
            if (len > max_name_len) len = max_name_len;
            std::memcpy(&all_names_buffer[i * chunk_size], nm.c_str(), len);
            for (int c = len; c < chunk_size; c++) {
                all_names_buffer[i * chunk_size + c] = '\0';
            }
        }

        for (int i = 0; i < size; i++) {
            name_sendcounts[i] = sendcounts[i] * chunk_size;
        }

        int off = 0;
        for (int i = 0; i < size; i++) {
            name_displs[i] = off;
            off += name_sendcounts[i];
        }
    }

    std::vector<char> local_name_buffer(local_n * chunk_size);
    MPI_Scatterv(all_names_buffer.data(), name_sendcounts.data(), name_displs.data(), MPI_CHAR,
                 local_name_buffer.data(), local_n * chunk_size, MPI_CHAR,
                 0, MPI_COMM_WORLD);

    std::vector<std::string> local_names(local_n);
    for (int i = 0; i < local_n; i++) {
        local_names[i] = std::string(&local_name_buffer[i * chunk_size]);
    }

    // local data for each process
    std::vector<double> local_masses(local_n);
    std::vector<Position> local_positions(local_n);
    std::vector<Velocity> local_velocities(local_n);
    std::vector<int> local_orbit_classes(local_n);

    // scatter masses to all processes
    MPI_Scatterv(masses.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 local_masses.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter positions
    MPI_Scatterv(positions.data(), sendcounts.data(), displs.data(), MPI_VECTOR,
                local_positions.data(), local_n, MPI_VECTOR, 0, MPI_COMM_WORLD);

    // Scatter velocities
    MPI_Scatterv(velocities.data(), sendcounts.data(), displs.data(), MPI_VECTOR,
                local_velocities.data(), local_n, MPI_VECTOR, 0, MPI_COMM_WORLD);

    // scatter orbit_classes to all processes
    MPI_Scatterv(orbit_classes.data(), sendcounts.data(), displs.data(), MPI_INT,
                 local_orbit_classes.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);

    LOG(rank, "process " << rank << " received " << local_n << " bodies starting at index " << displs[rank]);

    auto simulation_start = MPI_Wtime(); // start simulation timing

    double t = 0.0;      // current simulation time in days
    int step = 0;        // simulation step counter
    int vs_counter = 0;  // visualization counter

    // compute initial accelerations
    std::vector<Acceleration> local_accelerations(local_n);
    computeAccelerations(masses, positions, local_masses, local_positions, local_accelerations, G, theta, softening);

    // main simulation loop
    while (t < t_end_val) {
        // perform leapfrog integration, update positions and half-step velocities
        leapfrogIntegration(local_positions, local_velocities, local_accelerations, dt_val);

        // advance simulation time
        t += dt_val;
        step++;

        // gather updated positions from all processes for next acceleration calculation
        MPI_Allgatherv(local_positions.data(), local_n, MPI_VECTOR,
                    positions.data(), sendcounts.data(), displs.data(), MPI_VECTOR,
                    MPI_COMM_WORLD);

        // compute new accelerations based on updated positions
        computeAccelerations(masses, positions, local_masses, local_positions, local_accelerations, G, theta, softening);

        // update velocities by another half step using new accelerations
        #pragma omp parallel for
        for (int i = 0; i < local_n; i++) {
            local_velocities[i].vx += local_accelerations[i].ax * dt_val * 0.5;
            local_velocities[i].vy += local_accelerations[i].ay * dt_val * 0.5;
            local_velocities[i].vz += local_accelerations[i].az * dt_val * 0.5;
        }

        // check if it's time to save visualization data
        if (t >= vs_counter * vs_val) {
  
            // compute global energies
            double kinetic_energy = 0.0, potential_energy = 0.0, total_energy = 0.0, virial_equilibrium = 0.0;
            computeGlobalEnergiesParallel(masses, positions, local_velocities, G,
                                        rank, size, displs, sendcounts, local_n,
                                        kinetic_energy, potential_energy, total_energy, virial_equilibrium);

            int body_id_offset = displs[rank];
            // write the current state to vtp file
            writeVTPFile(rank, vs_counter,
                         local_masses, local_positions, local_velocities,
                         local_accelerations, local_names, local_orbit_classes,
                         body_id_offset,
                         kinetic_energy, potential_energy, total_energy, virial_equilibrium,
                         vs_dir);

            

            MPI_Barrier(MPI_COMM_WORLD); 
            // ensure all vtp files are written before updating pvd
            
            if (rank == 0) {
                // update the pvd file with the new timestep data
                updatePVDFile(pvdFilename, size, vs_counter, t, vs_dir);
                LOG(0, "saved state for visualization step " << vs_counter);
            }

            vs_counter++;
        }
    }

    auto simulation_end = MPI_Wtime(); // end simulation timing
    LOG(0, "simulation completed.");

    // output timing information and summed up distances
    if (rank == 0) {
        double dx = 0.0, dy = 0.0, dz = 0.0;
        for (int i = 0; i < num_bodies; ++i) {
            dx += positions[i].x;
            dy += positions[i].y;
            dz += positions[i].z;
        }

        std::cout << "SIMULATION_TIME: " << (simulation_end - simulation_start) << std::endl;
        std::cout << "TOTAL_TIME: " << (MPI_Wtime() - total_start) << std::endl;
        std::cout << "SUMMED_DIST: " << dx << " " << dy << " " << dz << std::endl;
    }


    // clean up mpi datatypes and finalize mpi
    MPI_Type_free(&MPI_VECTOR);
    MPI_Finalize();

    return 0;
}
