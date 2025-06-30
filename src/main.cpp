#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <filesystem>


#include "cxxopts.hpp"
#include "parse_time.h"
#include "body.h"
#include "io.h"
#include "morton_keys.h"
#include "linear_octree.h"
#include "load_balancing.h"
#include "exchange.h"
#include "traversal.h"
#include "policy.h"



// Forward declarations of MPI datatypes
MPI_Datatype MPI_POSITION, MPI_VELOCITY, MPI_ACCELERATION, MPI_NODE, MPI_ID;

int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // CLI parsing
    cxxopts::Options opt(argv[0], "Phase‑2 Barnes–Hut");
    opt.add_options()
        ("f,file",    "CSV file (mass,x,y,z,vx,vy,vz)",   cxxopts::value<std::string>())
        ("dt",        "Time step",                       cxxopts::value<std::string>()->default_value("1d"))
        ("tend",      "End time",                       cxxopts::value<std::string>()->default_value("1y"))
        ("vs",        "Visualization step interval", cxxopts::value<std::string>()->default_value("10d"))
        ("o,outdir",  "Output directory for visualization", cxxopts::value<std::string>()->default_value("sim_out"))
        ("b,bodies",  "number of bodies to simulate", cxxopts::value<int>()->default_value("-1"))
        // ("lb",        "Load balance policy: hist256",   cxxopts::value<std::string>()->default_value("hist256"))
        ("fc",        "Force calculation: tree|let",    cxxopts::value<std::string>()->default_value("tree"))
        ("theta",     "BH opening angle",               cxxopts::value<double>()->default_value("0.5"));
    const auto args = opt.parse(argc, argv);

    const double dt    = parseTime(args["dt"].as<std::string>());
    const double t_end = parseTime(args["tend"].as<std::string>());
    const double vs_interval = parseTime(args["vs"].as<std::string>());
    const std::string out_dir = args["outdir"].as<std::string>();
    const double theta = args["theta"].as<double>();
    const int nbodies = args["bodies"].as<int>();

    // LBPolicy lbPol = LBPolicy::Hist256;
    FCPolicy fcPol = (args["fc"].as<std::string>() == "let") ? FCPolicy::LET : FCPolicy::Tree;

    // Load data on rank 0 then scatter
    std::vector<Position>  init_pos;
    std::vector<Velocity>  init_vel;
    std::vector<double>    init_mass;
    std::vector<uint64_t>  init_ids;

    if (rank == 0) {
        if (!args.count("file")) {
            std::cerr << "Error: input file required (-f)\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (!readCSV(args["file"].as<std::string>(), init_ids, init_mass, init_pos, init_vel, nbodies))
            MPI_Abort(MPI_COMM_WORLD, 1);
        std::cout << "Loaded " << init_mass.size() << " bodies\n";
        std::filesystem::create_directories(out_dir); // Create output dir
    }

    // Broadcast total N
    int N_total = static_cast<int>(init_mass.size());
    MPI_Bcast(&N_total, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> counts(size), displs(size);
    int base = N_total / size, rem = N_total % size;
    for (int i = 0; i < size; ++i) {
        counts[i] = base + (i < rem);
        displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
    }

    std::vector<Position>  local_pos(counts[rank]);
    std::vector<Velocity>  local_vel(counts[rank]);
    std::vector<double>    local_mass(counts[rank]);
    std::vector<uint64_t>  local_ids(counts[rank]);

    // MPI datatypes 
    MPI_Type_contiguous(3, MPI_DOUBLE, &MPI_POSITION);  MPI_Type_commit(&MPI_POSITION);
    MPI_Type_contiguous(3, MPI_DOUBLE, &MPI_VELOCITY);  MPI_Type_commit(&MPI_VELOCITY);
    MPI_Type_contiguous(3, MPI_DOUBLE, &MPI_ACCELERATION); MPI_Type_commit(&MPI_ACCELERATION);
    MPI_Type_contiguous(1, MPI_UINT64_T, &MPI_ID);      MPI_Type_commit(&MPI_ID);

    int blk[3]   = {1, 1, 4};
    MPI_Aint dsp[3] = {offsetof(NodeRecord, prefix), offsetof(NodeRecord, depth), offsetof(NodeRecord, mass)};
    MPI_Datatype typ[3] = {MPI_UINT64_T,
                           MPI_UNSIGNED_CHAR, 
                           MPI_DOUBLE};
    MPI_Type_create_struct(3, blk, dsp, typ, &MPI_NODE); MPI_Type_commit(&MPI_NODE);

    // Scatter initial data
    MPI_Scatterv(init_pos.data(),  counts.data(), displs.data(), MPI_POSITION,
                 local_pos.data(), counts[rank], nullptr == init_pos.data() ? MPI_DATATYPE_NULL : MPI_POSITION,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(init_vel.data(),  counts.data(), displs.data(), MPI_VELOCITY,
                 local_vel.data(), counts[rank], nullptr == init_vel.data() ? MPI_DATATYPE_NULL : MPI_VELOCITY,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(init_mass.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 local_mass.data(), counts[rank], nullptr == init_mass.data() ? MPI_DATATYPE_NULL : MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Scatterv(init_ids.data(),  counts.data(), displs.data(), MPI_ID,
                local_ids.data(), counts[rank], (nullptr == init_ids.data() ? MPI_DATATYPE_NULL : MPI_ID),
                0, MPI_COMM_WORLD);

    // Main loop
    const double G = 1.48812e-34;
    std::vector<Acceleration> local_acc(local_pos.size());
    double next_vis_time = 0.0;
    int vis_step = 0;
    double t = 0.0;

    if (rank == 0)
        std::cout << "Starting simulation (dt=" << dt << ", tend=" << t_end << ")\n";


    for (int step = 0; t < t_end; ++step, t += dt) {
        // 1. Half‑kick (initial accel known from previous loop or assumed zero)
        if (t > 0) {
            for (size_t i = 0; i < local_pos.size(); ++i) {
                local_vel[i].x += 0.5 * dt * local_acc[i].x;
                local_vel[i].y += 0.5 * dt * local_acc[i].y;
                local_vel[i].z += 0.5 * dt * local_acc[i].z;
            }
        }

        // 2. Drift
        for (size_t i = 0; i < local_pos.size(); ++i) {
            local_pos[i].x += dt * local_vel[i].x;
            local_pos[i].y += dt * local_vel[i].y;
            local_pos[i].z += dt * local_vel[i].z;
        }

        // 3. Load balance (every step for now)
        BoundingBox global_bb;
        CodesAndNorm cn = rebalance_bodies(rank, size, local_pos, local_mass, local_vel, local_ids, global_bb);

        // 4. Build local tree
        OctreeMap my_tree = buildOctree2(cn.code, local_pos, local_mass);

        // 5. Exchange trees (placeholder)
        OctreeMap full_tree;
        switch (fcPol) {
            case FCPolicy::Tree:
                exchange_whole_trees(my_tree, full_tree, MPI_NODE, rank, size);
                break;
            case FCPolicy::LET:
                if (rank == 0)
                    std::cout << "LET exchange not yet implemented – falling back" << std::endl;
                exchange_whole_trees(my_tree, full_tree, MPI_NODE, rank, size);
                break;
            default:
                MPI_Abort(MPI_COMM_WORLD, 999);
        }

        // 6. Compute new accelerations
        local_acc.resize(local_pos.size());
        bhAccelerations(full_tree, cn.code,  local_pos, theta, G, 0.0, local_acc);

        // 7. Second half‑kick
        for (size_t i = 0; i < local_pos.size(); ++i) {
            local_vel[i].x += 0.5 * dt * local_acc[i].x;
            local_vel[i].y += 0.5 * dt * local_acc[i].y;
            local_vel[i].z += 0.5 * dt * local_acc[i].z;
        }
        // 8. Visualization
        if (t >= next_vis_time || step == 0) {
            writeSnapshot(rank, vis_step, local_ids, local_mass,
                        local_pos, local_vel, local_acc, out_dir);

            MPI_Barrier(MPI_COMM_WORLD);

            if (rank == 0) {
                updatePVDFile(args, size, vis_step, t, out_dir);
                std::cout << "Saved frame " << vis_step << " at t=" << t << std::endl;
            }
            next_vis_time += vs_interval;
            vis_step++;
        }
    }

    if (rank == 0) std::cout << "Simulation finished.\n";


    // Cleanup
    MPI_Type_free(&MPI_ID);
    MPI_Type_free(&MPI_POSITION);
    MPI_Type_free(&MPI_VELOCITY);
    MPI_Type_free(&MPI_ACCELERATION);
    MPI_Type_free(&MPI_NODE);
    MPI_Finalize();
    return 0;
}

