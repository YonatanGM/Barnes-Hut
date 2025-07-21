#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <filesystem>
#include <chrono>
#include <iomanip>


#include "cxxopts.hpp"
#include "parse_time.h"
#include "body.h"
#include "io.h"
#include "morton_keys.h"
#include "linear_octree.h"
#include "load_balancing.h"
#include "exchange.h"
#include "traversal.h"
#include "utility.h"
#include "policy.h"


// Forward declarations
double performReferenceAndErrorAnalysis(
    bool is_reference,
    const std::string& ref_dir,
    int rank,
    int size,
    int N_total,
    const std::vector<Position>& local_pos,
    const std::vector<uint64_t>& local_ids);

// MPI datatypes
MPI_Datatype MPI_POSITION, MPI_VELOCITY, MPI_ACCELERATION, MPI_NODE, MPI_ID, MPI_BoundingBox;


int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // CLI parsing
    cxxopts::Options opt(argv[0], "Phase‑2 Barnes–Hut");
    opt.add_options()
        ("h,help",    "Print usage")
        ("f,file",    "CSV file (mass,x,y,z,vx,vy,vz)",   cxxopts::value<std::string>())
        ("dt",        "Time step",                       cxxopts::value<std::string>()->default_value("1d"))
        ("tend",      "End time",                       cxxopts::value<std::string>()->default_value("1y"))
        ("vs",        "Visualization step interval", cxxopts::value<std::string>()->default_value("10d"))
        ("o,outdir",  "Output directory for visualization", cxxopts::value<std::string>()->default_value("sim_out"))
        ("b,bodies",  "number of bodies to simulate", cxxopts::value<int>()->default_value("-1"))
        ("fc", "Force calculation: tree|let|let_direct", cxxopts::value<std::string>()->default_value("let"))
        ("theta",     "BH opening angle",               cxxopts::value<double>()->default_value("1.05"))
        ("max-depth", "Maximum depth for LET traversal (0-21, 21 means full traversal)", cxxopts::value<int>()->default_value("21"))
        ("bucket-bits", "Number of bits for histogram buckets (2^n buckets)", cxxopts::value<int>()->default_value("18"))
        ("rebalance-interval", "Steps between load rebalancing", cxxopts::value<int>()->default_value("24"))
        ("r,reference", "Run as reference and save final positions", cxxopts::value<bool>()->default_value("false"));

    const auto args = opt.parse(argc, argv);
    if (args.count("help")) {
        if (rank == 0) {
            std::cout << opt.help() << std::endl;
        }
        MPI_Finalize();
        exit(0);
    }
    const double dt    = parseTime(args["dt"].as<std::string>());
    const double t_end = parseTime(args["tend"].as<std::string>());
    const double vs_interval = parseTime(args["vs"].as<std::string>());
    const std::string out_dir = args["outdir"].as<std::string>();
    const double theta = args["theta"].as<double>();
    const int nbodies = args["bodies"].as<int>();
    const int max_let_depth = args["max-depth"].as<int>();
    const int bucket_bits = args["bucket-bits"].as<int>();
    const int rebalance_interval = args["rebalance-interval"].as<int>();


    if (rank == 0) { // Only print from rank 0 to avoid clutter
        // std::cout << "DEBUG (main.cpp): CLI parsed bucket_bits = " << bucket_bits << std::endl;
    }

    const bool is_reference = args["reference"].as<bool>();
    const std::string ref_dir = "reference";

    FCPolicy fcPol;
    const std::string fc_str = args["fc"].as<std::string>();

    if (is_reference) {
        fcPol = FCPolicy::Tree;
        if (rank == 0) {
            std::cout << "Running in reference mode. Force calculation method will be 'tree'." << std::endl;
        }
    } else {
        if (fc_str == "tree") {
            fcPol = FCPolicy::Tree;
        } else if (fc_str == "let_direct") {
            fcPol = FCPolicy::LET_DirectSum;
        } else {
            fcPol = FCPolicy::LET;
        }
    }

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
        std::filesystem::create_directories(out_dir);

        if (is_reference) {
            std::filesystem::create_directories(ref_dir);
        }
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

    int blk_lengths[2] = {1, 1};
    MPI_Aint displacements[2] = {offsetof(BoundingBox, min), offsetof(BoundingBox, max)};
    MPI_Datatype types[2] = {MPI_POSITION, MPI_POSITION};
    MPI_Type_create_struct(2, blk_lengths, displacements, types, &MPI_BoundingBox);
    MPI_Type_commit(&MPI_BoundingBox);

    // Scatter initial data
    MPI_Scatterv(init_pos.data(), counts.data(), displs.data(), MPI_POSITION, local_pos.data(), counts[rank], MPI_POSITION, 0, MPI_COMM_WORLD);
    MPI_Scatterv(init_vel.data(), counts.data(), displs.data(), MPI_VELOCITY, local_vel.data(), counts[rank], MPI_VELOCITY, 0, MPI_COMM_WORLD);
    MPI_Scatterv(init_mass.data(), counts.data(), displs.data(), MPI_DOUBLE, local_mass.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(init_ids .data(), counts.data(), displs.data(), MPI_ID,     local_ids .data(), counts[rank], MPI_ID,     0, MPI_COMM_WORLD);

    // Main loop
    const double G = 1.48812e-34;
    std::vector<Acceleration> local_acc(local_pos.size());
    double next_vis_time = 0.0;
    int vis_step = 0;
    double t = 0.0;
    constexpr int REBALANCE_INTERVAL = 20;


    if (rank == 0)
        std::cout << "Starting simulation (dt=" << dt << ", tend=" << t_end << ")\n";

    auto sim_start_time = std::chrono::high_resolution_clock::now();
    auto last_vis_time = sim_start_time;

    std::vector<std::vector<uint64_t>> rank_domain_keys;
    std::vector<std::pair<long long, int>> last_global_hist;

    for (int step = 0; t < t_end; ++step, t += dt) {
        // 1. half kick
        if (t > 0) {
            #pragma omp parallel for
            for (size_t i = 0; i < local_pos.size(); ++i) {
                local_vel[i].x += 0.5 * dt * local_acc[i].x;
                local_vel[i].y += 0.5 * dt * local_acc[i].y;
                local_vel[i].z += 0.5 * dt * local_acc[i].z;
            }
        }

        // 2. drift
        #pragma omp parallel for
        for (size_t i = 0; i < local_pos.size(); ++i) {
            local_pos[i].x += dt * local_vel[i].x;
            local_pos[i].y += dt * local_vel[i].y;
            local_pos[i].z += dt * local_vel[i].z;
        }

        // 3. Domain decomposition and load balancing
        // 3a. Calculate the BBox and Morton codes for the current particle state.
        BoundingBox local_bb = compute_local_bbox(local_pos);
        BoundingBox global_bb = compute_global_bbox(local_bb);
        BoundingBox bbox_for_histogram_vis = global_bb;
        std::vector<uint64_t> codes = generateMortonCodes(local_pos, global_bb);

        // 3b. Update domain decomposition every step for LET accuracy
        std::vector<int> splitters;
        update_rank_domains(size, codes, bucket_bits, rank_domain_keys, last_global_hist, splitters);

         // 3c. Body migration periodically
        if (step % rebalance_interval == 0) {
            if (rank == 0) std::cout << "Step " << step << ": Rebalancing particles...\n";

            rebalance_bodies(rank, size, splitters, codes,
                             local_pos, local_mass, local_vel, local_ids,
                             bucket_bits);

            // After rebalancing, we must recacluate the BBox and codes
            local_bb = compute_local_bbox(local_pos);
            global_bb = compute_global_bbox(local_bb);
            codes = generateMortonCodes(local_pos, global_bb);
        }

        // 4. prepare for tree build
        sortBodiesByMortonKey(codes, local_pos, local_mass, local_vel, local_ids);

        // 5. build local tree
        OctreeMap my_tree = buildOctreeBottomUp(codes, local_pos, local_mass);

        // 6. Exchange trees
        OctreeMap full_tree;
        std::vector<NodeRecord> remote_nodes;
        std::vector<int> remote_node_counts; // For visualization

        switch (fcPol) {
            case FCPolicy::Tree:
                exchangeFullTrees(my_tree, full_tree, MPI_NODE, rank, size);
                break;
            case FCPolicy::LET:
                exchangeEssentialTrees(my_tree, remote_nodes, remote_node_counts, global_bb, theta, MPI_NODE, rank, size, rank_domain_keys, max_let_depth, bucket_bits);
                break;
            case FCPolicy::LET_DirectSum:
                exchangeEssentialTrees(my_tree, remote_nodes, remote_node_counts, global_bb, theta, MPI_NODE, rank, size, rank_domain_keys, max_let_depth, bucket_bits);
                break;
            default:
                MPI_Abort(MPI_COMM_WORLD, 999);
        }

        // 7. Compute new accelerations
        switch (fcPol) {
            case FCPolicy::Tree:
                computeAccelerations(full_tree, codes, local_pos, theta, G, 0.0, global_bb, local_acc);
                break;

            case FCPolicy::LET:
                computeAccelerationsWithLET(my_tree, remote_nodes, codes, local_pos, theta, G, 0.0, global_bb, local_acc, rank);
                break;

            case FCPolicy::LET_DirectSum:
                computeAccelerationsWithRemoteDirectSum(my_tree, remote_nodes, codes, local_pos, theta, G, 0.0, global_bb, local_acc);
                break;
        }

        // 8. Second half-kick
        #pragma omp parallel for
        for (size_t i = 0; i < local_pos.size(); ++i) {
            local_vel[i].x += 0.5 * dt * local_acc[i].x;
            local_vel[i].y += 0.5 * dt * local_acc[i].y;
            local_vel[i].z += 0.5 * dt * local_acc[i].z;
        }

        // 9. Visualization and report
        if (t >= next_vis_time || step == 0) {
            writeSnapshot(rank, vis_step, local_ids, local_mass,
                          local_pos, local_vel, local_acc, out_dir);

            if (fcPol == FCPolicy::LET) {
                writeReceivedLETs(rank, vis_step, remote_nodes, remote_node_counts, global_bb, out_dir);
            }

            if (rank == 0 && !last_global_hist.empty()) {
                // global_bb is computed right before tree building, so it's available and current
                writeHistogram(vis_step, last_global_hist, bbox_for_histogram_vis, out_dir, bucket_bits);
            }

            MPI_Barrier(MPI_COMM_WORLD);

            if (rank == 0) {
                auto current_vis_time = std::chrono::high_resolution_clock::now();
                auto vis_interval_duration = std::chrono::duration_cast<std::chrono::duration<double>>(current_vis_time - last_vis_time).count();
                last_vis_time = current_vis_time;

                updatePVDFile(args, size, vis_step, t, out_dir);
                std::cout << "Saved frame " << vis_step << " at t=" << t
                          << " (Time since last frame: " << std::fixed << std::setprecision(2) << vis_interval_duration << " s)" << std::endl;


                if (fcPol == FCPolicy::LET) {
                    updateReceivedLETPVDFile(args, size, vis_step, t, out_dir);
                }

                if (!last_global_hist.empty()) {
                    updateHistogramPVDFile(args, vis_step, t, out_dir);
                }
            }

            next_vis_time += vs_interval;
            vis_step++;
        }
    }

    auto sim_end_time = std::chrono::high_resolution_clock::now();

    double summed_dist_error = performReferenceAndErrorAnalysis(
        is_reference, ref_dir, rank, size, N_total, local_pos, local_ids
    );


    if (rank == 0) {
        auto total_sim_duration_s = std::chrono::duration_cast<std::chrono::duration<double>>(sim_end_time - sim_start_time).count();
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "TOTAL_TIME          " << total_sim_duration_s << std::endl;
        std::cout << "SUMMED_DIST_ERROR   " << summed_dist_error << std::endl;
    }

    // Cleanup
    MPI_Type_free(&MPI_ID);
    MPI_Type_free(&MPI_POSITION);
    MPI_Type_free(&MPI_VELOCITY);
    MPI_Type_free(&MPI_ACCELERATION);
    MPI_Type_free(&MPI_NODE);
    MPI_Finalize();
    return 0;
}



/**
 * @brief either saves a reference file or computes error against it
 * @return The total summed distance error
 */
double performReferenceAndErrorAnalysis(
    bool is_reference,
    const std::string& ref_dir,
    int rank,
    int size,
    int N_total,
    const std::vector<Position>& local_pos,
    const std::vector<uint64_t>& local_ids)
{
    double summed_dist_error = 0.0;

    if (is_reference) {
        // First, get the final, correct counts from each rank after any rebalancing.
        int local_count = local_pos.size();
        std::vector<int> final_counts;
        if (rank == 0) {
            final_counts.resize(size);
        }
        MPI_Gather(&local_count, 1, MPI_INT, final_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        // On rank 0, calculate the correct displacements for the Gatherv.
        std::vector<int> final_displs;
        if (rank == 0) {
            final_displs.resize(size);
            final_displs[0] = 0;
            for (int i = 1; i < size; ++i) {
                final_displs[i] = final_displs[i - 1] + final_counts[i - 1];
            }
        }

        // Now perform the Gatherv with the correct counts and displacements.
        std::vector<Position> final_positions;
        std::vector<uint64_t> final_ids;
        if (rank == 0) {
            final_positions.resize(N_total);
            final_ids.resize(N_total);
        }

        MPI_Gatherv(local_pos.data(), local_count, MPI_POSITION,
                    final_positions.data(), final_counts.data(), final_displs.data(), MPI_POSITION,
                    0, MPI_COMM_WORLD);
        MPI_Gatherv(local_ids.data(), local_count, MPI_ID,
                    final_ids.data(), final_counts.data(), final_displs.data(), MPI_ID,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            saveReferenceCSV(ref_dir, final_positions, final_ids);
            std::cout << "Reference positions saved to " << ref_dir << "/final_ref.csv" << std::endl;
        }
    } else {
        // In a normal run, check for the reference file and compute the error in a distributed manner.
        if (std::filesystem::exists(std::filesystem::path(ref_dir) / "final_ref.csv")) {
            try {
                std::vector<Position> ref_positions = loadReferenceCSV(ref_dir, N_total);
                summed_dist_error = computeDistanceSum(local_pos, local_ids, ref_positions);
            } catch (const std::runtime_error& e) {
                if (rank == 0) std::cerr << "Error during error analysis: " << e.what() << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        } else {
            if (rank == 0) std::cout << "Reference file not found at " << ref_dir << "/final_ref.csv, skipping error analysis." << std::endl;
        }
    }
    return summed_dist_error;
}