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



// Forward declarations of MPI datatypes
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
        ("fc",        "Force calculation: tree|let",    cxxopts::value<std::string>()->default_value("tree"))
        ("theta",     "BH opening angle",               cxxopts::value<double>()->default_value("1.05"))
        ("max-depth", "Maximum depth for LET traversal (0-21, 21 means full traversal)", cxxopts::value<int>()->default_value("21"))
        ("bucket-bits", "Number of bits for histogram buckets (2^n buckets)", cxxopts::value<int>()->default_value("18"))
        ("rebalance-interval", "Steps between load rebalancing", cxxopts::value<int>()->default_value("24"));

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

    // accumulators (in microseconds)
    long long agg_halfkick   = 0;
    long long agg_drift      = 0;
    long long agg_rebalance  = 0;
    long long agg_pretree    = 0;
    long long agg_build      = 0;
    long long agg_exchange   = 0;
    long long agg_accel      = 0;
    long long agg_kick2      = 0;
    long long agg_viz        = 0;
    auto sim_start_time = std::chrono::high_resolution_clock::now();
    auto last_vis_time = sim_start_time;

    if (rank == 0)
        std::cout << "Starting simulation (dt=" << dt << ", tend=" << t_end << ")\n";

    std::vector<std::vector<uint64_t>> rank_domain_keys;
    BoundingBox hist_global_bb;
    std::vector<std::pair<long long, int>> last_global_hist;
    for (int step = 0; t < t_end; ++step, t += dt) {
        // 1. Half-kick
        auto t1_start = std::chrono::high_resolution_clock::now();
        if (t > 0) {
            #pragma omp parallel for
            for (size_t i = 0; i < local_pos.size(); ++i) {
                local_vel[i].x += 0.5 * dt * local_acc[i].x;
                local_vel[i].y += 0.5 * dt * local_acc[i].y;
                local_vel[i].z += 0.5 * dt * local_acc[i].z;
            }
        }
        auto t1_end = std::chrono::high_resolution_clock::now();
        agg_halfkick += std::chrono::duration_cast<std::chrono::microseconds>(t1_end - t1_start).count();

        // 2. Drift
        auto t2_start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for (size_t i = 0; i < local_pos.size(); ++i) {
            local_pos[i].x += dt * local_vel[i].x;
            local_pos[i].y += dt * local_vel[i].y;
            local_pos[i].z += dt * local_vel[i].z;
        }
        auto t2_end = std::chrono::high_resolution_clock::now();
        agg_drift += std::chrono::duration_cast<std::chrono::microseconds>(t2_end - t2_start).count();

        // 3. Load balance
        auto t3_start = std::chrono::high_resolution_clock::now();
        if (step % rebalance_interval == 0) {
            if (rank == 0) std::cout << "Step " << step << ": Rebalancing particles...\n";

            BoundingBox current_local_bb = compute_local_bbox(local_pos);
            BoundingBox current_global_bb = compute_global_bbox(current_local_bb);
            hist_global_bb = current_global_bb;
            std::vector<uint64_t> current_codes = generateMortonCodes(local_pos, current_global_bb);
            rebalance_bodies(rank, size, current_codes, local_pos, local_mass, local_vel, local_ids, rank_domain_keys, last_global_hist, bucket_bits);
        }
        auto t3_end = std::chrono::high_resolution_clock::now();
        agg_rebalance += std::chrono::duration_cast<std::chrono::microseconds>(t3_end - t3_start).count();

        // 4. PREPARE FOR TREE BUILD
        auto t4_start = std::chrono::high_resolution_clock::now();
        BoundingBox local_bb = compute_local_bbox(local_pos);
        BoundingBox global_bb = compute_global_bbox(local_bb);

        std::vector<uint64_t> codes = generateMortonCodes(local_pos, global_bb);
         sortBodiesByMortonKey(codes, local_pos, local_mass, local_vel, local_ids);
        auto t4_end = std::chrono::high_resolution_clock::now();
        agg_pretree += std::chrono::duration_cast<std::chrono::microseconds>(t4_end - t4_start).count();

        // 5. BUILD LOCAL TREE
        auto t5_start = std::chrono::high_resolution_clock::now();
        OctreeMap my_tree = buildOctreeBottomUp(codes, local_pos, local_mass);
        // OctreeMap my_tree2 =  buildOctree2(codes, local_pos, local_mass);
        // compareFlattened(serializeTreeToRecords(my_tree) ,serializeTreeToRecords(my_tree2));
        auto t5_end = std::chrono::high_resolution_clock::now();
        agg_build += std::chrono::duration_cast<std::chrono::microseconds>(t5_end - t5_start).count();

        // 6. Exchange trees
        auto t6_start = std::chrono::high_resolution_clock::now();
        OctreeMap full_tree;
        std::vector<NodeRecord> remote_nodes;
        std::vector<int> remote_node_counts; // For visualization
        switch (fcPol) {
            case FCPolicy::Tree:
                exchangeFullTrees(my_tree, full_tree, MPI_NODE, rank, size);
                break;
            case FCPolicy::LET:
                exchangeEssentialTrees(my_tree, remote_nodes, remote_node_counts, global_bb, theta, MPI_NODE, rank, size, rank_domain_keys, max_let_depth);
                break;
            default:
                MPI_Abort(MPI_COMM_WORLD, 999);
        }
        auto t6_end = std::chrono::high_resolution_clock::now();
        agg_exchange += std::chrono::duration_cast<std::chrono::microseconds>(t6_end - t6_start).count();

        // 7. Compute new accelerations
        auto t7_start = std::chrono::high_resolution_clock::now();
        local_acc.resize(local_pos.size());

        if (fcPol == FCPolicy::Tree) {
            computeAccelerations(full_tree, codes, local_pos, theta, G, 0.0, global_bb, local_acc);
        } else { // FCPolicy::LET
            // BHFlat local_flat    = buildBHFlat_safe(my_tree, G);
            // computeAccelerationsWithRemoteDirectSum_cached(local_cache, remote_nodes, codes, local_pos,
            //                    theta, G, 0.0, global_bb, local_acc, rank);
            // computeAccelerationsWithRemoteDirectSum(my_tree, remote_nodes, codes, local_pos, theta, G, 0.0, global_bb, local_acc);
            // computeAccelerationsWithRemoteDirectSum_flat_remoteBH(local_flat, remote_nodes, codes, local_pos,
            //                             theta, G, 0.0, global_bb, local_acc, rank);
            computeAccelerationsWithLET(my_tree, remote_nodes, codes, local_pos, theta, G, 0.0, global_bb, local_acc, rank);

        }
        auto t7_end = std::chrono::high_resolution_clock::now();
        agg_accel += std::chrono::duration_cast<std::chrono::microseconds>(t7_end - t7_start).count();

        // 8. Second half-kick
        auto t8_start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for (size_t i = 0; i < local_pos.size(); ++i) {
            local_vel[i].x += 0.5 * dt * local_acc[i].x;
            local_vel[i].y += 0.5 * dt * local_acc[i].y;
            local_vel[i].z += 0.5 * dt * local_acc[i].z;
        }
        auto t8_end = std::chrono::high_resolution_clock::now();
        agg_kick2 += std::chrono::duration_cast<std::chrono::microseconds>(t8_end - t8_start).count();

        // 9. Visualization (and report)
        auto t9_start = std::chrono::high_resolution_clock::now();
        if (t >= next_vis_time || step == 0) {
            writeSnapshot(rank, vis_step, local_ids, local_mass,
                          local_pos, local_vel, local_acc, out_dir);

            if (fcPol == FCPolicy::LET) {
                writeReceivedLETs(rank, vis_step, remote_nodes, remote_node_counts, global_bb, out_dir);
            }

            if (rank == 0 && !last_global_hist.empty()) {
                // global_bb is computed right before tree building, so it's available and current.
                writeHistogram(vis_step, last_global_hist, hist_global_bb, out_dir);
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
                // compute totals
                long long total_us = agg_halfkick + agg_drift + agg_rebalance +
                                     agg_pretree + agg_build + agg_exchange +
                                     agg_accel + agg_kick2;
                double total_s = total_us / 1e6;

                // report total
                std::cout << std::fixed << std::setprecision(3)
                          << "  Total time since last frame: "
                          << total_us << " µs (" << total_s << " s)\n";

                // report percentages
                std::cout << std::fixed << std::setprecision(1)
                          << "  half-kick:   " << (100.0 * agg_halfkick  / total_us) << "%\n"
                          << "  drift:       " << (100.0 * agg_drift     / total_us) << "%\n"
                          << "  rebalance:   " << (100.0 * agg_rebalance / total_us) << "%\n"
                          << "  prep-tree:   " << (100.0 * agg_pretree   / total_us) << "%\n"
                          << "  build-tree:  " << (100.0 * agg_build     / total_us) << "%\n"
                          << "  exchange:    " << (100.0 * agg_exchange  / total_us) << "%\n"
                          << "  accel:       " << (100.0 * agg_accel     / total_us) << "%\n"
                          << "  second-kick: " << (100.0 * agg_kick2     / total_us) << "%\n";

                // reset accumulators
                agg_halfkick  = agg_drift  = agg_rebalance = 0;
                agg_pretree   = agg_build  = agg_exchange  = 0;
                agg_accel     = agg_kick2 = 0;
            }

            next_vis_time += vs_interval;
            vis_step++;
        }
        auto t9_end = std::chrono::high_resolution_clock::now();
        agg_viz += std::chrono::duration_cast<std::chrono::microseconds>(t9_end - t9_start).count();
    }

    auto sim_end_time = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        std::cout << "Simulation finished.\n";
        auto total_sim_duration_s = std::chrono::duration_cast<std::chrono::duration<double>>(sim_end_time - sim_start_time).count();
        std::cout << "Total simulation time: " << std::fixed << std::setprecision(2) << total_sim_duration_s << " s\n";
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