#pragma once

#include "body.h"
#include "morton_keys.h"
#include <mpi.h>
#include <vector>



/**
 * @brief Computes a global workload histogram and determines the optimal domain
 *        decomposition for each rank.
 *
 * This function is designed to be called every timestep to ensure the domain
 * information used for LET generation is always up-to-date with particle positions.
 * It does not move any particle data.
 *
 * @param size The total number of MPI ranks.
 * @param codes The Morton codes for the local particles.
 * @param bucket_bits The number of bits for the histogram resolution.
 * @param[out] out_rank_domain_keys A map where key is rank and value is the list of bucket prefixes it owns.
 * @param[out] out_global_hist The complete global histogram (count per bucket, assigned rank).
 * @param[out] out_splitters The calculated splitter keys used to partition the domain.
 */
void update_rank_domains(
    int size,
    const std::vector<uint64_t>& codes,
    int bucket_bits,
    std::vector<std::vector<uint64_t>>& out_rank_domain_keys,
    std::vector<std::pair<long long, int>>& out_global_hist,
    std::vector<int>& out_splitters
);

/**
 * @brief Redistributes bodies among MPI ranks based on pre-computed splitters.
 *
 * This function performs only the data migration step using MPI_Alltoallv.
 * It should be called periodically, as it is a communication-heavy operation.
 *
 * @param rank The MPI rank of the calling process.
 * @param size The total number of MPI ranks.
 * @param splitters The splitter keys calculated by update_rank_domains.
 * @param codes The Morton codes for the local particles.
 * @param[in,out] local_pos On input, the local positions; on output, the new set after redistribution.
 * @param[in,out] local_mass On input/output, the corresponding local masses.
 * @param[in,out] local_vel On input/output, the corresponding local velocities.
 * @param[in,out] local_ids On input/output, the corresponding local particle IDs.
 * @param bucket_bits The number of bits for the histogram resolution (must match update_rank_domains).
 */
void rebalance_bodies(
    int rank, int size,
    const std::vector<int>& splitters, // This is now an input parameter
    const std::vector<uint64_t>& codes,
    std::vector<Position> &local_pos,
    std::vector<double> &local_mass,
    std::vector<Velocity> &local_vel,
    std::vector<uint64_t> &local_ids,
    int bucket_bits);


/**
 * @brief Redistributes bodies among MPI ranks based on a workload histogram
 *
 * This function performs a complete load balancing step:
 * 1. Builds a global histogram of particle distribution
 * 2. Calculates optimal splitters to divide the workload evenly
 * 3. Uses MPI_Alltoallv to send and receive bodies, moving them to their new owner rank
 *
 * @param rank The MPI rank of the calling process.
 * @param size The total number of MPI ranks.
 * @param mortonData The Morton codes for the local particles.
 * @param[in,out] local_pos On input, the local positions; on output, the new set of positions after redistribution.
 * @param[in,out] local_mass On input/output, the corresponding local masses.
 * @param[in,out] local_vel On input/output, the corresponding local velocities.
 * @param[in,out] local_ids On input/output, the corresponding local particle IDs.
 * @param[out] out_rank_domain_keys A map where key is rank and value is the list of bucket prefixes it owns
 * @param[out] out_global_histogram The complete global histogram (count per bucket, assigned rank)
 * @param bucket_bits The number of bits used to determine the histogram buckets (e.g., 18 for 2^18 buckets)
 */
void rebalance_bodies(
    int rank, int size,
    std::vector<uint64_t> codes,
    std::vector<Position> &local_pos,
    std::vector<double> &local_mass,
    std::vector<Velocity> &local_vel,
    std::vector<uint64_t> &local_ids,
    std::vector<std::vector<uint64_t>>& rank_domain_keys,
    std::vector<std::pair<long long, int>>& global_hist_out,
    int bucket_bits);