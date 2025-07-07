#include "load_balancing.h"
#include <numeric>
#include <algorithm>
#include <cassert>


void rebalance_bodies(
    int rank, int size,
    const CodesAndNorm &cn,
    std::vector<Position> &local_pos,
    std::vector<double> &local_mass,
    std::vector<Velocity> &local_vel,
    std::vector<uint64_t> &local_ids) {

    (void)rank;

    constexpr int BUCKET_BITS  = 18;            // 2^12 = 4096 buckets
    constexpr int NUM_BUCKETS  = 1 << BUCKET_BITS;


    // Step 2: build local histogram & reduce to global
    std::vector<long long> local_hist(NUM_BUCKETS, 0);
    for (uint64_t key : cn.code) {
        unsigned bucket_idx = key >> (64 - BUCKET_BITS);
        ++local_hist[bucket_idx];
    }
    std::vector<long long> global_hist(NUM_BUCKETS, 0);
    MPI_Allreduce(local_hist.data(), global_hist.data(), NUM_BUCKETS,
                  MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);


    // Step 3: choose P‑1 splitter buckets (inclusive)
    long long total_particles = std::accumulate(global_hist.begin(),
                                                global_hist.end(), 0LL);
    const long long ideal = (total_particles + size - 1) / size; // ceil

    std::vector<int> splitters; // length P‑1
    splitters.reserve(size - 1);
    long long run_sum = 0;
    for (int i = 0; i < NUM_BUCKETS && splitters.size() < static_cast<size_t>(size - 1); ++i) {
        run_sum += global_hist[i];
        if (run_sum >= static_cast<long long>(splitters.size() + 1) * ideal)
            splitters.push_back(i);
    }

    // Step 4: build send buffers + Alltoallv migration
    std::vector<int> send_counts(size, 0);
    for (uint64_t key : cn.code) {
        unsigned b = key >> (64 - BUCKET_BITS);
        int dest = std::upper_bound(splitters.begin(), splitters.end(),
                                    static_cast<int>(b)) - splitters.begin();
        ++send_counts[dest];
    }

    // prefix sums for send displacements
    std::vector<int> sdispls(size, 0);
    for (int i = 1; i < size; ++i) sdispls[i] = sdispls[i - 1] + send_counts[i - 1];
    int total_send = sdispls.back() + send_counts.back();

    // contiguous send buffers
    std::vector<Position>  pos_send(total_send);
    std::vector<double>    mass_send(total_send);
    std::vector<Velocity>  vel_send(total_send);
    std::vector<uint64_t>  ids_send(total_send);

    // fill the send buffers in bucket order
    std::vector<int> cursor = sdispls;
    for (size_t i = 0; i < local_pos.size(); ++i) {
        unsigned b = cn.code[i] >> (64 - BUCKET_BITS);
        int dest = std::upper_bound(splitters.begin(), splitters.end(),
                                    static_cast<int>(b)) - splitters.begin();
        int idx = cursor[dest]++;
        pos_send[idx]  = local_pos[i];
        mass_send[idx] = local_mass[i];
        vel_send[idx]  = local_vel[i];
        ids_send[idx]  = local_ids[i];
    }

    // Exchange counts
    std::vector<int> recv_counts(size, 0);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // recv displacements and resize local vectors
    std::vector<int> rdispls(size, 0);
    for (int i = 1; i < size; ++i) rdispls[i] = rdispls[i - 1] + recv_counts[i - 1];
    int total_recv = rdispls.back() + recv_counts.back();

    local_pos.resize(total_recv);
    local_mass.resize(total_recv);
    local_vel.resize(total_recv);
    local_ids.resize(total_recv);

    // Convenience type handles
    extern MPI_Datatype MPI_POSITION, MPI_VELOCITY, MPI_ID;

    MPI_Alltoallv(pos_send.data(),  send_counts.data(), sdispls.data(), MPI_POSITION,
                  local_pos.data(), recv_counts.data(), rdispls.data(), MPI_POSITION,
                  MPI_COMM_WORLD);
    MPI_Alltoallv(mass_send.data(), send_counts.data(), sdispls.data(), MPI_DOUBLE,
                  local_mass.data(), recv_counts.data(), rdispls.data(), MPI_DOUBLE,
                  MPI_COMM_WORLD);
    MPI_Alltoallv(vel_send.data(),  send_counts.data(), sdispls.data(), MPI_VELOCITY,
                  local_vel.data(), recv_counts.data(), rdispls.data(), MPI_VELOCITY,
                  MPI_COMM_WORLD);
    MPI_Alltoallv(ids_send.data(),  send_counts.data(), sdispls.data(), MPI_ID,
                local_ids.data(), recv_counts.data(), rdispls.data(), MPI_ID, MPI_COMM_WORLD);

    // Recompute codes for the new local particles, then sort everything
    // tree building stage needs the data in sorted order
    // CodesAndNorm final_cn = mortonCodes(local_pos, global_bb);

    // std::vector<size_t> sort_idx(total_recv);
    // std::iota(sort_idx.begin(), sort_idx.end(), 0);
    // std::sort(sort_idx.begin(), sort_idx.end(), [&](size_t a, size_t b) {
    //     return final_cn.code[a] < final_cn.code[b];
    // });

    // auto permute = [&](auto &vec) {
    //     std::vector<std::decay_t<decltype(vec[0])>> tmp(vec.size());
    //     for (size_t i = 0; i < vec.size(); ++i) tmp[i] = vec[sort_idx[i]];
    //     vec.swap(tmp);
    // };

    // permute(local_pos);
    // permute(local_mass);
    // permute(local_vel);
    // permute(local_ids);
    // permute(final_cn.code);
    // permute(final_cn.norm);

    // return final_cn;
}
