#include "load_balancing.h"
#include <numeric>
#include <algorithm>
#include <cassert>


CodesAndNorm rebalance_bodies(
    int rank, int size,
    std::vector<Position> &local_pos,
    std::vector<double> &local_mass,
    std::vector<Velocity> &local_vel,
    std::vector<uint64_t>  local_ids,
    BoundingBox& out_global_bb) {

    (void)rank;

    constexpr int BUCKET_BITS  = 12;            // 2^12 = 4096 buckets
    constexpr int NUM_BUCKETS  = 1 << BUCKET_BITS;

    // -------------------------------------------------
    // Step 1: Global bounding box for key normalisation
    // -------------------------------------------------
    BoundingBox local_bb;
    if (!local_pos.empty()) {
        for (const auto &p : local_pos) {
            local_bb.min.x = std::min(local_bb.min.x, p.x);
            local_bb.min.y = std::min(local_bb.min.y, p.y);
            local_bb.min.z = std::min(local_bb.min.z, p.z);
            local_bb.max.x = std::max(local_bb.max.x, p.x);
            local_bb.max.y = std::max(local_bb.max.y, p.y);
            local_bb.max.z = std::max(local_bb.max.z, p.z);
        }
    }
    // ranks with 0 bodies keep ±inf (safe for MIN/MAX reductions)

    MPI_Allreduce(&local_bb.min, &out_global_bb.min, 3, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_bb.max, &out_global_bb.max, 3, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);

    // recompute Morton keys with this global box
    CodesAndNorm cn = mortonCodes(local_pos, out_global_bb);

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
    CodesAndNorm final_cn = mortonCodes(local_pos, out_global_bb);
    
    std::vector<size_t> sort_idx(total_recv);
    std::iota(sort_idx.begin(), sort_idx.end(), 0);
    std::sort(sort_idx.begin(), sort_idx.end(), [&](size_t a, size_t b) {
        return final_cn.code[a] < final_cn.code[b];
    });

    auto permute = [&](auto &vec) {
        std::vector<std::decay_t<decltype(vec[0])>> tmp(vec.size());
        for (size_t i = 0; i < vec.size(); ++i) tmp[i] = vec[sort_idx[i]];
        vec.swap(tmp);
    };

    permute(local_pos);
    permute(local_mass);
    permute(local_vel);
    permute(local_ids); 
    permute(final_cn.code);
    permute(final_cn.norm);

    return final_cn;
}

// static int rootOctant(const Position&p,const BBox&b){
//     int xi=p.x>=0.5*(b.xmin+b.xmax);
//     int yi=p.y>=0.5*(b.ymin+b.ymax);
//     int zi=p.z>=0.5*(b.zmin+b.zmax);
//     return xi|(yi<<1)|(zi<<2);
// }

// /* ================================================================ */
// void load_balance(LBPolicy               policy,
//                   std::vector<Position>& posG,
//                   std::vector<double>&   massG,
//                   int                    rank,
//                   int                    size,
//                   std::vector<int>&      sendCnt,
//                   std::vector<int>&      displ,
//                   std::vector<Position>& localPos,
//                   std::vector<double>&   localMass)
// {
//     const int N = posG.size();
//     sendCnt.assign(size,0); displ.assign(size,0);

//     /* ----------------------------------------------------------------
//      * 1) decide owner rank for every body (rank 0) and broadcast
//      * ---------------------------------------------------------------- */
//     std::vector<int> owner;                    // only filled on rank 0
//     if(rank==0)
//     {
//         owner.resize(N);
//         if(policy==LBPolicy::Octant)
//         {   /* --- simple 8→4 pairing ---------------------------------- */
//             BBox bb = bbox(posG);
//             std::array<int,8> count{}; for(auto&p:posG) ++count[rootOctant(p,bb)];

//             std::array<int,8> order{0,1,2,3,4,5,6,7};
//             std::sort(order.begin(),order.end(),
//                       [&](int a,int b){ return count[a]>count[b]; });
//             std::array<int,8> oct2rank{};
//             for(int r=0;r<4;r++){
//                 oct2rank[order[r]]   = r;
//                 oct2rank[order[7-r]] = r;
//             }
//             for(int i=0;i<N;i++) owner[i]=oct2rank[rootOctant(posG[i],bb)];
//         }
//         else
//         {   /* --- 256-bucket histogram split --------------------------- */
//             constexpr int DEPTH=4, B=1<<DEPTH;         // 256
//             auto key = mortonCodes(posG);
//             std::vector<int> hist(B,0);
//             for(auto k:key) ++hist[k>>(63-3*DEPTH);

//             std::vector<int> pref(B+1,0);
//             std::partial_sum(hist.begin(),hist.end(),pref.begin()+1);
//             int per = (N+size-1)/size;
//             std::vector<int> cut(size+1,0); cut.back()=B;
//             int r=1;
//             for(int i=0;i<B && r<size;i++)
//                 if(pref[i]>=per*r) cut[r++]=i;

//             for(int i=0;i<N;i++){
//                 int b = key[i]>>(63-3*DEPTH);
//                 int dest = std::lower_bound(cut.begin(),cut.end(),b)-cut.begin()-1;
//                 owner[i]=dest;
//             }
//         }
//     }

//     /* broadcast owner list */
//     if(rank!=0) owner.resize(N);
//     MPI_Bcast(owner.data(),N,MPI_INT,0,MPI_COMM_WORLD);

//     /* ----------------------------------------------------------------
//      * 2) build sendCnt & displ on all ranks (cheap)                    */
//      /* ---------------------------------------------------------------- */
//     for(int o:owner) ++sendCnt[o];
//     std::partial_sum(sendCnt.begin(),sendCnt.end()-1,displ.begin()+1);

//     /* ----------------------------------------------------------------
//      * 3) rank 0 packs contiguous buffers                              */
//      /* ---------------------------------------------------------------- */
//     std::vector<Position> packPos(N);
//     std::vector<double>   packMass(N);
//     if(rank==0){
//         std::vector<int> off(size,0);
//         for(int i=0;i<N;i++){
//             int r=owner[i];
//             int dst = displ[r]+off[r]++;
//             packPos [dst]=posG [i];
//             packMass[dst]=massG[i];
//         }
//         posG.swap(packPos);
//         massG.swap(packMass);
//     }

//     /* ----------------------------------------------------------------
//      * 4) scatter                                                       */
//      /* ---------------------------------------------------------------- */
//     MPI_Datatype MPI_POS; MPI_Type_contiguous(3,MPI_DOUBLE,&MPI_POS);
//     MPI_Type_commit(&MPI_POS);

//     int localN = sendCnt[rank];
//     localPos.resize(localN);
//     localMass.resize(localN);

//     MPI_Scatterv(posG .data(),sendCnt.data(),displ.data(),MPI_POS,
//                  localPos.data(),localN,MPI_POS,0,MPI_COMM_WORLD);
//     MPI_Scatterv(massG.data(),sendCnt.data(),displ.data(),MPI_DOUBLE,
//                  localMass.data(),localN,MPI_DOUBLE,0,MPI_COMM_WORLD);

//     MPI_Type_free(&MPI_POS);
// }

