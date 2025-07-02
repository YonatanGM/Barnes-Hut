#pragma once
#include <vector>
#include <array>
#include <mpi.h>

#include "body.h"
#include "policy.h"
#include "utility.h"
#include "morton_keys.h"
#include "linear_octree.h"


void rebalance_bodies(
    int rank, int size,
    const CodesAndNorm &cn,
    std::vector<Position> &local_pos,
    std::vector<double> &local_mass,
    std::vector<Velocity> &local_vel,
    std::vector<uint64_t> &local_ids);


/* Packs & scatters the bodies according to the chosen policy.
 *   posG/massG : global copies on *all* ranks (broadcast beforehand)
 *   sendCnt/disp:   scatter pattern returned for later reuse
 *   localPos/Mass : out-vectors with the slice owned by this rank
 */
// void load_balance(LBPolicy               policy,
//                   std::vector<Position>& posG,
//                   std::vector<double>&   massG,
//                   int                    rank,
//                   int                    size,
//                   std::vector<int>&      sendCnt,
//                   std::vector<int>&      displ,
//                   std::vector<Position>& localPos,
//                   std::vector<double>&   localMass);