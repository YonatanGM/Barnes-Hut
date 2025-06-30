#pragma once
#include <vector>
#include <mpi.h>

#include "body.h"
#include "policy.h"   
#include "linear_octree.h"
         

/* ------------ TREES (broadcast every node) ----------------------- */

void exchange_whole_trees(const OctreeMap &local_tree, OctreeMap &full_tree,
                          MPI_Datatype node_type, int rank, int size);

/* ------------ BODIES (broadcast positions+masses) ---------------- */
void exchange_bodies(const std::vector<Position>& lPos,
                     const std::vector<double>&   lMass,
                     std::vector<Position>&       gPos,
                     std::vector<double>&         gMass,
                     MPI_Datatype                 MPI_POS,
                     int                          rank,
                     int                          size,
                     const std::vector<int>&      recvCnt,
                     const std::vector<int>&      recvDisp);

/* ------------ LET push-once  ------------------------------------ */
void exchange_LET(const OctreeMap&  myTree,
                  OctreeMap&        merged,
                  LBPolicy          lbPol,
                  double            theta,
                  MPI_Datatype      MPI_NODE,
                  int               rank,
                  int               size);
