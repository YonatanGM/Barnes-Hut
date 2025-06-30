#include "exchange.h"
#include <numeric>

/* helper – flatten + MPI_Allgatherv ------------------------------- */
// static void allgather_nodes(const std::vector<NodeRecord>& sendBuf,
//                             MPI_Datatype MPI_NODE,
//                             int rank,int size,
//                             std::vector<NodeRecord>& recvBuf)
// {
//     int sN = sendBuf.size();
//     std::vector<int> nPer(size);
//     MPI_Allgather(&sN,1,MPI_INT,nPer.data(),1,MPI_INT,MPI_COMM_WORLD);
//     std::vector<int> dPer(size,0);
//     std::partial_sum(nPer.begin(),nPer.end()-1,dPer.begin()+1);
//     int tot = dPer.back()+nPer.back();
//     recvBuf.resize(tot);
//     MPI_Allgatherv(sendBuf.data(),sN,MPI_NODE,
//                    recvBuf.data(),nPer.data(),dPer.data(),
//                    MPI_NODE,MPI_COMM_WORLD);
// }

/* ================================================================= */

void exchange_whole_trees(const OctreeMap &local_tree, OctreeMap &full_tree,
                          MPI_Datatype node_type, int /*rank*/, int size) {
    std::vector<NodeRecord> send_buf = flattenTree(local_tree);
    int send_cnt = static_cast<int>(send_buf.size());

    std::vector<int> recv_counts(size);
    MPI_Allgather(&send_cnt, 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
                  MPI_COMM_WORLD);

    std::vector<int> displs(size, 0);
    for (int i = 1; i < size; ++i) displs[i] = displs[i - 1] + recv_counts[i - 1];
    int total_nodes = displs.back() + recv_counts.back();

    std::vector<NodeRecord> recv_buf(total_nodes);
    MPI_Allgatherv(send_buf.data(), send_cnt, node_type,
                   recv_buf.data(), recv_counts.data(), displs.data(), node_type,
                   MPI_COMM_WORLD);

    full_tree.clear();
    mergeIntoTree(full_tree, recv_buf);
}

/* ================================================================= */
// void exchange_bodies(const std::vector<Position>& lPos,
//                      const std::vector<double>&   lMass,
//                      std::vector<Position>&       gPos,
//                      std::vector<double>&         gMass,
//                      MPI_Datatype                 MPI_POS,
//                      int                          rank,
//                      int                          size,
//                      const std::vector<int>&      recvCnt,
//                      const std::vector<int>&      recvDisp)
// {
//     int Ntot = std::accumulate(recvCnt.begin(),recvCnt.end(),0);
//     gPos .resize(Ntot);
//     gMass.resize(Ntot);
//     MPI_Allgatherv(lPos .data(),lPos .size(),MPI_POS,
//                    gPos .data(),recvCnt.data(),recvDisp.data(),
//                    MPI_POS,MPI_COMM_WORLD);
//     MPI_Allgatherv(lMass.data(),lMass.size(),MPI_DOUBLE,
//                    gMass.data(),recvCnt.data(),recvDisp.data(),
//                    MPI_DOUBLE,MPI_COMM_WORLD);
// }

/* =================================================================
 * Extremely simple LET: ship every node with depth ≤ 2
 * (i.e. root + its 8 + 64 grandchildren).  Good enough to prove
 * the plumbing; replace with real MAC-based tagging later.
 * =============================================================== */
// static std::vector<NodeRecord>
// pickLET(const OctreeMap& tree)
// {
//     std::vector<NodeRecord> out;
//     out.reserve(100);
//     for(auto &[k,n] : tree)
//         if(k.depth <= 2 && n.mass>0)
//             out.push_back({k.prefix,k.depth,n.mass,
//                            n.comX,n.comY,n.comZ});
//     return out;
// }

// void exchange_LET(const OctreeMap& myTree,
//                   OctreeMap& merged,
//                   LBPolicy   /*lbPol*/,
//                   double     /*theta*/,
//                   MPI_Datatype MPI_NODE,
//                   int rank,int size)
// {
//     auto sendBuf = pickLET(myTree);
//     std::vector<NodeRecord> recvBuf;
//     allgather_nodes(sendBuf,MPI_NODE,rank,size,recvBuf);

//     merged = myTree;
//     mergeIntoTree(recvBuf,merged);
// }
