#include "linear_octree.h"
#include <algorithm>
#include <numeric>
#include <vector>
#include <iostream>
#include <oneapi/tbb/parallel_sort.h>
#include <omp.h>
#include <chrono>

// Gets the Morton prefix of a code at a specific depth.
static inline uint64_t mortonPrefix(uint64_t code, int depth) {
    if (depth == 0) return 0;
    // Creates a mask to keep the top 3*depth bits.
    int shift = 63 - 3 * depth;
    return code & (~0ULL << shift);
}



// ── Optimised, serial aggregation ───────────────────────────────────
OctreeMap buildOctree1(const std::vector<uint64_t>& mortonCodes,
                               const std::vector<Position>& positions,
                               const std::vector<double>&   masses)
{
    const size_t N = mortonCodes.size();
    if (N == 0) return {};

    constexpr uint8_t MAX_DEPTH = 21;
    OctreeMap tree;
    tree.reserve(N * 2);                // heuristic

    /* ----------------------------------------------------------------
       1. Insert / accumulate leaves and record their keys
       ---------------------------------------------------------------- */
    std::vector<OctreeKey> currentKeys;
    currentKeys.reserve(N);

    for (size_t i = 0; i < N; ++i)
    {
        OctreeKey leaf{ mortonCodes[i], MAX_DEPTH };
        OctreeNode& n = tree[leaf];      // create/find leaf
        double m      = masses[i];
        n.mass += m;
        n.comX += positions[i].x * m;
        n.comY += positions[i].y * m;
        n.comZ += positions[i].z * m;
        currentKeys.push_back(leaf);
    }

    std::sort(currentKeys.begin(), currentKeys.end(),
              [](const OctreeKey& a, const OctreeKey& b){
                  return a.prefix < b.prefix;
              });
    currentKeys.erase(std::unique(currentKeys.begin(), currentKeys.end()),
                      currentKeys.end());

    /* ----------------------------------------------------------------
       2. Bottom-up: generate parents only from keys we have
       ---------------------------------------------------------------- */
    for (int depth = MAX_DEPTH - 1; depth >= 0; --depth)
    {
        if (currentKeys.empty()) break;

        std::vector<OctreeKey> parentKeys;
        parentKeys.reserve(currentKeys.size() / 8 + 8);

        // 2a. Compute parents of this level’s children
        for (const auto& child : currentKeys)
            parentKeys.push_back({ mortonPrefix(child.prefix, depth),
                                   static_cast<uint8_t>(depth) });

        std::sort(parentKeys.begin(), parentKeys.end(),
                  [](const OctreeKey& a, const OctreeKey& b){
                      return a.prefix < b.prefix;
                  });
        parentKeys.erase(std::unique(parentKeys.begin(), parentKeys.end()),
                         parentKeys.end());

        // 2b. Aggregate child → parent  (serial, race-free)
        for (const auto& pk : parentKeys)
        {
            OctreeNode& p = tree[pk];    // create/find parent

            for (int oc = 0; oc < 8; ++oc)
            {
                uint64_t childPrefix = pk.prefix |
                    (uint64_t(oc) << (63 - 3*(depth+1)));
                auto it = tree.find({ childPrefix,
                                      static_cast<uint8_t>(depth+1) });
                if (it == tree.end()) continue;

                const OctreeNode& c = it->second;
                p.mass  += c.mass;
                p.comX  += c.comX;      // fixed copy
                p.comY  += c.comY;
                p.comZ  += c.comZ;
            }
        }

        // 2c. Parents become next iteration’s children
        currentKeys.swap(parentKeys);
    }

    /* ----------------------------------------------------------------
       3. Finalise centre-of-mass for every node
       ---------------------------------------------------------------- */
    for (auto& kv : tree)
    {
        OctreeNode& n = kv.second;
        if (n.mass > 0)
        {
            n.comX /= n.mass;
            n.comY /= n.mass;
            n.comZ /= n.mass;
        }
    }

    return tree;
}


OctreeMap buildOctree2(
    const std::vector<uint64_t>& mortonCodes,
    const std::vector<Position>& positions,
    const std::vector<double>&   masses) {

    const size_t N = mortonCodes.size();
    if (N == 0) return {};

    OctreeMap tree;
    constexpr uint8_t MAX_DEPTH = 21;

    // 1. Create leaf nodes and insert them into the tree
    for (size_t i = 0; i < N; ++i) {
        OctreeKey leaf_key = {mortonCodes[i], MAX_DEPTH};
        OctreeNode& node = tree[leaf_key]; // Creates node if it doesn't exist

        // Accumulate mass and weighted position for center-of-mass calculation
        node.mass += masses[i];
        node.comX += positions[i].x * masses[i];
        node.comY += positions[i].y * masses[i];
        node.comZ += positions[i].z * masses[i];
    }

    // 2. Build internal nodes from the bottom up
    // Iterate from the level just above the leaves up to the root.
    for (int d = MAX_DEPTH - 1; d >= 0; --d) {
        std::vector<OctreeKey> parent_keys;
        // Find all unique parent keys at this level
        for (const auto& [key, node] : tree) {
            if (key.depth == d + 1) {
                parent_keys.push_back({mortonPrefix(key.prefix, d), (uint8_t)d});
            }
        }
        // Sort and remove duplicates
        std::sort(parent_keys.begin(), parent_keys.end(), [](const auto& a, const auto& b) {
            return a.prefix < b.prefix;
        });
        parent_keys.erase(std::unique(parent_keys.begin(), parent_keys.end()), parent_keys.end());

        // For each unique parent, find its children and aggregate their properties
        for (const auto& p_key : parent_keys) {
            OctreeNode& parent_node = tree[p_key];

            // Iterate through all 8 possible child octants
            for (int octant = 0; octant < 8; ++octant) {
                uint64_t child_prefix = p_key.prefix | (static_cast<uint64_t>(octant) << (63 - 3 * (d + 1)));
                OctreeKey child_key = {child_prefix, (uint8_t)(d + 1)};

                auto it = tree.find(child_key);
                if (it != tree.end()) {
                    const OctreeNode& child_node = it->second;
                    parent_node.mass += child_node.mass;
                    parent_node.comX += child_node.comX;
                    parent_node.comY += child_node.comY;
                    parent_node.comZ += child_node.comZ;
                }
            }
        }
    }

    // 3. Finalize center-of-mass calculation for all nodes
    for (auto& [key, node] : tree) {
        if (node.mass > 0) {
            node.comX /= node.mass;
            node.comY /= node.mass;
            node.comZ /= node.mass;
        }
    }

    return tree;
}

// Compare two flattened octree vectors
void compareFlattened(const std::vector<NodeRecord> A,
                      const std::vector<NodeRecord> B,
                      double tol)
{
    // 1) sort by (depth,prefix)
    auto cmp = [](auto const &a, auto const &b) {
        if (a.depth != b.depth) return a.depth < b.depth;
        return a.prefix < b.prefix;
    };
    std::vector<NodeRecord> vA = A, vB = B;
    std::sort(vA.begin(), vA.end(), cmp);
    std::sort(vB.begin(), vB.end(), cmp);

    // 2) walk them in lock-step
    size_t i = 0, n = std::min(vA.size(), vB.size());
    bool allGood = true;
    for (; i < n; ++i) {
        auto &a = vA[i];
        auto &b = vB[i];
        if (a.prefix != b.prefix || a.depth != b.depth) {
            std::printf("Key mismatch at index %zu:\n", i);
            std::printf("  A: (%02u,0x%016llx)\n", a.depth, (unsigned long long)a.prefix);
            std::printf("  B: (%02u,0x%016llx)\n", b.depth, (unsigned long long)b.prefix);
            allGood = false;
            continue;
        }
        if (std::abs(a.mass - b.mass) > tol
         || std::abs(a.comX - b.comX) > tol
         || std::abs(a.comY - b.comY) > tol
         || std::abs(a.comZ - b.comZ) > tol)
        {
            std::printf("Value mismatch at (d=%u,0x%016llx):\n",
                        a.depth, (unsigned long long)a.prefix);
            std::printf("  A.mass=%.9e, B.mass=%.9e\n", a.mass, b.mass);
            std::printf("  A.COM=(%.9e,%.9e,%.9e)\n",
                        a.comX, a.comY, a.comZ);
            std::printf("  B.COM=(%.9e,%.9e,%.9e)\n",
                        b.comX, b.comY, b.comZ);
            allGood = false;
        }
    }

    // 3) check for size mismatches
    if (vA.size() != vB.size()) {
        std::printf("Size mismatch: A has %zu nodes, B has %zu nodes\n",
                    vA.size(), vB.size());
        allGood = false;
    }

    if (allGood) {
        std::cout << "✅ All " << n << " nodes match exactly!\n";
    } else {
        std::cout << "❌ Discrepancies detected.\n";
    }
}

// OctreeMap buildOctree2(
//     const std::vector<uint64_t>& code,   // SORTED ascending Morton keys
//     const std::vector<Position>& pos,    // co-sorted physical positions
//     const std::vector<double>&   mass    // co-sorted masses
// ) {
//     using clock = std::chrono::high_resolution_clock;
//     auto t0 = clock::now();

//     const size_t N = code.size();
//     if (N == 0) return {};

//     constexpr uint8_t MAX_L = 21;

//     /* ---------- 1. Branch scan ---------- */
//     auto t1 = clock::now();
//     auto lcpBits = [&](int i,int j)->int {
//         if (i<0||j<0||i>=int(N)||j>=int(N)) return -1;
//         uint64_t x = code[i] ^ code[j];
//         return x ? (63 - __builtin_clzll(x)) : 63;
//     };
//     struct Key { uint64_t prefix; uint8_t depth; };
//     std::vector<Key> branches;
//     branches.reserve(N>1? N-1 : 0);

//     for (int i = 0; i+1 < int(N); ++i) {
//         int lp = lcpBits(i, i-1), ln = lcpBits(i, i+1);
//         int delta = std::max(lp, ln);
//         int lvl   = delta / 3;
//         int dir   = (ln > lp ? +1 : -1);
//         int lo = 1, hi = N-1;
//         while (lo < hi) {
//             int mid = (lo + hi) >> 1;
//             if (lcpBits(i, i + dir*mid) > delta) lo = mid+1;
//             else                                 hi = mid;
//         }
//         branches.push_back({ mortonPrefix(code[i], lvl+1),
//                              static_cast<uint8_t>(lvl+1) });
//     }
//     auto t2 = clock::now();

//     /* ---------- 1b. Sort+dedupe branches ---------- */
//     std::sort(branches.begin(), branches.end(),
//               [](auto &A, auto &B){
//                   return A.depth < B.depth
//                       || (A.depth==B.depth && A.prefix < B.prefix);
//               });
//     branches.erase(std::unique(branches.begin(), branches.end(),
//                                [](auto &A, auto &B){
//                                    return A.depth==B.depth
//                                        && A.prefix==B.prefix;
//                                }),
//                    branches.end());
//     auto t3 = clock::now();

//     /* ---------- 2. Build key list ---------- */
//     std::vector<Key> allKeys;
//     allKeys.reserve((branches.size()+N)*(MAX_L+1) + 1);
//     allKeys.push_back({0ULL,0});  // root
//     for (auto &b : branches)
//         for (int d = b.depth; d >= 1; --d)
//             allKeys.push_back({ mortonPrefix(b.prefix,d), uint8_t(d) });
//     for (size_t i = 0; i < N; ++i) {
//         allKeys.push_back({ code[i], MAX_L });  // leaf
//         for (int d = MAX_L-1; d >= 1; --d)
//             allKeys.push_back({ mortonPrefix(code[i],d), uint8_t(d) });
//     }
//     auto t4 = clock::now();

//     /* ---------- 2b. Sort+dedupe allKeys ---------- */
//     std::sort(allKeys.begin(), allKeys.end(),
//               [](auto &A, auto &B){
//                   return A.depth < B.depth
//                       || (A.depth==B.depth && A.prefix < B.prefix);
//               });
//     allKeys.erase(std::unique(allKeys.begin(), allKeys.end(),
//                               [](auto &A, auto &B){
//                                   return A.depth==B.depth
//                                       && A.prefix==B.prefix;
//                               }),
//                   allKeys.end());
//     auto t5 = clock::now();

//     /* ---------- 3. Populate map ---------- */
//     OctreeMap tree;
//     tree.reserve(allKeys.size());
//     for (auto &k : allKeys)
//         tree[OctreeKey{k.prefix, k.depth}];
//     auto t6 = clock::now();

//     /* ---------- 4. Leaf accumulation ---------- */
//     for (size_t i = 0; i < N; ++i) {
//         auto &n = tree.at(OctreeKey{code[i], MAX_L});
//         n.mass  += mass[i];
//         n.comX  += mass[i] * pos[i].x;
//         n.comY  += mass[i] * pos[i].y;
//         n.comZ  += mass[i] * pos[i].z;
//     }
//     auto t7 = clock::now();

//     /* ---------- 5. Bottom-up reduce ---------- */
//     std::sort(allKeys.begin(), allKeys.end(),
//               [](auto &A, auto &B){
//                   return A.depth > B.depth
//                       || (A.depth==B.depth && A.prefix > B.prefix);
//               });
//     for (auto &k : allKeys) {
//         if (k.depth == MAX_L) continue;
//         auto &p = tree.at(OctreeKey{k.prefix, k.depth});
//         for (int oc = 0; oc < 8; ++oc) {
//             uint64_t child = k.prefix
//                            | (uint64_t(oc) << (63 - 3*(k.depth+1)));
//             auto it = tree.find(OctreeKey{child, uint8_t(k.depth+1)});
//             if (it == tree.end()) continue;
//             const auto &c = it->second;
//             p.mass  += c.mass;
//             p.comX  += c.comX;
//             p.comY  += c.comY;
//             p.comZ  += c.comZ;
//         }
//     }
//     auto t8 = clock::now();

//     /* ---------- 6. Finalise COM ---------- */
//     for (auto &kv : tree) {
//         auto &n = kv.second;
//         if (n.mass > 0) {
//             n.comX /= n.mass;
//             n.comY /= n.mass;
//             n.comZ /= n.mass;
//         }
//     }
//     auto t9 = clock::now();

//     /* ---------- 7. Timing report ---------- */
//     auto us = [](auto a, auto b){
//         return std::chrono::duration_cast<std::chrono::microseconds>(b-a).count();
//     };
//     std::cout << "[buildOctree2] µs: "
//               << "branch-scan="   << us(t1,t2)
//               << ", branch-sort=" << us(t2,t3)
//               << ", key-gen="     << us(t3,t4)
//               << ", key-sort="    << us(t4,t5)
//               << ", map-fill="    << us(t5,t6)
//               << ", accumulate="  << us(t6,t7)
//               << ", reduce="      << us(t7,t8)
//               << ", finalize="    << us(t8,t9)
//               << ", total="       << us(t0,t9)
//               << '\n';

//     return tree;
// }
// OctreeMap buildOctree2(
//     const std::vector<uint64_t>& code,   // SORTED ascending Morton keys
//     const std::vector<Position>& pos,    // co-sorted physical positions
//     const std::vector<double>&   mass    // co-sorted masses
// ) {
//     const size_t N = code.size();
//     if (N == 0) return {};

//     constexpr uint8_t MAX_L = 21;

//     // 1) Branch scan via longest-common-prefix (LCP) of adjacent leaves
//     auto lcpBits = [&](int i,int j)->int {
//         if (i<0||j<0||i>=int(N)||j>=int(N)) return -1;
//         uint64_t x = code[i] ^ code[j];
//         return x ? (63 - __builtin_clzll(x)) : 63;
//     };
//     struct Key { uint64_t prefix; uint8_t depth; };
//     std::vector<Key> branches;
//     branches.reserve(N>1? N-1 : 0);

//     for (int i = 0; i+1 < int(N); ++i) {
//         int lp = lcpBits(i, i-1), ln = lcpBits(i, i+1);
//         int delta = std::max(lp, ln);
//         int lvl   = delta / 3;
//         int dir   = (ln > lp ? +1 : -1);
//         int lo = 1, hi = N-1;
//         while (lo < hi) {
//             int mid = (lo + hi) >> 1;
//             if (lcpBits(i, i + dir*mid) > delta) lo = mid+1;
//             else                                 hi = mid;
//         }
//         branches.push_back({ mortonPrefix(code[i], lvl+1),
//                              static_cast<uint8_t>(lvl+1) });
//     }
//     // dedupe branches
//     std::sort(branches.begin(), branches.end(),
//               [](auto &A, auto &B){
//                   return A.depth < B.depth
//                       || (A.depth==B.depth && A.prefix < B.prefix);
//               });
//     branches.erase(std::unique(branches.begin(), branches.end(),
//                                [](auto &A, auto &B){
//                                    return A.depth==B.depth
//                                        && A.prefix==B.prefix;
//                                }),
//                    branches.end());

//     // 2) Build full key list: root + branch-ancestors + leaf-ancestors
//     std::vector<Key> allKeys;
//     allKeys.reserve((branches.size()+N)*(MAX_L+1) + 1);
//     allKeys.push_back({0ULL,0});  // root
//     for (auto &b : branches)
//         for (int d = b.depth; d >= 1; --d)
//             allKeys.push_back({ mortonPrefix(b.prefix,d), uint8_t(d) });
//     for (size_t i = 0; i < N; ++i) {
//         allKeys.push_back({ code[i], MAX_L });  // leaf
//         for (int d = MAX_L-1; d >= 1; --d)
//             allKeys.push_back({ mortonPrefix(code[i],d), uint8_t(d) });
//     }
//     // dedupe allKeys
//     std::sort(allKeys.begin(), allKeys.end(),
//               [](auto &A, auto &B){
//                   return A.depth < B.depth
//                       || (A.depth==B.depth && A.prefix < B.prefix);
//               });
//     allKeys.erase(std::unique(allKeys.begin(), allKeys.end(),
//                               [](auto &A, auto &B){
//                                   return A.depth==B.depth
//                                       && A.prefix==B.prefix;
//                               }),
//                   allKeys.end());

//     // 3) Populate map with empty nodes
//     OctreeMap tree;
//     tree.reserve(allKeys.size());
//     for (auto &k : allKeys)
//         tree[OctreeKey{k.prefix, k.depth}];

//     // 4) Accumulate leaf mass & weighted COM
//     for (size_t i = 0; i < N; ++i) {
//         auto &n = tree.at(OctreeKey{code[i], MAX_L});
//         n.mass  += mass[i];
//         n.comX  += mass[i] * pos[i].x;
//         n.comY  += mass[i] * pos[i].y;
//         n.comZ  += mass[i] * pos[i].z;
//     }

//     // 5) Bottom-up reduce (add children’s mass & mass×pos sums)
//     std::sort(allKeys.begin(), allKeys.end(),
//               [](auto &A, auto &B){
//                   return A.depth > B.depth
//                       || (A.depth==B.depth && A.prefix > B.prefix);
//               });
//     for (auto &k : allKeys) {
//         if (k.depth == MAX_L) continue;
//         auto &p = tree.at(OctreeKey{k.prefix, k.depth});
//         // uint64_t stride = 1ULL << (63 - 3*(k.depth+1));
//         for (int oc = 0; oc < 8; ++oc) {
//             uint64_t child = k.prefix
//                            | (uint64_t(oc) << (63 - 3*(k.depth+1)));
//             auto it = tree.find(OctreeKey{child, uint8_t(k.depth+1)});
//             if (it == tree.end()) continue;
//             const auto &c = it->second;
//             p.mass  += c.mass;
//             p.comX  += c.comX;
//             p.comY  += c.comY;
//             p.comZ  += c.comZ;
//         }
//     }

//     // 6) Finalize center-of-mass
//     for (auto &kv : tree) {
//         auto &n = kv.second;
//         if (n.mass > 0) {
//             n.comX /= n.mass;
//             n.comY /= n.mass;
//             n.comZ /= n.mass;
//         }
//     }

//     return tree;
// }
// OctreeMap buildOctree2(
//     const std::vector<uint64_t>& code,   // SORTED ascending Morton keys
//     const std::vector<Position>& pos,    // co-sorted physical positions
//     const std::vector<double>&   mass    // co-sorted masses
// ) {
//     const size_t N = code.size();
//     if (N == 0) return {};

//     constexpr uint8_t MAX_L = 21;

//     // 1) Branch scan via longest-common-prefix (LCP) of adjacent leaves
//     auto lcpBits = [&](int i,int j)->int {
//         if (i<0||j<0||i>=int(N)||j>=int(N)) return -1;
//         uint64_t x = code[i] ^ code[j];
//         return x ? (63 - __builtin_clzll(x)) : 63;
//     };
//     struct Key { uint64_t prefix; uint8_t depth; };
//     // pre-allocate exactly N-1 entries
//     std::vector<Key> branches;
//     if (N > 1) {
//         branches.resize(N - 1);
//         #pragma omp parallel for
//         for (int i = 0; i < int(N) - 1; ++i) {
//             int lp = lcpBits(i, i-1), ln = lcpBits(i, i+1);
//             int delta = std::max(lp, ln);
//             int lvl   = delta / 3;
//             int dir   = (ln > lp ? +1 : -1);
//             int lo = 1, hi = N - 1;
//             while (lo < hi) {
//                 int mid = (lo + hi) >> 1;
//                 if (lcpBits(i, i + dir*mid) > delta) lo = mid+1;
//                 else                                 hi = mid;
//             }
//             branches[i] = { mortonPrefix(code[i], lvl+1), static_cast<uint8_t>(lvl+1) };
//         }
//     }
//     // dedupe branches
//     oneapi::tbb::parallel_sort(branches.begin(), branches.end(),
//         [](auto &A, auto &B){
//             return A.depth < B.depth || (A.depth==B.depth && A.prefix < B.prefix);
//         });
//     branches.erase(std::unique(branches.begin(), branches.end(),
//         [](auto &A, auto &B){ return A.depth==B.depth && A.prefix==B.prefix; }),
//         branches.end());

//     // 2) Build full key list: root + branch-ancestors + leaf-ancestors
//     std::vector<Key> allKeys;
//     allKeys.reserve((branches.size()+N)*(MAX_L+1) + 1);
//     allKeys.push_back({0ULL,0});
//     for (auto &b : branches)
//         for (int d = b.depth; d >= 1; --d)
//             allKeys.push_back({ mortonPrefix(b.prefix,d), uint8_t(d) });
//     for (size_t i = 0; i < N; ++i) {
//         allKeys.push_back({ code[i], MAX_L });
//         for (int d = MAX_L-1; d >= 1; --d)
//             allKeys.push_back({ mortonPrefix(code[i],d), uint8_t(d) });
//     }
//     // dedupe allKeys
//     oneapi::tbb::parallel_sort(allKeys.begin(), allKeys.end(),
//         [](auto &A, auto &B){
//             return A.depth < B.depth || (A.depth==B.depth && A.prefix < B.prefix);
//         });
//     allKeys.erase(std::unique(allKeys.begin(), allKeys.end(),
//         [](auto &A, auto &B){ return A.depth==B.depth && A.prefix==B.prefix; }),
//         allKeys.end());

//     // 3) Populate map with empty nodes
//     OctreeMap tree;
//     tree.reserve(allKeys.size());
//     for (auto &k : allKeys)
//         tree[OctreeKey{k.prefix, k.depth}];

//     // 4) Accumulate leaf mass & weighted COM
//     #pragma omp parallel for
//     for (int i = 0; i < int(N); ++i) {
//         auto &n = tree.at(OctreeKey{code[i], MAX_L});
//         n.mass  += mass[i];
//         n.comX  += mass[i] * pos[i].x;
//         n.comY  += mass[i] * pos[i].y;
//         n.comZ  += mass[i] * pos[i].z;
//     }

//     // 5) Bottom-up reduce
//     oneapi::tbb::parallel_sort(allKeys.begin(), allKeys.end(),
//         [](auto &A, auto &B){
//             return A.depth > B.depth || (A.depth==B.depth && A.prefix > B.prefix);
//         });
//     for (auto &k : allKeys) {
//         if (k.depth == MAX_L) continue;
//         auto &p = tree.at(OctreeKey{k.prefix, k.depth});
//         for (int oc = 0; oc < 8; ++oc) {
//             uint64_t child = k.prefix | (uint64_t(oc) << (63 - 3*(k.depth+1)));
//             auto it = tree.find(OctreeKey{child, uint8_t(k.depth+1)});
//             if (it == tree.end()) continue;
//             const auto &c = it->second;
//             p.mass  += c.mass;
//             p.comX  += c.comX;
//             p.comY  += c.comY;
//             p.comZ  += c.comZ;
//         }
//     }

//     // 6) Finalize center-of-mass
//     for (auto &kv : tree) {
//         auto &n = kv.second;
//         if (n.mass > 0) {
//             n.comX /= n.mass;
//             n.comY /= n.mass;
//             n.comZ /= n.mass;
//         }
//     }
//     // std::vector<OctreeNode*> nodes;
//     // nodes.reserve(tree.size());
//     // for (auto &kv : tree)
//     //     nodes.push_back(&kv.second);
//     // #pragma omp parallel for
//     // for (int i = 0; i < int(nodes.size()); ++i) {
//     //     auto &n = *nodes[i];
//     //     if (n.mass > 0) {
//     //         n.comX /= n.mass;
//     //         n.comY /= n.mass;
//     //         n.comZ /= n.mass;
//     //     }
//     // }

//     return tree;
// }

std::vector<NodeRecord> flattenTree(const OctreeMap& tree) {
    std::vector<NodeRecord> out;
    out.reserve(tree.size());
    for (const auto& [key, node] : tree) {
        out.push_back({key.prefix, key.depth, node.mass, node.comX, node.comY, node.comZ});
    }
    return out;
}

void mergeIntoTree(OctreeMap& tree, const std::vector<NodeRecord>& records) {
    for (const auto& r : records) {
        OctreeKey key = {r.prefix, r.depth};
        OctreeNode& node = tree[key]; // Creates node if it doesn't exist or finds existing one

        // Store old mass and center of mass
        double m0 = node.mass;
        double comX0 = node.comX;
        double comY0 = node.comY;
        double comZ0 = node.comZ;

        // Get mass and CoM from the record to be merged
        double m1 = r.mass;
        double comX1 = r.comX;
        double comY1 = r.comY;
        double comZ1 = r.comZ;

        double total_mass = m0 + m1;
        if (total_mass == 0) continue;

        // Update center of mass by weighted average
        node.comX = (m0 * comX0 + m1 * comX1) / total_mass;
        node.comY = (m0 * comY0 + m1 * comY1) / total_mass;
        node.comZ = (m0 * comZ0 + m1 * comZ1) / total_mass;
        node.mass = total_mass;
    }
}


void printTree(const char* title, const OctreeMap& tree)
{
    // collect pointers so we can sort without copying OctreeNode
    std::vector<std::pair<OctreeKey,const OctreeNode*>> v;
    v.reserve(tree.size());
    for (const auto& kv : tree) v.push_back({kv.first,&kv.second});

    // depth primary, prefix secondary
    std::sort(v.begin(), v.end(),
              [](auto a, auto b)
              {
                  return (a.first.depth <  b.first.depth) ||
                         (a.first.depth == b.first.depth &&
                          a.first.prefix < b.first.prefix);
              });

    std::cout << '\n' << title << "\n";
    std::cout << "prefix(hex)          depth   mass       COM(x,y,z)\n";
    std::cout << "-----------------------------------------------------------\n";

    for (auto& e : v)
    {
        const OctreeKey&  k = e.first;
        const OctreeNode& n = *e.second;
        std::printf("0x%016llx  %3u   %-6.1f  (%.2e %.2e %.2e)\n",
                    static_cast<unsigned long long>(k.prefix),
                    k.depth,
                    n.mass,
                    n.comX, n.comY, n.comZ);
    }
}

/* ----------------------------------------------------------------
   buildOctreeLinear
   “Canonical” sparse hashed/octree builder (two-pass + ancestor fill)
   for Barnes–Hut on the CPU.
   ----------------------------------------------------------------*/
// OctreeMap buildOctree2(
//     const std::vector<uint64_t>& code,
//     const std::vector<Position>& pos,
//     const std::vector<double>&   mass)
// {
//     const size_t N = code.size();
//     if (N == 0) return {};

//     constexpr uint8_t MAX_L = 21;

//     // 1) Sort bodies by Morton key
//     std::vector<size_t> idx(N);
//     for (size_t i = 0; i < N; ++i) idx[i] = i;
//     std::sort(idx.begin(), idx.end(),
//               [&](size_t a, size_t b){ return code[a] < code[b]; });

//     // Build sorted arrays
//     std::vector<uint64_t> skey(N);
//     std::vector<Position> spos(N);
//     std::vector<double>   smass(N);
//     for (size_t i = 0; i < N; ++i) {
//         skey[i]  = code[idx[i]];
//         spos[i]  = pos[idx[i]];
//         smass[i] = mass[idx[i]];
//     }

//     // 2) Branch scan (LCP of adjacent leaves)
//     auto lcpBits = [&](int i,int j){
//         if (i<0||j<0||i>=int(N)||j>=int(N)) return -1;
//         uint64_t x = skey[i] ^ skey[j];
//         return x ? (63 - __builtin_clzll(x)) : 63;
//     };

//     struct Key { uint64_t prefix; uint8_t depth; };
//     std::vector<Key> branches;
//     branches.reserve(N-1);

//     for (int i = 0; i+1 < int(N); ++i) {
//         int lp = lcpBits(i, i-1);
//         int ln = lcpBits(i, i+1);
//         int delta = std::max(lp, ln);
//         int lvl = delta / 3;                // shared level
//         int dir = (ln > lp) ? +1 : -1;

//         int lo = 1, hi = N-1;
//         while (lo < hi) {
//             int mid = (lo + hi) >> 1;
//             if (lcpBits(i, i + dir*mid) > delta) lo = mid+1;
//             else hi = mid;
//         }
//         branches.push_back({
//             mortonPrefix(skey[i], lvl+1),
//             uint8_t(lvl+1)
//         });
//     }

//     // dedupe branches
//     std::sort(branches.begin(), branches.end(),
//               [](auto &A, auto &B){
//                   return A.depth < B.depth
//                       || (A.depth==B.depth && A.prefix < B.prefix);
//               });
//     branches.erase(std::unique(branches.begin(), branches.end(),
//                                [](auto &A, auto &B){
//                                   return A.depth==B.depth
//                                       && A.prefix==B.prefix;
//                                }),
//                    branches.end());

//     // 3) Build full key list:
//     //    – root
//     //    – every branch & *all* its ancestors
//     //    – every leaf & *all* its ancestors
//     std::vector<Key> allKeys;
//     allKeys.reserve((branches.size() + N) * (MAX_L + 1) + 1);

//     // root
//     allKeys.push_back({0ULL, 0});

//     // branches + their ancestors
//     for (auto &b : branches) {
//         for (int d = b.depth; d >= 1; --d) {
//             allKeys.push_back({
//                 mortonPrefix(b.prefix, d),
//                 uint8_t(d)
//             });
//         }
//     }

//     // leaves + their ancestors
//     for (size_t i = 0; i < N; ++i) {
//         // leaf
//         allKeys.push_back({ skey[i], MAX_L });
//         // ancestors
//         for (int d = MAX_L - 1; d >= 1; --d) {
//             allKeys.push_back({
//                 mortonPrefix(skey[i], d),
//                 uint8_t(d)
//             });
//         }
//     }

//     // dedupe the full set
//     std::sort(allKeys.begin(), allKeys.end(),
//               [](auto &A, auto &B){
//                   return A.depth < B.depth
//                       || (A.depth==B.depth && A.prefix < B.prefix);
//               });
//     allKeys.erase(std::unique(allKeys.begin(), allKeys.end(),
//                                [](auto &A, auto &B){
//                                   return A.depth==B.depth
//                                       && A.prefix==B.prefix;
//                                }),
//                    allKeys.end());

//     // 4) Populate map and accumulate leaf sums
//     OctreeMap tree;
//     tree.reserve(allKeys.size());
//     for (auto &k : allKeys)
//         tree[{k.prefix, k.depth}];    // ensure key exists

//     for (size_t i = 0; i < N; ++i) {
//         auto &n = tree[{skey[i], MAX_L}];
//         n.mass  += smass[i];
//         n.comX  += smass[i]*spos[i].x;
//         n.comY  += smass[i]*spos[i].y;
//         n.comZ  += smass[i]*spos[i].z;
//     }

//     // 5) Bottom-up reduce
//     std::sort(allKeys.begin(), allKeys.end(),
//               [](auto &A, auto &B){
//                   return A.depth > B.depth
//                       || (A.depth==B.depth && A.prefix > B.prefix);
//               });

//     for (auto &k : allKeys) {
//         if (k.depth == MAX_L) continue;
//         auto &p = tree[{k.prefix, k.depth}];
//         uint64_t stride = 1ULL << (63 - 3*(k.depth+1));
//         for (int oct = 0; oct < 8; ++oct) {
//             auto it = tree.find({
//                 k.prefix + stride*oct,
//                 uint8_t(k.depth+1)
//             });
//             if (it == tree.end()) continue;
//             auto &c = it->second;
//             p.mass  += c.mass;
//             p.comX  += c.comX;
//             p.comY  += c.comY;
//             p.comZ  += c.comZ;
//         }
//     }

//     // 6) Finalize center-of-mass
//     for (auto &kv : tree) {
//         auto &n = kv.second;
//         if (n.mass > 0) {
//             n.comX /= n.mass;
//             n.comY /= n.mass;
//             n.comZ /= n.mass;
//         }
//     }

//     return tree;
// }

// OctreeMap buildOctree2(
//     std::vector<uint64_t>& code,      // WILL be permuted in-place
//     std::vector<Position>& pos,       // WILL be permuted in-place
//     std::vector<double>&   mass       // WILL be permuted in-place
// ) {
//     const size_t N = code.size();
//     if (N == 0) return {};

//     constexpr uint8_t MAX_L = 21;

//     // 1) In-place co-sort of (code, pos, mass) by code
//     {
//         std::vector<size_t> idx(N);
//         std::iota(idx.begin(), idx.end(), 0);
//         std::sort(idx.begin(), idx.end(),
//                   [&](size_t a, size_t b){ return code[a] < code[b]; });
//         auto permute = [&](auto& vec){
//             using T = std::decay_t<decltype(vec[0])>;
//             std::vector<T> tmp(N);
//             for (size_t i = 0; i < N; ++i) tmp[i] = vec[idx[i]];
//             vec.swap(tmp);
//         };
//         permute(code);
//         permute(pos);
//         permute(mass);
//     }

//     // 2) Branch scan via longest-common-prefix (LCP) of adjacent leaves
//     auto lcpBits = [&](int i,int j)->int {
//         if (i<0||j<0||i>=int(N)||j>=int(N)) return -1;
//         uint64_t x = code[i] ^ code[j];
//         return x ? (63 - __builtin_clzll(x)) : 63;
//     };
//     struct Key { uint64_t prefix; uint8_t depth; };
//     std::vector<Key> branches;
//     branches.reserve(N>0? N-1 : 0);

//     for (int i = 0; i+1 < int(N); ++i) {
//         int lp = lcpBits(i, i-1), ln = lcpBits(i, i+1);
//         int delta = std::max(lp, ln);
//         int lvl   = delta / 3;
//         int dir   = (ln > lp ? +1 : -1);
//         int lo = 1, hi = N-1;
//         while (lo < hi) {
//             int mid = (lo + hi) >> 1;
//             if (lcpBits(i, i + dir*mid) > delta) lo = mid+1;
//             else                                 hi = mid;
//         }
//         branches.push_back({ mortonPrefix(code[i], lvl+1),
//                              static_cast<uint8_t>(lvl+1) });
//     }
//     // dedupe branches
//     std::sort(branches.begin(), branches.end(),
//               [](auto &A, auto &B){
//                   return A.depth < B.depth
//                       || (A.depth==B.depth && A.prefix < B.prefix);
//               });
//     branches.erase(std::unique(branches.begin(), branches.end(),
//                                [](auto &A, auto &B){
//                                    return A.depth==B.depth
//                                        && A.prefix==B.prefix;
//                                }),
//                    branches.end());

//     // 3) Build full key list: root + branch-ancestors + leaf-ancestors
//     std::vector<Key> allKeys;
//     allKeys.reserve((branches.size()+N)*(MAX_L+1) + 1);
//     allKeys.push_back({0ULL,0});  // root
//     for (auto &b : branches)
//         for (int d = b.depth; d >= 1; --d)
//             allKeys.push_back({ mortonPrefix(b.prefix,d), uint8_t(d) });
//     for (size_t i = 0; i < N; ++i) {
//         allKeys.push_back({ code[i], MAX_L });
//         for (int d = MAX_L-1; d >= 1; --d)
//             allKeys.push_back({ mortonPrefix(code[i],d), uint8_t(d) });
//     }
//     // dedupe allKeys
//     std::sort(allKeys.begin(), allKeys.end(),
//               [](auto &A, auto &B){
//                   return A.depth < B.depth
//                       || (A.depth==B.depth && A.prefix < B.prefix);
//               });
//     allKeys.erase(std::unique(allKeys.begin(), allKeys.end(),
//                               [](auto &A, auto &B){
//                                   return A.depth==B.depth
//                                       && A.prefix==B.prefix;
//                               }),
//                   allKeys.end());

//     // 4) Populate map with empty nodes
//     OctreeMap tree;
//     tree.reserve(allKeys.size());
//     for (auto &k : allKeys)
//         tree[OctreeKey{k.prefix, k.depth}];

//     // 5) Accumulate leaf mass & weighted COM
//     for (size_t i = 0; i < N; ++i) {
//         auto &n = tree.at(OctreeKey{code[i], MAX_L});
//         n.mass  += mass[i];
//         n.comX  += mass[i] * pos[i].x;
//         n.comY  += mass[i] * pos[i].y;
//         n.comZ  += mass[i] * pos[i].z;
//     }

//     // 6) Bottom-up reduce (add children’s mass & mass×pos sums)
//     std::sort(allKeys.begin(), allKeys.end(),
//               [](auto &A, auto &B){
//                   return A.depth > B.depth
//                       || (A.depth==B.depth && A.prefix > B.prefix);
//               });
//     for (auto &k : allKeys) {
//         if (k.depth == MAX_L) continue;
//         auto &p = tree.at(OctreeKey{k.prefix, k.depth});
//         uint64_t stride = 1ULL << (63 - 3*(k.depth+1));
//         for (int oc = 0; oc < 8; ++oc) {
//             uint64_t child = k.prefix
//                            | (uint64_t(oc) << (63 - 3*(k.depth+1)));
//             auto it = tree.find(OctreeKey{child, uint8_t(k.depth+1)});
//             if (it == tree.end()) continue;
//             const auto &c = it->second;
//             p.mass  += c.mass;
//             p.comX  += c.comX;
//             p.comY  += c.comY;
//             p.comZ  += c.comZ;
//         }
//     }

//     // 7) Finalize center-of-mass
//     for (auto &kv : tree) {
//         auto &n = kv.second;
//         if (n.mass > 0) {
//             n.comX /= n.mass;
//             n.comY /= n.mass;
//             n.comZ /= n.mass;
//         }
//     }
//     return tree;
// }

