#include "linear_octree.h"
#include <algorithm>
#include <numeric>
#include <vector>
#include <iostream>
#include <oneapi/tbb/parallel_sort.h>
#include <omp.h>
#include <morton_keys.h>
#include <chrono>



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

