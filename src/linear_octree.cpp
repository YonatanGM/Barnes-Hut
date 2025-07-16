#include "linear_octree.h"
#include <algorithm>
#include <numeric>
#include <vector>
#include <iostream>
#include <omp.h>
#include <morton_keys.h>
#include <chrono>

constexpr uint8_t BH_MAX_DEPTH = 21;


OctreeMap buildOctreeBottomUp(const std::vector<uint64_t>& generateMortonCodes,
                               const std::vector<Position>& positions,
                               const std::vector<double>&   masses)
{
    const size_t N = generateMortonCodes.size();
    if (N == 0) return {};

    constexpr uint8_t MAX_DEPTH = 21;
    OctreeMap tree;
    tree.reserve(N * 2);                // heuristic

    // 1. Insert / accumulate leaves and record their keys
    std::vector<OctreeKey> currentKeys;
    currentKeys.reserve(N);

    for (size_t i = 0; i < N; ++i)
    {
        OctreeKey leaf{ generateMortonCodes[i], MAX_DEPTH };
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


    // 2. Bottom up, generate parents only from keys we have
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

        // 2b. Aggregate child into parent
        for (const auto& pk : parentKeys)
        {
            OctreeNode& p = tree[pk];    // create or find parent

            for (int oc = 0; oc < 8; ++oc)
            {
                uint64_t childPrefix = pk.prefix |
                    (uint64_t(oc) << (63 - 3*(depth+1)));
                auto it = tree.find({ childPrefix,
                                      static_cast<uint8_t>(depth+1) });
                if (it == tree.end()) continue;

                const OctreeNode& c = it->second;
                p.mass  += c.mass;
                p.comX  += c.comX;
                p.comY  += c.comY;
                p.comZ  += c.comZ;
            }
        }

        // 2c. Parents become next iteration’s children
        currentKeys.swap(parentKeys);
    }


    // 3. Finalise COM for every node
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


// Compare two flattened octree vectors
void compareFlattened(const std::vector<NodeRecord> A,
                      const std::vector<NodeRecord> B,
                      double tol)
{
    // sort by (depth,prefix)
    auto cmp = [](auto const &a, auto const &b) {
        if (a.depth != b.depth) return a.depth < b.depth;
        return a.prefix < b.prefix;
    };
    std::vector<NodeRecord> vA = A, vB = B;
    std::sort(vA.begin(), vA.end(), cmp);
    std::sort(vB.begin(), vB.end(), cmp);

    // walk them in lock-step
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

    // check for size mismatches
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



std::vector<NodeRecord> serializeTreeToRecords(const OctreeMap& tree) {
    std::vector<NodeRecord> out;
    out.reserve(tree.size());
    for (const auto& [key, node] : tree) {
        out.push_back({key.prefix, key.depth, node.mass, node.comX, node.comY, node.comZ});
    }
    return out;
}

void mergeRecordsIntoTree(OctreeMap& tree, const std::vector<NodeRecord>& records) {
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




OctreeMap buildTreeFromPseudoLeaves(const std::vector<NodeRecord>& remote_nodes)
{
    OctreeMap tree;
    if (remote_nodes.empty()) {
        return tree;
    }

    tree.reserve(remote_nodes.size() * 4); // conservative reservation

    // Insert and accumulate all remote nodes
    // First, insert all nodes from the input list. This correctly handles the
    // case where multiple ranks send a node with the same key, by summing
    // their mass and mass-weighted CoM contributions
    for (const auto& r : remote_nodes) {
        double m = r.mass;
        if (!(m > 0.0) || !std::isfinite(m)) continue;

        uint8_t d = r.depth > BH_MAX_DEPTH ? BH_MAX_DEPTH : r.depth;
        OctreeKey k{mortonPrefix(r.prefix, d), d};

        OctreeNode& n = tree[k];
        n.mass += m;
        n.comX += r.comX * m;
        n.comY += r.comY * m;
        n.comZ += r.comZ * m;
        n.is_pseudo = true;
    }

    // Create all ancestor nodes
    // Now, for every node we just inserted, we ensure its parents exist by
    // walking up to the root. We only need to create the keys; the mass
    // aggregation will happen in the next step.
    OctreeMap temp_ancestors;
    for (const auto& kv : tree) {
        const OctreeKey& k = kv.first;
        for (int d = k.depth - 1; d >= 0; --d) {
            OctreeKey parent_key{mortonPrefix(k.prefix, d), (uint8_t)d};
            // The [] operator creates the node if it doesn't exist.
            temp_ancestors[parent_key];
        }
    }
    tree.insert(temp_ancestors.begin(), temp_ancestors.end());


    // Aggregate mass bottom up
    // Now that all nodes (leaves and ancestors) exist in the tree, we can
    // iterate from the deepest level upwards and sum the children's contributions
    // into their parents.
    for (int d = BH_MAX_DEPTH; d > 0; --d) {
        for (const auto& kv : tree) {
            const OctreeKey& child_key = kv.first;
            if (child_key.depth != d) continue;

            // Find the parent and add this child's contribution.
            OctreeKey parent_key = {mortonPrefix(child_key.prefix, d - 1), (uint8_t)(d - 1)};
            auto it = tree.find(parent_key);
            if (it != tree.end()) {
                OctreeNode& parent_node = it->second;
                const OctreeNode& child_node = kv.second;

                parent_node.mass += child_node.mass;
                parent_node.comX += child_node.comX; // Already mass-weighted
                parent_node.comY += child_node.comY;
                parent_node.comZ += child_node.comZ;
            }
        }
    }


    // After all aggregation is complete, divide by mass to get the final COM
    for (auto& kv : tree) {
        OctreeNode& n = kv.second;
        if (n.mass > 1e-12) {
            n.comX /= n.mass;
            n.comY /= n.mass;
            n.comZ /= n.mass;
        }
    }

    return tree;
}