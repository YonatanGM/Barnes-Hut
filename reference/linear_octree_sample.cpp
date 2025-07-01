// linear_octree.cpp
//
// Morton-based octree helpers for a Barnes–Hut solar-system code.
// g++ -std=c++17 -O3 linear_octree.cpp -o linear_octree


#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <iomanip>

struct Position
{
    double x, y, z;
};

struct OctreeKey
{
    uint64_t prefix;           // top 3·depth bits
    uint8_t  depth;            // 0..21

    bool operator==(const OctreeKey& k) const noexcept
    { return prefix == k.prefix && depth == k.depth; }
};

struct OctreeKeyHash
{
    size_t operator()(OctreeKey k) const noexcept
    { return std::hash<uint64_t>()(k.prefix ^ (uint64_t(k.depth) << 56)); }
};

struct OctreeNode         // what Barnes–Hut needs
{
    double mass {0};      // total mass in this cube
    double comX {0};      // centre-of-mass coordinates
    double comY {0};
    double comZ {0};
};

using OctreeMap = std::unordered_map<OctreeKey,OctreeNode,OctreeKeyHash>;

/* ----------------------------------------------------------------
   bit helpers – 63-bit Morton code (21 bits per axis)
   ----------------------------------------------------------------*/
static inline uint64_t spread21(uint32_t v)
{
    v &= 0x1FFFFF;
    uint64_t x = v;
    x = (x | (x << 32)) & 0x1F00000000FFFFULL;
    x = (x | (x << 16)) & 0x1F0000FF0000FFULL;
    x = (x | (x <<  8)) & 0x100F00F00F00F00FULL;
    x = (x | (x <<  4)) & 0x10C30C30C30C30C3ULL;
    x = (x | (x <<  2)) & 0x1249249249249249ULL;
    return x;
}
static inline uint64_t morton63(uint32_t xi,uint32_t yi,uint32_t zi)
{
    return spread21(xi) | (spread21(yi)<<1) | (spread21(zi)<<2);
}

/* ----------------------------------------------------------------
   Step 1: bounding-box normalisation → unit cube coords
   ----------------------------------------------------------------*/
std::vector<Position> normalizePositions(const std::vector<Position>& pos)
{
    const std::size_t n = pos.size();
    std::vector<Position> normed(n);
    if (n == 0) return normed;

    // Compute bounds
    double xmin = +INFINITY, ymin = +INFINITY, zmin = +INFINITY;
    double xmax = -INFINITY, ymax = -INFINITY, zmax = -INFINITY;
    for (auto& p : pos) {
        xmin = std::min(xmin, p.x); xmax = std::max(xmax, p.x);
        ymin = std::min(ymin, p.y); ymax = std::max(ymax, p.y);
        zmin = std::min(zmin, p.z); zmax = std::max(zmax, p.z);
    }

    // Avoid zero-span
    double dx = (xmax == xmin) ? 1.0 : (xmax - xmin);
    double dy = (ymax == ymin) ? 1.0 : (ymax - ymin);
    double dz = (zmax == zmin) ? 1.0 : (zmax - zmin);

    auto clamp01 = [&](double v, double lo, double span) {
        double t = (v - lo) / span;
        if (t <= 0.0) return 0.0;
        if (t >= 1.0) return std::nextafter(1.0, 0.0);
        return t;
    };

    // Fill normalized positions
    for (std::size_t i = 0; i < n; ++i) {
        normed[i].x = clamp01(pos[i].x, xmin, dx);
        normed[i].y = clamp01(pos[i].y, ymin, dy);
        normed[i].z = clamp01(pos[i].z, zmin, dz);
    }

    return normed;
}

/* ------------------------------------------------------------------ */
/* pull the 21   x–, y–, z–bits back out of a 63-bit Morton word      */
/* ------------------------------------------------------------------ */
static inline uint32_t compact21(uint64_t x)          // undo spread21()
{
    x &= 0x1249249249249249ULL;
    x = (x ^ (x >>  2)) & 0x10c30c30c30c30c3ULL;
    x = (x ^ (x >>  4)) & 0x100f00f00f00f00fULL;
    x = (x ^ (x >>  8)) & 0x1f0000ff0000ffULL;
    x = (x ^ (x >> 16)) & 0x1f00000000ffffULL;
    x = (x ^ (x >> 32)) & 0x1fffff;
    return uint32_t(x);
}

/* ------------------------------------------------------------------ */
/* de-interleave the *top 3·depth bits* of the prefix into (ix,iy,iz)  */
/* ------------------------------------------------------------------ */
static inline void decodePrefix(uint64_t prefix, int depth,
                                uint32_t &ix, uint32_t &iy, uint32_t &iz)
{
    if (depth == 0) { ix = iy = iz = 0; return; }

    /* Move the relevant 3·d bits down to the LSBs … */
    uint64_t bits = prefix >> (63 - 3*depth);           // keep 3·d bits
    /* … then un-shuffle */
    ix = compact21(bits);
    iy = compact21(bits >> 1);
    iz = compact21(bits >> 2);
}

void exportForPlot(const std::vector<Position>& bodies,
                   const OctreeMap&            tree,
                   const std::string&          bodyFile,
                   const std::string&          nodeFile)
{
    /* ---- 1) bodies.csv -------------------------------------------------- */
    std::ofstream bo(bodyFile);
    bo << "x,y,z\n";
    bo << std::fixed << std::setprecision(6);
    for (auto& b : bodies)
        bo << b.x << ',' << b.y << ',' << b.z << '\n';
    bo.close();

    /* ---- 2) nodes.csv --------------------------------------------------- */
    std::ofstream no(nodeFile);
    no << "prefix,depth,xmin,xmax,ymin,ymax,zmin,zmax\n";
    no << std::hex << std::showbase;

    for (auto& kv : tree)
    {
        uint64_t prefix = kv.first.prefix;
        int      d      = kv.first.depth;

        /* recover integer cell coordinates */
        uint32_t ix, iy, iz;
        decodePrefix(prefix, d, ix, iy, iz);

        double size = 1.0 / (1u << d);
        double xmin = ix * size, xmax = xmin + size;
        double ymin = iy * size, ymax = ymin + size;
        double zmin = iz * size, zmax = zmin + size;

        no << prefix << ','
           << std::dec << d << ','
           << xmin << ',' << xmax << ','
           << ymin << ',' << ymax << ','
           << zmin << ',' << zmax << '\n'
           << std::hex;                    // back to hex for next prefix
    }
    no.close();
}



/* ----------------------------------------------------------------
   bounding-box normalisation → Morton codes
   ----------------------------------------------------------------*/
std::vector<uint64_t> mortonCodes(const std::vector<Position>& pos)
{
    const std::size_t n = pos.size();
    std::vector<uint64_t> code(n);
    if (!n) return code;

    double xmin=+INFINITY,ymin=+INFINITY,zmin=+INFINITY;
    double xmax=-INFINITY,ymax=-INFINITY,zmax=-INFINITY;
    for (auto [x,y,z] : pos) {
        xmin = std::min(xmin,x); xmax = std::max(xmax,x);
        ymin = std::min(ymin,y); ymax = std::max(ymax,y);
        zmin = std::min(zmin,z); zmax = std::max(zmax,z);
    }
    double dx = (xmax==xmin)?1:xmax-xmin;
    double dy = (ymax==ymin)?1:ymax-ymin;
    double dz = (zmax==zmin)?1:zmax-zmin;
    constexpr uint32_t Q = (1u<<21) - 1;

    auto norm=[&](double v,double lo,double span){
        double t=(v-lo)/span;
        if (t<=0) return 0.0;
        if (t>=1) return std::nextafter(1.0,0.0);
        return t;
    };

    for (std::size_t i=0;i<n;++i) {
        uint32_t xi=uint32_t(norm(pos[i].x,xmin,dx)*Q);
        uint32_t yi=uint32_t(norm(pos[i].y,ymin,dy)*Q);
        uint32_t zi=uint32_t(norm(pos[i].z,zmin,dz)*Q);
        code[i]=morton63(xi,yi,zi);
    }
    return code;
}
static inline uint64_t mortonPrefix(uint64_t code,int depth)
{
    if (depth==0) return 0;
    int shift = 63 - 3*depth;
    return code & (~0ULL << shift);
}

/* ==============================================================
   buildOctreePerBody
   --------------------------------------------------------------
   Insert one body at a time, walking up to root and merging its
   mass into every ancestor.  Keeps the full 21-level hierarchy.
   ==============================================================*/
OctreeMap buildOctree1(const std::vector<uint64_t>& code,
                             const std::vector<Position>& pos,
                             const std::vector<double>&   mass)
{
    const std::size_t n = code.size();
    std::vector<std::size_t> order(n);
    for (std::size_t i = 0; i < n; ++i) order[i] = i;

    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b){ return code[a] < code[b]; });

    OctreeMap tree; tree.reserve(n * 2);
    constexpr int MAX = 21;

    auto merge = [](OctreeNode& node, double m, const Position& p)
    {
        double oldM = node.mass;
        double newM = oldM + m;
        node.comX = (oldM*node.comX + m*p.x) / newM;
        node.comY = (oldM*node.comY + m*p.y) / newM;
        node.comZ = (oldM*node.comZ + m*p.z) / newM;
        node.mass = newM;
    };

    for (std::size_t id : order)
    {
        double m = mass[id];

        /* leaf ------------------------------------------------ */
        OctreeKey key{code[id], static_cast<uint8_t>(MAX)};
        merge(tree[key], m, pos[id]);

        /* ancestors ------------------------------------------- */
        uint64_t pref = code[id];
        for (int d = MAX - 1; d >= 0; --d)
        {
            pref = mortonPrefix(pref, d);           // zero-out lower bits
            OctreeKey k{pref, static_cast<uint8_t>(d)};
            merge(tree[k], m, pos[id]);
        }
    }
    return tree;

}

/* ----------------------------------------------------------------
   buildOctreeLinear
   “Canonical” sparse hashed/octree builder (two-pass + ancestor fill)
   for Barnes–Hut on the CPU.
   ----------------------------------------------------------------*/
OctreeMap buildOctree2(
    const std::vector<uint64_t>& code,
    const std::vector<Position>& pos,
    const std::vector<double>&   mass)
{
    const size_t N = code.size();
    if (N == 0) return {};

    constexpr uint8_t MAX_L = 21;

    // 1) Sort bodies by Morton key
    std::vector<size_t> idx(N);
    for (size_t i = 0; i < N; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(),
              [&](size_t a, size_t b){ return code[a] < code[b]; });

    // Build sorted arrays
    std::vector<uint64_t> skey(N);
    std::vector<Position> spos(N);
    std::vector<double>   smass(N);
    for (size_t i = 0; i < N; ++i) {
        skey[i]  = code[idx[i]];
        spos[i]  = pos[idx[i]];
        smass[i] = mass[idx[i]];
    }

    // 2) Branch scan (LCP of adjacent leaves)
    auto lcpBits = [&](int i,int j){
        if (i<0||j<0||i>=int(N)||j>=int(N)) return -1;
        uint64_t x = skey[i] ^ skey[j];
        return x ? (63 - __builtin_clzll(x)) : 63;
    };

    struct Key { uint64_t prefix; uint8_t depth; };
    std::vector<Key> branches;
    branches.reserve(N-1);

    for (int i = 0; i+1 < int(N); ++i) {
        int lp = lcpBits(i, i-1);
        int ln = lcpBits(i, i+1);
        int delta = std::max(lp, ln);
        int lvl = delta / 3;                // shared level
        int dir = (ln > lp) ? +1 : -1;

        int lo = 1, hi = N-1;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (lcpBits(i, i + dir*mid) > delta) lo = mid+1;
            else hi = mid;
        }
        branches.push_back({
            mortonPrefix(skey[i], lvl+1),
            uint8_t(lvl+1)
        });
    }

    // dedupe branches
    std::sort(branches.begin(), branches.end(),
              [](auto &A, auto &B){
                  return A.depth < B.depth
                      || (A.depth==B.depth && A.prefix < B.prefix);
              });
    branches.erase(std::unique(branches.begin(), branches.end(),
                               [](auto &A, auto &B){
                                  return A.depth==B.depth
                                      && A.prefix==B.prefix;
                               }),
                   branches.end());

    // 3) Build full key list:
    //    – root
    //    – every branch & *all* its ancestors
    //    – every leaf & *all* its ancestors
    std::vector<Key> allKeys;
    allKeys.reserve((branches.size() + N) * (MAX_L + 1) + 1);

    // root
    allKeys.push_back({0ULL, 0});

    // branches + their ancestors
    for (auto &b : branches) {
        for (int d = b.depth; d >= 1; --d) {
            allKeys.push_back({
                mortonPrefix(b.prefix, d),
                uint8_t(d)
            });
        }
    }

    // leaves + their ancestors
    for (size_t i = 0; i < N; ++i) {
        // leaf
        allKeys.push_back({ skey[i], MAX_L });
        // ancestors
        for (int d = MAX_L - 1; d >= 1; --d) {
            allKeys.push_back({
                mortonPrefix(skey[i], d),
                uint8_t(d)
            });
        }
    }

    // dedupe the full set
    std::sort(allKeys.begin(), allKeys.end(),
              [](auto &A, auto &B){
                  return A.depth < B.depth
                      || (A.depth==B.depth && A.prefix < B.prefix);
              });
    allKeys.erase(std::unique(allKeys.begin(), allKeys.end(),
                               [](auto &A, auto &B){
                                  return A.depth==B.depth
                                      && A.prefix==B.prefix;
                               }),
                   allKeys.end());

    // 4) Populate map and accumulate leaf sums
    OctreeMap tree;
    tree.reserve(allKeys.size());
    for (auto &k : allKeys)
        tree[{k.prefix, k.depth}];    // ensure key exists

    for (size_t i = 0; i < N; ++i) {
        auto &n = tree[{skey[i], MAX_L}];
        n.mass  += smass[i];
        n.comX  += smass[i]*spos[i].x;
        n.comY  += smass[i]*spos[i].y;
        n.comZ  += smass[i]*spos[i].z;
    }

    // 5) Bottom-up reduce
    std::sort(allKeys.begin(), allKeys.end(),
              [](auto &A, auto &B){
                  return A.depth > B.depth
                      || (A.depth==B.depth && A.prefix > B.prefix);
              });

    for (auto &k : allKeys) {
        if (k.depth == MAX_L) continue;
        auto &p = tree[{k.prefix, k.depth}];
        uint64_t stride = 1ULL << (63 - 3*(k.depth+1));
        for (int oct = 0; oct < 8; ++oct) {
            auto it = tree.find({
                k.prefix + stride*oct,
                uint8_t(k.depth+1)
            });
            if (it == tree.end()) continue;
            auto &c = it->second;
            p.mass  += c.mass;
            p.comX  += c.comX;
            p.comY  += c.comY;
            p.comZ  += c.comZ;
        }
    }

    // 6) Finalize center-of-mass
    for (auto &kv : tree) {
        auto &n = kv.second;
        if (n.mass > 0) {
            n.comX /= n.mass;
            n.comY /= n.mass;
            n.comZ /= n.mass;
        }
    }

    return tree;
}



/* ----------------------------------------------------------------
   flatten – turn tree into a vector you can send over MPI
   ----------------------------------------------------------------*/
struct NodeRecord
{
    uint64_t prefix;
    uint8_t  depth;
    double   mass;
    double   comX;
    double   comY;
    double   comZ;
};

std::vector<NodeRecord> flatten(const OctreeMap& tree)
{
    std::vector<NodeRecord> out;
    out.reserve(tree.size());
    for (const auto& [k,n] : tree)
        out.push_back({k.prefix,k.depth,n.mass,n.comX,n.comY,n.comZ});
    return out;
}

/* ----------------------------------------------------------------
   merge – add records into an existing tree
   ----------------------------------------------------------------*/
void mergeIntoTree(const std::vector<NodeRecord>& rec, OctreeMap& tree)
{
    for (const auto& r : rec)
    {
        OctreeKey key{r.prefix,r.depth};
        OctreeNode& node = tree[key];

        double m0 = node.mass;
        double m1 = r.mass;
        double mt = m0 + m1;
        if (mt == 0) continue;

        node.comX = (m0*node.comX + m1*r.comX) / mt;
        node.comY = (m0*node.comY + m1*r.comY) / mt;
        node.comZ = (m0*node.comZ + m1*r.comZ) / mt;
        node.mass = mt;
    }
}


/* ------------------------------------------------------------
   printWholeTree  –  list every cube in depth-then-prefix order
   ------------------------------------------------------------ */
void printWholeTree(const char* title, const OctreeMap& tree)
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



// Compare two flattened octree vectors
void compareFlattened(const std::vector<NodeRecord>& A,
                      const std::vector<NodeRecord>& B,
                      double tol = 1e-9)
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


/* ----------------------------------------------------------------
   mortonCodesAndNorm  –  EXACTLY the old mortonCodes() algorithm,
   but also gives you the [0,1]³ positions it computed internally.
   ----------------------------------------------------------------*/
struct CodesAndNorm {
    std::vector<uint64_t> code;   // Morton 63-bit keys
    std::vector<Position> norm;   // normalised positions (1-ε max)
};

CodesAndNorm mortonCodesAndNorm(const std::vector<Position>& raw)
{
    const std::size_t n = raw.size();
    CodesAndNorm out{ std::vector<uint64_t>(n), std::vector<Position>(n) };
    if (!n) return out;

    /* ---------- identical bounding-box scan ---------- */
    double xmin=+INFINITY,ymin=+INFINITY,zmin=+INFINITY;
    double xmax=-INFINITY,ymax=-INFINITY,zmax=-INFINITY;
    for (auto [x,y,z] : raw) {
        xmin = std::min(xmin,x); xmax = std::max(xmax,x);
        ymin = std::min(ymin,y); ymax = std::max(ymax,y);
        zmin = std::min(zmin,z); zmax = std::max(zmax,z);
    }
    double dx = (xmax==xmin)?1:xmax-xmin;
    double dy = (ymax==ymin)?1:ymax-ymin;
    double dz = (zmax==zmin)?1:zmax-zmin;
    constexpr uint32_t Q = (1u<<21) - 1;

    auto norm = [&](double v,double lo,double span){
        double t=(v-lo)/span;
        if (t<=0) return 0.0;
        if (t>=1) return std::nextafter(1.0,0.0);
        return t;
    };

    /* ---------- loop: compute norm + Morton key ---------- */
    for (std::size_t i=0;i<n;++i) {
        double xn = norm(raw[i].x,xmin,dx);
        double yn = norm(raw[i].y,ymin,dy);
        double zn = norm(raw[i].z,zmin,dz);

        out.norm[i] = {xn,yn,zn};

        uint32_t xi=uint32_t(xn*Q);
        uint32_t yi=uint32_t(yn*Q);
        uint32_t zi=uint32_t(zn*Q);
        out.code[i]=morton63(xi,yi,zi);
    }
    return out;
}


int main()
{
    /* ---------- demo body set ---------- */
    std::vector<Position> bodies =
    {
        {1e7,2e7,3e7},{5e7,1e7,4e7},{2e7,8e7,9e7}, {2e7,9e7,9e7},
        {9e6,4e7,7e7},{6e7,3e7,2e7},{7e7,9e7,5e7}
    };
    std::vector<double> masses = {1,2,3,4,5,6};

    std::vector<uint64_t> codes = mortonCodes(bodies);

    auto nm   = mortonCodesAndNorm(bodies);
    auto &pos = nm.norm;         // already [0,1]³ exactly once
    auto &key = nm.code;        // 63-bit Morton keys

    /* ---------- build with the two approaches ---------- */
    OctreeMap perBodyTree = buildOctree1(key, pos, masses);
    OctreeMap scanTree    = buildOctree2(key, pos, masses);


    printWholeTree("Per-body tree (complete)",  perBodyTree);
    printWholeTree("Single-scan tree (complete)", scanTree);

    auto flatA = flatten(perBodyTree);
    auto flatB = flatten(scanTree);

    compareFlattened(flatA, flatB);

    exportForPlot(pos, perBodyTree, "bodies.csv", "nodes.csv");

    /* ---------- merge test ---------- */
    // std::vector<Position> bodiesA(bodies.begin(), bodies.begin()+3);
    // std::vector<Position> bodiesB(bodies.begin()+3, bodies.end());
    // std::vector<double>   massesA(masses.begin(), masses.begin()+3);
    // std::vector<double>   massesB(masses.begin()+3, masses.end());

    // OctreeMap treeA = buildOctree1(mortonCodes(bodiesA), bodiesA, massesA);
    // OctreeMap treeB = buildOctree1(mortonCodes(bodiesB), bodiesB, massesB);

    // OctreeMap merged;
    // mergeIntoTree(flatten(treeA), merged);
    // mergeIntoTree(flatten(treeB), merged);

    // printWholeTree("Merged tree (A+B)", merged);

    /* ---------- quick cross-checks ---------- */
    // const OctreeKey root{0,0};
    // std::cout << "\nroot mass   per-body "   << perBodyTree[root].mass
    //           << "   single-scan "           << scanTree[root].mass
    //           << "   merged "                << merged[root].mass << '\n';


    // bodies = normalizePositions(bodies);
    // writeOctreeSVG(bodies, flatA, "octree.svg");
}

