// morton_order.cpp
//
// Compute 64-bit Morton (Z-order) codes for a list of 3D points
// with double-precision coordinates in an arbitrary bounding box.
// Uses 21 bits per axis (total 63 bits), packed into a uint64_t.
//
// Compile with: g++ -std=c++11 morton_order.cpp -o morton_order
//

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <cmath>

// A simple struct to hold a 3D point in double precision.
struct Point3D {
    double x, y, z;
};

// Spread (aka “part1by2”) a 21-bit integer so that between each input bit
// there are two zero bits in the output. The result uses up to 63 bits.
//
// This bit‐magic sequence is designed to take the lower 21 bits of 'x'
// and place them in bits 0,3,6,...,60 of the resulting 64-bit integer.
static inline uint64_t part1by2_21(uint32_t x) {
    x &= 0x1FFFFF; // keep only lowest 21 bits
    uint64_t x64 = x;                       // promote to 64-bit
    x64 = (x64 | (x64 << 32)) & 0x1F00000000FFFFULL;
    x64 = (x64 | (x64 << 16)) & 0x1F0000FF0000FFULL;
    x64 = (x64 | (x64 <<  8)) & 0x100F00F00F00F00FULL;
    x64 = (x64 | (x64 <<  4)) & 0x10C30C30C30C30C3ULL;
    x64 = (x64 | (x64 <<  2)) & 0x1249249249249249ULL;
    return x64;
}

// Given three 21-bit integers (xi, yi, zi), interleave their bits to form
// a 64-bit Morton code. Bits of xi go to positions 0,3,6,...; bits of yi
// go to positions 1,4,7,...; bits of zi go to positions 2,5,8,...
static inline uint64_t morton3D_21(uint32_t xi, uint32_t yi, uint32_t zi) {
    uint64_t x_part = part1by2_21(xi);         // occupies bits 0,3,6,...
    uint64_t y_part = part1by2_21(yi) << 1;    // occupies bits 1,4,7,...
    uint64_t z_part = part1by2_21(zi) << 2;    // occupies bits 2,5,8,...
    return x_part | y_part | z_part;
}

// Compute Morton codes (21 bits per axis) for a list of 3D points.
// Steps:
//   1. Compute the axis-aligned bounding box of all points.
//   2. Normalize each point to [0,1) in x, y, z.
//   3. Quantize normalized coordinates to 21-bit integers.
//   4. Interleave bits to produce a 64-bit Morton code.
std::vector<uint64_t> computeMortonCodes(const std::vector<Point3D>& pts) {
    size_t N = pts.size();
    std::vector<uint64_t> morton_codes(N);

    if (N == 0) {
        return morton_codes;
    }

    // 1. Compute bounding box
    double xmin =  std::numeric_limits<double>::infinity();
    double ymin =  std::numeric_limits<double>::infinity();
    double zmin =  std::numeric_limits<double>::infinity();
    double xmax = -std::numeric_limits<double>::infinity();
    double ymax = -std::numeric_limits<double>::infinity();
    double zmax = -std::numeric_limits<double>::infinity();

    for (const auto& p : pts) {
        if (p.x < xmin) xmin = p.x;
        if (p.x > xmax) xmax = p.x;
        if (p.y < ymin) ymin = p.y;
        if (p.y > ymax) ymax = p.y;
        if (p.z < zmin) zmin = p.z;
        if (p.z > zmax) zmax = p.z;
    }

    // To avoid division by zero if all points have the same coordinate
    double dx = (xmax > xmin) ? (xmax - xmin) : 1.0;
    double dy = (ymax > ymin) ? (ymax - ymin) : 1.0;
    double dz = (zmax > zmin) ? (zmax - zmin) : 1.0;

    // 2. Normalize & 3. Quantize to 21-bit grid
    //
    // We map normalized coordinates to [0, 2^21 - 1] inclusive.
    const uint32_t MAX_21 = (1u << 21) - 1; // 0x1FFFFF

    for (size_t i = 0; i < N; ++i) {
        // Normalize to [0,1)
        double nx = (pts[i].x - xmin) / dx;
        double ny = (pts[i].y - ymin) / dy;
        double nz = (pts[i].z - zmin) / dz;

        // Clamp just in case rounding yields exactly 1.0
        if (nx < 0.0) nx = 0.0; else if (nx >= 1.0) nx = std::nextafter(1.0, 0.0);
        if (ny < 0.0) ny = 0.0; else if (ny >= 1.0) ny = std::nextafter(1.0, 0.0);
        if (nz < 0.0) nz = 0.0; else if (nz >= 1.0) nz = std::nextafter(1.0, 0.0);

        // Quantize to [0, MAX_21]
        uint32_t xi = static_cast<uint32_t>(nx * MAX_21);
        uint32_t yi = static_cast<uint32_t>(ny * MAX_21);
        uint32_t zi = static_cast<uint32_t>(nz * MAX_21);

        // 4. Interleave bits
        morton_codes[i] = morton3D_21(xi, yi, zi);
    }

    return morton_codes;
}

// Given Morton codes, return the permutation of indices that sorts them ascending.
std::vector<size_t> argsortMorton(const std::vector<uint64_t>& morton_codes) {
    size_t N = morton_codes.size();
    std::vector<size_t> indices(N);
    for (size_t i = 0; i < N; ++i) {
        indices[i] = i;
    }

    std::sort(indices.begin(), indices.end(),
        [&morton_codes](size_t a, size_t b) {
            return morton_codes[a] < morton_codes[b];
        }
    );

    return indices;
}

// int main() {
//     // Example usage: create some sample points (replace with your own data)
//     std::vector<Point3D> points = {
//         { 1e7,  2e7,  3e7},
//         { 5e7,  1e7,  4e7},
//         { 2e7,  8e7,  9e7},
//         { 9e6,  4e7,  7e7},
//         { 6e7,  3e7,  2e7},
//         { 7e7,  9e7,  5e7}
//     };

//     // Compute 64-bit Morton codes (21 bits per axis)
//     std::vector<uint64_t> codes = computeMortonCodes(points);

//     // Get sorted order of indices
//     std::vector<size_t> sorted_idx = argsortMorton(codes);

//     // Print results
//     std::cout << "Point index |      (x, y, z)       |    Morton code (hex)\n";
//     std::cout << "----------------------------------------------------------\n";
//     for (size_t rank = 0; rank < sorted_idx.size(); ++rank) {
//         size_t i = sorted_idx[rank];
//         std::printf("%11zu | (%9.2e, %9.2e, %9.2e) | 0x%016llx\n",
//                     i,
//                     points[i].x, points[i].y, points[i].z,
//                     static_cast<unsigned long long>(codes[i]));
//     }

//     return 0;
// }
