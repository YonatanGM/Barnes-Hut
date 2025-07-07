#include "morton_keys.h"
#include <algorithm>
#include <cmath>
#include <limits>


// Spreads the lower 21 bits of an integer to a 63-bit value,
// inserting two zero bits between each original bit.
// This is used to interleave the x, y, and z coordinates.
static inline uint64_t spread21(uint32_t v) {
    uint64_t x = v & 0x1FFFFF; // Mask to 21 bits
    x = (x | (x << 32)) & 0x1F00000000FFFFULL;
    x = (x | (x << 16)) & 0x1F0000FF0000FFULL;
    x = (x | (x <<  8)) & 0x100F00F00F00F00FULL;
    x = (x | (x <<  4)) & 0x10C30C30C30C30C3ULL;
    x = (x | (x <<  2)) & 0x1249249249249249ULL;
    return x;
}

uint64_t morton63(uint32_t xi, uint32_t yi, uint32_t zi) {
    // Interleave the spread bits of x, y, and z coordinates
    return spread21(xi) | (spread21(yi) << 1) | (spread21(zi) << 2);
}

std::vector<uint64_t> mortonCodes(const std::vector<Position>& pos) {
    const size_t n = pos.size();
    std::vector<uint64_t> codes(n);
    if (n == 0) return codes;

    // 1. Find the bounding box of all positions
    double x_min = std::numeric_limits<double>::max();
    double y_min = std::numeric_limits<double>::max();
    double z_min = std::numeric_limits<double>::max();
    double x_max = std::numeric_limits<double>::lowest();
    double y_max = std::numeric_limits<double>::lowest();
    double z_max = std::numeric_limits<double>::lowest();

    for (const auto& p : pos) {
        x_min = std::min(x_min, p.x);
        y_min = std::min(y_min, p.y);
        z_min = std::min(z_min, p.z);
        x_max = std::max(x_max, p.x);
        y_max = std::max(y_max, p.y);
        z_max = std::max(z_max, p.z);
    }

    // 2. Determine the span of the bounding box, avoiding division by zero
    double dx = (x_max == x_min) ? 1.0 : x_max - x_min;
    double dy = (y_max == y_min) ? 1.0 : y_max - y_min;
    double dz = (z_max == z_min) ? 1.0 : z_max - z_min;

    // Max value for a 21-bit integer
    constexpr uint32_t Q = (1u << 21) - 1;

    // 3. Normalize positions and convert to Morton codes
    auto normalize = [&](double val, double min_val, double span) {
        double t = (val - min_val) / span;
        if (t <= 0.0) return 0.0;
        // Clamp to just below 1.0 to ensure the integer value fits in 21 bits
        if (t >= 1.0) return std::nextafter(1.0, 0.0);
        return t;
    };

    for (size_t i = 0; i < n; ++i) {
        uint32_t xi = static_cast<uint32_t>(normalize(pos[i].x, x_min, dx) * Q);
        uint32_t yi = static_cast<uint32_t>(normalize(pos[i].y, y_min, dy) * Q);
        uint32_t zi = static_cast<uint32_t>(normalize(pos[i].z, z_min, dz) * Q);
        codes[i] = morton63(xi, yi, zi);
    }

    return codes;
}

CodesAndNorm mortonCodes(const std::vector<Position>& raw, const BoundingBox& box)
{
    const std::size_t n = raw.size();
    CodesAndNorm out{ std::vector<uint64_t>(n), std::vector<Position>(n) };
    if (!n) return out;

    // CORRECTED: Use box.min.x, box.max.x, etc.
    const double dx = (box.max.x == box.min.x) ? 1.0 : (box.max.x - box.min.x);
    const double dy = (box.max.y == box.min.y) ? 1.0 : (box.max.y - box.min.y);
    const double dz = (box.max.z == box.min.z) ? 1.0 : (box.max.z - box.min.z);

    constexpr uint32_t Q = (1u << 21) - 1;   // 21-bit quantiser

    auto norm = [](double v, double lo, double span)
    {
        double t = (v - lo) / span;
        if (t <= 0.0) return 0.0;
        if (t >= 1.0) return std::nextafter(1.0, 0.0);
        return t;
    };

    for (std::size_t i = 0; i < n; ++i)
    {
        // CORRECTED: Use box.min.x, box.min.y, etc.
        double xn = norm(raw[i].x, box.min.x, dx);
        double yn = norm(raw[i].y, box.min.y, dy);
        double zn = norm(raw[i].z, box.min.z, dz);

        out.norm[i] = { xn, yn, zn };

        uint32_t xi = static_cast<uint32_t>(xn * Q);
        uint32_t yi = static_cast<uint32_t>(yn * Q);
        uint32_t zi = static_cast<uint32_t>(zn * Q);
        out.code[i] = morton63(xi, yi, zi);
    }
    return out;
}
