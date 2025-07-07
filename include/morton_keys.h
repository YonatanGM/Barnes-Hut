#pragma once

#include "body.h"
#include "bounding_box.h"
#include <vector>
#include <cstdint>


// Computes a 63-bit Morton code from 21-bit integer coordinates for each axis.
uint64_t morton63(uint32_t xi, uint32_t yi, uint32_t zi);

// Calculates Morton codes for a vector of positions.
// This involves finding the bounding box and normalizing positions to a unit cube.
std::vector<uint64_t> mortonCodes(const std::vector<Position>& pos);


struct CodesAndNorm {
    std::vector<uint64_t> code;   // Morton 63-bit keys
    std::vector<Position> norm;   // normalised positions (1-ε max)
};

CodesAndNorm mortonCodes(const std::vector<Position>& raw, const BoundingBox& bb);          // new