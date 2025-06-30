#pragma once

// Defines the available policies for load balancing.
enum class LBPolicy {
    Octant,  // Simple geometric decomposition (for baseline tests)
    Hist256  // Histogram-based partitioning (for production)
};

// Defines the available policies for force calculation data exchange.
enum class FCPolicy {
    Tree,    // Exchange full trees (for validation)
    Bodies,  // Exchange raw particle data
    LET      // Exchange Locally Essential Trees (the goal)
};