#ifndef BODY_H
#define BODY_H

#include <string>

// Structure representing a celestial body
struct Body {
    double mass;     // Mass of the body
    double x, y, z;  // Position coordinates
    double vx, vy, vz; // Velocity components
    double ax, ay, az; // Acceleration components
    std::string name;  // Name of the body
};

#endif // BODY_H
