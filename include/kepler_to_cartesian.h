#ifndef KEPLER_TO_CARTESIAN_H
#define KEPLER_TO_CARTESIAN_H

#include "io.h"


// Structure representing orbital elements of a celestial body
struct OrbitalElements {
    // double mass;
    double eccentricity;
    double semiMajorAxis;
    double inclination;
    double argOfPeriapsis;
    double longOfAscNode;
    double meanAnomaly;
    double epoch;
};

// structure representing a  body (using state vectors)
struct Body {
    int index;             // unique index of body 
    double mass;        // mass of the body
    double x, y, z;     // position coordinates
    double vx, vy, vz;  // velocity components
    double ax, ay, az;  // acceleration components
    std::string name;   // name of the body
};

// Read 
// Convert orbital elements to Cartesian state vectors
Body* convertKeplerToCartesian(const OrbitalElements& elem);

#endif 
