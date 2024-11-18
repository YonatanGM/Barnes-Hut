#ifndef KEPLER_TO_CARTESIAN_H
#define KEPLER_TO_CARTESIAN_H

#include "io.h"
#include "body.h"

// Structure representing orbital elements of a celestial body
struct OrbitalElements {
    double mass;
    double eccentricity;
    double semiMajorAxis;
    double inclination;
    double argOfPeriapsis;
    double longOfAscNode;
    double meanAnomaly;
};

// Read 
// Convert orbital elements to Cartesian state vectors
Body* convertKeplerToCartesian(const OrbitalElements& elem, double mu);

#endif // KEPLER_TO_CARTESIAN_H
