
#ifndef BODY_H
#define BODY_H

#include <string>

// structure representing a celestial body
struct Body {
    int index;             // unique index of body 
    double mass;        // mass of the body
    double x, y, z;     // position coordinates
    double vx, vy, vz;  // velocity components
    double ax, ay, az;  // acceleration components
    // std::string name;   // name of the body
};

#endif 