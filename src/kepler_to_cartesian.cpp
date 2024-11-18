#include "kepler_to_cartesian.h"
#include <cmath>

// Convert orbital elements to Cartesian state vectors
Body* convertKeplerToCartesian(const OrbitalElements& elem, double mu) {
    // Convert angles from degrees to radians if necessary
    double inclination = elem.inclination;
    double argOfPeriapsis = elem.argOfPeriapsis;
    double longOfAscNode = elem.longOfAscNode;
    double meanAnomaly = elem.meanAnomaly;

    // Solve Kepler's Equation for Eccentric Anomaly (E)
    double E = meanAnomaly; // Initial guess
    const double tolerance = 1e-8;
    int maxIterations = 100;
    for (int i = 0; i < maxIterations; ++i) {
        double f = E - elem.eccentricity * sin(E) - meanAnomaly;
        double f_prime = 1 - elem.eccentricity * cos(E);
        double delta = -f / f_prime;
        E += delta;
        if (fabs(delta) < tolerance) {
            break;
        }
    }

    // Compute True Anomaly (ν)
    double trueAnomaly = 2 * atan2(
        sqrt(1 + elem.eccentricity) * sin(E / 2),
        sqrt(1 - elem.eccentricity) * cos(E / 2)
    );

    // Compute distance (r)
    double r = elem.semiMajorAxis * (1 - elem.eccentricity * cos(E));

    // Position in orbital plane
    double x_orb = r * cos(trueAnomaly);
    double y_orb = r * sin(trueAnomaly);

    // Velocity in orbital plane
    double h = sqrt(mu * elem.semiMajorAxis * (1 - elem.eccentricity * elem.eccentricity));
    double vx_orb = -mu / h * sin(E);
    double vy_orb = mu / h * sqrt(1 - elem.eccentricity * elem.eccentricity) * cos(E);

    // Rotate to inertial frame
    double cos_Omega = cos(longOfAscNode);
    double sin_Omega = sin(longOfAscNode);
    double cos_i = cos(inclination);
    double sin_i = sin(inclination);
    double cos_omega = cos(argOfPeriapsis);
    double sin_omega = sin(argOfPeriapsis);

    // Rotation matrix components
    double R11 = cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i;
    double R12 = -cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i;
    double R13 = sin_Omega * sin_i;
    double R21 = sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i;
    double R22 = -sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i;
    double R23 = -cos_Omega * sin_i;
    double R31 = sin_omega * sin_i;
    double R32 = cos_omega * sin_i;
    double R33 = cos_i;

    // Position in inertial frame
    double x = R11 * x_orb + R12 * y_orb;
    double y = R21 * x_orb + R22 * y_orb;
    double z = R31 * x_orb + R32 * y_orb;

    // Velocity in inertial frame
    double vx = R11 * vx_orb + R12 * vy_orb;
    double vy = R21 * vx_orb + R22 * vy_orb;
    double vz = R31 * vx_orb + R32 * vy_orb;

    // Create and return the Body object
    Body* body = new Body;
    body->mass = elem.mass;
    body->x = x;
    body->y = y;
    body->z = z;
    body->vx = vx;
    body->vy = vy;
    body->vz = vz;
    body->ax = 0.0;
    body->ay = 0.0;
    body->az = 0.0;

    return body;
}
