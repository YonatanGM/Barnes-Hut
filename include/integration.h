#ifndef INTEGRATION_H
#define INTEGRATION_H

#include <vector>
#include "body.h"

// Perform leapfrog integration on the list of bodies
void leapfrogIntegration(std::vector<Body*>& bodies, double dt);

#endif // INTEGRATION_H
