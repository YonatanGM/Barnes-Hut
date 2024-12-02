#ifndef INTEGRATION_H
#define INTEGRATION_H

#include "body.h"
#include <vector>
#include <omp.h>

// Perform leapfrog integration on the list of bodies
void leapfrogIntegration(std::vector<Body>& bodies, double dt);

#endif 
