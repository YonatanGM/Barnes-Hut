#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <memory>

const double G = 6.67430e-11; // Gravitational constant
const double TIME_STEP = 1.0; // Time step for integration
const int NUM_STEPS = 1000; // Number of simulation steps
const double THETA = 0.5; // Barnes-Hut threshold
const bool VISUALIZE_STEPS = true; // Whether to visualize intermediate steps

struct Body {
    double mass;
    double pos_x, pos_y, pos_z;
    double vel_x, vel_y, vel_z;
    double acc_x, acc_y, acc_z;
};

 void readCSV(const std::string& filename, std::vector<Body>& bodies) {
        std::ifstream file(filename);
        std::string line;

        if (!file.is_open()) {
            std::cerr << "Error: Unable to open file " << filename << std::endl;
            return;
        }

        // Skip header line
        std::getline(file, line);

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string item;
            Body body;

            try {
                std::getline(ss, item, ','); // id
                std::getline(ss, item, ','); // name
                std::getline(ss, item, ','); // class

                std::getline(ss, item, ',');
                body.mass = std::stod(item);

                std::getline(ss, item, ','); body.pos_x = std::stod(item);
                std::getline(ss, item, ','); body.pos_y = std::stod(item);
                std::getline(ss, item, ','); body.pos_z = std::stod(item);
                std::getline(ss, item, ','); body.vel_x = std::stod(item);
                std::getline(ss, item, ','); body.vel_y = std::stod(item);
                std::getline(ss, item, ','); body.vel_z = std::stod(item);

                body.acc_x = body.acc_y = body.acc_z = 0.0; // Initialize accelerations
                bodies.push_back(body);
            } catch (const std::exception& e) {
                std::cerr << "Exception: " << e.what() << " in line: " << line << std::endl;
            }
        }
    }

   void saveState(const std::vector<Body>& bodies, const std::string& filename) {
        std::ofstream outFile(filename);
        if (outFile.is_open()) {
            outFile << "pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,mass\n";
            for (const auto& body : bodies) {
                outFile << body.pos_x << "," << body.pos_y << "," << body.pos_z << ","
                        << body.vel_x << "," << body.vel_y << "," << body.vel_z << ","
                        << body.mass << "\n";
            }
            outFile.close();
        } else {
            std::cerr << "Error: Unable to open output file " << filename << std::endl;
        }
    }


struct OctreeNode {
    double mass;
    double centerX, centerY, centerZ; // Center of mass for this node
    double width; // Size of this region
    Body* body; // Pointer to body if it's a leaf node (or null if it’s internal)
    std::unique_ptr<OctreeNode> children[8]; // Eight children for octree nodes

    OctreeNode(double x, double y, double z, double w)
        : mass(0), centerX(x), centerY(y), centerZ(z), width(w), body(nullptr) {}

    bool isLeaf() const { return body != nullptr; }

    // Insert a body into the tree, subdividing if necessary
    void insert(Body* newBody) {
        if (isLeaf()) {
            if (body == nullptr) {
                // Empty leaf, place the body here
                body = newBody;
                centerX = newBody->pos_x;
                centerY = newBody->pos_y;
                centerZ = newBody->pos_z;
                mass = newBody->mass;
            } else {
                // Subdivide node
                Body* existingBody = body;
                body = nullptr; // Make this an internal node
                for (int i = 0; i < 8; ++i) {
                    double offsetX = (i & 1) ? width / 4 : -width / 4;
                    double offsetY = (i & 2) ? width / 4 : -width / 4;
                    double offsetZ = (i & 4) ? width / 4 : -width / 4;
                    children[i] = std::make_unique<OctreeNode>(centerX + offsetX, centerY + offsetY, centerZ + offsetZ, width / 2);
                }
                insert(existingBody); // Insert existing body into one of the children
                insert(newBody);      // Insert new body into one of the children
            }
        } else {
            // Internal node, add mass to center of mass and propagate insertion
            mass += newBody->mass;
            centerX = (centerX * (mass - newBody->mass) + newBody->pos_x * newBody->mass) / mass;
            centerY = (centerY * (mass - newBody->mass) + newBody->pos_y * newBody->mass) / mass;
            centerZ = (centerZ * (mass - newBody->mass) + newBody->pos_z * newBody->mass) / mass;
            int octant = (newBody->pos_x > centerX ? 1 : 0) | (newBody->pos_y > centerY ? 2 : 0) | (newBody->pos_z > centerZ ? 4 : 0);
            children[octant]->insert(newBody);
        }
    }


    // Calculate acceleration on body_i due to this node using Barnes-Hut criterion
    void computeForceOnBody(Body& body_i) {
        if (body == &body_i) return; // Skip self

        double dx = centerX - body_i.pos_x;
        double dy = centerY - body_i.pos_y;
        double dz = centerZ - body_i.pos_z;
        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

        if (isLeaf() || (width / dist < THETA)) {
            // Use this node's mass as an approximation
            double force = G * mass * body_i.mass / (dist * dist * dist);
            body_i.acc_x += force * dx;
            body_i.acc_y += force * dy;
            body_i.acc_z += force * dz;
        } else {
            // Recursively calculate force from children
            for (auto& child : children) {
                if (child) child->computeForceOnBody(body_i);
            }
        }
    }
};

void computeAccelerations(std::vector<Body>& bodies, double width) {
    // Build octree
    OctreeNode root(0, 0, 0, width);
    for (auto& body : bodies) {
        root.insert(&body);
    }
    // Calculate forces using octree
    for (auto& body : bodies) {
        body.acc_x = body.acc_y = body.acc_z = 0.0;
        root.computeForceOnBody(body);
    }
}

void leapfrogIntegration(std::vector<Body>& bodies, double width) {
    computeAccelerations(bodies, width);
    for (auto& body : bodies) {
        body.pos_x += body.vel_x * TIME_STEP + 0.5 * body.acc_x * TIME_STEP * TIME_STEP;
        body.pos_y += body.vel_y * TIME_STEP + 0.5 * body.acc_y * TIME_STEP * TIME_STEP;
        body.pos_z += body.vel_z * TIME_STEP + 0.5 * body.acc_z * TIME_STEP * TIME_STEP;

        body.vel_x += body.acc_x * TIME_STEP;
        body.vel_y += body.acc_y * TIME_STEP;
        body.vel_z += body.acc_z * TIME_STEP;
    }
}

int main() {
    std::vector<Body> bodies;
    readCSV("../simulation_data/planets_and_moons_state_vectors.csv", bodies);
    double simulationWidth = 1.0e12; // Set appropriate width based on data

    saveState(bodies, "../output/initial_state.csv");
    for (int step = 0; step < NUM_STEPS; ++step) {
        leapfrogIntegration(bodies, simulationWidth);
        if (VISUALIZE_STEPS && (step % 100 == 0)) {
            saveState(bodies, "../output/state_step_" + std::to_string(step) + ".csv");
            std::cout << "State at step " << step << " saved." << std::endl;
        }
    }
    saveState(bodies, "../output/final_state.csv");
    return 0;
}
