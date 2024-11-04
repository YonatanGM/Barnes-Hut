#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>

const double G = 6.67430e-11; // Gravitational constant
const double TIME_STEP = 1.0; // Time step for integration
const int NUM_STEPS = 1000; // Number of simulation steps
const bool VISUALIZE_STEPS = true; // Whether to visualize intermediate steps

struct Body {
    double mass;
    double pos_x, pos_y, pos_z;
    double vel_x, vel_y, vel_z;
    double acc_x, acc_y, acc_z; // For acceleration
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

void computeAccelerations(std::vector<Body>& bodies) {
    const size_t num_bodies = bodies.size();
    for (size_t i = 0; i < num_bodies; ++i) {
        Body& body_i = bodies[i];
        body_i.acc_x = body_i.acc_y = body_i.acc_z = 0.0;

        for (size_t j = 0; j < num_bodies; ++j) {
            if (i == j) continue;
            Body& body_j = bodies[j];

            double dx = body_j.pos_x - body_i.pos_x;
            double dy = body_j.pos_y - body_i.pos_y;
            double dz = body_j.pos_z - body_i.pos_z;
            double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
            double force = G * body_i.mass * body_j.mass / (dist * dist * dist);

            body_i.acc_x += force * dx;
            body_i.acc_y += force * dy;
            body_i.acc_z += force * dz;
        }
    }
}

void leapfrogIntegration(std::vector<Body>& bodies) {
    for (auto& body : bodies) {
        // Update positions
        body.pos_x += body.vel_x * TIME_STEP + 0.5 * body.acc_x * TIME_STEP * TIME_STEP;
        body.pos_y += body.vel_y * TIME_STEP + 0.5 * body.acc_y * TIME_STEP * TIME_STEP;
        body.pos_z += body.vel_z * TIME_STEP + 0.5 * body.acc_z * TIME_STEP * TIME_STEP;

        // Update velocities
        body.vel_x += body.acc_x * TIME_STEP;
        body.vel_y += body.acc_y * TIME_STEP;
        body.vel_z += body.acc_z * TIME_STEP;

        // Update accelerations
        computeAccelerations(bodies);
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

int main() {
    std::vector<Body> bodies;
    readCSV("../simulation_data/planets_and_moons_state_vectors.csv", bodies);
    
    // Save initial state
    saveState(bodies, "../output/initial_state.csv");

    // Visualization of initial state
    std::cout << "Initial state saved to initial_state.csv" << std::endl;

    for (int step = 0; step < NUM_STEPS; ++step) {
        leapfrogIntegration(bodies);

        if (VISUALIZE_STEPS && (step % 100 == 0)) { // Visualize every 100 steps
            saveState(bodies, "../output/state_step_" + std::to_string(step) + ".csv");
            std::cout << "State at step " << step << " saved." << std::endl;
        }
    }
    
    // Save final state
    saveState(bodies, "../output/final_state.csv");
    std::cout << "Final state saved to final_state.csv" << std::endl;
    
    return 0;
}
