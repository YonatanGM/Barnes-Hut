#ifndef BODY_H
#define BODY_H

/**
 * @brief Structure representing a body's position.
 */
struct Position {
    double x;
    double y;
    double z;
};

/**
 * @brief Structure representing a body's velocity.
 */
struct Velocity {
    double vx;
    double vy;
    double vz;
};

/**
 * @brief Structure representing a body's acceleration.
 */
struct Acceleration {
    double ax;
    double ay;
    double az;
};

#endif // BODY_H
