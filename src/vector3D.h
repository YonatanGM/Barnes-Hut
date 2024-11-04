// vector3D.h

#ifndef VECTOR3D_H
#define VECTOR3D_H

#include <cmath>

class Vector3D {
public:
    double x, y, z;

    Vector3D() : x(0), y(0), z(0) {}
    Vector3D(double x, double y, double z) : x(x), y(y), z(z) {}

    Vector3D operator+(const Vector3D& other) const {
        return Vector3D(x + other.x, y + other.y, z + other.z);
    }

    Vector3D operator-(const Vector3D& other) const {
        return Vector3D(x - other.x, y - other.y, z - other.z);
    }

    Vector3D operator*(double scalar) const {
        return Vector3D(x * scalar, y * scalar, z * scalar);
    }

    Vector3D operator/(double scalar) const {
        return Vector3D(x / scalar, y / scalar, z / scalar);
    }

    double magnitude() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    Vector3D normalize() const {
        double mag = magnitude();
        return Vector3D(x / mag, y / mag, z / mag);
    }
};

#endif // VECTOR3D_H
