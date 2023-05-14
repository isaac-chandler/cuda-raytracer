#pragma once

#include "common.cuh"

template<typename T>
COMMON T lerp(float amount, const T& a, const T& b)
{
    return (1.0f - amount) * a + amount * b;
}

struct Vec3 
{
    float x, y, z;

    COMMON Vec3 &operator*=(const Vec3& other)
    {
        x *= other.x;
        y *= other.y;
        z *= other.z;

        return *this;
    }

    COMMON Vec3 &operator+=(const Vec3& other)
    {
        x += other.x;
        y += other.y;
        z += other.z;

        return *this;
    }

    COMMON Vec3 &operator-=(const Vec3& other)
    {
        x -= other.x;
        y -= other.y;
        z -= other.z;

        return *this;
    }

    COMMON Vec3 &operator*=(float scalar)
    {
        x *= scalar;
        y *= scalar;
        z *= scalar;

        return *this;
    }

    COMMON Vec3 operator-() const
    {
        return {-x, -y, -z};
    }

    float &operator[](size_t index)
    {
        return (&x)[index];
    }

    const float &operator[](size_t index) const
    {
        return (&x)[index];
    }
};

COMMON inline float dot(const Vec3 &a, const Vec3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

COMMON inline Vec3 cross(const Vec3 &a, const Vec3 &b)
{
    return {
        a.y * b.z - a.z * b.y, 
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x, 
    };
}

COMMON inline float magnitude_squared(const Vec3 &vector)
{
    return vector.x * vector.x + vector.y * vector.y + vector.z * vector.z;
}

COMMON inline float magnitude(const Vec3 &vector)
{
    return sqrtf(magnitude_squared(vector));
}

COMMON inline Vec3 operator+(const Vec3 &a, const Vec3 &b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

COMMON inline Vec3 operator-(const Vec3 &a, const Vec3 &b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

COMMON inline Vec3 operator*(const Vec3 &a, const Vec3 &b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

COMMON inline Vec3 operator*(float scalar, const Vec3 &vector)
{
    return {scalar * vector.x, scalar * vector.y, scalar * vector.z};
}

COMMON inline Vec3 normalise(const Vec3 &vector)
{
    return (1.0f / magnitude(vector)) * vector;
}

COMMON inline float clamp(float x, float min_val, float max_val)
{
    return max(min(x, max_val), min_val);
}

COMMON inline float clamp01(float x)
{
    return clamp(x, 0, 1);
}

COMMON inline Vec3 min(const Vec3 &a, const Vec3 &b)
{
    return {
        min(a.x, b.x), 
        min(a.y, b.y), 
        min(a.z, b.z), 
    };
}

COMMON inline Vec3 max(const Vec3 &a, const Vec3 &b)
{
    return {
        max(a.x, b.x), 
        max(a.y, b.y), 
        max(a.z, b.z), 
    };
}