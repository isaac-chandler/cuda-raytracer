#pragma once

#include "common.cuh"

template<typename T>
COMMON T lerp(float amount, const T& a, const T& b)
{
    return (1.0f - amount) * a + amount * b;
}

COMMON inline float3& operator*=(float3& v, float3 k)
{
    v.x *= k.x;
    v.y *= k.y;
    v.z *= k.z;

    return v;
}

COMMON inline float3& operator+=(float3& v, float3 k)
{
    v.x += k.x;
    v.y += k.y;
    v.z += k.z;

    return v;
}

COMMON inline float3& operator-=(float3& v, float3 k)
{
    v.x -= k.x;
    v.y -= k.y;
    v.z -= k.z;

    return v;
}

COMMON inline float3& operator*=(float3& v, float k)
{
    v.x *= k;
    v.y *= k;
    v.z *= k;

    return v;
}

COMMON inline float3 operator-(const float3& v)
{
    return {-v.x, -v.y, -v.z};
}

COMMON inline float elem(const float3& v, int i)
{
    return (&v.x)[i];
}

COMMON inline float& elem(float3& v, int i)
{
    return (&v.x)[i];
}


COMMON inline float dot(const float3 &a, const float3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

COMMON inline float3 cross(const float3 &a, const float3 &b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

COMMON inline float magnitude_squared(const float3 &vector)
{
    return vector.x * vector.x + vector.y * vector.y + vector.z * vector.z;
}

COMMON inline float magnitude(const float3 &vector)
{
    return sqrtf(magnitude_squared(vector));
}

COMMON inline float3 operator+(const float3 &a, const float3 &b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

COMMON inline float3 operator-(const float3 &a, const float3 &b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

COMMON inline float3 operator*(const float3 &a, const float3 &b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

COMMON inline float3 operator*(float scalar, const float3 &vector)
{
    return {scalar * vector.x, scalar * vector.y, scalar * vector.z};
}

COMMON inline float3 normalise(const float3 &vector)
{
#ifdef __CUDA_ARCH__
    float mult = rsqrtf(magnitude_squared(vector));
#else
    float mult = 1.0f / magnitude(vector);
#endif
    return mult * vector;
}

COMMON inline float clamp(float x, float min_val, float max_val)
{
    return max(min(x, max_val), min_val);
}

COMMON inline float clamp01(float x)
{
    return clamp(x, 0, 1);
}

COMMON inline float3 min(const float3 &a, const float3 &b)
{
    return {
        min(a.x, b.x),
        min(a.y, b.y),
        min(a.z, b.z),
    };
}

COMMON inline float3 max(const float3 &a, const float3 &b)
{
    return {
        max(a.x, b.x),
        max(a.y, b.y),
        max(a.z, b.z),
    };
}
