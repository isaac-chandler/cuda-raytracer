#pragma once

#include "common.cuh"

struct xor_random
{
    unsigned long long state;
    unsigned long long inc;
};

COMMON inline unsigned int xor_rand(xor_random* rng)
{
    auto old_state = rng->state;

    rng->state = old_state * 6364136223846793005ULL + (rng->inc | 1);

    auto xor_shifted = (unsigned int) (((old_state >> 18) ^ old_state) >> 27);
    auto rot = (unsigned int) (old_state >> 59);

    return (xor_shifted >> rot) | (xor_shifted << ((-rot) & 31));
}

COMMON inline void xor_srand(xor_random* rng, unsigned int seed)
{
    rng->state = seed * 6839056345687307ULL;
    rng->inc = 820957824423429ULL;
    xor_rand(rng);
}

COMMON inline float random01(xor_random *rng)
{
    return xor_rand(rng) * (1.0f /  UINT_MAX);
}

COMMON inline float random02(xor_random *rng)
{
    return xor_rand(rng) * (2.0f /  UINT_MAX);
}

COMMON inline float random_radians(xor_random *rng)
{
    return xor_rand(rng) * (M_PI * 2 /  UINT_MAX);
}

inline float random_bidir(xor_random *rng)
{
    return random02(rng) - 1;
}

inline Vec3 random_in_sphere(xor_random *rng)
{
    Vec3 v;

    do {
        v = {random_bidir(rng), random_bidir(rng), random_bidir(rng)};
    } while (magnitude_squared(v) >  1);

    return v;
}

inline COMMON Vec3 random_on_sphere(xor_random *rng)
{
    float r1 = random_radians(rng);
    float r2 = random02(rng);

    float x = sqrtf(r2 * (2 - r2));

    return {
        cosf(r1) * x, 
        sinf(r1) * x, 
        1 - r2,
    };
}