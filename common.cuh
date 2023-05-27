#pragma once

#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define COMMON __host__ __device__

#define CUDA_CHECK(call) \
do {\
    const auto error = (call);\
    if (error != cudaSuccess)\
    {\
        std::cout << "Error " << #call << ' '  << cudaGetErrorString(error) << "\n";\
        exit(1);\
    }\
} while (0)