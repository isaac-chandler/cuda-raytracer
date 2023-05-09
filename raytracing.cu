#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cub/device/device_radix_sort.cuh>

#include "common.cuh"
#include "math.cuh"
#include "scene.cuh"
#include "random.cuh"

__constant__ Scene cuda_scene;

__global__ void cuda_generate_initial_rays(RayData *ray_data, unsigned int *ray_indices, unsigned int *ray_keys, int rays_per_pixel, int seed)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    cuda_scene.generate_initial_rays(ray_data, ray_indices, ray_keys, rays_per_pixel, index, seed);
}

__global__ void cuda_process_rays(RayData *ray_data, unsigned int *ray_indices, unsigned int *keys, int ray_count, int seed)
{
    int ray_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (ray_index < ray_count)
    {
        xor_random rng;
        xor_srand(&rng, ray_index * 4137874753 + 279220567 * seed);

        cuda_scene.process_ray(ray_data + ray_indices[ray_index], keys + ray_index, rng);
    }
}

__global__ void cuda_accumulate_rays(Vec3 *framebuffer, RayData *ray_data, int rays_per_pixel)
{
    int ray_index = blockIdx.x * blockDim.x + threadIdx.x;
    int framebuffer_index = ray_index / rays_per_pixel;

    atomicAdd(&framebuffer[framebuffer_index].x, ray_data[ray_index].collected_color.x);
    atomicAdd(&framebuffer[framebuffer_index].y, ray_data[ray_index].collected_color.y);
    atomicAdd(&framebuffer[framebuffer_index].z, ray_data[ray_index].collected_color.z);
}

#define CUDA_CHECK(call) \
do {\
    const auto error = (call);\
    if (error != cudaSuccess)\
    {\
        std::cout << "Error " << #call << ' '  << cudaGetErrorString(error) << "\n";\
        exit(1);\
    }\
} while (0)

#define MAX_RAYS_PER_PIXEL_PER_PASS 20

void accumulate_rays_to_framebuffer(Vec3 *framebuffer, RayData *ray_data, int total_rays, int rays_per_pixel)
{
    for (int i = 0; i < total_rays; i++)
    {
        if (isfinite(ray_data[i].collected_color.x) && isfinite(ray_data[i].collected_color.y) && isfinite(ray_data[i].collected_color.z))
            framebuffer[i / rays_per_pixel] += ray_data[i].collected_color;
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <scene>\n";
        exit(1);
    }

    bool sort = true;
    bool cpu = false;
    bool bvh = true;

    for (int i = 2; i < argc; i++)
    {
        if (strcmp(argv[i], "no_sort") == 0)
        {
            sort = false;
        }
        else if (strcmp(argv[i], "cpu") == 0)
        {
            cpu = true;
        }
        else if (strcmp(argv[i], "no_bvh") == 0)
        {
            bvh = false;
        }
    }

    Scene scene = {};
    load_scene(&scene, argv[1], bvh);

    std::vector<Vec3> framebuffer{(size_t) (scene.width * scene.height)};    
    std::vector<RayData> ray_data((size_t) scene.width * (size_t) scene.height * (size_t) MAX_RAYS_PER_PIXEL_PER_PASS);

    auto start_time = std::chrono::high_resolution_clock::now();
    decltype(start_time) end_time;
    int remaining_rays = scene.ray_count;

    if (cpu)
    {
        while (remaining_rays)
        {
            int rays_to_cast = min(remaining_rays, MAX_RAYS_PER_PIXEL_PER_PASS);
            remaining_rays -= rays_to_cast;

            int total_rays = rays_to_cast * scene.width * scene.height;
            #pragma omp parallel for schedule(dynamic, 1000)
            for (int i = 0; i < total_rays; i++)
            {
                scene.generate_initial_rays(&ray_data.front(), nullptr, nullptr, rays_to_cast, i, remaining_rays);
            }

            for (int i = 0; i < scene.bounces; i++)
            {
                #pragma omp parallel for schedule(dynamic, 1000)
                for (int i = 0; i < total_rays; i++)
                {
                    xor_random rng;
                    xor_srand(&rng, 1905678123 * i + 345903 * (remaining_rays * MAX_RAYS_PER_PIXEL_PER_PASS + i));
                    unsigned int ray_key;
                    scene.process_ray(&ray_data[i], &ray_key, rng);
                }
            }

            accumulate_rays_to_framebuffer(&framebuffer.front(), &ray_data.front(), total_rays, rays_to_cast);
        }

        end_time = std::chrono::high_resolution_clock::now();

    }

    auto cpu_time = std::chrono::duration<float>(end_time - start_time).count();

    if (cpu)
    {
        std::cout << "CPU Took " << cpu_time << "s\n";
    }
    
    std::vector<unsigned char> output_image;
    output_image.reserve(scene.width * scene.height * 3 * 2);

    if (cpu)
    {
        for (const auto &pixel : framebuffer)
        {
            float r = pixel.x / scene.ray_count;
            float g = pixel.y / scene.ray_count;
            float b = pixel.z / scene.ray_count;

            output_image.push_back((unsigned char) (sqrtf(r / (r + 1)) * 255.999f));
            output_image.push_back((unsigned char) (sqrtf(g / (g + 1)) * 255.999f));
            output_image.push_back((unsigned char) (sqrtf(b / (b + 1)) * 255.999f));
        }
    } 

    std::fill(framebuffer.begin(), framebuffer.end(), Vec3{0, 0, 0});

    cudaFuncAttributes attribs;
    cudaFuncGetAttributes(&attribs, cuda_generate_initial_rays);
    cudaFuncGetAttributes(&attribs, cuda_process_rays);
    cudaFuncGetAttributes(&attribs, cuda_accumulate_rays);

    start_time = std::chrono::high_resolution_clock::now();

    Scene scene_copy = scene;

    const auto material_count = scene.sphere_count + scene.triangle_count;

    RayData *cuda_ray_data;
    unsigned int *cuda_ray_keys[2];
    unsigned int *cuda_ray_indices[2];
    size_t cuda_sort_temp_storage_size = 0;
    void *cuda_sort_temp_storage;

    Vec3 *cuda_framebuffer;


    CUDA_CHECK(cudaMalloc(&cuda_ray_data,          ray_data.size()      * sizeof(RayData)));
    CUDA_CHECK(cudaMalloc(&cuda_ray_keys[0],       ray_data.size()      * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&cuda_ray_keys[1],       ray_data.size()      * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&cuda_ray_indices[0],    ray_data.size()      * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&cuda_ray_indices[1],    ray_data.size()      * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&scene_copy.spheres,     scene.sphere_count   * sizeof(Sphere)));
    CUDA_CHECK(cudaMalloc(&scene_copy.triangles,   scene.triangle_count * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&scene_copy.materials,   material_count       * sizeof(Material)));
    CUDA_CHECK(cudaMalloc(&scene_copy.bvh,         scene.bvh_node_count * sizeof(BvhNode)));
    CUDA_CHECK(cudaMalloc(&cuda_framebuffer,       framebuffer.size()   * sizeof(Vec3)));

    cudaStream_t scene_copy_stream;
    cudaEvent_t scene_copy_done;
    cudaStream_t framebuffer_stream;
    cudaEvent_t framebuffer_done;

    CUDA_CHECK(cudaStreamCreate(&scene_copy_stream));
    CUDA_CHECK(cudaEventCreateWithFlags(&scene_copy_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaStreamCreate(&framebuffer_stream));
    CUDA_CHECK(cudaEventCreateWithFlags(&framebuffer_done, cudaEventDisableTiming));

    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, cuda_sort_temp_storage_size, cuda_ray_keys[0], cuda_ray_keys[1], cuda_ray_indices[0], cuda_ray_indices[1], ray_data.size()));
    CUDA_CHECK(cudaMalloc(&cuda_sort_temp_storage, cuda_sort_temp_storage_size));

    CUDA_CHECK(cudaMemcpyAsync(scene_copy.spheres,     scene.spheres,   sizeof(Sphere)   * scene.sphere_count,   cudaMemcpyHostToDevice, scene_copy_stream));    
    CUDA_CHECK(cudaMemcpyAsync(scene_copy.triangles,   scene.triangles, sizeof(Triangle) * scene.triangle_count, cudaMemcpyHostToDevice, scene_copy_stream));    
    CUDA_CHECK(cudaMemcpyAsync(scene_copy.materials,   scene.materials, sizeof(Material) * material_count,       cudaMemcpyHostToDevice, scene_copy_stream));
    CUDA_CHECK(cudaMemcpyAsync(scene_copy.bvh,         scene.bvh,       sizeof(BvhNode)  * scene.bvh_node_count, cudaMemcpyHostToDevice, scene_copy_stream));

    CUDA_CHECK(cudaMemcpyToSymbolAsync(cuda_scene, &scene_copy, sizeof(Scene), 0, cudaMemcpyHostToDevice, scene_copy_stream));
    cudaEventRecord(scene_copy_done, scene_copy_stream);

    cudaMemsetAsync(cuda_framebuffer, 0, framebuffer.size() * sizeof(Vec3), framebuffer_stream);
    cudaEventRecord(framebuffer_done, framebuffer_stream);

    remaining_rays = scene.ray_count;


    while (remaining_rays)
    {
        int rays_to_cast = min(remaining_rays, MAX_RAYS_PER_PIXEL_PER_PASS);
        remaining_rays -= rays_to_cast;

        int total_rays = rays_to_cast * scene.width * scene.height;
        cuda_generate_initial_rays<<<(total_rays + 127) / 128, 128 >>>(cuda_ray_data, cuda_ray_indices[0], cuda_ray_keys[0], rays_to_cast, remaining_rays);

        
        for (int i = 0; i < scene.bounces; i++)
        {
            cudaStreamWaitEvent(0, scene_copy_done);
            cuda_process_rays<<<(total_rays + 127) / 128, 128>>>(cuda_ray_data, cuda_ray_indices[0], cuda_ray_keys[0], total_rays, remaining_rays * MAX_RAYS_PER_PIXEL_PER_PASS + i);

            if (sort && i + 1 != scene.bounces)
            {
                CUDA_CHECK(cub::DeviceRadixSort::SortPairs(cuda_sort_temp_storage, cuda_sort_temp_storage_size, 
                    cuda_ray_keys[0], cuda_ray_keys[1], 
                    cuda_ray_indices[0], cuda_ray_indices[1],
                    total_rays));

                std::swap(cuda_ray_indices[0], cuda_ray_indices[1]);
                std::swap(cuda_ray_keys[0], cuda_ray_keys[1]);
            }

        }

        cudaStreamWaitEvent(0, framebuffer_done);
        cuda_accumulate_rays<<<(total_rays + 127) / 128, 128>>>(cuda_framebuffer, cuda_ray_data, rays_to_cast);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&framebuffer.front(), cuda_framebuffer, framebuffer.size() * sizeof(Vec3), cudaMemcpyDeviceToHost));
    
    end_time = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration<float>(end_time - start_time).count();
    std::cout << "GPU Took " << gpu_time << "s\n";

    if (cpu)
    {
        std::cout << "Speedup " << (cpu_time / gpu_time) << "\n";
    }

    for (const auto &pixel : framebuffer)
    {
        float r = pixel.x / scene.ray_count;
        float g = pixel.y / scene.ray_count;
        float b = pixel.z / scene.ray_count;

        output_image.push_back((unsigned char) (sqrtf(r / (r + 1)) * 255.999f));
        output_image.push_back((unsigned char) (sqrtf(g / (g + 1)) * 255.999f));
        output_image.push_back((unsigned char) (sqrtf(b / (b + 1)) * 255.999f));
    }


    stbi_write_png("raytracing.png", scene.width, output_image.size() / scene.width / 3, 3, &output_image.front(), scene.width * 3);
    system("start raytracing.png");

    return 0;
}