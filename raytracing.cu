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

#define MAX_RAYS_PER_PIXEL_PER_PASS 20

void accumulate_rays_to_framebuffer(Vec3 *framebuffer, RayData *ray_data, int total_rays, int rays_per_pixel)
{
    for (int i = 0; i < total_rays; i++)
    {
        framebuffer[i / rays_per_pixel] += ray_data[i].collected_color;
    }
}

Vec3 *cpu_raytrace(Scene *scene)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    int remaining_rays = scene->ray_count;

    RayData *ray_data = new RayData[scene->width * scene->height * MAX_RAYS_PER_PIXEL_PER_PASS];
    Vec3 *framebuffer = new Vec3[scene->width * scene->height]{};

    while (remaining_rays)
    {
        int rays_to_cast = min(remaining_rays, MAX_RAYS_PER_PIXEL_PER_PASS);
        remaining_rays -= rays_to_cast;

        int total_rays = rays_to_cast * scene->width * scene->height;
        #pragma omp parallel for schedule(dynamic, 1000)
        for (int i = 0; i < total_rays; i++)
        {
            scene->generate_initial_rays(ray_data, nullptr, nullptr, rays_to_cast, i, remaining_rays);
        }

        for (int i = 0; i < scene->bounces; i++)
        {
            #pragma omp parallel for schedule(dynamic, 1000)
            for (int i = 0; i < total_rays; i++)
            {
                xor_random rng;
                xor_srand(&rng, 1905678123 * i + 345903 * (remaining_rays * MAX_RAYS_PER_PIXEL_PER_PASS + i));
                scene->process_ray(&ray_data[i], nullptr, rng);
            }
        }

        accumulate_rays_to_framebuffer(framebuffer, ray_data, total_rays, rays_to_cast);
    }

    delete[] ray_data;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration<float>(end_time - start_time).count();
    std::cout << "CPU Took " << cpu_time << "s\n";

    return framebuffer;
}

Vec3 *gpu_raytrace(const Scene *scene, bool sort)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    int pixel_count = scene->width * scene->height;
    Vec3 *cuda_framebuffer;
    CUDA_CHECK(cudaMalloc(&cuda_framebuffer, pixel_count   * sizeof(Vec3)));

    int max_ray_count = pixel_count * MAX_RAYS_PER_PIXEL_PER_PASS;
    RayData *cuda_ray_data;
    CUDA_CHECK(cudaMalloc(&cuda_ray_data, max_ray_count * sizeof(RayData)));

    unsigned int *cuda_ray_keys[2];
    CUDA_CHECK(cudaMalloc(&cuda_ray_keys[0],    max_ray_count * sizeof(unsigned int)));

    unsigned int *cuda_ray_indices[2];
    CUDA_CHECK(cudaMalloc(&cuda_ray_indices[0], max_ray_count * sizeof(unsigned int)));

    size_t cuda_sort_temp_storage_size = 0;
    void *cuda_sort_temp_storage;
    if (sort)
    {
        CUDA_CHECK(cudaMalloc(&cuda_ray_keys[1],    max_ray_count * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&cuda_ray_indices[1], max_ray_count * sizeof(unsigned int)));
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, cuda_sort_temp_storage_size, cuda_ray_keys[0], cuda_ray_keys[1], 
                                                   cuda_ray_indices[0], cuda_ray_indices[1], max_ray_count));
        CUDA_CHECK(cudaMalloc(&cuda_sort_temp_storage, cuda_sort_temp_storage_size));
    }

    cudaStream_t scene_copy_stream;
    cudaEvent_t scene_copy_done;
    cudaStream_t framebuffer_stream;
    cudaEvent_t framebuffer_done;

    CUDA_CHECK(cudaStreamCreate(&scene_copy_stream));
    CUDA_CHECK(cudaEventCreateWithFlags(&scene_copy_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaStreamCreate(&framebuffer_stream));
    CUDA_CHECK(cudaEventCreateWithFlags(&framebuffer_done, cudaEventDisableTiming));

    cuda_scene.copy_from_cpu_async(*scene, scene_copy_stream);
    cudaEventRecord(scene_copy_done, scene_copy_stream);

    cudaMemsetAsync(cuda_framebuffer, 0, pixel_count * sizeof(Vec3), framebuffer_stream);
    cudaEventRecord(framebuffer_done, framebuffer_stream);

    int remaining_rays = scene->ray_count;

    while (remaining_rays)
    {
        int rays_to_cast = min(remaining_rays, MAX_RAYS_PER_PIXEL_PER_PASS);
        remaining_rays -= rays_to_cast;

        int total_rays = rays_to_cast * scene->width * scene->height;
        cuda_generate_initial_rays<<<(total_rays + 127) / 128, 128 >>>(cuda_ray_data, cuda_ray_indices[0], cuda_ray_keys[0], rays_to_cast, remaining_rays);

        
        for (int i = 0; i < scene->bounces; i++)
        {
            cudaStreamWaitEvent(0, scene_copy_done);
            cuda_process_rays<<<(total_rays + 127) / 128, 128>>>(cuda_ray_data, cuda_ray_indices[0], cuda_ray_keys[0], total_rays, remaining_rays * MAX_RAYS_PER_PIXEL_PER_PASS + i);

            if (sort && i + 1 != scene->bounces)
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

    Vec3 *framebuffer = new Vec3[pixel_count];

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(framebuffer, cuda_framebuffer, pixel_count * sizeof(Vec3), cudaMemcpyDeviceToHost));

    cuda_scene.free_from_gpu();

    CUDA_CHECK(cudaFree(cuda_framebuffer));
    CUDA_CHECK(cudaFree(cuda_ray_data));
    CUDA_CHECK(cudaFree(cuda_ray_keys[0]));
    CUDA_CHECK(cudaFree(cuda_ray_indices[0]));
    if (sort)
    {
        CUDA_CHECK(cudaFree(cuda_ray_keys[1]));
        CUDA_CHECK(cudaFree(cuda_ray_indices[1]));
        CUDA_CHECK(cudaFree(cuda_sort_temp_storage));
    }

    cudaEventDestroy(framebuffer_done);
    cudaEventDestroy(scene_copy_done);
    cudaStreamDestroy(framebuffer_stream);
    cudaStreamDestroy(scene_copy_stream);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration<float>(end_time - start_time).count();
    std::cout << "GPU Took " << gpu_time << "s\n";

    return framebuffer;
}

void write_framebuffer_to_output_image(Scene *scene, std::vector<unsigned char> &output_image, const Vec3 *framebuffer)
{
    for (int pixel_index = 0; pixel_index < scene->width * scene->height; pixel_index++)
    {
        auto pixel = (scene->exposure / scene->ray_count) * framebuffer[pixel_index];

        float r = pixel.x;
        float g = pixel.y;
        float b = pixel.z;

        output_image.push_back((unsigned char) (sqrtf(r / (r + 1)) * 255.999f));
        output_image.push_back((unsigned char) (sqrtf(g / (g + 1)) * 255.999f));
        output_image.push_back((unsigned char) (sqrtf(b / (b + 1)) * 255.999f));
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <scene>\n";
        return 1;
    }

    bool sort = true;
    bool cpu = false;
    bool gpu = true;
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
        else if (strcmp(argv[i], "no_gpu") == 0)
        {
            gpu = false;
        }
    }

    if (!cpu && !gpu)
    {
        std::cout << "No raytracing hardware specified\n";
        return 2;
    }

    Scene scene = {};
    load_scene(&scene, argv[1], bvh);
    
    std::vector<unsigned char> output_image;
    if (cpu)
    {
        Vec3 *framebuffer = cpu_raytrace(&scene);

        write_framebuffer_to_output_image(&scene, output_image, framebuffer);
        delete[] framebuffer;
    }

    if (gpu)
    {
        cudaFuncAttributes attribs;
        cudaFuncGetAttributes(&attribs, cuda_generate_initial_rays);
        cudaFuncGetAttributes(&attribs, cuda_process_rays);
        cudaFuncGetAttributes(&attribs, cuda_accumulate_rays);

        Vec3 *framebuffer = gpu_raytrace(&scene, sort);

        write_framebuffer_to_output_image(&scene, output_image, framebuffer);
        delete[] framebuffer;
    }

    stbi_write_png("raytracing.png", scene.width, output_image.size() / scene.width / 3, 3, &output_image.front(), scene.width * 3);

    return 0;
}