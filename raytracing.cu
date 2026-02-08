#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "common.cuh"
#include "math.cuh"
#include "scene.cuh"
#include "random.cuh"

__constant__ Scene cuda_scene;

#define MAX_RAYS_PER_PASS (1 << 8)
#define MAX_PIXELS_PER_PASS (1 << 18)

int ceil_divide(int numerator, int divisor)
{
    return (numerator + divisor - 1) / divisor;
}

float3 *gpu_raytrace(const Scene *scene)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    int pixel_count = scene->width * scene->height;
    float3 *cuda_framebuffer;
    CUDA_CHECK(cudaMalloc(&cuda_framebuffer, pixel_count * sizeof(float3)));
    cuda_scene.copy_from_cpu(*scene);
    cudaMemset(cuda_framebuffer, 0, pixel_count * sizeof(float3));

    int seed = 1;

    for (int y = 0; y < scene->height; y += MAX_TILE_SIZE)
    {
        int tile_height = min(scene->height - y, MAX_TILE_SIZE);
        for (int x = 0; x < scene->width; x += MAX_TILE_SIZE)
        {
            int tile_width = min(scene->width - x, MAX_TILE_SIZE);

            for (int ray_index = 0; ray_index < scene->ray_count; ray_index += MAX_RAYS_PER_PASS)
            {

                int pass_ray_count = min(MAX_RAYS_PER_PASS, scene->ray_count - ray_index);
                process_rays<<<dim3(tile_width, tile_height, 1), pass_ray_count>>>(cuda_framebuffer, x, y, seed);

                seed += MAX_TILE_SIZE * MAX_TILE_SIZE * pass_ray_count;
            }
        }
    }

    float3 *framebuffer = new float3[pixel_count];

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(framebuffer, cuda_framebuffer, pixel_count * sizeof(float3), cudaMemcpyDeviceToHost));

    cuda_scene.free_from_gpu();

    CUDA_CHECK(cudaFree(cuda_framebuffer));

    auto end_time = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration<float>(end_time - start_time).count();
    std::cout << "GPU Took " << gpu_time << "s\n";

    return framebuffer;
}

void write_framebuffer_to_output_image(Scene *scene, std::vector<unsigned char> &output_image, const float3 *framebuffer)
{
    for (int pixel_index = 0; pixel_index < scene->width * scene->height; pixel_index++)
    {
        auto pixel = (scene->exposure / scene->ray_count) * framebuffer[pixel_index];

        float r = pixel.x;
        float g = pixel.y;
        float b = pixel.z;

        // Convert HDR float with arbitrary range to 0-255 byte
        // x / (x + 1) does HDR to SDR tone mapping (this is a very basic way to do it)
        // Square root applies approximate linear -> sRGB conversion
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

    Scene scene = {};
    load_scene(&scene, argv[1]);

    std::vector<unsigned char> output_image;
    float3 *framebuffer = gpu_raytrace(&scene);
    write_framebuffer_to_output_image(&scene, output_image, framebuffer);
    delete[] framebuffer;

    stbi_write_png("raytracing.png", scene.width, output_image.size() / scene.width / 3, 3, &output_image.front(), scene.width * 3);

    return 0;
}