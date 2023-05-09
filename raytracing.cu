#define _USE_MATH_DEFINES
#include <cmath>
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

#define COMMON __host__ __device__

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
};

struct __align__(16) Sphere
{
    Vec3 center;
    float radius;
};

struct __align__(16) Triangle
{
    Vec3 p1;
    Vec3 p2p1;
    Vec3 p3p1;
    Vec3 normal;
};

COMMON Sphere load_sphere(Sphere *sphere)
{
#ifdef __CUDA_ARCH__
    Sphere result;

    float4 first = __ldg((float4 *) sphere);

    result.center.x = first.x;
    result.center.y = first.y;
    result.center.z = first.z;
    result.radius = first.w;

    return result;
#else
    return *sphere;
#endif
}

COMMON Triangle load_triangle(Triangle *triangle)
{
#ifdef __CUDA_ARCH__
    Triangle result;

    float4 first = __ldg((float4 *) triangle);

    result.p1.x = first.x;
    result.p1.y = first.y;
    result.p1.z = first.z;
    result.p2p1.x = first.w;

    float4 second = __ldg(((float4 *) triangle) + 1);
    
    result.p2p1.y = second.x;
    result.p2p1.z = second.y;
    result.p3p1.x = second.z;
    result.p3p1.y = second.w;

    float4 third = __ldg(((float4 *) triangle) + 2);

    result.p3p1.z = third.x;
    result.normal.x = third.y;
    result.normal.y = third.z;
    result.normal.z = third.w;

    return result;
#else
    return *triangle;
#endif
}

struct Ray
{
    Vec3 origin;
    Vec3 direction;
};

struct Material 
{
    Vec3 diffuse_albedo;
    float metallicity;
    Vec3 specular_albedo;
    float roughness;
    Vec3 emitted;   
    float index_of_refraction;
};

COMMON Material load_material(Material *material)
{
#ifdef __CUDA_ARCH__
    Material result;

    float4 first = __ldg((float4 *) material);

    result.diffuse_albedo.x = first.x;
    result.diffuse_albedo.y = first.y;
    result.diffuse_albedo.z = first.z;
    result.metallicity = first.w;

    float4 second = __ldg(((float4 *) material) + 1);
    
    result.specular_albedo.x = second.x;
    result.specular_albedo.y = second.y;
    result.specular_albedo.z = second.z;
    result.roughness = second.w;

    float4 third = __ldg(((float4 *) material) + 2);

    result.emitted.x = third.x;
    result.emitted.y = third.y;
    result.emitted.z = third.z;
    result.index_of_refraction = third.w;

    return result;
#else
    return *material;
#endif
}

COMMON float dot(const Vec3 &a, const Vec3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

COMMON Vec3 cross(const Vec3 &a, const Vec3 &b)
{
    return {
        a.y * b.z - a.z * b.y, 
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x, 
    };
}

COMMON float magnitude_squared(const Vec3 &vector)
{
    return vector.x * vector.x + vector.y * vector.y + vector.z * vector.z;
}

COMMON float magnitude(const Vec3 &vector)
{
    return sqrtf(magnitude_squared(vector));
}

COMMON Vec3 operator+(const Vec3 &a, const Vec3 &b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

COMMON Vec3 operator-(const Vec3 &a, const Vec3 &b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

COMMON Vec3 operator*(const Vec3 &a, const Vec3 &b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

COMMON Vec3 operator*(float scalar, const Vec3 &vector)
{
    return {scalar * vector.x, scalar * vector.y, scalar * vector.z};
}

COMMON Vec3 normalise(const Vec3 &vector)
{
    return (1.0f / magnitude(vector)) * vector;
}

float clamp01(float x)
{
    return x > 1 ? 1 : (x < 0 ? 0 : x);
}

Vec3 min(const Vec3 &a, const Vec3 &b)
{
    return {
        min(a.x, b.x), 
        min(a.y, b.y), 
        min(a.z, b.z), 
    };
}

Vec3 max(const Vec3 &a, const Vec3 &b)
{
    return {
        max(a.x, b.x), 
        max(a.y, b.y), 
        max(a.z, b.z), 
    };
}

struct xor_random
{
    unsigned long long state;
    unsigned long long inc;
};

COMMON unsigned int xor_rand(xor_random* rng)
{
    auto old_state = rng->state;

    rng->state = old_state * 6364136223846793005ULL + (rng->inc | 1);

    auto xor_shifted = (unsigned int) (((old_state >> 18) ^ old_state) >> 27);
    auto rot = (unsigned int) (old_state >> 59);

    return (xor_shifted >> rot) | (xor_shifted << ((-rot) & 31));
}

COMMON void xor_srand(xor_random* rng, unsigned int seed)
{
    rng->state = seed * 6839056345687307ULL;
    rng->inc = 820957824423429ULL;
    xor_rand(rng);
}

COMMON float random01(xor_random *rng)
{
    return xor_rand(rng) * (1.0f /  UINT_MAX);
}

COMMON float random02(xor_random *rng)
{
    return xor_rand(rng) * (2.0f /  UINT_MAX);
}

COMMON float random_radians(xor_random *rng)
{
    return xor_rand(rng) * (M_PI * 2 /  UINT_MAX);
}

float random_bidir(xor_random *rng)
{
    return random02(rng) - 1;
}

Vec3 random_in_sphere(xor_random *rng)
{
    Vec3 v;

    do {
        v = {random_bidir(rng), random_bidir(rng), random_bidir(rng)};
    } while (magnitude_squared(v) >  1);

    return v;
}

COMMON Vec3 random_on_sphere(xor_random *rng)
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

struct __align__(16) RayData
{
    Ray ray;
    Vec3 transmitted_color;
    Vec3 collected_color;
};

__device__ unsigned short interleave_5(unsigned short x)
{
    x = (x | (x << 8)) & 0x100F;
    x = (x | (x << 4)) & 0x10C3;
    x = (x | (x << 2)) & 0x1249;

    return x;
}

__device__ unsigned short morton_code(const Vec3 &vec)
{
    unsigned short x = (unsigned short) (vec.x * 31.99);
    unsigned short y = (unsigned short) (vec.y * 31.99);
    unsigned short z = (unsigned short) (vec.z * 31.99);

    return interleave_5(x) | (interleave_5(y) << 1) | (interleave_5(z) << 2);
}

struct Scene
{
    Sphere   *spheres;
    int sphere_count;

    Triangle *triangles;
    int triangle_count;

    Material *materials;

    int width;
    int height;

    Vec3 sky_color;
    Vec3 camera_position;
    Vec3 forward;
    Vec3 up;
    float vertical_fov;

    Vec3 min_coord;
    Vec3 inv_dimensions;

    Vec3 scaled_right;
    Vec3 scaled_up;

    Vec3 near_plane_top_left;

    float inv_width;
    float inv_height;

    void precompute_cammera_data()
    {
        Vec3 right = cross(up, forward);
    
        float near_plane_height = 2.0f * std::tan(vertical_fov * 0.5f);
        float near_plane_width  = near_plane_height * width / height;

        scaled_right = near_plane_width * right;
        scaled_up = near_plane_height * up;

        near_plane_top_left = forward - 0.5f * scaled_right + 0.5f * scaled_up;

        inv_width = 1.0f / (width - 1);
        inv_height = 1.0f / (height - 1);
    }

    COMMON void generate_initial_rays(RayData *ray_data, unsigned int *ray_indices, unsigned int *ray_keys, int rays_per_pixel, int ray_index, int seed) const
    {
        xor_random rng;
        xor_srand(&rng, ray_index * 298592570346 + 709579 * seed);

        int framebuffer_index = ray_index / rays_per_pixel;

        int x = framebuffer_index % width;
        int y = framebuffer_index / width;

        if (y < height)
        {
#ifdef __CUDA_ARCH__
                ray_indices[ray_index] = ray_index;
                ray_keys[ray_index] = 0;
#endif
            float x_clamped = (x + random01(&rng)) * inv_width;
            float y_clamped = (y + random01(&rng)) * inv_height;

            RayData ray;

            ray.ray = {camera_position, normalise(near_plane_top_left + x_clamped * scaled_right - y_clamped * scaled_up)};
            ray.transmitted_color = {1, 1, 1};
            ray.collected_color = {0, 0, 0};

            ray_data[ray_index] = ray;
        }
    }

    COMMON void process_ray(RayData *ray_data_ptr, unsigned int *ray_key, xor_random rng) const
    {
#ifdef __CUDA_ARCH__
        if (*ray_key == 0xFFFF'FFFF)
            return;
#else
        if (ray_data_ptr->transmitted_color.x == 0 && ray_data_ptr->transmitted_color.y == 0 && ray_data_ptr->transmitted_color.z == 0)
            return;
#endif

        RayData ray_data = *ray_data_ptr;

        float closest_hit_distance = INFINITY;
        int closest_hit_index = -1;

        const auto ray = ray_data.ray;

        for (int i = 0; i < sphere_count; i++)
        {
            const auto sphere = load_sphere(&spheres[i]);

            Vec3 offset = sphere.center - ray.origin;
            

            float minus_half_b = dot(offset, ray.direction);
            float quarter_c = magnitude_squared(offset) - sphere.radius * sphere.radius;

            float quarter_discriminant = minus_half_b * minus_half_b - quarter_c;

            if (quarter_discriminant < 0)
                continue;

            float half_square_root = sqrtf(quarter_discriminant);

            float hit_distance = minus_half_b - half_square_root;

            if (hit_distance < closest_hit_distance && hit_distance >= 0.005)
            {
                closest_hit_distance = hit_distance;
                closest_hit_index = i;
                continue;
            }

            hit_distance = minus_half_b + half_square_root;

            if (hit_distance < closest_hit_distance && hit_distance >= 0.005)
            {
                closest_hit_distance = hit_distance;
                closest_hit_index = i;
                continue;
            }
        }

        for (int i = 0; i < triangle_count; i++)
        {
            const auto triangle = load_triangle(&triangles[i]);

            Vec3 h = cross(ray.direction, triangle.p3p1);
            float perpendicular_component = dot(h, triangle.p2p1);

            float x = dot(triangle.normal, ray.direction);

            if (perpendicular_component == 0)
                continue;

            float inv_perpendicular_component = 1 / perpendicular_component;

            Vec3 offset = ray.origin - triangle.p1;
            float u = dot(offset, h) * inv_perpendicular_component;

            if (u < 0 || u > 1)
                continue;

            Vec3 q = cross(offset, triangle.p2p1);
            float v = dot(ray.direction, q) * inv_perpendicular_component;

            if (v < 0 || u + v > 1)
                continue;

            float hit_distance = dot(triangle.p3p1, q) * inv_perpendicular_component;

            if (hit_distance < 0.005 || hit_distance >= closest_hit_distance)
                continue;

            closest_hit_distance = hit_distance;
            closest_hit_index = sphere_count + i;
        }

        if (closest_hit_index == -1)
        {
            ray_data.collected_color += sky_color * ray_data.transmitted_color;
            ray_data.transmitted_color = {0, 0, 0};
        }
        else
        {
            const auto hit_point = ray.origin + closest_hit_distance * ray.direction;
            ray_data.ray.origin = hit_point;

            Vec3 normal;
            if (closest_hit_index < sphere_count)
            {
                const auto hit_sphere = load_sphere(&spheres[closest_hit_index]);
                normal = (1 / hit_sphere.radius) * (hit_point - hit_sphere.center);
            }
            else
            {
                const auto hit_triangle = load_triangle(&triangles[closest_hit_index - sphere_count]);
                normal = hit_triangle.normal;
            }

            const auto material = load_material(&materials[closest_hit_index]);
            ray_data.collected_color += material.emitted * ray_data.transmitted_color;
            

            bool front_face = dot(normal, ray.direction) < 0;

            if (!front_face)
            {
                normal = -normal;
            }

            Vec3 rough_normal = normalise(normal + material.roughness * random_on_sphere(&rng));
            float cos_theta = dot(rough_normal, ray.direction);



            if (material.index_of_refraction == 0)
            {
                if (random01(&rng) <= material.metallicity)
                {
                    ray_data.transmitted_color *= material.specular_albedo;
                    ray_data.ray.direction = ray.direction - 2 * cos_theta * rough_normal;
                }
                else
                {
                    ray_data.transmitted_color *= material.diffuse_albedo;
                    ray_data.ray.direction = normalise(normal + random_on_sphere(&rng)); 
                }
            }
            else
            {
                float ior = material.index_of_refraction;
                float inv_ior = 1 / ior;

                if (front_face)
                {
                    float temp = inv_ior;
                    inv_ior = ior;
                    ior = temp;
                }

                float sin_theta_squared = 1 - cos_theta * cos_theta;

                float r0 = (1 - ior) / (1 + ior);
                r0 *= r0;

                float cosine = 1 + cos_theta;
                float reflectance = r0 + (1 - r0) * cosine * cosine * cosine * cosine * cosine;

                if (sin_theta_squared > inv_ior * inv_ior || random01(&rng) < reflectance)
                {
                    ray_data.transmitted_color *= material.specular_albedo;
                    ray_data.ray.direction = ray.direction - 2 * cos_theta * rough_normal;                
                }
                else
                {
                    ray_data.transmitted_color *= material.diffuse_albedo;
                    
                    Vec3 r_out_perp = ior * (ray.direction - cos_theta * rough_normal);
                    Vec3 r_out_parallel = -sqrtf(1 - magnitude_squared(r_out_perp)) * rough_normal;
                    ray_data.ray.direction = normalise(r_out_parallel + r_out_perp);
                }
            }
        }


#ifdef __CUDA_ARCH__
        if (ray_data.transmitted_color.x == 0 && ray_data.transmitted_color.y == 0 && ray_data.transmitted_color.z == 0)
            *ray_key = 0xFFFF'FFFF;
        else
            *ray_key = ((unsigned int) morton_code((ray_data.ray.origin - min_coord) * inv_dimensions) << 16) | (unsigned int) morton_code(0.5 * (ray_data.ray.direction + Vec3{1, 1, 1}));
#endif
        *ray_data_ptr = ray_data;
    }
};

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
    }

    Scene scene = {};
    scene.width = 1920;
    scene.height = 1080;
    int ray_count = 1;
    int bounces = 3;

    std::vector<Sphere> spheres;
    std::vector<Triangle> triangles;
    std::vector<Material> materials;

    {
        std::ifstream scene_file(argv[1]);

        std::unordered_map<std::string, Material> materials_map;
        std::vector<Material> sphere_materials;
        std::vector<Material> triangle_materials;

        for (std::string line; std::getline(scene_file, line);)
        {
            if (line.empty())
                continue;

            std::istringstream tokens(line);

            std::string token;
            std::getline(tokens, token, ' ' );


            if (token == "sky")
            {
                tokens >> scene.sky_color.x;
                tokens >> scene.sky_color.y;
                tokens >> scene.sky_color.z;
            }
            else if (token == "camera")
            {
                std::getline(tokens, token, ' ' );

                tokens >> scene.camera_position.x;
                tokens >> scene.camera_position.y;
                tokens >> scene.camera_position.z;

                std::getline(tokens, token, ' ' );
                std::getline(tokens, token, ' ' );

                tokens >> scene.forward.x;
                tokens >> scene.forward.y;
                tokens >> scene.forward.z;
                scene.forward = normalise(scene.forward);

                std::getline(tokens, token, ' ' );
                std::getline(tokens, token, ' ' );

                tokens >> scene.up.x;
                tokens >> scene.up.y;
                tokens >> scene.up.z;
                scene.up = normalise(scene.up);

                std::getline(tokens, token, ' ' );
                std::getline(tokens, token, ' ' );

                tokens >> scene.vertical_fov;
                scene.vertical_fov = scene.vertical_fov * (M_PI / 180);
            }
            else if (token == "material")
            {
                std::getline(tokens, token, ' ' );

                Material &material = materials_map[token];
                material.specular_albedo = {1, 1, 1};
                material.diffuse_albedo = {1, 1, 1};
                material.emitted = {0, 0, 0};
                material.metallicity = 0;
                material.roughness = 0;
                material.index_of_refraction = 0;

                while (std::getline(tokens, token, ' ' ))
                {
                    if (token == "diffuse")
                    {
                        tokens >> material.diffuse_albedo.x;
                        tokens >> material.diffuse_albedo.y;
                        tokens >> material.diffuse_albedo.z;
                    }
                    else if (token == "specular")
                    {
                        tokens >> material.specular_albedo.x;
                        tokens >> material.specular_albedo.y;
                        tokens >> material.specular_albedo.z;
                    }
                    else if (token == "emit")
                    {
                        tokens >> material.emitted.x;
                        tokens >> material.emitted.y;
                        tokens >> material.emitted.z;
                    }
                    else if (token == "metallicity")
                    {
                        tokens >> material.metallicity;
                    }
                    else if (token == "roughness")
                    {
                        tokens >> material.roughness;
                    }
                    else if (token == "ior")
                    {
                        tokens >> material.index_of_refraction;
                    }
                }
            }
            else if (token == "sphere")
            {
                std::getline(tokens, token, ' ' );

                sphere_materials.push_back(materials_map.at(token));

                Sphere sphere;

                tokens >> sphere.center.x;
                tokens >> sphere.center.y;
                tokens >> sphere.center.z;
                tokens >> sphere.radius;

                spheres.push_back(sphere);
            }
            else if (token == "triangle")
            {
                std::getline(tokens, token, ' ' );

                triangle_materials.push_back(materials_map.at(token));

                Triangle triangle;

                tokens >> triangle.p1.x;
                tokens >> triangle.p1.y;
                tokens >> triangle.p1.z;

                tokens >> triangle.p2p1.x;
                tokens >> triangle.p2p1.y;
                tokens >> triangle.p2p1.z;

                tokens >> triangle.p3p1.x;
                tokens >> triangle.p3p1.y;
                tokens >> triangle.p3p1.z;

                triangle.p2p1 -= triangle.p1;
                triangle.p3p1 -= triangle.p1;
                triangle.normal = normalise(cross(triangle.p3p1, triangle.p2p1));

                triangles.push_back(triangle);
            }
            else if (token == "quad")
            {
                std::getline(tokens, token, ' ' );

                triangle_materials.push_back(materials_map.at(token));
                triangle_materials.push_back(materials_map.at(token));

                Vec3 p1, p2, p3, p4;

                tokens >> p1.x;
                tokens >> p1.y;
                tokens >> p1.z;

                tokens >> p2.x;
                tokens >> p2.y;
                tokens >> p2.z;

                tokens >> p3.x;
                tokens >> p3.y;
                tokens >> p3.z;

                tokens >> p4.x;
                tokens >> p4.y;
                tokens >> p4.z;

                Triangle triangle;

                triangle.p1 = p1;
                triangle.p2p1 = p2 - p1;
                triangle.p3p1 = p3 - p1;

                triangle.normal = normalise(cross(triangle.p3p1, triangle.p2p1));

                triangles.push_back(triangle);

                triangle.p1 = p1;
                triangle.p2p1 = p3 - p1;
                triangle.p3p1 = p4 - p1;
                
                triangle.normal = normalise(cross(triangle.p3p1, triangle.p2p1));

                triangles.push_back(triangle);
            }
            else if (token == "image")
            {
                tokens >> scene.width;
                tokens >> scene.height;
                tokens >> ray_count;
                tokens >> bounces;
            }
        }

        materials = std::move(sphere_materials);
        materials.insert(materials.end(), triangle_materials.begin(), triangle_materials.end());
    }
    /*
    scene.vertical_fov = 90 * M_PI / 180;
    scene.camera_position = {0, 2, -2};
    scene.forward = normalise(Vec3{0, -0.5, 1});
    scene.up = {0, 1, 0};

    scene.sky_color = {0.2, 0.4, 0.9};

    std::vector<Sphere> spheres;
    std::vector<Triangle> triangles;
    std::vector<Material> materials;

    spheres.push_back({{3, 2, 4}, 3});
    spheres.push_back({{-3, 2, 4}, 3});
    spheres.push_back({{0, -10001, 2}, 10000});
    spheres.push_back({{200000, 200000, 300000}, 40000});

    Material sphere_material = {};
    sphere_material.diffuse_albedo = {0.7, 0.7, 0.7};
    sphere_material.specular_albedo = {1, 1, 1};
    sphere_material.metallicity = 0.05;

    Material glass_material = {};
    glass_material.diffuse_albedo = {1, 1, 1};
    glass_material.specular_albedo = {1, 1, 1};
    glass_material.index_of_refraction = 1.5;
    glass_material.roughness = 0.1;

    Material ground_material = {};
    ground_material.diffuse_albedo = {0.7, 0.6, 0.2};

    Material sun_material = {};
    sun_material.emitted = {40, 40, 40};

    materials.push_back(sphere_material);
    materials.push_back(glass_material);
    materials.push_back(ground_material);
    materials.push_back(sun_material);
    */

    scene.sphere_count = spheres.size();
    scene.spheres = &spheres.front();
    scene.triangle_count = triangles.size();
    scene.triangles = &triangles.front();
    scene.materials = &materials.front();

    Vec3 scene_max_coord = {-INFINITY, -INFINITY, -INFINITY};
    scene.min_coord = {INFINITY, INFINITY, INFINITY};

    for (const auto &sphere : spheres)
    {
        scene_max_coord = max(scene_max_coord, sphere.center + Vec3{sphere.radius, sphere.radius, sphere.radius});
        scene.min_coord = min(scene.min_coord, sphere.center - Vec3{sphere.radius, sphere.radius, sphere.radius});
    }

    for (const auto &triangle : triangles)
    {
        scene_max_coord = max(scene_max_coord, triangle.p1);
        scene_max_coord = max(scene_max_coord, triangle.p1 + triangle.p2p1);
        scene_max_coord = max(scene_max_coord, triangle.p1 + triangle.p3p1);
        scene.min_coord = min(scene.min_coord, triangle.p1);
        scene.min_coord = min(scene.min_coord, triangle.p1 + triangle.p2p1);
        scene.min_coord = min(scene.min_coord, triangle.p1 + triangle.p3p1);
    }

    scene_max_coord -= scene.min_coord;

    scene.inv_dimensions = {1 / scene_max_coord.x, 1 / scene_max_coord.y, 1 / scene_max_coord.z};
    scene.precompute_cammera_data();

    std::vector<Vec3> framebuffer{(size_t) (scene.width * scene.height)};    
    std::vector<RayData> ray_data((size_t) scene.width * (size_t) scene.height * (size_t) MAX_RAYS_PER_PIXEL_PER_PASS);

    auto start_time = std::chrono::high_resolution_clock::now();
    decltype(start_time) end_time;
    int remaining_rays = ray_count;

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

            for (int i = 0; i < bounces; i++)
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
            float r = pixel.x / ray_count;
            float g = pixel.y / ray_count;
            float b = pixel.z / ray_count;

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

    CUDA_CHECK(cudaMemcpyToSymbolAsync(cuda_scene, &scene_copy, sizeof(Scene), 0, cudaMemcpyHostToDevice, scene_copy_stream));
    cudaEventRecord(scene_copy_done, scene_copy_stream);

    cudaMemsetAsync(cuda_framebuffer, 0, framebuffer.size() * sizeof(Vec3), framebuffer_stream);
    cudaEventRecord(framebuffer_done, framebuffer_stream);

    remaining_rays = ray_count;


    while (remaining_rays)
    {
        int rays_to_cast = min(remaining_rays, MAX_RAYS_PER_PIXEL_PER_PASS);
        remaining_rays -= rays_to_cast;

        int total_rays = rays_to_cast * scene.width * scene.height;
        cuda_generate_initial_rays<<<(total_rays + 127) / 128, 128 >>>(cuda_ray_data, cuda_ray_indices[0], cuda_ray_keys[0], rays_to_cast, remaining_rays);

        
        for (int i = 0; i < bounces; i++)
        {
            cudaStreamWaitEvent(0, scene_copy_done);
            cuda_process_rays<<<(total_rays + 127) / 128, 128>>>(cuda_ray_data, cuda_ray_indices[0], cuda_ray_keys[0], total_rays, remaining_rays * MAX_RAYS_PER_PIXEL_PER_PASS + i);

            if (sort && i + 1 != bounces)
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
        float r = pixel.x / ray_count;
        float g = pixel.y / ray_count;
        float b = pixel.z / ray_count;

        output_image.push_back((unsigned char) (sqrtf(r / (r + 1)) * 255.999f));
        output_image.push_back((unsigned char) (sqrtf(g / (g + 1)) * 255.999f));
        output_image.push_back((unsigned char) (sqrtf(b / (b + 1)) * 255.999f));
    }


    stbi_write_png("raytracing.png", scene.width, output_image.size() / scene.width / 3, 3, &output_image.front(), scene.width * 3);
    system("start raytracing.png");

    return 0;
}