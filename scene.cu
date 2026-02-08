#include "scene.cuh"

#include <unordered_map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <immintrin.h>

#define MAX_BVH_DEPTH 30

// By using this we promise CUDA that the value we are reading will never be written by a kernel
// This allows the data to be loaded into L1 cache which is not coherent
template<typename T> __device__ T load_read_only(T *t)
{
    // return *t;
    static_assert(alignof(T) >= sizeof(float4), "load_read_only requires 16 byte alignment");

    constexpr int count = sizeof(T) / sizeof(float4);

    union Dummy{
        float4 dummy[count];
        T value;

        __device__ Dummy() {}
    };

    Dummy dummy;

    #pragma unroll
    for (int i = 0; i < count; i++)
    {
        dummy.dummy[i] = __ldg(((float4 *) t) + i);
    }

    return dummy.value;
}

void Scene::precompute_camera_data()
{
    float3 right = cross(up, forward);

    float near_plane_height = 2.0f * std::tan(vertical_fov * 0.5f);
    float near_plane_width  = near_plane_height * width / height;

    scaled_right = near_plane_width * right;
    scaled_up = near_plane_height * up;

    near_plane_top_left = forward - 0.5f * scaled_right + 0.5f * scaled_up;

    inv_width = 1.0f / (width - 1);
    inv_height = 1.0f / (height - 1);
}

// Branchless ray AABB intersection from https://tavianator.com/2022/ray_box_boundary.html
// (assuming there is hardware min/max which is true on all modern GPUs and CPUs)
COMMON bool ray_aabb_intersection(const Aabb &aabb, const Ray &ray, const float3 &n_inv, float &tmin, float tmax)
{
    tmin = 0.0f;

    float t1 = (aabb.min_bound.x - ray.origin.x) * n_inv.x;
    float t2 = (aabb.max_bound.x - ray.origin.x) * n_inv.x;

    tmin = min(max(t1, tmin), max(t2, tmin));
    tmax = max(min(t1, tmax), min(t2, tmax));

    t1 = (aabb.min_bound.y - ray.origin.y) * n_inv.y;
    t2 = (aabb.max_bound.y - ray.origin.y) * n_inv.y;

    tmin = min(max(t1, tmin), max(t2, tmin));
    tmax = max(min(t1, tmax), min(t2, tmax));

    t1 = (aabb.min_bound.z - ray.origin.z) * n_inv.z;
    t2 = (aabb.max_bound.z - ray.origin.z) * n_inv.z;

    tmin = min(max(t1, tmin), max(t2, tmin));
    tmax = max(min(t1, tmax), min(t2, tmax));

    return tmin <= tmax;
}

__device__ void bvh_closest_hit_distance(const Ray &ray, float &closest_hit_distance, int &closest_hit_index)
{
    extern __constant__ Scene cuda_scene;
    float3 n_inv = {1 / ray.direction.x, 1 / ray.direction.y, 1 / ray.direction.z};

    unsigned int node_index_stack[MAX_BVH_DEPTH + 1];
    float node_distance_stack[MAX_BVH_DEPTH + 1];
    int stack_count = 1;

    node_index_stack[0] = 0;
    node_distance_stack[0] = 0;

    while (stack_count)
    {
        stack_count--;
        float distance = node_distance_stack[stack_count];

        if (distance >= closest_hit_distance)
        {
            continue;
        }

        BvhNode node = cuda_scene.bvh[node_index_stack[stack_count]];


        if (node.is_leaf())
        {
            for (int i = node.child2; i < node.child1; i++)
            {
                const auto triangle = cuda_scene.triangles[i];

                // Möller–Trumbore ray-triangle intersection algorithm
                // Based on https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
                float3 h = cross(ray.direction, triangle.p3p1);
                float perpendicular_component = dot(h, triangle.p2p1);

                if (perpendicular_component == 0)
                    continue;

                float3 offset = ray.origin - triangle.p1;
                float u = dot(offset, h) * perpendicular_component;

                float3 q = cross(offset, triangle.p2p1);
                float v = dot(ray.direction, q) * perpendicular_component;

                if ((v <= 0) | (u + v > perpendicular_component * perpendicular_component) | (u <= 0)) [[likely]]
                    continue;

                float hit_distance = dot(triangle.p3p1, q) / perpendicular_component;

                if (hit_distance < 0.005 || hit_distance >= closest_hit_distance)
                    continue;

                closest_hit_distance = hit_distance;
                closest_hit_index = i;
            }
        }
        else
        {
            float hit1_distance, hit2_distance;

            bool hit1 = ray_aabb_intersection(cuda_scene.bvh[node.child1].aabb, ray, n_inv, hit1_distance, closest_hit_distance);
            bool hit2 = ray_aabb_intersection(cuda_scene.bvh[node.child2].aabb, ray, n_inv, hit2_distance, closest_hit_distance);

            if (hit1_distance < hit2_distance)
            {
                if (hit1)
                {
                    node_index_stack[stack_count] = node.child1;
                    node_distance_stack[stack_count] = hit1_distance;
                    stack_count++;
                }

                if (hit2)
                {
                    node_index_stack[stack_count] = node.child2;
                    node_distance_stack[stack_count] = hit2_distance;
                    stack_count++;
                }
            }
            else
            {
                if (hit2)
                {
                    node_index_stack[stack_count] = node.child2;
                    node_distance_stack[stack_count] = hit2_distance;
                    stack_count++;
                }

                if (hit1)
                {
                    node_index_stack[stack_count] = node.child1;
                    node_distance_stack[stack_count] = hit1_distance;
                    stack_count++;
                }
            }
        }
    }
}

void Scene::copy_from_cpu(const Scene &scene)
{
    Scene scene_copy = scene;

    int environment_map_size = scene.environment_map_width * scene.environment_map_height;

    CUDA_CHECK(cudaMalloc(&scene_copy.triangles,        scene.triangle_count * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&scene_copy.materials,        scene.material_count * sizeof(Material)));
    CUDA_CHECK(cudaMalloc(&scene_copy.material_indices, scene.triangle_count * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&scene_copy.bvh,              scene.bvh_node_count * sizeof(BvhNode)));
    CUDA_CHECK(cudaMalloc(&scene_copy.environment_map,  environment_map_size * sizeof(float3)));

    CUDA_CHECK(cudaMemcpy(scene_copy.triangles,        scene.triangles,        sizeof(Triangle) * scene.triangle_count, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scene_copy.materials,        scene.materials,        sizeof(Material) * scene.material_count, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scene_copy.material_indices, scene.material_indices, sizeof(uint16_t) * scene.triangle_count, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scene_copy.bvh,              scene.bvh,              sizeof(BvhNode)  * scene.bvh_node_count, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scene_copy.environment_map,  scene.environment_map,  sizeof(float3)     * environment_map_size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpyToSymbol(*this, &scene_copy, sizeof(Scene)));
}

void Scene::free_from_gpu()
{
    Scene scene_copy;
    CUDA_CHECK(cudaMemcpyFromSymbol(&scene_copy, *this, sizeof(Scene)));

    CUDA_CHECK(cudaFree(scene_copy.environment_map));
    CUDA_CHECK(cudaFree(scene_copy.material_indices));
    CUDA_CHECK(cudaFree(scene_copy.materials));
    CUDA_CHECK(cudaFree(scene_copy.bvh));
    CUDA_CHECK(cudaFree(scene_copy.triangles));
}


// Maybe it makes sense to pre-convert to a cubemap instead of doing this every time a ray misses
// Our test environment map is in a format used by the PBRTv4 ray tracer
// This code is based on https://github.com/mmp/pbrt-v4/blob/c4baa534042e2ec4eb245924efbcef477e096389/src/pbrt/util/math.cpp#L317
__device__ float3 equal_area_project_sphere_to_square(const float3 &direction)
{
    float x = abs(direction.x);
    float y = abs(direction.y);
    float z = abs(direction.z);

    float r = sqrt(1 - min(z, 1.0f));

    float a = max(x, y);
    float b = min(x, y);

    b = a == 0 ? 0 : b / a;

    float phi = (2 / M_PI) * atan(b);

    if (x < y)
    {
        phi = 1 - phi;
    }

    float v = phi * r;
    float u = r - v;

    if (direction.z < 0)
    {
        float old_v = v;
        v = 1 - u;
        u = 1 - old_v;
    }

    u = copysign(u, direction.x);
    v = copysign(v, direction.y);

    return {(u + 1) * 0.5f, (v + 1) * 0.5f, 0};
}

__global__ void process_rays(float3* framebuffer, int start_x, int start_y, int seed)
{
    extern __constant__ Scene cuda_scene;
    xor_random rng;
    int ray_index = threadIdx.x;
    int pixel_index = blockIdx.y * gridDim.x + blockIdx.x;
    xor_srand(&rng, seed + pixel_index * cuda_scene.ray_count + ray_index);

    int x = start_x + blockIdx.x;
    int y = start_y + blockIdx.y;

    int framebuffer_index = y * cuda_scene.width + x;

    float3 collected_color = {0, 0, 0};
    float3 transmitted_color = {1, 1, 1};

    float x_clamped = (x + random01(&rng)) * cuda_scene.inv_width;
    float y_clamped = (y + random01(&rng)) * cuda_scene.inv_height;
    Ray ray = {cuda_scene.camera_position, normalise(cuda_scene.near_plane_top_left + x_clamped * cuda_scene.scaled_right - y_clamped * cuda_scene.scaled_up)};

    for (int i = 0; i < cuda_scene.bounces; i++) {
        float closest_hit_distance = 1e30;

        int closest_hit_index = -1;

        bvh_closest_hit_distance(ray, closest_hit_distance, closest_hit_index);

        if (closest_hit_index == -1)
        {
            // Environment map in our test data is rotated and has y and z axes flipped,
            // apply a hardcoded transformation for now.
            float dir_x = ray.direction.x * -0.386527 + ray.direction.z * 0.922278;
            float dir_y = ray.direction.x * -0.922278 + ray.direction.z * -0.386527;
            float dir_z = ray.direction.y;

            float3 coords = equal_area_project_sphere_to_square({dir_x, dir_y, dir_z});
            float x = coords.x;
            float y = coords.y;

            // Nearest filtering
            int texel_x = (int) (clamp01(x) * (cuda_scene.environment_map_width  - 1) + 0.5);
            int texel_y = (int) (clamp01(y) * (cuda_scene.environment_map_height - 1) + 0.5);
            float3 sky_color = cuda_scene.environment_map[texel_y * cuda_scene.environment_map_height + texel_x];

            collected_color += sky_color * transmitted_color;
            transmitted_color = {0, 0, 0};
            break;
        }
        else
        {
            const auto hit_point = ray.origin + closest_hit_distance * ray.direction;
            ray.origin = hit_point;

            float3 normal;
            const auto hit_triangle = cuda_scene.triangles[closest_hit_index];
            normal = hit_triangle.normal;

            const auto material = cuda_scene.materials[cuda_scene.material_indices[closest_hit_index]];

            collected_color += material.emitted * transmitted_color;


            bool front_face = dot(normal, ray.direction) < 0;

            if (!front_face)
            {
                normal = -normal;
            }

            float3 rough_normal = normalise(normal + material.roughness * random_on_sphere(&rng));
            float cos_theta = dot(rough_normal, ray.direction);



            if (material.index_of_refraction == 0)
            {
                if (random01(&rng) <= material.metallicity)
                {
                    transmitted_color *= material.specular_albedo;
                    ray.direction = ray.direction - 2 * cos_theta * rough_normal;
                }
                else
                {
                    transmitted_color *= material.diffuse_albedo;
                    ray.direction = normalise(normal + random_on_sphere(&rng));
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
                    transmitted_color *= material.specular_albedo;
                    ray.direction = ray.direction - 2 * cos_theta * rough_normal;
                }
                else
                {
                    transmitted_color *= material.diffuse_albedo;

                    float3 r_out_perp = ior * (ray.direction - cos_theta * rough_normal);
                    float3 r_out_parallel = -sqrtf(1 - magnitude_squared(r_out_perp)) * rough_normal;
                    ray.direction = normalise(r_out_parallel + r_out_perp);
                }
            }
        }
    }

    const auto group = cooperative_groups::this_thread_block();
    const auto tile = cooperative_groups::tiled_partition<32>(group);

    cooperative_groups::reduce_update_async(tile, cuda::atomic_ref<float>(framebuffer[framebuffer_index].x), collected_color.x, cooperative_groups::plus<float>());
    cooperative_groups::reduce_update_async(tile, cuda::atomic_ref<float>(framebuffer[framebuffer_index].y), collected_color.y, cooperative_groups::plus<float>());
    cooperative_groups::reduce_update_async(tile, cuda::atomic_ref<float>(framebuffer[framebuffer_index].z), collected_color.z, cooperative_groups::plus<float>());
}

// Extremely hacky ply file loader for exactly the ply format we have
// will not work for most files
void load_ply(std::vector<Triangle> &triangles, const std::string &filename)
{
    std::ifstream ply_file(filename, std::ios_base::binary);

    std::string line;
    std::getline(ply_file, line);
    std::getline(ply_file, line);
    std::getline(ply_file, line);

    auto vertex_count = std::stoi(line.substr(15));

    std::getline(ply_file, line);
    std::getline(ply_file, line);
    std::getline(ply_file, line);
    std::getline(ply_file, line);
    std::getline(ply_file, line);
    std::getline(ply_file, line);
    std::getline(ply_file, line);
    std::getline(ply_file, line);
    std::getline(ply_file, line);

    auto face_count = std::stoi(line.substr(13));

    std::getline(ply_file, line);
    std::getline(ply_file, line);

    struct Vertex {
        float3 position;
        float3 normal;
        float u, v;
    };

    std::vector<Vertex> vertices;
    vertices.resize(vertex_count);
    ply_file.read(reinterpret_cast<char *>(&vertices.front()), sizeof(Vertex) * vertices.size());

    std::vector<int> indices;

    for (int i = 0; i < face_count; i++)
    {
        indices.resize(ply_file.get());
        ply_file.read(reinterpret_cast<char *>(&indices.front()), sizeof(int) * indices.size());

        for (int j = 2; j < indices.size(); j++)
        {
            Triangle triangle;

            triangle.p1 = vertices[indices[0]].position;
            triangle.p2p1 = vertices[indices[j - 1]].position;
            triangle.p3p1 = vertices[indices[j]].position;
            triangle.normal = normalise(cross(triangle.p3p1 - triangle.p1, triangle.p2p1 - triangle.p1));

            triangles.push_back(triangle);
        }
    }
}

float3 *load_pfm(const std::string &filename, int *width, int *height)
{
    std::ifstream file(filename, std::ios_base::binary);

    std::string line;
    std::getline(file, line);
    std::getline(file, line);

    std::stringstream ss(line);

    ss >> *width;
    ss >> *height;

    std::getline(file, line);

    float3 *image = new float3[*width * *height];
    file.read((char *) image, sizeof(float3) * *width * *height);

    return image;
}

void load_scene(Scene *scene, const char *filename)
{
    scene->width = 1920;
    scene->height = 1080;
    scene->ray_count = 1;
    scene->bounces = 3;

    std::vector<Triangle> triangles;

    std::ifstream scene_file(filename);

    std::unordered_map<std::string, uint16_t> materials_map;
    std::vector<Material> materials;
    std::vector<uint16_t> triangle_materials;

    for (std::string line; std::getline(scene_file, line);)
    {
        if (line.empty())
            continue;

        std::istringstream tokens(line);

        std::string token;
        std::getline(tokens, token, ' ' );


        if (token == "sky")
        {
            float r, g, b;

            tokens >> r;
            tokens >> g;
            tokens >> b;

            scene->environment_map = new float3{r, g, b};
            scene->environment_map_width = 1;
            scene->environment_map_height = 1;
        }
        else if (token == "sky_map")
        {
            std::getline(tokens, token, ' ' );

            scene->environment_map = load_pfm(token, &scene->environment_map_width, &scene->environment_map_height);
            std::cout << "Loaded environment map with size " << scene->environment_map_width << "," << scene->environment_map_height << "\n";

        }
        else if (token == "camera")
        {
            std::getline(tokens, token, ' ' );

            tokens >> scene->camera_position.x;
            tokens >> scene->camera_position.y;
            tokens >> scene->camera_position.z;

            std::getline(tokens, token, ' ' );
            std::getline(tokens, token, ' ' );

            tokens >> scene->forward.x;
            tokens >> scene->forward.y;
            tokens >> scene->forward.z;
            scene->forward = normalise(scene->forward);

            std::getline(tokens, token, ' ' );
            std::getline(tokens, token, ' ' );

            tokens >> scene->up.x;
            tokens >> scene->up.y;
            tokens >> scene->up.z;
            scene->up = normalise(scene->up);

            std::getline(tokens, token, ' ' );
            std::getline(tokens, token, ' ' );

            tokens >> scene->vertical_fov;
            scene->vertical_fov = scene->vertical_fov * (M_PI / 180);
        }
        else if (token == "material")
        {
            std::getline(tokens, token, ' ' );

            materials_map[token] = (uint16_t) materials.size();

            Material material;
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

            materials.push_back(material);
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

            triangle.normal = normalise(cross(triangle.p3p1 - triangle.p1, triangle.p2p1 - triangle.p1));

            triangles.push_back(triangle);
        }
        else if (token == "quad")
        {
            std::getline(tokens, token, ' ' );

            triangle_materials.push_back(materials_map.at(token));
            triangle_materials.push_back(materials_map.at(token));

            float3 p1, p2, p3, p4;

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
            triangle.p2p1 = p2;
            triangle.p3p1 = p3;
            triangle.normal = normalise(cross(triangle.p3p1 - triangle.p1, triangle.p2p1 - triangle.p1));

            triangles.push_back(triangle);

            triangle.p1 = p1;
            triangle.p2p1 = p3;
            triangle.p3p1 = p4;
            triangle.normal = normalise(cross(triangle.p3p1 - triangle.p1, triangle.p2p1 - triangle.p1));

            triangles.push_back(triangle);
        }
        else if (token == "ply")
        {
            std::getline(tokens, token, ' ' );

            const auto material = materials_map.at(token);

            size_t triangle_count = triangles.size();

            std::getline(tokens, token, ' ' );
            load_ply(triangles, token);

            for (; triangle_count < triangles.size(); triangle_count++)
            {
                triangle_materials.push_back(material);
            }
        }
        else if (token == "image")
        {
            tokens >> scene->width;
            tokens >> scene->height;
            tokens >> scene->ray_count;
            tokens >> scene->bounces;
            tokens >> scene->exposure;
        }
    }

    scene->triangle_count = triangles.size();
    scene->triangles = new Triangle[triangles.size()];
    std::copy(triangles.begin(), triangles.end(), scene->triangles);

    scene->material_indices = new uint16_t[triangle_materials.size()];
    std::copy(triangle_materials.begin(), triangle_materials.end(), scene->material_indices);

    scene->materials = new Material[materials.size()];
    std::copy(materials.begin(), materials.end(), scene->materials);
    scene->material_count = (uint16_t) materials.size();

    scene->precompute_camera_data();
    scene->generate_bvh(MAX_BVH_DEPTH);

    scene->min_coord = scene->bvh[0].aabb.min_bound;
    float3 scene_max_coord = scene->bvh[0].aabb.max_bound;
    scene->inv_dimensions = {1 / scene_max_coord.x, 1 / scene_max_coord.y, 1 / scene_max_coord.z};
}

void Aabb::expand(const float3 &other)
{
    min_bound = min(min_bound, other);
    max_bound = max(max_bound, other);
}

void Aabb::expand(const Triangle &other)
{
    expand(other.p1);
    expand(other.p2p1);
    expand(other.p3p1);
}

void Aabb::expand(const Aabb &other)
{
    min_bound = min(min_bound, other.min_bound);
    max_bound = max(max_bound, other.max_bound);
}

float Aabb::half_area() const
{
    float3 size = max_bound - min_bound;

    return size.x * size.y + size.x * size.z + size.y * size.z;
}

__device__ bool BvhNode::is_leaf() const
{
    return child2 <= child1;
}

struct BuilderAabb
{
    float4 min = {1e30, 1e30, 1e30, 0};
    float4 max = {-1e30, -1e30, -1e30, 0};

    void expand(const float3& v)
    {
        _mm_storeu_ps(&min.x, _mm_min_ps(_mm_loadu_ps(&min.x), _mm_loadu_ps(&v.x)));
        _mm_storeu_ps(&max.x, _mm_max_ps(_mm_loadu_ps(&max.x), _mm_loadu_ps(&v.x)));
    }

    void expand(const Triangle& triangle)
    {
        expand(triangle.p1);
        expand(triangle.p2p1);
        expand(triangle.p3p1);
    }

    void expand(const BuilderAabb& aabb)
    {
        _mm_storeu_ps(&min.x, _mm_min_ps(_mm_loadu_ps(&min.x), _mm_loadu_ps(&aabb.min.x)));
        _mm_storeu_ps(&max.x, _mm_max_ps(_mm_loadu_ps(&max.x), _mm_loadu_ps(&aabb.max.x)));
    }

    float half_area() const
    {
        __m128 size = _mm_sub_ps(_mm_loadu_ps(&max.x), _mm_loadu_ps(&min.x));

        __m128 axis_swizzle = _mm_permute_ps(size, 0b11001001);
        __m128 areas = _mm_mul_ps(size, axis_swizzle);

        return areas.m128_f32[0] + areas.m128_f32[1] + areas.m128_f32[2];
    }

    Aabb build() const
    {
        return {
            {min.x, min.y, min.z},
            {max.x, max.y, max.z},
        };
    }
};

struct BuilderBvhNode
{
    BuilderAabb aabb;
    int child1;
    int child2;

    BvhNode build()
    {
        return {
            aabb.build(),
            child1,
            child2,
        };
    }
};

struct BuilderTriangle
{
    float3 centroid;
    BuilderAabb aabb;
};

void Scene::generate_bvh(int max_depth)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    decltype(start_time) end_time;

    std::vector<BuilderBvhNode> bvh_nodes;
    bvh_nodes.reserve(triangle_count);

    auto &root = bvh_nodes.emplace_back();

    root.child2 = 0;
    root.child1 = triangle_count;

    struct NodeToSplit
    {
        int index;
        int depth_remaining;
    };

    std::vector<NodeToSplit> nodes_to_split;
    nodes_to_split.push_back({0, max_depth});

    std::vector<int> triangle_indices;
    std::vector<BuilderTriangle> builder_triangles;
    triangle_indices.resize(triangle_count);
    builder_triangles.resize(triangle_count);

    for (int i = 0; i < triangle_count; i++)
    {
        triangle_indices[i] = i;
        builder_triangles[i].aabb.expand(triangles[i]);
        builder_triangles[i].centroid = 0.5f * (builder_triangles[i].aabb.build().min_bound, builder_triangles[i].aabb.build().max_bound);
    }

    // Binned surface area heuristic BVH computation
    // Based on https://jacco.ompf2.com/2022/04/21/how-to-build-a-bvh-part-3-quick-builds/
    while (!nodes_to_split.empty())
    {
        const auto [node_index, depth_remaining] = nodes_to_split.back();
        nodes_to_split.pop_back();

        auto& node = bvh_nodes[node_index];

        __m128 v_min_centroid = _mm_set1_ps(1e30);
        __m128 v_max_centroid = _mm_set1_ps(-1e30);

        for (int i = node.child2; i < node.child1; i++)
        {
            node.aabb.expand(builder_triangles[triangle_indices[i]].aabb);
            const auto v_centroid = _mm_loadu_ps(&builder_triangles[triangle_indices[i]].centroid.x);
            v_min_centroid = _mm_min_ps(v_min_centroid, v_centroid);
            v_max_centroid = _mm_max_ps(v_max_centroid, v_centroid);
        }

        int our_count = node.child1 - node.child2;

        if (our_count <= 4 || depth_remaining == 0)
        {
            continue;
        }

        float our_cost = node.aabb.half_area() * our_count;

        struct Bin
        {
            BuilderAabb aabb;
            int triangle_count = 0;
        };

        constexpr int BINS = 64;

        int best_axis;
        float best_position;
        float best_cost = our_cost;

        float3 min_centroids{v_min_centroid.m128_f32[0], v_min_centroid.m128_f32[1], v_min_centroid.m128_f32[2]};
        float3 max_centroids{v_max_centroid.m128_f32[0], v_max_centroid.m128_f32[1], v_max_centroid.m128_f32[2]};

        for (int axis = 0; axis < 3; axis++)
        {
            float min_centroid = elem(min_centroids, axis);
            float max_centroid = elem(max_centroids, axis);

            if (min_centroid == max_centroid)
                continue;

            float scale = BINS / (max_centroid - min_centroid);

            Bin bins[BINS];

            for (int i = node.child2; i < node.child1; i++)
            {
                const auto &triangle = builder_triangles[triangle_indices[i]];
                auto &bin =  bins[std::min(BINS - 1, (int) ((elem(triangle.centroid, axis) - min_centroid) * scale))];

                bin.triangle_count++;

                bin.aabb.expand(triangle.aabb);
            }

            float left_area[BINS - 1], right_area[BINS - 1];
            int left_count[BINS - 1];
            int left_sum = 0;

            BuilderAabb left_box, right_box;

            for (int i = 0; i + 1 < BINS; i++)
            {
                left_sum += bins[i].triangle_count;
                left_count[i] = left_sum;
                left_box.expand(bins[i].aabb);
                left_area[i] = left_box.half_area();

                right_box.expand(bins[BINS - 1 - i].aabb);
                right_area[BINS - 2 - i] = right_box.half_area();
            }

            scale = (max_centroid - min_centroid) / BINS;

            for (int i = 0; i + 1 < BINS; i++)
            {
                float plane_cost = left_count[i] * left_area[i] + (our_count - left_count[i]) * right_area[i];

                if (plane_cost != 0 && plane_cost < best_cost)
                {
                    best_axis = axis;
                    best_position = min_centroid + scale * (i + 1);
                    best_cost = plane_cost;
                }
            }
        }

        if (best_cost >= our_cost)
        {
            continue;
        }

        int i = node.child2;
        int j = node.child1 - 1;

        while (i <= j)
        {
            if (elem(builder_triangles[triangle_indices[i]].centroid, best_axis) < best_position)
            {
                i++;
            }
            else
            {
                std::swap(triangle_indices[i], triangle_indices[j]);
                j--;
            }
        }

        if (i == node.child1 || i == node.child2)
        {
            continue;
        }

        int left_child_index = bvh_nodes.size();
        auto &left_child = bvh_nodes.emplace_back();

        int right_child_index = bvh_nodes.size();
        auto &right_child = bvh_nodes.emplace_back();

        left_child.child2 = node.child2;
        left_child.child1 = i;
        right_child.child2 = i;
        right_child.child1 = node.child1;

        nodes_to_split.push_back({left_child_index, depth_remaining - 1});
        nodes_to_split.push_back({right_child_index, depth_remaining - 1});

        node.child1 = left_child_index;
        node.child2 = right_child_index;
    }

    Triangle* new_triangles = new Triangle[triangle_count];
    uint16_t* new_material_indices = new uint16_t[triangle_count];

    for (int i = 0; i < triangle_count; i++)
    {
        const auto &triangle = triangles[triangle_indices[i]];
        auto& new_triangle = new_triangles[i];
        new_triangle.p1 = triangle.p1;
        new_triangle.p2p1 = triangle.p2p1 - triangle.p1;
        new_triangle.p3p1 = triangle.p3p1 - triangle.p1;
        new_triangle.normal = triangle.normal;
        new_material_indices[i] = material_indices[triangle_indices[i]];
    }

    delete[] triangles;
    delete[] material_indices;
    triangles = new_triangles;
    material_indices = new_material_indices;

    bvh_node_count = bvh_nodes.size();
    bvh = new BvhNode[bvh_nodes.size()];

    for (int i = 0; i < bvh_nodes.size(); i++)
    {
        bvh[i] = bvh_nodes[i].build();
    }

    end_time = std::chrono::high_resolution_clock::now();
    auto bvh_time = std::chrono::duration<float>(end_time - start_time).count();
    std::cout << "Triangle count: " << triangle_count << "\n";
    std::cout << "BVH Took " << (bvh_time * 1000) << "ms\n";
    std::cout << "Node count: " << bvh_node_count << "\n";
}