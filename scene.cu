#include "scene.cuh"

#include <unordered_map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>

#define MAX_BVH_DEPTH 30

// By using this we promise CUDA that the value we are reading will never be written by a kernel
// This allows the data to be loaded into L1 cache which is not coherent
template<typename T> COMMON T load_read_only(T *t)
{
    static_assert(alignof(T) >= sizeof(float4), "load_read_only requires 16 byte alignment");

#ifdef __CUDA_ARCH__
    constexpr int count = sizeof(T) / sizeof(float4);

    union Dummy{
        float4 dummy[count];
        T value;

        COMMON Dummy() {}
    };
    
    Dummy dummy;

    #pragma unroll
    for (int i = 0; i < count; i++)
    {
        dummy.dummy[i] = __ldg(((float4 *) t) + i);
    }

    return dummy.value;
#else
    return *t;
#endif
}

// Convert number of the form 0bABCDE to
// 0b00A00B00C00D00E
__device__ unsigned short interleave_5(unsigned short x)
{
    x = (x | (x << 8)) & 0b1000000001111;
    x = (x | (x << 4)) & 0x1000010100011;
    x = (x | (x << 2)) & 0b1001001001001;

    return x;
}

__device__ unsigned short morton_code(const Vec3 &vec)
{
    unsigned short x = (unsigned short) (vec.x * 31.99);
    unsigned short y = (unsigned short) (vec.y * 31.99);
    unsigned short z = (unsigned short) (vec.z * 31.99);

    return interleave_5(x) | (interleave_5(y) << 1) | (interleave_5(z) << 2);
}

void Scene::precompute_camera_data()
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

COMMON void Scene::generate_initial_rays(RayData *ray_data, unsigned int *ray_indices, unsigned int *ray_keys, int rays_per_pixel, int ray_index, int seed) const
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

// Branchless ray AABB intersection from https://tavianator.com/2022/ray_box_boundary.html
// (assuming there is hardware min/max which is true on all modern GPUs and CPUs)
COMMON bool ray_aabb_intersection(const Aabb &aabb, const Ray &ray, const Vec3 &n_inv, float &tmin, float tmax)
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

COMMON void Scene::bvh_closest_hit_distance(const Ray &ray, float &closest_hit_distance, int &closest_hit_index) const
{
    Vec3 n_inv = {1 / ray.direction.x, 1 / ray.direction.y, 1 / ray.direction.z};

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

        BvhNode node = load_read_only(&bvh[node_index_stack[stack_count]]);

        
        if (node.is_leaf())
        {
            for (int i = node.child2; i < node.child1; i++)
            {
                const auto triangle = load_read_only(&triangles[i]);

                // Möller–Trumbore ray-triangle intersection algorithm
                // Based on https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
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
        }
        else
        {
            float hit1_distance, hit2_distance;

            bool hit1 = ray_aabb_intersection(load_read_only(&bvh[node.child1]).aabb, ray, n_inv, hit1_distance, closest_hit_distance);
            bool hit2 = ray_aabb_intersection(load_read_only(&bvh[node.child2]).aabb, ray, n_inv, hit2_distance, closest_hit_distance);

            if (hit1 && hit2)
            {
                if (hit1_distance < hit2_distance)
                {
                    node_index_stack[stack_count] = node.child1;
                    node_distance_stack[stack_count] = hit1_distance;
                    stack_count++;

                    node_index_stack[stack_count] = node.child2;
                    node_distance_stack[stack_count] = hit2_distance;
                    stack_count++;
                }
                else
                {
                    node_index_stack[stack_count] = node.child2;
                    node_distance_stack[stack_count] = hit2_distance;
                    stack_count++;

                    node_index_stack[stack_count] = node.child1;
                    node_distance_stack[stack_count] = hit1_distance;
                    stack_count++;
                }
            }
            else if (hit1)
            {
                node_index_stack[stack_count] = node.child1;
                node_distance_stack[stack_count] = hit1_distance;
                stack_count++;
            }
            else if (hit2)
            {
                node_index_stack[stack_count] = node.child2;
                node_distance_stack[stack_count] = hit2_distance;
                stack_count++;
            }
        }
    }
}

void Scene::copy_from_cpu_async(const Scene &scene, cudaStream_t stream)
{
    Scene scene_copy = scene;

    int primitive_count = scene.triangle_count + scene.sphere_count;
    int environment_map_size = scene.environment_map_width * scene.environment_map_height;

    CUDA_CHECK(cudaMalloc(&scene_copy.spheres,          scene.sphere_count   * sizeof(Sphere)));
    CUDA_CHECK(cudaMalloc(&scene_copy.triangles,        scene.triangle_count * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&scene_copy.materials,        scene.material_count * sizeof(Material)));
    CUDA_CHECK(cudaMalloc(&scene_copy.material_indices, primitive_count      * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&scene_copy.bvh,              scene.bvh_node_count * sizeof(BvhNode)));
    CUDA_CHECK(cudaMalloc(&scene_copy.environment_map,  environment_map_size * sizeof(Vec3)));

    CUDA_CHECK(cudaMemcpyAsync(scene_copy.spheres,          scene.spheres,          sizeof(Sphere)   * scene.sphere_count,   cudaMemcpyHostToDevice, stream));    
    CUDA_CHECK(cudaMemcpyAsync(scene_copy.triangles,        scene.triangles,        sizeof(Triangle) * scene.triangle_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(scene_copy.materials,        scene.materials,        sizeof(Material) * scene.material_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(scene_copy.material_indices, scene.material_indices, sizeof(uint16_t) * primitive_count,      cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(scene_copy.bvh,              scene.bvh,              sizeof(BvhNode)  * scene.bvh_node_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(scene_copy.environment_map,  scene.environment_map,  sizeof(Vec3)     * environment_map_size, cudaMemcpyHostToDevice, stream));

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
    CUDA_CHECK(cudaFree(scene_copy.spheres));
}


// Maybe it makes sense to pre-convert to a cubemap instead of doing this every time a ray misses
// Our test environment map is in a format used by the PBRTv4 ray tracer
// This code is based on https://github.com/mmp/pbrt-v4/blob/c4baa534042e2ec4eb245924efbcef477e096389/src/pbrt/util/math.cpp#L317
COMMON Vec3 equal_area_project_sphere_to_square(const Vec3 &direction)
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

COMMON void Scene::process_ray(RayData *ray_data_ptr, unsigned int *ray_key, xor_random rng) const
{
#ifdef __CUDA_ARCH__
    if (*ray_key == 0xFFFF'FFFF)
        return;
#else
    if (ray_data_ptr->transmitted_color.x == 0 && ray_data_ptr->transmitted_color.y == 0 && ray_data_ptr->transmitted_color.z == 0)
        return;
#endif

    RayData ray_data = *ray_data_ptr;

    float closest_hit_distance = 1e30;

    int closest_hit_index = -1;

    const auto ray = ray_data.ray;

    for (int i = 0; i < sphere_count; i++)
    {
        const auto sphere = spheres[i];

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

    bvh_closest_hit_distance(ray, closest_hit_distance, closest_hit_index);

    if (closest_hit_index == -1)
    {
        // Environment map in our test data is rotated and has y and z axes flipped, 
        // apply a hardcoded transformation for now.
        float dir_x = ray.direction.x * -0.386527 + ray.direction.z * 0.922278;
        float dir_y = ray.direction.x * -0.922278 + ray.direction.z * -0.386527;
        float dir_z = ray.direction.y;

        Vec3 coords = equal_area_project_sphere_to_square({dir_x, dir_y, dir_z});
        float x = coords.x;
        float y = coords.y;

        // Nearest filtering
        int texel_x = (int) (clamp01(x) * (environment_map_width  - 1) + 0.5);
        int texel_y = (int) (clamp01(y) * (environment_map_height - 1) + 0.5);
        Vec3 sky_color = environment_map[texel_y * environment_map_height + texel_x];

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
            const auto hit_sphere = spheres[closest_hit_index];
            normal = (1 / hit_sphere.radius) * (hit_point - hit_sphere.center);
        }
        else
        {
            const auto hit_triangle = triangles[closest_hit_index - sphere_count];
            normal = hit_triangle.normal;
        }

        const auto material = load_read_only(&materials[material_indices[closest_hit_index]]);

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
        Vec3 position;
        Vec3 normal;
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
            triangle.normal = (1.0f / 3.0f) * (triangle.p1 + triangle.p2p1 + triangle.p3p1);

            triangles.push_back(triangle);
        }
    }
}

Vec3 *load_pfm(const std::string &filename, int *width, int *height)
{
    std::ifstream file(filename, std::ios_base::binary);

    std::string line;
    std::getline(file, line);
    std::getline(file, line);

    std::stringstream ss(line);
    
    ss >> *width;
    ss >> *height;

    std::getline(file, line);

    Vec3 *image = new Vec3[*width * *height];
    file.read((char *) image, sizeof(Vec3) * *width * *height);

    return image;
}

void load_scene(Scene *scene, const char *filename, bool use_bvh)
{
    scene->width = 1920;
    scene->height = 1080;
    scene->ray_count = 1;
    scene->bounces = 3;

    std::vector<Sphere> spheres;
    std::vector<Triangle> triangles;

    std::ifstream scene_file(filename);

    std::unordered_map<std::string, uint16_t> materials_map;
    std::vector<Material> materials;
    std::vector<uint16_t> sphere_materials;
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

            scene->environment_map = new Vec3{r, g, b};
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

            triangle.normal = (1.0f / 3.0f) * (triangle.p1 + triangle.p2p1 + triangle.p3p1);

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
            triangle.p2p1 = p2;
            triangle.p3p1 = p3;
            triangle.normal = (1.0f / 3.0f) * (triangle.p1 + triangle.p2p1 + triangle.p3p1);

            triangles.push_back(triangle);

            triangle.p1 = p1;
            triangle.p2p1 = p3;
            triangle.p3p1 = p4;
            triangle.normal = (1.0f / 3.0f) * (triangle.p1 + triangle.p2p1 + triangle.p3p1);

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

    scene->sphere_count = spheres.size();
    scene->spheres = new Sphere[spheres.size()];
    std::copy(spheres.begin(), spheres.end(), scene->spheres);
    
    scene->triangle_count = triangles.size();
    scene->triangles = new Triangle[triangles.size()];
    std::copy(triangles.begin(), triangles.end(), scene->triangles);

    scene->material_indices = new uint16_t[sphere_materials.size() + triangle_materials.size()];
    std::copy(sphere_materials.begin(), sphere_materials.end(), scene->material_indices);
    std::copy(triangle_materials.begin(), triangle_materials.end(), scene->material_indices + sphere_materials.size());

    scene->materials = new Material[materials.size()];
    std::copy(materials.begin(), materials.end(), scene->materials);
    scene->material_count = (uint16_t) materials.size();

    scene->precompute_camera_data();
    scene->generate_bvh(use_bvh ? MAX_BVH_DEPTH : 0);

    scene->min_coord = scene->bvh[0].aabb.min_bound;
    Vec3 scene_max_coord = scene->bvh[0].aabb.max_bound;
    for (const auto &sphere : spheres)
    {
        scene_max_coord = max(scene_max_coord, sphere.center + Vec3{sphere.radius, sphere.radius, sphere.radius});
        scene->min_coord = min(scene->min_coord, sphere.center - Vec3{sphere.radius, sphere.radius, sphere.radius});
    }

    scene->inv_dimensions = {1 / scene_max_coord.x, 1 / scene_max_coord.y, 1 / scene_max_coord.z};
}

void Aabb::expand(const Vec3 &other)
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
    Vec3 size = max_bound - min_bound;

    return size.x * size.y + size.x * size.z + size.y * size.z;
}

COMMON bool BvhNode::is_leaf() const
{
    return child2 <= child1;
}

// Binned surface area heuristic BVH computation
// Based on https://jacco.ompf2.com/2022/04/21/how-to-build-a-bvh-part-3-quick-builds/
void BvhNode::maybe_split(const Scene *scene, std::vector<BvhNode> &bvh_nodes, int max_depth)
{
    for (int i = child2; i < child1; i++)
    {
        aabb.expand(scene->triangles[i]);
    }

    int our_count = child1 - child2;

    if (our_count <= 4 || max_depth == 0)
    {
        return;
    }

    float our_cost = aabb.half_area() * our_count;

    struct Bin
    {
        Aabb aabb;
        int triangle_count = 0;
    };

    constexpr int BINS = 8;

    int best_axis;
    float best_position;
    float best_cost = our_cost;

    for (int axis = 0; axis < 3; axis++)
    {
        float min_centroid = 1e30;
        float max_centroid = -1e30;

        for (int i = child2; i < child1; i++)
        {
            const auto &triangle = scene->triangles[i];
            min_centroid = min(min_centroid, triangle.normal[axis]);
            max_centroid = max(max_centroid, triangle.normal[axis]);
        }

        if (min_centroid == max_centroid)
            continue;

        float scale = BINS / (max_centroid - min_centroid);

        Bin bins[BINS];

        for (int i = child2; i < child1; i++)
        {
            const auto &triangle = scene->triangles[i];
            auto &bin =  bins[std::min(BINS - 1, (int) ((triangle.normal[axis] - min_centroid) * scale))];

            bin.triangle_count++;

            bin.aabb.expand(triangle);
        }

        float left_area[BINS - 1], right_area[BINS - 1];
        int left_count[BINS - 1];
        int left_sum = 0;

        Aabb left_box, right_box;

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
        return;
    }

    int i = child2;
    int j = child1 - 1;

    while (i <= j)
    {
        if (scene->triangles[i].normal[best_axis] < best_position)
        {
            i++;
        }
        else
        {
            std::swap(scene->triangles[i], scene->triangles[j]);
            std::swap(scene->material_indices[scene->sphere_count + i], scene->material_indices[scene->sphere_count + j]);
            j--;
        }
    }

    if (i == child1 || i == child2)
    {
        return;
    }

    int left_child_index = bvh_nodes.size();
    bvh_nodes.emplace_back();
    auto &left_child = bvh_nodes.back();

    int right_child_index = bvh_nodes.size();
    bvh_nodes.emplace_back();
    auto &right_child = bvh_nodes.back();

    left_child.child2 = child2;
    left_child.child1 = i;
    right_child.child2 = i;
    right_child.child1 = child1;

    left_child.maybe_split(scene, bvh_nodes, max_depth - 1);
    right_child.maybe_split(scene, bvh_nodes, max_depth - 1);

    child1 = left_child_index;
    child2 = right_child_index;
}

void Scene::generate_bvh(int max_depth)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    decltype(start_time) end_time;

    std::vector<BvhNode> bvh_nodes;
    // Not only for speed, needed so that references will never be invalidated
    bvh_nodes.reserve(triangle_count * 2);

    bvh_nodes.emplace_back();
    auto &root = bvh_nodes.back();

    root.child2 = 0;
    root.child1 = triangle_count;
    
    root.maybe_split(this, bvh_nodes, max_depth);

    bvh_node_count = bvh_nodes.size();
    bvh = new BvhNode[bvh_nodes.size()];
    std::copy(bvh_nodes.begin(), bvh_nodes.end(), bvh);

    end_time = std::chrono::high_resolution_clock::now();
    auto bvh_time = std::chrono::duration<float>(end_time - start_time).count();
    std::cout << "Triangle count: " << triangle_count << "\n";
    std::cout << "BVH Took " << (bvh_time * 1000) << "ms\n";
    std::cout << "Node count: " << bvh_node_count << "\n";

    for (int i = 0; i < triangle_count; i++)
    {
        auto &triangle = triangles[i];
        triangle.p2p1 -= triangle.p1;
        triangle.p3p1 -= triangle.p1;
        triangle.normal = normalise(cross(triangle.p3p1, triangle.p2p1));
    }
}