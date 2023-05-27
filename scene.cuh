#pragma once

#include "common.cuh"
#include "math.cuh"
#include "random.cuh"

#include <vector>

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

struct Ray
{
    Vec3 origin;
    Vec3 direction;
};

struct __align__(16) Material 
{
    Vec3 diffuse_albedo;
    float metallicity;
    Vec3 specular_albedo;
    float roughness;
    Vec3 emitted;   
    float index_of_refraction;
};

struct __align__(16) RayData
{
    Ray ray;
    Vec3 transmitted_color;
    Vec3 collected_color;
    Vec3 hit_point; // New member variable for storing the hit point
};

struct Aabb {
    Vec3 min_bound = { 1e30,  1e30,  1e30};
    Vec3 max_bound = {-1e30, -1e30, -1e30};

    void expand(const Vec3 &other);
    void expand(const Triangle &other);
    void expand(const Aabb &other);
    float half_area() const;
};

struct __align__(32) BvhNode {
    Aabb aabb;
    int child1;
    int child2;

    void maybe_split(const struct Scene *scene, std::vector<BvhNode> &bvh_nodes, int max_depth);
};

struct Scene
{
    Sphere   *spheres;
    int sphere_count;

    Triangle *triangles;
    int triangle_count;

    uint16_t *material_indices;
    Material *materials;
    uint16_t material_count;

    BvhNode *bvh;
    int bvh_node_count;

    int width;
    int height;

    Vec3 *environment_map;
    int environment_map_width, environment_map_height;
    Vec3 camera_position;
    Vec3 forward;
    Vec3 up;
    float vertical_fov;

    float exposure;

    Vec3 min_coord;
    Vec3 inv_dimensions;

    Vec3 scaled_right;
    Vec3 scaled_up;

    Vec3 near_plane_top_left;

    float inv_width;
    float inv_height;

    int bounces;
    int ray_count;

    void precompute_cammera_data();

    void generate_bvh(int max_depth);

    COMMON void bvh_closest_hit_distance(const Ray &ray, float &closest_hit_distance, int &closest_hit_index) const;

    COMMON void generate_initial_rays(RayData *ray_data, unsigned int *ray_indices, unsigned int *ray_keys, int rays_per_pixel, int ray_index, int seed) const;

    COMMON void process_ray(RayData *ray_data_ptr, unsigned int *ray_key, xor_random rng) const;
};

void load_scene(Scene *scene, const char *filename, bool use_bvh);