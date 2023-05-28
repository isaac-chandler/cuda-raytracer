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

// Triangles have two different representations during the program
// If p1, p2 and p3 are the triangle vertices then:
// During ray tracing: 
//  - p1 = p1
//  - p2p1 = p2 - p1
//  - p3p1 = p3 - p1
//  - normal = (p2 - p1) x (p3 - p1) [x means cross product]
// During BVH construction: 
//  - p1 = p1
//  - p2p1 = p2
//  - p3p1 = p3
//  - normal = (p1 + p2 + p3) / 3 [centroid]
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

// For all colors x = red, y = green, z = blue
struct __align__(16) Material 
{
    // Color light is tinted when reflected diffusely
    // 
    Vec3 diffuse_albedo;
    // Portion of reflections that are specular (mirror-like) instead of diffuse (scattering) (0-1)
    float metallicity;         
    // Color light is tinted with when reflecting diffusely
    // Should be (1, 1, 1) for glossy materials
    // Colored for metals, components should be (0-1)
    Vec3 specular_albedo;      
    // How much randomness is added to metallic reflections (0-1)
    float roughness;           
    // Color and strength of light emitted by surface (can be > 1)
    Vec3 emitted;              
    // Index of refraction 0 means material is opaque (0 or > 1)
    float index_of_refraction;
};

struct __align__(16) RayData
{
    Ray ray;
    // Tint due to all albedos applied to this ray
    Vec3 transmitted_color;
    // Actual light along this array from emissive materials or skybox
    Vec3 collected_color;
};

struct Aabb {
    // Use very large floats instead of infinity since special IEEE float
    // values MAY be slower to operate on (definitely denormals but maybe infinities are fine?)
    Vec3 min_bound = { 1e30,  1e30,  1e30};
    Vec3 max_bound = {-1e30, -1e30, -1e30};

    void expand(const Vec3 &other);
    void expand(const Triangle &other);
    void expand(const Aabb &other);
    float half_area() const;
};

struct __align__(32) BvhNode {
    Aabb aabb;
    // For non-leaf nodes
    //  - child1 < child2
    //  - child1 is the index in the BVH array of this node's child
    //  - child2 is the index in the BVH array of this node's child
    // For leaf nodes
    //  - child1 >= child2
    //  - child2 is the start index in the triangle array of this node's triangle list
    //  - child1 is the end index in the triangle array of this node's triangle list
    int child1;
    int child2;

    // Split this BVH node into two leaves if there is a split plane with a lower cost than the 
    // unsplit node
    void maybe_split(const struct Scene *scene, std::vector<BvhNode> &bvh_nodes, int max_depth);

    COMMON bool is_leaf() const;
};

struct Scene
{
    Sphere *spheres;
    int sphere_count;

    Triangle *triangles;
    int triangle_count;

    // Material index associated with each primitive, indices are stored instead of the whole material
    // since there may be hundreds of thousands of triangles in a mesh with the exact same material
    // Spheres are associated with material_indices[0..sphere_count - 1]
    // Triangles are associated with material_indices[sphere_count .. sphere_count + triangle_count - 1]
    uint16_t *material_indices;
    Material *materials;
    uint16_t material_count;

    // Bounding volume hierarchy containing all triangles
    // Spheres are not included here for simplicity and speed
    // since spheres were mainly added for early testing and aren't 
    // used in 3D modelling software so we won't have any scenes with
    // a huge number of spheres.
    BvhNode *bvh;
    int bvh_node_count;

    // Output image dimensions
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

    // Precomputed values for ray generation based on camera settings
    Vec3 scaled_right;
    Vec3 scaled_up;

    Vec3 near_plane_top_left;

    float inv_width;
    float inv_height;

    int bounces;
    int ray_count;

    void precompute_camera_data();

    void generate_bvh(int max_depth);

    void copy_from_cpu_async(const Scene &cpu_scene, cudaStream_t stream);
    void free_from_gpu();

    COMMON void bvh_closest_hit_distance(const Ray &ray, float &closest_hit_distance, int &closest_hit_index) const;

    COMMON void generate_initial_rays(RayData *ray_data, unsigned int *ray_indices, unsigned int *ray_keys, int rays_per_pixel, int ray_index, int seed) const;

    COMMON void process_ray(RayData *ray_data_ptr, unsigned int *ray_key, xor_random rng) const;
};

void load_scene(Scene *scene, const char *filename, bool use_bvh);