# GPU Raytracer SOFTENG751
SOFTENG751 2023

Isaac Chandler

Yogesh Dangwal

Daniel Cutfield

# Compiling

For Windows: 
- Tested using CUDA 12.1 and Visual Studio 2019
- In `build.bat` modify -arch sm_61 to the architecture version of your GPU
  - CUDA supports -arch native in theory but this just crashed the compiler for me 
  - Omitting this option entirely is possible but will likely cause a large performance penalty (40% for me)
- Run `Developer Command Prompt for VS [Year]`
- `cd` to the project root
- Run `build.bat`

For Linux (untested):
- In `build.sh` modify -arch sm_61 to the architecture version of your GPU
- Run `build.sh` (might need to `chmod +x` it)

# Running
- Windows `raytracing.exe <scene file> [options]` 
- Linux (untested) `./raytracing <scene file> [options]` 
- The output image will be `raytracing.png`
- Scene files included with the project are
  - `cornell.scene` - Classic diffuse cornell box
  - `cornell_plus.scene` - Cornell box with some metallic and glass spheres
  - `spheres.scene` - Spheres of various materials with a sun light
  - `teapot.scene` - Utah teapot illuminated by an environment map (Adapted from scene by Benedikt Bitterli)
  - `lamp.scene` - A lamp (Adapted from scene by Benedikt Bitterli)
- Options are
  - `no_sort` Disable ray reordering
  - `cpu` Run CPU ray tracing
  - `no_gpu` Do not run GPU ray tracing 
  - `no_bvh` Do not use a BVH (not recommended for `teapot.scene` or `lamp.scene`)

# Acknowledgements
- Ray tracing basics: https://raytracing.github.io/
- Teapot and lamp scenes: Bennedikt Bitterli, Rendering resources, 2016, https://benedikt-bitterli.me/resources/
- Ray triangle intersection: https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
- BVH construction algorithm: https://jacco.ompf2.com/2022/04/21/how-to-build-a-bvh-part-3-quick-builds/
- Ray AABB intersection: https://tavianator.com/2022/ray_box_boundary.html
- PNG Image output `stb_image_write` library by Sean T. Barrett, https://github.com/nothings/stb
- Environment map projection: From PBRTv4 ray tracer - https://github.com/mmp/pbrt-v4/blob/c4baa534042e2ec4eb245924efbcef477e096389/src/pbrt/util/math.cpp#L317
- Cornell box scene measurements https://www.graphics.cornell.edu/online/box/data.html
- Random number generation https://www.pcg-random.org/download.html

# Scene file format
Commands: 
<pre>image <i>width height rays_per_pixel bounces exposure</i> </pre>
This is the command you are most likely to modify, it will determine the 
image quality and runtime.
<pre>material <i>name property property property ...</i></pre>
<pre>sphere <i>x y z radius</i></pre>
<pre>triangle <i>material_name x0 y0 z0 x1 y1 z1 x2 y2 z2</i></pre>
<pre>quad <i>material_name x0 y0 z0 x1 y1 x1 x2 y2 z2 x3 y3 z3</i></pre>
Equivalent to 2 triangles with corners (0, 1, 2) and (0, 2, 3)
<pre>ply <i>material_name file_path</i></pre>
Load a list of triangles from a ply model file. **This is not a complete ply loader and will likely not work with arbitrary ply files**
<pre>sky_map <i>file_path</i></pre>
Load a PFM (HDR image) file for light emitted when a ray hits nothing
<pre>sky <i>r g b</i></pre>
Equivalent to above but uses a single color instead of loading an image
<pre>camera position <i>x y z</i> forward <i>x y z</i> up <i>x y z</i> fov <i>degrees_vertical</i></pre>

Material Properties:
See comment on `Material` struct in `scene.cu`
<pre>diffuse <i>r g b</i></pre>
Default: `1 1 1`
<pre>specular <i>r g b</i></pre>
Default: `1 1 1`
<pre>emit <i>r g b</i></pre>
Default: `0 0 0`
<pre>metallicity <i>value</i></pre>
Default: `0`
<pre>roughness <i>value</i></pre>
Default: `0`
<pre>ior <i>index_of_refraction</i></pre>
Default: `0`
