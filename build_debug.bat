nvcc -O0 -G -g -dc raytracing.cu
nvcc -O0 -G -g -dc scene.cu
nvcc -O0 -G -g -o raytracing_debug.exe scene.obj raytracing.obj