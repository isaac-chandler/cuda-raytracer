#!/bin/sh
nvcc --use_fast_math -O3 -g -G -lineinfo -dopt=on -dlto -src-in-ptx -Xcompiler -ffast-math -Xcompiler -fopenmp -o raytracing -arch sm_61 --restrict raytracing.cu scene.cu