//
// Created by egi on 5/11/19.
//

#ifndef FDTD_COMMON_DEFS_H
#define FDTD_COMMON_DEFS_H

#ifdef __CUDACC__
#define CPU_GPU __device__ __host__
#else
#define CPU_GPU
#endif

#endif //FDTD_COMMON_DEFS_H
