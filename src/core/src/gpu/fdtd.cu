#include <cuda_runtime.h>
#include <iostream>

#include "core/common/sources.h"
#include "core/common/curl.h"
#include "core/gpu/fdtd.cuh"

template <typename float_type>
__global__ void fdtd_update_h_kernel (
    const grid_topology topology,
    const grid_geometry geometry,
    const float_type * __restrict__ ez,
    const float_type * __restrict__ mh,
    float_type * __restrict__ hx,
    float_type * __restrict__ hy)
{
  const unsigned int cell_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (cell_id < topology.get_cells_count ())
    fdtd_2d_update_h (cell_id, topology, geometry, ez, mh, hx, hy);
}

template <typename float_type>
__global__ void fdtd_update_e_kernel (
    float_type t,
    const float_type C0_p_dt,
    const grid_topology topology,
    const grid_geometry geometry,
    const float_type * __restrict__ er,
    const float_type * __restrict__ hx,
    const float_type * __restrict__ hy,
    float_type * __restrict__ dz,
    float_type * __restrict__ ez,

    unsigned int sources_count,
    const float_type * __restrict__ sources_frequencies,
    const unsigned int * __restrict__ sources_offsets)
{
  const unsigned int cell_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (cell_id < topology.get_cells_count ())
    fdtd_2d_update_e (
        cell_id, t, C0_p_dt, topology, geometry, er, hx, hy, dz, ez,
        sources_count, sources_frequencies, sources_offsets);
}

template <typename float_type>
void fdtd_step_gpu (
    float_type t,
    const float_type C0_p_dt,

    const grid_topology &topology,
    const grid_geometry &geometry,

    const float_type *mh,
    const float_type *er,
    float_type *ez,
    float_type *dz,
    float_type *hx,
    float_type *hy,

    unsigned int sources_count,
    const float_type * __restrict__ sources_frequencies,
    const unsigned int * __restrict__ sources_offsets)
{
  constexpr unsigned int threads_per_block = 1024;
  const unsigned int blocks_count = (topology.get_cells_count () + threads_per_block - 1) / threads_per_block;

  fdtd_update_h_kernel<<<blocks_count, threads_per_block>>> (topology, geometry, ez, mh, hx, hy);
  fdtd_update_e_kernel<<<blocks_count, threads_per_block>>> (t, C0_p_dt, topology, geometry, er, hx, hy, dz, ez, sources_count, sources_frequencies, sources_offsets);
}

#define GEN_FDTD_INSTANCE_FOR(type)                                                                \
  template void fdtd_step_gpu<type>(type, const type, const grid_topology &, const grid_geometry &,\
                                    const type *mh, const type *er, type *ez, type *dz, type *hx,  \
                                    type *hy, unsigned int sources_count,                          \
                                    const type *sources_frequencies,                               \
                                    const unsigned int *sources_offsets);

GEN_FDTD_INSTANCE_FOR (float)
GEN_FDTD_INSTANCE_FOR (double)

#undef GEN_FDTD_INSTANCE_FOR
