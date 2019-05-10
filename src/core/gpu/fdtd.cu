#include <cuda_runtime.h>
#include <iostream>

#include "core/common/sources.h"

template <typename float_type>
__global__ void fdtd_update_h_kernel (
    unsigned int nx, unsigned int ny,
    const float_type dx, const float_type dy,
    const float_type * __restrict__ ez,
    const float_type * __restrict__ mh,
    float_type * __restrict__ hx,
    float_type * __restrict__ hy)
{
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int idx = j * nx + i;

  __shared__ float_type cache[33 * 33];

  const unsigned int si = threadIdx.x;
  const unsigned int sj = threadIdx.y;
  const unsigned int sidx = sj * (blockDim.x + 1) + si;

  if (j < ny && i < nx)
  {
    cache[sidx] = ez[idx];

    if (sj == blockDim.y - 1 || j == ny - 1)
      cache[(sj + 1) * (blockDim.x + 1) + si] = j < ny - 1 ? ez[(j + 1) * nx + i] : ez[0 * nx + i];

    if (si == blockDim.x - 1 || i == nx - 1)
      cache[sidx + 1] = i < nx - 1 ? ez[idx + 1] : ez[j * nx + 0];
  }

  __syncthreads ();

  if (j < ny && i < nx)
  {
    const float_type cez = cache[sidx];
    const float_type cex =  (cache[(sj + 1) * (blockDim.x + 1) + si] - cez) / dy;
    const float_type cey = -(cache[sidx + 1] - cez) / dx;

    // update_h
    hx[idx] -= mh[idx] * cex;
    hy[idx] -= mh[idx] * cey;
  }
}

template <typename float_type>
__device__ static float_type update_curl_h (
    unsigned int i,
    unsigned int j,
    const unsigned int nx,
    const unsigned int ny,
    const float_type dx,
    const float_type dy,
    const float_type * __restrict__ hx,
    const float_type * __restrict__ hy)
{
  // TODO For now assume that only periodic boundary conditions exist
  const unsigned int curr_idx   = (j + 0) * nx + i;
  const unsigned int prev_idx_i = i > 0 ? (j + 0) * nx + i - 1 : j * nx + nx - 1;
  const unsigned int prev_idx_j = j > 0 ? (j - 1) * nx + i     : (ny - 1) * nx + i;

  return (hy[curr_idx] - hy[prev_idx_i]) / dx
       - (hx[curr_idx] - hx[prev_idx_j]) / dy;
}

template <typename float_type>
__global__ void fdtd_update_e_kernel (
    float_type t,
    unsigned int nx,
    unsigned int ny,
    const float_type C0_p_dt,
    const float_type dx,
    const float_type dy,
    const float_type * __restrict__ er,
    const float_type * __restrict__ hx,
    const float_type * __restrict__ hy,
    float_type * __restrict__ dz,
    float_type * __restrict__ ez,

    unsigned int sources_count,
    const float_type * __restrict__ sources_frequencies,
    const unsigned int * __restrict__ sources_offsets)
{
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int idx = j * nx + i;

  if (j < ny && i < nx)
  {
    const float_type chz = update_curl_h (i, j, nx, ny, dx, dy, hx, hy);

    dz[idx] += C0_p_dt * chz; // update d = C0 * dt * curl Hz

    // TODO Extract into separate kernel
    for (unsigned int source_id = 0; source_id < sources_count; source_id++)
      if (sources_offsets[source_id] == idx)
        dz[idx] += calculate_source (t, sources_frequencies[source_id]);

    ez[idx] = dz[idx] / er[idx]; // update e
  }
}

template <typename float_type>
void fdtd_step_gpu (
    float_type t,
    const float_type dt,
    unsigned int nx, unsigned int ny,
    const float_type dx, const float_type dy,
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
  constexpr auto C0 = static_cast<float_type> (299792458); /// Speed of light [metres per second]

  dim3 block_size = dim3 (32, 32);
  dim3 grid_size;

  grid_size.x = (nx + block_size.x - 1) / block_size.x;
  grid_size.y = (ny + block_size.y - 1) / block_size.y;

  // TODO Calculate block sizes
  fdtd_update_h_kernel<<<grid_size, block_size>>> (nx, ny, dx, dy, ez, mh, hx, hy);

  // TODO Update source
  fdtd_update_e_kernel<<<grid_size, block_size>>> (t, nx, ny, C0 * dt, dx, dy, er, hx, hy, dz, ez, sources_count, sources_frequencies, sources_offsets);
}

#define GEN_FDTD_INSTANCE_FOR(type)                                                               \
  template void fdtd_step_gpu<type>(type t, const type dt, unsigned int nx, unsigned int ny,      \
                                    const type dx, const type dy, const type *mh, const type *er, \
                                    type *ez, type *dz, type *hx, type *hy,                       \
                                    unsigned int sources_count, const type *sources_frequencies,  \
                                    const unsigned int *sources_offsets);

GEN_FDTD_INSTANCE_FOR (float)
GEN_FDTD_INSTANCE_FOR (double)

#undef GEN_FDTD_INSTANCE_FOR
