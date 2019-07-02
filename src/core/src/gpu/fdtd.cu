#include <cuda_runtime.h>
#include <iostream>

#include "core/common/sources.h"
#include "core/common/curl.h"

constexpr bool USE_SHARED_H_UPDATE = true;

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

  if (j < ny && i < nx)
  {
    const float_type cex = update_curl_ex (i, j, nx, ny, dy, ez);
    const float_type cey = update_curl_ey (i, j, nx, dx, ez);

    // update_h
    hx[idx] -= mh[idx] * cex;
    hy[idx] -= mh[idx] * cey;
  }
}

template <typename float_type, int tile_size>
__global__ void fdtd_update_h_shared_kernel (
    unsigned int nx, unsigned int ny,
    const float_type dx, const float_type dy,
    const float_type * __restrict__ ez,
    const float_type * __restrict__ mh,
    float_type * __restrict__ hx,
    float_type * __restrict__ hy)
{
  __shared__ float_type cache[tile_size + 1][tile_size + 1];
  const unsigned int tx = threadIdx.x;
  const unsigned int ty = threadIdx.y;
  const unsigned int i = blockIdx.x * blockDim.x + tx;
  const unsigned int j = blockIdx.y * blockDim.y + ty;
  const unsigned int curr_idx = j * nx + i;

  if (j < ny && i < nx)
  {
    cache[ty][tx] = ez[curr_idx];

    if (tx == tile_size - 1 || i == nx - 1)
    {
      const unsigned int next_idx_i = i < nx - 1 ? j * nx + i + 1 : j * nx + 0;
      cache[ty][tx + 1] = ez[next_idx_i];
    }

    if (ty == tile_size - 1 || j == ny - 1)
    {
      const unsigned int next_idx_j = j < ny - 1 ? (j + 1) * nx + i : 0 * nx + i;
      cache[ty + 1][tx] = ez[next_idx_j];
    }
  }

  __syncthreads ();

  if (j < ny && i < nx)
  {
    const float_type m_h = mh[curr_idx];

    const float_type cex = (cache[ty + 1][tx] - cache[ty][tx]) / dy;
    const float_type cey = -(cache[ty][tx + 1] - cache[ty][tx]) / dx;
    hx[curr_idx] -= m_h * cex;
    hy[curr_idx] -= m_h * cey;
  }
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
  constexpr int tile_size = 16;

  dim3 block_size = dim3 (tile_size, tile_size);
  dim3 grid_size;

  grid_size.x = (nx + block_size.x - 1) / block_size.x;
  grid_size.y = (ny + block_size.y - 1) / block_size.y;
  grid_size.z = 1;

  if (USE_SHARED_H_UPDATE)
    fdtd_update_h_shared_kernel<float_type, tile_size><<<grid_size, block_size>>> (nx, ny, dx, dy, ez, mh, hx, hy);
  else
    fdtd_update_h_kernel<<<grid_size, block_size>>> (nx, ny, dx, dy, ez, mh, hx, hy);
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
