#include "core/gpu/euler_2d.cuh"
#include "core/gpu/reduce.cuh"
#include "core/grid/grid.h"

#include <cuda_runtime.h>
#include <algorithm>

template <class float_type, int warps_count>
__global__ void euler_2d_calculate_dt_gpu_kernel (
  unsigned int n_cells,
  float_type gamma,
  float_type *workspace,
  const float_type *p_rho,
  const float_type *p_u,
  const float_type *p_v,
  const float_type *p_p)
{
  const unsigned int first_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int stride = blockDim.x * gridDim.x;

  float_type max_speed {};

  for (unsigned int idx = first_idx; idx < n_cells; idx += stride)
    {
      const float_type rho = p_rho[idx];
      const float_type p = p_p[idx];
      const float_type a = speed_of_sound_in_gas (gamma, p, rho);
      const float_type u = p_u[idx];
      const float_type v = p_v[idx];

      max_speed = fmax (max_speed, fmax (fabs (u + a), fabs (u - a)));
      max_speed = fmax (max_speed, fmax (fabs (v + a), fabs (v - a)));
    }

  max_speed = block_reduce <float_type, reduce_operation::max, warps_count> (max_speed);

  if (threadIdx.x == 0)
    atomicMax (workspace, max_speed);
}

template <class float_type>
float_type euler_2d_calculate_dt_gpu (
  unsigned int n_cells,
  float_type gamma,
  float_type cfl,
  float_type min_len,
  float_type *workspace,
  const float_type *p_rho,
  const float_type *p_u,
  const float_type *p_v,
  const float_type *p_p)
{
  cudaMemset (workspace, 0, sizeof (float_type));

  constexpr int warps_per_block = 32;
  constexpr int warp_size = 32;
  constexpr int threads_per_block = warps_per_block * warp_size;

  const int blocks = std::min ((n_cells + threads_per_block - 1) / threads_per_block, 1024u);

  euler_2d_calculate_dt_gpu_kernel<float_type, warp_size> <<<blocks, threads_per_block>>> (
      n_cells, gamma, workspace, p_rho, p_u, p_v, p_p);

  float_type max_speed {};
  cudaMemcpy (&max_speed, workspace, sizeof (float_type), cudaMemcpyDeviceToHost);

  float_type new_dt = cfl * min_len / max_speed;
  return new_dt;
}

template <class float_type>
__global__ void euler_2d_calculate_next_time_step_gpu_kernel (
    float_type dt,
    float_type gamma,

    const grid_topology topology,
    const grid_geometry geometry,

    const float_type *p_rho,
    float_type *p_rho_next,
    const float_type *p_u,
    float_type *p_u_next,
    const float_type *p_v,
    float_type *p_v_next,
    const float_type *p_p,
    float_type *p_p_next)
{
  const unsigned int cell_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (cell_id < topology.get_cells_count ())
    euler_2d_calculate_next_cell_values (
        cell_id, dt, gamma, topology, geometry,
        p_rho, p_rho_next, p_u, p_u_next, p_v, p_v_next, p_p, p_p_next);
}

template <class float_type>
void euler_2d_calculate_next_time_step_gpu (
    float_type dt,
    float_type gamma,

    const grid_topology &topology,
    const grid_geometry &geometry,

    const float_type *p_rho,
    float_type *p_rho_next,
    const float_type *p_u,
    float_type *p_u_next,
    const float_type *p_v,
    float_type *p_v_next,
    const float_type *p_p,
    float_type *p_p_next)
{
  constexpr int threads_per_block = 1024;
  const unsigned int blocks = (topology.get_cells_count () + threads_per_block - 1) / threads_per_block;

  euler_2d_calculate_next_time_step_gpu_kernel <<<blocks, threads_per_block>>> (
      dt, gamma, topology, geometry,
      p_rho, p_rho_next, p_u, p_u_next, p_v, p_v_next, p_p, p_p_next);
}

#define GEN_EULER_2D_INSTANCE_FOR(type)                     \
  template type euler_2d_calculate_dt_gpu <type>(           \
      unsigned int n_cells, type gamma, type cfl,           \
      type min_len, type *workspace, const type *p_rho,     \
      const type *p_u, const type *p_v, const type *p_p);   \
  template void euler_2d_calculate_next_time_step_gpu (     \
      type dt, type gamma,                                  \
      const grid_topology &, const grid_geometry &,         \
      const type *p_rho, type *p_rho_next,                  \
      const type *p_u, type *p_u_next, const type *p_v,     \
      type *p_v_next, const type *p_p, type *p_p_next);

GEN_EULER_2D_INSTANCE_FOR (float)
GEN_EULER_2D_INSTANCE_FOR (double)

#undef GEN_EULER_2D_INTERFACE_INSTANCE_FOR
