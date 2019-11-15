#include "core/gpu/euler_2d.cuh"
#include "core/gpu/reduce.cuh"
#include "core/grid/grid.h"

#include <cuda_runtime.h>
#include <algorithm>

template<>
inline CPU_GPU float max_speed (float v_c, float v_n, float u_c, float u_n)
{
  const float zero = 0.0f;
  const float splus  = fmaxf (zero, fmaxf (u_c + v_c, u_n + v_n));
  const float sminus = fminf (zero, fminf (u_c - v_c, u_n - v_n));
  return fmaxf (splus, -sminus);
}

template <class float_type, int warps_count>
__global__ void euler_2d_calculate_dt_gpu_kernel (
  float_type gamma,

  const grid_topology topology,
  const grid_geometry geometry,

  float_type *workspace,
  const float_type *p_rho,
  const float_type *p_u,
  const float_type *p_v,
  const float_type *p_p)
{
  const unsigned int first_cell_id = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int stride = blockDim.x * gridDim.x;

  float_type min_len = std::numeric_limits<float_type>::max ();
  float_type max_speed = std::numeric_limits<float_type>::min ();

  for (unsigned int cell_id = first_cell_id; cell_id < topology.get_cells_count (); cell_id += stride)
    {
      const float_type rho = p_rho[cell_id];
      const float_type p = p_p[cell_id];
      const float_type a = speed_of_sound_in_gas (gamma, p, rho);
      const float_type u = p_u[cell_id];
      const float_type v = p_v[cell_id];

      max_speed = fmaxf (max_speed, fmaxf (fabsf (u + a), fabsf (u - a)));
      max_speed = fmaxf (max_speed, fmaxf (fabsf (v + a), fabsf (v - a)));

      for (unsigned int edge_id = 0; edge_id < topology.get_edges_count (cell_id); edge_id++)
      {
        const float_type edge_len = geometry.get_edge_area (cell_id, edge_id);
        if (edge_len < min_len)
          min_len = edge_len;
      }
    }

  min_len   = block_reduce <float_type, reduce_operation::min, warps_count> (min_len);
  max_speed = block_reduce <float_type, reduce_operation::max, warps_count> (max_speed);

  if (threadIdx.x == 0)
  {
    atomicMin (workspace + 0, min_len);
    atomicMax (workspace + 1, max_speed);
  }
}

template <class float_type>
float_type euler_2d_calculate_dt_gpu (
  float_type gamma,
  float_type cfl,

  const grid_topology &topology,
  const grid_geometry &geometry,

  float_type *workspace,
  const float_type *p_rho,
  const float_type *p_u,
  const float_type *p_v,
  const float_type *p_p)
{
  float_type cpu_workspace_copy[2];
  float_type &min_len = cpu_workspace_copy[0];
  float_type &max_speed = cpu_workspace_copy[1];

  min_len = std::numeric_limits<float_type>::max ();
  max_speed = std::numeric_limits<float_type>::min ();

  cudaMemcpy (workspace, cpu_workspace_copy, 2 * sizeof (float_type), cudaMemcpyHostToDevice);

  constexpr int warps_per_block = 32;
  constexpr int warp_size = 32;
  constexpr int threads_per_block = warps_per_block * warp_size;

  const int blocks = std::min ((topology.get_cells_count () + threads_per_block - 1) / threads_per_block, 1024u);

  euler_2d_calculate_dt_gpu_kernel<float_type, warp_size> <<<blocks, threads_per_block>>> (
      gamma, topology, geometry, workspace, p_rho, p_u, p_v, p_p);

  cudaMemcpy (cpu_workspace_copy, workspace, 2 * sizeof (float_type), cudaMemcpyDeviceToHost);

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
      type gamma, type cfl,                                 \
      const grid_topology &, const grid_geometry &,         \
      type *workspace, const type *p_rho,                   \
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
