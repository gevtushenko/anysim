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
  {
    float_type q_c[4];
    float_type q_n[4];

    float_type Q_c[4];
    float_type Q_n[4];

    float_type F_c[4];
    float_type F_n[4];

    float_type F_sigma[4];  /// Edge flux in local coordinate system
    float_type f_sigma[4];  /// Edge flux in global coordinate system

    fill_state_vector (cell_id, gamma, p_rho, p_u, p_v, p_p, q_c);

    float_type flux[4] = {0.0, 0.0, 0.0, 0.0};

    for (int edge = 0; edge < 4; edge++)
    {
      const unsigned int j = topology.get_neighbor_id (cell_id, edge);

      fill_state_vector (j, gamma, p_rho, p_u, p_v, p_p, q_n);
      rotate_vector_to_edge_coordinates (edge, q_c, Q_c);
      rotate_vector_to_edge_coordinates (edge, q_n, Q_n);
      fill_flux_vector (Q_c, F_c, gamma);
      fill_flux_vector (Q_n, F_n, gamma);

      const float_type U_c = Q_c[1] / Q_c[0];
      const float_type U_n = Q_n[1] / Q_n[0];

      rusanov_scheme (gamma, p_p[cell_id], p_p[j], p_rho[cell_id], p_rho[j], U_c, U_n, F_c, F_n, Q_n, Q_c, F_sigma);

      rotate_vector_from_edge_coordinates (edge, F_sigma, f_sigma);

      for (int c = 0; c < 4; c++)
        flux[c] += geometry.get_edge_area (cell_id, edge) * f_sigma[c];
    }

    float_type new_q[4];
    for (int c = 0; c < 4; c++)
      new_q[c] = q_c[c] - (dt / geometry.get_cell_volume (cell_id)) * (flux[c]);

    const float_type rho = new_q[0];
    const float_type u   = new_q[1] / rho;
    const float_type v   = new_q[2] / rho;
    const float_type E   = new_q[3] / rho;

    p_rho_next[cell_id] = rho;
    p_u_next[cell_id] = u;
    p_v_next[cell_id] = v;
    p_p_next[cell_id] = calculate_p (gamma, E, u, v, rho);
  }
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
