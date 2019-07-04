//
// Created by egi on 7/2/19.
//

#include "core/gpu/euler_2d.cuh"
#include "core/gpu/euler_2d_interface.h"
#include "cpp/common_funcs.h"

template <class float_type>
float_type euler_2d_calculate_dt_gpu_interface (
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
#ifdef GPU_BUILD
  return euler_2d_calculate_dt_gpu (n_cells, gamma, cfl, min_len, workspace, p_rho, p_u, p_v, p_p);
#else
  cpp_unreferenced (n_cells, gamma, cfl, min_len, workspace, p_rho, p_u, p_v, p_p);
  return {};
#endif
}

template <class float_type>
void euler_2d_calculate_next_time_step_gpu_interface (
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
#ifdef GPU_BUILD
  euler_2d_calculate_next_time_step_gpu (dt, gamma, topology, geometry, p_rho, p_rho_next, p_u, p_u_next, p_v, p_v_next, p_p, p_p_next);
#else
  cpp_unreferenced (nx, ny, dt, gamma, cell_area, edge_lengths, p_rho, p_rho_next, p_u, p_u_next, p_v, p_v_next, p_p, p_p_next, topology, geometry);
#endif
}

#define GEN_EULER_2D_INTERFACE_INSTANCE_FOR(type)                 \
  template type euler_2d_calculate_dt_gpu_interface <type>(       \
      unsigned int n_cells, type gamma, type cfl, type min_len,   \
      type *workspace, const type *p_rho, const type *p_u,        \
      const type *p_v,const type *p_p);                           \
  template void euler_2d_calculate_next_time_step_gpu_interface ( \
      type dt, type gamma,                                        \
      const grid_topology &, const grid_geometry &,               \
      const type *p_rho, type *p_rho_next,                        \
      const type *p_u, type *p_u_next, const type *p_v,           \
      type *p_v_next, const type *p_p, type *p_p_next);

GEN_EULER_2D_INTERFACE_INSTANCE_FOR (float)
GEN_EULER_2D_INTERFACE_INSTANCE_FOR (double)

#undef GEN_EULER_2D_INTERFACE_INSTANCE_FOR
