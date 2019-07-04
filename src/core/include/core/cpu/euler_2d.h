//
// Created by egi on 2/18/18.
//

#ifndef BENCHMARK_EULER_AOS_H
#define BENCHMARK_EULER_AOS_H

#ifdef GPU_BUILD
#include <cuda_runtime.h>
#endif

#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>

#include "core/grid/grid.h"
#include "core/gpu/euler_2d.cuh"
#include "core/gpu/euler_2d_interface.h"
#include "core/config/configuration.h"
#include "core/solver/workspace.h"
#include "core/cpu/thread_pool.h"
#include "core/solver/solver.h"
#include "cpp/common_funcs.h"

template<class float_type>
class euler_2d : public solver
{
  constexpr static int LEFT = 0;
  constexpr static int BOTTOM = 1;
  constexpr static int RIGHT = 2;
  constexpr static int TOP = 3;

  bool use_gpu = false;

  float_type cfl = 0.1;
  float_type gamma = 1.4;

  float_type normals_x[4];
  float_type normals_y[4];

  const grid *solver_grid = nullptr;

public:
  euler_2d (
      thread_pool &threads_arg,
      workspace &solver_workspace_arg)
    : solver (threads_arg, solver_workspace_arg)
  {
    normals_x[LEFT] = -1.0f;  normals_y[LEFT] = 0.0f;
    normals_x[BOTTOM] = 0.0f; normals_y[BOTTOM] = -1.0f;
    normals_x[RIGHT] = 1.0f;  normals_y[RIGHT] = 0.0f;
    normals_x[TOP] = 0.0f;    normals_y[TOP] = 1.0f;
  }

  ~euler_2d () override = default;

  void fill_configuration_scheme (configuration &config, std::size_t config_id) final
  {
    config.create_node (config_id, "cfl", 0.1);
    config.create_node (config_id, "gamma", 1.4);
  }

  bool is_gpu_supported () const final
  {
    return true;
  }

  void apply_configuration (const configuration &config, std::size_t solver_id, grid *solver_grid_arg, int gpu_num) final
  {
    cpp_unreferenced (gpu_num);

    const auto solver_children = config.children_for (solver_id);

    auto cfl_id = solver_children[0];
    auto gamma_id = solver_children[1];
    cfl = config.get_node_value (cfl_id);
    gamma = config.get_node_value (gamma_id);

    solver_grid = solver_grid_arg;

#ifdef GPU_BUILD
    use_gpu = gpu_num >= 0;

    if (use_gpu)
      {
        solver_grid_arg->create_field<float_type> ("gpu_rho", memory_holder_type::device, 2);
        solver_grid_arg->create_field<float_type> ("gpu_u",   memory_holder_type::device, 2);
        solver_grid_arg->create_field<float_type> ("gpu_v",   memory_holder_type::device, 2);
        solver_grid_arg->create_field<float_type> ("gpu_p",   memory_holder_type::device, 2);
        solver_workspace.allocate ("gpu_edge_length", memory_holder_type::device, 4 * sizeof (float_type));
        solver_workspace.allocate ("euler_workspace", memory_holder_type::device, 2 * sizeof (float_type));
      }
#endif

    solver_grid_arg->create_field<float_type> ("rho", memory_holder_type::host, 2);
    solver_grid_arg->create_field<float_type> ("u",   memory_holder_type::host, 2);
    solver_grid_arg->create_field<float_type> ("v",   memory_holder_type::host, 2);
    solver_grid_arg->create_field<float_type> ("p",   memory_holder_type::host, 2);

    auto rho_1 = reinterpret_cast<float_type *> (solver_workspace.get ("rho", 0));
    auto u_1   = reinterpret_cast<float_type *> (solver_workspace.get ("u", 0));
    auto v_1   = reinterpret_cast<float_type *> (solver_workspace.get ("v", 0));
    auto p_1   = reinterpret_cast<float_type *> (solver_workspace.get ("p", 0));

    // TODO Extract initialization
#if 0
    const float_type circle_x = 0.5;
    const float_type circle_y = 0.5;
    const float_type circle_rad = 0.1;
#endif

    ///
    const float_type x_0 = 1.0;
    const float_type y_0 = 1.5;

    const auto topology = solver_grid->gen_topology_wrapper ();
    const auto geometry = solver_grid->gen_geometry_wrapper ();

    for (unsigned int cell_id = 0; cell_id < topology.get_cells_count (); cell_id++)
    {
      const float_type cx = geometry.get_cell_center_x (cell_id);
      const float_type cy = geometry.get_cell_center_y (cell_id);

      if (cx < x_0)
      {
        rho_1[cell_id] = 1.0;
        p_1[cell_id] = 1.0;
        v_1[cell_id] = 0.0;
        u_1[cell_id] = 0.0;
      }
      else
      {
        if (cy < y_0)
        {
          rho_1[cell_id] = 1.0;
          p_1[cell_id] = 0.1;
          v_1[cell_id] = 0.0;
          u_1[cell_id] = 0.0;
        }
        else
        {
          rho_1[cell_id] = 0.125;
          p_1[cell_id] = 0.1;
          v_1[cell_id] = 0.0;
          u_1[cell_id] = 0.0;
        }
      }
    }

#ifdef GPU_BUILD
    if (use_gpu)
      {
        auto rho = reinterpret_cast<float_type *> (solver_workspace.get ("gpu_rho", 0));
        auto u   = reinterpret_cast<float_type *> (solver_workspace.get ("gpu_u", 0));
        auto v   = reinterpret_cast<float_type *> (solver_workspace.get ("gpu_v", 0));
        auto p   = reinterpret_cast<float_type *> (solver_workspace.get ("gpu_p", 0));

        const unsigned int cells_count = topology.get_cells_count ();

        cudaMemcpyAsync (rho, rho_1, cells_count * sizeof (float_type), cudaMemcpyHostToDevice);
        cudaMemcpyAsync (u,   u_1,   cells_count * sizeof (float_type), cudaMemcpyHostToDevice);
        cudaMemcpyAsync (v,   v_1,   cells_count * sizeof (float_type), cudaMemcpyHostToDevice);
        cudaMemcpyAsync (p,   p_1,   cells_count * sizeof (float_type), cudaMemcpyHostToDevice);
      }
#endif
  }

  float_type calculate_dt_cpu (
    unsigned int thread_id,
    unsigned int total_threads,

    const grid_topology &topology,
    const grid_geometry &geometry,

    const float_type *p_rho,
    const float_type *p_u,
    const float_type *p_v,
    const float_type *p_p) const
  {
    float_type max_speed = std::numeric_limits<float_type>::min ();
    float_type min_len = std::numeric_limits<float_type>::max ();

    auto yr = work_range::split (solver_grid->get_cells_number (), thread_id, total_threads);

    for (unsigned int cell_id = yr.chunk_begin; cell_id < yr.chunk_end; cell_id++)
    {
      const float_type rho = p_rho[cell_id];
      const float_type p = p_p[cell_id];
      const float_type a = speed_of_sound_in_gas (gamma, p, rho);
      const float_type u = p_u[cell_id];
      const float_type v = p_v[cell_id];

      max_speed = std::max (max_speed, std::max (std::fabs (u + a), std::fabs (u - a)));
      max_speed = std::max (max_speed, std::max (std::fabs (v + a), std::fabs (v - a)));

      for (unsigned int edge_id = 0; edge_id < topology.get_edges_count (cell_id); edge_id++)
      {
        const float_type edge_len = geometry.get_edge_area (cell_id, edge_id);
        if (edge_len < min_len)
          min_len = edge_len;
      }
    }

    float_type new_dt = cfl * min_len / max_speed;
    threads.reduce_min (thread_id, new_dt);
    return new_dt;
  }

  float_type calculate_dt (
    unsigned int thread_id,
    unsigned int total_threads,

    const grid_topology &topology,
    const grid_geometry &geometry,

    const float_type *p_rho,
    const float_type *p_u,
    const float_type *p_v,
    const float_type *p_p) const
  {
#ifdef GPU_BUILD
    if (use_gpu)
    {
      float_type dt = std::numeric_limits<float_type>::max ();
      if (is_main_thread (thread_id))
      {
        dt = euler_2d_calculate_dt_gpu_interface (
            gamma, cfl, topology, geometry,
            reinterpret_cast<float_type *> (solver_workspace.get ("euler_workspace")),
            p_rho, p_u, p_v, p_p);
      }
      threads.reduce_min (thread_id, dt);
      return dt;
    }
    else
#endif
    {
      return calculate_dt_cpu (thread_id, total_threads, topology, geometry, p_rho, p_u, p_v, p_p);
    }
  }

  void solve_cpu (
      unsigned int thread_id,
      unsigned int total_threads,
      float_type dt,

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
    auto yr = work_range::split (solver_grid->get_cells_number (), thread_id, total_threads);

    for (unsigned int cell_id = yr.chunk_begin; cell_id < yr.chunk_end; cell_id++)
      euler_2d_calculate_next_cell_values (
          cell_id, dt, gamma, topology, geometry,
          p_rho, p_rho_next, p_u, p_u_next, p_v, p_v_next, p_p, p_p_next);
  }

  double solve (unsigned int step, unsigned int thread_id, unsigned int total_threads) final
  {
    const std::string prefix = use_gpu ? "gpu_" : "";

    auto p_rho      = reinterpret_cast<float_type *> (solver_workspace.get (prefix + "rho", (step + 0) % 2));
    auto p_rho_next = reinterpret_cast<float_type *> (solver_workspace.get (prefix + "rho", (step + 1) % 2));
    auto p_u        = reinterpret_cast<float_type *> (solver_workspace.get (prefix + "u",   (step + 0) % 2));
    auto p_u_next   = reinterpret_cast<float_type *> (solver_workspace.get (prefix + "u",   (step + 1) % 2));
    auto p_v        = reinterpret_cast<float_type *> (solver_workspace.get (prefix + "v",   (step + 0) % 2));
    auto p_v_next   = reinterpret_cast<float_type *> (solver_workspace.get (prefix + "v",   (step + 1) % 2));
    auto p_p        = reinterpret_cast<float_type *> (solver_workspace.get (prefix + "p",   (step + 0) % 2));
    auto p_p_next   = reinterpret_cast<float_type *> (solver_workspace.get (prefix + "p",   (step + 1) % 2));

    const auto topology = solver_grid->gen_topology_wrapper ();
    const auto geometry = solver_grid->gen_geometry_wrapper ();

    const float_type dt = calculate_dt (thread_id, total_threads, topology, geometry, p_rho, p_u, p_v, p_p);

#ifdef GPU_BUILD
    if (use_gpu)
    {
      if (is_main_thread (thread_id))
        euler_2d_calculate_next_time_step_gpu_interface (
            dt, gamma, topology, geometry,
            p_rho, p_rho_next, p_u, p_u_next,
            p_v, p_v_next, p_p, p_p_next);
    }
    else
#endif
    {
      solve_cpu (
          thread_id, total_threads, dt,
          topology, geometry,
          p_rho, p_rho_next, p_u, p_u_next, p_v,
          p_v_next, p_p, p_p_next);
    }

    threads.barrier ();
    return dt;
  }
};

#endif //BENCHMARK_MESH_AOS_H
