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

  unsigned int nx = 0;
  unsigned int ny = 0;

  float_type cfl = 0.1;
  float_type gamma = 1.4;

  float_type dx = 1.0;
  float_type dy = 1.0;

  float_type edge_lengths[4];
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
    dx = solver_grid->dx;
    dy = solver_grid->dy;

    edge_lengths[LEFT] = edge_lengths[RIGHT] = dx;
    edge_lengths[BOTTOM] = edge_lengths[TOP] = dy;

    nx = solver_grid->nx;
    ny = solver_grid->ny;

#ifdef GPU_BUILD
    use_gpu = gpu_num >= 0;

    if (use_gpu)
      {
        solver_grid_arg->create_field<float_type> ("gpu_rho", memory_holder_type::device, 2);
        solver_grid_arg->create_field<float_type> ("gpu_u",   memory_holder_type::device, 2);
        solver_grid_arg->create_field<float_type> ("gpu_v",   memory_holder_type::device, 2);
        solver_grid_arg->create_field<float_type> ("gpu_p",   memory_holder_type::device, 2);
        solver_workspace.allocate ("gpu_edge_length", memory_holder_type::device, 4 * sizeof (float_type));
        solver_workspace.allocate ("euler_workspace", memory_holder_type::device, sizeof (float_type));

        cudaMemcpy (solver_workspace.get ("gpu_edge_length"), edge_lengths, 4 * sizeof (float_type), cudaMemcpyHostToDevice);
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

    for (unsigned int y = 0; y < ny; ++y)
    {
      for (unsigned int x = 0; x < nx; ++x)
      {
        auto i = y * nx + x;

        const float_type lbx = x * dx;
        const float_type lby = y * dy;

        if (lbx < x_0)
        {
          rho_1[i] = 1.0;
          p_1[i] = 1.0;
          v_1[i] = 0.0;
          u_1[i] = 0.0;
        }
        else
        {
          if (lby < y_0)
          {
            rho_1[i] = 1.0;
            p_1[i] = 0.1;
            v_1[i] = 0.0;
            u_1[i] = 0.0;
          }
          else
          {
            rho_1[i] = 0.125;
            p_1[i] = 0.1;
            v_1[i] = 0.0;
            u_1[i] = 0.0;
          }
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

        cudaMemcpyAsync (rho, rho_1, nx * ny * sizeof (float_type), cudaMemcpyHostToDevice);
        cudaMemcpyAsync (u,   u_1,   nx * ny * sizeof (float_type), cudaMemcpyHostToDevice);
        cudaMemcpyAsync (v,   v_1,   nx * ny * sizeof (float_type), cudaMemcpyHostToDevice);
        cudaMemcpyAsync (p,   p_1,   nx * ny * sizeof (float_type), cudaMemcpyHostToDevice);
      }
#endif
  }

  float_type calculate_dt_cpu (
    unsigned int thread_id,
    unsigned int total_threads,
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

      for (unsigned int edge_id = 0; edge_id < solver_grid->get_edges_count (cell_id); edge_id++)
      {
        const float_type edge_len = solver_grid->get_edge_area (edge_id);
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
            nx * ny,
            gamma,
            cfl,
            std::min (dx, dy),
            reinterpret_cast<float_type *> (solver_workspace.get ("euler_workspace")),
            p_rho, p_u, p_v, p_p);
      }
      threads.reduce_min (thread_id, dt);
      return dt;
    }
    else
#endif
    {
      return calculate_dt_cpu (thread_id, total_threads, p_rho, p_u, p_v, p_p);
    }
  }

  void solve_cpu (
      unsigned int thread_id,
      unsigned int total_threads,
      float_type dt,
      const float_type *p_rho,
      float_type *p_rho_next,
      const float_type *p_u,
      float_type *p_u_next,
      const float_type *p_v,
      float_type *p_v_next,
      const float_type *p_p,
      float_type *p_p_next)
  {
    float_type q_c[4];
    float_type q_n[4];

    float_type Q_c[4];
    float_type Q_n[4];

    float_type F_c[4];
    float_type F_n[4];

    float_type F_sigma[4];  /// Edge flux in local coordinate system
    float_type f_sigma[4];  /// Edge flux in global coordinate system

    auto yr = work_range::split (solver_grid->get_cells_number (), thread_id, total_threads);

    for (unsigned int cell_id = yr.chunk_begin; cell_id < yr.chunk_end; cell_id++)
    {
      fill_state_vector (cell_id, gamma, p_rho, p_u, p_v, p_p, q_c);

      float_type flux[4] = {0.0, 0.0, 0.0, 0.0};

      /// Edge flux
      for (unsigned int edge_id = 0; edge_id < solver_grid->get_edges_count (cell_id); edge_id++)
      {
        const unsigned int neighbor_id = solver_grid->get_neighbor_id (cell_id, edge_id);

        fill_state_vector (neighbor_id, gamma, p_rho, p_u, p_v, p_p, q_n);
        rotate_vector_to_edge_coordinates (edge_id, q_c, Q_c);
        rotate_vector_to_edge_coordinates (edge_id, q_n, Q_n);
        fill_flux_vector (Q_c, F_c, gamma);
        fill_flux_vector (Q_n, F_n, gamma);

        const float_type U_c = Q_c[1] / Q_c[0];
        const float_type U_n = Q_n[1] / Q_n[0];

        rusanov_scheme (
            gamma,
            p_p[cell_id], p_p[neighbor_id],
            p_rho[cell_id], p_rho[neighbor_id],
            U_c, U_n, F_c, F_n, Q_n, Q_c, F_sigma);

        rotate_vector_from_edge_coordinates (edge_id, F_sigma, f_sigma);

        for (int c = 0; c < 4; c++)
          flux[c] += solver_grid->get_edge_area (edge_id) * f_sigma[c];
      }

      float_type new_q[4];
      for (int c = 0; c < 4; c++)
        new_q[c] = q_c[c] - (dt / solver_grid->get_cell_volume (cell_id)) * (flux[c]);

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

    const float_type dt = calculate_dt (thread_id, total_threads, p_rho, p_u, p_v, p_p);

#ifdef GPU_BUILD
    if (use_gpu)
    {
      if (is_main_thread (thread_id))
        euler_2d_calculate_next_time_step_gpu_interface (
            nx, ny, dt, gamma, dx * dy,
            reinterpret_cast<const float_type *> (solver_workspace.get ("gpu_edge_length")),
            p_rho, p_rho_next, p_u, p_u_next,
            p_v, p_v_next, p_p, p_p_next);
    }
    else
#endif
    {
      solve_cpu (
          thread_id, total_threads, dt,
          p_rho, p_rho_next, p_u, p_u_next, p_v,
          p_v_next, p_p, p_p_next);
    }

    threads.barrier ();
    return dt;
  }
};

#endif //BENCHMARK_MESH_AOS_H
