//
// Created by egi on 6/2/19.
//

#ifdef GPU_BUILD
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#endif

#include "core/config/configuration.h"
#include "core/gpu/fdtd_gpu_interface.h"
#include "core/gpu/coloring.cuh"
#include "core/cpu/sources_holder.h"
#include "core/cpu/thread_pool.h"
#include "core/solver/solver.h"
#include "core/common/curl.h"
#include "cpp/common_funcs.h"
#include "core/gpu/fdtd.cuh"
#include "core/grid/grid.h"

#include <iostream>
#include <chrono>
#include <memory>
#include <cmath>

constexpr double C0 = 299792458; /// Speed of light [metres per second]

#ifndef ANYSIM_FDTD_2D_H
#define ANYSIM_FDTD_2D_H

enum class boundary_condition
{
  dirichlet, periodic
};

template <class float_type>
class region_initializer
{
public:
  region_initializer () = delete;
  region_initializer (unsigned int nx_arg, unsigned int ny_arg)
    : nx (nx_arg)
    , ny (ny_arg)
  { }

  virtual void fill_region (float_type *er, float_type *hr) const = 0;

protected:
  const unsigned int nx, ny;
};

template <class float_type>
class rectangular_region_initializer : public region_initializer<float_type>
{
public:
  rectangular_region_initializer () = delete;
  rectangular_region_initializer (
    unsigned int nx,
    unsigned int ny,
    unsigned int region_i_start_arg,
    unsigned int region_j_start_arg,
    unsigned int width_arg,
    unsigned int height_arg,
    float_type er_value_arg,
    float_type hr_value_arg)
    : region_initializer<float_type> (nx, ny)
    , region_i_start (region_i_start_arg)
    , region_j_start (region_j_start_arg)
    , width (width_arg)
    , height (height_arg)
    , er_value (er_value_arg)
    , hr_value (hr_value_arg)
  { }

  void fill_region (float_type *er, float_type *hr) const final
  {
    using ri = region_initializer<float_type>;

    for (unsigned int j = region_j_start; j < region_j_start + height; j++)
      {
        for (unsigned int i = region_i_start; i < region_i_start + width; i++)
          {
            er[j * ri::nx + i] = er_value;
            hr[j * ri::nx + i] = hr_value;
          }
      }
  }

protected:
  const unsigned int region_i_start, region_j_start, width, height;
  const float_type er_value, hr_value;
};

template <class float_type>
class fdtd_2d : public solver
{
  boundary_condition left_bc, bottom_bc, right_bc, top_bc;

  bool use_gpu = false;
  unsigned int nx = 0;
  unsigned int ny = 0;

  float_type dx = 0.1;
  float_type dy = 0.1;
  float_type dt = 0.1;
  float_type t = 0.0;

  float_type *m_h = nullptr;
  float_type *dz = nullptr;
  float_type *ez = nullptr;
  float_type *hx = nullptr;
  float_type *hy = nullptr;
  float_type *er = nullptr;
  float_type *hr = nullptr; /// Materials properties (for now assume mu_xx = mu_yy)

  float_type *d_mh = nullptr;
  float_type *d_er = nullptr;

  unsigned int sources_count = 0;
  float_type *d_sources_frequencies = nullptr;
  unsigned int *d_sources_offsets = nullptr;

  grid *solver_grid = nullptr;

  std::unique_ptr<sources_holder<float_type>> sources;

public:
  fdtd_2d () = delete;
  fdtd_2d (
    thread_pool &threads_arg,
    workspace &solver_workspace_arg)
  : solver (threads_arg, solver_workspace_arg)
  , left_bc (boundary_condition::periodic)
  , bottom_bc (boundary_condition::periodic)
  , right_bc (boundary_condition::periodic)
  , top_bc (boundary_condition::periodic)
  { }

  void fill_configuration_scheme (configuration &config, std::size_t config_id) final
  {
    config.create_node (config_id, "cfl", 0.5);
    const auto source_scheme_id = config.create_group("source_scheme");
    config.create_node (source_scheme_id, "frequency", 1E+8);
    config.create_node (source_scheme_id, "x", 0.5);
    config.create_node (source_scheme_id, "y", 0.5);
    config.create_array(config_id, "sources", source_scheme_id);
  }

  bool is_gpu_supported () const final
  {
    return true;
  }

  void apply_configuration (const configuration &config, std::size_t solver_id, grid *solver_grid_arg, int gpu_num) final
  {
    solver_grid = solver_grid_arg;

    dx = solver_grid->dx;
    dy = solver_grid->dy;

    const auto solver_children = config.children_for (solver_id);

    auto cfl_id = solver_children[0];
    auto sources_id = solver_children[1];
    const float_type cfl = config.get_node_value (cfl_id);
    dt = cfl * std::min (dx, dy) / C0;
    t = 0.0; /// Reset time

    nx = solver_grid->nx;
    ny = solver_grid->ny;

    sources = std::make_unique<sources_holder<float_type>> ();
    for (auto &source_id: config.children_for (sources_id))
    {
      const auto source_children = config.children_for (source_id);
      const double frequency = config.get_node_value (source_children[0]);
      const double x = config.get_node_value (source_children[1]);
      const double y = config.get_node_value (source_children[2]);

      const unsigned int grid_x = std::ceil (x / dx);
      const unsigned int grid_y = std::ceil (y / dy);

      sources->append_source (frequency, nx * grid_y + grid_x);
    }

    use_gpu = gpu_num >= 0;
    memory_holder_type holder = use_gpu ? memory_holder_type::device : memory_holder_type::host;

    solver_grid->create_field<float_type> ("ez", holder, 1);
    solver_grid->create_field<float_type> ("dz", holder, 1);
    solver_grid->create_field<float_type> ("hx", holder, 1);
    solver_grid->create_field<float_type> ("hy", holder, 1);
    solver_grid->create_field<float_type> ("hr", memory_holder_type::host, 1);
    solver_grid->create_field<float_type> ("er", memory_holder_type::host, 1);
    solver_grid->create_field<float_type> ("mh", memory_holder_type::host, 1);

    m_h = reinterpret_cast<float_type *> (solver_workspace.get ("mh"));
    dz  = reinterpret_cast<float_type *> (solver_workspace.get ("dz"));
    ez  = reinterpret_cast<float_type *> (solver_workspace.get ("ez"));
    hx  = reinterpret_cast<float_type *> (solver_workspace.get ("hx"));
    hy  = reinterpret_cast<float_type *> (solver_workspace.get ("hy"));
    er  = reinterpret_cast<float_type *> (solver_workspace.get ("er"));
    hr  = reinterpret_cast<float_type *> (solver_workspace.get ("hr"));

    std::fill_n (er, nx * ny, 1.0);
    std::fill_n (hr, nx * ny, 1.0);

    for (unsigned int i = 0; i < nx * ny; i++)
      m_h[i] = C0 * dt / hr[i];

#ifdef GPU_BUILD
    if (use_gpu)
    {
      solver_grid->create_field<float_type> ("gpu_er", memory_holder_type::device, 1);
      solver_grid->create_field<float_type> ("gpu_mh", memory_holder_type::device, 1);

      d_mh = reinterpret_cast<float_type *> (solver_workspace.get ("gpu_mh"));
      d_er = reinterpret_cast<float_type *> (solver_workspace.get ("gpu_er"));

      cudaMemset (ez, 0, nx * ny * sizeof (float_type));
      cudaMemset (hx, 0, nx * ny * sizeof (float_type));
      cudaMemset (hy, 0, nx * ny * sizeof (float_type));

      sources_count = sources->get_sources_count ();

      cudaMalloc (&d_sources_frequencies, sources_count * sizeof (float_type));
      cudaMalloc (&d_sources_offsets, sources_count * sizeof (unsigned int));

      cudaMemcpy (d_mh, m_h, nx * ny * sizeof (float_type), cudaMemcpyHostToDevice);
      cudaMemcpy (d_er, er,  nx * ny * sizeof (float_type), cudaMemcpyHostToDevice);

      cudaMemcpy (d_sources_frequencies, sources->get_sources_frequencies (), sources_count * sizeof (float_type), cudaMemcpyHostToDevice);
      cudaMemcpy (d_sources_offsets, sources->get_sources_offsets (), sources_count * sizeof (unsigned int), cudaMemcpyHostToDevice);
    }
    else
#endif
    {
      std::fill_n (hx, nx * ny, 0.0);
      std::fill_n (hy, nx * ny, 0.0);
      std::fill_n (ez, nx * ny, 0.0);
      std::fill_n (dz, nx * ny, 0.0);
    }

    // TODO Apply initializers
  }

  void handle_grid_change () final { }

  void update_h (
    unsigned int thread_id,
    unsigned int total_threads,
    const grid_topology &topology,
    const grid_geometry &geometry)
  {
    auto yr = work_range::split (topology.get_cells_count (), thread_id, total_threads);
    for (unsigned int cell_id = yr.chunk_begin; cell_id < yr.chunk_end; cell_id++)
      fdtd_2d_update_h (cell_id, topology, geometry, ez, m_h, hx, hy);
  }

  void update_e (
    unsigned int thread_id,
    unsigned int total_threads,
    const sources_holder<float_type> &s,
    const grid_topology &topology,
    const grid_geometry &geometry)
  {
    const float_type C0_p_dt = C0 * dt;

    auto sources_offsets = s.get_sources_offsets ();
    auto sources_frequencies = s.get_sources_frequencies ();

    auto yr = work_range::split (topology.get_cells_count (), thread_id, total_threads);
    for (unsigned int cell_id = yr.chunk_begin; cell_id < yr.chunk_end; cell_id++)
      fdtd_2d_update_e (
          cell_id, t, C0_p_dt, topology, geometry, er, hx, hy, dz, ez,
          s.get_sources_count (), sources_frequencies, sources_offsets);
  }

  /// Ez mode
  double solve (unsigned int /* step */, unsigned int thread_id, unsigned int total_threads) final
  {
    threads.barrier ();
    if (is_main_thread (thread_id))
      t += dt;

    const auto topology = solver_grid->gen_topology_wrapper ();
    const auto geometry = solver_grid->gen_geometry_wrapper ();

#ifdef GPU_BUILD
    if (use_gpu)
    {
      if (is_main_thread (thread_id))
        solve_gpu (topology, geometry);
    }
    else
#endif
    {
      solve_cpu (thread_id, total_threads, topology, geometry);
    }

    return dt;
  }

private:
  void solve_cpu (
      unsigned int thread_id,
      unsigned int total_threads,
      const grid_topology &topology,
      const grid_geometry &geometry)
  {
    update_h (thread_id, total_threads, topology, geometry);
    threads.barrier ();
    update_e (thread_id, total_threads, *sources, topology, geometry);
  }

#ifdef GPU_BUILD
  void solve_gpu (
      const grid_topology &topology,
      const grid_geometry &geometry)
  {
    fdtd_step<float_type> (t, C0 * dt, topology, geometry, d_mh, d_er, ez, dz, hx, hy, sources_count, d_sources_frequencies, d_sources_offsets);

    auto cuda_error = cudaGetLastError ();
    if (cuda_error != cudaSuccess)
    {
      std::cerr << "Error on GPU: " << cudaGetErrorString (cuda_error) << std::endl;
      return;
    }
  }
#endif
};


#endif //ANYSIM_FDTD_2D_H
