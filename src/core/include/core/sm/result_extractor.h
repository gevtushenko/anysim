//
// Created by egi on 6/16/19.
//

#ifndef ANYSIM_RESULT_EXTRACTOR_H
#define ANYSIM_RESULT_EXTRACTOR_H

#include "core/pm/project_manager.h"
#include "core/solver/workspace.h"
#include "core/cpu/thread_pool.h"
#include "core/gpu/coloring.cuh"
#include "core/grid/grid.h"

class result_extractor
{
public:
  result_extractor () = default;

  virtual ~result_extractor () = default;
  virtual void extract (
    unsigned int thread_id,
    unsigned int threads_count,
    thread_pool &threads) = 0;
};

class cpu_results_visualizer : public result_extractor
{
private:
  template <class data_type>
  void render (unsigned int thread_id, unsigned int threads_count, thread_pool &threads)
  {
    const auto &solver_grid = pm.get_grid ();
    const auto &solver_workspace = pm.get_solver_workspace ();
    const unsigned int nx = solver_grid.nx;
    const unsigned int ny = solver_grid.ny;
    auto yr = work_range::split (ny, thread_id, threads_count);
    auto data = reinterpret_cast<const data_type*> (solver_workspace.get (target_name));

    if (!data)
      return;

    data_type min = std::numeric_limits<data_type>::max ();
    data_type max = std::numeric_limits<data_type>::min ();

    for (unsigned int j = yr.chunk_begin; j < yr.chunk_end; j++)
    {
      for (unsigned int i = 0; i < nx; i++)
      {
        const data_type val = data[j * nx + i];
        if (val > max) max = val;
        if (val < min) min = val;
      }
    }

    threads.reduce_min (thread_id, min);
    threads.reduce_max (thread_id, max);

    for (unsigned int j = yr.chunk_begin; j < yr.chunk_end; j++)
      for (unsigned int i = 0; i < nx; i++)
        for (unsigned int k = 0; k < 4; k++)
          fill_vertex_color (data[j * nx + i], colors + 3 * 4 * (j * nx + i) + 3 * k, min, max);
  }

public:
  explicit cpu_results_visualizer (project_manager &pm_arg)
    : result_extractor ()
    , pm (pm_arg)
  { }

  void set_target (const std::string &target, float *colors_arg)
  {
    colors = colors_arg;
    target_name = target;
  }

  void extract (
    unsigned int thread_id,
    unsigned int threads_count,
    thread_pool &threads) final
  {
    if (pm.is_double_precision_used ())
      render<double> (thread_id, threads_count, threads);
    else
      render<float> (thread_id, threads_count, threads);
  }

private:
  float *colors = nullptr;
  std::string target_name;

  project_manager &pm;
};

class gpu_results_visualizer : public result_extractor
{
private:
  template <class data_type>
  void render (unsigned int thread_id, unsigned int /* threads_count */, thread_pool & /* threads */)
  {
    const auto &solver_grid = pm.get_grid ();
    const auto &solver_workspace = pm.get_solver_workspace ();
    const unsigned int nx = solver_grid.nx;
    const unsigned int ny = solver_grid.ny;
    auto data = reinterpret_cast<const data_type*> (solver_workspace.get (target_name));

    if (!data)
      return;

    if (is_main_thread (thread_id))
      fill_colors (nx, ny, data, colors);
  }

public:
  explicit gpu_results_visualizer (project_manager &pm_arg)
    : result_extractor ()
    , pm (pm_arg)
  { }

  void set_target (const std::string &target, float *colors_arg)
  {
    colors = colors_arg;
    target_name = target;
  }

  void extract (
    unsigned int thread_id,
    unsigned int threads_count,
    thread_pool &threads) final
  {
    if (pm.is_double_precision_used ())
      render<double> (thread_id, threads_count, threads);
    else
      render<float> (thread_id, threads_count, threads);
  }

private:
  float *colors = nullptr;
  std::string target_name;

  project_manager &pm;
};

class hybrid_results_visualizer : public result_extractor
{
public:
  explicit hybrid_results_visualizer (project_manager &pm_arg)
    : result_extractor ()
    , pm (pm_arg)
    , cpu_visualizer (pm)
    , gpu_visualizer (pm)
    { }

  void extract (
    unsigned int thread_id,
    unsigned int threads_count,
    thread_pool &threads) final
  {
    get_extractor ()->extract (thread_id, threads_count, threads);
  }

  void set_target (const std::string &target, float *colors_arg)
  {
    if (pm.get_use_gpu ())
      gpu_visualizer.set_target (target, colors_arg);
    else
      cpu_visualizer.set_target (target, colors_arg);
  }

private:
  result_extractor *get_extractor ()
  {
    if (pm.get_use_gpu ())
      return &gpu_visualizer;
    else
      return &cpu_visualizer;
  }

private:
  project_manager &pm;
  cpu_results_visualizer cpu_visualizer;
  gpu_results_visualizer gpu_visualizer;
};

#endif  // ANYSIM_RESULT_EXTRACTOR_H
