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
    unsigned int threads_count) = 0;
};

class cpu_results_visualizer : public result_extractor
{
private:
  template <class data_type>
  void render (unsigned int thread_id, unsigned int threads_count)
  {
    const auto &solver_grid = pm.get_grid ();
    const auto &solver_workspace = pm.get_solver_workspace ();
    const unsigned int nx = solver_grid.nx;
    const unsigned int ny = solver_grid.ny;
    auto yr = work_range::split (ny, thread_id, threads_count);
    auto data = reinterpret_cast<const data_type*> (solver_workspace.get (target_name));

    if (!data)
      return;

    for (unsigned int j = yr.chunk_begin; j < yr.chunk_end; j++)
      for (unsigned int i = 0; i < nx; i++)
        for (unsigned int k = 0; k < 4; k++)
          fill_vertex_color (data[j * nx + i], colors + 3 * 4 * (j * nx + i) + 3 * k);
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
    unsigned int threads_count) final
  {
    if (pm.is_double_precision_used ())
      render<double> (thread_id, threads_count);
    else
      render<float> (thread_id, threads_count);
  }

private:
  float *colors = nullptr;
  std::string target_name;

  project_manager &pm;
};

#endif  // ANYSIM_RESULT_EXTRACTOR_H
