//
// Created by egi on 6/16/19.
//

#ifndef ANYSIM_RESULT_EXTRACTOR_H
#define ANYSIM_RESULT_EXTRACTOR_H

#include "core/solver/workspace.h"
#include "core/cpu/thread_pool.h"
#include "core/gpu/coloring.cuh"

class result_extractor
{
public:
  result_extractor () = delete;
  explicit result_extractor (const workspace &solver_workspace_arg)
    : solver_workspace (solver_workspace_arg)
  {
  }

  virtual ~result_extractor () = default;
  virtual void extract (
    unsigned int thread_id,
    unsigned int threads_count) = 0;

protected:
  const workspace &solver_workspace;
};

class cpu_results_visualizer : public result_extractor
{
  const unsigned int nx = 0;
  const unsigned int ny = 0;
  float *colors = nullptr;

  std::string target_name;

public:
  cpu_results_visualizer (
      const workspace &solver_workspace,
      unsigned int nx_arg,
      unsigned int ny_arg,
      float *colors_arg)
    : result_extractor (solver_workspace)
    , nx (nx_arg)
    , ny (ny_arg)
    , colors (colors_arg)
  { }

  void set_target (const std::string &target)
  {
    target_name = target;
  }

  void extract (
    unsigned int thread_id,
    unsigned int threads_count) final
  {
    auto yr = work_range::split (ny, thread_id, threads_count);

    // TODO Move this cast into wrapper
    auto data = reinterpret_cast<const double *> (solver_workspace.get (target_name));

    if (!data)
      return;

    for (unsigned int j = yr.chunk_begin; j < yr.chunk_end; j++)
      for (unsigned int i = 0; i < nx; i++)
        for (unsigned int k = 0; k < 4; k++)
          fill_vertex_color (data[j * nx + i], colors + 3 * 4 * (j * nx + i) + 3 * k);
  }
};

#endif  // ANYSIM_RESULT_EXTRACTOR_H
