//
// Created by egi on 6/8/19.
//

#include "core/sm/simulation_manager.h"
#include "core/config/configuration.h"
#include "core/sm/result_extractor.h"
#include "core/solver/solver.h"
#include "core/cpu/euler_2d.h"
#include "core/cpu/fdtd_2d.h"

solver *solver_abstract_method (
    const std::string &solver_arg,
    bool use_double_precision_arg,
    thread_pool &threads,
    workspace &workspace_arg)
{
  if (solver_arg == "euler_2d")
  {
    if (use_double_precision_arg)
    {
      return new euler_2d<double> (threads, workspace_arg);
    }
    else
    {
      return new euler_2d<float> (threads, workspace_arg);
    }
  }
  if (solver_arg == "fdtd_2d")
  {
    if (use_double_precision_arg)
    {
      return new fdtd_2d<double> (threads, workspace_arg);
    }
    else
    {
      return new fdtd_2d<float> (threads, workspace_arg);
    }
  }

  return nullptr;
}

simulation_manager::simulation_manager (
    const std::string &solver_arg,
    double max_simulation_time_arg,
    bool use_double_precision_arg,
    workspace &workspace_arg)
  : max_time (max_simulation_time_arg)
  , solver_workspace (workspace_arg)
  , solver_context (
      solver_abstract_method (
          solver_arg,
          use_double_precision_arg,
          threads,
          solver_workspace))
{
}

void simulation_manager::fill_configuration_scheme (configuration &scheme, std::size_t scheme_id)
{
  if (solver_context)
    solver_context->fill_configuration_scheme (scheme, scheme_id);
}

void simulation_manager::apply_configuration (
    const configuration &config,
    std::size_t config_id,
    grid *solver_grid,
    int gpu_num)
{
#ifdef GPU_BUILD
  if (gpu_num >= 0)
  {
    threads.execute ([=] (unsigned int thread_id, unsigned int) {
      if (is_main_thread (thread_id))
        cudaSetDevice (gpu_num);
    });
  }
#endif

  step = 0;
  time = 0.0;

  if (solver_context)
    solver_context->apply_configuration (config, config_id, solver_grid, gpu_num);
}

bool simulation_manager::is_gpu_supported () const
{
  return solver_context->is_gpu_supported ();
}

bool simulation_manager::calculate_next_time_step (result_extractor **extractors, unsigned int extractors_count)
{
  if (!solver_context)
    return false;

  const auto calculation_begin = std::chrono::high_resolution_clock::now ();
  const unsigned int steps_until_render = 10;

  solver_workspace.set_active_layer ("rho", 0);
  threads.execute ([&] (unsigned int thread_id, unsigned int threads_count) {
    double report_time = 0.0;
    for (unsigned int local_step = 0; local_step < steps_until_render; local_step++)
    {
      report_time += solver_context->solve (step + local_step, thread_id, threads_count);
      if (time + report_time > max_time)
        break;
    }

    threads.barrier ();
    time += report_time;

    for (unsigned int eid = 0; eid < extractors_count; eid++)
      extractors[eid]->extract (thread_id, threads_count, threads);
  });

  const auto calculation_end = std::chrono::high_resolution_clock::now ();
  const std::chrono::duration<double> duration = calculation_end - calculation_begin;
  std::cout << "Computation of time " << time << " completed in " << duration.count () << "s\n";

  step += steps_until_render;

  return time < max_time; /// Calculation continuation condition
}

