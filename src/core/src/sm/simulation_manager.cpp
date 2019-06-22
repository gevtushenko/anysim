//
// Created by egi on 6/8/19.
//

#include "core/sm/simulation_manager.h"
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
    bool use_double_precision_arg,
    workspace &workspace_arg)
  : solver_workspace (workspace_arg)
  , solver_context (
      solver_abstract_method (
          solver_arg,
          use_double_precision_arg,
          threads,
          solver_workspace))
{
}

void simulation_manager::fill_configuration_scheme (configuration_node &solver_scheme)
{
  if (solver_context)
    solver_context->fill_configuration_scheme (solver_scheme);
}

void simulation_manager::apply_configuration (
    const configuration_node &solver_config,
    grid &solver_grid)
{
  if (solver_context)
    solver_context->apply_configuration (solver_config, solver_grid);
}

bool simulation_manager::calculate_next_time_step (result_extractor **extractors, unsigned int extractors_count)
{
  if (!solver_context)
    return false;

  const auto calculation_begin = std::chrono::high_resolution_clock::now ();
  const unsigned int steps_until_render = 100;

  solver_workspace.set_active_layer ("rho", 0);
  threads.execute ([&] (unsigned int thread_id, unsigned int threads_count) {
    for (unsigned int local_step = 0; local_step < steps_until_render; local_step++)
      solver_context->solve (step + local_step, thread_id, threads_count);
    for (unsigned int eid = 0; eid < extractors_count; eid++)
      extractors[eid]->extract (thread_id, threads_count, threads);
  });

  const auto calculation_end = std::chrono::high_resolution_clock::now ();
  const std::chrono::duration<double> duration = calculation_end - calculation_begin;
  std::cout << "Computation completed in " << duration.count () << "s\n";

  step += steps_until_render;

  return step < 3000;
}

