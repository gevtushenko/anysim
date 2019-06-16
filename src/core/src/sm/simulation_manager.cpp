//
// Created by egi on 6/8/19.
//

#include "core/sm/simulation_manager.h"
#include "core/sm/result_extractor.h"
#include "core/solver/solver.h"
#include "core/cpu/euler_2d.h"

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

  return nullptr;
}

simulation_manager::simulation_manager (
    const std::string &solver_arg,
    bool use_double_precision_arg,
    workspace &workspace_arg)
  : solver_workspace (workspace_arg)
  , solver_context (solver_abstract_method (solver_arg, use_double_precision_arg, threads, solver_workspace))
{
}

void simulation_manager::fill_configuration_scheme (configuration &scheme)
{
  if (solver_context)
    solver_context->fill_configuration_scheme (scheme);
}

void simulation_manager::apply_configuration (const configuration &config)
{
  if (solver_context)
    solver_context->apply_configuration (config);
}

void simulation_manager::calculate_next_time_step ()
{
  if (!solver_context)
    return;

  threads.execute ([&] (unsigned int thread_id, unsigned int threads_count) {
    for (unsigned int step = 0; step < 100; step++)
      solver_context->solve (step, thread_id, threads_count);
    for (auto &extractor: extractors)
      extractor->extract (thread_id, threads_count);
  });
}

void simulation_manager::append_extractor (result_extractor *extractor)
{
  extractors.push_back (extractor);
}

