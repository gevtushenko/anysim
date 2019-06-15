//
// Created by egi on 6/8/19.
//

#include "core/sm/simulation_manager.h"
#include "core/solver/solver.h"
#include "core/cpu/euler_2d.h"

solver *solver_abstract_method (const std::string &solver_arg, bool use_double_precision_arg, thread_pool &threads)
{
  if (solver_arg == "euler_2d")
  {
    if (use_double_precision_arg)
    {
      return new euler_2d<double> (threads);
    }
    else
    {
      return new euler_2d<float> (threads);
    }
  }

  return nullptr;
}

simulation_manager::simulation_manager (const std::string &solver_arg, bool use_double_precision_arg)
  : solver_context (solver_abstract_method (solver_arg, use_double_precision_arg, threads))
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
    for (unsigned int step = 0; step < 1000; step++)
      solver_context->solve (step, thread_id, threads_count);
  });
}
